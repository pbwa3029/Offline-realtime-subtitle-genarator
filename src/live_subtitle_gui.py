import gc
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ARGOS_DEVICE_TYPE"] = "cpu"

import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd

# Import faster_whisper BEFORE argostranslate so ctranslate2 loads cleanly
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download, try_to_load_from_cache
from tqdm.auto import tqdm

try:
    import argostranslate.package as argos_pkg
    import argostranslate.translate as argos_trans
    ARGOS_AVAILABLE = True
except BaseException:
    ARGOS_AVAILABLE = False

try:
    from deep_translator import MyMemoryTranslator
    MYMEMORY_AVAILABLE = True
except BaseException:
    MYMEMORY_AVAILABLE = False

MODELS = {
    "large-v3-turbo (~1.6 GB - Fastest + Offline Argos Translate)": ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
    "large-v3 (~3.1 GB - Highest Accuracy)": ("large-v3", "Systran/faster-whisper-large-v3"),
    "medium (~1.5 GB - Balanced)": ("medium", "Systran/faster-whisper-medium"),
    "small (~484 MB - Lightweight)": ("small", "Systran/faster-whisper-small"),
}

INPUT_LANGUAGES = {
    "Auto-Detect": None,
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Hindi": "hi",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "English": "en",
    "Tamil": "ta",
    "Sinhala": "si",
}

ARGOS_PACK_CHOICES = {
    "Japanese → English (ja → en)": ("ja", "en", "Japanese → English"),
    "Korean → English (ko → en)": ("ko", "en", "Korean → English"),
    "Chinese → English (zh → en)": ("zh", "en", "Chinese → English"),
    "Spanish → English (es → en)": ("es", "en", "Spanish → English"),
    "French → English (fr → en)": ("fr", "en", "French → English"),
    "German → English (de → en)": ("de", "en", "German → English"),
    "Russian → English (ru → en)": ("ru", "en", "Russian → English"),
    "Hindi → English (hi → en)": ("hi", "en", "Hindi → English"),
    "Arabic → English (ar → en)": ("ar", "en", "Arabic → English"),
    "Portuguese → English (pt → en)": ("pt", "en", "Portuguese → English"),
    "Italian → English (it → en)": ("it", "en", "Italian → English"),
}

OUTPUT_OPTIONS = {
    "English (Offline Argos Translate - Instant + Turbo)": ("argos", "en"),
    "English (Offline Whisper Native - Use large-v3)": ("whisper_translate", "en"),
    "Same as Input (Transcribe Only)": ("transcribe", None),
    "English (MyMemory Online Backup)": ("mymemory", "en"),
    "Sinhala (MyMemory Online)": ("mymemory", "si"),
    "Tamil (MyMemory Online)": ("mymemory", "ta"),
    "Spanish (Offline Argos / Backup)": ("argos", "es"),
    "French (Offline Argos / Backup)": ("argos", "fr"),
}

TARGET_SAMPLE_RATE = 16000
STEP_SECONDS = 0.25
SILENCE_THRESHOLD = 0.010


class SubtitleOverlay(tk.Toplevel):
    """Frameless, always-on-top draggable subtitle bar that auto-expands for full sentences."""

    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.85)
        self.configure(bg="#111111")

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.box_width = 900
        x = (screen_w - self.box_width) // 2
        y = screen_h - 190
        self.geometry(f"+{x}+{y}")

        self._drag_x = 0
        self._drag_y = 0

        self.label = tk.Label(
            self,
            text="Subtitles will appear here — drag me anywhere",
            font=("Segoe UI", 16, "bold"),
            fg="#FFFFFF",
            bg="#111111",
            wraplength=self.box_width - 40,
            justify="center",
            width=64,
            padx=20,
            pady=14,
        )
        self.label.pack(expand=True, fill="both")

        for widget in (self, self.label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        x = self.winfo_x() + (event.x - self._drag_x)
        y = self.winfo_y() + (event.y - self._drag_y)
        self.geometry(f"+{x}+{y}")

    def update_text(self, text):
        self.label.config(text=text)
        self.update_idletasks()


class SubtitleControllerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Multi-Language Offline Subtitles")
        self.root.geometry("515x685")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.model = None
        self.loaded_model_size = None
        self.running = False
        self.downloading = False
        self.audio_queue = queue.Queue()
        self.overlay = None

        self._build_ui()
        self.check_model_status()
        self.check_argos_status()

    def _get_audio_devices(self):
        devices = sd.query_devices()
        input_devices = {}
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                name = f"[{idx}] {dev['name'][:48]}"
                input_devices[name] = idx
        return input_devices

    def _build_ui(self):
        pad = {"padx": 14, "pady": 4}

        # --- 1. Whisper Model Management Frame ---
        model_frame = tk.LabelFrame(self.root, text="1. Whisper Speech Model Setup", padx=10, pady=8)
        model_frame.pack(fill="x", padx=14, pady=(10, 6))

        self.model_var = tk.StringVar(value=list(MODELS.keys())[0])
        self.model_combo = ttk.Combobox(
            model_frame, textvariable=self.model_var, values=list(MODELS.keys()), state="readonly", width=58
        )
        self.model_combo.pack(anchor="w", pady=(0, 6))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.check_model_status())

        status_row = tk.Frame(model_frame)
        status_row.pack(fill="x")

        self.model_status_var = tk.StringVar(value="Checking local cache...")
        self.model_status_lbl = tk.Label(status_row, textvariable=self.model_status_var, font=("Segoe UI", 9, "bold"))
        self.model_status_lbl.pack(side="left")

        self.download_btn = tk.Button(
            status_row, text="⬇ Download Whisper", bg="#0067c0", fg="white", command=self.start_download
        )
        self.download_btn.pack(side="right")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            model_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=(8, 2))

        self.dl_detail_var = tk.StringVar(value="")
        tk.Label(model_frame, textvariable=self.dl_detail_var, fg="#555555", font=("Segoe UI", 8)).pack(anchor="w")

        # --- 2. Offline Translator Packs Frame ---
        argos_frame = tk.LabelFrame(self.root, text="2. Offline Argos Translator Pack Selector", padx=10, pady=8)
        argos_frame.pack(fill="x", padx=14, pady=4)

        self.argos_pack_var = tk.StringVar(value=list(ARGOS_PACK_CHOICES.keys())[0])
        self.argos_pack_combo = ttk.Combobox(
            argos_frame, textvariable=self.argos_pack_var, values=list(ARGOS_PACK_CHOICES.keys()), state="readonly", width=58
        )
        self.argos_pack_combo.pack(anchor="w", pady=(0, 6))
        self.argos_pack_combo.bind("<<ComboboxSelected>>", lambda e: self.check_argos_status())

        argos_top = tk.Frame(argos_frame)
        argos_top.pack(fill="x")

        self.argos_status_var = tk.StringVar(value="Checking selected pack...")
        self.argos_status_lbl = tk.Label(argos_top, textvariable=self.argos_status_var, font=("Segoe UI", 9, "bold"))
        self.argos_status_lbl.pack(side="left")

        btn_box = tk.Frame(argos_top)
        btn_box.pack(side="right")

        self.argos_single_btn = tk.Button(
            btn_box, text="⬇ Download Pack", bg="#4a5568", fg="white", command=self.start_selected_argos_install
        )
        self.argos_single_btn.pack(side="left", padx=(0, 4))

        self.argos_all_btn = tk.Button(
            btn_box, text="⬇ Install All → EN", bg="#2b6cb0", fg="white", command=self.start_all_argos_install
        )
        self.argos_all_btn.pack(side="left")

        self.argos_detail_var = tk.StringVar(value="")
        tk.Label(argos_frame, textvariable=self.argos_detail_var, fg="#555555", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        # --- 3. Audio, Language & Speed Settings ---
        settings_frame = tk.LabelFrame(self.root, text="3. Audio, Language & Speed Settings", padx=10, pady=8)
        settings_frame.pack(fill="x", padx=14, pady=6)

        tk.Label(settings_frame, text="Audio Source (Select Stereo Mix / VB-Cable for Video):").pack(anchor="w")
        self.device_map = self._get_audio_devices()
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            settings_frame, textvariable=self.device_var, values=list(self.device_map.keys()), state="readonly", width=58
        )
        if self.device_map:
            self.device_combo.current(0)
        self.device_combo.pack(pady=(2, 6))

        tk.Label(settings_frame, text="Input Language (Spoken in Video):").pack(anchor="w")
        self.in_lang_var = tk.StringVar(value="Japanese")
        self.in_lang_combo = ttk.Combobox(
            settings_frame, textvariable=self.in_lang_var, values=list(INPUT_LANGUAGES.keys()), state="readonly", width=58
        )
        self.in_lang_combo.pack(pady=(2, 6))

        tk.Label(settings_frame, text="Output Subtitle Mode / Target Language:").pack(anchor="w")
        self.out_opt_var = tk.StringVar(value=list(OUTPUT_OPTIONS.keys())[0])
        self.out_opt_combo = ttk.Combobox(
            settings_frame, textvariable=self.out_opt_var, values=list(OUTPUT_OPTIONS.keys()), state="readonly", width=58
        )
        self.out_opt_combo.pack(pady=(2, 8))

        # --- Speed & Sentence Completeness Sliders ---
        self.pause_var = tk.DoubleVar(value=0.85)
        self.max_buf_var = tk.DoubleVar(value=5.5)

        pause_row = tk.Frame(settings_frame)
        pause_row.pack(fill="x", pady=(2, 0))
        self.pause_lbl = tk.Label(pause_row, text="Pause Wait (Fast ← 0.85s → Full Sentence):", font=("Segoe UI", 8, "bold"))
        self.pause_lbl.pack(side="left")

        self.pause_slider = tk.Scale(
            settings_frame,
            from_=0.30,
            to=2.00,
            resolution=0.05,
            orient="horizontal",
            variable=self.pause_var,
            showvalue=False,
            command=lambda v: self.pause_lbl.config(text=f"Pause Wait (Fast ← {float(v):.2f}s → Full Sentence):"),
        )
        self.pause_slider.pack(fill="x", pady=(0, 4))

        buf_row = tk.Frame(settings_frame)
        buf_row.pack(fill="x", pady=(2, 0))
        self.buf_lbl = tk.Label(buf_row, text="Max Sentence Window (Short ← 5.5s → Long):", font=("Segoe UI", 8, "bold"))
        self.buf_lbl.pack(side="left")

        self.buf_slider = tk.Scale(
            settings_frame,
            from_=2.0,
            to=10.0,
            resolution=0.5,
            orient="horizontal",
            variable=self.max_buf_var,
            showvalue=False,
            command=lambda v: self.buf_lbl.config(text=f"Max Sentence Window (Short ← {float(v):.1f}s → Long):"),
        )
        self.buf_slider.pack(fill="x")

        # --- 4. Runtime Status & Controls ---
        self.status_var = tk.StringVar(value="Status: Ready")
        tk.Label(self.root, textvariable=self.status_var, fg="#2b579a").pack(anchor="w", **pad)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)

        self.start_btn = tk.Button(
            btn_frame, text="▶ Start Subtitles", width=18, bg="#1f7a1f", fg="white", command=self.start
        )
        self.start_btn.pack(side="left", padx=8)

        self.stop_btn = tk.Button(
            btn_frame, text="■ Stop", width=12, state="disabled", command=self.stop
        )
        self.stop_btn.pack(side="left", padx=8)

    def check_model_status(self):
        _, repo_id = MODELS[self.model_var.get()]
        cached_path = try_to_load_from_cache(repo_id=repo_id, filename="model.bin")
        if isinstance(cached_path, str) and os.path.exists(cached_path):
            self.model_status_var.set("Status: ✓ Downloaded & Ready")
            self.model_status_lbl.config(fg="#1f7a1f")
            self.progress_var.set(100)
            self.dl_detail_var.set("Cached locally on your PC.")
            self.download_btn.config(text="Re-verify / Update", state="normal")
        else:
            self.model_status_var.set("Status: ↓ Not Downloaded Yet")
            self.model_status_lbl.config(fg="#b85c00")
            self.progress_var.set(0)
            self.dl_detail_var.set("Click 'Download Whisper' before starting.")
            self.download_btn.config(text="⬇ Download Whisper", state="normal")

    def _is_pair_installed(self, src_code, tgt_code):
        if not ARGOS_AVAILABLE:
            return False
        try:
            installed = argos_trans.get_installed_languages()
            from_lang = next((l for l in installed if l.code == src_code), None)
            to_lang = next((l for l in installed if l.code == tgt_code), None)
            return bool(from_lang and to_lang and from_lang.get_translation(to_lang))
        except BaseException:
            return False

    def check_argos_status(self):
        if not ARGOS_AVAILABLE:
            self.argos_status_var.set("Argos not installed (Using MyMemory backup)")
            self.argos_status_lbl.config(fg="#b85c00")
            self.argos_single_btn.config(state="disabled")
            self.argos_all_btn.config(state="disabled")
            return

        src, tgt, label = ARGOS_PACK_CHOICES[self.argos_pack_var.get()]
        is_installed = self._is_pair_installed(src, tgt)

        installed_pairs = [
            info[2].split(" → ")[0]
            for info in ARGOS_PACK_CHOICES.values()
            if self._is_pair_installed(info[0], info[1])
        ]

        if is_installed:
            self.argos_status_var.set("Selected Pack: ✓ Installed Offline")
            self.argos_status_lbl.config(fg="#1f7a1f")
            self.argos_single_btn.config(text="✓ Installed", state="disabled")
        else:
            self.argos_status_var.set("Selected Pack: ↓ Not Downloaded")
            self.argos_status_lbl.config(fg="#b85c00")
            self.argos_single_btn.config(text="⬇ Download Pack", state="normal")

        if installed_pairs:
            self.argos_detail_var.set(
                f"Installed ({len(installed_pairs)}/{len(ARGOS_PACK_CHOICES)}): " + ", ".join(installed_pairs)
            )
        else:
            self.argos_detail_var.set("No offline packs installed yet.")

        self.argos_all_btn.config(
            state="normal" if len(installed_pairs) < len(ARGOS_PACK_CHOICES) else "disabled"
        )

    def start_selected_argos_install(self):
        src, tgt, label = ARGOS_PACK_CHOICES[self.argos_pack_var.get()]
        if self._is_pair_installed(src, tgt):
            return
        self._run_argos_batch([(src, tgt, label)])

    def start_all_argos_install(self):
        missing = [
            (src, tgt, label)
            for src, tgt, label in ARGOS_PACK_CHOICES.values()
            if not self._is_pair_installed(src, tgt)
        ]
        if not missing:
            messagebox.showinfo("Already Installed", "All → English packs are already installed!")
            return
        self._run_argos_batch(missing)

    def _run_argos_batch(self, target_triples):
        if not ARGOS_AVAILABLE or self.downloading:
            return
        self.downloading = True
        self.argos_pack_combo.config(state="disabled")
        self.argos_single_btn.config(state="disabled")
        self.argos_all_btn.config(state="disabled")
        self.download_btn.config(state="disabled")
        self.start_btn.config(state="disabled")

        def _worker():
            try:
                self.root.after(0, lambda: self.argos_status_var.set("Fetching Argos package index..."))
                self.root.after(0, lambda: self.argos_status_lbl.config(fg="#0067c0"))
                argos_pkg.update_package_index()
                available = argos_pkg.get_available_packages()

                installed_count = 0
                total = len(target_triples)

                for idx, (src_code, tgt_code, label) in enumerate(target_triples, start=1):
                    self.root.after(
                        0,
                        lambda i=idx, t=total, n=label: self.argos_detail_var.set(
                            f"Downloading ({i}/{t}): {n} (~100 MB)..."
                        ),
                    )
                    pkg = next((p for p in available if p.from_code == src_code and p.to_code == tgt_code), None)
                    if pkg:
                        argos_pkg.install_from_path(pkg.download())
                        installed_count += 1
                        self.root.after(0, self.check_argos_status)

                self.root.after(0, self._reset_download_ui)
                self.root.after(0, self.check_argos_status)
                self.root.after(
                    0,
                    lambda c=installed_count: messagebox.showinfo(
                        "Offline Packs Ready", f"Successfully installed {c} offline language pack(s)!"
                    ),
                )
            except BaseException as exc:
                err = str(exc)
                self.root.after(0, self._reset_download_ui)
                self.root.after(0, self.check_argos_status)
                self.root.after(0, lambda: messagebox.showerror("Argos Install Error", err))

        threading.Thread(target=_worker, daemon=True).start()

    def start_download(self):
        if self.downloading or self.running:
            return
        self.downloading = True
        self.download_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.model_combo.config(state="disabled")
        self.argos_pack_combo.config(state="disabled")
        self.argos_single_btn.config(state="disabled")
        self.argos_all_btn.config(state="disabled")
        self.model_status_var.set("Status: Downloading...")
        self.model_status_lbl.config(fg="#0067c0")
        self.progress_var.set(0)

        _, repo_id = MODELS[self.model_var.get()]
        threading.Thread(target=self._download_worker, args=(repo_id,), daemon=True).start()

    def _download_worker(self, repo_id):
        app_ref = self

        class GuiTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if self.total and self.total > 1_000_000:
                    pct = min(100.0, (self.n / self.total) * 100.0)
                    done_mb = self.n / (1024 * 1024)
                    total_mb = self.total / (1024 * 1024)
                    app_ref.root.after(0, lambda: app_ref.progress_var.set(pct))
                    app_ref.root.after(
                        0,
                        lambda: app_ref.dl_detail_var.set(
                            f"Downloading weights: {done_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)"
                        ),
                    )

        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
                tqdm_class=GuiTqdm,
            )
            self.root.after(0, self._on_download_complete)
        except BaseException as exc:
            err_msg = str(exc)
            self.root.after(0, lambda: messagebox.showerror("Download Error", err_msg))
            self.root.after(0, self._reset_download_ui)

    def _on_download_complete(self):
        self._reset_download_ui()
        self.check_model_status()
        messagebox.showinfo("Download Complete", "The Whisper model has finished downloading and is ready!")

    def _reset_download_ui(self):
        self.downloading = False
        self.download_btn.config(state="normal")
        self.start_btn.config(state="normal")
        self.model_combo.config(state="readonly")
        self.argos_pack_combo.config(state="readonly")
        self.check_argos_status()

    def start(self):
        if self.running or self.downloading:
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.download_btn.config(state="disabled")
        self.argos_pack_combo.config(state="disabled")
        self.argos_single_btn.config(state="disabled")
        self.argos_all_btn.config(state="disabled")
        self.model_combo.config(state="disabled")
        self.stop_btn.config(state="normal")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        if self.overlay is None or not self.overlay.winfo_exists():
            self.overlay = SubtitleOverlay(self.root)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.download_btn.config(state="normal")
        self.model_combo.config(state="readonly")
        self.argos_pack_combo.config(state="readonly")
        self.stop_btn.config(state="disabled")
        if not self.status_var.get().startswith("Error:"):
            self.status_var.set("Status: Stopped")
        self.check_argos_status()
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.destroy()

    def on_close(self):
        self.stop()
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
        self.root.destroy()

    def _audio_callback(self, indata, frames, time, status):
        if self.running:
            self.audio_queue.put(indata.copy())

    def _resample_to_16k(self, audio, orig_sr):
        if orig_sr == TARGET_SAMPLE_RATE or len(audio) == 0:
            return audio
        duration = len(audio) / float(orig_sr)
        target_len = int(round(duration * TARGET_SAMPLE_RATE))
        if target_len <= 0:
            return audio
        return np.interp(
            np.linspace(0.0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    def _translate_text(self, text, detected_lang, target_lang, mode):
        src = detected_lang if detected_lang else "ja"
        if src == target_lang:
            return text

        if mode == "argos" and ARGOS_AVAILABLE:
            try:
                res = argos_trans.translate(text, src, target_lang)
                if res and res.strip():
                    return res
            except BaseException:
                pass

        if MYMEMORY_AVAILABLE:
            try:
                mem_src = "ja-JP" if src == "ja" else src
                mem_tgt = "en-US" if target_lang == "en" else target_lang
                res = MyMemoryTranslator(source=mem_src, target=mem_tgt).translate(text)
                if res and res.strip():
                    return res
            except BaseException as exc:
                self.root.after(0, lambda err=str(exc)[:45]: self.status_var.set(f"Translate Warn: {err}"))

        return text

    def _worker_loop(self):
        try:
            model_size, _ = MODELS[self.model_var.get()]
            if self.model is None or self.loaded_model_size != model_size:
                self.root.after(0, lambda: self.status_var.set(f"Status: Loading {model_size} onto GPU..."))
                try:
                    self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                except BaseException:
                    self.root.after(0, lambda: self.status_var.set(f"Status: CUDA unavailable, loading {model_size} on CPU..."))
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                self.loaded_model_size = model_size
                self.root.after(0, self.check_model_status)

            device_idx = self.device_map.get(self.device_var.get(), None)
            dev_info = sd.query_devices(device_idx, "input")
            native_sr = int(dev_info.get("default_samplerate", TARGET_SAMPLE_RATE))

            self.root.after(0, lambda: self.status_var.set(f"Status: Listening at {native_sr}Hz..."))

            speech_buffer = []
            silence_duration = 0.0

            with sd.InputStream(
                device=device_idx,
                samplerate=native_sr,
                channels=1,
                dtype="float32",
                blocksize=int(native_sr * STEP_SECONDS),
                callback=self._audio_callback,
            ) as stream:
                while self.running and stream.active:
                    try:
                        block = self.audio_queue.get(timeout=0.5).flatten()
                    except queue.Empty:
                        continue

                    rms = np.abs(block).mean()

                    if rms >= SILENCE_THRESHOLD:
                        speech_buffer.append(block)
                        silence_duration = 0.0
                    elif speech_buffer:
                        speech_buffer.append(block)
                        silence_duration += STEP_SECONDS

                    buffered_sec = len(speech_buffer) * STEP_SECONDS

                    # Read live slider values so adjustments apply immediately while video plays
                    pause_limit = float(self.pause_var.get())
                    max_buf_limit = float(self.max_buf_var.get())

                    if speech_buffer and (silence_duration >= pause_limit or buffered_sec >= max_buf_limit):
                        raw_clip = np.concatenate(speech_buffer)
                        speech_buffer.clear()
                        silence_duration = 0.0

                        audio_clip = self._resample_to_16k(raw_clip, native_sr)

                        selected_lang = INPUT_LANGUAGES[self.in_lang_var.get()]
                        mode, target_lang = OUTPUT_OPTIONS[self.out_opt_var.get()]
                        whisper_task = "translate" if mode == "whisper_translate" else "transcribe"

                        segments, info = self.model.transcribe(
                            audio_clip,
                            language=selected_lang,
                            task=whisper_task,
                            beam_size=1,
                            best_of=1,
                            without_timestamps=True,
                            condition_on_previous_text=False,
                            vad_filter=True,
                        )

                        lines = [seg.text.strip() for seg in segments if seg.text.strip()]
                        if lines and self.running:
                            raw_text = " ".join(lines)
                            actual_src_lang = selected_lang if selected_lang else info.language

                            if mode in ("argos", "mymemory") and target_lang:
                                final_text = self._translate_text(raw_text, actual_src_lang, target_lang, mode)
                            else:
                                final_text = raw_text

                            if final_text and self.running:
                                self.root.after(0, lambda t=final_text: self._safe_update_overlay(t))

            if self.running:
                self.root.after(0, lambda: self.status_var.set("Error: Audio stream stopped unexpectedly."))
                self.root.after(0, self.stop)

        except BaseException as exc:
            traceback.print_exc()
            err_msg = f"{type(exc).__name__}: {exc}"
            self.root.after(0, lambda: self.status_var.set(f"Error: {err_msg[:55]}"))
            self.root.after(0, lambda: messagebox.showerror("Worker Stopped", err_msg))
            self.root.after(0, self.stop)

    def _safe_update_overlay(self, text):
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.update_text(text)


if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleControllerApp(root)
    root.mainloop()