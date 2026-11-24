# Whisper Real-Time Offline Live Subtitles

A GPU-accelerated desktop application built with Python and Tkinter that captures live system or microphone audio, transcribes speech using **Faster-Whisper**, translates it offline via **Argos Translate** (or online via **MyMemory**), and displays the captions in a frameless, always-on-top draggable subtitle overlay.

---

## Repository Structure

```text
├── live_subtitle_gui.py   # Main GUI application & draggable subtitle overlay
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation & setup guide
├── LICENSE                # MIT License
└── .gitignore             # Excludes local caches, virtualenvs, and model binaries
```

## Features

- Real-Time Speech Recognition: Powered by faster-whisper (large-v3-turbo, large-v3, medium, and small) with automatic CUDA GPU acceleration (int8_float16) and CPU fallback.

- 100% Offline Multi-Language Translation: Built-in Argos Translate pack selector lets you download offline neural translation packs (Japanese, Korean, Chinese, Spanish, French, German, Russian, Hindi, Arabic, Portuguese, Italian → English) directly from the GUI with zero API keys or rate limits.

- Built-In Whisper Model Manager: Check local cache status and download Whisper weights directly from the UI with a live progress bar.

- Draggable, Always-on-Top Subtitle Overlay: Frameless floating subtitle bar that stays above fullscreen or windowed video players and auto-expands vertically for multi-line sentences.

- Live Latency & Sentence Controls: Real-time sliders for Pause Wait (0.30s – 2.00s) and Max Sentence Window (2.0s – 10.0s) let you tune speed vs. full-sentence accuracy while a video is playing.

- Adaptive Audio Resampling: Automatically detects your audio device's native sample rate (e.g., 44.1 kHz or 48 kHz) and resamples cleanly to 16 kHz for Whisper.

## Dependencies (requirements.txt)

All required third-party Python packages are listed in [requirements.txt]

```bash
faster-whisper>=1.0.0
sounddevice>=0.4.6
numpy>=1.24.0
huggingface-hub>=0.20.0
tqdm>=4.65.0
argostranslate>=1.9.0
deep-translator>=1.11.4
```
## Package Overview
| Package | Role in Project | 
| :--- | :--- |
| faster-whisper | CTranslate2-optimized Whisper engine for fast local speech-to-text |
| sounddevice | Captures live audio streams from microphones or virtual loopback devices |
| numpy | "Audio buffer processing, RMS silence detection, and 16 kHz resampling" |
| huggingface-hub | Downloads and caches Whisper model weights locally |
| tqdm | Powers the real-time download progress bar in the GUI |
| argostranslate | Offline neural machine translation engine (no internet or API key required) |
| deep-translator | "Provides MyMemoryTranslator as an online fallback for additional languages (e.g., Sinhala, Tamil)" |
| tkinter | Built-in Python standard library used for the control panel and floating overlay |

## How the Program Works

1. Audio Capture & Resampling (sounddevice + numpy):
Opens an input stream at your selected device's native sample rate in 0.25s micro-blocks and calculates the Root Mean Square (RMS) energy of each block to distinguish active speech from silence.

2. Pause-Triggered Buffering:
Accumulates speech blocks until either a natural pause exceeds your Pause Wait slider setting or continuous speech reaches your Max Sentence Window limit, then resamples the clip to 16,000 Hz mono float32.

3. Speech Recognition (faster-whisper):
Runs WhisperModel with vad_filter=True and beam_size=1 to transcribe the spoken audio (or auto-detect the spoken language when set to Auto-Detect).

4. Translation (argostranslate / deep-translator):

  - In Offline Argos Translate mode, transcribed text is translated locally on CPU (50 – 100 ms) using the installed language pack.

  - If a pack is missing or an online language (such as Sinhala or Tamil) is chosen, it falls back to MyMemoryTranslator.

5. Floating Subtitle Overlay (tkinter.Toplevel):
Updates a borderless, semi-transparent (alpha=0.85), always-on-top window that can be dragged anywhere on screen.

# Installation & How to Run
1. Clone the Repository
```Bash
git clone [https://github.com/](https://github.com/)pbwa3029/Offline-realtime-subtitle-genarator.git
cd Offline-realtime-subtitle-genarator
```
2. Install Requirements
```Bash
  pip install -r requirements.txt
```
3. Enable System Audio Routing (To Caption Videos/Browsers)
   To translate audio playing from YouTube, VLC, or media players instead of your microphone:
     1. Open Windows Sound Control Panel (Win + R → type mmsys.cpl → press Enter).
     2. Go to the Recording tab, right-click in the empty area, and enable Show Disabled Devices.
     3. Right-click Stereo Mix and select Enable (if your PC lacks Stereo Mix, install a free virtual audio cable such as VB-Cable).
4. Run the Application
```Bash
  python live_subtitle_gui.py
```  
5. First-Time Setup in the GUI
  1. **Download Whisper Model**: In Section 1, select large-v3-turbo (recommended for speed) and click ⬇ Download Whisper. Wait for the progress bar to reach 100%.
  2. **Download Offline Translation Pack** : In Section 2, select your source language pair (e.g., Japanese → English (ja → en)) from the dropdown and click ⬇ Download Pack (or ⬇ Install All → EN).
  3. **Configure Audio & Language** : In Section 3, select Stereo Mix (or your microphone) as the Audio Source, pick your Input Language (or Auto-Detect), and select English (Offline Argos Translate).
  4. **Start & Position Subtitles** : Click  **Start Subtitles** , then click and drag the dark floating subtitle bar right underneath your video player.

## Tuning Tips
- Sentences getting cut off early? Increase Pause Wait to 0.90s – 1.20s and Max Sentence Window to 6.0s – 7.5s (especially helpful for Japanese and Korean, where verbs appear at the end of sentences).
- Subtitles lagging behind fast dialogue? Lower Pause Wait to 0.40s – 0.55s and Max Sentence Window to 3.5s.
- Background music preventing pauses from triggering? Increase SILENCE_THRESHOLD in live_subtitle_gui.py from 0.010 to 0.025.

## License
This project is open-source and available under the [MIT License](.LICENCE).
