# ClearComms — React Frontend

The app has been converted from Streamlit to a **React (Vite)** frontend with a **FastAPI** backend. On-device inference (Qualcomm NPU / QNN) is unchanged.

## Run the app

### 1. Backend (FastAPI)

From the **project root** (where `pipeline/` and `backend/` live), with your Python venv activated:

```bash
pip install fastapi uvicorn[standard] python-multipart
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

> **Windows:** If you see `WinError 10013` (access forbidden) on port 8001, try another port, e.g. `--port 5000`, and set the same port in `frontend/vite.config.ts` → `proxy["/api"].target`.

### 2. Frontend (React)

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api` to the backend at port 8001.

## Branch

All changes are on the **revised-frontend** branch.

## Features (unchanged)

- Upload audio (WAV, FLAC, OGG, MP3, M4A)
- Optional radio preprocess (bandpass + gate) and peak normalization
- Whisper transcription via ONNX (QNN/NPU when models are in `models/`)
- Transcript, performance metrics, and export (transcript.txt, metadata.json)
- Audio playback: original, prepared 16 kHz, and (if enabled) radio-filtered

## Offline TTS with Piper

The React demo includes a **"Speak cleaned transcript (offline)"** button when Piper is configured.
The backend uses `PIPER_MODEL_PATH` and optionally `PIPER_BIN`.

### 1. Download voice model (Windows)

From **project root** in PowerShell:

```powershell
.\setup_piper.ps1
```

This creates `assets\voices\` and downloads `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`.

### 2. Set environment and start backend

In the **same** PowerShell where you start the backend (venv activated):

```powershell
$env:PIPER_MODEL_PATH="C:\Users\hackathon user\Documents\qualhack\simple-whisper-transcription\assets\voices\en_US-lessac-medium.onnx"
# Only if piper is not on PATH:
# $env:PIPER_BIN="C:\path\to\piper.exe"

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

Checks:
- File exists: `...\assets\voices\en_US-lessac-medium.onnx`
- Matching config in same folder: `...\en_US-lessac-medium.onnx.json`
- `piper --help` works (or set `PIPER_BIN` to your `piper.exe`)
- Restart backend after setting env vars

### 3. Install Piper (if not on PATH)

On Windows you need the Piper binary (e.g. from [Piper releases](https://github.com/rhasspy/piper/releases)) or the Python package:

```powershell
pip install piper-tts pathvalidate
piper --help
```

If TTS fails with `ModuleNotFoundError: No module named 'pathvalidate'`, run `pip install pathvalidate`.

If you use a downloaded `piper.exe`, set `$env:PIPER_BIN` to its full path.

### 4. Quick test

With backend running:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/tts" -ContentType "application/json" -Body '{"text":"hello"}' -OutFile tts.wav
# Then play tts.wav
```

Or with curl (if installed):

```bash
curl -X POST http://127.0.0.1:8001/api/tts -H "Content-Type: application/json" -d "{\"text\":\"hello\"}" --output tts.wav
```

## Design

- Black and white, Apple-inspired layout
- System font stack (SF Pro on macOS), clean typography, subtle borders and shadows
