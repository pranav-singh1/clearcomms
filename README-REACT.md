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

The React demo includes a **"Speak cleaned transcript (offline)"** button in the output panel.
It calls `POST /api/tts` on the FastAPI backend, which runs Piper locally and returns `audio/wav`.

### 1. Install Piper

Install Piper so `piper` is available on your PATH, or set `PIPER_BIN` to the executable path.

Examples:

```bash
# Linux (pip package that provides Piper CLI)
pip install piper-tts

# Verify
piper --help
```

On Windows, you can also download Piper binaries and point `PIPER_BIN` to `piper.exe`.

### 2. Download a voice model

Download a Piper ONNX voice model and place it under `assets/voices/` (or any local path), for example:

```bash
mkdir -p assets/voices
# Example files (names vary by selected voice):
# - en_US-lessac-medium.onnx
# - en_US-lessac-medium.onnx.json
```

Important: keep both files together. Piper needs:
- `your-voice.onnx`
- `your-voice.onnx.json`

### 3. Configure environment variables

Set these before starting backend:

```bash
# required
export PIPER_MODEL_PATH=/absolute/path/to/assets/voices/your-voice.onnx

# optional (default is `piper`)
export PIPER_BIN=/absolute/path/to/piper
```

Windows PowerShell:

```powershell
$env:PIPER_MODEL_PATH="C:\path\to\assets\voices\your-voice.onnx"
$env:PIPER_BIN="C:\path\to\piper.exe"   # optional
```

### 4. Manual test

With backend running:

```bash
curl -X POST http://127.0.0.1:8001/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"hello"}' \
  --output tts.wav
```

Expected:
- HTTP 200
- response content type `audio/wav`
- playable `tts.wav`

## Design

- Black and white, Apple-inspired layout
- System font stack (SF Pro on macOS), clean typography, subtle borders and shadows
