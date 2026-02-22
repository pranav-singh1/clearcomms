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

## Online TTS with Deepgram

The React demo includes a **"Speak cleaned transcript"** button when Deepgram is configured.
The backend calls Deepgram directly; no API key is exposed to the browser.

### 1. Set environment and start backend

In the **same** terminal where you start the backend (venv activated):

```bash
export DEEPGRAM_API_KEY="dg_..."
export DEEPGRAM_TTS_MODEL="aura-2-thalia-en"
export DEEPGRAM_TTS_CACHE_MAX="50"
export DEEPGRAM_TTS_CACHE_TTL_SEC="600"

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

Windows PowerShell:

```powershell
$env:DEEPGRAM_API_KEY="dg_..."
$env:DEEPGRAM_TTS_MODEL="aura-2-thalia-en"
$env:DEEPGRAM_TTS_CACHE_MAX="50"
$env:DEEPGRAM_TTS_CACHE_TTL_SEC="600"

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

Notes:
- Deepgram TTS requires internet access. The rest of the pipeline can remain offline.
- If `DEEPGRAM_API_KEY` is missing, the UI disables the TTS button and shows a tooltip.

### 2. Quick test

With backend running:

```bash
curl -X POST http://127.0.0.1:8001/api/tts -H "Content-Type: application/json" -d "{\"text\":\"hello\"}" --output tts.mp3
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/api/tts" -ContentType "application/json" -Body '{"text":"hello"}' -OutFile tts.mp3
```

## Design

- Black and white, Apple-inspired layout
- System font stack (SF Pro on macOS), clean typography, subtle borders and shadows
