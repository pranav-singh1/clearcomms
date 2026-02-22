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

## Design

- Black and white, Apple-inspired layout
- System font stack (SF Pro on macOS), clean typography, subtle borders and shadows
