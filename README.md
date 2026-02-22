# ClearComms

**Fully offline AI that converts noisy radio communication into accurate transcripts and structured incident summaries.** Optimized to run locally on Qualcomm AI laptops with no internet connection.

This project improves transcription reliability in emergency and field communication using optimized on-device speech recognition and local language models. It is based on [simple-whisper-transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription) and extends it with structured extraction via on-device LLaMA (Qualcomm Genie).

---

## Table of Contents

- [Problem](#problem)
- [Solution](#solution)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [What You Need Installed](#what-you-need-installed)
- [Quick Start: Run Backend & Frontend](#quick-start-run-backend--frontend)
- [Whisper Models (ONNX)](#whisper-models-onnx)
- [Optional: On-Device LLaMA (Genie)](#optional-on-device-llama-genie)
- [Optional: TTS (Deepgram)](#optional-tts-deepgram)
- [Code Organization](#code-organization)
- [Datasets & References](#datasets--references)
- [Building an Executable](#building-an-executable)
- [Contributing](#contributing)

---

## Problem

First responders and field teams rely on radios that produce noisy, clipped, and hard-to-understand audio. This leads to:

- Misheard instructions  
- Missed location or hazard details  
- Slower and less effective response  

Internet access is often unavailable in these environments, making cloud solutions unreliable.

**ClearComms** solves this by running transcription and structuring **fully on device**.

---

## Solution

ClearComms processes radio audio through three local stages:

### 1. Offline Speech Recognition

Audio is transcribed locally using an optimized Whisper model (e.g. [Whisper Base En](https://aihub.qualcomm.com/compute/models/whisper_base_en?domain=Audio) from Qualcomm AI Hub, or Whisper large-v3-turbo).

Engineering focus includes:

- Running Whisper with **ONNX Runtime** (QNN/NPU on Qualcomm hardware)
- Model optimization and quantization
- Parameter tuning for noisy radio audio
- Low-latency on-device inference

### 2. Structured Incident Extraction

The transcript can be processed by a **local LLM** (e.g. LLaMA via Qualcomm Genie) to turn raw speech into structured outputs.

**Example**

| Raw transcript | Structured output |
|----------------|-------------------|
| *unit 12 need backup at 5th street possible fire* | **Location:** 5th Street<br>**Request:** Backup<br>**Incident:** Fire<br>**Urgency:** High |

This makes communication faster to interpret and act on.

### 3. Offline End-to-End Pipeline

```
Radio Audio
    ↓
Whisper (ONNX, on device)
    ↓
Transcript
    ↓
Local LLaMA (Genie) [optional]
    ↓
Structured incident / action summary
```

Everything runs fully offline.

---

## Key Features

- **Fully offline operation** — No internet required for transcription or LLaMA.
- **Optimized Whisper** — ONNX Runtime with QNN/NPU on Qualcomm hardware.
- **Structured extraction** — On-device LLaMA (Genie) for action items and suggested actions.
- **Designed for Qualcomm AI hardware** — Snapdragon X Elite (e.g. Dell Latitude 7455).
- **Fast, reliable transcription** in noisy environments.
- **Simple UI** — React (Vite) frontend, FastAPI backend; upload or record, then see raw transcript + Llama output.

---

## Tech Stack

- **Python** (backend, pipeline)
- **Whisper** (ONNX Runtime, Qualcomm AI Hub / QNN)
- **LLaMA** (local inference via Qualcomm Genie)
- **FastAPI** (backend API)
- **React + Vite** (frontend)
- **Qualcomm AI Hub** tools, ONNX Runtime

---

## What You Need Installed

- **Python 3.11** (recommended; project uses 3.11.9 on Windows)
- **Node.js & npm** (for the React frontend)
- **FFmpeg** — for audio. On Windows: download from [FFmpeg Windows builds](https://github.com/BtbN/FFmpeg-Builds/releases), extract (e.g. to `C:\Program Files\ffmpeg`), and add the `bin` folder to your PATH.
- **Qualcomm AI / QAIRT** (for Genie/Llama; only if using on-device LLaMA)
- **Windows** (tested on Windows 11; Snapdragon X Elite)

No GPU or cloud account required for core transcription; NPU/Genie are used when available.

---

## Quick Start: Run Backend & Frontend

### 1. Clone and prepare Python env

```powershell
git clone https://github.com/thatrandomfrenchdude/simple-whisper-transcription.git
cd simple-whisper-transcription

python -m venv whisper-venv
.\whisper-venv\Scripts\Activate.ps1   # Windows

pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart
```

### 2. Whisper models (required for transcription)

- Create a `models` folder at the project root.
- Add ONNX encoder/decoder (e.g. from Qualcomm AI Hub or your own export).  
  Example (AI Hub):  
  `python -m qai_hub_models.models.whisper_base_en.export --target-runtime onnx`  
  then copy the generated encoder/decoder from `build` into `models`.
- If you use a different variant (e.g. large-v3-turbo), point `config.yaml` at those ONNX files and set `model_variant` accordingly.

### 3. Config

Create `config.yaml` in the project root (see [Whisper Models (ONNX)](#whisper-models-onnx) for a minimal example). At minimum you need `encoder_path` and `decoder_path` under `models/`.

### 4. Start the backend

From the **project root**, with the venv activated:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

If port 8001 is in use (e.g. `WinError 10013`), use another port (e.g. `--port 5000`) and set the same port in `frontend/vite.config.ts` → `proxy["/api"].target`.

### 5. Start the frontend

In a **second terminal**:

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The UI proxies `/api` to the backend. You can upload audio (WAV, FLAC, OGG, MP3, M4A) or record from the microphone, then run the pipeline and see the raw transcript and (if enabled) the Llama-revised output.

---

## Whisper Models (ONNX)

- **Sample rate:** 16 kHz mono (handled by the pipeline).
- **Config:** In `config.yaml` set at least:
  - `encoder_path`: e.g. `models/WhisperEncoder.onnx`
  - `decoder_path`: e.g. `models/WhisperDecoder.onnx`
  - Optionally `model_variant` (e.g. `base_en` or `large_v3_turbo`) to match your ONNX export.
- **Hardware:** Built and tested on Snapdragon X Elite (e.g. Dell Latitude 7455, 32 GB RAM, Windows 11). ONNX runs with QNN when the models are present; otherwise CPU fallback.

---

## Optional: On-Device LLaMA (Genie)

The app can run **Qualcomm Genie** (`genie-t2t-run.exe`) to revise or structure the Whisper transcript (e.g. action items, suggested actions). Genie is a native executable; Python only shells out—no separate Python venv for Llama.

### Setup Genie (PowerShell)

1. **Dot-source the env script** so `genie-t2t-run.exe` is on PATH:
   ```powershell
   . .\scripts\setup_genie_env.ps1
   ```
2. **Set the Genie bundle directory** (folder with `genie_config.json`):
   ```powershell
   $env:GENIE_BUNDLE_DIR = "C:\path\to\your\llama\bundle"
   ```

### Test revision

```powershell
python scripts/run_llama_revision.py --text "bravo two copy that we are oscar mike"
```

### Enable in the app

Before starting the backend:

```powershell
$env:ENABLE_LLAMA_REVISION = "1"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

The frontend will show the raw Whisper transcript immediately and display **Loading...** in the “Reconstructed transcript (Llama)” box until the revision request returns. Optional env: `GENIE_CONFIG`, `GENIE_EXE`, `GENIE_TIMEOUT_S` (see `llama_on_device/README.md`).

---

## Optional: TTS (Deepgram)

The React UI can speak the **Whisper** transcript (TTS uses the raw transcript, not the Llama output). TTS is **online** (Deepgram); the rest of the pipeline stays offline.

In the same terminal where you start the backend (venv activated):

**PowerShell:**

```powershell
$env:DEEPGRAM_API_KEY = "dg_..."
$env:DEEPGRAM_TTS_MODEL = "aura-2-arcas-en"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

If `DEEPGRAM_API_KEY` is not set, the TTS button is disabled in the UI.

---

## Code Organization

- **Backend:** `backend/main.py` — FastAPI app; transcribe endpoint, optional `/api/revise` for Llama, TTS endpoints.
- **Pipeline:** `pipeline/asr.py` (Whisper), `pipeline/enhance.py` (radio DSP), `pipeline/audio_io.py` (load/resample/save).
- **Whisper model:** `src/model.py` — ONNX + QNN wrapper; supports base_en and large_v3_turbo (and variants via config).
- **Live CLI:** `src/LiveTranscriber.py` — Live mic → Whisper (no React).
- **Llama on device:** `llama_on_device/` — Prompts and Genie subprocess (`genie_llama.py`, `prompts.py`).
- **Frontend:** `frontend/` — React (Vite); upload/record, transcribe, show raw + reconstructed transcript, latency, export.

---

## Datasets & References

- **LibriSpeech ASR corpus** (OpenSLR 12): [https://www.openslr.org/12](https://www.openslr.org/12) — Large-scale (1000 hours) read English speech at 16 kHz; useful for training or evaluating ASR in clean and “other” conditions.
- **Whisper:** Qualcomm AI Hub [Whisper Base En](https://aihub.qualcomm.com/compute/models/whisper_base_en?domain=Audio); OpenAI Whisper large-v3-turbo for larger models.
- **Base repo:** [simple-whisper-transcription](https://github.com/thatrandomfrenchdude/simple-whisper-transcription).

---

## Building an Executable

To build a standalone Whisper transcriber (no React/backend):

1. With the venv activated: `.\build.ps1` or `python build_executable.py`
2. Find `WhisperTranscriber.exe` in `dist/`
3. Copy `config.yaml`, `mel_filters.npz`, and the `models/` folder into `dist/`
4. Run the executable (see [BUILD_EXECUTABLE.md](BUILD_EXECUTABLE.md) for details)

---

## Demo Summary

- **Input:** Noisy walkie-talkie (or any) audio — upload or record from the mic.  
- **Output:** Raw transcript (Whisper) + optional structured/revised summary (Llama), plus latency and export.  
- **Environment:** All core processing runs locally with no internet.

---

## Why It Matters

ClearComms makes critical communication understandable and actionable in the exact environments where reliability matters most: when networks are down, latency is critical, and every word can change the outcome.

---

## Contributing

Contributions are welcome. Please review [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) ([code of conduct](CODE_OF_CONDUCT.md)).
