"""
ClearComms API — FastAPI backend for React frontend.
Exposes transcription and model-status; keeps on-device inference unchanged.
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root for pipeline imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from pipeline.asr import transcribe
from pipeline.enhance import enhance_audio
from pipeline.audio_io import load_mono, normalize_peak, resample, save_wav, WHISPER_SR

app = FastAPI(title="ClearComms API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODELS_DIR = _ROOT / "models"


def _model_files_present() -> bool:
    enc = _ROOT / "models" / "WhisperEncoder.onnx"
    dec = _ROOT / "models" / "WhisperDecoder.onnx"
    return enc.exists() and dec.exists()


@app.get("/api/model-status")
def model_status():
    return {"models_found": _model_files_present()}


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    apply_radio_filter: str = Form("true"),
    normalize: str = Form("true"),
):
    apply_radio = apply_radio_filter.lower() in ("true", "1", "yes")
    do_normalize = normalize.lower() in ("true", "1", "yes")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    if suffix.lower() not in (".wav", ".flac", ".ogg", ".mp3", ".m4a"):
        raise HTTPException(400, "Unsupported format. Use WAV, FLAC, OGG, MP3, or M4A.")

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read upload: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(contents)
        input_path = Path(tf.name)

    tmp_dir = _ROOT / "runs"
    tmp_dir.mkdir(exist_ok=True)
    pre16_path = tmp_dir / "preprocessed_16k.wav"
    filt_path = tmp_dir / "radio_filtered_16k.wav"

    try:
        audio, sr = load_mono(str(input_path))
        audio_16k, _ = resample(audio, sr, WHISPER_SR)
        if do_normalize:
            audio_16k = normalize_peak(audio_16k)
        save_wav(str(pre16_path), audio_16k, WHISPER_SR)

        if apply_radio:
            filtered = enhance_audio(audio_16k, WHISPER_SR)
            save_wav(str(filt_path), filtered, WHISPER_SR)
            asr_input = filt_path
        else:
            asr_input = pre16_path

        # Keep preprocessed clip internal; return only user-facing streams.
        audio_filtered_b64 = None
        if apply_radio and filt_path.exists():
            audio_filtered_b64 = base64.b64encode(filt_path.read_bytes()).decode("utf-8")

        duration_sec = round(len(audio) / max(sr, 1), 2)
        payload = {
            "success": True,
            "error": None,
            "text": "",
            "meta": {},
            "audio_filtered_b64": audio_filtered_b64,
            "apply_radio_filter": apply_radio,
            "duration_sec": duration_sec,
            "sample_rate_original": sr,
        }

        try:
            t0 = time.time()
            text, meta = transcribe(str(asr_input), WHISPER_SR)
            ui_total_ms = (time.time() - t0) * 1000.0
            payload["text"] = (text or "").strip() or "(no transcript)"
            payload["meta"] = {**meta, "ui_total_ms": round(ui_total_ms, 1)}
        except FileNotFoundError as e:
            payload["success"] = False
            payload["error"] = "ONNX encoder/decoder not found. Place WhisperEncoder.onnx and WhisperDecoder.onnx in models/."
        except Exception as e:
            payload["success"] = False
            payload["error"] = str(e)

        return payload
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try:
            os.remove(str(input_path))
        except Exception:
            pass
