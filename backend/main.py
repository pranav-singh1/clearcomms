"""
ClearComms API — FastAPI backend for React frontend.
Exposes transcription and model-status; keeps on-device inference unchanged.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
_MAX_TTS_CHARS = 2000


class TTSRequest(BaseModel):
    text: str


def _model_files_present() -> bool:
    enc = _ROOT / "models" / "WhisperEncoder.onnx"
    dec = _ROOT / "models" / "WhisperDecoder.onnx"
    return enc.exists() and dec.exists()


@app.get("/api/model-status")
def model_status():
    return {"models_found": _model_files_present()}


def _tts_available() -> bool:
    model_path_raw = os.getenv("PIPER_MODEL_PATH", "").strip()
    if not model_path_raw:
        return False
    model_path = Path(model_path_raw)
    if not model_path.is_file():
        return False
    config_path = model_path.with_suffix(model_path.suffix + ".json")
    return config_path.is_file()


@app.get("/api/tts-status")
def tts_status():
    return {"available": _tts_available()}


@app.post("/api/tts")
def api_tts(payload: TTSRequest):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    if len(text) > _MAX_TTS_CHARS:
        raise HTTPException(400, f"text is too long (max {_MAX_TTS_CHARS} chars)")

    piper_bin = os.getenv("PIPER_BIN", "piper")
    model_path_raw = os.getenv("PIPER_MODEL_PATH", "").strip()
    if not model_path_raw:
        raise HTTPException(500, "Missing PIPER_MODEL_PATH. Set it to your Piper .onnx voice model file.")

    model_path = Path(model_path_raw)
    if not model_path.is_file():
        raise HTTPException(500, f"Piper model not found: {model_path}")

    model_config_path = model_path.with_suffix(model_path.suffix + ".json")
    if not model_config_path.is_file():
        raise HTTPException(
            500,
            f"Missing Piper model config: {model_config_path}. Piper expects the matching .json next to the .onnx file.",
        )

    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            output_path = Path(tf.name)

        cmd = [piper_bin, "--model", str(model_path), "--output_file", str(output_path)]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=90,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip() or "Unknown Piper error."
            raise HTTPException(500, f"Offline TTS failed. Check Piper installation/model. Details: {stderr}")

        if not output_path.exists():
            raise HTTPException(500, "Offline TTS failed: Piper did not create an output WAV file.")

        wav_bytes = output_path.read_bytes()
        if not wav_bytes:
            raise HTTPException(500, "Offline TTS failed: Piper returned an empty WAV file.")

        return Response(content=wav_bytes, media_type="audio/wav")
    except FileNotFoundError:
        raise HTTPException(500, f"Piper executable not found: '{piper_bin}'. Set PIPER_BIN or install Piper.")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Offline TTS timed out while running Piper.")
    finally:
        if output_path and output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass


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

        # Always encode preprocessed clips for playback (even if transcribe fails)
        audio_prepared_b64 = None
        audio_filtered_b64 = None
        if pre16_path.exists():
            audio_prepared_b64 = base64.b64encode(pre16_path.read_bytes()).decode("utf-8")
        if apply_radio and filt_path.exists():
            audio_filtered_b64 = base64.b64encode(filt_path.read_bytes()).decode("utf-8")

        duration_sec = round(len(audio) / max(sr, 1), 2)
        payload = {
            "success": True,
            "error": None,
            "text": "",
            "cleaned_transcript": "",
            "meta": {},
            "audio_prepared_b64": audio_prepared_b64,
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
            payload["cleaned_transcript"] = payload["text"]
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
