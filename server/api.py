from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.signal import butter, lfilter

from pipeline.audio_io import load_audio, normalize_peak, resample_to_16k, safe_wav_bytes
from pipeline.asr import transcribe as asr_transcribe
from pipeline.cleanup import cleanup_transcript
from pipeline.extract import extract_incident
from pipeline.llm_client import LLMConfig, cleanup_and_extract

app = FastAPI(title="ClearComms API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Audio preprocessing
# -----------------------------


def bandpass_radio(x: np.ndarray, sr: int, lo: int = 300, hi: int = 3400, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def soft_gate(x: np.ndarray, thr: float = 0.02) -> np.ndarray:
    mag = np.abs(x)
    gate = np.where(mag < thr, mag / max(thr, 1e-6), 1.0)
    return (x * gate).astype(np.float32)


# -----------------------------
# ASR backends
# -----------------------------


def _mock_asr(wav_path: str) -> tuple[str, dict]:
    name = Path(wav_path).stem.lower()
    canned = (
        "engine 12 respond 235 mapple street smoke visible need backup"
        if "maple" in name or "sample" in name
        else "unit 4 to dispatch patient injured requesting medical assistance"
    )
    return canned, {
        "backend": "mock",
        "asr_latency_ms": 5.0,
        "audio_duration_sec": None,
        "realtime_factor": None,
    }


def _run_asr(backend: str, wav_path: str, sr_hint: int) -> tuple[str, dict]:
    if backend == "mock":
        return _mock_asr(wav_path)
    return asr_transcribe(wav_path, sr_hint)


# -----------------------------
# Helpers
# -----------------------------


def _demo_audio_path(demo_name: str) -> Optional[Path]:
    demo_dir = Path("radio_dispatch_filter") / "radio_audio"
    candidate = (demo_dir / demo_name).resolve()
    if demo_dir.exists() and candidate.exists() and candidate.suffix.lower() in {".wav", ".flac", ".ogg"}:
        return candidate
    return None


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg"}:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(upload.file.read())
        return tf.name


def _save_bytes_to_temp_wav(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        tf.write(wav_bytes)
        return tf.name


# -----------------------------
# API models
# -----------------------------


class ExtractRequest(BaseModel):
    text: str
    mode: str = "cleanup_and_extract"  # or "extract_only"
    llm_backend: str = "on_device"  # on_device|mock|openai|ollama
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_max_tokens: int = 256


# -----------------------------
# Routes
# -----------------------------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/demos")
def list_demos() -> dict:
    demo_dir = Path("radio_dispatch_filter") / "radio_audio"
    if not demo_dir.exists():
        return {"files": []}
    files = sorted(p.name for p in demo_dir.glob("*.wav"))
    return {"files": files}


@app.post("/api/process")
def process_audio(
    file: Optional[UploadFile] = File(default=None),
    demo_name: Optional[str] = Form(default=None),
    asr_backend: str = Form(default="whisper"),
    llm_backend: str = Form(default="on_device"),
    llm_base_url: Optional[str] = Form(default=None),
    llm_model: Optional[str] = Form(default=None),
    llm_max_tokens: int = Form(default=256),
    use_radio_bp: bool = Form(default=True),
    use_gate: bool = Form(default=True),
    do_normalize: bool = Form(default=True),
) -> dict:
    if not file and not demo_name:
        raise HTTPException(status_code=400, detail="Provide file or demo_name")

    input_path = None
    cleanup_paths: list[str] = []
    try:
        if file:
            input_path = _save_upload_to_temp(file)
            cleanup_paths.append(input_path)
        elif demo_name:
            demo_path = _demo_audio_path(demo_name)
            if not demo_path:
                raise HTTPException(status_code=404, detail="Demo file not found")
            input_path = str(demo_path)

        clip = load_audio(input_path)
        x, sr = clip.samples, clip.sr

        pre_t0 = time.time()
        x16, sr16 = resample_to_16k(x, sr)
        if use_radio_bp:
            x16 = bandpass_radio(x16, sr16)
        if use_gate:
            x16 = soft_gate(x16, thr=0.02)
        if do_normalize:
            x16 = normalize_peak(x16, peak=0.95)
        pre_ms = (time.time() - pre_t0) * 1000.0

        processed_wav_bytes = safe_wav_bytes(x16, sr16)
        processed_path = _save_bytes_to_temp_wav(processed_wav_bytes)
        cleanup_paths.append(processed_path)

        asr_t0 = time.time()
        raw_text, asr_meta = _run_asr(asr_backend, processed_path, sr16)
        asr_ms = (time.time() - asr_t0) * 1000.0

        llm_meta = {}
        cleanup_meta = {}
        extract_meta = {}
        if llm_backend in {"mock", "openai", "ollama"}:
            cfg = LLMConfig(
                mode=llm_backend,
                base_url=llm_base_url or ("http://localhost:1234" if llm_backend == "openai" else "http://localhost:11434"),
                model=llm_model or "llama-3.1-8b-instruct",
                max_tokens=int(llm_max_tokens),
            )
            llm_out, llm_meta = cleanup_and_extract(raw_text, cfg)
            cleaned = (llm_out.get("cleaned_transcript") or "").strip() or raw_text
            incident = {
                "request_type": llm_out.get("request_type"),
                "urgency": llm_out.get("urgency"),
                "location": llm_out.get("location"),
                "units": llm_out.get("units"),
                "hazards": llm_out.get("hazards"),
                "actions": llm_out.get("actions"),
                "uncertainties": llm_out.get("uncertainties"),
            }
        elif llm_backend == "on_device":
            cleaned, cleanup_meta = cleanup_transcript(raw_text)
            incident, extract_meta = extract_incident(cleaned)
            llm_meta = {"llm_backend": "on_device"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown llm_backend: {llm_backend}")

        duration_sec = float(len(x16) / max(sr16, 1))
        total_ms = pre_ms + asr_ms
        total_ms += float(llm_meta.get("llm_latency_ms", 0.0))
        total_ms += float(cleanup_meta.get("cleanup_latency_ms", 0.0))
        total_ms += float(extract_meta.get("extract_latency_ms", 0.0))

        return {
            "raw_transcript": raw_text,
            "cleaned_transcript": cleaned,
            "incident": incident,
            "meta": {
                "preprocess_ms": round(pre_ms, 1),
                "asr_ms_wall": round(asr_ms, 1),
                "audio_duration_sec": round(duration_sec, 3),
                "end_to_end_ms": round(total_ms, 1),
                "end_to_end_realtime_factor": round((total_ms / 1000.0) / max(duration_sec, 1e-3), 3),
                "asr_meta": asr_meta,
                "llm_meta": llm_meta,
                "cleanup_meta": cleanup_meta,
                "extract_meta": extract_meta,
            },
        }
    finally:
        for p in cleanup_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.post("/api/extract")
def extract_from_text(payload: ExtractRequest) -> dict:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    cleaned = payload.text.strip()
    llm_meta = {}
    cleanup_meta = {}
    extract_meta = {}

    if payload.mode == "extract_only":
        incident, extract_meta = extract_incident(cleaned)
        return {
            "cleaned_transcript": cleaned,
            "incident": incident,
            "meta": {
                "llm_meta": llm_meta,
                "cleanup_meta": cleanup_meta,
                "extract_meta": extract_meta,
            },
        }

    if payload.llm_backend in {"mock", "openai", "ollama"}:
        cfg = LLMConfig(
            mode=payload.llm_backend,
            base_url=payload.llm_base_url or ("http://localhost:1234" if payload.llm_backend == "openai" else "http://localhost:11434"),
            model=payload.llm_model or "llama-3.1-8b-instruct",
            max_tokens=int(payload.llm_max_tokens),
        )
        llm_out, llm_meta = cleanup_and_extract(cleaned, cfg)
        cleaned = (llm_out.get("cleaned_transcript") or "").strip() or cleaned
        incident = {
            "request_type": llm_out.get("request_type"),
            "urgency": llm_out.get("urgency"),
            "location": llm_out.get("location"),
            "units": llm_out.get("units"),
            "hazards": llm_out.get("hazards"),
            "actions": llm_out.get("actions"),
            "uncertainties": llm_out.get("uncertainties"),
        }
    elif payload.llm_backend == "on_device":
        cleaned, cleanup_meta = cleanup_transcript(cleaned)
        incident, extract_meta = extract_incident(cleaned)
        llm_meta = {"llm_backend": "on_device"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown llm_backend: {payload.llm_backend}")

    return {
        "cleaned_transcript": cleaned,
        "incident": incident,
        "meta": {
            "llm_meta": llm_meta,
            "cleanup_meta": cleanup_meta,
            "extract_meta": extract_meta,
        },
    }
