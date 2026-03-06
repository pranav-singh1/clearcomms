"""
pipeline/asr.py — Whisper via openai-whisper (PyTorch, works on Mac).
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _PROJECT_ROOT / "config.yaml"

_WHISPER_SR = 16_000  # Whisper expects 16 kHz

# Lazy-loaded backend singleton
_backend = None


def _load_config():
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _init_backend():
    global _backend
    if _backend is not None:
        return

    cfg = _load_config()
    variant = cfg.get("model_variant", "base.en")

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from src.model import get_whisper_model

    model = get_whisper_model(variant)
    _backend = {"model": model, "cfg": cfg}
    print(f"[ASR] Loaded Whisper ({variant}) via openai-whisper (PyTorch)")


def transcribe(audio_path, sr):
    """
    Transcribe an audio file with Whisper.

    Args:
        audio_path: path to audio file (WAV, FLAC, etc.)
        sr: sample rate of the audio file (used for metadata only)

    Returns:
        (transcript_text, metadata_dict)
    """
    _init_backend()

    import soundfile as sf

    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_sec = len(audio) / file_sr

    model = _backend["model"]
    t0 = time.time()
    result = model.transcribe(str(audio_path), language="en", fp16=False)
    latency_ms = (time.time() - t0) * 1000

    text = (result.get("text") or "").strip()

    meta = {
        "asr_latency_ms": round(latency_ms, 1),
        "audio_duration_sec": round(duration_sec, 2),
        "realtime_factor": round(latency_ms / 1000 / max(duration_sec, 0.001), 3),
        "backend": "whisper-pytorch",
        "model_variant": _backend["cfg"].get("model_variant", "base.en"),
    }

    return text, meta
