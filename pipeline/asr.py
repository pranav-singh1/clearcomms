"""
pipeline/asr.py — Whisper via on-device ONNX Runtime (models/).
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


def _resample(audio, orig_sr, target_sr):
    """Resample audio to target sample rate using scipy."""
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample

    num_samples = int(len(audio) * target_sr / orig_sr)
    return resample(audio, num_samples).astype(np.float32)


def _init_backend():
    global _backend
    if _backend is not None:
        return

    cfg = _load_config()
    variant = cfg.get("model_variant", "base_en")

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from src.model import make_whisper_app

    encoder_path = _PROJECT_ROOT / cfg.get("encoder_path", "models/WhisperEncoder.onnx")
    decoder_path = _PROJECT_ROOT / cfg.get("decoder_path", "models/WhisperDecoder.onnx")

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder model not found at {encoder_path}. "
            "Place WhisperEncoder.onnx in models/ (or set encoder_path in model/config.yaml)."
        )
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Decoder model not found at {decoder_path}. "
            "Place WhisperDecoder.onnx in models/ (or set decoder_path in model/config.yaml)."
        )

    app = make_whisper_app(str(encoder_path), str(decoder_path), variant, cfg)
    _backend = {"app": app, "cfg": cfg}
    print(f"[ASR] Loaded on-device Whisper ({variant}) from models/")


def transcribe(audio_path, sr):
    """
    Transcribe an audio file with Whisper via on-device ONNX (models/).

    Args:
        audio_path: path to audio file (WAV, FLAC, etc.)
        sr: sample rate of the audio file

    Returns:
        (transcript_text, metadata_dict)
    """
    _init_backend()

    import soundfile as sf

    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_sec = len(audio) / file_sr
    audio_16k = _resample(audio, file_sr, _WHISPER_SR)

    app = _backend["app"]
    t0 = time.time()
    text = app.transcribe(audio_16k, _WHISPER_SR)
    latency_ms = (time.time() - t0) * 1000

    text = text.strip()

    meta = {
        "asr_latency_ms": round(latency_ms, 1),
        "audio_duration_sec": round(duration_sec, 2),
        "realtime_factor": round(latency_ms / 1000 / max(duration_sec, 0.001), 3),
        "backend": "onnx",
        "model_variant": _backend["cfg"].get("model_variant", "base_en"),
    }

    return text, meta
