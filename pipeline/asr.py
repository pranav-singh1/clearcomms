"""
pipeline/asr.py — Whisper via ONNX Runtime.

Two backends, tried in order:
  1. QNN-accelerated (Qualcomm hardware) via model.model
  2. ONNX Runtime CPU via HuggingFace optimum (dev / fallback)
"""

import logging
import os
import sys
import time
import warnings

# Suppress transformers/optimum deprecation noise before they're imported
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _logger_name in ("transformers", "optimum", "optimum.onnxruntime"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import numpy as np
import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _PROJECT_ROOT / "model" / "config.yaml"

_VARIANT_TO_HF = {
    "base_en": "openai/whisper-base.en",
    "small_en": "openai/whisper-small.en",
    "medium_en": "openai/whisper-medium.en",
    "large_en": "openai/whisper-large-v3",
}

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

    # --- Backend 1: QNN-accelerated (Qualcomm hardware) ---
    try:
        root = str(_PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)

        from model.model import make_whisper_app

        encoder = _PROJECT_ROOT / cfg.get("encoder_path", "models/WhisperEncoder.onnx")
        decoder = _PROJECT_ROOT / cfg.get("decoder_path", "models/WhisperDecoder.onnx")

        if encoder.exists() and decoder.exists():
            app = make_whisper_app(str(encoder), str(decoder), variant, cfg)
            _backend = {"type": "qnn", "app": app, "cfg": cfg}
            print(f"[ASR] Loaded QNN-accelerated Whisper ({variant})")
            return
    except Exception:
        pass

    # --- Backend 2: ONNX Runtime CPU via optimum ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        model_id = _VARIANT_TO_HF.get(variant, "openai/whisper-base.en")

        # Cache exported ONNX model locally under models/
        onnx_dir = _PROJECT_ROOT / "models" / f"whisper-{variant}-onnx"

        print(f"[ASR] Loading {model_id} via ONNX Runtime ...")

        if onnx_dir.exists():
            model = ORTModelForSpeechSeq2Seq.from_pretrained(str(onnx_dir))
        else:
            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_id, export=True
            )
            onnx_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(onnx_dir))

        processor = WhisperProcessor.from_pretrained(model_id)

    _backend = {
        "type": "ort",
        "model": model,
        "processor": processor,
        "cfg": cfg,
    }
    print(f"[ASR] Ready — Whisper {variant} on ONNX Runtime CPU")


def transcribe(audio_path, sr):
    """
    Transcribe an audio file with Whisper via ONNX Runtime.

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

    # Whisper requires 16 kHz
    audio_16k = _resample(audio, file_sr, _WHISPER_SR)

    if _backend["type"] == "qnn":
        app = _backend["app"]
        t0 = time.time()
        text = app.transcribe(audio_16k, _WHISPER_SR)
        latency_ms = (time.time() - t0) * 1000

    elif _backend["type"] == "ort":
        model = _backend["model"]
        processor = _backend["processor"]

        t0 = time.time()
        inputs = processor(
            audio_16k, sampling_rate=_WHISPER_SR, return_tensors="pt"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_ids = model.generate(inputs.input_features)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        latency_ms = (time.time() - t0) * 1000

    text = text.strip()

    meta = {
        "asr_latency_ms": round(latency_ms, 1),
        "audio_duration_sec": round(duration_sec, 2),
        "realtime_factor": round(latency_ms / 1000 / max(duration_sec, 0.001), 3),
        "backend": _backend["type"],
        "model_variant": _backend["cfg"].get("model_variant", "base_en"),
    }

    return text, meta
