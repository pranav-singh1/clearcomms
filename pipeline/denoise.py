"""
pipeline/denoise.py — Neural speech denoising via fine-tuned Facebook Denoiser.

Uses a dns64 model fine-tuned on paired radio/clean audio.
Falls back to pretrained dns64 if no fine-tuned checkpoint exists.
"""

import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT_PATH = _PROJECT_ROOT / "models" / "denoiser" / "best_denoiser.th"

_model = None
_device = None


def _init_model():
    global _model, _device

    if _model is not None:
        return

    from denoiser import pretrained

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = pretrained.dns64()

    if _CHECKPOINT_PATH.exists():
        print(f"[Denoise] Loading fine-tuned checkpoint: {_CHECKPOINT_PATH.name}")
        state = torch.load(str(_CHECKPOINT_PATH), map_location=_device, weights_only=True)
        model.load_state_dict(state)
    else:
        print("[Denoise] No fine-tuned checkpoint found, using pretrained dns64")

    model.to(_device)
    model.eval()
    _model = model
    print(f"[Denoise] Ready on {_device}")


def denoise_audio(audio, sr):
    """
    Denoise audio using the fine-tuned Denoiser model.

    Args:
        audio: numpy float32 array, mono
        sr: sample rate

    Returns:
        (denoised_audio, metadata)
        denoised_audio: numpy float32 array at 16kHz
        metadata: dict with denoise_latency_ms
    """
    _init_model()

    from denoiser.dsp import convert_audio

    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    x = convert_audio(x, sr, _model.sample_rate, _model.chin)
    x = x.to(_device)

    t0 = time.time()
    with torch.no_grad():
        out = _model(x)[0]
    latency_ms = (time.time() - t0) * 1000

    out_np = out.squeeze().cpu().numpy().astype(np.float32)

    # Normalize peak
    peak = np.max(np.abs(out_np)) + 1e-9
    out_np = 0.95 * out_np / peak

    meta = {
        "denoise_latency_ms": round(latency_ms, 1),
        "checkpoint": _CHECKPOINT_PATH.name if _CHECKPOINT_PATH.exists() else "pretrained",
    }
    return out_np, meta
