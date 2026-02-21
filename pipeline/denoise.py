"""
pipeline/denoise.py — Neural speech denoising via Facebook Denoiser.

Lazy-loads the pretrained dns64 model on first call.
"""

import time
import numpy as np
import torch

_model = None


def _init_model():
    global _model
    if _model is not None:
        return

    from denoiser import pretrained

    print("[Denoise] Loading Facebook Denoiser (dns64) ...")
    _model = pretrained.dns64()
    _model.eval()
    print("[Denoise] Ready")


def denoise_audio(audio, sr):
    """
    Denoise audio using Facebook's pretrained Denoiser.

    Args:
        audio: numpy float32 array, mono
        sr: sample rate (will be resampled to 16kHz if needed)

    Returns:
        (denoised_audio, metadata_dict)
        denoised_audio: numpy float32 array at original sr
        metadata: dict with denoise_latency_ms
    """
    _init_model()

    from scipy.signal import resample

    orig_sr = sr
    x = audio.astype(np.float32)

    # Denoiser expects 16kHz
    if sr != 16000:
        n_samples = int(len(x) * 16000 / sr)
        x = resample(x, n_samples).astype(np.float32)
        sr = 16000

    # numpy -> torch: [1, 1, samples]
    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        denoised = _model(tensor)[0]
    latency_ms = (time.time() - t0) * 1000

    # torch -> numpy
    out = denoised.squeeze().numpy().astype(np.float32)

    # Resample back to original sr if we changed it
    if orig_sr != 16000:
        n_samples = int(len(out) * orig_sr / 16000)
        out = resample(out, n_samples).astype(np.float32)

    # Normalize peak
    peak = np.max(np.abs(out)) + 1e-9
    out = 0.95 * out / peak

    meta = {"denoise_latency_ms": round(latency_ms, 1)}
    return out, meta
