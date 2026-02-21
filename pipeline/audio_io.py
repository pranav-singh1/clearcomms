"""pipeline/audio_io.py

Small utilities for loading and preparing audio for Whisper.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WHISPER_SR = 16_000

def load_mono(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)

def resample(audio: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SR) -> Tuple[np.ndarray, int]:
    if orig_sr == target_sr:
        return audio.astype(np.float32), orig_sr
    g = int(np.gcd(orig_sr, target_sr))
    up = target_sr // g
    down = orig_sr // g
    y = resample_poly(audio.astype(np.float32), up, down).astype(np.float32)
    return y, target_sr

def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(audio)) + 1e-9)
    return (audio / m * peak).astype(np.float32)

def save_wav(path: str, audio: np.ndarray, sr: int = WHISPER_SR) -> None:
    sf.write(path, np.clip(audio, -1.0, 1.0).astype(np.float32), sr)
