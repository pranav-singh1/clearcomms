cd /mnt/c/Users/sogan_lfd6wlv/clearcomms

cat > pipeline/audio_io.py <<'PY'
"""Audio IO helpers used by the Streamlit demo.

This keeps the UI runnable on a normal laptop (no Qualcomm-specific pieces).

Dependencies are intentionally minimal: numpy, soundfile, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf


@dataclass
class AudioClip:
    samples: np.ndarray  # float32 mono
    sr: int


def load_audio(path: str) -> AudioClip:
    """Load audio as mono float32."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return AudioClip(samples=x.astype(np.float32), sr=int(sr))


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample using scipy.signal.resample (good enough for hackathon demo)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    from scipy.signal import resample

    n = int(round(len(audio) * float(target_sr) / float(orig_sr)))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return resample(audio, n).astype(np.float32)


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int]:
    x = resample(audio, orig_sr, 16000)
    return x, 16000


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize to [-peak, +peak]."""
    x = audio.astype(np.float32)
    m = float(np.max(np.abs(x)) + 1e-9)
    return (peak * x / m).astype(np.float32)


def safe_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Serialize to WAV bytes for Streamlit playback without touching disk."""
    import io

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()
PY
