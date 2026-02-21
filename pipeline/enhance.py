import numpy as np
from scipy.signal import butter, lfilter

def _bandpass(x, sr, lo=300, hi=3400, order=4):
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return lfilter(b, a, x).astype(np.float32)

def _soft_gate(x, thr=0.02):
    # Simple noise gate for walkie-talkie hiss / static
    mag = np.abs(x)
    gate = np.where(mag < thr, mag / max(thr, 1e-6), 1.0)
    return (x * gate).astype(np.float32)

def enhance_audio(audio, sr):
    x = audio.astype(np.float32)
    x = _bandpass(x, sr)
    x = _soft_gate(x, thr=0.02)
    # normalize
    peak = np.max(np.abs(x)) + 1e-9
    x = 0.95 * x / peak
    return x
