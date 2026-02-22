import numpy as np
from scipy.signal import butter, lfilter

def _bandpass(x, sr, lo=300, hi=3400, order=4):
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return lfilter(b, a, x).astype(np.float32)

def _soft_gate(x, thr=0.02):
    mag = np.abs(x)
    gate = np.where(mag < thr, mag / max(thr, 1e-6), 1.0)
    return (x * gate).astype(np.float32)

def enhance_audio(audio, sr, intensity=0.5):
    """
    intensity: 0.0 = very mild (wide band, light gate)
               0.5 = baseline (300-3400 Hz, gate thr=0.02) — original behaviour
               1.0 = heavy radio (narrow band, aggressive gate, added static)
    """
    t = max(0.0, min(1.0, float(intensity)))
    x = audio.astype(np.float32)

    # Bandpass: 150-5000 Hz at t=0, 300-3400 Hz at t=0.5, 700-2500 Hz at t=1.0
    lo = 150 + t * (700 - 150)
    hi = 5000 + t * (2500 - 5000)
    x = _bandpass(x, sr, lo=int(lo), hi=int(hi))

    # Gate: 0.005 at t=0, 0.02 at t=0.5, 0.07 at t=1.0
    gate_thr = 0.005 + t * (0.07 - 0.005)
    x = _soft_gate(x, thr=gate_thr)

    # Static noise: none below t=0.5, ramps up to ~2% amplitude at t=1.0
    if t > 0.5:
        noise_scale = (t - 0.5) * 2.0 * 0.02
        noise = np.random.default_rng().standard_normal(len(x)).astype(np.float32)
        x = x + noise * noise_scale

    peak = np.max(np.abs(x)) + 1e-9
    x = 0.95 * x / peak
    return x
