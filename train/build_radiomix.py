import os, random, json
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.signal import butter, lfilter

def bandpass(x, sr, lo=300, hi=3400, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return lfilter(b, a, x).astype(np.float32)

def add_clipping(x, clip=0.35):
    return np.clip(x, -clip, clip).astype(np.float32)

def dropout_bursts(x, sr, p=0.3, max_ms=120):
    y = x.copy()
    if random.random() > p:
        return y
    bursts = random.randint(1, 4)
    for _ in range(bursts):
        dur = int(sr * (random.randint(20, max_ms) / 1000.0))
        start = random.randint(0, max(0, len(y) - dur - 1))
        y[start:start+dur] = 0.0
    return y.astype(np.float32)

def mix_snr(clean, noise, snr_db):
    # scale noise to achieve desired SNR
    eps = 1e-9
    c_pow = np.mean(clean**2) + eps
    n_pow = np.mean(noise**2) + eps
    target_n_pow = c_pow / (10**(snr_db/10))
    scale = np.sqrt(target_n_pow / n_pow)
    return (clean + noise * scale).astype(np.float32)

def load_mono(path, sr_target=16000):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr != sr_target:
        # simple resample using numpy (good enough for dataset gen; you can swap to resample_poly)
        t_old = np.linspace(0, 1, len(x), endpoint=False)
        t_new = np.linspace(0, 1, int(len(x)*sr_target/sr), endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)
        sr = sr_target
    peak = np.max(np.abs(x)) + 1e-9
    return (x/peak).astype(np.float32), sr

def main(clean_dir, noise_dir, out_dir, n_examples=2000, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    clean_files = list(Path(clean_dir).rglob("*.wav"))
    noise_files = list(Path(noise_dir).rglob("*.wav"))

    out_dir = Path(out_dir)
    (out_dir / "noisy").mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "manifest.jsonl"

    with manifest.open("w", encoding="utf-8") as f:
        for i in tqdm(range(n_examples)):
            cpath = random.choice(clean_files)
            npath = random.choice(noise_files)

            clean, sr = load_mono(str(cpath), 16000)
            noise, _ = load_mono(str(npath), 16000)

            # match lengths (simple crop/loop)
            if len(noise) < len(clean):
                reps = int(np.ceil(len(clean) / len(noise)))
                noise = np.tile(noise, reps)
            noise = noise[:len(clean)]

            snr = random.uniform(-5, 20)
            noisy = mix_snr(clean, noise, snr)

            # radio channel simulation
            noisy = bandpass(noisy, sr)
            noisy = add_clipping(noisy, clip=random.uniform(0.2, 0.6))
            noisy = dropout_bursts(noisy, sr, p=0.6)

            # target: clean band-limited (more fair for radio)
            target = bandpass(clean, sr)

            noisy_path = out_dir / "noisy" / f"{i:06d}.wav"
            clean_path = out_dir / "clean" / f"{i:06d}.wav"
            sf.write(noisy_path, noisy, sr)
            sf.write(clean_path, target, sr)

            rec = {
                "noisy": str(noisy_path),
                "clean": str(clean_path),
                "snr_db": round(snr, 2)
            }
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--out_dir", default="datasets/radiomix_train")
    ap.add_argument("--n_examples", type=int, default=2000)
    args = ap.parse_args()
    main(args.clean_dir, args.noise_dir, args.out_dir, args.n_examples)
