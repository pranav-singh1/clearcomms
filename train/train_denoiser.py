"""
train/train_denoiser.py — Fine-tune Facebook Denoiser (dns64) on paired radio data.

Uses the existing clean/radio pairs in radio_dispatch_filter/.

Usage:
    python train/train_denoiser.py                        # defaults
    python train/train_denoiser.py --epochs 50 --lr 1e-4  # custom
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RadioPairsDataset(Dataset):
    """Pre-loads all paired clean/radio audio into memory, returns random crops."""

    def __init__(self, clean_dir, radio_dir, crop_seconds=3.0, sr=16000):
        from scipy.signal import resample as scipy_resample

        self.sr = sr
        self.crop_len = int(crop_seconds * sr)

        clean_files = sorted(Path(clean_dir).glob("*.flac")) + sorted(Path(clean_dir).glob("*.wav"))
        radio_files = sorted(Path(radio_dir).glob("*.wav"))

        clean_map = {f.stem: f for f in clean_files}

        # Pre-load everything into memory (80 pairs ≈ 30MB)
        self.pairs = []
        print("[Dataset] Pre-loading audio into memory ...")
        for rf in radio_files:
            clean_stem = rf.stem.replace("_radio", "")
            if clean_stem not in clean_map:
                continue

            clean_audio, csr = sf.read(str(clean_map[clean_stem]), dtype="float32")
            radio_audio, rsr = sf.read(str(rf), dtype="float32")

            if clean_audio.ndim > 1:
                clean_audio = clean_audio.mean(axis=1)
            if radio_audio.ndim > 1:
                radio_audio = radio_audio.mean(axis=1)

            # Resample to target sr
            if csr != sr:
                n = int(len(clean_audio) * sr / csr)
                clean_audio = scipy_resample(clean_audio, n).astype(np.float32)
            if rsr != sr:
                n = int(len(radio_audio) * sr / rsr)
                radio_audio = scipy_resample(radio_audio, n).astype(np.float32)

            # Align lengths
            min_len = min(len(clean_audio), len(radio_audio))
            self.pairs.append((clean_audio[:min_len], radio_audio[:min_len]))

        log(f"[Dataset] Loaded {len(self.pairs)} pairs into memory")

    def __len__(self):
        return len(self.pairs) * 20

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)
        clean, radio = self.pairs[pair_idx]

        # Random crop
        if len(clean) > self.crop_len:
            start = random.randint(0, len(clean) - self.crop_len)
            clean = clean[start:start + self.crop_len]
            radio = radio[start:start + self.crop_len]
        else:
            pad = self.crop_len - len(clean)
            clean = np.pad(clean, (0, pad))
            radio = np.pad(radio, (0, pad))

        # Copy to avoid mutating cached data
        clean = clean.copy()
        radio = radio.copy()

        # Normalize by radio peak
        peak = max(np.max(np.abs(radio)), 1e-6)
        radio = radio / peak
        clean = clean / peak

        return (
            torch.from_numpy(radio).unsqueeze(0),
            torch.from_numpy(clean).unsqueeze(0),
        )


# ---------------------------------------------------------------------------
# Loss: L1 + multi-resolution STFT
# ---------------------------------------------------------------------------

def stft_loss(predicted, target, fft_sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss — captures both spectral magnitude and convergence."""
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=predicted.device)

        pred_stft = torch.stft(predicted.squeeze(1), n_fft, hop, window=window, return_complex=True)
        tgt_stft = torch.stft(target.squeeze(1), n_fft, hop, window=window, return_complex=True)

        pred_mag = pred_stft.abs()
        tgt_mag = tgt_stft.abs()

        # Spectral convergence + log magnitude
        loss += (pred_mag - tgt_mag).abs().mean()
        loss += (torch.log(pred_mag + 1e-7) - torch.log(tgt_mag + 1e-7)).abs().mean()

    return loss / len(fft_sizes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"[Train] Device: {device}")

    # Load pretrained model
    from denoiser import pretrained
    model = pretrained.dns64().to(device)
    model.train()
    log(f"[Train] Loaded dns64 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Dataset
    dataset = RadioPairsDataset(
        clean_dir=args.clean_dir,
        radio_dir=args.radio_dir,
        crop_seconds=args.crop_sec,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for radio_batch, clean_batch in loader:
            radio_batch = radio_batch.to(device)
            clean_batch = clean_batch.to(device)

            denoised = model(radio_batch)

            # Combined loss
            l1 = nn.functional.l1_loss(denoised, clean_batch)
            spec = stft_loss(denoised, clean_batch)
            loss = l1 + args.stft_weight * spec

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        log(f"  Epoch {epoch:3d}/{args.epochs} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = save_dir / "best_denoiser.th"
            torch.save(model.state_dict(), ckpt_path)

    # Save final
    final_path = save_dir / "final_denoiser.th"
    torch.save(model.state_dict(), final_path)
    print(f"\n[Train] Done. Best loss: {best_loss:.4f}")
    log(f"  Best:  {save_dir / 'best_denoiser.th'}")
    log(f"  Final: {final_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", default="radio_dispatch_filter/clean_audio")
    p.add_argument("--radio_dir", default="radio_dispatch_filter/radio_audio")
    p.add_argument("--save_dir", default="models/denoiser")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--crop_sec", type=float, default=3.0)
    p.add_argument("--stft_weight", type=float, default=0.5)
    args = p.parse_args()
    train(args)
