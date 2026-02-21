"""
Smoke tests for pipeline/asr.py

Run from project root:
    python tests/test_asr.py

Creates runs/test_outputs/ with a clear before/after for each sample:
    runs/test_outputs/
      sample_01/
        1_radio_input.wav        <- noisy radio audio (before)
        2_enhanced.wav           <- after bandpass + noise gate
        3_transcript.txt         <- what Whisper produced from each
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf
from pipeline.asr import transcribe
from pipeline.enhance import enhance_audio


RADIO_DIR = Path("radio_dispatch_filter/radio_audio")
CLEAN_DIR = Path("radio_dispatch_filter/clean_audio")
OUT_DIR = Path("runs/test_outputs")

NUM_SAMPLES = 5


def run_pipeline_samples():
    """Run the full pipeline on several samples and save everything."""

    # Fresh output dir each run
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    radio_files = sorted(RADIO_DIR.glob("*.wav"))[:NUM_SAMPLES]
    assert radio_files, f"No WAV files in {RADIO_DIR}"

    for i, radio_path in enumerate(radio_files, 1):
        sample_dir = OUT_DIR / f"sample_{i:02d}"
        sample_dir.mkdir()

        # --- 1. Copy radio input (the "before") ---
        input_dst = sample_dir / "1_radio_input.wav"
        shutil.copy2(radio_path, input_dst)

        # --- 2. Enhance (bandpass + noise gate → the "after" audio) ---
        audio, sr = sf.read(str(radio_path), always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        enhanced = enhance_audio(audio, sr)
        enhanced_dst = sample_dir / "2_enhanced.wav"
        sf.write(str(enhanced_dst), enhanced, sr)

        # --- 3. Transcribe both and save results ---
        text_raw, meta_raw = transcribe(str(input_dst), sr)
        text_enh, meta_enh = transcribe(str(enhanced_dst), sr)

        # Also get clean baseline if available
        clean_name = radio_path.stem.replace("_radio", "") + ".flac"
        clean_path = CLEAN_DIR / clean_name
        if clean_path.exists():
            shutil.copy2(clean_path, sample_dir / "0_clean_original.flac")
            text_clean, meta_clean = transcribe(str(clean_path), 16000)
        else:
            text_clean, meta_clean = "(no clean file found)", {}

        transcript_text = (
            f"=== CLEAN (original, no distortion) ===\n"
            f"{text_clean}\n\n"
            f"=== RADIO (distorted input) ===\n"
            f"{text_raw}\n"
            f"  latency: {meta_raw['asr_latency_ms']:.0f}ms  |  "
            f"duration: {meta_raw['audio_duration_sec']:.1f}s\n\n"
            f"=== ENHANCED (bandpass + noise gate → Whisper) ===\n"
            f"{text_enh}\n"
            f"  latency: {meta_enh['asr_latency_ms']:.0f}ms  |  "
            f"duration: {meta_enh['audio_duration_sec']:.1f}s\n"
        )

        (sample_dir / "3_transcript.txt").write_text(transcript_text)

        meta_out = {"radio": meta_raw, "enhanced": meta_enh}
        if meta_clean:
            meta_out["clean"] = meta_clean
        (sample_dir / "4_metadata.json").write_text(
            json.dumps(meta_out, indent=2)
        )

        # Print summary
        short = radio_path.stem[:45]
        print(f"sample_{i:02d}  {short}")
        print(f"  clean:    {text_clean[:80]}")
        print(f"  radio:    {text_raw[:80]}")
        print(f"  enhanced: {text_enh[:80]}")
        print()

    print(f"Output saved to: {OUT_DIR.resolve()}")


def verify_metadata():
    """Quick check that metadata has all expected keys."""
    radio_file = next(RADIO_DIR.glob("*.wav"))
    _, meta = transcribe(str(radio_file), 44100)

    expected = {"asr_latency_ms", "audio_duration_sec", "realtime_factor",
                "backend", "model_variant"}
    missing = expected - set(meta.keys())
    assert not missing, f"Missing metadata keys: {missing}"
    print(f"Metadata keys OK: {sorted(meta.keys())}")


if __name__ == "__main__":
    run_pipeline_samples()
    verify_metadata()
    print("\nALL TESTS PASSED")
