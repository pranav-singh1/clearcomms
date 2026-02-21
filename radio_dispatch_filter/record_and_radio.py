import argparse
import datetime as dt
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

import make_radio_dataset as radio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record from mic and convert to dispatch-style radio audio."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional base name for output files (without extension).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate in Hz (default: 16000).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the radio output when finished.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional existing .wav/.mp3 path to process instead of recording.",
    )
    parser.add_argument(
        "--use-external-noise",
        action="store_true",
        help="Enable mixing vetted files from radio_noise into the recording.",
    )
    return parser.parse_args()


def ensure_dirs(clean_dir, radio_dir, noise_dir):
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(radio_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)


def default_stem(prefix):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def record_with_afrecord(output_path, seconds, sample_rate):
    cmd = [
        "afrecord",
        "-f",
        "WAVE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        f"{seconds:.2f}",
        output_path,
    ]
    print(f"Recording for {seconds:.1f}s...")
    print("Speak now.")
    subprocess.run(cmd, check=True)


def ensure_python_package(module_name, pip_name):
    if importlib.util.find_spec(module_name) is not None:
        return
    print(f"Installing missing recording package: {pip_name}")
    subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)


def record_with_sounddevice(output_path, seconds, sample_rate):
    ensure_python_package("sounddevice", "sounddevice")
    sd = importlib.import_module("sounddevice")

    frames = max(1, int(round(seconds * sample_rate)))
    print(f"Recording for {seconds:.1f}s via Python sounddevice...")
    print("Speak now.")

    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = audio.reshape(-1).astype("float32")
    radio.write_float_as_wav(output_path, audio, sample_rate)


def maybe_play(path):
    player = shutil.which("afplay")
    if player is None:
        print("Could not auto-play: 'afplay' not found.")
        return
    subprocess.run([player, path], check=False)


def main():
    args = parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if args.sample_rate < 8000:
        raise ValueError("--sample-rate must be >= 8000")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_dir, "clean_audio")
    radio_dir = os.path.join(base_dir, "radio_audio")
    noise_dir = os.path.join(base_dir, "radio_noise")
    ensure_dirs(clean_dir, radio_dir, noise_dir)

    if args.input:
        source_path = os.path.abspath(args.input)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Input file not found: {source_path}")
        stem = args.name.strip() or os.path.splitext(os.path.basename(source_path))[0]
    else:
        stem = args.name.strip() or default_stem("mic")
        source_path = os.path.join(clean_dir, f"{stem}.wav")
        if shutil.which("afrecord") is not None:
            record_with_afrecord(source_path, args.seconds, args.sample_rate)
        else:
            try:
                record_with_sounddevice(source_path, args.seconds, args.sample_rate)
            except Exception as exc:
                raise RuntimeError(
                    "No working recorder backend found. "
                    "Install 'afrecord' or allow Python package installation for sounddevice. "
                    "You can also pass --input <file.wav|file.mp3>."
                ) from exc

    noise_catalog = []
    if args.use_external_noise:
        radio.ENABLE_EXTERNAL_NOISE_MIX = True
        noise_catalog, rejected_noise = radio.build_noise_catalog(noise_dir)
        if rejected_noise:
            print(f"Ignored {len(rejected_noise)} unsuitable noise file(s).")
    else:
        radio.ENABLE_EXTERNAL_NOISE_MIX = False
        print("External noise mixing disabled (recommended for clean dispatch realism).")

    radio_audio, sr = radio.process_file(source_path, noise_catalog)
    output_path = os.path.join(radio_dir, f"{stem}_radio.wav")
    radio.write_float_as_wav(output_path, radio_audio, sr)

    print(f"Input:  {source_path}")
    print(f"Output: {output_path}")

    if args.play:
        maybe_play(output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
