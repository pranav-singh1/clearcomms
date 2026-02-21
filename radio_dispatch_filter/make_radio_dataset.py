import importlib.util
import math
import os
import subprocess
import sys
import tarfile
import urllib.request

# =====================
# CONFIG
# =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "clean_audio")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "radio_audio")
NOISE_FOLDER = os.path.join(BASE_DIR, "radio_noise")
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac"}

# Strict dispatch realism defaults.
# Keep external file mixing OFF unless you explicitly want it.
ENABLE_EXTERNAL_NOISE_MIX = False
EXTERNAL_NOISE_MIX_PROB = 0.75
MIN_NOISE_FLATNESS = 0.18
NOISE_NAME_BLOCKLIST = (
    "music",
    "song",
    "beat",
    "melody",
    "instrumental",
    "bpm",
    "raven",
    "test1",
)

# Skip obviously non-dispatch content from clean inputs.
SKIP_MUSICY_INPUT_NAMES = True
INPUT_NAME_BLOCKLIST = (
    "music",
    "song",
    "beat",
    "melody",
    "instrumental",
    "bpm",
    "raven",
)

AUTO_DOWNLOAD_SAMPLE = True
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_jackson_0.wav",
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/1_jackson_0.wav",
]
SAMPLE_FILENAME = "auto_sample.wav"

# Auto-bootstrap clean speech set (so you don't have to add files manually).
AUTO_BOOTSTRAP_CLEAN_SET = os.environ.get("RADIO_AUTO_BOOTSTRAP", "1") != "0"
MIN_CLEAN_FILE_COUNT = int(os.environ.get("RADIO_MIN_CLEAN_FILES", "30"))
PREFER_LIBRISPEECH_BOOTSTRAP = os.environ.get("RADIO_USE_LIBRISPEECH", "1") != "0"
LIBRISPEECH_TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_CACHE_DIR = os.path.join(BASE_DIR, ".cache")
LIBRISPEECH_ARCHIVE_PATH = os.path.join(LIBRISPEECH_CACHE_DIR, "test-clean.tar.gz")
LIBRISPEECH_PROGRESS_EVERY = 10
FSDD_BASE_URL = (
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/"
)
FSDD_SPEAKER = "jackson"
FSDD_MAX_INDEX = 49
FSDD_DOWNLOAD_ATTEMPT_MULTIPLIER = 6

# Set to an integer for reproducible output.
RANDOM_SEED = None


def ensure_dependencies():
    required = {
        "numpy": "numpy",
        "scipy": "scipy",
        "miniaudio": "miniaudio",
    }
    missing = [
        pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    print(
        f"Installing missing packages in current Python ({sys.executable}): "
        f"{', '.join(missing)}"
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_dependencies()

import miniaudio
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, chirp, resample_poly, sosfilt


if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)


# =====================
# GENERAL HELPERS
# =====================

def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-9))


def fit_length(x, target_len):
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(x) < target_len:
        reps = int(np.ceil(target_len / len(x)))
        x = np.tile(x, reps)
    if len(x) > target_len:
        start = np.random.randint(0, len(x) - target_len + 1)
        x = x[start : start + target_len]
    return x.astype(np.float32, copy=False)


def resample_audio(x, source_sr, target_sr):
    if int(source_sr) == int(target_sr):
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(source_sr) // g
    return resample_poly(x, up, down).astype(np.float32)


def normalize_audio(x, peak=0.95):
    max_abs = float(np.max(np.abs(x)) + 1e-9)
    if max_abs <= 0.0:
        return x
    return x * (peak / max_abs)


def name_has_blocked_keyword(path, blocklist):
    name = os.path.basename(path).lower()
    return any(token in name for token in blocklist)


def estimate_spectral_flatness(audio):
    if len(audio) < 2048:
        return 0.0
    frame = 2048
    hop = 1024
    window = np.hanning(frame).astype(np.float32)
    flatness_vals = []
    for start in range(0, len(audio) - frame + 1, hop):
        seg = audio[start : start + frame] * window
        power = np.square(np.abs(np.fft.rfft(seg))) + 1e-12
        geom = np.exp(np.mean(np.log(power)))
        arith = np.mean(power)
        flatness_vals.append(float(geom / arith))
    if not flatness_vals:
        return 0.0
    return float(np.median(np.asarray(flatness_vals)))


# =====================
# RADIO EFFECT FUNCTIONS
# =====================

def bandpass(data, sr, low=500, high=2500, order=8):
    nyq = 0.5 * sr
    safe_low = max(low, 20)
    safe_high = min(high, nyq * 0.95)
    if safe_low >= safe_high:
        return data
    # SOS form is much more numerically stable than direct-form IIR at higher orders.
    sos = butter(order, [safe_low / nyq, safe_high / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, data.astype(np.float64))
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def distort(data, gain):
    return np.tanh(data * gain).astype(np.float32)


def clip(data, clip_level):
    return np.clip(data, -clip_level, clip_level).astype(np.float32)


def add_static_bed(data, sr, amount):
    static = np.random.normal(0.0, 1.0, len(data)).astype(np.float32)
    static = bandpass(static, sr, low=350, high=3400, order=4)

    # Slow amplitude movement to mimic changing RF noise floor.
    mod_freq = np.random.uniform(0.5, 2.0)
    t = np.arange(len(data), dtype=np.float32) / float(sr)
    mod = 0.6 + 0.4 * np.sin(2.0 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi))
    static = static * mod

    static = static / (float(np.max(np.abs(static))) + 1e-9)
    return data + static * amount


def add_bursty_dropouts(data, sr):
    out = data.copy()
    n = len(out)
    if n < 8:
        return out

    # Radio fade/chop style micro dropouts (10-50 ms).
    burst_count = np.random.randint(4, 12)
    for _ in range(burst_count):
        dur = int(sr * np.random.uniform(0.01, 0.05))
        if dur <= 1 or dur >= n:
            continue
        start = np.random.randint(0, n - dur)
        attenuation = np.random.uniform(0.12, 0.45)
        out[start : start + dur] *= attenuation

        pop_len = min(int(sr * 0.002), n - start)
        if pop_len > 1:
            out[start : start + pop_len] += np.random.normal(0, 0.025, pop_len).astype(
                np.float32
            )

    return out


def add_interference_bursts(data, sr):
    out = data.copy()
    n = len(out)
    if n < 8:
        return out

    burst_count = np.random.randint(1, 3)
    for _ in range(burst_count):
        dur = int(sr * np.random.uniform(0.02, 0.12))
        if dur <= 4 or dur >= n:
            continue
        start = np.random.randint(0, n - dur)

        t = np.arange(dur, dtype=np.float32) / float(sr)
        t1 = float(max(dur - 1, 1)) / float(sr)
        sweep = chirp(
            t,
            f0=np.random.uniform(600, 1200),
            f1=np.random.uniform(1800, 3200),
            t1=t1,
            method="linear",
        ).astype(np.float32)
        noise = np.random.normal(0, 1.0, dur).astype(np.float32)
        env = np.hanning(dur).astype(np.float32)
        burst = (0.7 * sweep + 0.3 * noise) * env * np.random.uniform(0.012, 0.055)
        out[start : start + dur] += burst

    return out


def apply_squelch_gate(data, sr):
    if len(data) == 0:
        return data

    env_window = max(1, int(sr * 0.01))
    env_kernel = np.ones(env_window, dtype=np.float32) / env_window
    envelope = np.convolve(np.abs(data), env_kernel, mode="same")

    percentile = np.random.uniform(40, 65)
    threshold = np.percentile(envelope, percentile) * np.random.uniform(0.8, 1.1)
    gate = (envelope > threshold).astype(np.float32)

    smooth = max(1, int(sr * 0.008))
    if smooth > 1:
        win = np.hanning(smooth * 2 + 1).astype(np.float32)
        win /= float(np.sum(win) + 1e-9)
        gate = np.convolve(gate, win, mode="same")

    gate = 0.12 + 0.88 * np.clip(gate, 0.0, 1.0)
    return data * gate


def codec_crunch(data, sr):
    # Simulate low-bitrate narrowband radio path (around 11k and coarse quantization).
    narrow_sr = 11025
    low = resample_audio(data, sr, narrow_sr)

    # Mu-law style compand and coarse quantization.
    mu = 255.0
    companded = np.sign(low) * np.log1p(mu * np.abs(low)) / np.log1p(mu)
    levels = float(np.random.choice([128, 256]))
    q = (levels - 1.0) / 2.0
    quantized = np.round((companded + 1.0) * q) / q - 1.0
    expanded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(mu)) / mu)

    restored = resample_audio(expanded.astype(np.float32), narrow_sr, sr)
    return fit_length(restored, len(data))


def make_tone(sr, freq, duration_s, amplitude):
    length = max(1, int(sr * duration_s))
    t = np.arange(length, dtype=np.float32) / float(sr)
    tone = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    fade = max(1, int(0.006 * sr))
    if fade * 2 < length:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env = np.ones(length, dtype=np.float32)
        env[:fade] = ramp
        env[-fade:] = ramp[::-1]
        tone *= env
    return tone * amplitude


def make_click(sr, duration_s, amplitude):
    length = max(1, int(sr * duration_s))
    click = np.random.normal(0.0, 1.0, length).astype(np.float32)
    click = np.concatenate(([click[0]], np.diff(click))).astype(np.float32)
    decay = np.exp(-np.linspace(0.0, 4.0, length, dtype=np.float32))
    return click * decay * amplitude


def add_ptt_clicks_and_beeps(data, sr):
    out = data.copy()
    n = len(out)
    if n < 32:
        return out

    # PTT click at start and occasional tail click.
    start_click = make_click(
        sr,
        duration_s=np.random.uniform(0.002, 0.007),
        amplitude=np.random.uniform(0.01, 0.04),
    )
    out[: len(start_click)] += start_click[:n]
    if np.random.rand() < 0.40:
        end_click = make_click(
            sr,
            duration_s=np.random.uniform(0.002, 0.007),
            amplitude=np.random.uniform(0.01, 0.035),
        )
        end_len = min(len(end_click), n)
        out[n - end_len :] += end_click[:end_len]

    # Rare tiny tones: mostly clicks/static instead of game-like beeps.
    if np.random.rand() < 0.12:
        start_tone = make_tone(
            sr,
            freq=np.random.uniform(950, 1180),
            duration_s=np.random.uniform(0.010, 0.026),
            amplitude=np.random.uniform(0.01, 0.03),
        )
        st_len = min(len(start_tone), n)
        out[:st_len] += start_tone[:st_len]

    if np.random.rand() < 0.06:
        end_tone = make_tone(
            sr,
            freq=np.random.uniform(780, 940),
            duration_s=np.random.uniform(0.008, 0.020),
            amplitude=np.random.uniform(0.008, 0.025),
        )
        et_len = min(len(end_tone), n)
        out[n - et_len :] += end_tone[:et_len]

    # Rare tiny marker beep mid-transmission.
    if np.random.rand() < 0.03 and n > int(0.3 * sr):
        marker = make_tone(
            sr,
            freq=np.random.uniform(1350, 1800),
            duration_s=np.random.uniform(0.008, 0.018),
            amplitude=np.random.uniform(0.006, 0.02),
        )
        start = np.random.randint(int(0.2 * n), max(int(0.2 * n) + 1, int(0.8 * n)))
        mk_len = min(len(marker), n - start)
        out[start : start + mk_len] += marker[:mk_len]

    return out


def add_transmission_edges(data, sr):
    out = data.copy()
    n = len(out)
    if n < 16:
        return out

    # Start-channel open chirp/pop (very short).
    pre_len = max(1, min(n, int(sr * np.random.uniform(0.01, 0.025))))
    t = np.arange(pre_len, dtype=np.float32) / float(sr)
    t1 = float(max(pre_len - 1, 1)) / float(sr)
    pre = chirp(
        t,
        f0=np.random.uniform(1200, 1700),
        f1=np.random.uniform(1800, 2400),
        t1=t1,
        method="linear",
    ).astype(np.float32)
    pre *= np.hanning(pre_len).astype(np.float32)
    out[:pre_len] += pre * np.random.uniform(0.005, 0.02)

    pop = np.random.normal(0, 1.0, pre_len).astype(np.float32)
    pop = bandpass(pop, sr, low=900, high=3200, order=4)
    out[:pre_len] += pop * np.random.uniform(0.0025, 0.010)

    # End-channel squelch tail.
    squelch_len = max(1, min(n, int(sr * np.random.uniform(0.02, 0.06))))
    sq = np.random.normal(0, 1.0, squelch_len).astype(np.float32)
    sq = bandpass(sq, sr, low=1200, high=3500, order=4)
    decay = np.exp(-np.linspace(0.0, 4.0, squelch_len, dtype=np.float32))
    sq *= decay * np.random.uniform(0.006, 0.028)
    out[n - squelch_len :] += sq[: min(squelch_len, n)]

    return out


def add_wind_buffeting(data, sr, amount):
    if amount <= 0.0 or len(data) == 0:
        return data
    wind = np.random.normal(0.0, 1.0, len(data)).astype(np.float32)
    wind = bandpass(wind, sr, low=25, high=180, order=4)
    t = np.arange(len(data), dtype=np.float32) / float(sr)
    gust = 0.5 + 0.5 * np.sin(
        2.0 * np.pi * np.random.uniform(0.08, 0.22) * t + np.random.uniform(0, 2 * np.pi)
    )
    wind *= gust.astype(np.float32)
    wind /= float(np.max(np.abs(wind)) + 1e-9)
    return data + wind * amount


def add_impulse_noise(data, sr):
    out = data.copy()
    n = len(out)
    if n < 8:
        return out
    event_count = np.random.randint(2, 7)
    for _ in range(event_count):
        width = max(1, int(sr * np.random.uniform(0.0008, 0.003)))
        start = np.random.randint(0, n)
        end = min(n, start + width)
        span = end - start
        if span <= 0:
            continue
        pulse = np.random.normal(0, 1.0, span).astype(np.float32)
        env = np.exp(-np.linspace(0.0, 4.0, span, dtype=np.float32))
        out[start:end] += pulse * env * np.random.uniform(0.006, 0.03)
    return out


def apply_agc(data, target_rms, max_gain=12.0):
    current = rms(data)
    gain = min(max_gain, target_rms / (current + 1e-9))
    return data * gain


def mix_external_noise(data, sr, noise_catalog):
    if not ENABLE_EXTERNAL_NOISE_MIX:
        return data
    if not noise_catalog:
        return data
    if np.random.rand() > EXTERNAL_NOISE_MIX_PROB:
        return data

    noise_path = np.random.choice(noise_catalog)
    noise, noise_sr = read_audio_as_float(noise_path)
    if len(noise) == 0:
        return data

    noise = resample_audio(noise, noise_sr, sr)
    noise = fit_length(noise, len(data))
    noise = bandpass(noise, sr, low=250, high=3600, order=4)

    speech_rms = max(0.02, rms(data))
    desired_snr_db = np.random.uniform(1.5, 10.0)
    desired_noise_rms = speech_rms / (10 ** (desired_snr_db / 20.0))
    noise_scale = desired_noise_rms / (rms(noise) + 1e-9)
    return data + noise * noise_scale


# =====================
# I/O HELPERS
# =====================

def read_wav_as_float(filepath):
    sr, audio = wavfile.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio.astype(np.float32), axis=1)

    if audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(audio.dtype, np.integer):
        max_abs = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_abs
    else:
        audio = audio.astype(np.float32)

    return audio, int(sr)


def read_miniaudio_file_as_float(filepath):
    decoded = miniaudio.decode_file(
        filepath,
        output_format=miniaudio.SampleFormat.FLOAT32,
    )
    sr = int(decoded.sample_rate)
    channels = int(decoded.nchannels)
    audio = np.asarray(decoded.samples, dtype=np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def read_mp3_as_float(filepath):
    return read_miniaudio_file_as_float(filepath)


def read_flac_as_float(filepath):
    return read_miniaudio_file_as_float(filepath)


def read_audio_as_float(filepath):
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == ".wav":
        return read_wav_as_float(filepath)
    if ext == ".mp3":
        return read_mp3_as_float(filepath)
    if ext == ".flac":
        return read_flac_as_float(filepath)
    raise ValueError(f"Unsupported input extension: {ext}")


def write_float_as_wav(filepath, audio, sr):
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(filepath, int(sr), pcm16)


def list_audio_files(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def select_dispatch_input_files(files):
    accepted = []
    skipped = []
    for path in files:
        if SKIP_MUSICY_INPUT_NAMES and name_has_blocked_keyword(path, INPUT_NAME_BLOCKLIST):
            skipped.append(path)
            continue
        accepted.append(path)
    return accepted, skipped


def build_noise_catalog(folder):
    candidates = list_audio_files(folder)
    accepted = []
    rejected = []
    for path in candidates:
        try:
            if name_has_blocked_keyword(path, NOISE_NAME_BLOCKLIST):
                rejected.append((path, "blocked-name"))
                continue

            noise, sr = read_audio_as_float(path)
            if len(noise) < max(1, int(sr * 0.5)):
                rejected.append((path, "too-short"))
                continue

            flatness = estimate_spectral_flatness(noise[: min(len(noise), int(sr * 20))])
            if flatness < MIN_NOISE_FLATNESS:
                rejected.append((path, f"tonal(flatness={flatness:.3f})"))
                continue
            accepted.append(path)
        except Exception as exc:
            rejected.append((path, f"decode-error({exc})"))
    return accepted, rejected


# =====================
# PROCESS FILE
# =====================

def process_file(filepath, noise_catalog):
    audio, sr = read_audio_as_float(filepath)
    if len(audio) == 0:
        return audio, sr

    # Core dispatch chain.
    audio = bandpass(
        audio,
        sr,
        low=np.random.uniform(470, 560),
        high=np.random.uniform(2300, 2700),
        order=8,
    )
    audio = codec_crunch(audio, sr)
    audio = apply_squelch_gate(audio, sr)
    audio = add_bursty_dropouts(audio, sr)
    audio = add_static_bed(audio, sr, amount=np.random.uniform(0.006, 0.025))
    audio = add_wind_buffeting(audio, sr, amount=np.random.uniform(0.0, 0.012))
    audio = add_impulse_noise(audio, sr)
    audio = mix_external_noise(audio, sr, noise_catalog)
    audio = add_interference_bursts(audio, sr)
    audio = add_transmission_edges(audio, sr)
    audio = add_ptt_clicks_and_beeps(audio, sr)
    audio = distort(audio, gain=np.random.uniform(3.2, 5.3))
    audio = clip(audio, clip_level=np.random.uniform(0.32, 0.56))
    audio = apply_agc(audio, target_rms=np.random.uniform(0.09, 0.12))
    audio = normalize_audio(audio, peak=np.random.uniform(0.90, 0.98))

    return audio.astype(np.float32), sr


def ensure_input_audio_exists():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    inputs = list_audio_files(INPUT_FOLDER)
    if inputs or not AUTO_DOWNLOAD_SAMPLE:
        return inputs

    target = os.path.join(INPUT_FOLDER, SAMPLE_FILENAME)
    for url in SAMPLE_URLS:
        try:
            print(f"No input audio found. Downloading sample: {url}")
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded sample to: {target}")
            return list_audio_files(INPUT_FOLDER)
        except Exception as exc:
            print(f"Download failed from {url}: {exc}")

    raise RuntimeError(
        f"No .wav/.mp3/.flac files found in '{INPUT_FOLDER}' and sample download failed. "
        f"Add at least one WAV, MP3, or FLAC file, then run again."
    )


def ensure_librispeech_archive():
    os.makedirs(LIBRISPEECH_CACHE_DIR, exist_ok=True)
    if os.path.exists(LIBRISPEECH_ARCHIVE_PATH):
        return True
    try:
        print(f"Downloading LibriSpeech test-clean archive from {LIBRISPEECH_TEST_CLEAN_URL}")
        urllib.request.urlretrieve(LIBRISPEECH_TEST_CLEAN_URL, LIBRISPEECH_ARCHIVE_PATH)
        print(f"Saved archive: {LIBRISPEECH_ARCHIVE_PATH}")
        return True
    except Exception as exc:
        print(f"LibriSpeech download failed: {exc}")
        return False


def bootstrap_librispeech_clean_audio(existing_files, needed):
    if not PREFER_LIBRISPEECH_BOOTSTRAP or needed <= 0:
        return list_audio_files(INPUT_FOLDER)
    if not ensure_librispeech_archive():
        return list_audio_files(INPUT_FOLDER)

    extracted = 0
    try:
        with tarfile.open(LIBRISPEECH_ARCHIVE_PATH, "r:gz") as archive:
            members = [
                m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith(".flac")
            ]
            np.random.shuffle(members)

            for member in members:
                if extracted >= needed:
                    break
                parts = member.name.split("/")
                if len(parts) >= 4:
                    speaker = parts[-3]
                    chapter = parts[-2]
                    utterance = os.path.splitext(parts[-1])[0]
                    local_name = f"librispeech_{speaker}_{chapter}_{utterance}.flac"
                else:
                    local_name = f"librispeech_{os.path.basename(member.name)}"

                local_path = os.path.join(INPUT_FOLDER, local_name)
                if os.path.exists(local_path):
                    continue

                source = archive.extractfile(member)
                if source is None:
                    continue
                with source:
                    with open(local_path, "wb") as out_file:
                        out_file.write(source.read())
                extracted += 1
                if extracted % LIBRISPEECH_PROGRESS_EVERY == 0 or extracted == needed:
                    print(f"Extracted {extracted}/{needed} LibriSpeech clips...")
    except Exception as exc:
        print(f"LibriSpeech extraction failed: {exc}")

    if extracted > 0:
        print(f"Added {extracted} LibriSpeech clip(s) to '{INPUT_FOLDER}'.")
    return list_audio_files(INPUT_FOLDER)


def bootstrap_clean_audio_batch(existing_files):
    if not AUTO_BOOTSTRAP_CLEAN_SET:
        return existing_files
    if len(existing_files) >= MIN_CLEAN_FILE_COUNT:
        return existing_files

    start_count = len(existing_files)
    needed = MIN_CLEAN_FILE_COUNT - start_count
    print(
        f"Found {start_count} clean file(s). "
        f"Auto-downloading up to {needed} additional clean clips..."
    )

    existing_files = bootstrap_librispeech_clean_audio(existing_files, needed)
    if len(existing_files) >= MIN_CLEAN_FILE_COUNT:
        return existing_files

    needed = MIN_CLEAN_FILE_COUNT - len(existing_files)
    candidates = []
    for index in range(FSDD_MAX_INDEX + 1):
        for digit in range(10):
            remote_name = f"{digit}_{FSDD_SPEAKER}_{index}.wav"
            local_name = f"auto_{remote_name}"
            local_path = os.path.join(INPUT_FOLDER, local_name)
            if os.path.exists(local_path):
                continue
            candidates.append((f"{FSDD_BASE_URL}{remote_name}", local_path))

    if len(candidates) == 0:
        return list_audio_files(INPUT_FOLDER)

    np.random.shuffle(candidates)
    max_attempts = min(
        len(candidates),
        max(needed, 1) * FSDD_DOWNLOAD_ATTEMPT_MULTIPLIER,
    )
    downloaded = 0
    attempts = 0

    for url, local_path in candidates:
        if downloaded >= needed or attempts >= max_attempts:
            break
        attempts += 1
        try:
            urllib.request.urlretrieve(url, local_path)
            downloaded += 1
            if downloaded % 5 == 0 or downloaded == needed:
                print(f"Downloaded {downloaded}/{needed} clean clips...")
        except Exception:
            continue

    if downloaded == 0:
        print("Could not download additional clean clips; using currently available inputs.")
    else:
        print(f"Downloaded {downloaded} extra clean clip(s).")

    return list_audio_files(INPUT_FOLDER)


# =====================
# BATCH PROCESS
# =====================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(NOISE_FOLDER, exist_ok=True)
    input_files = ensure_input_audio_exists()
    input_files = bootstrap_clean_audio_batch(input_files)
    input_files, skipped_inputs = select_dispatch_input_files(input_files)
    if skipped_inputs:
        print(f"Skipping {len(skipped_inputs)} music-like clean input file(s).")
        for p in skipped_inputs[:5]:
            print("  skipped:", os.path.basename(p))

    noise_catalog, rejected_noise = build_noise_catalog(NOISE_FOLDER)
    if ENABLE_EXTERNAL_NOISE_MIX:
        if noise_catalog:
            print(f"Using {len(noise_catalog)} external noise file(s) from '{NOISE_FOLDER}'.")
        else:
            print(
                f"No valid external noise files found in '{NOISE_FOLDER}'. "
                "Using synthetic radio static/interference only."
            )
    else:
        noise_catalog = []
        print("External noise mixing disabled (strict dispatch mode).")

    if rejected_noise:
        print(f"Ignored {len(rejected_noise)} unsuitable noise file(s).")
        for path, reason in rejected_noise[:5]:
            print(f"  ignored: {os.path.basename(path)} ({reason})")

    if not input_files:
        raise RuntimeError(
            "No valid clean input files to process after filtering. "
            "Add speech clips to clean_audio and run again."
        )

    created = 0
    for input_path in input_files:
        filename = os.path.basename(input_path)
        stem, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"{stem}_radio.wav")

        radio_audio, sr = process_file(input_path, noise_catalog)
        write_float_as_wav(output_path, radio_audio, sr)
        created += 1
        print("Created:", output_path)

    print(f"Done. Generated {created} file(s) in '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()
