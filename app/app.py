import os
import tempfile
import time

import numpy as np
import streamlit as st
from scipy.signal import butter, lfilter

from pipeline.audio_io import load_audio, resample_to_16k, normalize
from pipeline.asr_qnn import QNNWhisper

st.set_page_config(page_title="CLEARCOMMS", layout="wide")
st.title("CLEARCOMMS — Offline Radio Transcription (Snapdragon NPU)")

@st.cache_resource
def get_asr():
    return QNNWhisper("config.yaml")

def bandpass_radio(x: np.ndarray, sr: int, lo=300, hi=3400, order=4) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return lfilter(b, a, x).astype(np.float32)

uploaded = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg"])

colL, colR = st.columns([1, 1])

with colL:
    use_radio_bp = st.checkbox("Apply radio bandpass (300–3400 Hz)", value=True)
    do_normalize = st.checkbox("Normalize audio", value=True)

if uploaded:
    # Save to temp so soundfile can load it reliably
    suffix = os.path.splitext(uploaded.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded.read())
        tmp_path = tf.name

    # Load + convert
    x, sr = load_audio(tmp_path)
    x, sr = resample_to_16k(x, sr, 16000)
    if use_radio_bp:
        x = bandpass_radio(x, sr)
    if do_normalize:
        x = normalize(x)

    with colL:
        st.subheader("Audio (uploaded)")
        st.audio(open(tmp_path, "rb").read())

        st.subheader("Audio (processed 16k mono)")
        # write processed to wav for playback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf2:
            import soundfile as sf
            sf.write(tf2.name, x, sr)
            st.audio(open(tf2.name, "rb").read())

    # Transcribe
    asr = get_asr()
    with colR:
        st.subheader("Transcription")
        t0 = time.time()
        text, meta = asr.transcribe(x)
        total_ms = (time.time() - t0) * 1000.0

        st.write(text if text else "*No transcript produced.*")

        st.subheader("Performance / Proof")
        st.json({
            "provider_hint": "QNNExecutionProvider should be active (check your console logs / ORT providers).",
            "asr_meta": meta,
            "ui_total_ms": round(total_ms, 1),
        })

    # cleanup temp file
    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Upload a short WAV clip. The app converts to 16k mono → (optional) radio bandpass → Whisper (QNN/NPU) → transcript.")
