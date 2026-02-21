import streamlit as st
import numpy as np
import soundfile as sf
from pathlib import Path
from pipeline.enhance import enhance_audio
from pipeline.asr import transcribe

st.set_page_config(page_title="CLEARCOMMS", layout="wide")
st.title("CLEARCOMMS — Offline Radio Speech Recovery")

uploaded = st.file_uploader("Upload WAV/FLAC/MP3 (WAV safest)", type=["wav", "flac", "mp3"])

if uploaded:
    Path("runs").mkdir(exist_ok=True)
    in_path = Path("runs/input.wav")
    out_path = Path("runs/enhanced.wav")

    # Streamlit gives bytes; write to disk
    in_path.write_bytes(uploaded.read())

    # Read audio
    audio, sr = sf.read(str(in_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    st.subheader("Input Audio")
    st.audio(in_path.read_bytes())

    st.subheader("Enhancing…")
    enhanced = enhance_audio(audio, sr)
    sf.write(str(out_path), enhanced, sr)

    st.subheader("Enhanced Audio")
    st.audio(out_path.read_bytes())

    st.subheader("Transcribing…")
    raw_text, meta = transcribe(out_path, sr)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Transcript")
        st.write(raw_text)
    with col2:
        st.markdown("### Timing / Performance")
        st.json(meta)

    st.download_button("Download incident JSON", data=meta.get("incident_json","{}"), file_name="incident.json")
