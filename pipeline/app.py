"""pipeline/app.py

ClearComms Streamlit UI (offline-first).

This UI is intentionally simple and demo-focused:
  - Upload an audio clip
  - (Optional) apply a light "radio" preprocess (bandpass + gate)
  - Run Whisper (QNN EP if ONNX encoder/decoder are present)
  - Show transcript + latency + backend proof

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

from pipeline.asr import transcribe
from pipeline.enhance import enhance_audio
from pipeline.audio_io import load_mono, normalize_peak, resample, save_wav, WHISPER_SR


_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "model" / "config.yaml"


def _model_files_present() -> bool:
    enc = _ROOT / "models" / "WhisperEncoder.onnx"
    dec = _ROOT / "models" / "WhisperDecoder.onnx"
    return enc.exists() and dec.exists()


def run_streamlit_app() -> None:
    st.set_page_config(page_title="ClearComms", layout="wide")
    st.title("ClearComms — Offline Radio Transcription")

    with st.sidebar:
        st.header("Controls")
        apply_radio_filter = st.checkbox(
            "Apply radio preprocess (bandpass + light gate)",
            value=True,
            help="Not a denoiser model — just a fast DSP filter that often helps radio audio.",
        )
        normalize = st.checkbox("Normalize loudness (peak)", value=True)
        st.caption("Whisper expects 16 kHz mono; we convert automatically.")

        st.divider()
        st.subheader("Model backend")
        if _model_files_present():
            st.success("Found ONNX encoder/decoder → QNN/NPU backend should load")
        else:
            st.warning(
                "ONNX encoder/decoder not found in ./models. "
                "ASR may fall back to HuggingFace ONNX export (requires internet the first time)."
            )
        st.caption("Config: model/config.yaml")

    st.subheader("Input source")
    source_mode = st.radio(
        "Choose input type",
        ["Upload file", "Record microphone", "Repo demo clip"],
        horizontal=True,
    )

    uploaded = None
    recorded = None
    input_path: Path | None = None
    input_bytes: bytes | None = None
    input_label = ""
    cleanup_input = False

    if source_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload an audio clip (WAV recommended).",
            type=["wav", "flac", "ogg", "mp3", "m4a"],
        )
        if not uploaded:
            st.info("Upload a clip to run: source -> preprocess -> Whisper -> transcript.")
            return

        suffix = Path(uploaded.name).suffix or ".wav"
        input_bytes = uploaded.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(input_bytes)
            input_path = Path(tf.name)
        input_label = uploaded.name
        cleanup_input = True

    elif source_mode == "Record microphone":
        recorded = st.audio_input("Record audio from your microphone")
        if recorded:
            st.caption("Mic preview")
            st.audio(recorded)
        if not recorded:
            st.info("Record audio to run the full pipeline.")
            return

        input_bytes = recorded.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tf.write(input_bytes)
            input_path = Path(tf.name)
        input_label = "mic_recording.wav"
        cleanup_input = True

    else:
        demo_dir = _ROOT / "radio_dispatch_filter" / "radio_audio"
        demo_files = sorted(demo_dir.glob("*.wav")) if demo_dir.exists() else []
        if not demo_files:
            st.warning(f"No demo WAV files found in {demo_dir}")
            return

        picked = st.selectbox("Pick a demo radio clip", [p.name for p in demo_files], index=0)
        input_path = demo_dir / picked
        input_bytes = input_path.read_bytes()
        input_label = picked
        cleanup_input = False

    if input_path is None:
        st.error("No input selected.")
        return

    tmp_dir = Path("runs")
    tmp_dir.mkdir(exist_ok=True)
    pre16_path = tmp_dir / "preprocessed_16k.wav"
    filt_path = tmp_dir / "radio_filtered_16k.wav"

    audio, sr = load_mono(str(input_path))
    audio_16k, _ = resample(audio, sr, WHISPER_SR)
    if normalize:
        audio_16k = normalize_peak(audio_16k)
    save_wav(str(pre16_path), audio_16k, WHISPER_SR)

    if apply_radio_filter:
        filtered = enhance_audio(audio_16k, WHISPER_SR)
        save_wav(str(filt_path), filtered, WHISPER_SR)
        asr_input = filt_path
    else:
        asr_input = pre16_path

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original input")
        st.audio(input_bytes if input_bytes is not None else input_path.read_bytes())
        st.caption(f"Loaded: {sr} Hz, {len(audio)/max(sr,1):.2f}s")
        st.caption(f"Source: {source_mode} | {input_label}")

        st.subheader("Prepared (16 kHz mono)")
        st.audio(pre16_path.read_bytes())
        if apply_radio_filter:
            st.subheader("After radio preprocess")
            st.audio(filt_path.read_bytes())

    with col2:
        st.subheader("Transcript")
        try:
            t0 = time.time()
            text, meta = transcribe(str(asr_input), WHISPER_SR)
            total_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            st.error(
                "Transcription failed. If you're offline, make sure the ONNX encoder/decoder exist in ./models. "
                "(models/WhisperEncoder.onnx and models/WhisperDecoder.onnx)"
            )
            st.code(str(e))
            return

        st.write(text if text else "(no transcript)")

        st.subheader("Performance")
        st.json({**meta, "ui_total_ms": round(total_ms, 1)})

        st.subheader("Export")
        st.download_button(
            "Download transcript.txt",
            data=(text + "\n"),
            file_name="transcript.txt",
        )
        st.download_button(
            "Download metadata.json",
            data=json.dumps({"meta": meta}, indent=2),
            file_name="metadata.json",
        )

        with st.expander("Debug"):
            st.write("ASR input file:", str(asr_input))
            st.write("Model config path:", str(_CFG))
            st.write("Tip: add ./models/*.onnx to .gitignore; don’t commit weights.")

    try:
        if cleanup_input and input_path and input_path.exists():
            os.remove(str(input_path))
    except Exception:
        pass
