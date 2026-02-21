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
from typing import Callable, Optional, Tuple

import streamlit as st

from pipeline.enhance import enhance_audio
from pipeline.audio_io import load_mono, normalize_peak, resample, save_wav, WHISPER_SR


_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "model" / "config.yaml"
_TRANSCRIBE_FN: Optional[Callable[[str, int], Tuple[str, dict]]] = None
_TRANSCRIBE_IMPORT_ERROR: Optional[Exception] = None


def _model_files_present() -> bool:
    enc = _ROOT / "models" / "WhisperEncoder.onnx"
    dec = _ROOT / "models" / "WhisperDecoder.onnx"
    return enc.exists() and dec.exists()


def _get_transcribe_fn() -> Tuple[Optional[Callable[[str, int], Tuple[str, dict]]], Optional[Exception]]:
    global _TRANSCRIBE_FN, _TRANSCRIBE_IMPORT_ERROR

    if _TRANSCRIBE_FN is not None or _TRANSCRIBE_IMPORT_ERROR is not None:
        return _TRANSCRIBE_FN, _TRANSCRIBE_IMPORT_ERROR

    try:
        from pipeline.asr import transcribe as _transcribe
    except Exception as exc:
        _TRANSCRIBE_IMPORT_ERROR = exc
        return None, _TRANSCRIBE_IMPORT_ERROR

    _TRANSCRIBE_FN = _transcribe
    return _TRANSCRIBE_FN, None


def _demo_transcribe(duration_sec: float, clip_name: str) -> Tuple[str, dict]:
    templates = [
        "Engine 12 to dispatch, smoke visible near Maple Street. Requesting backup.",
        "Unit 4 arrived on scene. One patient stable, preparing transport.",
        "Dispatch, possible traffic collision at Pine and 3rd. Need medical support.",
        "Team reporting heavy interference on channel. Repeating location now.",
        "Responder to base, hazard contained. Continuing perimeter check.",
    ]
    idx = (len(clip_name) + int(duration_sec * 10)) % len(templates)
    text = templates[idx]

    fake_latency_ms = max(90.0, min(1200.0, 110.0 + duration_sec * 45.0))
    meta = {
        "asr_latency_ms": round(fake_latency_ms, 1),
        "audio_duration_sec": round(duration_sec, 2),
        "realtime_factor": round(fake_latency_ms / 1000.0 / max(duration_sec, 0.001), 3),
        "backend": "demo",
        "model_variant": "ui_mock",
        "note": "Demo mode: synthetic transcript for UI testing.",
    }
    return text, meta


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
        model_files_present = _model_files_present()
        if model_files_present:
            st.success("Found ONNX encoder/decoder → QNN/NPU backend should load")
        else:
            st.warning(
                "ONNX encoder/decoder not found in ./models. "
                "ASR may fall back to HuggingFace ONNX export (requires internet the first time)."
            )
        st.caption("Config: model/config.yaml")

        st.divider()
        st.subheader("Runtime mode")
        demo_mode = st.checkbox(
            "UI demo mode (no Whisper required)",
            value=not model_files_present,
            help="Use synthetic transcript + timings so you can test the UI without models or NPU.",
        )
        fallback_to_demo = st.checkbox(
            "Fallback to demo transcript if ASR fails",
            value=True,
            help="Keeps the UI usable even when model load/transcription fails.",
        )
        if demo_mode:
            st.info("Demo mode is active. Transcript output is synthetic.")

    uploaded = st.file_uploader(
        "Upload an audio clip (WAV recommended).",
        type=["wav", "flac", "ogg", "mp3", "m4a"],
    )

    if not uploaded:
        st.info(
            "Upload a clip to run: upload → (optional) radio preprocess → Whisper → transcript. "
            "Everything runs locally on the laptop."
        )
        return

    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded.read())
        input_path = Path(tf.name)

    try:
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
            st.subheader("Original upload")
            st.audio(input_path.read_bytes())
            duration_sec = len(audio) / max(sr, 1)
            st.caption(f"Loaded: {sr} Hz, {duration_sec:.2f}s")

            st.subheader("Prepared (16 kHz mono)")
            st.audio(pre16_path.read_bytes())
            if apply_radio_filter:
                st.subheader("After radio preprocess")
                st.audio(filt_path.read_bytes())

        with col2:
            st.subheader("Transcript")
            t0 = time.time()

            mode_used = "demo" if demo_mode else "asr"
            asr_error: Optional[Exception] = None

            if demo_mode:
                text, meta = _demo_transcribe(duration_sec, uploaded.name)
            else:
                transcribe_fn, import_error = _get_transcribe_fn()
                if transcribe_fn is None:
                    asr_error = import_error
                    if fallback_to_demo:
                        st.warning("ASR backend failed to load. Falling back to demo transcript.")
                        mode_used = "demo_fallback"
                        text, meta = _demo_transcribe(duration_sec, uploaded.name)
                    else:
                        st.error("ASR backend failed to load and demo fallback is disabled.")
                        st.code(str(import_error))
                        return
                else:
                    try:
                        text, meta = transcribe_fn(str(asr_input), WHISPER_SR)
                    except Exception as exc:
                        asr_error = exc
                        if fallback_to_demo:
                            st.warning("Transcription failed. Falling back to demo transcript.")
                            mode_used = "demo_fallback"
                            text, meta = _demo_transcribe(duration_sec, uploaded.name)
                        else:
                            st.error(
                                "Transcription failed. If you're offline, make sure ONNX files exist in ./models."
                            )
                            st.code(str(exc))
                            return

            total_ms = (time.time() - t0) * 1000.0
            st.write(text if text else "(no transcript)")

            st.subheader("Performance")
            st.json({**meta, "ui_total_ms": round(total_ms, 1), "ui_mode": mode_used})

            st.subheader("Export")
            st.download_button(
                "Download transcript.txt",
                data=(text + "\n"),
                file_name="transcript.txt",
            )
            st.download_button(
                "Download metadata.json",
                data=json.dumps({"meta": meta, "ui_mode": mode_used}, indent=2),
                file_name="metadata.json",
            )

            with st.expander("Debug"):
                st.write("ASR input file:", str(asr_input))
                st.write("Model config path:", str(_CFG))
                st.write("Tip: add ./models/*.onnx to .gitignore; don’t commit weights.")
                if asr_error is not None:
                    st.write("ASR error:", str(asr_error))
    finally:
        try:
            os.remove(str(input_path))
        except Exception:
            pass
