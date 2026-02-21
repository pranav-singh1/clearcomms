"""pipeline/app.py

ClearComms Streamlit UI (offline-first).

Pipeline stages:
  1. Upload audio clip
  2. Preprocess (resample, bandpass, normalize)
  3. Whisper ASR (QNN/NPU or ONNX CPU fallback)
  4. Transcript cleanup (LLaMA or rule-based)
  5. Structured incident extraction (LLaMA or rule-based)
  6. Display + export

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
import yaml

from pipeline.enhance import enhance_audio
from pipeline.audio_io import load_mono, normalize_peak, resample, save_wav, WHISPER_SR
from pipeline.cleanup import cleanup_transcript
from pipeline.extract import extract_incident


_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "model" / "config.yaml"
_TRANSCRIBE_FN: Optional[Callable[[str, int], Tuple[str, dict]]] = None
_TRANSCRIBE_IMPORT_ERROR: Optional[Exception] = None


def _resolve_model_paths() -> Tuple[Path, Path]:
    defaults = ("models/WhisperEncoder.onnx", "models/WhisperDecoder.onnx")
    if _CFG.exists():
        try:
            with open(_CFG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            enc_cfg = Path(cfg.get("encoder_path", defaults[0]))
            dec_cfg = Path(cfg.get("decoder_path", defaults[1]))
        except Exception:
            enc_cfg, dec_cfg = Path(defaults[0]), Path(defaults[1])
    else:
        enc_cfg, dec_cfg = Path(defaults[0]), Path(defaults[1])

    enc = enc_cfg if enc_cfg.is_absolute() else (_ROOT / enc_cfg)
    dec = dec_cfg if dec_cfg.is_absolute() else (_ROOT / dec_cfg)
    return enc, dec


def _model_files_present() -> bool:
    enc, dec = _resolve_model_paths()
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
        "engine 12 to dispatch smoke visible near mapple street requesting back up",
        "unit 4 arrived on scene one patient stable preparing transport to med center",
        "dispatch poss traffic collision at pine and 3rd need med support",
        "team reporting heavy interference on channel repeating location now",
        "responder to base hazard contained continuing perimeter check eng 7 standing by",
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
        enc_model_path, dec_model_path = _resolve_model_paths()
        model_files_present = enc_model_path.exists() and dec_model_path.exists()
        if model_files_present:
            st.success("Found ONNX encoder/decoder in config paths.")
        else:
            st.warning(
                "Configured ONNX encoder/decoder not found. "
                "ASR may fall back to HuggingFace ONNX export (internet needed once)."
            )
        st.caption(f"Encoder: {enc_model_path}")
        st.caption(f"Decoder: {dec_model_path}")
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
            "Upload a clip to run the full pipeline: "
            "preprocess → Whisper → cleanup → incident extraction. "
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
            pipeline_t0 = time.time()

            # --- Layer 3: ASR ---
            st.subheader("Raw Transcript")

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

            st.write(text if text else "(no transcript)")

            # --- Layer 4: Transcript Cleanup ---
            st.subheader("Cleaned Transcript")
            cleaned_text, cleanup_meta = cleanup_transcript(text)
            st.write(cleaned_text if cleaned_text else "(no output)")

            if text != cleaned_text:
                with st.expander("Side-by-side comparison"):
                    raw_col, clean_col = st.columns(2)
                    with raw_col:
                        st.caption("Raw")
                        st.text(text)
                    with clean_col:
                        st.caption("Cleaned")
                        st.text(cleaned_text)

            # --- Layer 5: Structured Incident Extraction ---
            st.subheader("Incident Report")
            incident, extract_meta = extract_incident(cleaned_text)

            urgency_colors = {
                "critical": "red", "high": "orange",
                "medium": "blue", "low": "green",
            }
            urg = incident.get("urgency", "medium")
            st.markdown(f"**Urgency:** :{urgency_colors.get(urg, 'blue')}[{urg.upper()}]")
            st.markdown(f"**Type:** {incident.get('request_type', 'unknown')}")
            st.markdown(f"**Location:** {incident.get('location', 'unknown')}")

            if incident.get("units"):
                st.markdown(f"**Units:** {', '.join(incident['units'])}")
            if incident.get("hazards"):
                st.markdown(f"**Hazards:** {', '.join(incident['hazards'])}")
            if incident.get("actions"):
                st.markdown(f"**Actions:** {', '.join(incident['actions'])}")

            total_pipeline_ms = (time.time() - pipeline_t0) * 1000.0

            # --- Layer 7: Performance ---
            st.subheader("Performance")
            all_meta = {
                **meta,
                **cleanup_meta,
                **extract_meta,
                "total_pipeline_ms": round(total_pipeline_ms, 1),
                "ui_mode": mode_used,
            }
            st.json(all_meta)
            if all_meta.get("qnn_status") == "failed":
                st.warning("QNN backend failed; currently running CPU fallback.")
                if all_meta.get("qnn_error"):
                    st.code(str(all_meta["qnn_error"]))

            # --- Export ---
            st.subheader("Export")
            incident_json_str = json.dumps(incident, indent=2)

            st.download_button(
                "Download incident.json",
                data=incident_json_str,
                file_name="incident.json",
            )
            st.download_button(
                "Download transcript.txt",
                data=(cleaned_text + "\n"),
                file_name="transcript.txt",
            )
            st.download_button(
                "Download full_report.json",
                data=json.dumps(
                    {"incident": incident, "raw_transcript": text,
                     "cleaned_transcript": cleaned_text, "meta": all_meta},
                    indent=2,
                ),
                file_name="full_report.json",
            )

            with st.expander("Debug"):
                st.write("ASR input file:", str(asr_input))
                st.write("Model config path:", str(_CFG))
                st.write("Cleanup method:", cleanup_meta.get("cleanup_method"))
                st.write("Extract method:", extract_meta.get("extract_method"))
                if asr_error is not None:
                    st.write("ASR error:", str(asr_error))
    finally:
        try:
            os.remove(str(input_path))
        except Exception:
            pass
