"""pipeline/app.py

ClearComms Streamlit UI (offline-first) with an operator-first review flow.
"""

from __future__ import annotations

import base64
import html
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import streamlit as st
import yaml

from pipeline.audio_io import WHISPER_SR, load_mono, normalize_peak, resample, save_wav
from pipeline.cleanup import cleanup_transcript
from pipeline.enhance import enhance_audio
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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 6% 0%, #ebf4ff 0%, #f7fafe 35%, #fffaf3 75%, #fffefb 100%);
            font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
            color: #10243f;
        }
        .cc-hero {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(245, 250, 255, 0.92));
            border: 1px solid #d8e3f0;
            border-radius: 14px;
            padding: 1.0rem 1.1rem;
            margin-bottom: 1.0rem;
            box-shadow: 0 8px 24px rgba(16, 36, 63, 0.08);
        }
        .cc-hero-title {
            margin: 0;
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .cc-hero-sub {
            margin: 0.35rem 0 0 0;
            color: #4b5f79;
            font-size: 0.92rem;
            line-height: 1.35rem;
        }
        .cc-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: 0.6rem;
            margin: 0.3rem 0 0.7rem 0;
        }
        .cc-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #d8e3f0;
            border-radius: 12px;
            padding: 0.7rem 0.75rem;
        }
        .cc-label {
            margin: 0;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #4e6078;
            font-weight: 600;
        }
        .cc-value {
            margin: 0.18rem 0 0 0;
            font-size: 1.02rem;
            color: #10243f;
            font-weight: 700;
            line-height: 1.2rem;
        }
        .cc-urgency-critical { border-left: 5px solid #b42318; }
        .cc-urgency-high { border-left: 5px solid #ea580c; }
        .cc-urgency-medium { border-left: 5px solid #1d4ed8; }
        .cc-urgency-low { border-left: 5px solid #15803d; }
        /* Force readable button colors across Streamlit themes and versions. */
        .stButton > button,
        div[data-testid="stButton"] > button,
        button[kind="primary"],
        button[kind="secondary"],
        .stDownloadButton > button,
        div[data-testid="stDownloadButton"] > button,
        .stBaseButton-primary,
        .stBaseButton-secondary {
            background: linear-gradient(135deg, #1f4d8f 0%, #153f77 100%) !important;
            background-color: #1f4d8f !important;
            color: #ffffff !important;
            border: 1px solid #133965 !important;
            font-weight: 700 !important;
            text-shadow: none !important;
        }
        .stButton > button:hover,
        div[data-testid="stButton"] > button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover,
        .stBaseButton-primary:hover,
        .stBaseButton-secondary:hover {
            background: linear-gradient(135deg, #255aa6 0%, #1d4e8f 100%) !important;
            background-color: #255aa6 !important;
            color: #ffffff !important;
            border-color: #0f2f53 !important;
        }
        .stButton > button:disabled,
        div[data-testid="stButton"] > button:disabled,
        button[kind="primary"]:disabled,
        button[kind="secondary"]:disabled,
        .stDownloadButton > button:disabled,
        div[data-testid="stDownloadButton"] > button:disabled,
        .stBaseButton-primary:disabled,
        .stBaseButton-secondary:disabled {
            background: #7f8ea3 !important;
            background-color: #7f8ea3 !important;
            color: #ffffff !important;
            border-color: #708197 !important;
            opacity: 0.95 !important;
        }
        .stButton > button * ,
        .stDownloadButton > button * {
            color: #ffffff !important;
        }
        /* Fix low-contrast status + alert colors across theme variants. */
        div[data-testid="stStatusWidget"],
        div[data-testid="stStatus"],
        details[data-testid="stExpander"].st-status,
        [data-testid="stStatusWidget"] summary,
        [data-testid="stStatusWidget"] details,
        .stStatus,
        .stStatus summary,
        .stStatus details {
            background: #eaf3ff !important;
            background-color: #eaf3ff !important;
            border: 1px solid #b8d0ea !important;
            border-radius: 12px !important;
        }
        div[data-testid="stStatusWidget"] *,
        div[data-testid="stStatus"] *,
        .stStatus *,
        .stStatus summary *,
        .stStatus details *,
        [data-testid="stStatusWidget"] summary *,
        [data-testid="stStatusWidget"] label,
        [data-testid="stStatusWidget"] span,
        [data-testid="stStatusWidget"] p,
        [data-testid="stStatusWidget"] div {
            color: #123e6f !important;
            fill: #123e6f !important;
            -webkit-text-fill-color: #123e6f !important;
        }
        /* Completed/running/error status icon colors */
        [data-testid="stStatusWidget"] svg,
        .stStatus svg {
            color: #1f4d8f !important;
            fill: #1f4d8f !important;
        }
        /* Nuclear override for the collapsed status bar — covers all Streamlit versions */
        [data-testid="stStatusWidget"] > summary,
        [data-testid="stStatusWidget"] > details > summary,
        details[open] > summary,
        details:not([open]) > summary {
            background: #eaf3ff !important;
            background-color: #eaf3ff !important;
            color: #123e6f !important;
            -webkit-text-fill-color: #123e6f !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid #c9d8ea !important;
        }
        div[data-testid="stAlert"] *,
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {
            color: #10243f !important;
            fill: #10243f !important;
            -webkit-text-fill-color: #10243f !important;
        }
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] * {
            color: #10243f !important;
        }
        .cc-audio-wrap {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #d8e3f0;
            border-radius: 12px;
            padding: 0.4rem 0.55rem;
            margin-bottom: 0.5rem;
        }
        .cc-audio-wrap audio {
            width: 100%;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe(value: Any) -> str:
    return html.escape(str(value) if value is not None else "unknown")


def _guess_audio_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".ogg":
        return "audio/ogg"
    return "audio/wav"


def _render_audio_player(audio_bytes: bytes, mime_type: str) -> None:
    if not audio_bytes:
        st.info("Audio unavailable.")
        return

    payload = base64.b64encode(audio_bytes).decode("ascii")
    st.markdown(
        f"""
        <div class="cc-audio-wrap">
          <audio controls preload="metadata">
            <source src="data:{mime_type};base64,{payload}" type="{mime_type}">
            Your browser does not support audio playback.
          </audio>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _urgency_card_class(urgency: str) -> str:
    normalized = urgency.strip().lower()
    if normalized in {"critical", "high", "medium", "low"}:
        return f"cc-urgency-{normalized}"
    return "cc-urgency-medium"


def _build_review_flags(incident: dict, cleaned_text: str) -> list[str]:
    flags: list[str] = []

    location = str(incident.get("location", "")).strip().lower()
    if not location or location == "unknown":
        flags.append("Location is missing or unknown")

    if not incident.get("units"):
        flags.append("No responding units detected")

    if not incident.get("actions"):
        flags.append("No requested action detected")

    if len(cleaned_text.split()) < 4:
        flags.append("Transcript is very short and may be incomplete")

    return flags


def _run_pipeline(
    uploaded_name: str,
    uploaded_bytes: bytes,
    apply_radio_filter: bool,
    normalize: bool,
    demo_mode: bool,
    fallback_to_demo: bool,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    def tick(label: str) -> None:
        if stage_cb:
            stage_cb(label)

    suffix = Path(uploaded_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded_bytes)
        input_path = Path(tf.name)

    pre16_path: Optional[Path] = None
    filt_path: Optional[Path] = None
    asr_error: Optional[Exception] = None

    try:
        tick("Preparing audio")
        tmp_dir = Path("runs")
        tmp_dir.mkdir(exist_ok=True)

        run_id = str(int(time.time() * 1000))
        base_name = Path(uploaded_name).stem.replace(" ", "_")[:42] or "clip"
        pre16_path = tmp_dir / f"{base_name}_{run_id}_preprocessed_16k.wav"
        filt_path = tmp_dir / f"{base_name}_{run_id}_radio_filtered_16k.wav"

        audio, sr = load_mono(str(input_path))
        duration_sec = len(audio) / max(sr, 1)

        audio_16k, _ = resample(audio, sr, WHISPER_SR)
        if normalize:
            audio_16k = normalize_peak(audio_16k)
        save_wav(str(pre16_path), audio_16k, WHISPER_SR)

        if apply_radio_filter:
            filtered = enhance_audio(audio_16k, WHISPER_SR)
            save_wav(str(filt_path), filtered, WHISPER_SR)
            asr_input = filt_path
            asr_input_label = "radio_filtered_16k.wav"
        else:
            asr_input = pre16_path
            asr_input_label = "preprocessed_16k.wav"

        tick("Running transcription")
        mode_used = "demo" if demo_mode else "asr"
        if demo_mode:
            text, meta = _demo_transcribe(duration_sec, uploaded_name)
        else:
            transcribe_fn, import_error = _get_transcribe_fn()
            if transcribe_fn is None:
                asr_error = import_error
                if not fallback_to_demo:
                    raise RuntimeError(str(import_error))
                mode_used = "demo_fallback"
                text, meta = _demo_transcribe(duration_sec, uploaded_name)
            else:
                try:
                    text, meta = transcribe_fn(str(asr_input), WHISPER_SR)
                except Exception as exc:
                    asr_error = exc
                    if not fallback_to_demo:
                        raise
                    mode_used = "demo_fallback"
                    text, meta = _demo_transcribe(duration_sec, uploaded_name)

        tick("Cleaning transcript")
        cleaned_text, cleanup_meta = cleanup_transcript(text)

        tick("Extracting incident fields")
        incident, extract_meta = extract_incident(cleaned_text)

        total_pipeline_ms = (time.time() - (int(run_id) / 1000.0)) * 1000.0
        all_meta = {
            **meta,
            **cleanup_meta,
            **extract_meta,
            "total_pipeline_ms": round(total_pipeline_ms, 1),
            "ui_mode": mode_used,
        }

        result = {
            "run_id": run_id,
            "uploaded_name": uploaded_name,
            "duration_sec": round(duration_sec, 2),
            "source_sr": sr,
            "raw_transcript": text,
            "cleaned_transcript": cleaned_text,
            "incident": incident,
            "meta": all_meta,
            "cleanup_meta": cleanup_meta,
            "extract_meta": extract_meta,
            "asr_error": str(asr_error) if asr_error is not None else "",
            "asr_input_label": asr_input_label,
            "input_audio_mime": _guess_audio_mime(uploaded_name),
            "input_audio_bytes": input_path.read_bytes(),
            "prepared_audio_mime": "audio/wav",
            "prepared_audio_bytes": pre16_path.read_bytes() if pre16_path.exists() else b"",
            "filtered_audio_mime": "audio/wav",
            "filtered_audio_bytes": (
                filt_path.read_bytes() if (apply_radio_filter and filt_path.exists()) else b""
            ),
        }
        return result
    finally:
        for p in (input_path, pre16_path, filt_path):
            if p is None:
                continue
            try:
                os.remove(str(p))
            except Exception:
                pass


def run_streamlit_app() -> None:
    st.set_page_config(page_title="ClearComms", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="cc-hero">
          <p class="cc-hero-title">ClearComms - Offline Radio Triage Console</p>
          <p class="cc-hero-sub">
            Run a clip through preprocess, ASR, cleanup, and incident extraction.
            The layout prioritizes fast operator review before export.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline Controls")
        apply_radio_filter = st.checkbox(
            "Apply radio preprocess (bandpass + light gate)",
            value=True,
            help="Fast DSP filter for radio bandwidth; not the neural denoiser model.",
        )
        normalize = st.checkbox("Normalize loudness (peak)", value=True)
        st.caption("Whisper expects 16 kHz mono. Conversion is automatic.")

        st.divider()
        st.subheader("ASR Backend")
        enc_model_path, dec_model_path = _resolve_model_paths()
        model_files_present = _model_files_present()
        if model_files_present:
            st.success("ONNX encoder/decoder found in config paths.")
        else:
            st.warning(
                "Configured ONNX encoder/decoder missing. "
                "ASR may fall back to ONNX export (internet needed once)."
            )
        st.caption(f"Encoder: {enc_model_path}")
        st.caption(f"Decoder: {dec_model_path}")
        st.caption("Config: model/config.yaml")

        st.divider()
        st.subheader("Runtime Mode")
        demo_mode = st.checkbox(
            "UI demo mode (no Whisper required)",
            value=not model_files_present,
            help="Use synthetic transcript/timing to validate UX without model runtime.",
        )
        fallback_to_demo = st.checkbox(
            "Fallback to demo transcript if ASR fails",
            value=True,
            help="Keeps review workflow available if model loading/transcription fails.",
        )
        if demo_mode:
            st.info("Demo mode is active. Output transcript is synthetic.")

    uploaded = st.file_uploader(
        "Upload an audio clip (WAV recommended)",
        type=["wav", "flac", "ogg", "mp3", "m4a"],
    )

    run_col, clear_col = st.columns([2, 1])
    run_clicked = run_col.button(
        "Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )
    clear_clicked = clear_col.button("Clear Results", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("cc_last_result", None)
        st.rerun()

    if run_clicked:
        if uploaded is None:
            st.warning("Upload a clip first.")
        else:
            with st.status("Pipeline running...", expanded=True) as status:
                try:
                    result = _run_pipeline(
                        uploaded_name=uploaded.name,
                        uploaded_bytes=uploaded.getvalue(),
                        apply_radio_filter=apply_radio_filter,
                        normalize=normalize,
                        demo_mode=demo_mode,
                        fallback_to_demo=fallback_to_demo,
                        stage_cb=lambda s: status.update(label=s, state="running"),
                    )
                except Exception as exc:
                    status.update(label="Pipeline failed", state="error")
                    st.error(
                        "Processing failed. If running offline, ensure ONNX models exist in ./models "
                        "or enable demo mode."
                    )
                    st.code(str(exc))
                    return
                status.update(label="Pipeline complete", state="complete")
            st.session_state["cc_last_result"] = result

    result = st.session_state.get("cc_last_result")
    if not result:
        st.info(
            "Upload a clip and click Run Pipeline to start. "
            "Result view keeps the last successful run until cleared."
        )
        return

    incident = result["incident"]
    all_meta = result["meta"]
    raw_text = result["raw_transcript"] or "(no transcript)"
    cleaned_text = result["cleaned_transcript"] or "(no cleaned transcript)"
    urgency = str(incident.get("urgency", "medium")).strip().lower()

    if all_meta.get("ui_mode") in {"demo", "demo_fallback"}:
        st.warning(
            "Demo transcript mode is active. Use only for UX flow checks, not operational decisions."
        )
    if all_meta.get("qnn_status") == "failed":
        st.warning("QNN backend failed; currently using CPU fallback.")

    summary_html = f"""
    <div class="cc-summary-grid">
      <div class="cc-card {_urgency_card_class(urgency)}">
        <p class="cc-label">Urgency</p>
        <p class="cc-value">{_safe(urgency.upper() if urgency else "MEDIUM")}</p>
      </div>
      <div class="cc-card">
        <p class="cc-label">Request Type</p>
        <p class="cc-value">{_safe(incident.get("request_type", "unknown"))}</p>
      </div>
      <div class="cc-card">
        <p class="cc-label">Location</p>
        <p class="cc-value">{_safe(incident.get("location", "unknown"))}</p>
      </div>
      <div class="cc-card">
        <p class="cc-label">Runtime</p>
        <p class="cc-value">{_safe(all_meta.get("backend", "unknown"))} | {_safe(all_meta.get("ui_mode", "asr"))}</p>
      </div>
      <div class="cc-card">
        <p class="cc-label">Latency</p>
        <p class="cc-value">{_safe(all_meta.get("total_pipeline_ms", "n/a"))} ms</p>
      </div>
      <div class="cc-card">
        <p class="cc-label">Clip</p>
        <p class="cc-value">{_safe(result.get("uploaded_name", "unknown"))}</p>
      </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    review_flags = _build_review_flags(incident, cleaned_text)
    if review_flags:
        st.error("Review focus: " + " | ".join(review_flags))
    else:
        st.success("No immediate review flags detected from rule checks.")

    review_tab, audio_tab, data_tab = st.tabs(["Review", "Audio", "Data + Export"])

    with review_tab:
        left, right = st.columns(2)
        with left:
            st.caption("Raw transcript")
            st.text_area(
                "Raw transcript",
                value=raw_text,
                height=200,
                key=f"raw_transcript_{result['run_id']}",
                label_visibility="collapsed",
                disabled=True,
            )
        with right:
            st.caption("Cleaned transcript")
            st.text_area(
                "Cleaned transcript",
                value=cleaned_text,
                height=200,
                key=f"cleaned_transcript_{result['run_id']}",
                label_visibility="collapsed",
                disabled=True,
            )

        st.subheader("Extracted Incident Fields")
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.markdown(f"**Units:** {', '.join(incident.get('units', [])) or 'none'}")
        with col_b:
            st.markdown(f"**Hazards:** {', '.join(incident.get('hazards', [])) or 'none'}")
        with col_c:
            st.markdown(f"**Actions:** {', '.join(incident.get('actions', [])) or 'none'}")

    with audio_tab:
        st.caption(
            "Quick A/B listen before sign-off. This helps confirm preprocess effects on intelligibility."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original upload**")
            _render_audio_player(result["input_audio_bytes"], result.get("input_audio_mime", "audio/wav"))
            st.caption(f"{result['source_sr']} Hz | {result['duration_sec']:.2f}s")
        with col2:
            st.markdown("**Prepared (16 kHz mono)**")
            _render_audio_player(result["prepared_audio_bytes"], result.get("prepared_audio_mime", "audio/wav"))
            if result["filtered_audio_bytes"]:
                st.markdown("**After radio preprocess**")
                _render_audio_player(
                    result["filtered_audio_bytes"],
                    result.get("filtered_audio_mime", "audio/wav"),
                )

    with data_tab:
        incident_json_str = json.dumps(incident, indent=2)
        full_report = json.dumps(
            {
                "incident": incident,
                "raw_transcript": result["raw_transcript"],
                "cleaned_transcript": result["cleaned_transcript"],
                "meta": all_meta,
            },
            indent=2,
        )

        btn1, btn2, btn3 = st.columns(3)
        btn1.download_button(
            "Download incident.json",
            data=incident_json_str,
            file_name="incident.json",
            use_container_width=True,
        )
        btn2.download_button(
            "Download transcript.txt",
            data=(result["cleaned_transcript"] + "\n"),
            file_name="transcript.txt",
            use_container_width=True,
        )
        btn3.download_button(
            "Download full_report.json",
            data=full_report,
            file_name="full_report.json",
            use_container_width=True,
        )

        with st.expander("Performance metadata", expanded=True):
            st.json(all_meta)

        with st.expander("Debug details"):
            st.write("ASR input stage:", result.get("asr_input_label"))
            st.write("Model config path:", str(_CFG))
            st.write("Cleanup method:", result["cleanup_meta"].get("cleanup_method"))
            st.write("Extract method:", result["extract_meta"].get("extract_method"))
            if result.get("asr_error"):
                st.write("ASR error:", result.get("asr_error"))
