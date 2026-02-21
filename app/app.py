import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st
from scipy.signal import butter, lfilter

from pipeline.audio_io import load_audio, normalize_peak, resample_to_16k, safe_wav_bytes
from pipeline.llm_client import LLMConfig, cleanup_and_extract
from pipeline.text_utils import html_diff, word_error_rate


# -----------------------------
# Page
# -----------------------------

st.set_page_config(page_title="CLEARCOMMS", layout="wide")
st.title("CLEARCOMMS — Offline Radio Transcription → Incident Card")

st.markdown(
    """
<style>
.cc-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; margin-right:6px; border:1px solid rgba(0,0,0,0.06); }
.cc-ok { background:#dcfce7; color:#14532d; }
.cc-warn { background:#ffedd5; color:#7c2d12; }
.cc-bad { background:#fee2e2; color:#7f1d1d; }
.cc-kv { padding:10px 12px; border:1px solid #e5e7eb; border-radius:14px; }
.cc-k { font-size:12px; color:#6b7280; margin-bottom:2px; }
.cc-v { font-size:16px; font-weight:600; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Audio preprocessing
# -----------------------------


def bandpass_radio(x: np.ndarray, sr: int, lo=300, hi=3400, order=4) -> np.ndarray:
    nyq = 0.5 * sr
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return lfilter(b, a, x).astype(np.float32)


def soft_gate(x: np.ndarray, thr: float = 0.02) -> np.ndarray:
    mag = np.abs(x)
    gate = np.where(mag < thr, mag / max(thr, 1e-6), 1.0)
    return (x * gate).astype(np.float32)


# -----------------------------
# ASR backends (local Whisper or mock)
# -----------------------------


@st.cache_resource
def _load_asr_module():
    # Imported lazily so UI can still run even if optimum/transformers aren't installed.
    from pipeline.asr import transcribe

    return transcribe


def asr_transcribe(mode: str, wav_path: str, sr_hint: int) -> tuple[str, dict]:
    if mode == "Mock (no model)":
        # Mimic radio transcript style.
        name = Path(wav_path).stem.lower()
        canned = (
            "engine 12 respond 235 mapple street smoke visible need backup"
            if "maple" in name or "sample" in name
            else "unit 4 to dispatch patient injured requesting medical assistance"
        )
        return canned, {
            "backend": "mock",
            "asr_latency_ms": 5.0,
            "audio_duration_sec": None,
            "realtime_factor": None,
        }

    transcribe = _load_asr_module()
    return transcribe(wav_path, sr_hint)


# -----------------------------
# Helpers
# -----------------------------


def incident_note(card: Dict[str, Any]) -> str:
    """Deterministic note for the demo (no hallucination)."""
    rt = (card.get("request_type") or "unknown").upper()
    urg = (card.get("urgency") or "unknown").upper()
    loc = card.get("location")
    units = card.get("units") or []
    haz = card.get("hazards") or []
    acts = card.get("actions") or []

    header = f"{rt} | {urg}"
    if loc:
        header += f" | {loc}"

    lines = [header]
    if units:
        lines.append("Units: " + ", ".join(units))
    if haz:
        lines.append("Hazards: " + ", ".join(haz))
    if acts:
        lines.append("Actions/Requests: " + ", ".join(acts))

    return "\n".join(lines)


def kv(label: str, value: str):
    st.markdown(
        f"<div class='cc-kv'><div class='cc-k'>{label}</div><div class='cc-v'>{value}</div></div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar controls
# -----------------------------


with st.sidebar:
    st.header("Runtime")

    asr_mode = st.selectbox(
        "ASR backend",
        ["Mock (no model)", "Local Whisper (pipeline/asr.py)"],
        index=0,
        help="Mock lets you test the UI on any laptop. Local Whisper uses pipeline/asr.py (CPU on normal laptops, QNN on Qualcomm).",
    )

    st.divider()
    st.header("LLM cleanup + incident extraction")

    llm_mode = st.selectbox(
        "LLM backend",
        ["Mock (no model)", "OpenAI-compatible local server", "Ollama"],
        index=0,
        help="Mock works everywhere. OpenAI-compatible works with LM Studio / vLLM servers. Ollama works with its local server.",
    )

    llm_base_url = st.text_input(
        "LLM base URL",
        value="http://localhost:1234" if llm_mode == "OpenAI-compatible local server" else "http://localhost:11434",
    )

    llm_model = st.text_input(
        "LLM model name",
        value="llama-3.1-8b-instruct",
        help="Ollama examples: llama3.1:8b-instruct. LM Studio/vLLM: use the server's model name.",
    )

    llm_max_tokens = st.slider("Max output tokens", 64, 512, 256, step=32)

    st.divider()
    st.header("Audio preprocessing")

    use_radio_bp = st.checkbox("Bandpass 300–3400 Hz", value=True)
    use_gate = st.checkbox("Soft noise gate", value=True)
    do_normalize = st.checkbox("Peak normalize", value=True)

    st.divider()
    st.header("UX")
    show_diff = st.checkbox("Show diff (raw → cleaned)", value=True)
    show_note = st.checkbox("Show incident note", value=True)
    allow_manual_edit = st.checkbox("Allow manual transcript edits", value=True)


# -----------------------------
# Main UI
# -----------------------------


tab_run, tab_eval, tab_help = st.tabs(["Run", "Eval", "Help"])

with tab_run:
    uploaded = st.file_uploader(
        "Upload audio (WAV/FLAC/OGG recommended)",
        type=["wav", "flac", "ogg"],
    )

    demo_dir = Path("radio_dispatch_filter/radio_audio")
    demo_files = sorted(demo_dir.glob("*.wav")) if demo_dir.exists() else []

    picked = "(none)"
    if demo_files:
        picked = st.selectbox(
            "…or pick a demo radio clip from the repo",
            ["(none)"] + [p.name for p in demo_files],
            index=0,
        )

    def _save_upload_to_temp(upload) -> str:
        suffix = os.path.splitext(upload.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(upload.read())
            return tf.name

    def _save_bytes_to_temp_wav(wav_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tf.write(wav_bytes)
            return tf.name

    input_path = None
    if uploaded:
        input_path = _save_upload_to_temp(uploaded)
    elif picked != "(none)":
        input_path = str(demo_dir / picked)

    if not input_path:
        st.info(
            "To test UI on your normal laptop: set **ASR=Mock** and **LLM=Mock**. "
            "You can still upload audio and see the full UX flow."
        )
        st.stop()

    colL, colR = st.columns([1, 1])

    # Load + preprocess
    clip = load_audio(input_path)
    x, sr = clip.samples, clip.sr

    with colL:
        st.subheader("Input audio")
        st.audio(open(input_path, "rb").read())
        st.caption(f"Loaded: {Path(input_path).name} | {sr} Hz | {len(x)/max(sr,1):.2f}s")

    pre_t0 = time.time()
    x16, sr16 = resample_to_16k(x, sr)
    if use_radio_bp:
        x16 = bandpass_radio(x16, sr16)
    if use_gate:
        x16 = soft_gate(x16, thr=0.02)
    if do_normalize:
        x16 = normalize_peak(x16, peak=0.95)
    pre_ms = (time.time() - pre_t0) * 1000.0

    processed_wav_bytes = safe_wav_bytes(x16, sr16)
    processed_path = _save_bytes_to_temp_wav(processed_wav_bytes)

    with colL:
        st.subheader("Processed audio (16 kHz mono)")
        st.audio(processed_wav_bytes)

    # ASR
    asr_t0 = time.time()
    raw_text, asr_meta = asr_transcribe(asr_mode, processed_path, sr16)
    asr_ms = (time.time() - asr_t0) * 1000.0

    # LLM cleanup + incident extraction
    llm_cfg = LLMConfig(
        mode=(
            "mock"
            if llm_mode == "Mock (no model)"
            else "openai"
            if llm_mode == "OpenAI-compatible local server"
            else "ollama"
        ),
        base_url=llm_base_url,
        model=llm_model,
        max_tokens=int(llm_max_tokens),
    )

    llm_out, llm_meta = cleanup_and_extract(raw_text, llm_cfg)
    cleaned = (llm_out.get("cleaned_transcript") or "").strip() or raw_text

    # TTS (ONLINE) - do not let demo crash if offline
    enable_tts = st.sidebar.checkbox("Text-to-speech (Deepgram) [ONLINE]", value=False)
    deepgram_key = st.sidebar.text_input("Deepgram API Key", type="password", value=os.getenv("DEEPGRAM_API_KEY",""))
    deepgram_voice = st.sidebar.text_input("Deepgram voice/model", value="aura-asteria-en")

    if enable_tts:
        try:
            from pipeline.tts_deepgram import deepgram_tts_mp3
            mp3_bytes = deepgram_tts_mp3(cleaned, deepgram_key, model=deepgram_voice)
            st.subheader("Cleaned transcript (TTS)")
            st.audio(mp3_bytes, format="audio/mp3")
        except Exception as e:
            st.warning(f"TTS failed (continuing): {e}")


    # Right pane: transcripts + card
    with colR:
        st.subheader("Status")

        pill_backend = asr_meta.get("backend", "mock")
        if pill_backend == "qnn":
            st.markdown("<span class='cc-pill cc-ok'>ASR: QNN/NPU</span>", unsafe_allow_html=True)
        elif pill_backend == "ort":
            st.markdown("<span class='cc-pill cc-warn'>ASR: ORT CPU</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='cc-pill cc-warn'>ASR: mock</span>", unsafe_allow_html=True)

        if llm_meta.get("llm_backend") == "mock":
            st.markdown("<span class='cc-pill cc-warn'>LLM: mock</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='cc-pill cc-ok'>LLM: live</span>", unsafe_allow_html=True)

        st.divider()

        st.subheader("Raw transcript (ASR)")
        st.write(raw_text if raw_text else "(no transcript)")

        st.subheader("Cleaned transcript")

        # Optional manual edit loop (very judge-friendly: "human in the loop")
        if allow_manual_edit:
            st.caption("Optional: edit transcript then rerun extraction (helps in high-stress ops)")
            edited = st.text_area("Edit transcript", value=cleaned, height=120)
            if st.button("Re-run extraction using edited text"):
                llm_out, llm_meta = cleanup_and_extract(edited, llm_cfg)
                cleaned = (llm_out.get("cleaned_transcript") or "").strip() or edited

        st.write(cleaned if cleaned else "(no cleaned transcript)")

        if show_diff:
            with st.expander("Show diff (raw → cleaned)", expanded=True):
                st.markdown(html_diff(raw_text, cleaned), unsafe_allow_html=True)

        st.subheader("Incident Card")
        c1, c2, c3 = st.columns(3)
        with c1:
            kv("Request type", (llm_out.get("request_type") or "unknown").upper())
        with c2:
            urg = (llm_out.get("urgency") or "unknown").upper()
            urg_cls = "cc-ok" if urg in ("LOW", "MEDIUM") else "cc-bad" if urg == "HIGH" else "cc-warn"
            kv("Urgency", f"<span class='cc-pill {urg_cls}'>{urg}</span>")
        with c3:
            kv("Location", str(llm_out.get("location") or "(unknown)"))

        st.markdown("#### Full JSON")
        st.json(llm_out)

        # Fast-scanning chips (judge-friendly)
        def _chips(title: str, items):
            if not items:
                return
            st.markdown(f"**{title}:**")
            html = ""
            for it in items:
                html += f"<span class='cc-pill cc-ok'>{str(it)}</span>"
            st.markdown(html, unsafe_allow_html=True)

        _chips("Units", llm_out.get("units") or [])
        _chips("Hazards", llm_out.get("hazards") or [])
        _chips("Actions", llm_out.get("actions") or [])
        if llm_out.get("uncertainties"):
            st.markdown("**Uncertainties:**")
            for u in llm_out.get("uncertainties") or []:
                st.markdown(f"- {u}")

        if show_note:
            st.markdown("#### Incident note")
            st.code(incident_note(llm_out), language="text")

        st.download_button(
            "Download incident JSON",
            data=json.dumps(llm_out, indent=2).encode("utf-8"),
            file_name="incident.json",
            mime="application/json",
        )
        st.download_button(
            "Download incident note",
            data=incident_note(llm_out).encode("utf-8"),
            file_name="incident_note.txt",
            mime="text/plain",
        )

        st.subheader("Performance / Proof")
        duration_sec = float(len(x16) / max(sr16, 1))
        total_ms = pre_ms + asr_ms + float(llm_meta.get("llm_latency_ms", 0.0))

        st.json(
            {
                "preprocess_ms": round(pre_ms, 1),
                "asr_ms_wall": round(asr_ms, 1),
                "asr_meta": asr_meta,
                "llm_meta": llm_meta,
                "audio_duration_sec": round(duration_sec, 3),
                "end_to_end_ms": round(total_ms, 1),
                "end_to_end_realtime_factor": round((total_ms / 1000.0) / max(duration_sec, 1e-3), 3),
            }
        )

    # Cleanup temp file if it was an upload
    try:
        if uploaded and input_path and os.path.exists(input_path):
            os.remove(input_path)
    except Exception:
        pass

    try:
        if processed_path and os.path.exists(processed_path):
            os.remove(processed_path)
    except Exception:
        pass


with tab_eval:
    st.subheader("Quick eval on paired dataset")
    st.caption(
        "If your repo has radio_dispatch_filter/radio_audio and clean_audio paired files, "
        "this computes a quick WER improvement using clean Whisper output as a reference (hackathon-friendly metric)."
    )

    radio_dir = Path("radio_dispatch_filter/radio_audio")
    clean_dir = Path("radio_dispatch_filter/clean_audio")
    has_pairs = radio_dir.exists() and clean_dir.exists()

    if not has_pairs:
        st.warning("Paired dataset folders not found. Expected: radio_dispatch_filter/radio_audio + radio_dispatch_filter/clean_audio")
        st.stop()

    n = st.slider("How many pairs", 1, 30, 10)

    if st.button("Run eval"):
        radios = sorted(radio_dir.glob("*.wav"))[:n]
        rows = []

        llm_cfg = LLMConfig(
            mode=(
                "mock"
                if llm_mode == "Mock (no model)"
                else "openai"
                if llm_mode == "OpenAI-compatible local server"
                else "ollama"
            ),
            base_url=llm_base_url,
            model=llm_model,
            max_tokens=int(llm_max_tokens),
        )

        for rp in radios:
            clean_name = rp.stem.replace("_radio", "") + ".flac"  # same heuristic used in your tests
            cp = clean_dir / clean_name
            if not cp.exists():
                continue

            # reference transcript (clean)
            ref_text, _ = asr_transcribe(asr_mode, str(cp), 16000)

            # noisy transcript (radio)
            hyp_raw, _ = asr_transcribe(asr_mode, str(rp), 16000)
            hyp_cleaned = cleanup_and_extract(hyp_raw, llm_cfg)[0].get("cleaned_transcript", hyp_raw)

            wer_raw = word_error_rate(ref_text, hyp_raw)
            wer_cleaned = word_error_rate(ref_text, hyp_cleaned)

            rows.append(
                {
                    "file": rp.name,
                    "WER_raw": round(wer_raw, 3),
                    "WER_cleaned": round(wer_cleaned, 3),
                    "delta": round(wer_raw - wer_cleaned, 3),
                }
            )

        if rows:
            st.dataframe(rows, use_container_width=True)
            avg_raw = sum(r["WER_raw"] for r in rows) / len(rows)
            avg_clean = sum(r["WER_cleaned"] for r in rows) / len(rows)
            st.write(f"Average WER: raw={avg_raw:.3f} → cleaned={avg_clean:.3f} (Δ={avg_raw-avg_clean:+.3f})")
        else:
            st.warning("No paired samples matched your naming convention.")


with tab_help:
    st.subheader("Run this UI on your normal laptop (no models)")
    st.code(
        """
cd clearcomms-main
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
streamlit run app/app.py

In the sidebar:
- ASR backend = Mock (no model)
- LLM backend = Mock (no model)
""".strip(),
        language="bash",
    )

    st.subheader("Hook in Llama (when ready)")
    st.markdown(
        """
- If you have an **OpenAI-compatible** local server (LM Studio / vLLM):
  - Set **LLM backend = OpenAI-compatible local server**
  - Base URL: `http://localhost:1234` (or your server)
  - Model: whatever the server exposes

- If you run **Ollama**:
  - Set **LLM backend = Ollama**
  - Base URL: `http://localhost:11434`
  - Model: e.g. `llama3.1:8b-instruct`
"""
    )
