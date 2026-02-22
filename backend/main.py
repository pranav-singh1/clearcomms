"""
ClearComms API — FastAPI backend for React frontend.
Exposes transcription and model-status; keeps on-device inference unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root for pipeline imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from pipeline.asr import transcribe
from pipeline.enhance import enhance_audio
from pipeline.audio_io import load_mono, normalize_peak, resample, save_wav, WHISPER_SR

app = FastAPI(title="ClearComms API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODELS_DIR = _ROOT / "models"
_MAX_TTS_CHARS = 2000
_TTS_CACHE_MAX = max(int(os.getenv("DEEPGRAM_TTS_CACHE_MAX", "50")), 0)
_TTS_CACHE_TTL_SEC = max(int(os.getenv("DEEPGRAM_TTS_CACHE_TTL_SEC", "600")), 0)
_TTS_TIMEOUT_SEC = max(float(os.getenv("DEEPGRAM_TTS_TIMEOUT_SEC", "15")), 1.0)
_DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/speak"
_DEEPGRAM_DEFAULT_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-arcas-en").strip() or "aura-2-arcas-en"
_DEEPGRAM_DEFAULT_SPEED = 1.15
_TTS_ENCODING = "mp3"
_TTS_STREAM_CHUNK_SIZE = 4096
_TTS_CACHE: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()
_TTS_CACHE_LOCK = threading.Lock()


class TTSRequest(BaseModel):
    text: str


def _tts_cache_get(key: str) -> bytes | None:
    if _TTS_CACHE_MAX <= 0:
        return None
    with _TTS_CACHE_LOCK:
        item = _TTS_CACHE.get(key)
        if item is None:
            return None
        created_at, value = item
        if _TTS_CACHE_TTL_SEC > 0 and (time.time() - created_at) > _TTS_CACHE_TTL_SEC:
            _TTS_CACHE.pop(key, None)
            return None
        _TTS_CACHE.move_to_end(key)
        return value


def _tts_cache_set(key: str, audio_bytes: bytes) -> None:
    if _TTS_CACHE_MAX <= 0:
        return
    with _TTS_CACHE_LOCK:
        _TTS_CACHE[key] = (time.time(), audio_bytes)
        _TTS_CACHE.move_to_end(key)
        if _TTS_CACHE_TTL_SEC > 0:
            now = time.time()
            expired = [k for k, (ts, _) in _TTS_CACHE.items() if (now - ts) > _TTS_CACHE_TTL_SEC]
            for k in expired:
                _TTS_CACHE.pop(k, None)
        while len(_TTS_CACHE) > _TTS_CACHE_MAX:
            _TTS_CACHE.popitem(last=False)

def _model_files_present() -> bool:
    enc = _ROOT / "models" / "WhisperEncoder.onnx"
    dec = _ROOT / "models" / "WhisperDecoder.onnx"
    return enc.exists() and dec.exists()


@app.get("/api/model-status")
def model_status():
    return {"models_found": _model_files_present()}


def _tts_available() -> bool:
    return bool(os.getenv("DEEPGRAM_API_KEY", "").strip())


def _tts_model() -> str:
    return os.getenv("DEEPGRAM_TTS_MODEL", "").strip() or _DEEPGRAM_DEFAULT_MODEL


def _tts_speed() -> float:
    raw = os.getenv("DEEPGRAM_TTS_SPEED", "").strip()
    if not raw:
        return _DEEPGRAM_DEFAULT_SPEED
    try:
        speed = float(raw)
    except ValueError:
        return 1.0
    # Deepgram Aura-2 speed control expects a 0.7–1.5 multiplier (Early Access); out-of-range values fall back to 1.0.
    if speed < 0.7 or speed > 1.5:
        return 1.0
    return speed


def _decode_error_payload(payload: bytes) -> str:
    if not payload:
        return ""
    try:
        text = payload.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""
    if not text:
        return ""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get("error") or data.get("message") or data)
    except Exception:
        pass
    return text


def _synthesize_with_deepgram(text: str, model: str, encoding: str, speed: float) -> bytes:
    # Deepgram TTS API:
    # Endpoint: POST https://api.deepgram.com/v1/speak
    # Auth header: Authorization: Token <DEEPGRAM_API_KEY>
    # Content-Type: application/json
    # Query string: model=aura-2-arcas-en (or configured model)
    # Optional query: encoding=mp3 (default is mp3)
    # Optional query: speed=1.15 (speaking rate multiplier; default 1.0)
    # JSON body: { "text": "Hello ..." }
    # Response: binary audio stream (content-type audio/mpeg), often chunked.
    key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "TTS unavailable: DEEPGRAM_API_KEY is not configured.")

    query = urllib.parse.urlencode({"model": model, "encoding": encoding, "speed": speed})
    url = f"{_DEEPGRAM_ENDPOINT}?{query}"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TTS_TIMEOUT_SEC) as resp:
            body = resp.read()
            status = getattr(resp, "status", 200)
            if status != 200:
                err_text = _decode_error_payload(body)
                raise HTTPException(status, err_text or "Deepgram TTS request failed.")
            if not body:
                raise HTTPException(502, "Deepgram TTS returned empty audio.")
            return body
    except urllib.error.HTTPError as e:
        err_body = e.read()
        err_text = _decode_error_payload(err_body)
        raise HTTPException(e.code, err_text or "Deepgram TTS request failed.")
    except socket.timeout:
        raise HTTPException(504, "Deepgram TTS request timed out.")
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None) or "Network error."
        raise HTTPException(502, f"Deepgram TTS request failed: {reason}")


@app.get("/api/tts-status")
def tts_status():
    available = _tts_available()
    reason = None if available else "Missing DEEPGRAM_API_KEY."
    return {"available": available, "model": _tts_model(), "reason": reason}


@app.post("/api/tts")
def api_tts(payload: TTSRequest):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    if len(text) > _MAX_TTS_CHARS:
        raise HTTPException(400, f"text is too long (max {_MAX_TTS_CHARS} chars)")
    if not _tts_available():
        raise HTTPException(503, "TTS unavailable: DEEPGRAM_API_KEY is not configured.")

    model = _tts_model()
    speed = _tts_speed()
    cache_key = hashlib.sha256(f"{model}|{_TTS_ENCODING}|{speed}|{text}".encode("utf-8")).hexdigest()
    cached_audio = _tts_cache_get(cache_key)
    if cached_audio is not None:
        return Response(content=cached_audio, media_type="audio/mpeg", headers={"X-TTS-Cache": "HIT"})

    audio_bytes = _synthesize_with_deepgram(text, model, _TTS_ENCODING, speed)
    _tts_cache_set(cache_key, audio_bytes)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"X-TTS-Cache": "MISS", "X-TTS-Model": model},
    )


def _iter_cached_audio(audio_bytes: bytes, chunk_size: int):
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i : i + chunk_size]


def _open_deepgram_stream(text: str, model: str, encoding: str, speed: float) -> urllib.response.addinfourl:
    key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "TTS unavailable: DEEPGRAM_API_KEY is not configured.")
    query = urllib.parse.urlencode({"model": model, "encoding": encoding, "speed": speed})
    url = f"{_DEEPGRAM_ENDPOINT}?{query}"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Token {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=_TTS_TIMEOUT_SEC)
        status = getattr(resp, "status", 200)
        if status != 200:
            err_text = _decode_error_payload(resp.read())
            raise HTTPException(status, err_text or "Deepgram TTS request failed.")
        return resp
    except urllib.error.HTTPError as e:
        err_body = e.read()
        err_text = _decode_error_payload(err_body)
        raise HTTPException(e.code, err_text or "Deepgram TTS request failed.")
    except socket.timeout:
        raise HTTPException(504, "Deepgram TTS request timed out.")
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", None) or "Network error."
        raise HTTPException(502, f"Deepgram TTS request failed: {reason}")


@app.post("/api/tts-stream")
def api_tts_stream(payload: TTSRequest):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    if len(text) > _MAX_TTS_CHARS:
        raise HTTPException(400, f"text is too long (max {_MAX_TTS_CHARS} chars)")
    if not _tts_available():
        raise HTTPException(503, "TTS unavailable: DEEPGRAM_API_KEY is not configured.")

    model = _tts_model()
    speed = _tts_speed()
    cache_key = hashlib.sha256(f"{model}|{_TTS_ENCODING}|{speed}|{text}".encode("utf-8")).hexdigest()
    cached_audio = _tts_cache_get(cache_key)
    if cached_audio is not None:
        return StreamingResponse(
            iter(_iter_cached_audio(cached_audio, _TTS_STREAM_CHUNK_SIZE)),
            media_type="audio/mpeg",
            headers={"X-TTS-Cache": "HIT", "X-TTS-Model": model},
        )

    resp = _open_deepgram_stream(text, model, _TTS_ENCODING, speed)
    buffer = bytearray()

    def _stream():
        try:
            while True:
                chunk = resp.read(_TTS_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                buffer.extend(chunk)
                yield chunk
        finally:
            resp.close()
            if buffer:
                _tts_cache_set(cache_key, bytes(buffer))

    return StreamingResponse(
        _stream(),
        media_type="audio/mpeg",
        headers={"X-TTS-Cache": "MISS", "X-TTS-Model": model},
    )


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    apply_radio_filter: str = Form("true"),
    normalize: str = Form("true"),
    source: str = Form("file"),
    radio_intensity: str = Form("50"),
):
    apply_radio = apply_radio_filter.lower() in ("true", "1", "yes")
    do_normalize = normalize.lower() in ("true", "1", "yes")
    is_mic = source.lower() == "mic"
    raw_intensity = max(0.0, min(100.0, float(radio_intensity)))
    # Map 50 -> 0.0 (no change), 100 -> 0.7 (toned-down max effect).
    intensity = max(0.0, (raw_intensity - 50.0) / 50.0) * 0.7

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    if suffix.lower() not in (".wav", ".flac", ".ogg", ".mp3", ".m4a"):
        raise HTTPException(400, "Unsupported format. Use WAV, FLAC, OGG, MP3, or M4A.")

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read upload: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(contents)
        input_path = Path(tf.name)

    tmp_dir = _ROOT / "runs"
    tmp_dir.mkdir(exist_ok=True)
    pre16_path = tmp_dir / "preprocessed_16k.wav"
    filt_path = tmp_dir / "radio_filtered_16k.wav"

    try:
        audio, sr = load_mono(str(input_path))
        audio_16k, _ = resample(audio, sr, WHISPER_SR)
        if do_normalize:
            audio_16k = normalize_peak(audio_16k)
        save_wav(str(pre16_path), audio_16k, WHISPER_SR)

        if apply_radio:
            filtered = enhance_audio(audio_16k, WHISPER_SR, intensity)
            if is_mic:
                # Mic audio is already clean — run a second enhancement pass
                # after the radio filter to further clarify the signal.
                filtered = enhance_audio(filtered, WHISPER_SR, intensity)
            save_wav(str(filt_path), filtered, WHISPER_SR)
            asr_input = filt_path
        else:
            asr_input = pre16_path

        # Always encode preprocessed clips for playback (even if transcribe fails)
        audio_prepared_b64 = None
        audio_filtered_b64 = None
        if pre16_path.exists():
            audio_prepared_b64 = base64.b64encode(pre16_path.read_bytes()).decode("utf-8")
        if apply_radio and filt_path.exists():
            audio_filtered_b64 = base64.b64encode(filt_path.read_bytes()).decode("utf-8")

        duration_sec = round(len(audio) / max(sr, 1), 2)
        payload = {
            "success": True,
            "error": None,
            "text": "",
            "cleaned_transcript": "",
            "meta": {},
            "audio_prepared_b64": audio_prepared_b64,
            "audio_filtered_b64": audio_filtered_b64,
            "apply_radio_filter": apply_radio,
            "duration_sec": duration_sec,
            "sample_rate_original": sr,
        }

        try:
            t0 = time.time()
            text, meta = transcribe(str(asr_input), WHISPER_SR)
            ui_total_ms = (time.time() - t0) * 1000.0
            payload["text"] = (text or "").strip() or "(no transcript)"
            payload["cleaned_transcript"] = payload["text"]
            payload["meta"] = {**meta, "ui_total_ms": round(ui_total_ms, 1)}
        except FileNotFoundError as e:
            payload["success"] = False
            payload["error"] = "ONNX encoder/decoder not found. Place WhisperEncoder.onnx and WhisperDecoder.onnx in models/."
        except Exception as e:
            payload["success"] = False
            payload["error"] = str(e)

        return payload
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try:
            os.remove(str(input_path))
        except Exception:
            pass
