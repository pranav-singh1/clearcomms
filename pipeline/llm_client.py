"""LLM cleanup + incident extraction.

Designed so the Streamlit UI can run on any laptop.

Modes:
  - mock: no model needed (fast UI testing)
  - openai: OpenAI-compatible local server (LM Studio, vLLM, etc.)
  - ollama: Ollama local server

All calls are local HTTP by default (offline once the model is installed).

Key goal for hackathon reliability:
- If the LLM is down / returns bad JSON, we fall back to deterministic rules.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple


LLMMode = Literal["mock", "openai", "ollama"]


@dataclass
class LLMConfig:
    mode: LLMMode = "mock"
    base_url: str = "http://localhost:1234"  # LM Studio default
    model: str = "llama-3.1-8b-instruct"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 30.0


REQUIRED_KEYS = [
    "cleaned_transcript",
    "request_type",
    "urgency",
    "location",
    "units",
    "hazards",
    "actions",
    "uncertainties",
]


INCIDENT_SCHEMA_HINT = {
    "cleaned_transcript": "string",
    "request_type": "fire|medical|rescue|hazard|other|unknown",
    "urgency": "low|medium|high|unknown",
    "location": "string|null",
    "units": "list[string]",
    "hazards": "list[string]",
    "actions": "list[string]",
    "uncertainties": "list[string] (short human notes)",
}


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _build_prompt(transcript: str) -> Tuple[str, str]:
    system = (
        "You are a safety-critical dispatch assistant. "
        "You MUST be conservative and never invent details. "
        "Only use information explicitly present in the transcript. "
        "If a field is unknown, output null (for location) or 'unknown'. "
        "Output MUST be valid JSON only, no prose, no markdown."
    )

    user = (
        "Given the raw radio transcript below, do two things:\n"
        "1) Produce a cleaned transcript with corrected casing/punctuation and fixed obvious ASR garbles.\n"
        "2) Extract an incident card as JSON fields.\n\n"
        f"JSON schema (types only): {json.dumps(INCIDENT_SCHEMA_HINT)}\n\n"
        "Raw transcript:\n"
        f"{transcript}\n\n"
        "Return JSON with exactly these keys: "
        "cleaned_transcript, request_type, urgency, location, units, hazards, actions, uncertainties."
    )
    return system, user


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove ```json ... ``` blocks if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response."""
    text = _strip_code_fences(text)

    # If response contains extra tokens, try to find a JSON object.
    if not text.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)

    # Tiny repairs: trailing commas
    text = re.sub(r",\s*\}", "}", text)
    text = re.sub(r",\s*\]", "]", text)

    return json.loads(text)


def _validate_and_fill(out: Dict[str, Any], fallback_transcript: str) -> Dict[str, Any]:
    """Ensure required keys exist with sane types, fill defaults."""
    fixed: Dict[str, Any] = dict(out) if isinstance(out, dict) else {}

    fixed.setdefault("cleaned_transcript", fallback_transcript.strip())
    fixed.setdefault("request_type", "unknown")
    fixed.setdefault("urgency", "unknown")
    fixed.setdefault("location", None)
    fixed.setdefault("units", [])
    fixed.setdefault("hazards", [])
    fixed.setdefault("actions", [])
    fixed.setdefault("uncertainties", [])

    # Coerce list types
    for k in ("units", "hazards", "actions", "uncertainties"):
        if not isinstance(fixed.get(k), list):
            fixed[k] = [str(fixed[k])]

    # Normalize strings
    fixed["request_type"] = str(fixed.get("request_type") or "unknown").lower()
    fixed["urgency"] = str(fixed.get("urgency") or "unknown").lower()

    # Keep location as string or None
    loc = fixed.get("location")
    if loc is not None:
        loc = str(loc).strip()
        fixed["location"] = loc if loc else None

    # Only keep expected keys (helps judge safety)
    return {k: fixed.get(k) for k in REQUIRED_KEYS}


def _mock_cleanup(transcript: str) -> Dict[str, Any]:
    t = transcript.strip()

    fixes = {
        "mapple": "Maple",
        "streat": "Street",
        "st": "St",
        "ave": "Ave",
        "rd": "Rd",
        "bkup": "backup",
    }

    words = []
    for w in t.split():
        key = re.sub(r"[^a-zA-Z]", "", w).lower()
        rep = fixes.get(key)
        if rep:
            suffix = w[len(re.sub(r"[^a-zA-Z]", "", w)) :]
            words.append(rep + suffix)
        else:
            words.append(w)

    cleaned = " ".join(words)
    if cleaned and cleaned[-1].isalnum():
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0:1].upper() + cleaned[1:]

    units = re.findall(r"\b(?:engine|unit|medic|ladder)\s*\d+\b", cleaned, flags=re.I)
    units = [u.title().replace("  ", " ") for u in units]

    loc = None
    m = re.search(
        r"\b\d{1,5}\s+[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})*\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b",
        cleaned,
    )
    if m:
        loc = m.group(0)

    hazards = []
    for k in ("smoke", "fire", "downed lines", "gas", "leak", "crowd", "shots", "weapon"):
        if re.search(rf"\b{k.split()[0]}\b", cleaned, flags=re.I):
            hazards.append(k)

    request_type = "unknown"
    if re.search(r"\bsmoke\b|\bfire\b|\bflames\b", cleaned, flags=re.I):
        request_type = "fire"
    elif re.search(r"\bmedical\b|\binjured\b|\bambulance\b", cleaned, flags=re.I):
        request_type = "medical"
    elif re.search(r"\bsearch\b|\brescue\b|\bmissing\b", cleaned, flags=re.I):
        request_type = "rescue"
    elif hazards:
        request_type = "hazard"

    urgency = "unknown"
    if re.search(r"\bimmediate\b|\burgent\b|\bmayday\b|\bneed backup\b", cleaned, flags=re.I):
        urgency = "high"

    actions = []
    for a in ("respond", "evacuate", "need backup", "send ambulance", "stage", "hold"):
        if re.search(rf"\b{re.escape(a)}\b", cleaned, flags=re.I):
            actions.append(a)

    uncertainties = []
    if loc is None:
        uncertainties.append("Location not clearly stated")
    if not units:
        uncertainties.append("Unit/callsign not detected")

    return {
        "cleaned_transcript": cleaned,
        "request_type": request_type,
        "urgency": urgency,
        "location": loc,
        "units": units,
        "hazards": hazards,
        "actions": actions,
        "uncertainties": uncertainties,
    }


def cleanup_and_extract(transcript: str, cfg: Optional[LLMConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (incident_json, meta)."""
    cfg = cfg or LLMConfig()
    t0 = time.time()

    if cfg.mode == "mock":
        out = _mock_cleanup(transcript)
        latency_ms = (time.time() - t0) * 1000.0
        return _validate_and_fill(out, transcript), {
            "llm_backend": "mock",
            "llm_latency_ms": round(latency_ms, 1),
        }

    system, user = _build_prompt(transcript)

    try:
        if cfg.mode == "openai":
            url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
            payload = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            }
            resp = _post_json(url, payload, timeout_s=cfg.timeout_s)
            content = resp["choices"][0]["message"]["content"]
            out = _extract_json(content)

        elif cfg.mode == "ollama":
            url = cfg.base_url.rstrip("/") + "/api/chat"
            payload = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
            }
            resp = _post_json(url, payload, timeout_s=cfg.timeout_s)
            content = resp.get("message", {}).get("content", "")
            out = _extract_json(content)

        else:
            raise ValueError(f"Unknown LLM mode: {cfg.mode}")

        out = _validate_and_fill(out, transcript)
        latency_ms = (time.time() - t0) * 1000.0
        meta = {
            "llm_backend": cfg.mode,
            "llm_latency_ms": round(latency_ms, 1),
            "llm_model": cfg.model,
            "llm_base_url": cfg.base_url,
        }
        return out, meta

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
        out = _mock_cleanup(transcript)
        out = _validate_and_fill(out, transcript)
        out.setdefault("uncertainties", [])
        out["uncertainties"].append(f"LLM fallback used: {type(e).__name__}")

        latency_ms = (time.time() - t0) * 1000.0
        meta = {
            "llm_backend": f"{cfg.mode}_fallback",
            "llm_latency_ms": round(latency_ms, 1),
            "llm_model": cfg.model,
            "llm_base_url": cfg.base_url,
            "error": f"{type(e).__name__}: {e}",
        }
        return out, meta
