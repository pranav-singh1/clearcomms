"""pipeline/extract.py — Structured incident extraction (Layer 5).

Takes a cleaned transcript and produces a structured incident JSON with:
  - request_type, urgency, location, hazards, units, actions

Two backends:
  1. LLaMA 3 8B Instruct (if model is loaded)
  2. Rule-based keyword extraction (always works, no model needed)
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable, Tuple

INCIDENT_SCHEMA = {
    "request_type": "unknown",
    "urgency": "medium",
    "location": "unknown",
    "hazards": [],
    "units": [],
    "actions": [],
}

_SYSTEM_PROMPT = """You extract structured incident data from emergency radio transcripts.

Output ONLY a JSON object with exactly these fields:
- "request_type": one of "fire", "medical", "rescue", "hazard", "traffic", "police", "other"
- "urgency": one of "low", "medium", "high", "critical"
- "location": street address or location description, or "unknown"
- "hazards": list of hazards mentioned (e.g. "smoke", "downed lines", "chemical spill")
- "units": list of units mentioned (e.g. "Engine 12", "Unit 4", "Medic 7")
- "actions": list of requested or reported actions (e.g. "send backup", "evacuate", "transport patient")

Rules:
- Only extract information explicitly stated in the transcript
- Use "unknown" or empty lists for fields with no information
- Output valid JSON only, no explanation or markdown"""

# --- Rule-based keyword extraction fallback ---

_REQUEST_TYPES = {"fire", "medical", "rescue", "hazard", "traffic", "police", "other", "unknown"}
_URGENCY_LEVELS = {"low", "medium", "high", "critical"}

_TYPE_KEYWORDS = {
    "fire": ["fire", "smoke", "burn", "flame", "arson"],
    "medical": ["medical", "patient", "injury", "injured", "ambulance", "cardiac", "breathing", "trauma", "transport"],
    "rescue": ["rescue", "trapped", "missing", "search", "extricate"],
    "hazard": ["hazard", "hazmat", "chemical", "spill", "downed lines", "gas leak", "radiation"],
    "traffic": ["traffic", "collision", "accident", "crash", "vehicle", "mvp", "mva"],
    "police": ["police", "suspect", "arrest", "weapon", "shots", "robbery", "assault"],
}

_CRITICAL_URGENCY_TERMS = [
    "critical",
    "mayday",
    "life threatening",
    "code 3",
    "immediate",
    "emergency emergency",
]

_HIGH_URGENCY_TERMS = [
    "urgent",
    "backup",
    "need backup",
    "send backup",
    "high priority",
    "respond now",
    "emergency",
]

_LOW_URGENCY_TERMS = [
    "non-emergency",
    "non emergency",
    "routine",
    "standby",
    "advisory",
    "no rush",
]

_UNIT_PATTERN = re.compile(
    r"(engine|unit|truck|ladder|medic|rescue|squad|battalion|chief|ambulance|ems|car)\s*\d+",
    re.IGNORECASE,
)

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|Lane|Ln|Way|Court|Ct|Highway|Hwy|Place|Pl)"
)
_DIRECTION = r"(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West)\.?"

_ADDRESS_PATTERN = re.compile(
    rf"\b\d{{1,6}}\s+(?:(?:{_DIRECTION})\s+)?[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+){{0,2}}\s+{_STREET_SUFFIX}\b",
    re.IGNORECASE,
)

_PREPOSITION_STREET_PATTERN = re.compile(
    rf"\b(?:at|near|on)\s+((?:(?:{_DIRECTION})\s+)?[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+){{0,2}}\s+{_STREET_SUFFIX})\b",
    re.IGNORECASE,
)

_CROSS_STREET_PATTERN = re.compile(
    r"\b(?:at|near|on)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)\s+(?:and|&|/)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?)\b",
    re.IGNORECASE,
)

_ACTION_KEYWORDS = [
    "respond", "send backup", "backup", "evacuate", "transport",
    "stage", "set up command", "start triage", "need medical",
    "request additional", "shut down", "block", "divert",
    "search", "rescue", "contain", "ventilate", "suppress",
]


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for raw in values:
        val = re.sub(r"\s+", " ", raw).strip(" .,;:")
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _normalize_list_field(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [v.strip() for v in value.split(",")]
        return _dedupe_strings(candidates)
    if isinstance(value, list):
        candidates = [v for v in value if isinstance(v, str)]
        return _dedupe_strings(candidates)
    return []


def _format_unit(raw_unit: str) -> str:
    tokens = re.sub(r"\s+", " ", raw_unit).strip().split(" ")
    if not tokens:
        return raw_unit
    if tokens[0].lower() == "ems":
        tokens[0] = "EMS"
    else:
        tokens[0] = tokens[0].capitalize()
    return " ".join(tokens)


def _clean_location(raw_location: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_location).strip(" .,;:")
    return cleaned if cleaned else "unknown"


def _extract_location(text: str) -> str:
    address_match = _ADDRESS_PATTERN.search(text)
    if address_match:
        return _clean_location(address_match.group(0))

    street_match = _PREPOSITION_STREET_PATTERN.search(text)
    if street_match:
        return _clean_location(street_match.group(1))

    cross_match = _CROSS_STREET_PATTERN.search(text)
    if cross_match:
        return _clean_location(f"{cross_match.group(1)} and {cross_match.group(2)}")

    return "unknown"


def _detect_urgency(lower_text: str) -> str:
    # "non-emergency" must win over generic "emergency".
    if any(_contains_phrase(lower_text, term) for term in _LOW_URGENCY_TERMS[:2]):
        return "low"
    if any(_contains_phrase(lower_text, term) for term in _CRITICAL_URGENCY_TERMS):
        return "critical"
    if any(_contains_phrase(lower_text, term) for term in _HIGH_URGENCY_TERMS):
        return "high"
    if any(_contains_phrase(lower_text, term) for term in _LOW_URGENCY_TERMS[2:]):
        return "low"
    return "medium"


def _sanitize_incident(result: dict[str, Any]) -> dict:
    incident = dict(INCIDENT_SCHEMA)
    if not isinstance(result, dict):
        return incident

    request_type = str(result.get("request_type", "")).strip().lower()
    if request_type in _REQUEST_TYPES:
        incident["request_type"] = request_type
    elif request_type:
        incident["request_type"] = "other"

    urgency = str(result.get("urgency", "")).strip().lower()
    if urgency in _URGENCY_LEVELS:
        incident["urgency"] = urgency

    location = result.get("location")
    if isinstance(location, str):
        incident["location"] = _clean_location(location)

    incident["hazards"] = _normalize_list_field(result.get("hazards"))
    incident["units"] = _normalize_list_field(result.get("units"))
    incident["actions"] = _normalize_list_field(result.get("actions"))
    return incident


def _rule_based_extract(text: str) -> dict:
    """Keyword-based structured extraction — no model needed."""
    lower = text.lower()
    result = dict(INCIDENT_SCHEMA)

    # Request type
    for rtype, keywords in _TYPE_KEYWORDS.items():
        if any(_contains_phrase(lower, kw) for kw in keywords):
            result["request_type"] = rtype
            break

    # Urgency
    result["urgency"] = _detect_urgency(lower)

    # Location
    result["location"] = _extract_location(text)

    # Units
    result["units"] = _dedupe_strings(_format_unit(m.group(0)) for m in _UNIT_PATTERN.finditer(text))

    # Hazards
    hazard_terms = ["smoke", "fire", "chemical", "gas leak", "downed lines",
                    "flooding", "explosion", "radiation", "collapse", "spill"]
    result["hazards"] = _dedupe_strings(h for h in hazard_terms if _contains_phrase(lower, h))

    # Actions
    result["actions"] = _dedupe_strings(a for a in _ACTION_KEYWORDS if _contains_phrase(lower, a))

    return _sanitize_incident(result)


def extract_incident(cleaned_text: str) -> Tuple[dict, dict]:
    """Extract structured incident data from a cleaned transcript.

    Returns:
        (incident_dict, metadata_dict)
    """
    if not cleaned_text.strip():
        return dict(INCIDENT_SCHEMA), {"extract_method": "skip", "extract_latency_ms": 0.0}

    from pipeline import llm

    t0 = time.time()

    if llm.is_available():
        result = llm.chat_json(_SYSTEM_PROMPT, cleaned_text, max_tokens=512)
        if result:
            incident = _sanitize_incident(result)
            method = "llama3_8b_instruct"
        else:
            incident = _rule_based_extract(cleaned_text)
            method = "rules_fallback"
    else:
        incident = _rule_based_extract(cleaned_text)
        method = "rules"

    latency_ms = (time.time() - t0) * 1000

    meta = {
        "extract_method": method,
        "extract_latency_ms": round(latency_ms, 1),
    }

    return incident, meta
