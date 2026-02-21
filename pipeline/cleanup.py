"""pipeline/cleanup.py — Transcript cleanup (Layer 4).

Takes raw Whisper output and fixes:
  - Capitalization and punctuation
  - Common ASR garbles (e.g. "mapple" → "maple")
  - Abbreviation expansion (e.g. "med" → "medical")
  - Repeated words from radio jitter

Two backends:
  1. LLaMA 3 8B Instruct (if model is loaded)
  2. Rule-based fallback (always works, no model needed)
"""

from __future__ import annotations

import re
import time
from typing import Tuple

_SYSTEM_PROMPT = """You clean up emergency radio transcripts. The input is raw automatic speech recognition output from noisy radio audio.

Rules:
- Fix spelling errors and ASR mishearings
- Add proper capitalization and punctuation
- Expand common abbreviations (e.g. "med" → "medical", "eng" → "engine", "dept" → "department")
- Remove repeated words caused by radio jitter or echo
- Do NOT add information that was not in the original
- Do NOT summarize or rephrase — keep the speaker's words
- Return ONLY the cleaned transcript, nothing else"""

ABBREVIATIONS = {
    r"\bmed\b": "medical",
    r"\bmeds\b": "medications",
    r"\beng\b": "engine",
    r"\bdept\b": "department",
    r"\bsgt\b": "sergeant",
    r"\blt\b": "lieutenant",
    r"\bcapt\b": "captain",
    r"\bave\b": "avenue",
    r"\bst\b": "street",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bhwy\b": "highway",
    r"\brd\b": "road",
    r"\bpd\b": "police department",
    r"\bfd\b": "fire department",
    r"\bems\b": "emergency medical services",
    r"\brespo\b": "respond",
    r"\bxmit\b": "transmit",
    r"\baffirm\b": "affirmative",
    r"\bneg\b": "negative",
    r"\bposs\b": "possible",
    r"\bveh\b": "vehicle",
    r"\bsubj\b": "subject",
    r"\binj\b": "injured",
}

GARBLE_FIXES = {
    "mapple": "maple",
    "back up": "backup",
    "fire truck": "firetruck",
    "break break": "break,",
}


def _rule_based_cleanup(text: str) -> str:
    """Fast rule-based transcript cleanup — no model needed."""
    if not text.strip():
        return text

    # Remove repeated words (e.g. "go go go" → "go", "the the" → "the")
    cleaned = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Fix known garbles
    for wrong, right in GARBLE_FIXES.items():
        cleaned = re.sub(re.escape(wrong), right, cleaned, flags=re.IGNORECASE)

    # Expand abbreviations
    for pattern, expansion in ABBREVIATIONS.items():
        cleaned = re.sub(pattern, expansion, cleaned, flags=re.IGNORECASE)

    # Re-run de-duplication after replacements (e.g. "back up back up" -> "backup")
    cleaned = re.sub(r"\b(\w+)(\s+\1){1,}\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Capitalize first letter of each sentence
    cleaned = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), cleaned)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Add period at end if missing punctuation
    cleaned = cleaned.strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned


def cleanup_transcript(raw_text: str) -> Tuple[str, dict]:
    """Clean up a raw Whisper transcript.

    Returns:
        (cleaned_text, metadata_dict)
    """
    if not raw_text.strip():
        return raw_text, {"cleanup_method": "skip", "cleanup_latency_ms": 0.0}

    from pipeline import llm

    t0 = time.time()

    if llm.is_available():
        result = llm.chat(_SYSTEM_PROMPT, raw_text, max_tokens=len(raw_text) * 2 + 128)
        if result:
            cleaned = result
            method = "llama3_8b_instruct"
        else:
            cleaned = _rule_based_cleanup(raw_text)
            method = "rules_fallback"
    else:
        cleaned = _rule_based_cleanup(raw_text)
        method = "rules"

    latency_ms = (time.time() - t0) * 1000

    meta = {
        "cleanup_method": method,
        "cleanup_latency_ms": round(latency_ms, 1),
    }

    return cleaned, meta
