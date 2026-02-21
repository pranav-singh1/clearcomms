"""Regression tests for structured incident extraction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import llm
from pipeline.extract import extract_incident


def _force_rules_backend() -> None:
    os.environ["LLM_MODEL_PATH"] = "/tmp/clearcomms-no-llm.gguf"
    llm._llm = None
    llm._llm_error = None


def test_non_emergency_is_low_urgency() -> None:
    _force_rules_backend()
    incident, _ = extract_incident("non-emergency standby at 100 Main Street")
    assert incident["urgency"] == "low"


def test_engine_number_not_parsed_as_address_prefix() -> None:
    _force_rules_backend()
    incident, _ = extract_incident("Engine 12 to dispatch smoke near Maple Street requesting backup")
    assert not incident["location"].lower().startswith("12 to dispatch")
    assert incident["location"].lower() in {"maple street", "unknown"}


def test_cross_street_location_extraction() -> None:
    _force_rules_backend()
    incident, _ = extract_incident("Unit 4 staged at Pine and 3rd")
    assert incident["location"].lower() == "pine and 3rd"


def test_llm_json_is_sanitized() -> None:
    original_is_available = llm.is_available
    original_chat_json = llm.chat_json

    try:
        llm.is_available = lambda: True
        llm.chat_json = lambda *_args, **_kwargs: {
            "request_type": "alien",   # invalid -> coerce to "other"
            "urgency": "panic",        # invalid -> keep default "medium"
            "location": 123,           # invalid -> keep default "unknown"
            "hazards": "smoke, fire",
            "units": ["Engine 12", "Engine 12", None],
            "actions": "respond, backup",
        }

        incident, meta = extract_incident("Engine 12 respond to 235 Maple Street")

        assert incident["request_type"] == "other"
        assert incident["urgency"] == "medium"
        assert incident["location"] == "unknown"
        assert incident["hazards"] == ["smoke", "fire"]
        assert incident["units"] == ["Engine 12"]
        assert incident["actions"] == ["respond", "backup"]
        assert meta["extract_method"] == "llama3_8b_instruct"
    finally:
        llm.is_available = original_is_available
        llm.chat_json = original_chat_json


if __name__ == "__main__":
    test_non_emergency_is_low_urgency()
    test_engine_number_not_parsed_as_address_prefix()
    test_cross_street_location_extraction()
    test_llm_json_is_sanitized()
    print("tests/test_extract.py: ALL TESTS PASSED")
