"""Regression tests for transcript cleanup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import llm
from pipeline.cleanup import cleanup_transcript


def _force_rules_backend() -> None:
    os.environ["LLM_MODEL_PATH"] = "/tmp/clearcomms-no-llm.gguf"
    llm._llm = None
    llm._llm_error = None


def test_rule_cleanup_basics() -> None:
    _force_rules_backend()
    raw = "eng 12 to dispatch smoke near mapple st need back up go go go"
    cleaned, meta = cleanup_transcript(raw)

    assert meta["cleanup_method"].startswith("rules")
    assert "engine 12" in cleaned.lower()
    assert "maple street" in cleaned.lower()
    assert "back up" not in cleaned.lower()
    assert "go go" not in cleaned.lower()
    assert cleaned.endswith(".")


def test_cleanup_empty_input() -> None:
    _force_rules_backend()
    cleaned, meta = cleanup_transcript("")

    assert cleaned == ""
    assert meta["cleanup_method"] == "skip"
    assert meta["cleanup_latency_ms"] == 0.0


if __name__ == "__main__":
    test_rule_cleanup_basics()
    test_cleanup_empty_input()
    print("tests/test_cleanup.py: ALL TESTS PASSED")
