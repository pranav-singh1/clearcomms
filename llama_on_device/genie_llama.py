"""
Run on-device Llama via Qualcomm Genie (genie-t2t-run.exe) to revise transcripts.
Uses a prompt file to avoid quoting issues; no new dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from llama_on_device.prompts import build_revision_prompt


def revise_transcript(transcript: str) -> str:
    """
    Revise a transcript using on-device Llama (Genie). Calls genie-t2t-run.exe
    as a subprocess with the prompt written to a file in the Genie bundle dir.

    Configuration via environment:
        GENIE_BUNDLE_DIR (required): path to folder containing genie_config.json
        GENIE_CONFIG (optional): default ${GENIE_BUNDLE_DIR}/genie_config.json
        GENIE_EXE (optional): default genie-t2t-run.exe
        GENIE_TIMEOUT_S (optional): default 60

    The subprocess is run with: genie-t2t-run.exe -c <config> -p "<prompt>"
    (cwd=GENIE_BUNDLE_DIR). The prompt is also written to prompt.txt for
    debugging. Output is parsed for text between [BEGIN]: and [END].

    Args:
        transcript: Raw transcript string from Whisper.

    Returns:
        Revised transcript string.

    Raises:
        ValueError: If GENIE_BUNDLE_DIR is missing or invalid.
        RuntimeError: If Genie fails or output cannot be parsed (includes
            last ~2000 chars of combined stdout+stderr).
    """
    bundle_dir = os.getenv("GENIE_BUNDLE_DIR", "").strip()
    if not bundle_dir:
        raise ValueError(
            "GENIE_BUNDLE_DIR is not set. Set it to the path of the folder "
            "containing genie_config.json (e.g. your Llama Genie bundle directory). "
            "Example (PowerShell): $env:GENIE_BUNDLE_DIR = 'C:\\path\\to\\bundle'"
        )
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_dir():
        raise ValueError(
            f"GENIE_BUNDLE_DIR is not a directory: {bundle_dir}. "
            "Set GENIE_BUNDLE_DIR to the folder containing genie_config.json."
        )

    config_path = Path(os.getenv("GENIE_CONFIG", "").strip() or str(bundle_path / "genie_config.json"))
    if not config_path.is_file():
        raise ValueError(
            f"Genie config file not found: {config_path}. "
            "Ensure GENIE_BUNDLE_DIR points to a folder with genie_config.json, "
            "or set GENIE_CONFIG to the correct config path."
        )

    prompt_text = build_revision_prompt(transcript)
    (bundle_path / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    # Use config basename when config is inside bundle dir so Genie sees a relative path from cwd
    config_arg = config_path.name if config_path.resolve().parent == bundle_path.resolve() else str(config_path)
    exe = os.getenv("GENIE_EXE", "").strip() or "genie-t2t-run.exe"
    timeout_s = int(os.getenv("GENIE_TIMEOUT_S", "60").strip() or "60")
    if timeout_s <= 0:
        timeout_s = 60

    cmd: list[str] = [exe, "-c", config_arg, "-p", prompt_text]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(bundle_path),
            capture_output=True,
            timeout=timeout_s,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Genie executable not found: {exe}. "
            "Dot-source the env script so genie-t2t-run.exe is on PATH: "
            ". .\\scripts\\setup_genie_env.ps1"
        ) from e
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        tail = out[-2000:] if len(out) > 2000 else out
        raise RuntimeError(
            f"Genie timed out after {timeout_s}s. Last 2000 chars of output:\n{tail}"
        ) from e

    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        tail = combined[-2000:] if len(combined) > 2000 else combined
        raise RuntimeError(
            f"Genie exited with code {result.returncode}. Last 2000 chars:\n{tail}"
        )

    # Parse text between [BEGIN]: and [END]
    begin_m = re.search(r"\[BEGIN\]:\s*", combined, re.IGNORECASE | re.DOTALL)
    end_m = re.search(r"\s*\[END\]", combined, re.IGNORECASE | re.DOTALL)
    if begin_m is not None and end_m is not None and end_m.end() > begin_m.end():
        revised = combined[begin_m.end() : end_m.start()].strip()
        return revised

    tail = combined[-2000:] if len(combined) > 2000 else combined
    raise RuntimeError(
        "Could not parse Genie output: expected [BEGIN]: ... [END]. "
        f"Last 2000 chars of output:\n{tail}"
    )
