"""pipeline/llm.py — Shared on-device LLaMA 3 8B Instruct inference.

Uses llama-cpp-python to run a GGUF-quantized model locally.
Works on Mac (dev) and Qualcomm laptops (demo) with no internet.

Place your GGUF model file in: models/llama-3-8b-instruct.gguf
(or set LLM_MODEL_PATH env var to override)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "llama-3-8b-instruct.gguf"

_llm = None
_llm_error: Optional[Exception] = None


def _init_llm():
    global _llm, _llm_error
    if _llm is not None or _llm_error is not None:
        return

    model_path = os.environ.get("LLM_MODEL_PATH", str(_DEFAULT_MODEL_PATH))

    if not Path(model_path).exists():
        _llm_error = FileNotFoundError(
            f"LLM model not found at {model_path}. "
            "Download a GGUF quantized LLaMA 3 8B Instruct model and place it there."
        )
        return

    try:
        from llama_cpp import Llama

        print(f"[LLM] Loading {Path(model_path).name} ...")
        _llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("[LLM] Ready — LLaMA 3 8B Instruct loaded on-device")
    except Exception as exc:
        _llm_error = exc


def is_available() -> bool:
    _init_llm()
    return _llm is not None


def get_error() -> Optional[Exception]:
    _init_llm()
    return _llm_error


def chat(system_prompt: str, user_message: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Run a single chat turn with the loaded LLaMA model.

    Returns the assistant's response text, or empty string on failure.
    """
    _init_llm()
    if _llm is None:
        return ""

    response = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|eot_id|>"],
    )

    return response["choices"][0]["message"]["content"].strip()


def chat_json(system_prompt: str, user_message: str, max_tokens: int = 512) -> Optional[dict]:
    """Like chat(), but parses the response as JSON. Returns None on parse failure."""
    raw = chat(system_prompt, user_message, max_tokens=max_tokens, temperature=0.0)
    if not raw:
        return None

    # Extract JSON from response (model may wrap it in markdown fences)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None
