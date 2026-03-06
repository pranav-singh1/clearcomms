"""
Revise transcripts using Google Gemini API.
Replaces the previous Qualcomm Genie (on-device Llama) subprocess approach.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from llama_on_device.prompts import build_revision_prompt


def revise_transcript(transcript: str) -> str:
    """
    Revise a transcript using Google Gemini API.

    Configuration via environment:
        GEMINI_API_KEY (required): Google Gemini API key
        GEMINI_MODEL (optional): model name, default "gemini-2.0-flash"

    Args:
        transcript: Raw transcript string from Whisper.

    Returns:
        Revised transcript string.

    Raises:
        ValueError: If GEMINI_API_KEY is missing.
        RuntimeError: If Gemini API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Set it to your Google Gemini API key. "
            "Example: export GEMINI_API_KEY='your-api-key-here'"
        )

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.0-flash"
    system_prompt, user_prompt = build_revision_prompt(transcript)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        revised = response.text.strip()
        if not revised:
            raise RuntimeError("Gemini returned an empty response.")
        return revised
    except Exception as e:
        if "API key" in str(e) or "api_key" in str(e):
            raise ValueError(str(e)) from e
        raise RuntimeError(f"Gemini API call failed: {e}") from e
