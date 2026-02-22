"""
Llama 3 formatted prompts for transcript analysis (action items + optional reconstruction).
"""


def build_revision_prompt(transcript: str) -> str:
    """
    Build a Llama 3 chat-formatted prompt for extracting action items and suggested
    actions from a radio/dispatch transcript. If the transcript seems like nonsense,
    suggest reconstruction instead.

    Args:
        transcript: Raw transcript string from ASR.

    Returns:
        Full prompt string in Llama 3 format.
    """
    system = (
        "You are an AI assistant for first responders. Your ONLY job is to rewrite a noisy "
        "radio/dispatch transcript into a clean, readable transcript.\n\n"

        "Strict output rules (must follow):\n"
        "- Output ONLY the cleaned transcript text.\n"
        "- Do NOT include analysis, summaries, action items, recommendations, or meta commentary.\n"
        "- Do NOT use headings (e.g., 'Transcript Analysis', 'Action Items').\n"
        "- Do NOT use bullet points or lists.\n"
        "- Do NOT add any extra text before or after the transcript.\n\n"

        "Editing rules:\n"
        "- Preserve meaning; do NOT invent new facts.\n"
        "- Fix punctuation, casing, and obvious ASR errors.\n"
        "- Keep numbers, unit IDs/callsigns, addresses, mile markers, and locations exactly unless clearly wrong.\n"
        "- If words are cut off but strongly inferable in first-responder context, reconstruct them and wrap them as "
        "[predicted: ...].\n"
        "- If something is not confidently inferable, keep it but mark as [unclear: ...].\n\n"

        "If the transcript is mostly unintelligible/corrupted:\n"
        "- Still output ONLY a best-effort cleaned transcript.\n"
        "- Use [unclear: ...] for large unclear spans.\n"
        "- Do NOT say 'unusable' or suggest actions; no commentary.\n"
    )
    # Llama 3 chat format matching Genie: begin_of_text, then system/user/assistant
    parts = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>",
        "",
        system,
        "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "",
        transcript.strip(),
        "<|eot_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "",
    ]
    return "\n".join(parts)
