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
        "You are an AI assistant for first responders. You analyze radio, dispatch, and "
        "emergency communications and output concise, actionable summaries.\n\n"

        "Your task:\n"
        "1. Extract action items from the transcript (who, what, where, when, priority).\n"
        "2. List suggested next actions (e.g. dispatch unit, request backup, confirm location).\n"
        "3. Be concise: use short bullets or a few lines; no long prose.\n\n"

        "If the transcript is mostly nonsense (heavy static, unintelligible, or clearly "
        "corrupted), do NOT invent content. Instead output briefly that the transcript "
        "appears unusable and suggest reconstruction or a clearer audio source.\n\n"

        "Otherwise preserve key details (callsigns, locations, numbers) and do not invent "
        "facts. Output only your analysis: action items, suggested actions, and do "
        "not do any special formatting. Just the text."
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
