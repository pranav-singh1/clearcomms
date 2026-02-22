"""
On-device Llama revision via Qualcomm Genie (genie-t2t-run.exe).
Optional post-processing for Whisper transcripts: clean noisy radio/dispatch text.
"""

from llama_on_device.genie_llama import revise_transcript
from llama_on_device.prompts import build_revision_prompt

__all__ = ["revise_transcript", "build_revision_prompt"]
