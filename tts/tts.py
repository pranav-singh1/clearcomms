"""Deepgram text-to-speech utility for ClearComms transcripts.

This script is built for the "patched" transcript flow:
  Whisper raw text -> cleanup_transcript (LLaMA/rules) -> Deepgram TTS audio.

Examples:
  python tts/tts.py --full-report runs/full_report.json --output runs/tts_cleaned.mp3
  python tts/tts.py --text "Engine 12 respond to Maple Street." --output runs/tts.mp3
  python tts/tts.py --transcript-file runs/raw_transcript.txt --patch-with-llama
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Tuple


DEFAULT_API_BASE = "https://api.deepgram.com/v1/speak"
DEFAULT_MODEL = "aura-2-thalia-en"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow running as: python /absolute/path/to/tts/tts.py
# so imports like "from pipeline.cleanup import ..." still resolve.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_text_inputs(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve text input source from CLI args."""
    if args.text:
        text = args.text.strip()
        if not text:
            raise ValueError("--text provided but empty.")
        return text, "text"

    if args.full_report:
        report_path = Path(args.full_report)
        if not report_path.exists():
            raise FileNotFoundError(f"full_report not found: {report_path}")

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        cleaned = str(payload.get("cleaned_transcript", "")).strip()
        raw = str(payload.get("raw_transcript", "")).strip()

        if cleaned:
            return cleaned, "full_report.cleaned_transcript"
        if raw:
            return raw, "full_report.raw_transcript"
        raise ValueError(f"No transcript fields found in {report_path}")

    if args.transcript_file:
        transcript_path = Path(args.transcript_file)
        if not transcript_path.exists():
            raise FileNotFoundError(f"transcript file not found: {transcript_path}")

        text = transcript_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Transcript file is empty: {transcript_path}")
        return text, "transcript_file"

    raise ValueError("Provide one input source: --text, --full-report, or --transcript-file.")


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    """Minimal .env loader (no external dependency)."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        if not key:
            continue

        # Preserve already-exported shell env values.
        os.environ.setdefault(key, value)


def _infer_encoding(output_path: Path, explicit_encoding: str | None) -> str:
    if explicit_encoding:
        return explicit_encoding.strip().lower()

    ext = output_path.suffix.lower()
    if ext == ".wav":
        return "linear16"
    if ext == ".opus":
        return "opus"
    if ext == ".flac":
        return "flac"
    return "mp3"


def _maybe_patch_with_llama(text: str, enable_patch: bool) -> Tuple[str, str]:
    if not enable_patch:
        return text, "skip"

    # Reuse existing cleanup layer. If LLaMA is loaded, it uses LLaMA;
    # otherwise it falls back to the rule-based cleanup.
    from pipeline.cleanup import cleanup_transcript

    cleaned, meta = cleanup_transcript(text)
    method = str(meta.get("cleanup_method", "unknown"))
    return cleaned, method


def _deepgram_tts(
    text: str,
    output_path: Path,
    api_key: str,
    model: str,
    encoding: str,
    sample_rate: int | None = None,
    container: str | None = None,
    api_base: str = DEFAULT_API_BASE,
) -> None:
    query = {
        "model": model,
        "encoding": encoding,
    }
    if sample_rate is not None:
        query["sample_rate"] = str(sample_rate)
    if container:
        query["container"] = container

    url = f"{api_base}?{urllib.parse.urlencode(query)}"
    body = json.dumps({"text": text}).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Deepgram TTS request failed: HTTP {exc.code} - {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Deepgram TTS request failed: {exc.reason}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


def main() -> int:
    _load_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Generate speech audio from ClearComms transcript text using Deepgram TTS."
    )
    parser.add_argument("--text", help="Direct input text to synthesize.")
    parser.add_argument("--transcript-file", help="Path to a text file containing transcript text.")
    parser.add_argument(
        "--full-report",
        help="Path to exported full_report.json from pipeline/app.py (uses cleaned_transcript when present).",
    )
    parser.add_argument(
        "--patch-with-llama",
        action="store_true",
        help="Patch/cleanup the input text via pipeline.cleanup before TTS.",
    )
    parser.add_argument(
        "--output",
        default="runs/tts_output.mp3",
        help="Output audio file path (default: runs/tts_output.mp3).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPGRAM_TTS_MODEL", DEFAULT_MODEL),
        help=f"Deepgram TTS model (default: {DEFAULT_MODEL} or DEEPGRAM_TTS_MODEL env).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Audio encoding, e.g. mp3/linear16/opus/flac. If omitted, inferred from output extension.",
    )
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional output sample rate.")
    parser.add_argument("--container", default=None, help="Optional container, e.g. wav.")
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Deepgram TTS endpoint base (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPGRAM_API_KEY",
        help="Environment variable name that stores the Deepgram API key.",
    )

    args = parser.parse_args()
    output_path = Path(args.output).resolve()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"Missing API key. Set {args.api_key_env} in your environment.", file=sys.stderr)
        return 2

    try:
        input_text, source = _read_text_inputs(args)
        final_text, patch_method = _maybe_patch_with_llama(input_text, args.patch_with_llama)
        encoding = _infer_encoding(output_path, args.encoding)

        # Common convenience: if user wants a .wav file and didn't set container, set wav container.
        container = args.container
        if container is None and output_path.suffix.lower() == ".wav" and encoding == "linear16":
            container = "wav"

        _deepgram_tts(
            text=final_text,
            output_path=output_path,
            api_key=api_key,
            model=args.model,
            encoding=encoding,
            sample_rate=args.sample_rate,
            container=container,
            api_base=args.api_base,
        )
    except Exception as exc:
        print(f"TTS failed: {exc}", file=sys.stderr)
        return 1

    print(f"[TTS] Source: {source}")
    print(f"[TTS] Patch method: {patch_method}")
    print(f"[TTS] Model: {args.model}")
    print(f"[TTS] Output: {output_path}")
    print("[TTS] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
