# On-Device Llama Revision (Qualcomm Genie)

Optional post-processing for Whisper transcripts: clean noisy radio/dispatch text using on-device Llama via **Qualcomm Genie** (`genie-t2t-run.exe`). No separate Python venv is required for Llama—Genie is a native executable; Python only shells out.

## Setup

1. **Dot-source the Genie env script** (PowerShell) so `genie-t2t-run.exe` is on PATH and QAIRT/ADSP vars are set:
   ```powershell
   . .\scripts\setup_genie_env.ps1
   ```
   You must **dot-source** (`. .\path\to\script.ps1`) so the environment variables apply to the current session.

2. **Set the Genie bundle directory** (folder containing `genie_config.json`):
   ```powershell
   $env:GENIE_BUNDLE_DIR = "C:\path\to\your\llama\bundle"
   ```

## Test

```powershell
python scripts/run_llama_revision.py --text "bravo two copy that we are oscar mike"
```

Or from a file:

```powershell
python scripts/run_llama_revision.py --in transcript.txt
```

The revised transcript is printed to stdout. If `GENIE_BUNDLE_DIR` is missing or Genie fails, the script exits nonzero with a helpful message.

## Enable in the pipeline

Set the environment variable before starting the backend:

```powershell
$env:ENABLE_LLAMA_REVISION = "1"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

When `ENABLE_LLAMA_REVISION=1`, after Whisper produces a transcript the backend optionally calls Genie to revise it and returns the revised text. If revision fails, the original Whisper transcript is kept and an error is recorded in `meta.llama_revision_error`.

## Optional env vars

| Variable | Description |
|----------|-------------|
| `GENIE_BUNDLE_DIR` | **Required.** Path to folder containing `genie_config.json`. |
| `GENIE_CONFIG` | Config file path; default `${GENIE_BUNDLE_DIR}/genie_config.json`. |
| `GENIE_EXE` | Executable name or path; default `genie-t2t-run.exe`. |
| `GENIE_TIMEOUT_S` | Subprocess timeout in seconds; default `60`. |
