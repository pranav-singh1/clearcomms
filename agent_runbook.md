# Agent Runbook: ClearComms

This file is for AI agents and teammates who need exact commands to set up, verify, and run the project.

## 0. Repo root

```bash
cd /Users/pranavsingh/clearcomms
```

## 1. Python environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

## 2. Install dependencies

Base dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional LLM dependencies (for GGUF LLaMA cleanup/extract):

```bash
python -m pip install -r requirements-llm.txt
```

## 3. Run setup doctor (required before debugging)

```bash
python tools/snapdragon_doctor.py
```

Interpretation:
- `FAIL`: must fix before expecting pipeline to run correctly.
- `WARN`: fallback may still work, but target backend may not.

## 4. Model files expected

Whisper QNN files (from Qualcomm export):
- `models/WhisperEncoder.onnx`
- `models/WhisperDecoder.onnx`

Optional LLM file:
- `models/llama-3-8b-instruct.gguf`

## 5. Backend selection controls

### Auto mode (default)

```bash
unset CLEARCOMMS_ASR_BACKEND 2>/dev/null || true
python -m streamlit run app/app.py
```

### Force QNN/NPU mode (fail fast if broken)

macOS/Linux:

```bash
export CLEARCOMMS_ASR_BACKEND=qnn
export QNN_BACKEND_PATH=/absolute/path/to/libQnnHtp.so
python -m streamlit run app/app.py
```

Windows PowerShell:

```powershell
$env:CLEARCOMMS_ASR_BACKEND = "qnn"
$env:QNN_BACKEND_PATH = "C:\absolute\path\to\QnnHtp.dll"
python -m streamlit run app/app.py
```

### Force ORT CPU fallback

```bash
export CLEARCOMMS_ASR_BACKEND=ort
python -m streamlit run app/app.py
```

## 6. Streamlit run command (standard)

```bash
python -m streamlit run app/app.py
```

In UI:
- `UI demo mode (no Whisper required)` lets you test UI without ASR models.
- Performance JSON shows `backend`, `requested_backend`, `qnn_status`, `qnn_error`.

## 7. Tests

Run cleanup/extract regressions:

```bash
python tests/test_cleanup.py
python tests/test_extract.py
```

Run ASR smoke pipeline:

```bash
python tests/test_asr.py
```

## 8. Live transcriber (mic)

```bash
python model/live_transcriber.py
```

## 9. Troubleshooting quick commands

Check active interpreter:

```bash
which python
python -c "import sys; print(sys.executable)"
python -m pip -V
```

Check ONNX Runtime providers:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

Check Streamlit import:

```bash
python - <<'PY'
import streamlit
print(streamlit.__version__)
PY
```

## 10. One-pass bootstrap (agent-friendly)

```bash
cd /Users/pranavsingh/clearcomms && \
python -m pip install -r requirements.txt && \
python tools/snapdragon_doctor.py
```
