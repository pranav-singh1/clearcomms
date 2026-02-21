# Snapdragon Setup Guide (ClearComms)

Goal: make code/setup predictable so the main blocker is only model downloads.

## 1. Create a clean Python environment

On the Qualcomm laptop, from repo root (`/Users/pranavsingh/clearcomms` equivalent path on device):

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
```

## 2. Install project dependencies

Base app + ASR/UI:

```bash
python -m pip install -r requirements.txt
```

Optional LLM cleanup/extraction:

```bash
python -m pip install -r requirements-llm.txt
```

If `llama-cpp-python` fails to build, you can still run the full app with rule-based cleanup/extraction.

## 3. Install Snapdragon/QNN runtime dependencies

You need an ONNX Runtime build that includes `QNNExecutionProvider` and Qualcomm runtime libraries.

Typical source: Qualcomm AI Hub / AI Lab model export package for your target laptop.

After installing runtime files, make sure:

- `QnnHtp.dll` is on `PATH`, or
- `QNN_BACKEND_PATH` points to the exact DLL file.

Example (PowerShell):

```powershell
$env:QNN_BACKEND_PATH = "C:\path\to\QnnHtp.dll"
```

## 4. Place model files

Whisper QNN export files expected by `model/config.yaml`:

- `models/WhisperEncoder.onnx`
- `models/WhisperDecoder.onnx`

Optional LLM model:

- `models/llama-3-8b-instruct.gguf`

You can override paths with config or env vars:

- `LLM_MODEL_PATH` for GGUF
- `QNN_BACKEND_PATH` for runtime DLL

## 5. Run environment doctor

```bash
python tools/snapdragon_doctor.py
```

This checks:

- installed Python modules
- ONNX Runtime providers (`QNNExecutionProvider`)
- model files from config
- QNN backend DLL visibility
- optional LLM backend readiness

## 6. Run app

```bash
python -m streamlit run app/app.py
```

Optional strict QNN mode (fail fast instead of CPU fallback):

```bash
set CLEARCOMMS_ASR_BACKEND=qnn
python -m streamlit run app/app.py
```

## 7. C++ / low-level build issues (common)

If teammates hit low-level build errors:

1. `llama-cpp-python` compile errors:
   - install C++ build tools (Visual Studio Build Tools with C++ workload)
   - or skip `requirements-llm.txt` and use rule fallback

2. `onnxruntime` or provider mismatch:
   - confirm doctor output lists `QNNExecutionProvider`
   - if missing, install the Qualcomm-provided ORT/QNN package for your device

3. `QnnHtp.dll` not found:
   - set `QNN_BACKEND_PATH` to absolute DLL path
   - or add containing folder to `PATH`

## 8. What success looks like

In the Streamlit app performance metadata:

- `"backend": "qnn"` means Snapdragon QNN/NPU path is active.
- `"backend": "ort"` means CPU fallback is active.
- If fallback happened, app now reports `qnn_status` and `qnn_error` for debugging.
