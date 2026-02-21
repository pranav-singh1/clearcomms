# Streamlit Run Guide (ClearComms)

Use this to install and run the UI at `/Users/pranavsingh/clearcomms/app/app.py`.

## 1. Go to project root

```bash
cd /Users/pranavsingh/clearcomms
```

## 2. Install dependencies

Choose the Python you want to run with. Install into that same interpreter.

### Option A: Conda Python (`/opt/miniconda3/bin/python3`)

```bash
/opt/miniconda3/bin/python3 -m pip install -r requirements.txt
/opt/miniconda3/bin/python3 -m pip install streamlit
```

### Option B: System Python (example: `/Library/Frameworks/.../python3`)

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip install -r requirements.txt
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip install streamlit
```

## 3. Run Streamlit

Again, use the same interpreter you installed into.

### Conda run

```bash
/opt/miniconda3/bin/python3 -m streamlit run /Users/pranavsingh/clearcomms/app/app.py
```

### System Python run

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m streamlit run /Users/pranavsingh/clearcomms/app/app.py
```

## 4. Test UI without Whisper/QNN

In the sidebar:

- Turn on `UI demo mode (no Whisper required)`.
- Keep `Fallback to demo transcript if ASR fails` enabled.

You can now upload audio and test the full UI even without Qualcomm Whisper setup.

## Troubleshooting

If you get `No module named streamlit`, you installed Streamlit into a different Python.

Check which interpreter is being used:

```bash
which python3
python3 -m pip -V
python3 -c "import sys; print(sys.executable)"
```

Verify Streamlit in that interpreter:

```bash
python3 -c "import streamlit; print(streamlit.__version__)"
```
