# app/app.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.app import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app()
