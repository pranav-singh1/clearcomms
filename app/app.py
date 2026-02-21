"""app/app.py

Streamlit entrypoint for ClearComms.

Run from repo root:
    streamlit run app/app.py
"""

from pipeline.app import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app()
