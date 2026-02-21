"""Snapdragon setup doctor for ClearComms.

Run:
    python tools/snapdragon_doctor.py
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    status: str
    name: str
    detail: str


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _resolve_model_path(project_root: Path, path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _print_results(results: list[CheckResult]) -> int:
    status_icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    fail_count = 0
    warn_count = 0
    for r in results:
        print(f"{status_icon[r.status]} {r.name}: {r.detail}")
        if r.status == "FAIL":
            fail_count += 1
        elif r.status == "WARN":
            warn_count += 1

    print()
    print(f"Summary: {len(results)} checks, {fail_count} fail, {warn_count} warn")
    return 1 if fail_count else 0


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "model" / "config.yaml"
    results: list[CheckResult] = []

    results.append(
        CheckResult(
            "PASS",
            "Python",
            f"{sys.executable} ({sys.version.split()[0]})",
        )
    )
    results.append(
        CheckResult(
            "PASS",
            "Platform",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        )
    )

    required_modules = [
        "numpy",
        "scipy",
        "soundfile",
        "yaml",
        "streamlit",
        "transformers",
        "optimum",
    ]
    for module_name in required_modules:
        if _has_module(module_name):
            results.append(CheckResult("PASS", f"Module:{module_name}", "installed"))
        else:
            results.append(CheckResult("FAIL", f"Module:{module_name}", "missing"))

    # onnxruntime is required for ASR path.
    if _has_module("onnxruntime"):
        results.append(CheckResult("PASS", "Module:onnxruntime", "installed"))
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        if "QNNExecutionProvider" in providers:
            results.append(
                CheckResult("PASS", "ONNX Runtime QNN EP", f"available providers: {providers}")
            )
        else:
            results.append(
                CheckResult(
                    "WARN",
                    "ONNX Runtime QNN EP",
                    f"missing QNNExecutionProvider; current providers: {providers}",
                )
            )
    else:
        results.append(CheckResult("FAIL", "Module:onnxruntime", "missing"))
        providers = []

    if _has_module("qai_hub_models"):
        results.append(CheckResult("PASS", "Module:qai_hub_models", "installed"))
    else:
        results.append(
            CheckResult(
                "WARN",
                "Module:qai_hub_models",
                "missing; required for QNN Whisper wrapper in model/model.py",
            )
        )

    if not _has_module("yaml"):
        results.append(CheckResult("FAIL", "Config parser", "PyYAML missing"))
        return _print_results(results)

    import yaml

    if not cfg_path.exists():
        results.append(CheckResult("FAIL", "Config", f"missing {cfg_path}"))
        return _print_results(results)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    variant = str(cfg.get("model_variant", "base_en"))
    supported_variants = {"base_en", "small_en", "medium_en", "large_en", "large_v3_turbo"}
    if variant in supported_variants:
        results.append(CheckResult("PASS", "Whisper variant", variant))
    else:
        results.append(
            CheckResult(
                "FAIL",
                "Whisper variant",
                f"{variant} unsupported. Choose one of {sorted(supported_variants)}",
            )
        )

    encoder_path = _resolve_model_path(project_root, cfg.get("encoder_path", "models/WhisperEncoder.onnx"))
    decoder_path = _resolve_model_path(project_root, cfg.get("decoder_path", "models/WhisperDecoder.onnx"))
    if encoder_path.exists():
        results.append(CheckResult("PASS", "QNN encoder file", str(encoder_path)))
    else:
        results.append(
            CheckResult(
                "WARN",
                "QNN encoder file",
                f"missing {encoder_path} (download from AI Hub/AI Lab export)",
            )
        )
    if decoder_path.exists():
        results.append(CheckResult("PASS", "QNN decoder file", str(decoder_path)))
    else:
        results.append(
            CheckResult(
                "WARN",
                "QNN decoder file",
                f"missing {decoder_path} (download from AI Hub/AI Lab export)",
            )
        )

    qnn_backend = os.environ.get("QNN_BACKEND_PATH", cfg.get("qnn_backend_path", "QnnHtp.dll"))
    qnn_backend_path = Path(qnn_backend)
    if qnn_backend_path.is_absolute():
        if qnn_backend_path.exists():
            results.append(CheckResult("PASS", "QNN backend library", str(qnn_backend_path)))
        else:
            results.append(CheckResult("WARN", "QNN backend library", f"missing {qnn_backend_path}"))
    else:
        found = shutil.which(qnn_backend)
        if found:
            results.append(CheckResult("PASS", "QNN backend library", f"found on PATH: {found}"))
        else:
            results.append(
                CheckResult(
                    "WARN",
                    "QNN backend library",
                    f"{qnn_backend} not found on PATH (set QNN_BACKEND_PATH or update PATH)",
                )
            )

    # LLM is optional; only warn if model file exists but runtime missing.
    llm_model_path = Path(os.environ.get("LLM_MODEL_PATH", str(project_root / "models" / "llama-3-8b-instruct.gguf")))
    llm_module = _has_module("llama_cpp")
    if llm_model_path.exists() and llm_module:
        results.append(CheckResult("PASS", "LLM backend", f"{llm_model_path} + llama_cpp available"))
    elif llm_model_path.exists() and not llm_module:
        results.append(CheckResult("WARN", "LLM backend", "GGUF present but llama_cpp not installed"))
    elif not llm_model_path.exists():
        results.append(CheckResult("WARN", "LLM backend", f"model not found at {llm_model_path} (rules fallback works)"))

    requested_backend = os.environ.get("CLEARCOMMS_ASR_BACKEND", "auto").strip().lower()
    if requested_backend == "qnn" and "QNNExecutionProvider" not in providers:
        results.append(
            CheckResult(
                "FAIL",
                "Requested backend",
                "CLEARCOMMS_ASR_BACKEND=qnn but QNNExecutionProvider is unavailable",
            )
        )

    exit_code = _print_results(results)
    if exit_code:
        print("\nRecommended next step: fix FAIL items, then rerun doctor.")
    else:
        print("\nDoctor checks passed. If models are present, run:")
        print("  python -m streamlit run app/app.py")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
