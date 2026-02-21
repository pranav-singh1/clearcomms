# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import onnxruntime
from qai_hub_models.models._shared.whisper.model import Whisper
from qai_hub_models.models.whisper_base_en import App as WhisperApp


def get_onnxruntime_session_with_qnn_ep(path, cfg):
    options = onnxruntime.SessionOptions()
    # Optional: enable profiling when you want evidence
    # options.enable_profiling = True

    provider_options = {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": cfg.get("htp_performance_mode", "burst"),
        "high_power_saver": "sustained_high_performance",
        "enable_htp_fp16_precision": cfg.get("enable_htp_fp16_precision", "1"),
        "htp_graph_finalization_optimization_mode": "3",
    }

    session = onnxruntime.InferenceSession(
        path,
        sess_options=options,
        providers=["QNNExecutionProvider"],
        provider_options=[provider_options],
    )
    return session


class ONNXEncoderWrapper:
    def __init__(self, encoder_path, cfg):
        self.session = get_onnxruntime_session_with_qnn_ep(encoder_path, cfg)

    def to(self, *args):
        return self

    def __call__(self, audio):
        return self.session.run(None, {"audio": audio})


class ONNXDecoderWrapper:
    def __init__(self, decoder_path, cfg):
        self.session = get_onnxruntime_session_with_qnn_ep(decoder_path, cfg)

    def to(self, *args):
        return self

    def __call__(self, x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self):
        return self.session.run(
            None,
            {
                "x": x.astype(np.int32),
                "index": np.array(index),
                "k_cache_cross": k_cache_cross,
                "v_cache_cross": v_cache_cross,
                "k_cache_self": k_cache_self,
                "v_cache_self": v_cache_self,
            },
        )


# Standard Whisper architecture configs (must match your ONNX export!)
WHISPER_ARCH = {
    "base_en":   {"num_decoder_blocks": 6,  "num_heads": 8,  "attention_dim": 512},
    "small_en":  {"num_decoder_blocks": 12, "num_heads": 12, "attention_dim": 768},
    "medium_en": {"num_decoder_blocks": 24, "num_heads": 16, "attention_dim": 1024},
    "large_en":  {"num_decoder_blocks": 32, "num_heads": 20, "attention_dim": 1280},
}


class WhisperONNX(Whisper):
    def __init__(self, encoder_path, decoder_path, variant: str, cfg):
        arch = WHISPER_ARCH.get(variant)
        if arch is None:
            raise ValueError(f"Unknown variant '{variant}'. Use one of: {list(WHISPER_ARCH.keys())}")

        super().__init__(
            ONNXEncoderWrapper(encoder_path, cfg),
            ONNXDecoderWrapper(decoder_path, cfg),
            num_decoder_blocks=arch["num_decoder_blocks"],
            num_heads=arch["num_heads"],
            attention_dim=arch["attention_dim"],
        )


def make_whisper_app(encoder_path: str, decoder_path: str, variant: str, cfg):
    # If you switch model variants, also switch to the correct WhisperApp class if Qualcomm provides it
    # For hackathon simplicity, WhisperApp usually still works as long as underlying Whisper model matches.
    model = WhisperONNX(encoder_path, decoder_path, variant, cfg)
    return WhisperApp(model)
