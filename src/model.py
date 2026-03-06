"""
Whisper model wrapper using openai-whisper (PyTorch).
Works on Mac (CPU/MPS) without Qualcomm QNN dependencies.
"""
import whisper


_model_cache: dict[str, whisper.Whisper] = {}


def get_whisper_model(variant: str = "base.en") -> whisper.Whisper:
    """Load (and cache) a Whisper model by variant name."""
    # Normalize variant names from config format
    variant = variant.replace("_", ".").replace("-", ".")
    if variant in _model_cache:
        return _model_cache[variant]
    model = whisper.load_model(variant)
    _model_cache[variant] = model
    return model
