"""Speaker encoders (voice-print extractors)."""

from . import dsp as _dsp  # noqa: F401  (registers "dsp")
from . import ecapa as _ecapa  # noqa: F401  (registers "ecapa", imported lazily inside)
from .base import (
    EncoderSpec,
    SpeakerEncoder,
    available_encoders,
    average_embeddings,
    get_encoder,
    l2_normalize,
    register_encoder,
)

__all__ = [
    "EncoderSpec",
    "SpeakerEncoder",
    "average_embeddings",
    "available_encoders",
    "get_encoder",
    "l2_normalize",
    "register_encoder",
]
