"""Voice imitation backends."""

from . import dspvc as _dspvc  # noqa: F401  (registers "dspvc")
from . import neural as _neural  # noqa: F401  (registers "xtts", "yourtts")
from .base import SynthResult, VoiceSynthesizer, available_synths, get_synth, register_synth

__all__ = [
    "SynthResult",
    "VoiceSynthesizer",
    "available_synths",
    "get_synth",
    "register_synth",
]
