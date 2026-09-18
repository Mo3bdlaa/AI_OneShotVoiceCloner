"""Voice-imitation backend interface.

Two genuinely different capabilities hide behind the word "cloning":

``tts``
    text -> speech in a target voice. Needs a generative model trained on
    thousands of hours; there is no way to approximate it with signal processing.
``vc``
    speech -> the same speech in a target voice. A recording already supplies the
    words, the timing and the prosody, so only the *timbre* has to be changed --
    which a signal-processing method can attempt.

Backends declare which of the two they support, so the CLI can fail with a clear
message instead of a stack trace when a user asks a conversion-only backend to
read a sentence out loud.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..audio import TARGET_SR


@dataclass(frozen=True)
class SynthResult:
    """Generated audio plus whatever the backend wants to report about it."""

    wav: np.ndarray
    sample_rate: int
    backend: str
    info: dict

    @property
    def duration(self) -> float:
        return float(self.wav.size) / float(self.sample_rate)


class VoiceSynthesizer:
    """Base class for text-to-speech and voice-conversion backends.

    Deliberately not an abstract base class: a backend implements ``synthesize``,
    ``convert``, or both, and declares which through :attr:`capabilities`. The
    default implementations raise a message naming what the backend *can* do,
    which is more useful than an instantiation error.
    """

    name: str = "base"
    version: str = "0"
    capabilities: frozenset[str] = frozenset()
    sample_rate: int = TARGET_SR
    #: Free-text note surfaced by ``voxprint backends`` so users can judge
    #: quality and licensing before downloading a multi-gigabyte model.
    notes: str = ""

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def synthesize(
        self,
        text: str,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        *,
        language: str = "en",
        **kwargs,
    ) -> SynthResult:
        """Speak ``text`` in the voice of ``reference``."""
        raise NotImplementedError(
            f"backend {self.name!r} cannot do text-to-speech; it supports: "
            f"{', '.join(sorted(self.capabilities)) or 'nothing'}"
        )

    def convert(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        **kwargs,
    ) -> SynthResult:
        """Re-voice the speech in ``source`` to sound like ``reference``."""
        raise NotImplementedError(
            f"backend {self.name!r} cannot do voice conversion; it supports: "
            f"{', '.join(sorted(self.capabilities)) or 'nothing'}"
        )

    def available(self) -> tuple[bool, str]:
        """``(ready, reason)`` -- whether the backend can run right now."""
        return True, "ready"

    def describe(self) -> dict:
        ready, reason = self.available()
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": sorted(self.capabilities),
            "sample_rate": self.sample_rate,
            "ready": ready,
            "status": reason,
            "notes": self.notes,
        }


_REGISTRY: dict[str, Callable[..., VoiceSynthesizer]] = {}


def register_synth(name: str) -> Callable[[Callable[..., VoiceSynthesizer]], Callable[..., VoiceSynthesizer]]:
    def decorator(factory):
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_synths() -> list[str]:
    return sorted(_REGISTRY)


def get_synth(name: str = "dspvc", **kwargs) -> VoiceSynthesizer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown synthesis backend {name!r}; available: {', '.join(available_synths())}")
    return _REGISTRY[name](**kwargs)
