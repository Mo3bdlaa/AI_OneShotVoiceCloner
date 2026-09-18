"""Speaker-encoder interface and registry.

An encoder turns a variable-length waveform into one fixed-length, L2-normalised
vector -- the *voice print*. Everything above this layer (enrolment, scoring,
identification) only talks to this interface, so swapping a DSP encoder for a
neural one changes nothing else in the system.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from ..audio import TARGET_SR, preprocess


@dataclass(frozen=True)
class EncoderSpec:
    """Identity of the encoder that produced a set of embeddings.

    Stored alongside every voice print. Comparing two embeddings from different
    encoders is meaningless but numerically silent, so the gallery refuses it by
    checking this record rather than trusting the caller.
    """

    name: str
    version: str
    dim: int
    sample_rate: int

    def as_dict(self) -> dict:
        return asdict(self)

    def compatible_with(self, other: EncoderSpec) -> bool:
        return (self.name, self.version, self.dim) == (other.name, other.version, other.dim)


class SpeakerEncoder(abc.ABC):
    """Base class for every voice-print extractor."""

    name: str = "base"
    version: str = "0"
    dim: int = 0
    sample_rate: int = TARGET_SR

    #: Minimum usable speech after silence removal. Below this the embedding is
    #: dominated by whichever phonemes happened to be spoken, not by the speaker.
    min_speech_seconds: float = 1.0

    @property
    def spec(self) -> EncoderSpec:
        return EncoderSpec(self.name, self.version, self.dim, self.sample_rate)

    @abc.abstractmethod
    def _embed_prepared(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Embed already pre-processed audio. Implemented by subclasses."""

    def embed(self, wav: np.ndarray, sr: int = TARGET_SR, *, prepared: bool = False) -> np.ndarray:
        """Return the L2-normalised voice print for ``wav``.

        ``prepared=True`` skips the front-end, for callers that already ran
        :func:`voxprint.audio.preprocess` (for example when embedding several
        windows cut from one long recording).
        """
        wav = np.asarray(wav, dtype=np.float32)
        if not prepared:
            wav = preprocess(wav, sr)
        if wav.size < int(self.sample_rate * 0.1):
            raise ValueError(
                f"only {wav.size / self.sample_rate:.2f}s of speech after silence removal; "
                "need a longer or louder recording"
            )
        vec = np.asarray(self._embed_prepared(wav, sr), dtype=np.float64).ravel()
        if vec.size != self.dim:
            raise RuntimeError(f"{self.name} produced {vec.size} dims, expected {self.dim}")
        return l2_normalize(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))

    def embed_many(self, clips: Iterable[np.ndarray], sr: int = TARGET_SR) -> np.ndarray:
        """Embed several clips -> ``(n_clips, dim)``."""
        vecs = [self.embed(clip, sr) for clip in clips]
        return np.stack(vecs) if vecs else np.zeros((0, self.dim))

    def embed_windows(
        self,
        wav: np.ndarray,
        sr: int = TARGET_SR,
        window_seconds: float = 3.0,
        hop_seconds: float = 1.5,
    ) -> np.ndarray:
        """Slide a window over one recording -> ``(n_windows, dim)``.

        Averaging window embeddings is markedly more stable than embedding the
        whole file at once, because it stops one loud or atypical stretch of
        audio from dominating the time-averaged statistics.
        """
        wav = preprocess(np.asarray(wav, dtype=np.float32), sr)
        win = int(self.sample_rate * window_seconds)
        hop = max(1, int(self.sample_rate * hop_seconds))
        if wav.size <= win:
            return self.embed(wav, sr, prepared=True)[None, :]
        starts = range(0, wav.size - win + 1, hop)
        return np.stack([self.embed(wav[s : s + win], sr, prepared=True) for s in starts])

    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "dim": self.dim,
            "sample_rate": self.sample_rate,
            "min_speech_seconds": self.min_speech_seconds,
            "doc": (self.__doc__ or "").strip().splitlines()[0] if self.__doc__ else "",
        }


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

def l2_normalize(vec: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return np.asarray(vec, dtype=np.float64) / np.maximum(norm, 1e-12)


def average_embeddings(vecs: np.ndarray) -> np.ndarray:
    """Centroid of several embeddings, renormalised to the unit sphere.

    This is the standard way to fuse multiple enrolment takes: the mean cancels
    session noise while the speaker direction reinforces itself.
    """
    vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float64))
    if vecs.shape[0] == 0:
        raise ValueError("cannot average zero embeddings")
    return l2_normalize(vecs.mean(axis=0))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, Callable[..., SpeakerEncoder]] = {}


def register_encoder(name: str) -> Callable[[Callable[..., SpeakerEncoder]], Callable[..., SpeakerEncoder]]:
    def decorator(factory):
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


def get_encoder(name: str = "dsp", **kwargs) -> SpeakerEncoder:
    """Instantiate an encoder by name.

    Heavy backends are imported lazily inside their factory, so importing
    :mod:`voxprint` never pulls in PyTorch.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; available: {', '.join(available_encoders())}")
    return _REGISTRY[name](**kwargs)
