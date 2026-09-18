"""Embedding standardisation -- the step that makes cosine scoring work.

Why this module exists
----------------------
Raw statistics-based voice prints occupy a very narrow cone: every human voice
shares most of its spectral structure, so *all* pairs score near 1.0 and the
speaker-specific part is buried in the last two decimal places. Measured on the
synthetic test cast, raw cosine gives same-speaker 0.99 and different-speaker
0.98 -- unusable.

Standardising each dimension by the mean and standard deviation of a reference
population re-weights the space so that dimensions which actually vary between
speakers dominate the dot product. The same measurement, with a held-out
reference, gives same-speaker 0.82-0.89 against different-speaker below 0.57.

This is the embedding-space analogue of z-scoring features before a distance
computation, and it is why neural encoders, which are trained to produce an
already-isotropic space, do not need it nearly as much.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .encoders.base import EncoderSpec, l2_normalize

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class EmbeddingStandardizer:
    """Per-dimension mean/scale normaliser fitted on a reference population."""

    mean: np.ndarray
    scale: np.ndarray
    n_samples: int = 0
    source: str = "identity"
    encoder: str = ""
    encoder_version: str = ""

    # -- construction ------------------------------------------------------ #

    @classmethod
    def identity(cls, dim: int) -> EmbeddingStandardizer:
        """A no-op standardiser, for encoders whose space is already isotropic."""
        return cls(mean=np.zeros(dim), scale=np.ones(dim), n_samples=0, source="identity")

    @classmethod
    def fit(
        cls,
        vectors: Iterable[np.ndarray],
        spec: EncoderSpec | None = None,
        source: str = "fitted",
        min_samples: int = 8,
    ) -> EmbeddingStandardizer:
        """Fit on a set of embeddings drawn from *different* speakers.

        Fitting on too few samples is worse than not fitting at all: the
        estimated variance then mostly describes the handful of speakers in the
        set, and scores against anyone else become arbitrary. Below
        ``min_samples`` this returns the identity instead of a bad estimate.
        """
        mat = np.atleast_2d(np.asarray(list(vectors), dtype=np.float64))
        dim = mat.shape[1] if mat.size else (spec.dim if spec else 0)
        if mat.shape[0] < min_samples:
            std = cls.identity(dim)
        else:
            mean = mat.mean(axis=0)
            scale = mat.std(axis=0)
            # Floor the scale relative to the typical spread so that a dimension
            # which happened to be constant in the reference set cannot explode
            # into the dominant term for an unseen voice.
            floor = max(float(np.median(scale)) * 1e-2, 1e-9)
            std = cls(mean=mean, scale=np.maximum(scale, floor), n_samples=mat.shape[0], source=source)
        if spec is not None:
            std.encoder, std.encoder_version = spec.name, spec.version
        return std

    # -- use --------------------------------------------------------------- #

    @property
    def is_identity(self) -> bool:
        return self.source == "identity"

    @property
    def dim(self) -> int:
        return int(self.mean.size)

    def transform(self, vec: np.ndarray) -> np.ndarray:
        """Standardise and re-normalise one embedding (or a stack of them)."""
        arr = np.asarray(vec, dtype=np.float64)
        if self.is_identity:
            return l2_normalize(arr)
        if arr.shape[-1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim embedding, got {arr.shape[-1]}")
        return l2_normalize((arr - self.mean) / self.scale)

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        return self.transform(vec)

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | os.PathLike[str]) -> None:
        parent = os.path.dirname(os.fspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        meta = json.dumps(
            {
                "n_samples": self.n_samples,
                "source": self.source,
                "encoder": self.encoder,
                "encoder_version": self.encoder_version,
            }
        )
        np.savez_compressed(path, mean=self.mean, scale=self.scale, meta=np.array(meta))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> EmbeddingStandardizer:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item())) if "meta" in data else {}
            return cls(
                mean=np.asarray(data["mean"], dtype=np.float64),
                scale=np.asarray(data["scale"], dtype=np.float64),
                n_samples=int(meta.get("n_samples", 0)),
                source=str(meta.get("source", "file")),
                encoder=str(meta.get("encoder", "")),
                encoder_version=str(meta.get("encoder_version", "")),
            )

    def describe(self) -> dict:
        return {
            "source": self.source,
            "dim": self.dim,
            "n_samples": self.n_samples,
            "encoder": self.encoder,
            "encoder_version": self.encoder_version,
            "identity": self.is_identity,
        }


def reference_path(spec: EncoderSpec) -> str:
    return os.path.join(_DATA_DIR, f"reference_{spec.name}_v{spec.version}.npz")


def load_default_reference(spec: EncoderSpec) -> EmbeddingStandardizer:
    """Load the packaged reference statistics for an encoder, if any.

    The shipped reference for the DSP encoder is built by
    ``tools/build_reference.py`` from a bank of procedurally generated voices.
    It is a *prior*, not ground truth: refitting on real enrolled speakers, which
    :meth:`voxprint.gallery.Gallery.refit_standardizer` does automatically once
    enough of them exist, is strictly better.
    """
    path = reference_path(spec)
    if os.path.exists(path):
        std = EmbeddingStandardizer.load(path)
        if std.dim == spec.dim:
            return std
    return EmbeddingStandardizer.identity(spec.dim)
