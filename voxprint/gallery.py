"""The voice-print store: enrolment, persistence, and gallery-level statistics.

A gallery holds one :class:`VoicePrint` per speaker. Each print keeps both the
individual utterance embeddings and their centroid, because the two answer
different questions: the centroid is what you match against, while the spread of
the utterances tells you how *trustworthy* that centroid is.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .encoders.base import EncoderSpec, average_embeddings, l2_normalize
from .normalization import EmbeddingStandardizer, load_default_reference

SCHEMA_VERSION = 2
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@-]{0,63}$")


class GalleryError(RuntimeError):
    """Raised for enrolment and persistence problems."""


class ConsentError(GalleryError):
    """Raised when a voice is enrolled without a consent record."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_speaker_id(speaker_id: str) -> str:
    speaker_id = (speaker_id or "").strip()
    if not _ID_RE.match(speaker_id):
        raise GalleryError(
            f"invalid speaker id {speaker_id!r}: use 1-64 characters from letters, digits, "
            "space, dot, underscore, hyphen or @"
        )
    return speaker_id


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConsentRecord:
    """Who agreed to have this voice modelled, and for what.

    Stored with the print rather than in a separate log so it cannot drift away
    from the data it authorises. A voice print is biometric data and the cloning
    backends can reproduce a person's voice from it, so enrolment without this
    record is refused by default.
    """

    granted_by: str
    statement: str
    purpose: str = ""
    granted_at: str = field(default_factory=_now)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> ConsentRecord | None:
        if not data:
            return None
        return cls(
            granted_by=str(data.get("granted_by", "")),
            statement=str(data.get("statement", "")),
            purpose=str(data.get("purpose", "")),
            granted_at=str(data.get("granted_at", _now())),
        )


# --------------------------------------------------------------------------- #
# Voice print
# --------------------------------------------------------------------------- #

@dataclass
class VoicePrint:
    """One enrolled speaker."""

    speaker_id: str
    centroid: np.ndarray
    utterances: np.ndarray
    display_name: str = ""
    consent: ConsentRecord | None = None
    total_seconds: float = 0.0
    sources: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def n_utterances(self) -> int:
        return int(self.utterances.shape[0])

    def add(self, embeddings: np.ndarray, seconds: float = 0.0, sources: Sequence[str] = ()) -> None:
        """Append new enrolment material and recompute the centroid."""
        new = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        if new.shape[1] != self.utterances.shape[1]:
            raise GalleryError(
                f"embedding dimension mismatch: print has {self.utterances.shape[1]}, got {new.shape[1]}"
            )
        self.utterances = np.vstack([self.utterances, new])
        self.centroid = average_embeddings(self.utterances)
        self.total_seconds += float(seconds)
        self.sources.extend(str(s) for s in sources)
        self.updated_at = _now()

    def cohesion(self) -> float:
        """Mean pairwise cosine between this speaker's own utterances.

        A low value means the enrolment takes disagree with each other -- usually
        different microphones, background noise, or a second person in the room.
        It is the single best warning sign that a print will not match reliably,
        and it is available without any labelled data.
        """
        if self.n_utterances < 2:
            return float("nan")
        vecs = l2_normalize(self.utterances, axis=1)
        sim = vecs @ vecs.T
        iu = np.triu_indices(self.n_utterances, k=1)
        return float(sim[iu].mean())

    def quality_report(self, min_seconds: float = 6.0) -> dict:
        """Human-readable warnings about this print's enrolment quality."""
        warnings: list[str] = []
        if self.total_seconds and self.total_seconds < min_seconds:
            warnings.append(
                f"only {self.total_seconds:.1f}s of speech enrolled; {min_seconds:.0f}s+ is much more stable"
            )
        if self.n_utterances < 2:
            warnings.append("single enrolment take: no way to measure consistency, and no session averaging")
        coh = self.cohesion()
        if self.n_utterances >= 2 and coh < 0.55:
            warnings.append(f"enrolment takes disagree (cohesion {coh:.2f}); check for noise or mixed speakers")
        return {
            "speaker_id": self.speaker_id,
            "utterances": self.n_utterances,
            "total_seconds": round(self.total_seconds, 2),
            "cohesion": None if np.isnan(coh) else round(coh, 4),
            "warnings": warnings,
        }

    def meta_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "consent": self.consent.as_dict() if self.consent else None,
            "total_seconds": self.total_seconds,
            "sources": list(self.sources),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #

class Gallery:
    """A named collection of voice prints produced by one encoder."""

    def __init__(
        self,
        spec: EncoderSpec,
        standardizer: EmbeddingStandardizer | None = None,
        *,
        require_consent: bool = True,
        threshold: float | None = None,
        identification_threshold: float | None = None,
    ):
        self.spec = spec
        self.prints: dict[str, VoicePrint] = {}
        self.require_consent = require_consent
        #: Threshold for 1-to-1 verification.
        self.threshold = threshold
        #: Threshold for 1-to-N identification. Higher, because the score being
        #: tested is a maximum over every enrolled speaker -- see
        #: :mod:`voxprint.scoring`.
        self.identification_threshold = identification_threshold
        self.standardizer = standardizer or load_default_reference(spec)
        self.created_at = _now()

    # -- container protocol ------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.prints)

    def __contains__(self, speaker_id: object) -> bool:
        return speaker_id in self.prints

    def __iter__(self) -> Iterator[VoicePrint]:
        return iter(self.prints.values())

    def ids(self) -> list[str]:
        return sorted(self.prints)

    def get(self, speaker_id: str) -> VoicePrint:
        try:
            return self.prints[speaker_id]
        except KeyError as exc:
            raise GalleryError(f"no enrolled speaker with id {speaker_id!r}") from exc

    # -- enrolment --------------------------------------------------------- #

    def enroll(
        self,
        speaker_id: str,
        embeddings: np.ndarray,
        *,
        display_name: str = "",
        consent: ConsentRecord | None = None,
        seconds: float = 0.0,
        sources: Sequence[str] = (),
        notes: str = "",
        replace: bool = False,
    ) -> VoicePrint:
        """Add a speaker, or extend an existing one with more material."""
        speaker_id = validate_speaker_id(speaker_id)
        vecs = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        if vecs.shape[0] == 0:
            raise GalleryError("no embeddings supplied")
        if vecs.shape[1] != self.spec.dim:
            raise GalleryError(
                f"gallery expects {self.spec.dim}-dim embeddings from encoder "
                f"{self.spec.name!r}, got {vecs.shape[1]}"
            )

        existing = self.prints.get(speaker_id)
        if existing is not None and not replace:
            if self.require_consent and existing.consent is None and consent is None:
                raise ConsentError(f"speaker {speaker_id!r} has no consent record on file")
            existing.add(vecs, seconds=seconds, sources=sources)
            if consent is not None:
                existing.consent = consent
            if display_name:
                existing.display_name = display_name
            if notes:
                existing.notes = notes
            return existing

        if self.require_consent and consent is None:
            raise ConsentError(
                f"enrolling {speaker_id!r} requires a consent record. A voice print is biometric "
                "data and can drive the cloning backends; record who agreed and why. "
                "Pass consent=ConsentRecord(...), or construct the Gallery with require_consent=False "
                "for test fixtures and synthetic voices."
            )

        print_ = VoicePrint(
            speaker_id=speaker_id,
            centroid=average_embeddings(vecs),
            utterances=vecs,
            display_name=display_name or speaker_id,
            consent=consent,
            total_seconds=float(seconds),
            sources=[str(s) for s in sources],
            notes=notes,
        )
        self.prints[speaker_id] = print_
        return print_

    def remove(self, speaker_id: str) -> None:
        """Delete a speaker. Erasure has to be as easy as enrolment."""
        if speaker_id not in self.prints:
            raise GalleryError(f"no enrolled speaker with id {speaker_id!r}")
        del self.prints[speaker_id]

    def rename(self, speaker_id: str, new_id: str) -> None:
        new_id = validate_speaker_id(new_id)
        if new_id in self.prints:
            raise GalleryError(f"speaker id {new_id!r} already exists")
        print_ = self.prints.pop(speaker_id)
        print_.speaker_id = new_id
        print_.updated_at = _now()
        self.prints[new_id] = print_

    # -- matrices ---------------------------------------------------------- #

    def centroid_matrix(self, standardize: bool = True) -> tuple[list[str], np.ndarray]:
        """``(ids, matrix)`` of enrolled centroids, ready for a dot product."""
        ids = self.ids()
        if not ids:
            return [], np.zeros((0, self.spec.dim))
        mat = np.stack([self.prints[i].centroid for i in ids])
        return ids, (self.standardizer.transform(mat) if standardize else mat)

    def all_utterances(self, standardize: bool = False) -> tuple[list[str], np.ndarray]:
        """Every enrolment embedding with its speaker label -- used for calibration."""
        labels: list[str] = []
        rows: list[np.ndarray] = []
        for sid in self.ids():
            for vec in self.prints[sid].utterances:
                labels.append(sid)
                rows.append(vec)
        mat = np.stack(rows) if rows else np.zeros((0, self.spec.dim))
        return labels, (self.standardizer.transform(mat) if standardize and rows else mat)

    def refit_standardizer(self, min_speakers: int = 8, min_samples: int = 24) -> bool:
        """Re-estimate the reference statistics from the enrolled population.

        Real enrolled voices are a far better reference than the shipped prior --
        but only once there are enough of them. Fitting a 373-dimensional
        per-dimension variance on a handful of vectors mostly describes those
        particular speakers, and measurably weakens held-out scores; below these
        thresholds the existing standardiser is kept instead.
        """
        if len(self.prints) < min_speakers:
            return False
        _, mat = self.all_utterances(standardize=False)
        if mat.shape[0] < min_samples:
            return False
        self.standardizer = EmbeddingStandardizer.fit(
            mat, spec=self.spec, source="gallery", min_samples=min_samples
        )
        return not self.standardizer.is_identity

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the whole gallery to one ``.npz`` file, atomically."""
        path = os.fspath(path)
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "encoder": self.spec.as_dict(),
            "created_at": self.created_at,
            "saved_at": _now(),
            "threshold": self.threshold,
            "identification_threshold": self.identification_threshold,
            "require_consent": self.require_consent,
            "standardizer": self.standardizer.describe(),
            "speakers": [],
        }
        for idx, sid in enumerate(self.ids()):
            print_ = self.prints[sid]
            arrays[f"utt_{idx}"] = print_.utterances.astype(np.float64)
            arrays[f"cen_{idx}"] = print_.centroid.astype(np.float64)
            entry = print_.meta_dict()
            entry["index"] = idx
            manifest["speakers"].append(entry)

        arrays["manifest"] = np.array(json.dumps(manifest, ensure_ascii=False))
        arrays["std_mean"] = self.standardizer.mean.astype(np.float64)
        arrays["std_scale"] = self.standardizer.scale.astype(np.float64)

        # The suffix must be ".npz": numpy appends that extension to any other
        # name, which would leave the rename pointing at the empty stub file.
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Gallery:
        if not os.path.exists(path):
            raise GalleryError(f"gallery file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"].item()))
            if int(manifest.get("schema_version", 0)) > SCHEMA_VERSION:
                raise GalleryError(
                    f"gallery was written by a newer version (schema "
                    f"{manifest['schema_version']} > {SCHEMA_VERSION})"
                )
            spec = EncoderSpec(**manifest["encoder"])
            std_info = manifest.get("standardizer", {})
            standardizer = EmbeddingStandardizer(
                mean=np.asarray(data["std_mean"], dtype=np.float64),
                scale=np.asarray(data["std_scale"], dtype=np.float64),
                n_samples=int(std_info.get("n_samples", 0)),
                source=str(std_info.get("source", "file")),
                encoder=spec.name,
                encoder_version=spec.version,
            )
            gallery = cls(
                spec,
                standardizer=standardizer,
                require_consent=bool(manifest.get("require_consent", True)),
                threshold=manifest.get("threshold"),
                # Absent in schema 1: such a gallery falls back to the
                # verification threshold, which is what it was using anyway.
                identification_threshold=manifest.get("identification_threshold"),
            )
            gallery.created_at = str(manifest.get("created_at", _now()))

            for entry in manifest["speakers"]:
                idx = entry["index"]
                gallery.prints[entry["speaker_id"]] = VoicePrint(
                    speaker_id=entry["speaker_id"],
                    centroid=np.asarray(data[f"cen_{idx}"], dtype=np.float64),
                    utterances=np.asarray(data[f"utt_{idx}"], dtype=np.float64),
                    display_name=entry.get("display_name", ""),
                    consent=ConsentRecord.from_dict(entry.get("consent")),
                    total_seconds=float(entry.get("total_seconds", 0.0)),
                    sources=list(entry.get("sources", [])),
                    notes=entry.get("notes", ""),
                    created_at=entry.get("created_at", _now()),
                    updated_at=entry.get("updated_at", _now()),
                )
        return gallery

    # -- reporting --------------------------------------------------------- #

    def summary(self) -> dict:
        return {
            "encoder": self.spec.as_dict(),
            "speakers": len(self.prints),
            "utterances": sum(p.n_utterances for p in self.prints.values()),
            "total_seconds": round(sum(p.total_seconds for p in self.prints.values()), 2),
            "threshold": self.threshold,
            "identification_threshold": self.identification_threshold,
            "standardizer": self.standardizer.describe(),
            "require_consent": self.require_consent,
        }
