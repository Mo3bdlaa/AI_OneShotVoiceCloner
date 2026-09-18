"""Audio I/O and pre-processing primitives.

Everything downstream assumes the same contract: a 1-D ``float32`` array in
``[-1, 1]`` together with an explicit sample rate. Keeping that contract in one
place means the encoders never have to care where the audio came from.
"""

from __future__ import annotations

import io
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

TARGET_SR = 16_000
"""Working sample rate. 16 kHz is the standard for speaker recognition: it keeps
the formant range intact while halving the compute versus 32 kHz."""


class AudioError(RuntimeError):
    """Raised when audio cannot be read or is unusable for enrolment."""


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def _require_soundfile():
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AudioError(
            "soundfile is required for audio I/O. Install it with "
            "`pip install soundfile` (it needs libsndfile on the system)."
        ) from exc
    return sf


def load_audio(
    source: str | os.PathLike[str] | BinaryIO | bytes,
    sr: int = TARGET_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Read an audio file and return ``(wav, sr)`` ready for feature extraction.

    ``source`` may be a path, an open binary file object, or raw bytes, which
    makes the same function usable from the CLI and from an HTTP upload.
    """
    sf = _require_soundfile()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        data, file_sr = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises a zoo of exception types
        raise AudioError(f"could not decode audio: {exc}") from exc

    if data.size == 0:
        raise AudioError("audio file contains no samples")

    wav = data.mean(axis=1) if mono else data
    if file_sr != sr:
        wav = resample(wav, file_sr, sr)
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def save_audio(path: str | os.PathLike[str], wav: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a mono float waveform to disk, clipping instead of wrapping."""
    sf = _require_soundfile()
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, np.clip(wav, -1.0, 1.0).astype(np.float32), sr)


# --------------------------------------------------------------------------- #
# Basic transforms
# --------------------------------------------------------------------------- #

def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Rational resampling via polyphase filtering (anti-aliased)."""
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    from scipy.signal import resample_poly  # noqa: PLC0415

    g = math.gcd(int(orig_sr), int(target_sr))
    up, down = target_sr // g, orig_sr // g
    out = resample_poly(np.asarray(wav, dtype=np.float64), up, down, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def to_mono(wav: np.ndarray) -> np.ndarray:
    return wav if wav.ndim == 1 else wav.mean(axis=1).astype(np.float32)


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav if m < 1e-9 else (wav * (peak / m)).astype(np.float32)


def rms_normalize(wav: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    """Loudness-align a clip so level differences do not leak into the embedding.

    Without this, "loud recording" becomes a feature the encoder can latch onto,
    and the same speaker recorded twice at different gains scores as two people.
    """
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) if wav.size else 0.0
    if rms < 1e-9:
        return wav.astype(np.float32)
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    return np.clip(wav * gain, -1.0, 1.0).astype(np.float32)


def preemphasis(wav: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """First-order high-pass that flattens the ~-6 dB/octave glottal roll-off."""
    if coeff <= 0:
        return wav.astype(np.float32)
    out = np.empty_like(wav, dtype=np.float32)
    out[0] = wav[0]
    out[1:] = wav[1:] - coeff * wav[:-1]
    return out


def frame_signal(wav: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split into overlapping frames -> ``(n_frames, frame_len)``.

    Uses a strided view so long files do not get copied twice; callers that
    mutate the result must copy first.
    """
    if wav.size < frame_len:
        pad = np.zeros(frame_len - wav.size, dtype=wav.dtype)
        wav = np.concatenate([wav, pad])
    n_frames = 1 + (wav.size - frame_len) // hop_len
    shape = (n_frames, frame_len)
    strides = (wav.strides[0] * hop_len, wav.strides[0])
    return np.lib.stride_tricks.as_strided(wav, shape=shape, strides=strides)


# --------------------------------------------------------------------------- #
# Voice activity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VadResult:
    """Frame-level speech decision plus the frame geometry used to compute it."""

    mask: np.ndarray       # bool, one entry per frame
    frame_len: int
    hop_len: int
    speech_ratio: float

    def to_sample_mask(self, n_samples: int) -> np.ndarray:
        """Expand the frame decision back onto the sample timeline."""
        out = np.zeros(n_samples, dtype=bool)
        for i, keep in enumerate(self.mask):
            if keep:
                start = i * self.hop_len
                out[start : min(start + self.frame_len, n_samples)] = True
        return out


def energy_vad(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    rel_db: float = 30.0,
    abs_floor_db: float = -55.0,
) -> VadResult:
    """Adaptive energy VAD.

    A frame counts as speech when it is within ``rel_db`` of the loudest frame
    *and* above an absolute floor. The relative part adapts to recording level;
    the absolute part stops a silent file from being read as all-speech.
    """
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frames = frame_signal(wav, frame_len, hop_len)
    energy = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    db = 20.0 * np.log10(energy + 1e-12)
    peak_db = float(db.max()) if db.size else -120.0
    mask = (db > peak_db - rel_db) & (db > abs_floor_db)
    ratio = float(mask.mean()) if mask.size else 0.0
    return VadResult(mask=mask, frame_len=frame_len, hop_len=hop_len, speech_ratio=ratio)


def trim_silence(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    rel_db: float = 35.0,
    pad_ms: float = 30.0,
) -> np.ndarray:
    """Drop leading/trailing silence, keeping a short pad so plosives survive."""
    vad = energy_vad(wav, sr, rel_db=rel_db)
    idx = np.flatnonzero(vad.mask)
    if idx.size == 0:
        return wav
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, idx[0] * vad.hop_len - pad)
    end = min(wav.size, idx[-1] * vad.hop_len + vad.frame_len + pad)
    return wav[start:end]


def keep_speech(wav: np.ndarray, sr: int = TARGET_SR, rel_db: float = 30.0) -> np.ndarray:
    """Concatenate only the speech frames -- silence carries no speaker identity."""
    vad = energy_vad(wav, sr, rel_db=rel_db)
    sample_mask = vad.to_sample_mask(wav.size)
    kept = wav[sample_mask]
    return kept if kept.size >= sr * 0.2 else wav


def preprocess(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    *,
    do_trim: bool = True,
    do_vad: bool = True,
    target_dbfs: float = -23.0,
) -> np.ndarray:
    """Standard front-end applied identically at enrolment and at query time.

    Any asymmetry here shows up as a systematic score shift, so both paths must
    call this one function.
    """
    wav = to_mono(np.asarray(wav, dtype=np.float32))
    wav = remove_dc(wav)
    if do_trim:
        wav = trim_silence(wav, sr)
    if do_vad:
        wav = keep_speech(wav, sr)
    return rms_normalize(wav, target_dbfs)


def remove_dc(wav: np.ndarray) -> np.ndarray:
    return (wav - float(np.mean(wav))).astype(np.float32) if wav.size else wav


def duration_seconds(wav: np.ndarray, sr: int = TARGET_SR) -> float:
    return float(wav.size) / float(sr)


def concat_with_gap(clips: Iterable[np.ndarray], sr: int = TARGET_SR, gap_ms: float = 120.0) -> np.ndarray:
    """Join clips with a short silence, e.g. when merging enrolment takes."""
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    parts: list[np.ndarray] = []
    for clip in clips:
        if parts:
            parts.append(gap)
        parts.append(np.asarray(clip, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
