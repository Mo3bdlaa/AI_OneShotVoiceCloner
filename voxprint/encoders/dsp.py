"""Signal-processing speaker encoder -- the always-available default.

Design
------
Instead of learning a representation, this encoder concatenates several
*hand-chosen views* of the voice, each of which captures a different physical
property of the speaker:

===============  ===========================================================
block            what it measures
===============  ===========================================================
``mfcc_cmvn``    articulation, with the recording channel normalised away
``mfcc_raw``     absolute spectral envelope -- vocal-tract size (channel bound)
``delta1/2``     articulation *dynamics*: how fast the speaker moves
``lpcc``         all-pole vocal-tract model -- resonator anatomy
``ltas``         long-term average spectrum -- overall timbre
``shape``        brightness, spread, flatness -- voice quality / breathiness
``prosody``      pitch register and its variability
===============  ===========================================================

Each block is centred and L2-normalised on its own before being weighted and
concatenated, so a block with 80 dimensions cannot drown out one with 9.

Honesty about accuracy
----------------------
This is a *baseline*. On clean, matched-channel audio it separates speakers well
enough to be useful; across different microphones or noisy rooms it degrades
sharply, because half of its blocks are channel-sensitive by construction. Where
accuracy matters, use the ``ecapa`` encoder, which is a neural network trained
discriminatively on thousands of speakers. The value of this one is that it has
no model download, no GPU, and no licence -- so the whole pipeline is testable
and the neural backend becomes a drop-in upgrade rather than a prerequisite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import TARGET_SR
from ..features import (
    EPS,
    cmvn,
    deltas,
    estimate_f0,
    log_mel_spectrogram,
    lpc_cepstrum,
    mfcc,
    power_spectrogram,
    robust_stats,
    spectral_shape,
)
from .base import SpeakerEncoder, l2_normalize, register_encoder

# Rough population priors used to put prosody features on a comparable scale.
# They are not fitted to any corpus -- they only need to be the right order of
# magnitude so that no single prosodic number dominates the block.
_PROSODY_CENTER = np.array([4.95, 0.35, 4.80, 5.15, 0.55, 0.55, 0.06, 0.18, 0.15])
_PROSODY_SCALE = np.array([0.40, 0.25, 0.42, 0.42, 0.25, 0.20, 0.05, 0.12, 0.10])


@dataclass(frozen=True)
class BlockWeights:
    """Relative contribution of each feature block to the final vector.

    Pitch (``prosody``) is weighted highest per unit of information: it is a
    small block but one of the most speaker-specific signals available.
    """

    mfcc_cmvn: float = 1.0
    mfcc_raw: float = 0.7
    delta1: float = 0.8
    delta2: float = 0.5
    lpcc: float = 0.9
    ltas: float = 0.8
    shape: float = 0.5
    prosody: float = 1.1

    def as_dict(self) -> dict[str, float]:
        return {
            "mfcc_cmvn": self.mfcc_cmvn,
            "mfcc_raw": self.mfcc_raw,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "lpcc": self.lpcc,
            "ltas": self.ltas,
            "shape": self.shape,
            "prosody": self.prosody,
        }


@dataclass
class DspEncoderConfig:
    n_mfcc: int = 20
    n_mels: int = 40
    n_fft: int = 512
    lpc_order: int = 16
    n_lpcc: int = 16
    weights: BlockWeights = field(default_factory=BlockWeights)


class DspSpeakerEncoder(SpeakerEncoder):
    """Hand-engineered voice print: MFCC/LPCC statistics fused with prosody."""

    name = "dsp"
    version = "1"
    sample_rate = TARGET_SR
    min_speech_seconds = 1.5

    def __init__(self, config: DspEncoderConfig | None = None):
        self.config = config or DspEncoderConfig()
        c = self.config
        self._block_dims = {
            "mfcc_cmvn": 4 * c.n_mfcc,
            "mfcc_raw": 2 * c.n_mfcc,
            "delta1": 4 * c.n_mfcc,
            "delta2": 2 * c.n_mfcc,
            "lpcc": 4 * c.n_lpcc,
            "ltas": c.n_mels,
            "shape": 4 * 5,
            "prosody": _PROSODY_CENTER.size,
        }
        self.dim = sum(self._block_dims.values())

    # -- blocks ------------------------------------------------------------ #

    def blocks(self, wav: np.ndarray, sr: int = TARGET_SR) -> dict[str, np.ndarray]:
        """Compute every block separately -- useful for debugging and ablations."""
        c = self.config
        coeffs = mfcc(wav, sr=sr, n_mfcc=c.n_mfcc, n_mels=c.n_mels, n_fft=c.n_fft)
        normed = cmvn(coeffs)
        d1 = deltas(normed)
        d2 = deltas(d1)

        logmel = log_mel_spectrogram(wav, sr=sr, n_mels=c.n_mels, n_fft=c.n_fft)
        power = power_spectrogram(wav, sr=sr, n_fft=c.n_fft)
        shape = _scale_shape(spectral_shape(power, sr=sr))
        lpcc = lpc_cepstrum(wav, sr=sr, order=c.lpc_order, n_cep=c.n_lpcc)

        return {
            "mfcc_cmvn": robust_stats(normed),
            "mfcc_raw": np.concatenate([coeffs.mean(axis=0), coeffs.std(axis=0)]),
            "delta1": robust_stats(d1),
            "delta2": np.concatenate([d2.mean(axis=0), d2.std(axis=0)]),
            "lpcc": robust_stats(lpcc),
            "ltas": logmel.mean(axis=0),
            "shape": robust_stats(shape),
            "prosody": _prosody_vector(wav, sr),
        }

    def _embed_prepared(self, wav: np.ndarray, sr: int) -> np.ndarray:
        weights = self.config.weights.as_dict()
        parts = []
        for key, vec in self.blocks(wav, sr).items():
            vec = np.nan_to_num(np.asarray(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if vec.size != self._block_dims[key]:  # pragma: no cover - guards a coding error
                raise RuntimeError(f"block {key} has {vec.size} dims, expected {self._block_dims[key]}")
            # Centre within the block: removes a constant offset shared by all
            # speakers, which otherwise inflates every cosine score toward 1.
            vec = vec - vec.mean()
            parts.append(l2_normalize(vec) * weights[key])
        return np.concatenate(parts)

    def describe(self) -> dict:
        info = super().describe()
        info["blocks"] = dict(self._block_dims)
        info["weights"] = self.config.weights.as_dict()
        return info


def _scale_shape(shape: np.ndarray) -> np.ndarray:
    """Put the five spectral descriptors on roughly one scale (log where in Hz)."""
    out = shape.copy()
    out[:, 0] = np.log(out[:, 0] + 1.0)   # centroid  (Hz)
    out[:, 1] = np.log(out[:, 1] + 1.0)   # spread    (Hz)
    out[:, 2] = np.log(out[:, 2] + 1.0)   # rolloff   (Hz)
    out[:, 3] = np.log(out[:, 3] + EPS)   # flatness  (ratio)
    return out


def _prosody_vector(wav: np.ndarray, sr: int) -> np.ndarray:
    """Nine pitch-derived numbers, standardised against population priors."""
    track = estimate_f0(wav, sr=sr)
    voiced = track.voiced_f0
    if voiced.size < 3:
        return np.zeros(_PROSODY_CENTER.size, dtype=np.float64)

    log_f0 = np.log(voiced)
    p25, p50, p75 = np.percentile(log_f0, [25, 50, 75])
    p10, p90 = np.percentile(log_f0, [10, 90])

    # Frame-to-frame pitch change on contiguous voiced runs: a jitter proxy that
    # separates steady voices from tremulous ones.
    idx = np.flatnonzero(track.voiced)
    contiguous = np.flatnonzero(np.diff(idx) == 1)
    if contiguous.size:
        a = np.log(track.f0[idx[contiguous]])
        b = np.log(track.f0[idx[contiguous + 1]])
        jitter = float(np.mean(np.abs(b - a)))
    else:
        jitter = 0.0

    raw = np.array(
        [
            p50,
            p75 - p25,
            p10,
            p90,
            float(track.voiced.mean()),
            float(track.clarity.mean()),
            jitter,
            float(log_f0.std()),
            float(track.clarity.std()),
        ],
        dtype=np.float64,
    )
    return (raw - _PROSODY_CENTER) / _PROSODY_SCALE


@register_encoder("dsp")
def _make_dsp(**kwargs) -> DspSpeakerEncoder:
    weights = kwargs.pop("weights", None)
    config = DspEncoderConfig(**kwargs)
    if weights is not None:
        config.weights = weights if isinstance(weights, BlockWeights) else BlockWeights(**weights)
    return DspSpeakerEncoder(config)
