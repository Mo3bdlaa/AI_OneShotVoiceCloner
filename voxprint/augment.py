"""Acoustic augmentations, used to make calibration realistic.

Why this exists
---------------
Measured on twelve speakers, with a threshold calibrated on clean enrolment
audio and queries degraded with white noise:

=========  ==================  ====================  ==================  ====================
condition  dsp rank-1          dsp accepted          ecapa rank-1        ecapa accepted
=========  ==================  ====================  ====================  ==================
clean      1.000               0.972                 1.000               1.000
40 dB SNR  1.000               0.028                 0.917               0.083
20 dB SNR  0.333               0.000                 0.917               0.000
10 dB SNR  0.083               0.000                 0.833               0.000
=========  ==================  ====================  ====================  ==================

ECAPA keeps *ranking* the right speaker first far longer than the DSP encoder --
that is the real difference between a trained encoder and a hand-built one. But
**both** collapse to zero acceptance, because both were given a threshold derived
from clean-against-clean comparisons, and any condition mismatch shifts every
score downward.

So the threshold, not the encoder, is the binding constraint under mismatch. This
module generates the degraded audio that lets calibration see that shift and
account for it; :meth:`voxprint.pipeline.VoiceLab.calibrate` with ``robust=True``
uses it.

All augmentations are deterministic given a seed, so a calibration can be
reproduced exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR, rms_normalize


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Individual degradations
# --------------------------------------------------------------------------- #

def add_noise(
    wav: np.ndarray,
    snr_db: float,
    sr: int = TARGET_SR,
    color: str = "white",
    seed: int | None = None,
) -> np.ndarray:
    """Mix in noise at a given signal-to-noise ratio.

    ``color`` selects the spectral shape. Pink noise (-3 dB per octave) is a much
    closer match to real room and traffic noise than white, which puts most of its
    energy where speech has least.
    """
    rng = _rng(seed)
    noise = rng.standard_normal(wav.size)
    if color == "pink":
        noise = _pink(noise)
    elif color != "white":
        raise ValueError(f"unknown noise colour {color!r}; use 'white' or 'pink'")

    signal_power = float(np.mean(np.asarray(wav, dtype=np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return np.clip(wav + scale * noise, -1.0, 1.0).astype(np.float32)


def _pink(white: np.ndarray) -> np.ndarray:
    """Shape white noise to a 1/f power spectrum."""
    spectrum = np.fft.rfft(white)
    freqs = np.arange(spectrum.size)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0.0
    out = np.fft.irfft(spectrum, n=white.size)
    return out / (np.std(out) + 1e-12)


def add_reverb(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    rt60: float = 0.3,
    direct_to_reverb_db: float = 6.0,
    seed: int | None = None,
) -> np.ndarray:
    """Convolve with a synthetic room impulse response.

    The RIR is exponentially decaying noise, which is a crude but spectrally fair
    stand-in for a real room: it smears the spectral envelope over time in the
    same way, and that smearing is what a speaker encoder actually suffers from.
    """
    rng = _rng(seed)
    length = max(8, int(sr * rt60))
    decay = np.exp(-6.9 * np.arange(length) / length)   # -60 dB across rt60
    rir = rng.standard_normal(length) * decay
    rir[0] += 10.0 ** (direct_to_reverb_db / 20.0)      # direct path
    rir /= np.linalg.norm(rir) + 1e-12

    wet = np.convolve(np.asarray(wav, dtype=np.float64), rir, mode="full")[: wav.size]
    return rms_normalize(wet.astype(np.float32))


def band_limit(wav: np.ndarray, sr: int = TARGET_SR, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """Restrict to a telephone passband -- the classic channel mismatch."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    high = min(high_hz / nyq, 0.99)
    low = max(low_hz / nyq, 1e-4)
    sos = butter(4, [low, high], btype="band", output="sos")
    return rms_normalize(sosfiltfilt(sos, np.asarray(wav, dtype=np.float64)).astype(np.float32))


def soft_clip(wav: np.ndarray, drive: float = 4.0) -> np.ndarray:
    """Saturate the waveform, as a recording with the gain set too high would."""
    return rms_normalize(np.tanh(drive * np.asarray(wav, dtype=np.float64)).astype(np.float32))


def spectral_tilt(wav: np.ndarray, db_per_khz: float = 1.5, sr: int = TARGET_SR) -> np.ndarray:
    """Apply a smooth spectral slope -- a stand-in for a different microphone."""
    spectrum = np.fft.rfft(np.asarray(wav, dtype=np.float64))
    freqs = np.fft.rfftfreq(wav.size, 1.0 / sr)
    gain = 10.0 ** (db_per_khz * freqs / 1000.0 / 20.0)
    out = np.fft.irfft(spectrum * gain, n=wav.size)
    return rms_normalize(out.astype(np.float32))


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Augmentation:
    """One named degradation."""

    name: str
    apply: Callable[[np.ndarray, int, int], np.ndarray]

    def __call__(self, wav: np.ndarray, sr: int = TARGET_SR, seed: int = 0) -> np.ndarray:
        return self.apply(wav, sr, seed)


def default_suite() -> list[Augmentation]:
    """The conditions a calibration should expect to meet in the field.

    Chosen to span the failure modes separately rather than to be exhaustive:
    additive noise at two severities, room acoustics, channel bandwidth, a
    different microphone response, and clipping. Six variants per enrolment take
    keeps a robust calibration to a few seconds per speaker.
    """
    return [
        Augmentation("noise_30db", lambda w, sr, s: add_noise(w, 30.0, sr, "pink", seed=s)),
        Augmentation("noise_20db", lambda w, sr, s: add_noise(w, 20.0, sr, "pink", seed=s + 1)),
        Augmentation("noise_white_25db", lambda w, sr, s: add_noise(w, 25.0, sr, "white", seed=s + 2)),
        Augmentation("reverb", lambda w, sr, s: add_reverb(w, sr, rt60=0.35, seed=s + 3)),
        Augmentation("telephone", lambda w, sr, s: band_limit(w, sr)),
        Augmentation("mic_tilt", lambda w, sr, s: spectral_tilt(w, 2.0, sr)),
    ]


def light_suite() -> list[Augmentation]:
    """A cheaper suite for enrolment-time augmentation of the voice print itself."""
    return [
        Augmentation("noise_30db", lambda w, sr, s: add_noise(w, 30.0, sr, "pink", seed=s)),
        Augmentation("reverb", lambda w, sr, s: add_reverb(w, sr, rt60=0.25, seed=s + 1)),
    ]


def augment_all(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    suite: list[Augmentation] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Apply every augmentation in a suite -> ``{name: waveform}``."""
    return {aug.name: aug(wav, sr, seed) for aug in (suite if suite is not None else default_suite())}
