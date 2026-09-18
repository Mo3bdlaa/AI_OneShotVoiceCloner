"""STFT analysis/synthesis and time-scale tools used by the conversion backend.

Separate from :mod:`voxprint.features` because the goal here is different:
features throw information away on purpose, while these transforms have to be
invertible enough to reconstruct audio you can listen to.
"""

from __future__ import annotations

import numpy as np

from .audio import TARGET_SR


def hann(n: int) -> np.ndarray:
    """Periodic Hann window -- the periodic form is what makes COLA hold."""
    return np.hanning(n + 1)[:n].astype(np.float64)


def stft(wav: np.ndarray, n_fft: int = 1024, hop: int = 256, window: np.ndarray | None = None) -> np.ndarray:
    """``(n_frames, n_fft // 2 + 1)`` complex spectrogram, centred and reflect-padded."""
    win = hann(n_fft) if window is None else window
    padded = np.pad(np.asarray(wav, dtype=np.float64), n_fft // 2, mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop
    frames = np.stack([padded[i * hop : i * hop + n_fft] * win for i in range(max(n_frames, 0))])
    return np.fft.rfft(frames, n=n_fft, axis=1)


def istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256, length: int | None = None) -> np.ndarray:
    """Weighted overlap-add inverse of :func:`stft`."""
    win = hann(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1)
    n_frames = frames.shape[0]
    out = np.zeros((n_frames - 1) * hop + n_fft, dtype=np.float64)
    norm = np.zeros_like(out)
    for i in range(n_frames):
        start = i * hop
        out[start : start + n_fft] += frames[i] * win
        norm[start : start + n_fft] += win**2
    out /= np.maximum(norm, 1e-8)
    out = out[n_fft // 2 :]
    if length is not None:
        out = out[:length] if out.size >= length else np.pad(out, (0, length - out.size))
    return out.astype(np.float32)


def phase_vocoder(wav: np.ndarray, rate: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Change duration by ``rate`` without changing pitch.

    Standard phase-vocoder: interpolate magnitudes between analysis frames while
    integrating the *instantaneous frequency* so that partials stay phase-coherent
    instead of smearing.
    """
    if abs(rate - 1.0) < 1e-6:
        return np.asarray(wav, dtype=np.float32)

    spec = stft(wav, n_fft=n_fft, hop=hop)
    n_frames, n_bins = spec.shape
    time_steps = np.arange(0, n_frames - 1, rate)
    expected_advance = hop * 2.0 * np.pi * np.arange(n_bins) / n_fft

    magnitude = np.abs(spec)
    phase = np.angle(spec)
    out = np.zeros((time_steps.size, n_bins), dtype=np.complex128)
    accumulated = phase[0].copy()

    for i, t in enumerate(time_steps):
        lo = int(np.floor(t))
        frac = t - lo
        mag = (1.0 - frac) * magnitude[lo] + frac * magnitude[min(lo + 1, n_frames - 1)]
        out[i] = mag * np.exp(1j * accumulated)

        delta = phase[min(lo + 1, n_frames - 1)] - phase[lo] - expected_advance
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))  # wrap to (-pi, pi]
        accumulated = accumulated + expected_advance + delta

    return istft(out, n_fft=n_fft, hop=hop)


def pitch_shift(wav: np.ndarray, factor: float, sr: int = TARGET_SR, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Multiply pitch by ``factor`` while keeping the duration.

    Stretch in time by ``factor``, then resample by the same factor: the resample
    scales pitch and duration together, and the stretch cancels the duration
    change. Formants move with the pitch, which is a known limitation -- it makes
    a large shift sound artificial, so callers should clamp the factor.
    """
    if abs(factor - 1.0) < 1e-3:
        return np.asarray(wav, dtype=np.float32)
    from scipy.signal import resample_poly

    stretched = phase_vocoder(wav, 1.0 / factor, n_fft=n_fft, hop=hop)
    num, den = _ratio(1.0 / factor)
    shifted = resample_poly(stretched.astype(np.float64), num, den)
    target_len = int(np.asarray(wav).size)
    if shifted.size >= target_len:
        return shifted[:target_len].astype(np.float32)
    return np.pad(shifted, (0, target_len - shifted.size)).astype(np.float32)


def _ratio(value: float, max_den: int = 512) -> tuple[int, int]:
    """Rational approximation for polyphase resampling."""
    from fractions import Fraction

    frac = Fraction(float(value)).limit_denominator(max_den)
    return max(1, frac.numerator), max(1, frac.denominator)


def cepstral_envelope(log_magnitude: np.ndarray, n_coeff: int = 32) -> np.ndarray:
    """Smooth a log spectrum by keeping only the low-quefrency cepstrum.

    This is what separates the *envelope* (vocal tract, slow variation across
    frequency) from the *fine structure* (harmonics of the pitch, fast variation).
    Voice conversion works on the envelope and leaves the fine structure alone.
    """
    arr = np.atleast_2d(np.asarray(log_magnitude, dtype=np.float64))
    n_bins = arr.shape[1]
    # Mirror the spectrum so the DCT of the real cepstrum stays real and even.
    mirrored = np.concatenate([arr, arr[:, -2:0:-1]], axis=1)
    cep = np.fft.irfft(mirrored, axis=1).real
    cep[:, n_coeff:-n_coeff] = 0.0
    smoothed = np.fft.rfft(cep, axis=1).real[:, :n_bins]
    return smoothed if log_magnitude.ndim > 1 else smoothed[0]
