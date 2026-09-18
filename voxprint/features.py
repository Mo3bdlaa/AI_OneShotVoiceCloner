"""Acoustic features, implemented on plain NumPy.

Keeping the feature stack dependency-free (NumPy + a little SciPy) is a
deliberate choice: the fingerprint half of this project then runs anywhere,
including on a machine that will never have PyTorch installed. The neural
encoder in :mod:`voxprint.encoders.ecapa` brings its own features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR, frame_signal, preemphasis

EPS = 1e-10


# --------------------------------------------------------------------------- #
# Mel scale
# --------------------------------------------------------------------------- #

def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Slaney/HTK-style log mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = 40,
    n_fft: int = 512,
    sr: int = TARGET_SR,
    fmin: float = 40.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular mel filters -> ``(n_mels, n_fft // 2 + 1)``, area-normalised.

    ``fmin`` defaults to 40 Hz rather than 0 so that rumble and mains hum stay
    out of the first band, where they would otherwise dominate the log energy.
    """
    fmax = fmax if fmax is not None else sr / 2.0
    n_bins = n_fft // 2 + 1
    bin_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float64)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (bin_freqs - left) / max(center - left, EPS)
        falling = (right - bin_freqs) / max(right - center, EPS)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
        area = fb[m].sum()
        if area > EPS:
            fb[m] /= area
    return fb


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #

def power_spectrogram(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    preemph: float = 0.97,
) -> np.ndarray:
    """``(n_frames, n_fft // 2 + 1)`` power spectrum with a Hann window."""
    wav = preemphasis(np.asarray(wav, dtype=np.float32), preemph)
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frames = frame_signal(wav, frame_len, hop_len).astype(np.float64)
    window = np.hanning(frame_len + 1)[:frame_len]
    frames = frames * window
    if frame_len < n_fft:
        frames = np.pad(frames, ((0, 0), (0, n_fft - frame_len)))
    spec = np.fft.rfft(frames[:, :n_fft], n=n_fft, axis=1)
    return (np.abs(spec) ** 2) / float(n_fft)


def log_mel_spectrogram(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = 40,
    n_fft: int = 512,
    **kwargs,
) -> np.ndarray:
    """``(n_frames, n_mels)`` log-compressed mel energies."""
    power = power_spectrogram(wav, sr=sr, n_fft=n_fft, **kwargs)
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr)
    return np.log(power @ fb.T + EPS)


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II basis -> ``(n_out, n_in)``."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    basis *= np.sqrt(2.0 / n_in)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def mfcc(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    n_mfcc: int = 20,
    n_mels: int = 40,
    n_fft: int = 512,
    **kwargs,
) -> np.ndarray:
    """``(n_frames, n_mfcc)`` cepstral coefficients, C0 included.

    C0 is kept because it is a coarse loudness/spectral-tilt term that still
    differs between speakers once the waveform has been loudness-normalised.
    """
    logmel = log_mel_spectrogram(wav, sr=sr, n_mels=n_mels, n_fft=n_fft, **kwargs)
    return logmel @ dct_matrix(n_mfcc, n_mels).T


def deltas(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-based temporal derivative over +/- ``width`` frames.

    Deltas carry speaking-style information (how fast articulators move) that
    static spectra miss, and they are largely immune to channel colouration.
    """
    if feat.shape[0] == 0:
        return feat.copy()
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    out = np.zeros_like(feat, dtype=np.float64)
    for i in range(1, width + 1):
        out += i * (padded[width + i : padded.shape[0] - width + i] - padded[width - i : padded.shape[0] - width - i])
    return out / denom


def cmvn(feat: np.ndarray, mean_only: bool = False) -> np.ndarray:
    """Cepstral mean (and variance) normalisation -- a channel compensator.

    Convolutional channel effects (microphone, room) are additive in the log
    spectral domain, so subtracting the utterance mean removes most of them. It
    also removes some speaker identity, which is why the DSP encoder keeps both
    normalised and un-normalised statistics.
    """
    if feat.shape[0] == 0:
        return feat.copy()
    out = feat - feat.mean(axis=0, keepdims=True)
    if not mean_only:
        out = out / (feat.std(axis=0, keepdims=True) + EPS)
    return out


# --------------------------------------------------------------------------- #
# Source features (pitch)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class F0Track:
    """Per-frame fundamental frequency with a voiced/unvoiced decision."""

    f0: np.ndarray          # Hz, 0.0 where unvoiced
    voiced: np.ndarray      # bool
    clarity: np.ndarray     # normalised autocorrelation peak in [0, 1]

    @property
    def voiced_f0(self) -> np.ndarray:
        return self.f0[self.voiced]


def estimate_f0(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    fmin: float = 60.0,
    fmax: float = 400.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    clarity_threshold: float = 0.35,
) -> F0Track:
    """Autocorrelation pitch tracker (NCCF).

    Pitch is the single most speaker-discriminative scalar available, and it is
    the one feature a spectral-envelope encoder systematically under-weights, so
    it is estimated separately. The 40 ms window is long enough to hold two
    periods at 60 Hz.
    """
    wav = np.asarray(wav, dtype=np.float64)
    frame_len = max(8, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frames = np.array(frame_signal(wav.astype(np.float32), frame_len, hop_len), dtype=np.float64)
    if frames.shape[0] == 0:
        empty = np.zeros(0)
        return F0Track(empty, empty.astype(bool), empty)

    frames = frames - frames.mean(axis=1, keepdims=True)

    min_lag = max(2, int(sr / fmax))
    max_lag = min(frame_len - 1, int(sr / fmin))
    if max_lag <= min_lag:
        zeros = np.zeros(frames.shape[0])
        return F0Track(zeros, zeros.astype(bool), zeros)

    # Autocorrelation through the frequency domain: O(n log n) instead of O(n^2).
    n_fft = 1 << int(np.ceil(np.log2(2 * frame_len)))
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, n=n_fft, axis=1)[:, : max_lag + 1]

    energy = acf[:, :1]
    nccf = acf / (energy + EPS)

    lags = np.arange(min_lag, max_lag + 1)
    band = nccf[:, min_lag : max_lag + 1]
    best = np.argmax(band, axis=1)
    clarity = band[np.arange(band.shape[0]), best]
    lag = lags[best].astype(np.float64)

    # Parabolic interpolation around the peak for sub-sample lag resolution.
    left = np.clip(best - 1, 0, band.shape[1] - 1)
    right = np.clip(best + 1, 0, band.shape[1] - 1)
    y0 = band[np.arange(band.shape[0]), left]
    y1 = clarity
    y2 = band[np.arange(band.shape[0]), right]
    denom = (y0 - 2.0 * y1 + y2)
    shift = np.where(np.abs(denom) > EPS, 0.5 * (y0 - y2) / np.where(np.abs(denom) > EPS, denom, 1.0), 0.0)
    lag = lag + np.clip(shift, -1.0, 1.0)

    voiced = (clarity >= clarity_threshold) & (energy[:, 0] > EPS)
    f0 = np.where(voiced, sr / np.maximum(lag, EPS), 0.0)
    voiced &= (f0 >= fmin) & (f0 <= fmax)
    f0 = np.where(voiced, f0, 0.0)
    return F0Track(f0=f0, voiced=voiced, clarity=clarity)


# --------------------------------------------------------------------------- #
# Vocal-tract features (LPC)
# --------------------------------------------------------------------------- #

def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Solve the Yule-Walker equations for LPC coefficients.

    Returns ``(a, err)`` where ``a[0] == 1`` and the all-pole model is
    ``1 / sum(a[k] z^-k)``.
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    err = float(r[0])
    if err <= EPS:
        return a, EPS
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1]) if i > 1 else r[i]
        k = -acc / err
        a_prev = a[1:i].copy()
        a[i] = k
        if i > 1:
            a[1:i] = a_prev + k * a_prev[::-1]
        err *= (1.0 - k * k)
        if err <= EPS:
            err = EPS
            break
    return a, err


def lpc_cepstrum(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    order: int = 16,
    n_cep: int = 16,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """``(n_frames, n_cep)`` LPC cepstral coefficients (LPCC).

    LPC models the vocal tract as a resonant tube, so LPCCs describe *anatomy*
    fairly directly. They complement MFCCs, which describe the perceived
    spectrum, and the two disagree in useful ways on the same speaker.
    """
    wav = preemphasis(np.asarray(wav, dtype=np.float32))
    frame_len = max(order * 2, int(round(sr * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    frames = np.array(frame_signal(wav, frame_len, hop_len), dtype=np.float64)
    if frames.shape[0] == 0:
        return np.zeros((0, n_cep))
    frames = frames * np.hanning(frame_len + 1)[:frame_len]

    n_fft = 1 << int(np.ceil(np.log2(2 * frame_len)))
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, n=n_fft, axis=1)[:, : order + 1]

    out = np.zeros((frames.shape[0], n_cep), dtype=np.float64)
    for t in range(frames.shape[0]):
        a, err = levinson_durbin(acf[t], order)
        out[t] = _lpc_to_cepstrum(a, err, n_cep)
    return out


def _lpc_to_cepstrum(a: np.ndarray, err: float, n_cep: int) -> np.ndarray:
    """Recursive LPC -> cepstrum conversion (Atal's relations)."""
    order = a.size - 1
    c = np.zeros(n_cep, dtype=np.float64)
    if n_cep == 0:
        return c
    c[0] = np.log(max(err, EPS))
    for n in range(1, n_cep):
        acc = 0.0
        for k in range(1, min(n, order) + 1):
            acc += (1.0 - k / n) * a[k] * c[n - k]
        c[n] = -a[n] if n <= order else 0.0
        c[n] -= acc
    return c


# --------------------------------------------------------------------------- #
# Aggregate descriptors
# --------------------------------------------------------------------------- #

def spectral_shape(power: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Per-frame ``(centroid, spread, rolloff85, flatness, slope)``.

    Cheap global descriptors of voice *quality* -- breathiness and brightness
    show up here long before they show up in a mel filterbank.
    """
    n_bins = power.shape[1]
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    total = power.sum(axis=1) + EPS

    centroid = (power @ freqs) / total
    spread = np.sqrt((power @ (freqs**2)) / total - centroid**2 + EPS)

    cumulative = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]

    log_p = np.log(power + EPS)
    flatness = np.exp(log_p.mean(axis=1)) / (power.mean(axis=1) + EPS)

    log_f = np.log(freqs + 1.0)
    log_f = log_f - log_f.mean()
    slope = (log_p - log_p.mean(axis=1, keepdims=True)) @ log_f / (np.sum(log_f**2) + EPS)

    return np.stack([centroid, spread, rolloff, flatness, slope], axis=1)


def robust_stats(feat: np.ndarray) -> np.ndarray:
    """Concatenate ``[mean, std, p10, p90]`` over time -> ``(4 * n_dims,)``.

    Percentiles rather than skew/kurtosis: higher moments are extremely noisy on
    the 3-10 second clips this system is built for.
    """
    if feat.shape[0] == 0:
        return np.zeros(4 * feat.shape[1], dtype=np.float64)
    return np.concatenate(
        [
            feat.mean(axis=0),
            feat.std(axis=0),
            np.percentile(feat, 10, axis=0),
            np.percentile(feat, 90, axis=0),
        ]
    ).astype(np.float64)
