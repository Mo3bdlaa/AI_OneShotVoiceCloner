"""Procedurally generated voices.

Real speaker-recognition corpora cannot be shipped in a repository, so voices are
built here from a source-filter model instead: a jittered glottal pulse train
(the "source", i.e. the vocal folds) driving a cascade of resonators (the
"filter", i.e. the vocal tract). A speaker is then a fixed pitch register plus a
fixed set of formant frequencies -- exactly the properties a voice print is
supposed to capture.

Two things depend on this module:

* the test suite, which renders the same phoneme sequence through different
  speakers to check that the encoder keys on *who* is speaking rather than on
  *what* is being said;
* ``tools/build_reference.py``, which embeds a large bank of random voices to
  estimate the reference statistics used by
  :mod:`voxprint.normalization`.

These are not real voices and no claim is made that they model human variation
faithfully. They model it well enough to be a usable prior and a deterministic,
dependency-free test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16_000

# Formant targets for five vowels, as multipliers applied to a speaker's own
# formant set. Shared across speakers so "the same sentence" is reproducible.
VOWEL_SCALES = {
    "a": (1.00, 1.00, 1.00),
    "i": (0.45, 1.85, 1.15),
    "u": (0.42, 0.72, 0.95),
    "e": (0.72, 1.50, 1.05),
    "o": (0.62, 0.78, 0.98),
}


@dataclass
class SyntheticSpeaker:
    """A reproducible artificial voice."""

    name: str
    f0: float                                    # mean pitch, Hz
    formants: tuple[float, float, float]         # F1, F2, F3 for the neutral vowel
    bandwidths: tuple[float, float, float] = (90.0, 110.0, 160.0)
    jitter: float = 0.012                        # cycle-to-cycle pitch instability
    breathiness: float = 0.02                    # aspiration noise mixed into the source
    tilt: float = 0.0                            # extra spectral slope, dB/kHz
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def say(self, phonemes: str = "aiueo", seconds: float = 3.0, sr: int = SR, seed: int | None = None) -> np.ndarray:
        """Render a vowel sequence in this speaker's voice."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        segments = [c for c in phonemes if c in VOWEL_SCALES]
        if not segments:
            segments = ["a"]
        seg_len = max(1, int(sr * seconds / len(segments)))

        out = []
        for ch in segments:
            scale = VOWEL_SCALES[ch]
            targets = tuple(f * s for f, s in zip(self.formants, scale, strict=True))
            out.append(self._segment(targets, seg_len, sr, rng))
        wav = np.concatenate(out)

        # A short fade stops the concatenation clicks from becoming features.
        fade = int(0.005 * sr)
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
        return _normalize(wav)

    # -- source-filter internals ------------------------------------------- #

    def _segment(self, formants, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
        source = self._glottal_source(n, sr, rng)
        sig = source
        for freq, bw in zip(formants, self.bandwidths, strict=True):
            sig = _resonator(sig, freq, bw, sr)
        if self.tilt:
            sig = _apply_tilt(sig, self.tilt, sr)
        return _normalize(sig.astype(np.float32))

    def _glottal_source(self, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
        """Jittered pulse train plus aspiration noise."""
        sig = np.zeros(n, dtype=np.float64)
        pos = 0.0
        while pos < n:
            i = int(pos)
            if i < n:
                sig[i] = 1.0
            period = sr / max(self.f0 * (1.0 + self.jitter * rng.standard_normal()), 20.0)
            pos += period
        # A one-pole low-pass approximates the glottal flow derivative's tilt.
        sig = _one_pole(sig, 0.92)
        sig += self.breathiness * rng.standard_normal(n)
        return sig


def _resonator(x: np.ndarray, freq: float, bandwidth: float, sr: int) -> np.ndarray:
    """Two-pole IIR resonator -- one formant."""
    from scipy.signal import lfilter

    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * freq / sr
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    b = [(1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * theta) + r * r)]
    return lfilter(b, a, x)


def _one_pole(x: np.ndarray, coeff: float) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter([1.0 - coeff], [1.0, -coeff], x)


def _apply_tilt(x: np.ndarray, db_per_khz: float, sr: int) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    gain = 10.0 ** (db_per_khz * freqs / 1000.0 / 20.0)
    return np.fft.irfft(spec * gain, n=x.size)


def _normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak * 0.9).astype(np.float32) if peak > 1e-9 else x.astype(np.float32)


def random_speaker(seed: int) -> SyntheticSpeaker:
    """Draw a voice from plausible ranges -- used to build the reference bank."""
    rng = np.random.default_rng(seed)
    return SyntheticSpeaker(
        name=f"rand{seed}",
        f0=float(np.exp(rng.uniform(np.log(80.0), np.log(260.0)))),
        formants=(
            float(rng.uniform(430.0, 830.0)),
            float(rng.uniform(1030.0, 2080.0)),
            float(rng.uniform(2180.0, 3250.0)),
        ),
        bandwidths=(
            float(rng.uniform(60.0, 130.0)),
            float(rng.uniform(80.0, 160.0)),
            float(rng.uniform(110.0, 220.0)),
        ),
        jitter=float(rng.uniform(0.004, 0.028)),
        breathiness=float(rng.uniform(0.0, 0.07)),
        tilt=float(rng.uniform(-2.5, 2.5)),
        seed=seed,
    )


#: A small, deliberately varied cast: two low voices, two high, one mid.
CAST: list[SyntheticSpeaker] = [
    SyntheticSpeaker("omar", f0=104.0, formants=(560.0, 1180.0, 2480.0), jitter=0.010, seed=11),
    SyntheticSpeaker("hana", f0=212.0, formants=(720.0, 1680.0, 2920.0), jitter=0.016, breathiness=0.04, seed=22),
    SyntheticSpeaker("sami", f0=126.0, formants=(640.0, 1420.0, 2650.0), jitter=0.008, tilt=-1.5, seed=33),
    SyntheticSpeaker("laila", f0=188.0, formants=(680.0, 1880.0, 3050.0), jitter=0.020, breathiness=0.05, seed=44),
    SyntheticSpeaker("khaled", f0=148.0, formants=(600.0, 1300.0, 2300.0), jitter=0.012, tilt=1.0, seed=55),
]


#: Phoneme sequences standing in for different "sentences".
PHRASES = ("aiueo", "oeuia", "uaeoi", "eioua", "auoie", "iouae")


def cast_by_name() -> dict[str, SyntheticSpeaker]:
    return {s.name: s for s in CAST}


def add_noise(wav: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix in white noise at a given SNR -- used to test robustness claims."""
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(wav.astype(np.float64) ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(wav.size) * np.sqrt(noise_power)
    return _normalize((wav + noise).astype(np.float32))
