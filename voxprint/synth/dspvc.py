"""Signal-processing voice conversion -- the offline, dependency-free backend.

What it does
------------
Two transformations, applied to a recording of someone speaking:

1. **Pitch register.** Shift the source's median F0 onto the target's, using a
   phase vocoder so the timing survives.
2. **Timbre.** Compute the long-term average spectral envelope of both voices,
   and apply the difference as a smooth equalisation curve. That moves the
   source's resonances toward the target's without touching the harmonic fine
   structure, which carries the words.

What it does not do
-------------------
It does not change *how* the person speaks -- rhythm, accent, the small timing
and pitch gestures that make a voice recognisable to a friend all belong to the
source speaker and stay there. It also shifts formants along with the pitch,
which is audible on large shifts.

So: the output is recognisably "the source speaker, re-coloured toward the
target", not the target speaker. The honest way to read it is as a *timbre
transfer*, and it is included because it runs anywhere, in real time, with no
model download -- which makes it a useful baseline, a fallback, and a way to
exercise the whole pipeline in tests. For actual voice cloning, use the
``xtts`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import TARGET_SR, preprocess, rms_normalize
from ..features import estimate_f0
from ..spectral import cepstral_envelope, istft, pitch_shift, stft
from .base import SynthResult, VoiceSynthesizer, register_synth


@dataclass
class DspVcConfig:
    n_fft: int = 1024
    hop: int = 256
    envelope_coeff: int = 28
    #: Cap on the equalisation curve. Beyond roughly 12 dB the correction starts
    #: amplifying noise in bands where the source has no energy to shape.
    max_gain_db: float = 12.0
    #: Cap on the pitch shift, in semitones. A larger shift is possible but the
    #: formant movement that comes with it makes the result sound synthetic.
    max_semitones: float = 9.0
    #: Fraction of the pitch correction to apply; below 1.0 keeps some of the
    #: source speaker's register, which often sounds more natural.
    pitch_strength: float = 1.0
    envelope_strength: float = 1.0


class DspVoiceConverter(VoiceSynthesizer):
    """Pitch-register matching plus long-term spectral-envelope transfer."""

    name = "dspvc"
    version = "1"
    capabilities = frozenset({"vc"})
    sample_rate = TARGET_SR
    notes = "No model download, runs on CPU in real time. Timbre transfer, not identity cloning."

    def __init__(self, config: DspVcConfig | None = None, **kwargs):
        self.config = config or DspVcConfig(**kwargs)

    def convert(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        **kwargs,
    ) -> SynthResult:
        cfg = self.config
        pitch_strength = float(kwargs.pop("pitch_strength", cfg.pitch_strength))
        envelope_strength = float(kwargs.pop("envelope_strength", cfg.envelope_strength))

        src = preprocess(np.asarray(source, dtype=np.float32), sr, do_vad=False)
        ref = preprocess(np.asarray(reference, dtype=np.float32), sr, do_vad=True)
        if src.size < cfg.n_fft or ref.size < cfg.n_fft:
            raise ValueError("source and reference must each be longer than one analysis frame")

        # -- 1. pitch register --------------------------------------------- #
        src_f0 = _median_f0(src, sr)
        ref_f0 = _median_f0(ref, sr)
        if src_f0 > 0 and ref_f0 > 0:
            semitones = 12.0 * np.log2(ref_f0 / src_f0) * pitch_strength
            semitones = float(np.clip(semitones, -cfg.max_semitones, cfg.max_semitones))
        else:
            semitones = 0.0
        factor = 2.0 ** (semitones / 12.0)
        shifted = pitch_shift(src, factor, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop)

        # -- 2. spectral envelope ------------------------------------------ #
        spec = stft(shifted, n_fft=cfg.n_fft, hop=cfg.hop)
        src_env = _average_envelope(shifted, cfg, sr)
        ref_env = _average_envelope(ref, cfg, sr)

        gain_db = (ref_env - src_env) * (20.0 / np.log(10.0)) * envelope_strength
        # Smooth first, then clip: smoothing a clipped curve can overshoot the
        # cap again, so the clip has to be the last thing that touches it.
        gain_db = cepstral_envelope(gain_db[None, :], n_coeff=cfg.envelope_coeff)[0]
        gain_db = np.clip(gain_db, -cfg.max_gain_db, cfg.max_gain_db)
        gain = 10.0 ** (gain_db / 20.0)

        out = istft(spec * gain[None, :], n_fft=cfg.n_fft, hop=cfg.hop, length=shifted.size)
        out = rms_normalize(out, -23.0)

        return SynthResult(
            wav=out,
            sample_rate=sr,
            backend=self.name,
            info={
                "source_f0": round(src_f0, 2),
                "reference_f0": round(ref_f0, 2),
                "semitones": round(semitones, 2),
                "max_gain_db": round(float(np.max(np.abs(gain_db))), 2),
                "pitch_strength": pitch_strength,
                "envelope_strength": envelope_strength,
            },
        )


def _median_f0(wav: np.ndarray, sr: int) -> float:
    track = estimate_f0(wav, sr=sr)
    voiced = track.voiced_f0
    return float(np.median(voiced)) if voiced.size >= 3 else 0.0


def _average_envelope(wav: np.ndarray, cfg: DspVcConfig, sr: int) -> np.ndarray:
    """Long-term average log-magnitude envelope over the energetic frames.

    Silent and near-silent frames are excluded: their spectrum is room noise, and
    including it drags the envelope toward whatever the recording room sounds
    like rather than the speaker.
    """
    spec = stft(wav, n_fft=cfg.n_fft, hop=cfg.hop)
    magnitude = np.abs(spec)
    energy = magnitude.sum(axis=1)
    if energy.size == 0:
        return np.zeros(spec.shape[1])
    keep = energy > np.percentile(energy, 40.0)
    if not keep.any():
        keep = np.ones_like(energy, dtype=bool)
    log_mag = np.log(magnitude[keep] + 1e-9)
    return cepstral_envelope(log_mag.mean(axis=0)[None, :], n_coeff=cfg.envelope_coeff)[0]


@register_synth("dspvc")
def _make_dspvc(**kwargs) -> DspVoiceConverter:
    return DspVoiceConverter(**kwargs)
