"""Measured evaluation on procedurally generated voices.

A project that claims to recognise voices should be able to say how well it does
so, on demand, without a corpus download. :func:`run_selftest` builds a gallery
of synthetic speakers, calibrates it exactly the way a user would, and then
scores *held-out* recordings -- takes that were never part of enrolment and never
touched the standardiser.

The resulting numbers are an upper bound. Synthetic voices are clean, perfectly
matched in recording conditions, and differ from each other in the exact
parameters the encoder was built to measure. Real speech is none of those things.
Read the output as "this is what the pipeline does when nothing goes wrong".
"""

from __future__ import annotations

import numpy as np

from .encoders import get_encoder
from .gallery import Gallery
from .scoring import PlattScaler, calibrate, identify
from .synthetic import PHRASES, random_speaker

ENROLL_PHRASES = PHRASES[:3]
HELDOUT_PHRASES = PHRASES[3:]


def build_gallery(
    encoder,
    n_speakers: int = 12,
    seconds: float = 4.0,
    seed_base: int = 100_000,
) -> tuple[Gallery, list]:
    """Enrol ``n_speakers`` synthetic voices with three takes each."""
    gallery = Gallery(encoder.spec, require_consent=False)
    speakers = []
    for i in range(n_speakers):
        speaker = random_speaker(seed=seed_base + i)
        vecs = [
            encoder.embed(speaker.say(phrase, seconds, seed=seed_base + 1000 * (j + 1) + i))
            for j, phrase in enumerate(ENROLL_PHRASES)
        ]
        gallery.enroll(f"spk{i:02d}", np.stack(vecs), seconds=seconds * len(ENROLL_PHRASES))
        speakers.append(speaker)
    return gallery, speakers


def run_selftest(
    encoder: str = "dsp",
    n_speakers: int = 12,
    seconds: float = 4.0,
    n_strangers: int = 12,
) -> dict:
    """Enrol, calibrate, then measure closed-set accuracy and open-set rejection."""
    enc = get_encoder(encoder)
    gallery, speakers = build_gallery(enc, n_speakers=n_speakers, seconds=seconds)

    # The user's own workflow: refit the reference statistics on the enrolled
    # population, then calibrate a threshold from the enrolment scores.
    gallery.refit_standardizer()
    cal = calibrate(gallery)
    gallery.threshold = cal.threshold
    scaler = PlattScaler.fit(cal.target_scores, cal.impostor_scores)

    # Held-out takes: different phrases, different seeds, never seen before.
    hits = 0
    trials = 0
    for i, speaker in enumerate(speakers):
        for j, phrase in enumerate(HELDOUT_PHRASES):
            wav = speaker.say(phrase, seconds, seed=900_000 + 17 * i + j)
            result = identify(enc.embed(wav), gallery, scaler=scaler)
            trials += 1
            hits += int(result.accepted and result.best.speaker_id == f"spk{i:02d}")

    # Strangers: voices that were never enrolled must be rejected.
    rejected = 0
    for s in range(n_strangers):
        stranger = random_speaker(seed=700_000 + s)
        wav = stranger.say(PHRASES[s % len(PHRASES)], seconds, seed=800_000 + s)
        rejected += int(not identify(enc.embed(wav), gallery).accepted)

    return {
        "encoder": f"{enc.name} v{enc.version}",
        "dim": enc.dim,
        "n_speakers": n_speakers,
        "seconds": seconds,
        "eer": round(cal.eer, 4),
        "threshold": round(cal.threshold, 4),
        "closed_set_accuracy": round(hits / max(trials, 1), 4),
        "closed_set_trials": trials,
        "open_set_rejection": round(rejected / max(n_strangers, 1), 4),
        "open_set_trials": n_strangers,
        "target_mean": round(float(cal.target_scores.mean()), 4) if cal.n_target else None,
        "impostor_mean": round(float(cal.impostor_scores.mean()), 4) if cal.n_impostor else None,
    }


def conversion_report(backend: str = "dspvc", seconds: float = 3.0) -> dict:
    """Quantify what a conversion backend actually achieves.

    Reports the target similarity of the source before and after conversion.
    Whether the gap closes, and by how much, is the only meaningful answer to
    "does this clone the voice?" -- and it is far more informative than listening
    once and forming an impression.
    """
    from .normalization import load_default_reference
    from .synth import get_synth

    enc = get_encoder("dsp")
    std = load_default_reference(enc.spec)
    synth = get_synth(backend)

    source_speaker = random_speaker(seed=310)
    target_speaker = random_speaker(seed=420)
    source = source_speaker.say(PHRASES[0], seconds, seed=1)
    target = target_speaker.say(PHRASES[1], seconds, seed=2)

    result = synth.convert(source, target, enc.sample_rate)
    embed = lambda w: std.transform(enc.embed(w, enc.sample_rate))  # noqa: E731
    e_src, e_tgt, e_out = embed(source), embed(target), embed(result.wav)

    return {
        "backend": backend,
        "before": round(float(e_src @ e_tgt), 4),
        "after": round(float(e_out @ e_tgt), 4),
        "drift_from_source": round(float(e_out @ e_src), 4),
        "info": result.info,
    }
