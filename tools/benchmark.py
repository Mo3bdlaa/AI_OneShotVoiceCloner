#!/usr/bin/env python3
"""Reproduce every number quoted in the README and the docs.

    python tools/benchmark.py                  # dsp only
    python tools/benchmark.py --encoders dsp ecapa

Each section prints a table that appears, in the same form, in the documentation.
Running this is how those tables are kept honest: if a change moves a number, the
change shows up here rather than in a reader's disappointment.

Every voice comes from :mod:`voxprint.synthetic`, and enrolled speakers and
strangers are drawn from one mutually distinct pool -- see
:func:`voxprint.synthetic.distinct_speakers` for why that matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxprint.augment import add_noise  # noqa: E402
from voxprint.encoders import get_encoder  # noqa: E402
from voxprint.evaluate import build_gallery  # noqa: E402
from voxprint.normalization import load_default_reference  # noqa: E402
from voxprint.scoring import calibrate, identify  # noqa: E402
from voxprint.synthetic import PHRASES, SR, distinct_speakers  # noqa: E402

HELD_OUT = PHRASES[3:]


def _held_out_queries(speakers, seconds):
    for i, speaker in enumerate(speakers):
        for j, phrase in enumerate(HELD_OUT):
            yield i, speaker.say(phrase, seconds, seed=900_000 + 17 * i + j)


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


# --------------------------------------------------------------------------- #

def bench_accuracy(encoder_name: str, n_speakers: int, n_strangers: int, seconds: float) -> dict:
    """Clean closed-set accuracy, open-set rejection, and the score distributions."""
    enc = get_encoder(encoder_name)
    pool = distinct_speakers(n_speakers + n_strangers, seed=1)
    speakers, strangers = pool[:n_speakers], pool[n_speakers:]
    gallery, speakers = build_gallery(enc, n_speakers=n_speakers, seconds=seconds, speakers=speakers)
    cal = calibrate(gallery)
    gallery.threshold = cal.threshold

    hits = trials = 0
    for i, wav in _held_out_queries(speakers, seconds):
        result = identify(enc.embed(wav), gallery)
        trials += 1
        hits += int(result.accepted and result.best.speaker_id == f"spk{i:02d}")

    rejected = sum(
        int(not identify(enc.embed(s.say(PHRASES[k % len(PHRASES)], seconds, seed=800_000 + k)), gallery).accepted)
        for k, s in enumerate(strangers)
    )
    return {
        "encoder": enc.name,
        "eer": cal.eer,
        "threshold": cal.threshold,
        "closed_set": hits / max(trials, 1),
        "open_set": rejected / max(len(strangers), 1),
        "target_mean": float(cal.target_scores.mean()),
        "impostor_mean": float(cal.impostor_scores.mean()),
    }


def bench_noise(encoder_name: str, n_speakers: int, seconds: float) -> list[dict]:
    """Ranking versus acceptance as the query degrades -- the key diagnostic."""
    enc = get_encoder(encoder_name)
    gallery, speakers = build_gallery(
        enc, n_speakers=n_speakers, seconds=seconds, speakers=distinct_speakers(n_speakers, seed=1)
    )
    cal = calibrate(gallery)
    gallery.threshold = cal.threshold

    rows = []
    conditions = [("clean", None)] + [(f"{s} dB SNR", float(s)) for s in (40, 30, 20, 10)]
    for label, snr in conditions:
        rank1 = accepted = trials = 0
        for i, wav in _held_out_queries(speakers, seconds):
            query = wav if snr is None else add_noise(wav, snr, SR, "pink", seed=3)
            result = identify(enc.embed(query), gallery, threshold=-2.0)
            hit = result.best.speaker_id == f"spk{i:02d}"
            trials += 1
            rank1 += hit
            accepted += int(hit and result.best.score >= cal.threshold)
        rows.append({"condition": label, "rank1": rank1 / trials, "accepted": accepted / trials})
    return rows


def bench_refit(encoder_name: str, n_speakers: int, n_strangers: int, seconds: float) -> list[dict]:
    """Does refitting the standardiser on the gallery help or hurt at this size?"""
    enc = get_encoder(encoder_name)
    pool = distinct_speakers(n_speakers + n_strangers, seed=1)
    speakers, strangers = pool[:n_speakers], pool[n_speakers:]

    rows = []
    for refit in (False, True):
        gallery, speakers = build_gallery(enc, n_speakers=n_speakers, seconds=seconds, speakers=speakers)
        gallery.standardizer = load_default_reference(enc.spec)
        fitted = gallery.refit_standardizer(min_speakers=0, min_samples=0) if refit else False
        cal = calibrate(gallery)
        gallery.threshold = cal.threshold

        hits = trials = 0
        for i, wav in _held_out_queries(speakers, seconds):
            result = identify(enc.embed(wav), gallery)
            trials += 1
            hits += int(result.accepted and result.best.speaker_id == f"spk{i:02d}")
        false_accepts = sum(
            int(identify(enc.embed(s.say(PHRASES[k % len(PHRASES)], seconds, seed=800_000 + k)), gallery).accepted)
            for k, s in enumerate(strangers)
        )
        rows.append({
            "reference": "gallery" if fitted else "packaged",
            "threshold": cal.threshold,
            "closed_set": hits / max(trials, 1),
            "false_accepts": false_accepts,
            "strangers": len(strangers),
        })
    return rows


def bench_conversion() -> dict:
    """How far the signal-processing converter actually moves the voice print."""
    from voxprint.synth import get_synth

    enc = get_encoder("dsp")
    std = load_default_reference(enc.spec)
    source_speaker, target_speaker = distinct_speakers(2, seed=31)
    source = source_speaker.say(PHRASES[0], 3.0, seed=1)
    target = target_speaker.say(PHRASES[1], 3.0, seed=2)
    out = get_synth("dspvc").convert(source, target, SR).wav

    embed = lambda w: std.transform(enc.embed(w, SR))  # noqa: E731
    e_src, e_tgt, e_out = embed(source), embed(target), embed(out)
    return {
        "before": float(e_src @ e_tgt),
        "after": float(e_out @ e_tgt),
        "drift_from_source": float(e_out @ e_src),
    }


def bench_watermark() -> dict:
    """Detection margin, and the conditions that destroy the mark."""
    from voxprint.audio import resample
    from voxprint.watermark import detect_watermark, embed_watermark

    clean_peaks, marked, phone, band12 = [], [], [], []
    for k, speaker in enumerate(distinct_speakers(10, seed=4)):
        clip = speaker.say("aiueoaiueo", 6.0, seed=k)
        mark = embed_watermark(clip)
        clean_peaks.append(detect_watermark(clip).confidence)
        marked.append(detect_watermark(mark).confidence)
        phone.append(detect_watermark(resample(resample(mark, SR, 8000), 8000, SR)).present)
        band12.append(detect_watermark(resample(resample(mark, SR, 12000), 12000, SR)).present)
    return {
        "unmarked_peak_z": max(clean_peaks),
        "marked_min_z": min(marked),
        "survives_12khz": f"{sum(band12)}/10",
        "survives_8khz": f"{sum(phone)}/10",
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoders", nargs="+", default=["dsp"])
    parser.add_argument("--speakers", type=int, default=12)
    parser.add_argument("--strangers", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--skip", nargs="*", default=[], choices=["accuracy", "noise", "refit", "conversion", "watermark"])
    args = parser.parse_args()

    if "accuracy" not in args.skip:
        section(f"Clean accuracy ({args.speakers} enrolled, {args.strangers} strangers, {args.seconds:.0f}s takes)")
        print(f"  {'encoder':<8} {'EER':>7} {'thresh':>8} {'closed':>8} {'open-set':>9} {'target':>8} {'impostor':>9}")
        for name in args.encoders:
            r = bench_accuracy(name, args.speakers, args.strangers, args.seconds)
            print(f"  {r['encoder']:<8} {r['eer']:>7.4f} {r['threshold']:>8.3f} {r['closed_set']:>8.3f} "
                  f"{r['open_set']:>9.3f} {r['target_mean']:>+8.3f} {r['impostor_mean']:>+9.3f}")

    if "noise" not in args.skip:
        section("Noise: ranking degrades far more slowly than acceptance")
        for name in args.encoders:
            print(f"  {name}:")
            print(f"    {'condition':<12} {'rank-1':>8} {'accepted':>10}")
            for row in bench_noise(name, args.speakers, args.seconds):
                print(f"    {row['condition']:<12} {row['rank1']:>8.3f} {row['accepted']:>10.3f}")

    if "refit" not in args.skip:
        section(f"Standardiser reference: packaged prior vs refit on {args.speakers} speakers")
        for name in args.encoders:
            print(f"  {name}:")
            print(f"    {'reference':<10} {'thresh':>8} {'closed':>8} {'false accepts':>15}")
            for row in bench_refit(name, args.speakers, args.strangers, args.seconds):
                print(f"    {row['reference']:<10} {row['threshold']:>8.3f} {row['closed_set']:>8.3f} "
                      f"{row['false_accepts']:>10}/{row['strangers']}")

    if "conversion" not in args.skip:
        section("Signal-processing voice conversion (similarity to the target)")
        r = bench_conversion()
        print(f"  before {r['before']:+.3f}  ->  after {r['after']:+.3f}  "
              f"(drift from source {r['drift_from_source']:+.3f})")

    if "watermark" not in args.skip:
        section("Watermark detection margin")
        r = bench_watermark()
        print(f"  unmarked peak z {r['unmarked_peak_z']:.2f} | marked min z {r['marked_min_z']:.2f}")
        print(f"  survives 12 kHz band-limiting: {r['survives_12khz']} | telephone 8 kHz: {r['survives_8khz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
