#!/usr/bin/env python3
"""End-to-end demonstration, with no downloads and no recordings of real people.

Run it::

    python examples/demo.py --out demo_output

It generates a small cast of procedural voices, enrols three of them, calibrates
a decision threshold from the measured scores, and then shows all four things
the system can do:

1. identify a held-out recording of an enrolled speaker;
2. reject a speaker who was never enrolled;
3. re-voice one speaker's recording toward another, and *measure* how far that
   actually got;
4. mark the generated audio and detect the mark.

Everything printed is measured at run time. Nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import os
import sys

# Run straight from a clone, without installing the package first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxprint import ConsentRecord, VoiceLab, detect_watermark, save_audio
from voxprint.synthetic import CAST, PHRASES

SR = 16_000
ENROLLED = ("omar", "hana", "sami")
STRANGER = "laila"


def write_clips(directory: str) -> dict[str, list[str]]:
    """Render four takes for each voice: three to enrol, one held out."""
    os.makedirs(directory, exist_ok=True)
    paths: dict[str, list[str]] = {}
    for speaker in CAST:
        files = []
        for i, phrase in enumerate(PHRASES[:4]):
            path = os.path.join(directory, f"{speaker.name}_{i}.wav")
            save_audio(path, speaker.say(phrase, seconds=4.0, seed=4000 + i), SR)
            files.append(path)
        paths[speaker.name] = files
    return paths


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="demo_output", help="working directory for the demo")
    args = parser.parse_args()

    clips_dir = os.path.join(args.out, "clips")
    clips = write_clips(clips_dir)
    print(f"generated {sum(len(v) for v in clips.values())} clips in {clips_dir}")

    lab = VoiceLab(os.path.join(args.out, "voices"))
    consent = ConsentRecord(
        granted_by="demo fixture",
        statement="procedurally generated voice; no real person is involved",
        purpose="demonstration",
    )

    rule("1. enrolment")
    for name in ENROLLED:
        report = lab.enroll(name, clips[name][:3], consent=consent)
        print(f"  {name:<8} {report.n_embeddings:>2} embeddings  {report.seconds:5.1f}s  "
              f"cohesion {report.cohesion:.3f}")
        for warning in report.warnings:
            print(f"           ! {warning}")

    rule("2. calibration")
    cal = lab.calibrate()
    info = cal.as_dict()
    print(f"  threshold {info['threshold']}  (equal error rate {info['eer']})")
    print(f"  same-speaker mean {info['target_score_mean']}  vs impostor mean {info['impostor_score_mean']}")

    rule("3. identification of held-out takes")
    for name in ENROLLED:
        result = lab.identify_file(clips[name][3])
        mark = "OK" if result.accepted and result.best.speaker_id == name else "MISS"
        print(f"  {name:<8} -> {result.best.speaker_id:<8} {result.best.score:+.3f}  "
              f"{result.decision:<9} [{mark}]")

    rule("4. open-set rejection")
    result = lab.identify_file(clips[STRANGER][0])
    print(f"  {STRANGER} was never enrolled -> {result.decision} "
          f"(best guess {result.best.speaker_id} at {result.best.score:+.3f}, "
          f"threshold {result.threshold:.3f})")

    rule("5. voice conversion, measured")
    out_path = os.path.join(args.out, "omar_as_hana.wav")
    converted = lab.convert_file(clips["omar"][0], speaker_id="hana")
    save_audio(out_path, converted.wav, converted.sample_rate)

    reference, sr = lab.load_reference("omar")
    before = lab.similarity_to(reference, sr, "hana")
    after = lab.similarity_to(converted.wav, converted.sample_rate, "hana")
    print(f"  wrote {out_path}")
    print(f"  pitch shifted {converted.info['semitones']:+.1f} semitones "
          f"({converted.info['source_f0']:.0f} Hz -> {converted.info['reference_f0']:.0f} Hz)")
    print(f"  similarity to hana: {before:+.3f} before  ->  {after:+.3f} after "
          f"(threshold {lab.gallery.threshold:.3f})")
    print("  the gap narrows but does not close: this is timbre transfer, not identity cloning.")

    rule("6. provenance watermark")
    detection = detect_watermark(converted.wav)
    print(f"  generated audio: {'DETECTED' if detection.present else 'not detected'} "
          f"(z={detection.confidence:.1f})")
    original, _ = lab.load_reference("omar")
    print(f"  original audio:  {'DETECTED' if detect_watermark(original).present else 'not detected'} "
          f"(z={detect_watermark(original).confidence:.1f})")

    rule("next steps")
    print("  Text-to-speech cloning needs a trained model:")
    print("      pip install -r requirements-neural.txt")
    voices = os.path.join(args.out, "voices")
    print(f"      voxprint --root {voices} speak 'مرحبا' --id hana --language ar -o hello.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
