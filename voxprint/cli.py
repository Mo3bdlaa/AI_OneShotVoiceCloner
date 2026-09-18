"""Command-line interface.

Every command is a thin wrapper over :class:`voxprint.pipeline.VoiceLab`; the
argument parsing and the output formatting live here, and nothing else does.
Each command also accepts ``--json`` so the tool composes with scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from . import __version__
from .audio import load_audio, save_audio
from .encoders import available_encoders, get_encoder
from .gallery import ConsentError, ConsentRecord, GalleryError
from .pipeline import VoiceLab
from .synth import available_synths, get_synth
from .watermark import DEFAULT_KEY, DEFAULT_TAG, detect_watermark, embed_watermark

DEFAULT_ROOT = os.environ.get("VOXPRINT_HOME", "voices")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def _emit(payload: Any, as_json: bool, lines: list[str] | None = None) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_encode))
    else:
        for line in lines or []:
            print(line)


def _encode(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _lab(args) -> VoiceLab:
    return VoiceLab(args.root, encoder=args.encoder, require_consent=not args.no_consent_check)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_enroll(args) -> int:
    lab = _lab(args)
    consent = None
    if args.consent:
        consent = ConsentRecord(
            granted_by=args.consent_by or args.id,
            statement=args.consent,
            purpose=args.purpose or "",
        )
    try:
        report = lab.enroll(
            args.id,
            args.files,
            consent=consent,
            display_name=args.name or "",
            notes=args.notes or "",
            keep_audio=not args.no_audio,
            replace=args.replace,
        )
    except ConsentError:
        raise ValueError(
            f"enrolling {args.id!r} requires a consent record. A voice print is biometric data "
            "and can drive the cloning backends, so record who agreed and to what:\n"
            f"    voxprint enroll --id {args.id} --consent \"agreed to voice enrolment on <date>\" ...\n"
            "Use --no-consent-check for synthetic voices and test fixtures."
        ) from None

    lines = [
        f"enrolled {report.speaker_id}: {report.n_embeddings} embeddings from "
        f"{len(report.files)} file(s), {report.seconds:.1f}s of speech"
    ]
    if not np.isnan(report.cohesion):
        lines.append(f"  consistency of enrolment takes: {report.cohesion:.3f}")
    if report.reference_audio:
        lines.append(f"  reference audio: {report.reference_audio}")
    for w in report.warnings:
        lines.append(f"  ! {w}")
    if len(lab.gallery) >= 2 and lab.gallery.threshold is None:
        lines.append("  next: run `voxprint calibrate` to set a decision threshold from measured scores")
    _emit(report.as_dict(), args.json, lines)
    return 0


def cmd_identify(args) -> int:
    lab = _lab(args)
    if len(lab.gallery) == 0:
        _warn("gallery is empty; enrol someone first")
        _emit({"decision": "empty", "candidates": []}, args.json, ["gallery is empty"])
        return 1

    result = lab.identify_file(
        args.file, top_k=args.top, threshold=args.threshold, min_margin=args.min_margin
    )
    lines = []
    if not result.calibrated:
        lines.append("! threshold is an uncalibrated default; run `voxprint calibrate`")
    verdict = {
        "match": f"MATCH: {result.best.speaker_id}",
        "unknown": "UNKNOWN: no enrolled speaker passed the threshold",
        "ambiguous": "AMBIGUOUS: top candidates are too close to call",
        "empty": "gallery is empty",
    }[result.decision]
    lines.append(f"{verdict}   (threshold {result.threshold:.3f}, margin {result.margin:.3f})")
    for i, cand in enumerate(result.candidates, 1):
        prob = f"  p={cand.probability:.2f}" if cand.probability is not None else ""
        lines.append(f"  {i}. {cand.speaker_id:<20} {cand.score:+.4f}{prob}")
    _emit(result.as_dict(), args.json, lines)
    return 0 if result.accepted else 2


def cmd_verify(args) -> int:
    lab = _lab(args)
    result = lab.verify_file(args.file, args.id, threshold=args.threshold)
    lines = []
    if not result.calibrated:
        lines.append("! threshold is an uncalibrated default; run `voxprint calibrate`")
    lines.append(
        f"{'ACCEPT' if result.accepted else 'REJECT'}  {args.id}  "
        f"score {result.score:+.4f}  threshold {result.threshold:.4f}"
    )
    _emit(result.as_dict(), args.json, lines)
    return 0 if result.accepted else 2


def cmd_list(args) -> int:
    lab = _lab(args)
    payload = [p.quality_report() | {"display_name": p.display_name,
                                     "consent": p.consent.as_dict() if p.consent else None}
               for p in lab.speakers()]
    lines = [f"{len(payload)} enrolled speaker(s) in {lab.root}"]
    for entry in payload:
        coh = "n/a" if entry["cohesion"] is None else f"{entry['cohesion']:.3f}"
        consent = "consent recorded" if entry["consent"] else "NO CONSENT RECORD"
        lines.append(
            f"  {entry['speaker_id']:<20} {entry['utterances']:>3} takes  "
            f"{entry['total_seconds']:>6.1f}s  cohesion {coh}  [{consent}]"
        )
        for w in entry["warnings"]:
            lines.append(f"      ! {w}")
    _emit(payload, args.json, lines)
    return 0


def cmd_remove(args) -> int:
    lab = _lab(args)
    lab.remove(args.id)
    _emit({"removed": args.id}, args.json, [f"removed {args.id} (voice print and reference audio)"])
    return 0


def cmd_calibrate(args) -> int:
    lab = _lab(args)
    if len(lab.gallery) < 2:
        _warn("calibration needs at least two enrolled speakers")
        return 1
    result = lab.calibrate(criterion=args.criterion, max_far=args.max_far)
    info = result.as_dict()
    lines = [
        f"threshold {info['threshold']}  (criterion: {info['criterion']})",
        f"  equal error rate      {info['eer']:.4f}",
        f"  false accepts at thr. {info['far_at_threshold']}",
        f"  false rejects at thr. {info['frr_at_threshold']}",
        f"  trials: {info['target_pairs']} same-speaker, {info['impostor_pairs']} impostor",
        "  note: measured on enrolment audio only, so real-world error will be higher",
    ]
    _emit(info, args.json, lines)
    return 0


def cmd_convert(args) -> int:
    lab = _lab(args)
    result = lab.convert_file(
        args.file,
        speaker_id=args.id,
        reference_path=args.reference,
        backend=args.backend,
        watermark=not args.no_watermark,
    )
    save_audio(args.out, result.wav, result.sample_rate)
    payload = {"out": args.out, "backend": result.backend,
               "duration": round(result.duration, 2), **result.info}
    lines = [f"wrote {args.out}  ({result.duration:.1f}s, backend {result.backend})"]
    if "semitones" in result.info:
        lines.append(
            f"  pitch shifted {result.info['semitones']:+.1f} semitones "
            f"({result.info['source_f0']:.0f} Hz -> {result.info['reference_f0']:.0f} Hz target)"
        )
    if args.id and args.id in lab.gallery:
        score = lab.similarity_to(result.wav, result.sample_rate, args.id)
        payload["similarity_to_target"] = round(score, 4)
        lines.append(f"  similarity of the output to the target's voice print: {score:+.3f}")
        if lab.gallery.threshold is not None:
            passes = score >= lab.gallery.threshold
            lines.append(
                f"  {'would' if passes else 'would NOT'} be recognised as {args.id} "
                f"(threshold {lab.gallery.threshold:.3f})"
            )
    _emit(payload, args.json, lines)
    return 0


def cmd_speak(args) -> int:
    lab = _lab(args)
    result = lab.speak(
        args.text,
        speaker_id=args.id,
        reference_path=args.reference,
        backend=args.backend,
        language=args.language,
        watermark=not args.no_watermark,
    )
    save_audio(args.out, result.wav, result.sample_rate)
    payload = {"out": args.out, "backend": result.backend,
               "duration": round(result.duration, 2), **result.info}
    lines = [f"wrote {args.out}  ({result.duration:.1f}s, backend {result.backend})"]
    if args.id and args.id in lab.gallery:
        score = lab.similarity_to(result.wav, result.sample_rate, args.id)
        payload["similarity_to_target"] = round(score, 4)
        lines.append(f"  similarity of the output to the target's voice print: {score:+.3f}")
    _emit(payload, args.json, lines)
    return 0


def cmd_watermark(args) -> int:
    wav, sr = load_audio(args.file)
    if args.action == "add":
        if not args.out:
            _warn("`watermark add` needs --out")
            return 1
        save_audio(args.out, embed_watermark(wav, key=args.key, tag=args.tag), sr)
        _emit({"out": args.out, "tag": args.tag}, args.json, [f"wrote watermarked audio to {args.out}"])
        return 0

    detection = detect_watermark(wav, key=args.key, tag=args.tag)
    lines = [
        f"{'DETECTED' if detection.present else 'not detected'}  "
        f"(z={detection.confidence:.2f}, {detection.blocks} blocks)"
    ]
    if not detection.present:
        lines.append("  absence proves nothing: the mark does not survive heavy processing")
    _emit(detection.as_dict(), args.json, lines)
    return 0 if detection.present else 2


def cmd_backends(args) -> int:
    encoders = []
    for name in available_encoders():
        try:
            encoders.append(get_encoder(name).describe())
        except Exception as exc:  # a backend may fail to construct without its deps
            encoders.append({"name": name, "error": str(exc)})
    synths = [get_synth(name).describe() for name in available_synths()]

    lines = ["encoders (voice fingerprinting):"]
    for info in encoders:
        status = info.get("error", f"{info.get('dim')} dims @ {info.get('sample_rate')} Hz")
        lines.append(f"  {info['name']:<14} {status}")
    lines.append("")
    lines.append("synthesis backends (imitation):")
    for info in synths:
        caps = "+".join(info["capabilities"])
        lines.append(f"  {info['name']:<14} [{caps:<7}] {'ready' if info['ready'] else info['status']}")
        if info["notes"]:
            lines.append(f"                 {info['notes']}")
    _emit({"encoders": encoders, "synthesizers": synths}, args.json, lines)
    return 0


def cmd_info(args) -> int:
    lab = _lab(args)
    info = lab.summary()
    lines = [
        f"root           {info['root']}",
        f"encoder        {info['encoder']['name']} v{info['encoder']['version']} ({info['encoder']['dim']} dims)",
        f"speakers       {info['speakers']}",
        f"utterances     {info['utterances']}",
        f"audio enrolled {info['total_seconds']:.1f}s",
        f"threshold      {info['threshold'] if info['threshold'] is not None else 'not calibrated'}",
        f"standardizer   {info['standardizer']['source']} (n={info['standardizer']['n_samples']})",
    ]
    _emit(info, args.json, lines)
    return 0


def cmd_selftest(args) -> int:
    """Run the pipeline on procedurally generated voices and report real numbers."""
    from .evaluate import run_selftest

    report = run_selftest(encoder=args.encoder, n_speakers=args.speakers, seconds=args.seconds)
    lines = [
        f"encoder {report['encoder']}  |  {report['n_speakers']} synthetic speakers, "
        f"{report['seconds']:.0f}s enrolment each",
        f"  equal error rate        {report['eer']:.4f}",
        f"  threshold (EER)         {report['threshold']:.4f}",
        f"  closed-set accuracy     {report['closed_set_accuracy']:.3f}",
        f"  open-set rejection      {report['open_set_rejection']:.3f}",
        f"  same-speaker score      {report['target_mean']:+.3f}",
        f"  impostor score          {report['impostor_mean']:+.3f}",
        "",
        "  These are synthetic voices with matched recording conditions -- an upper",
        "  bound. Real speech over real microphones scores materially worse.",
    ]
    _emit(report, args.json, lines)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxprint",
        description="Fingerprint a voice, recognise it later, and imitate it.",
    )
    parser.add_argument("--version", action="version", version=f"voxprint {__version__}")
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"gallery directory (default: {DEFAULT_ROOT})")
    parser.add_argument("--encoder", default="dsp", choices=available_encoders(), help="voice-print encoder")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--no-consent-check",
        action="store_true",
        help="allow enrolment without a consent record (for test fixtures and synthetic voices)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enroll", help="register a speaker from one or more recordings")
    p.add_argument("files", nargs="+")
    p.add_argument("--id", required=True, help="speaker identifier")
    p.add_argument("--name", help="display name")
    p.add_argument("--consent", help="what the speaker agreed to (required unless --no-consent-check)")
    p.add_argument("--consent-by", help="who granted it, if not the speaker")
    p.add_argument("--purpose", help="what the voice print will be used for")
    p.add_argument("--notes")
    p.add_argument("--no-audio", action="store_true", help="do not keep reference audio (disables cloning)")
    p.add_argument("--replace", action="store_true", help="overwrite instead of appending takes")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("identify", help="find which enrolled speaker a recording belongs to")
    p.add_argument("file")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--threshold", type=float)
    p.add_argument("--min-margin", type=float, default=0.0,
                   help="reject as ambiguous when the top two are closer than this")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("verify", help="check a claimed identity")
    p.add_argument("file")
    p.add_argument("--id", required=True)
    p.add_argument("--threshold", type=float)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("list", help="list enrolled speakers")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("remove", help="delete a speaker and their audio")
    p.add_argument("--id", required=True)
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("calibrate", help="set the decision threshold from measured scores")
    p.add_argument("--criterion", default="eer", choices=("eer", "far"))
    p.add_argument("--max-far", type=float, default=0.01, help="target false-accept rate for --criterion far")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("convert", help="re-voice an existing recording toward a target voice")
    p.add_argument("file")
    p.add_argument("--id", help="enrolled target speaker")
    p.add_argument("--reference", help="target voice as an audio file instead")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--backend", default="dspvc", choices=available_synths())
    p.add_argument("--no-watermark", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("speak", help="synthesise text in a target voice (needs a neural backend)")
    p.add_argument("text")
    p.add_argument("--id", help="enrolled target speaker")
    p.add_argument("--reference", help="target voice as an audio file instead")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--backend", default="xtts", choices=available_synths())
    p.add_argument("--language", default="en")
    p.add_argument("--no-watermark", action="store_true")
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("watermark", help="add or check the synthetic-audio marker")
    p.add_argument("action", choices=("add", "check"))
    p.add_argument("file")
    p.add_argument("-o", "--out")
    p.add_argument("--key", default=DEFAULT_KEY)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.set_defaults(func=cmd_watermark)

    p = sub.add_parser("backends", help="show available encoders and synthesis backends")
    p.set_defaults(func=cmd_backends)

    p = sub.add_parser("info", help="summarise the gallery")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("selftest", help="measure accuracy on procedurally generated voices")
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--seconds", type=float, default=4.0)
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (GalleryError, ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
