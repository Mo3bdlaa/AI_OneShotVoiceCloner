#!/usr/bin/env python3
"""Measure how a so-vits-svc model improves with training.

    python tools/svc_curve.py --workspace voices/svc/me \
        --source vocal.wav --target-dir ~/recordings/me_heldout \
        --source-dir ~/recordings/someone_else \
        --epochs 5 15 40 80 150

Trains in phases, converting and scoring after each, so the output is a curve
rather than a single number. One measurement cannot tell you whether training is
working, only where it happened to be when you stopped.

Scoring uses the ECAPA encoder against a **held-out** set of the target's
recordings -- files the model never trained on. Scoring against the training
audio would measure memorisation.

Two things this exists to answer:

* Where does the number get to, and where does it stop paying? Training is
  expensive enough that "more" is not a plan.
* Is a trained model worth it at all for your material? Compare the result
  against `voxprint revoice --backend knnvc`, which needs no training. For
  speech, kNN-VC usually wins; the case for so-vits-svc is singing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxprint.audio import load_audio  # noqa: E402
from voxprint.encoders import get_encoder  # noqa: E402
from voxprint.svc import SvcWorkspace, infer, set_training_budget, train  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def voice_print(encoder, directory: Path, limit: int = 8) -> np.ndarray:
    """Centroid of a speaker's recordings, L2-normalised."""
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)[:limit]
    if not files:
        raise SystemExit(f"no audio under {directory}")
    vectors = []
    for path in files:
        wav, sr = load_audio(str(path), sr=16000)
        vectors.append(encoder.embed(wav, 16000))
    centroid = np.mean(vectors, axis=0)
    return centroid / np.linalg.norm(centroid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, help="a prepared so-vits-svc workspace")
    parser.add_argument("--speaker", required=True, help="speaker name inside that workspace")
    parser.add_argument("--source", required=True, help="the clip to convert at each phase")
    parser.add_argument("--target-dir", required=True, help="held-out recordings of the target voice")
    parser.add_argument("--source-dir", required=True, help="recordings of the source voice")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 15, 40, 80, 150],
                        help="cumulative epoch targets; training resumes between them")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="only used to label the output; 0 reads it from the filelist")
    parser.add_argument("--out", default="svc_curve.json")
    parser.add_argument("--encoder", default="ecapa")
    args = parser.parse_args()

    workspace = SvcWorkspace(root=Path(args.workspace), speaker=args.speaker)
    if not workspace.config_path.exists():
        raise SystemExit(f"{workspace.config_path} not found; run `voxprint train-svc --prepare-only` first")

    steps_per_epoch = args.steps_per_epoch
    if not steps_per_epoch:
        filelist = workspace.root / "filelists" / "44k" / "train.txt"
        n_files = sum(1 for _ in filelist.open()) if filelist.exists() else 0
        steps_per_epoch = max(1, n_files // max(args.batch_size, 1))

    encoder = get_encoder(args.encoder)
    target = voice_print(encoder, Path(args.target_dir))
    source = voice_print(encoder, Path(args.source_dir))

    def score(path) -> tuple[float, float]:
        wav, sr = load_audio(str(path), sr=16000)
        vec = encoder.embed(wav, 16000)
        return float(vec @ target), float(vec @ source)

    to_target, to_source = score(args.source)
    results = [{"epochs": 0, "steps": 0, "to_target": to_target, "to_source": to_source}]
    print(f"baseline (unconverted): ->target {to_target:+.3f}  ->source {to_source:+.3f}\n", flush=True)

    out_dir = Path(args.out).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_epochs in args.epochs:
        set_training_budget(workspace, epochs=target_epochs, batch_size=args.batch_size)
        started = time.time()
        train(workspace)
        elapsed = time.time() - started

        converted = out_dir / f"epoch{target_epochs:04d}.wav"
        infer(workspace, args.source, converted, auto_predict_f0=False)
        to_target, to_source = score(converted)
        results.append({
            "epochs": target_epochs,
            "steps": target_epochs * steps_per_epoch,
            "to_target": to_target,
            "to_source": to_source,
            "train_seconds": round(elapsed, 1),
        })
        print(f"epoch {target_epochs:>4} (~{target_epochs * steps_per_epoch:>6} steps)  "
              f"->target {to_target:+.3f}  ->source {to_source:+.3f}   +{elapsed / 60:.0f} min", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nwrote {args.out} and {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
