#!/usr/bin/env python3
"""Rebuild the packaged reference statistics for an encoder.

Usage
-----
    python tools/build_reference.py --encoder dsp --voices 400

The reference is a per-dimension mean and standard deviation estimated over a
large, varied population of voices. :mod:`voxprint.normalization` uses it to
spread embeddings out before cosine scoring; without it the DSP encoder's scores
all sit within a hundredth of 1.0 and no threshold can separate them.

By default the population is procedurally generated (see
:mod:`voxprint.synthetic`). That is a *prior*, not a substitute for real data --
if you have a corpus of real speakers, point ``--audio-dir`` at it and the
statistics will be estimated from that instead, which is strictly better.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxprint.audio import load_audio  # noqa: E402
from voxprint.encoders import get_encoder  # noqa: E402
from voxprint.normalization import EmbeddingStandardizer, reference_path  # noqa: E402
from voxprint.synthetic import PHRASES, random_speaker  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def synthetic_embeddings(encoder, n_voices: int, seconds: float, verbose: bool) -> np.ndarray:
    vecs = []
    for i in range(n_voices):
        speaker = random_speaker(seed=10_000 + i)
        phrase = PHRASES[i % len(PHRASES)]
        wav = speaker.say(phrase, seconds=seconds, seed=20_000 + i)
        vecs.append(encoder.embed(wav, encoder.sample_rate))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_voices} voices embedded", file=sys.stderr)
    return np.stack(vecs)


def corpus_embeddings(encoder, audio_dir: str, limit: int, verbose: bool) -> np.ndarray:
    files = sorted(
        p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES
    )[:limit]
    if not files:
        raise SystemExit(f"no audio files found under {audio_dir}")
    vecs = []
    for i, path in enumerate(files):
        try:
            wav, sr = load_audio(path, sr=encoder.sample_rate)
            vecs.append(encoder.embed(wav, sr))
        except Exception as exc:  # a corpus always has a few unreadable files
            print(f"  skipping {path}: {exc}", file=sys.stderr)
            continue
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} files embedded", file=sys.stderr)
    return np.stack(vecs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="dsp")
    parser.add_argument("--voices", type=int, default=400, help="number of synthetic voices")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--audio-dir", default=None, help="use a real corpus instead of synthetic voices")
    parser.add_argument("--limit", type=int, default=2000, help="max files to read from --audio-dir")
    parser.add_argument("--out", default=None, help="output path (defaults to the packaged location)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    encoder = get_encoder(args.encoder)
    verbose = not args.quiet
    if verbose:
        print(f"encoder={encoder.name} v{encoder.version} dim={encoder.dim}", file=sys.stderr)

    if args.audio_dir:
        mat = corpus_embeddings(encoder, args.audio_dir, args.limit, verbose)
        source = f"corpus:{os.path.basename(args.audio_dir.rstrip('/'))}"
    else:
        mat = synthetic_embeddings(encoder, args.voices, args.seconds, verbose)
        source = f"synthetic:{args.voices}"

    std = EmbeddingStandardizer.fit(mat, spec=encoder.spec, source=source)
    if std.is_identity:
        raise SystemExit("not enough samples to fit a reference")

    out = args.out or reference_path(encoder.spec)
    std.save(out)
    print(f"wrote {out}  (n={std.n_samples}, dim={std.dim}, source={std.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
