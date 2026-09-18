"""Evaluation on a real speech corpus.

The numbers in :mod:`voxprint.evaluate` come from procedurally generated voices,
which are clean, perfectly matched in recording conditions, and differ from each
other in exactly the parameters the encoder measures. They establish that the
pipeline is correct. They say very little about accuracy on real speech.

This module closes that gap. Point it at a folder of real recordings and it
reports the same metrics measured properly:

* **speaker-disjoint strangers.** A fraction of speakers is held out entirely and
  never enrolled, so open-set rejection is measured against people the system has
  genuinely never heard -- not against held-out clips of enrolled speakers.
* **trial-based error rates as well as enrolment-based ones.** Calibration can
  only use enrolment audio, which makes its error rate optimistic. The trial EER,
  computed from held-out queries against enrolled centroids, is the number that
  reflects deployment.
* **ranking accuracy reported separately from acceptance.** Condition mismatch
  shows up as a score shift long before it shows up as a ranking failure, so
  collapsing the two hides the most actionable diagnostic.

Corpus layout::

    corpus/
      speaker_a/  utt1.wav  utt2.wav  utt3.wav  utt4.wav
      speaker_b/  ...

Any layout that puts one folder per speaker works; folder names become speaker
ids. Common corpora in this shape, or one command away from it: VoxCeleb,
LibriSpeech, Mozilla Common Voice (after grouping by ``client_id``), and your own
recordings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoders import SpeakerEncoder, get_encoder
from .gallery import Gallery
from .scoring import calibrate, equal_error_rate, identify

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".aac"}


class CorpusError(RuntimeError):
    """Raised when a corpus is missing, mislaid out, or too small to evaluate."""


@dataclass
class CorpusSplit:
    """Who is enrolled, what is queried, and who is a stranger."""

    enrollment: dict[str, list[Path]] = field(default_factory=dict)
    trials: dict[str, list[Path]] = field(default_factory=dict)
    strangers: dict[str, list[Path]] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "enrolled_speakers": len(self.enrollment),
            "enrollment_files": sum(len(v) for v in self.enrollment.values()),
            "trial_files": sum(len(v) for v in self.trials.values()),
            "stranger_speakers": len(self.strangers),
            "stranger_files": sum(len(v) for v in self.strangers.values()),
        }


def load_corpus(root: str | Path, min_files: int = 2) -> dict[str, list[Path]]:
    """Discover ``{speaker_id: [files]}`` from a folder-per-speaker layout."""
    root = Path(root)
    if not root.is_dir():
        raise CorpusError(f"{root} is not a directory")

    corpus: dict[str, list[Path]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        files = sorted(p for p in entry.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if len(files) >= min_files:
            corpus[entry.name] = files

    if not corpus:
        raise CorpusError(
            f"no speakers found under {root}. Expected one sub-directory per speaker, each with "
            f"at least {min_files} audio files ({', '.join(sorted(AUDIO_SUFFIXES))})."
        )
    return corpus


def split_corpus(
    corpus: dict[str, list[Path]],
    enroll_files: int = 3,
    stranger_fraction: float = 0.25,
    max_trials_per_speaker: int = 5,
    seed: int = 0,
) -> CorpusSplit:
    """Partition into enrolment, trial and stranger sets.

    Strangers are whole speakers, not spare clips: measuring open-set rejection
    against held-out clips of *enrolled* speakers would answer a different and much
    easier question.
    """
    rng = random.Random(seed)
    speakers = sorted(corpus)
    rng.shuffle(speakers)

    n_strangers = int(len(speakers) * stranger_fraction)
    if len(speakers) - n_strangers < 2:
        n_strangers = max(0, len(speakers) - 2)

    stranger_ids = set(speakers[:n_strangers])
    split = CorpusSplit()

    for speaker in sorted(corpus):
        files = list(corpus[speaker])
        rng.shuffle(files)
        if speaker in stranger_ids:
            split.strangers[speaker] = files[:max_trials_per_speaker]
            continue
        enrolment = files[:enroll_files]
        trials = files[enroll_files : enroll_files + max_trials_per_speaker]
        if not enrolment or not trials:
            # Too few files to both enrol and query: skip rather than reuse a file
            # for both, which would inflate every score for that speaker.
            continue
        split.enrollment[speaker] = enrolment
        split.trials[speaker] = trials

    if len(split.enrollment) < 2:
        raise CorpusError(
            "need at least two speakers with more than "
            f"{enroll_files} files each (found {len(split.enrollment)}). "
            "Lower --enroll-files or use a larger corpus."
        )
    return split


def _embed_files(encoder: SpeakerEncoder, paths: list[Path]) -> tuple[np.ndarray, list[str]]:
    """Embed each file, collecting rather than raising on unreadable ones."""
    from .audio import load_audio

    vectors: list[np.ndarray] = []
    problems: list[str] = []
    for path in paths:
        try:
            wav, sr = load_audio(path, sr=encoder.sample_rate)
            vectors.append(encoder.embed(wav, sr))
        except Exception as exc:
            problems.append(f"{path.name}: {exc}")
    mat = np.stack(vectors) if vectors else np.zeros((0, encoder.dim))
    return mat, problems


def evaluate_corpus(
    root: str | Path,
    encoder: str | SpeakerEncoder = "dsp",
    *,
    enroll_files: int = 3,
    stranger_fraction: float = 0.25,
    max_trials_per_speaker: int = 5,
    max_speakers: int | None = None,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    """Run the full protocol on a real corpus and return measured metrics."""
    enc = get_encoder(encoder) if isinstance(encoder, str) else encoder
    corpus = load_corpus(root)
    if max_speakers is not None:
        corpus = {k: corpus[k] for k in sorted(corpus)[:max_speakers]}
    split = split_corpus(
        corpus,
        enroll_files=enroll_files,
        stranger_fraction=stranger_fraction,
        max_trials_per_speaker=max_trials_per_speaker,
        seed=seed,
    )

    problems: list[str] = []
    gallery = Gallery(enc.spec, require_consent=False)

    for i, (speaker, paths) in enumerate(sorted(split.enrollment.items()), 1):
        vectors, errs = _embed_files(enc, paths)
        problems.extend(errs)
        if vectors.shape[0]:
            gallery.enroll(speaker, vectors)
        if progress:
            print(f"  enrolled {i}/{len(split.enrollment)}: {speaker}", flush=True)

    if len(gallery) < 2:
        raise CorpusError("fewer than two speakers survived enrolment; check the reported file errors")

    gallery.refit_standardizer()
    cal = calibrate(gallery)
    gallery.threshold = cal.threshold

    # -- held-out trials ---------------------------------------------------- #
    ids, centroids = gallery.centroid_matrix(standardize=True)
    index = {sid: i for i, sid in enumerate(ids)}

    rank1 = accepted_correct = n_trials = 0
    trial_target: list[float] = []
    trial_impostor: list[float] = []
    per_speaker: dict[str, dict] = {}

    for speaker, paths in sorted(split.trials.items()):
        if speaker not in index:
            continue
        vectors, errs = _embed_files(enc, paths)
        problems.extend(errs)
        hits = accepts = 0
        for vec in vectors:
            scores = centroids @ gallery.standardizer.transform(vec)
            target = index[speaker]
            n_trials += 1
            hit = int(np.argmax(scores) == target)
            accept = int(hit and scores[target] >= cal.threshold)
            rank1 += hit
            accepted_correct += accept
            hits += hit
            accepts += accept
            trial_target.append(float(scores[target]))
            trial_impostor.extend(float(s) for i, s in enumerate(scores) if i != target)
        if vectors.shape[0]:
            per_speaker[speaker] = {
                "trials": int(vectors.shape[0]),
                "rank1": round(hits / vectors.shape[0], 3),
                "accepted": round(accepts / vectors.shape[0], 3),
            }

    trial_eer, trial_threshold = equal_error_rate(np.asarray(trial_target), np.asarray(trial_impostor))

    # -- strangers ---------------------------------------------------------- #
    stranger_trials = stranger_rejected = 0
    for _speaker, paths in sorted(split.strangers.items()):
        vectors, errs = _embed_files(enc, paths)
        problems.extend(errs)
        for vec in vectors:
            stranger_trials += 1
            stranger_rejected += int(not identify(vec, gallery).accepted)

    return {
        "corpus": str(root),
        "encoder": f"{enc.name} v{enc.version}",
        "dim": enc.dim,
        "split": split.summary(),
        "enrolment_eer": round(cal.eer, 4),
        "threshold": round(cal.threshold, 4),
        "trial_eer": None if np.isnan(trial_eer) else round(float(trial_eer), 4),
        "trial_eer_threshold": round(float(trial_threshold), 4),
        "rank1_accuracy": round(rank1 / n_trials, 4) if n_trials else None,
        "accepted_and_correct": round(accepted_correct / n_trials, 4) if n_trials else None,
        "trials": n_trials,
        "open_set_rejection": round(stranger_rejected / stranger_trials, 4) if stranger_trials else None,
        "stranger_trials": stranger_trials,
        "target_score_mean": round(float(np.mean(trial_target)), 4) if trial_target else None,
        "impostor_score_mean": round(float(np.mean(trial_impostor)), 4) if trial_impostor else None,
        "per_speaker": per_speaker,
        "file_errors": problems[:20],
        "file_error_count": len(problems),
    }
