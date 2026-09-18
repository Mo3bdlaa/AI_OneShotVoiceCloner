"""Tests for the real-corpus evaluation harness.

The harness is exercised with a corpus of synthetic voices written to disk in the
layout it expects, which checks the protocol -- splitting, enrolment, trials,
strangers -- without needing a corpus we cannot ship.
"""

from __future__ import annotations

import pytest

from voxprint import corpus as C
from voxprint.audio import save_audio
from voxprint.synthetic import PHRASES, SR, random_speaker


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    """Eight speakers, six takes each, one folder per speaker."""
    root = tmp_path_factory.mktemp("corpus")
    for i in range(8):
        speaker = random_speaker(seed=50_000 + i)
        folder = root / f"spk{i:02d}"
        folder.mkdir()
        for j in range(6):
            wav = speaker.say(PHRASES[j % len(PHRASES)], seconds=3.0, seed=60_000 + 7 * i + j)
            save_audio(folder / f"take{j}.wav", wav, SR)
    return root


def test_load_corpus_finds_every_speaker(corpus_dir):
    found = C.load_corpus(corpus_dir)
    assert len(found) == 8
    assert all(len(v) == 6 for v in found.values())


def test_load_corpus_skips_speakers_with_too_few_files(corpus_dir, tmp_path):
    sparse = tmp_path / "sparse"
    (sparse / "only_one").mkdir(parents=True)
    save_audio(sparse / "only_one" / "a.wav", random_speaker(seed=1).say("aiueo", 2.0), SR)
    with pytest.raises(C.CorpusError, match="no speakers found"):
        C.load_corpus(sparse, min_files=2)


def test_load_corpus_rejects_a_missing_directory(tmp_path):
    with pytest.raises(C.CorpusError, match="not a directory"):
        C.load_corpus(tmp_path / "nope")


def test_split_holds_out_whole_speakers_as_strangers(corpus_dir):
    split = C.split_corpus(C.load_corpus(corpus_dir), stranger_fraction=0.25, seed=0)
    assert set(split.enrollment) & set(split.strangers) == set()
    assert len(split.strangers) == 2
    assert len(split.enrollment) == 6


def test_split_never_reuses_a_file_for_enrolment_and_query(corpus_dir):
    split = C.split_corpus(C.load_corpus(corpus_dir), enroll_files=3, seed=1)
    for speaker, enrolment in split.enrollment.items():
        assert not set(enrolment) & set(split.trials[speaker])


def test_split_is_deterministic_for_a_seed(corpus_dir):
    found = C.load_corpus(corpus_dir)
    a = C.split_corpus(found, seed=7)
    b = C.split_corpus(found, seed=7)
    assert a.enrollment == b.enrollment
    assert a.strangers == b.strangers


def test_split_refuses_a_corpus_that_is_too_small(corpus_dir):
    with pytest.raises(C.CorpusError, match="at least two speakers"):
        C.split_corpus(C.load_corpus(corpus_dir), enroll_files=99)


def test_evaluate_corpus_reports_every_metric(corpus_dir):
    report = C.evaluate_corpus(corpus_dir, enroll_files=3, stranger_fraction=0.25, seed=0)

    assert report["split"]["enrolled_speakers"] == 6
    assert report["split"]["stranger_speakers"] == 2
    assert report["trials"] > 0
    assert report["file_error_count"] == 0

    # Held-out trials from matched-condition synthetic voices should be easy.
    assert report["rank1_accuracy"] >= 0.9
    assert report["trial_eer"] <= 0.15
    assert report["target_score_mean"] > report["impostor_score_mean"]
    # Every per-speaker entry must name a speaker that was actually enrolled.
    assert set(report["per_speaker"]) <= {f"spk{i:02d}" for i in range(8)}
    assert report["per_speaker"], "per-speaker breakdown must not be empty"


def test_trial_eer_is_reported_separately_from_enrolment_eer(corpus_dir):
    """The two are different measurements and must not be conflated."""
    report = C.evaluate_corpus(corpus_dir, seed=2)
    assert "enrolment_eer" in report and "trial_eer" in report
    assert report["enrolment_eer"] is not None
    assert report["trial_eer"] is not None


def test_unreadable_files_are_counted_not_fatal(corpus_dir, tmp_path):
    broken = tmp_path / "broken"
    for i in range(3):
        src = corpus_dir / f"spk{i:02d}"
        dst = broken / f"spk{i:02d}"
        dst.mkdir(parents=True)
        for path in sorted(src.glob("*.wav")):
            dst.joinpath(path.name).write_bytes(path.read_bytes())
        dst.joinpath("corrupt.wav").write_bytes(b"this is not audio")

    report = C.evaluate_corpus(broken, enroll_files=2, stranger_fraction=0.0, seed=0)
    assert report["file_error_count"] >= 1
    assert report["trials"] > 0


def test_max_speakers_caps_the_run(corpus_dir):
    report = C.evaluate_corpus(corpus_dir, max_speakers=4, stranger_fraction=0.25, seed=0)
    assert report["split"]["enrolled_speakers"] + report["split"]["stranger_speakers"] <= 4
