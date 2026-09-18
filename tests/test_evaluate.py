
from voxprint.evaluate import build_gallery, conversion_report, run_selftest


def test_build_gallery_enrols_every_speaker(encoder):
    gallery, speakers = build_gallery(encoder, n_speakers=4, seconds=2.0)
    assert len(gallery) == 4
    assert len(speakers) == 4
    assert all(p.n_utterances == 3 for p in gallery)


def test_selftest_meets_its_published_numbers():
    """Guards the accuracy claims in the README against silent regressions."""
    report = run_selftest(n_speakers=10, seconds=3.0, n_strangers=8)
    assert report["closed_set_accuracy"] >= 0.85
    assert report["open_set_rejection"] >= 0.85
    assert report["eer"] <= 0.10
    assert report["target_mean"] > report["impostor_mean"] + 0.4


def test_conversion_report_quantifies_the_gap():
    """Reported as a contrast, since absolute cosines move with the reference."""
    report = conversion_report()
    assert report["contrast_after"] > report["contrast_before"], "conversion must close some of the gap"
    assert report["gain"] > 0.3
    assert report["contrast_after"] < 0.5, "timbre transfer, not identity cloning"


def test_distinct_speakers_are_actually_distinct():
    """The fixture must not hand the evaluation two copies of one voice."""
    import itertools

    from voxprint.synthetic import MIN_VOICE_DISTANCE, distinct_speakers, voice_distance

    speakers = distinct_speakers(12, seed=1)
    assert len(speakers) == 12
    pairs = [voice_distance(a, b) for a, b in itertools.combinations(speakers, 2)]
    assert min(pairs) >= MIN_VOICE_DISTANCE


def test_distinct_speakers_is_deterministic():
    from voxprint.synthetic import distinct_speakers, voice_distance

    a = distinct_speakers(6, seed=3)
    b = distinct_speakers(6, seed=3)
    assert all(voice_distance(x, y) == 0.0 for x, y in zip(a, b, strict=True))


def test_distinct_speakers_reports_when_it_cannot_pack_enough():
    import pytest

    from voxprint.synthetic import distinct_speakers

    with pytest.raises(ValueError, match="could only place"):
        distinct_speakers(50, seed=1, min_distance=3.0, max_attempts=2000)


def test_selftest_draws_strangers_from_the_same_distinct_pool():
    """Strangers must be as different from enrolled speakers as speakers are from each other."""
    report = run_selftest(n_speakers=8, seconds=3.0, n_strangers=8)
    assert report["min_voice_distance"] > 0
    assert report["open_set_rejection"] >= 0.85
