
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
    report = conversion_report()
    assert report["after"] > report["before"]
    assert report["after"] < 0.8, "DSP conversion is timbre transfer, not identity cloning"
