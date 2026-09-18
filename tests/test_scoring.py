import numpy as np
import pytest

from voxprint import scoring
from voxprint.encoders.base import EncoderSpec, l2_normalize
from voxprint.gallery import Gallery

SPEC = EncoderSpec("dsp", "1", 16, 16000)


def cluster(seed, n=4, spread=0.05, dim=16):
    """n embeddings around one direction -- a speaker with several takes."""
    rng = np.random.default_rng(seed)
    base = l2_normalize(rng.standard_normal(dim))
    return l2_normalize(base + spread * rng.standard_normal((n, dim)), axis=1), base


@pytest.fixture
def gallery():
    g = Gallery(SPEC, require_consent=False)
    for i in range(6):
        takes, _ = cluster(seed=i)
        g.enroll(f"spk{i}", takes, seconds=12.0)
    return g


def test_cosine_bounds_and_symmetry():
    a = np.array([1.0, 0.0, 0.0])
    assert scoring.cosine(a, a) == pytest.approx(1.0)
    assert scoring.cosine(a, -a) == pytest.approx(-1.0)
    assert scoring.cosine(a, np.array([0.0, 2.0, 0.0])) == pytest.approx(0.0)


def test_identify_finds_the_right_speaker(gallery):
    takes, base = cluster(seed=2)
    result = scoring.identify(base, gallery, threshold=0.5)
    assert result.best.speaker_id == "spk2"
    assert result.accepted


def test_identify_rejects_an_unknown_voice(gallery):
    _, stranger = cluster(seed=999)
    result = scoring.identify(stranger, gallery, threshold=0.9)
    assert result.decision == "unknown"
    assert not result.accepted


def test_identify_flags_an_ambiguous_tie(gallery):
    ids, centroids = gallery.centroid_matrix()
    blend = l2_normalize(centroids[0] + centroids[1])
    result = scoring.identify(blend, gallery, threshold=-1.0, min_margin=0.5)
    assert result.decision == "ambiguous"


def test_identify_on_an_empty_gallery():
    result = scoring.identify(np.ones(16), Gallery(SPEC, require_consent=False))
    assert result.decision == "empty"
    assert result.best is None


def test_identify_respects_top_k(gallery):
    _, base = cluster(seed=1)
    assert len(scoring.identify(base, gallery, top_k=3).candidates) == 3


def test_uncalibrated_results_say_so(gallery):
    _, base = cluster(seed=1)
    assert scoring.identify(base, gallery).calibrated is False
    gallery.threshold = 0.5
    assert scoring.identify(base, gallery).calibrated is True


def test_verify_accepts_the_true_speaker_and_rejects_others(gallery):
    _, base = cluster(seed=3)
    assert scoring.verify(base, gallery, "spk3", threshold=0.5).accepted
    assert not scoring.verify(base, gallery, "spk4", threshold=0.5).accepted


def test_verify_on_an_unknown_id_is_an_error(gallery):
    from voxprint.gallery import GalleryError

    with pytest.raises(GalleryError):
        scoring.verify(np.ones(16), gallery, "nobody")


def test_equal_error_rate_on_separable_distributions():
    target = np.linspace(0.6, 1.0, 100)
    impostor = np.linspace(-0.2, 0.2, 100)
    eer, threshold = scoring.equal_error_rate(target, impostor)
    assert eer == pytest.approx(0.0, abs=0.01)
    assert 0.2 < threshold <= 0.6


def test_equal_error_rate_on_identical_distributions():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(2000)
    eer, _ = scoring.equal_error_rate(scores, rng.standard_normal(2000))
    assert eer == pytest.approx(0.5, abs=0.05)


def test_equal_error_rate_handles_empty_input():
    eer, threshold = scoring.equal_error_rate(np.zeros(0), np.zeros(0))
    assert np.isnan(eer)
    assert threshold == scoring.DEFAULT_THRESHOLD


def test_threshold_for_far_bounds_false_accepts():
    target = np.linspace(0.5, 1.0, 200)
    impostor = np.linspace(-0.5, 0.45, 200)
    threshold = scoring.threshold_for_far(target, impostor, max_far=0.01)
    assert float(np.mean(impostor >= threshold)) <= 0.01


def test_collect_trial_scores_is_leave_one_out(gallery):
    target, impostor = scoring.collect_trial_scores(gallery)
    assert target.size == 6 * 4          # every take against the other three
    assert impostor.size == 6 * 4 * 5    # every take against every other speaker
    assert target.mean() > impostor.mean()


def test_calibrate_separates_and_sets_a_usable_threshold(gallery):
    result = scoring.calibrate(gallery)
    assert result.eer < 0.05
    far, frr = result.rates_at(result.threshold)
    assert far <= 0.1 and frr <= 0.1


def test_calibrate_with_a_false_accept_budget(gallery):
    result = scoring.calibrate(gallery, criterion="far", max_far=0.0)
    assert float(np.mean(result.impostor_scores >= result.threshold)) == 0.0


def test_calibrate_rejects_an_unknown_criterion(gallery):
    with pytest.raises(ValueError, match="criterion"):
        scoring.calibrate(gallery, criterion="magic")


def test_platt_scaler_is_monotonic_and_calibrated():
    target = np.random.default_rng(0).normal(0.8, 0.1, 500)
    impostor = np.random.default_rng(1).normal(0.0, 0.1, 500)
    scaler = scoring.PlattScaler.fit(target, impostor)
    assert scaler.probability(0.8) > 0.9
    assert scaler.probability(0.0) < 0.1
    probs = [scaler.probability(s) for s in np.linspace(-0.5, 1.5, 20)]
    assert all(b >= a for a, b in zip(probs, probs[1:], strict=False))


def test_platt_scaler_degrades_gracefully():
    scaler = scoring.PlattScaler.fit(np.zeros(0), np.zeros(0))
    assert scaler.probability(0.5) == pytest.approx(0.6224, abs=1e-3)


def test_calibration_with_one_take_per_speaker_is_reported_unusable():
    """A gallery with single takes has no same-speaker trials to balance."""
    g = Gallery(SPEC, require_consent=False)
    for i in range(4):
        g.enroll(f"spk{i}", cluster(seed=i, n=1)[0])

    result = scoring.calibrate(g)
    assert result.n_target == 0
    assert not result.usable
    assert any("two enrolment takes" in w for w in result.warnings)


def test_calibration_dict_is_strict_json():
    """NaN is not valid JSON: a bare NaN token breaks JSON.parse in a browser."""
    import json

    g = Gallery(SPEC, require_consent=False)
    for i in range(3):
        g.enroll(f"spk{i}", cluster(seed=i, n=1)[0])

    payload = json.dumps(scoring.calibrate(g).as_dict())

    def reject(constant):
        raise AssertionError(f"non-JSON constant in output: {constant}")

    parsed = json.loads(payload, parse_constant=reject)
    assert parsed["eer"] is None
    assert parsed["usable"] is False


def test_a_healthy_calibration_is_usable(gallery):
    result = scoring.calibrate(gallery)
    assert result.usable
    assert result.warnings == []
