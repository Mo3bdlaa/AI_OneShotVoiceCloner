import numpy as np
import pytest

from voxprint.encoders.base import EncoderSpec, l2_normalize
from voxprint.gallery import ConsentError, ConsentRecord, Gallery, GalleryError, validate_speaker_id

SPEC = EncoderSpec("dsp", "1", 16, 16000)
CONSENT = ConsentRecord(granted_by="Test Speaker", statement="agreed for testing")


def vecs(n, seed=0, dim=16):
    return l2_normalize(np.random.default_rng(seed).standard_normal((n, dim)), axis=1)


@pytest.fixture
def gallery():
    return Gallery(SPEC)


def test_enrolment_requires_consent(gallery):
    with pytest.raises(ConsentError, match="biometric"):
        gallery.enroll("alice", vecs(3))


def test_consent_can_be_waived_for_fixtures():
    g = Gallery(SPEC, require_consent=False)
    assert g.enroll("alice", vecs(3)).n_utterances == 3


def test_enrolment_stores_takes_and_centroid(gallery):
    print_ = gallery.enroll("alice", vecs(3), consent=CONSENT, seconds=9.0)
    assert print_.n_utterances == 3
    assert np.linalg.norm(print_.centroid) == pytest.approx(1.0)
    assert print_.total_seconds == 9.0
    assert print_.consent.granted_by == "Test Speaker"


def test_second_enrolment_appends(gallery):
    gallery.enroll("alice", vecs(2), consent=CONSENT, seconds=4.0)
    print_ = gallery.enroll("alice", vecs(2, seed=1), seconds=4.0)
    assert print_.n_utterances == 4
    assert print_.total_seconds == 8.0


def test_replace_discards_previous_takes(gallery):
    gallery.enroll("alice", vecs(5), consent=CONSENT)
    print_ = gallery.enroll("alice", vecs(2, seed=9), consent=CONSENT, replace=True)
    assert print_.n_utterances == 2


def test_dimension_mismatch_is_rejected(gallery):
    with pytest.raises(GalleryError, match="16-dim"):
        gallery.enroll("alice", np.zeros((2, 8)), consent=CONSENT)


@pytest.mark.parametrize("bad", ["", " ", "a" * 65, "no/slashes", "\n"])
def test_invalid_speaker_ids_are_rejected(bad):
    with pytest.raises(GalleryError):
        validate_speaker_id(bad)


def test_cohesion_is_high_for_consistent_takes(gallery):
    base = l2_normalize(np.random.default_rng(0).standard_normal(16))
    noise = np.random.default_rng(1).standard_normal((4, 16)) * 0.02
    print_ = gallery.enroll("alice", l2_normalize(base + noise, axis=1), consent=CONSENT)
    assert print_.cohesion() > 0.95


def test_cohesion_is_low_for_mixed_speakers(gallery):
    print_ = gallery.enroll("mixed", vecs(6, seed=7), consent=CONSENT)
    assert print_.cohesion() < 0.5
    assert any("disagree" in w for w in print_.quality_report()["warnings"])


def test_single_take_warns_and_has_no_cohesion(gallery):
    print_ = gallery.enroll("alice", vecs(1), consent=CONSENT, seconds=2.0)
    assert np.isnan(print_.cohesion())
    warnings = print_.quality_report()["warnings"]
    assert any("single enrolment take" in w for w in warnings)
    assert any("6s+" in w or "of speech enrolled" in w for w in warnings)


def test_remove_and_rename(gallery):
    gallery.enroll("alice", vecs(2), consent=CONSENT)
    gallery.rename("alice", "alice2")
    assert "alice2" in gallery and "alice" not in gallery
    gallery.remove("alice2")
    assert len(gallery) == 0
    with pytest.raises(GalleryError):
        gallery.remove("alice2")


def test_rename_onto_an_existing_id_is_refused(gallery):
    gallery.enroll("a", vecs(2), consent=CONSENT)
    gallery.enroll("b", vecs(2, seed=2), consent=CONSENT)
    with pytest.raises(GalleryError, match="already exists"):
        gallery.rename("a", "b")


def test_save_and_load_roundtrip(tmp_path, gallery):
    gallery.enroll("alice", vecs(3), consent=CONSENT, seconds=9.0, notes="hi")
    gallery.enroll("bob", vecs(2, seed=2), consent=CONSENT, seconds=6.0)
    gallery.threshold = 0.42
    gallery.save(tmp_path / "g.npz")

    loaded = Gallery.load(tmp_path / "g.npz")
    assert loaded.ids() == ["alice", "bob"]
    assert loaded.threshold == 0.42
    assert loaded.get("alice").notes == "hi"
    assert loaded.get("alice").consent.granted_by == "Test Speaker"
    assert np.allclose(loaded.get("alice").utterances, gallery.get("alice").utterances)
    assert np.allclose(loaded.get("bob").centroid, gallery.get("bob").centroid)


def test_save_is_atomic_on_failure(tmp_path, gallery, monkeypatch):
    gallery.enroll("alice", vecs(2), consent=CONSENT)
    gallery.save(tmp_path / "g.npz")
    original = (tmp_path / "g.npz").read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("voxprint.gallery.np.savez_compressed", boom)
    with pytest.raises(OSError):
        gallery.save(tmp_path / "g.npz")
    assert (tmp_path / "g.npz").read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_loading_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(GalleryError, match="not found"):
        Gallery.load(tmp_path / "nope.npz")


def test_newer_schema_is_refused(tmp_path, gallery, monkeypatch):
    gallery.enroll("alice", vecs(2), consent=CONSENT)
    monkeypatch.setattr("voxprint.gallery.SCHEMA_VERSION", 99)
    gallery.save(tmp_path / "g.npz")
    monkeypatch.setattr("voxprint.gallery.SCHEMA_VERSION", 1)
    with pytest.raises(GalleryError, match="newer version"):
        Gallery.load(tmp_path / "g.npz")


def test_centroid_matrix_is_ordered_by_id(gallery):
    gallery.enroll("zed", vecs(2), consent=CONSENT)
    gallery.enroll("amy", vecs(2, seed=3), consent=CONSENT)
    ids, mat = gallery.centroid_matrix()
    assert ids == ["amy", "zed"]
    assert mat.shape == (2, 16)


def test_refit_standardizer_needs_enough_speakers(gallery):
    for i in range(4):
        gallery.enroll(f"s{i}", vecs(3, seed=i), consent=CONSENT)
    assert gallery.refit_standardizer() is False

    for i in range(4, 10):
        gallery.enroll(f"s{i}", vecs(3, seed=i), consent=CONSENT)
    assert gallery.refit_standardizer() is True
    assert gallery.standardizer.source == "gallery"


def test_both_thresholds_survive_a_save_and_load(tmp_path, gallery):
    gallery.enroll("alice", vecs(3), consent=CONSENT)
    gallery.threshold = 0.31
    gallery.identification_threshold = 0.62
    gallery.save(tmp_path / "g.npz")

    loaded = Gallery.load(tmp_path / "g.npz")
    assert loaded.threshold == 0.31
    assert loaded.identification_threshold == 0.62


def test_a_schema_1_gallery_loads_without_an_identification_threshold(tmp_path, gallery, monkeypatch):
    """Older galleries must still open, with the field simply absent."""
    gallery.enroll("alice", vecs(2), consent=CONSENT)
    gallery.threshold = 0.4
    gallery.save(tmp_path / "g.npz")

    import json

    import numpy as _np

    with _np.load(tmp_path / "g.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["schema_version"] = 1
    manifest.pop("identification_threshold", None)
    arrays["manifest"] = _np.array(json.dumps(manifest))
    _np.savez_compressed(tmp_path / "old.npz", **arrays)

    loaded = Gallery.load(tmp_path / "old.npz")
    assert loaded.threshold == 0.4
    assert loaded.identification_threshold is None
