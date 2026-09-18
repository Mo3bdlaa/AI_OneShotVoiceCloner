import os

import pytest

from voxprint.gallery import ConsentError, ConsentRecord, GalleryError
from voxprint.pipeline import VoiceLab
from voxprint.watermark import detect_watermark

CONSENT = ConsentRecord(granted_by="test fixture", statement="synthetic voice, no real person")


@pytest.fixture
def lab(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    for name, files in clips.items():
        lab.enroll(name, files[:3], consent=CONSENT)
    lab.calibrate()
    return lab


def test_enrolment_reports_what_it_did(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    report = lab.enroll("omar", clips["omar"][:3], consent=CONSENT)
    assert report.speaker_id == "omar"
    assert report.n_embeddings >= 3
    assert report.seconds > 5.0
    assert report.cohesion > 0.5
    assert os.path.exists(report.reference_audio)


def test_enrolment_without_consent_is_refused(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    with pytest.raises(ConsentError, match="biometric"):
        lab.enroll("omar", clips["omar"][:1])


def test_consent_check_can_be_disabled(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices", require_consent=False)
    assert lab.enroll("omar", clips["omar"][:1]).speaker_id == "omar"


def test_unreadable_files_are_reported_not_fatal(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    report = lab.enroll("omar", [*clips["omar"][:2], str(tmp_path / "missing.wav")], consent=CONSENT)
    assert any("missing.wav" in w for w in report.warnings)
    assert report.n_embeddings >= 2


def test_all_files_unreadable_is_an_error(tmp_path):
    lab = VoiceLab(tmp_path / "voices")
    with pytest.raises(ValueError, match="no usable audio"):
        lab.enroll("ghost", [str(tmp_path / "a.wav")], consent=CONSENT)


def test_gallery_persists_between_sessions(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:2], consent=CONSENT)

    reopened = VoiceLab(tmp_path / "voices")
    assert reopened.gallery.ids() == ["omar"]
    assert reopened.gallery.get("omar").consent.granted_by == "test fixture"


def test_encoder_mismatch_is_refused(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:1], consent=CONSENT)
    with pytest.raises(GalleryError, match="not comparable"):
        VoiceLab(tmp_path / "voices", encoder="ecapa")


def test_identify_held_out_takes(lab, clips):
    for name, files in clips.items():
        result = lab.identify_file(files[3])       # the take never enrolled
        assert result.best.speaker_id == name, f"{name} misidentified as {result.best.speaker_id}"
        assert result.accepted


def test_identify_rejects_a_voice_that_was_never_enrolled(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    for name in ("omar", "hana", "sami"):
        lab.enroll(name, clips[name][:3], consent=CONSENT)
    lab.calibrate()
    assert not lab.identify_file(clips["laila"][0]).accepted


def test_verify_accepts_truth_and_rejects_a_false_claim(lab, clips):
    assert lab.verify_file(clips["sami"][3], "sami").accepted
    assert not lab.verify_file(clips["sami"][3], "hana").accepted


def test_calibration_sets_and_persists_a_threshold(lab, tmp_path):
    assert lab.gallery.threshold is not None
    assert VoiceLab(tmp_path / "voices").gallery.threshold == pytest.approx(lab.gallery.threshold)


def test_remove_deletes_the_print_and_the_audio(lab):
    path = lab.reference_audio_path("omar")
    assert os.path.exists(path)
    lab.remove("omar")
    assert "omar" not in lab.gallery
    assert not os.path.exists(path)


def test_conversion_writes_watermarked_audio(lab, clips):
    result = lab.convert_file(clips["omar"][0], speaker_id="hana")
    assert result.wav.size > 0
    assert result.info["watermarked"] is True
    assert detect_watermark(result.wav).present


def test_conversion_can_skip_the_watermark(lab, clips):
    result = lab.convert_file(clips["omar"][0], speaker_id="hana", watermark=False)
    assert not detect_watermark(result.wav).present


def test_conversion_accepts_a_reference_file_instead_of_an_id(lab, clips):
    assert lab.convert_file(clips["omar"][0], reference_path=clips["hana"][0]).wav.size > 0


def test_conversion_needs_some_target(lab, clips):
    with pytest.raises(ValueError, match="speaker_id or a reference"):
        lab.convert_file(clips["omar"][0])


def test_similarity_to_quantifies_the_result(lab, clips):
    """Conversion must move the voice toward the target without reaching it."""
    source = clips["omar"][0]
    converted = lab.convert_file(source, speaker_id="hana", watermark=False)

    after = lab.similarity_to(converted.wav, converted.sample_rate, "hana")
    wav, sr = lab.load_reference("omar")
    before = lab.similarity_to(wav, sr, "hana")

    assert after > before
    assert after < lab.gallery.threshold, "DSP conversion must not pass as the target"


def test_speak_explains_why_it_cannot_run(lab):
    with pytest.raises((RuntimeError, ValueError), match="not usable|text-to-speech"):
        lab.speak("hello", speaker_id="hana", backend="xtts")


def test_speak_rejects_a_conversion_only_backend(lab):
    with pytest.raises(ValueError, match="cannot generate speech from text"):
        lab.speak("hello", speaker_id="hana", backend="dspvc")


def test_cloning_without_stored_audio_explains_itself(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:1], consent=CONSENT, keep_audio=False)
    with pytest.raises(GalleryError, match="no reference audio"):
        lab.convert_file(clips["hana"][0], speaker_id="omar")


def test_summary_reports_the_state(lab):
    info = lab.summary()
    assert info["speakers"] == 5
    assert info["encoder"]["name"] == "dsp"
    assert info["threshold"] is not None
