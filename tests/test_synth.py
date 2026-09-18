import numpy as np
import pytest

from voxprint.encoders import get_encoder
from voxprint.features import estimate_f0
from voxprint.normalization import load_default_reference
from voxprint.synth import available_synths, get_synth
from voxprint.synthetic import SR


@pytest.fixture(scope="module")
def converter():
    return get_synth("dspvc")


def test_registry_lists_the_backends():
    assert {"dspvc", "xtts", "yourtts"} <= set(available_synths())


def test_unknown_backend_is_rejected():
    with pytest.raises(KeyError, match="unknown synthesis backend"):
        get_synth("nope")


def test_dspvc_cannot_do_text_to_speech(converter):
    with pytest.raises(NotImplementedError, match="text-to-speech"):
        converter.synthesize("hello", np.zeros(SR, dtype=np.float32), SR)


def test_neural_backends_report_their_status():
    info = get_synth("xtts").describe()
    assert info["capabilities"] == ["tts", "vc"]
    assert "CPML" in info["notes"], "the licence restriction must be visible"


def test_conversion_preserves_duration(converter, cast):
    source = cast["omar"].say("aiueo", 3.0, seed=1)
    result = converter.convert(source, cast["hana"].say("oeuia", 3.0, seed=2), SR)
    assert result.duration == pytest.approx(3.0, rel=0.15)
    assert np.all(np.isfinite(result.wav))


def test_conversion_moves_pitch_toward_the_target(converter, cast):
    source = cast["omar"].say("aiueo", 3.0, seed=1)      # ~104 Hz
    target = cast["hana"].say("oeuia", 3.0, seed=2)      # ~212 Hz
    result = converter.convert(source, target, SR)

    f0 = lambda w: float(np.median(estimate_f0(w, sr=SR).voiced_f0))
    assert f0(source) < f0(result.wav) <= f0(target) * 1.05
    assert result.info["semitones"] > 0


def test_pitch_shift_is_capped(converter, cast):
    result = converter.convert(
        cast["omar"].say("aiueo", 3.0, seed=1), cast["hana"].say("oeuia", 3.0, seed=2), SR
    )
    assert abs(result.info["semitones"]) <= converter.config.max_semitones


def test_pitch_strength_zero_leaves_pitch_alone(converter, cast):
    result = converter.convert(
        cast["omar"].say("aiueo", 3.0, seed=1),
        cast["hana"].say("oeuia", 3.0, seed=2),
        SR,
        pitch_strength=0.0,
    )
    assert result.info["semitones"] == 0.0


def test_conversion_moves_the_voice_print_toward_the_target(converter, cast):
    """The honest measurement: closer to the target, but not close enough to match."""
    encoder = get_encoder("dsp")
    std = load_default_reference(encoder.spec)
    embed = lambda w: std.transform(encoder.embed(w, SR))

    source = cast["omar"].say("aiueo", 3.0, seed=1)
    target = cast["hana"].say("oeuia", 3.0, seed=2)
    out = converter.convert(source, target, SR).wav

    before = float(embed(source) @ embed(target))
    after = float(embed(out) @ embed(target))
    assert after > before + 0.3, "conversion should close part of the gap"
    assert after < 0.75, "it should not be mistaken for real identity cloning"


def test_conversion_rejects_audio_shorter_than_a_frame(converter):
    with pytest.raises(ValueError, match="longer than one analysis frame"):
        converter.convert(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32), SR)
