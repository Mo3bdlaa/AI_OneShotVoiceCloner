"""Tests for the optional neural backends.

Skipped when the dependencies are absent, so the default suite stays
download-free. Run them with the neural extras installed:

    pip install -r requirements-neural.txt
    COQUI_TOS_AGREED=1 pytest tests/test_neural_backends.py

The ECAPA tests download an ~80 MB checkpoint on first run; the XTTS ones
download ~2 GB, so they additionally require ``VOXPRINT_TEST_XTTS=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from voxprint.encoders import get_encoder
from voxprint.features import estimate_f0
from voxprint.synth import get_synth
from voxprint.synthetic import SR, distinct_speakers

ecapa_available = get_encoder("ecapa").available()[0]
xtts_available = get_synth("xtts").available()[0] and os.environ.get("VOXPRINT_TEST_XTTS") == "1"

needs_ecapa = pytest.mark.skipif(not ecapa_available, reason="speechbrain/torch not installed")
needs_xtts = pytest.mark.skipif(not xtts_available, reason="set VOXPRINT_TEST_XTTS=1 with coqui-tts installed")


# --------------------------------------------------------------------------- #
# Availability reporting works without the dependencies present
# --------------------------------------------------------------------------- #

def test_missing_and_broken_are_reported_differently(monkeypatch):
    """"Not installed" and "installed but broken" need different fixes."""
    synth = get_synth("xtts")

    import builtins

    real_import = builtins.__import__

    def fail_with(exc):
        def fake_import(name, *args, **kwargs):
            if name == "TTS" or name.startswith("TTS."):
                raise exc
            return real_import(name, *args, **kwargs)
        return fake_import

    monkeypatch.setattr(builtins, "__import__", fail_with(ModuleNotFoundError("No module named 'TTS'")))
    ready, reason = synth.available()
    assert not ready and "not installed" in reason

    monkeypatch.setattr(builtins, "__import__", fail_with(ImportError("cannot import name 'isin_mps_friendly'")))
    ready, reason = synth.available()
    assert not ready
    assert "dependency conflict" in reason
    assert "isin_mps_friendly" in reason


def test_licence_must_be_accepted_explicitly(monkeypatch):
    """The backend must not accept a non-commercial licence on the user's behalf."""
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    synth = get_synth("xtts")
    ready, reason = synth.available()
    assert not ready
    assert "Coqui Public Model License" in reason
    with pytest.raises(RuntimeError, match="COQUI_TOS_AGREED"):
        synth.synthesize("hello", np.zeros(SR, dtype=np.float32), SR)


# --------------------------------------------------------------------------- #
# ECAPA
# --------------------------------------------------------------------------- #

@needs_ecapa
def test_ecapa_produces_a_unit_192_vector():
    enc = get_encoder("ecapa")
    vec = enc.embed(distinct_speakers(1, seed=3)[0].say("aiueo", 3.0, seed=1), SR)
    assert vec.shape == (192,)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert np.all(np.isfinite(vec))


@needs_ecapa
def test_ecapa_separates_distinct_speakers():
    enc = get_encoder("ecapa")
    a, b = distinct_speakers(2, seed=5)
    same = float(enc.embed(a.say("aiueo", 3.0, seed=1), SR) @ enc.embed(a.say("oeuia", 3.0, seed=2), SR))
    diff = float(enc.embed(a.say("aiueo", 3.0, seed=1), SR) @ enc.embed(b.say("oeuia", 3.0, seed=2), SR))
    assert same > diff


@needs_ecapa
def test_a_gallery_will_not_mix_encoders(tmp_path, clips):
    """Embeddings from different encoders are not comparable; mixing must fail."""
    from voxprint.gallery import ConsentRecord, GalleryError
    from voxprint.pipeline import VoiceLab

    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:1],
               consent=ConsentRecord(granted_by="fixture", statement="synthetic"))
    with pytest.raises(GalleryError, match="not comparable"):
        VoiceLab(tmp_path / "voices", encoder="ecapa")


# --------------------------------------------------------------------------- #
# XTTS
# --------------------------------------------------------------------------- #

@needs_xtts
def test_xtts_speaks_english_and_arabic():
    synth = get_synth("xtts")
    reference = distinct_speakers(1, seed=7)[0].say("aiueoaiueoaiueo", 10.0, seed=1)
    for language, text in (("en", "A short check of the backend."), ("ar", "اختبار قصير للنظام.")):
        result = synth.synthesize(text, reference, SR, language=language)
        assert result.duration > 0.5, language
        assert result.sample_rate >= 16_000
        assert np.all(np.isfinite(result.wav))
        assert float(np.sqrt(np.mean(result.wav.astype(np.float64) ** 2))) > 0.005, f"{language} is silent"


@needs_xtts
def test_xtts_rejects_an_unsupported_language():
    synth = get_synth("xtts")
    reference = distinct_speakers(1, seed=7)[0].say("aiueo", 6.0, seed=1)
    with pytest.raises(ValueError, match="not supported"):
        synth.synthesize("hello", reference, SR, language="xx")


@needs_xtts
def test_xtts_reproduces_the_reference_pitch_register():
    """The decisive check that the reference audio conditions the output."""
    from voxprint.audio import resample

    synth = get_synth("xtts")
    low, high = sorted(distinct_speakers(2, seed=77), key=lambda s: s.f0)
    text = "This sentence is used to compare two reference voices."

    registers = {}
    for tag, speaker in (("low", low), ("high", high)):
        result = synth.synthesize(text, speaker.say("aiueoaiueoaiueo", 10.0, seed=1), SR, language="en")
        wav = resample(result.wav, result.sample_rate, SR)
        voiced = estimate_f0(wav, sr=SR).voiced_f0
        registers[tag] = float(np.median(voiced))

    assert registers["low"] < registers["high"], registers
    # It should land near the reference, not merely on the right side of it.
    assert registers["low"] == pytest.approx(low.f0, rel=0.35), registers
    assert registers["high"] == pytest.approx(high.f0, rel=0.35), registers
