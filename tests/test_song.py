"""Tests for the song re-voicing pipeline.

Separation needs Demucs and a model download, so those tests skip unless it is
installed. The mixing and orchestration logic is tested without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxprint import song
from voxprint.audio import save_audio
from voxprint.synthetic import SR, distinct_speakers


@pytest.fixture
def fake_song(tmp_path):
    """Speech over a chord pad -- something with a vocal and a backing track."""
    voice = distinct_speakers(1, seed=3)[0].say("aiueoaiueo", 6.0, seed=1)
    t = np.arange(voice.size) / SR
    backing = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (110, 165, 220)).astype(np.float32)
    path = tmp_path / "song.wav"
    save_audio(path, np.clip(voice + backing, -1, 1), SR)
    return str(path), voice, backing


def test_remix_matches_lengths_and_avoids_clipping():
    vocal = np.ones(100, dtype=np.float32) * 0.9
    backing = np.ones(150, dtype=np.float32) * 0.9
    mixed = song.remix(vocal, backing)
    assert mixed.size == 150
    assert np.max(np.abs(mixed)) <= 1.0


def test_remix_gain_changes_the_vocal_level():
    vocal = np.full(100, 0.2, dtype=np.float32)
    backing = np.zeros(100, dtype=np.float32)
    quiet = song.remix(vocal, backing, vocal_gain_db=-20.0)
    loud = song.remix(vocal, backing, vocal_gain_db=0.0)
    assert np.max(np.abs(quiet)) < np.max(np.abs(loud))


def test_revoice_rejects_a_backend_that_cannot_convert(fake_song):
    from voxprint.synth import get_synth

    path, _, _ = fake_song

    class TextOnly:
        name = "text-only"

        def supports(self, capability):
            return capability == "tts"

    with pytest.raises(ValueError, match="cannot do voice conversion"):
        song.revoice_song(path, np.zeros(SR, dtype=np.float32), SR, converter=TextOnly())

    assert get_synth("dspvc").supports("vc")


def test_revoice_without_separation_uses_the_input_as_the_vocal(fake_song):
    """The offline DSP converter is enough to exercise the orchestration."""
    from voxprint.synth import get_synth

    path, _, _ = fake_song
    reference = distinct_speakers(2, seed=11)[1].say("oeuia", 6.0, seed=2)

    wav, sr, info = song.revoice_song(
        path, reference, SR, converter=get_synth("dspvc"), separate_vocals=False
    )
    assert wav.size > SR
    assert np.all(np.isfinite(wav))
    assert np.max(np.abs(wav)) <= 1.0
    assert info["separation"].startswith("none")
    assert info["backend"] == "dspvc"


def test_separation_reports_a_missing_demucs_clearly(fake_song, monkeypatch):
    path, _, _ = fake_song
    monkeypatch.setattr(song, "demucs_available", lambda: (False, "demucs not installed"))
    with pytest.raises(song.SeparationError, match="demucs not installed"):
        song.separate(path)


def test_demucs_availability_is_a_pair():
    ready, reason = song.demucs_available()
    assert isinstance(ready, bool)
    assert reason


@pytest.mark.skipif(not song.demucs_available()[0], reason="demucs not installed")
def test_separation_recovers_the_vocal(fake_song, tmp_path):
    """The separated vocal must track the true vocal, not the backing track."""
    path, voice, backing = fake_song
    stems = song.separate(path, sr=SR)

    assert stems.vocal.size > 0
    assert stems.instrumental.size > 0
    assert stems.method.startswith("demucs")

    def corr(a, b):
        n = min(a.size, b.size)
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    assert corr(stems.vocal, voice) > corr(stems.vocal, backing)

    paths = song.write_stems(stems, tmp_path / "stems")
    assert all(np.all(np.isfinite(x)) for x in (stems.vocal, stems.instrumental))
    assert set(paths) == {"vocal", "instrumental"}
