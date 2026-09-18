import numpy as np
import pytest

from voxprint import audio


def test_load_save_roundtrip(tmp_path, tone):
    path = tmp_path / "t.wav"
    audio.save_audio(path, tone, audio.TARGET_SR)
    wav, sr = audio.load_audio(path)
    assert sr == audio.TARGET_SR
    assert wav.dtype == np.float32
    assert np.allclose(wav, tone, atol=1e-4)


def test_load_resamples_to_target(tmp_path, tone):
    audio.save_audio(tmp_path / "t.wav", tone, 8000)
    wav, sr = audio.load_audio(tmp_path / "t.wav", sr=16000)
    assert sr == 16000
    assert wav.size == pytest.approx(tone.size * 2, rel=0.01)


def test_load_rejects_garbage(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not audio at all")
    with pytest.raises(audio.AudioError):
        audio.load_audio(tmp_path / "bad.wav")


def test_resample_preserves_frequency(tone):
    down = audio.resample(tone, 16000, 8000)
    assert down.size == pytest.approx(tone.size // 2, rel=0.01)
    spectrum = np.abs(np.fft.rfft(down))
    peak_hz = np.fft.rfftfreq(down.size, 1 / 8000)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(150, abs=3)


def test_rms_normalize_hits_target_level(tone):
    out = audio.rms_normalize(tone * 0.01, target_dbfs=-20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms_db == pytest.approx(-20.0, abs=0.5)


def test_rms_normalize_survives_silence():
    assert np.all(audio.rms_normalize(np.zeros(1000, dtype=np.float32)) == 0.0)


def test_trim_silence_removes_leading_and_trailing(tone):
    padded = np.concatenate([np.zeros(8000, dtype=np.float32), tone, np.zeros(8000, dtype=np.float32)])
    trimmed = audio.trim_silence(padded)
    assert trimmed.size < padded.size
    assert trimmed.size >= tone.size * 0.9


def test_energy_vad_flags_speech_not_silence(tone):
    signal = np.concatenate([np.zeros(16000, dtype=np.float32), tone])
    vad = audio.energy_vad(signal)
    assert 0.2 < vad.speech_ratio < 0.85
    assert not vad.mask[:50].any()


def test_preprocess_is_idempotent_in_level(tone):
    once = audio.preprocess(tone)
    twice = audio.preprocess(once)
    rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    assert rms(once) == pytest.approx(rms(twice), rel=0.05)


def test_frame_signal_pads_short_input():
    frames = audio.frame_signal(np.zeros(100, dtype=np.float32), 400, 160)
    assert frames.shape == (1, 400)


def test_concat_with_gap_inserts_silence(tone):
    joined = audio.concat_with_gap([tone, tone], gap_ms=100.0)
    assert joined.size == tone.size * 2 + int(0.1 * audio.TARGET_SR)


def test_preemphasis_boosts_high_frequencies():
    rng = np.random.default_rng(0)
    low = np.cumsum(rng.standard_normal(4000)).astype(np.float32)  # brown noise
    out = audio.preemphasis(low)
    assert np.std(np.diff(out)) < np.std(np.diff(low)) * 2.0
    assert np.var(out) < np.var(low)
