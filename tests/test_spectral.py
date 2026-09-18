import numpy as np
import pytest

from voxprint import spectral
from voxprint.features import estimate_f0
from voxprint.synthetic import SR


def test_stft_istft_reconstructs_the_signal(tone):
    back = spectral.istft(spectral.stft(tone), length=tone.size)
    assert np.max(np.abs(back[1000:-1000] - tone[1000:-1000])) < 1e-6


def test_stft_shape():
    spec = spectral.stft(np.zeros(16000, dtype=np.float32), n_fft=1024, hop=256)
    assert spec.shape[1] == 513
    assert spec.shape[0] == pytest.approx(16000 / 256, rel=0.05)


@pytest.mark.parametrize("rate", [0.5, 2.0])
def test_phase_vocoder_scales_duration(tone, rate):
    out = spectral.phase_vocoder(tone, rate)
    assert out.size == pytest.approx(tone.size / rate, rel=0.1)


def test_phase_vocoder_preserves_pitch(tone):
    out = spectral.phase_vocoder(tone, 1.5)
    assert float(np.median(estimate_f0(out, sr=SR).voiced_f0)) == pytest.approx(150, rel=0.05)


@pytest.mark.parametrize("factor", [0.75, 1.5])
def test_pitch_shift_moves_pitch_and_keeps_duration(tone, factor):
    out = spectral.pitch_shift(tone, factor, sr=SR)
    assert out.size == tone.size
    assert float(np.median(estimate_f0(out, sr=SR).voiced_f0)) == pytest.approx(150 * factor, rel=0.05)


def test_pitch_shift_is_a_noop_at_unity(tone):
    assert np.array_equal(spectral.pitch_shift(tone, 1.0, sr=SR), tone)


def test_cepstral_envelope_smooths_harmonic_ripple(tone):
    log_mag = np.log(np.abs(spectral.stft(tone)) + 1e-9)
    env = spectral.cepstral_envelope(log_mag, n_coeff=24)
    assert env.shape == log_mag.shape
    # The envelope must vary less across frequency than the raw spectrum.
    assert np.mean(np.abs(np.diff(env, axis=1))) < np.mean(np.abs(np.diff(log_mag, axis=1)))
