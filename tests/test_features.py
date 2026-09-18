import numpy as np
import pytest

from voxprint import features as F
from voxprint.synthetic import SR, SyntheticSpeaker


def test_mel_filterbank_is_area_normalised_and_ordered():
    fb = F.mel_filterbank(n_mels=24, n_fft=512, sr=SR)
    assert fb.shape == (24, 257)
    assert np.all(fb >= 0)
    assert np.allclose(fb.sum(axis=1), 1.0, atol=1e-9)
    peaks = np.argmax(fb, axis=1)
    assert np.all(np.diff(peaks) > 0), "filter centres must increase with index"


def test_mel_conversion_roundtrips():
    hz = np.array([50.0, 440.0, 1000.0, 7000.0])
    assert np.allclose(F.mel_to_hz(F.hz_to_mel(hz)), hz, rtol=1e-9)


def test_mfcc_shape_and_finiteness(tone):
    coeffs = F.mfcc(tone, sr=SR, n_mfcc=20)
    assert coeffs.shape[1] == 20
    assert coeffs.shape[0] == pytest.approx(tone.size / (SR * 0.01), rel=0.05)
    assert np.all(np.isfinite(coeffs))


def test_dct_matrix_is_orthonormal():
    basis = F.dct_matrix(16, 16)
    assert np.allclose(basis @ basis.T, np.eye(16), atol=1e-9)


def test_cmvn_zero_means_unit_variance(tone):
    normed = F.cmvn(F.mfcc(tone))
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(normed.std(axis=0), 1.0, atol=1e-6)


def test_cmvn_removes_a_constant_channel_offset(tone):
    coeffs = F.mfcc(tone)
    shifted = coeffs + 3.7                      # a fixed channel colouration
    assert np.allclose(F.cmvn(coeffs), F.cmvn(shifted), atol=1e-8)


def test_deltas_are_zero_on_constant_input():
    assert np.allclose(F.deltas(np.ones((30, 5))), 0.0)


def test_deltas_follow_a_linear_ramp():
    feat = np.arange(40, dtype=np.float64)[:, None] * np.ones((1, 3))
    d = F.deltas(feat)
    assert np.allclose(d[5:-5], 1.0, atol=1e-9)


@pytest.mark.parametrize("hz", [85.0, 140.0, 220.0, 330.0])
def test_f0_tracker_recovers_a_known_pitch(hz):
    t = np.arange(SR * 1.5) / SR
    wav = (0.6 * np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32)
    track = F.estimate_f0(wav, sr=SR)
    assert track.voiced.mean() > 0.8
    assert float(np.median(track.voiced_f0)) == pytest.approx(hz, rel=0.03)


def test_f0_tracker_reports_noise_as_unvoiced():
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
    assert F.estimate_f0(noise, sr=SR).voiced.mean() < 0.5


def test_f0_tracker_handles_silence():
    track = F.estimate_f0(np.zeros(SR, dtype=np.float32), sr=SR)
    assert track.voiced_f0.size == 0


def test_levinson_durbin_matches_a_known_ar_process():
    # x[n] = 0.8 x[n-1] + e  ->  a = [1, -0.8]
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    for n in range(1, x.size):
        x[n] = 0.8 * x[n - 1] + rng.standard_normal()
    r = np.correlate(x, x, mode="full")[x.size - 1 : x.size + 3] / x.size
    a, err = F.levinson_durbin(r, 1)
    assert a[1] == pytest.approx(-0.8, abs=0.05)
    assert err > 0


def test_lpc_cepstrum_separates_two_vocal_tracts():
    a = SyntheticSpeaker("a", f0=120.0, formants=(500.0, 1500.0, 2500.0), seed=1)
    b = SyntheticSpeaker("b", f0=120.0, formants=(750.0, 1100.0, 3000.0), seed=2)
    ca = F.lpc_cepstrum(a.say("a", 2.0, seed=1)).mean(axis=0)
    cb = F.lpc_cepstrum(b.say("a", 2.0, seed=1)).mean(axis=0)
    assert np.linalg.norm(ca - cb) > 0.1


def test_spectral_shape_tracks_brightness():
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    centroid = lambda w: F.spectral_shape(F.power_spectrogram(w, sr=SR), sr=SR)[:, 0].mean()
    assert centroid(high) > centroid(low) * 3


def test_robust_stats_layout_and_empty_input():
    feat = np.random.default_rng(0).standard_normal((100, 7))
    stats = F.robust_stats(feat)
    assert stats.shape == (28,)
    assert np.allclose(stats[:7], feat.mean(axis=0))
    assert F.robust_stats(np.zeros((0, 7))).shape == (28,)
