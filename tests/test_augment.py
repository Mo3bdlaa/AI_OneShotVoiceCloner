import numpy as np
import pytest

from voxprint import augment
from voxprint.features import estimate_f0
from voxprint.synthetic import SR, distinct_speakers


@pytest.fixture(scope="module")
def clip():
    return distinct_speakers(1, seed=9)[0].say("aiueo", 3.0, seed=1)


@pytest.mark.parametrize("snr_db", [30.0, 20.0, 10.0])
@pytest.mark.parametrize("color", ["white", "pink"])
def test_add_noise_hits_the_requested_snr(clip, snr_db, color):
    out = augment.add_noise(clip, snr_db, SR, color, seed=0)
    measured = 10 * np.log10(
        np.mean(clip.astype(np.float64) ** 2) / np.mean((out - clip).astype(np.float64) ** 2)
    )
    assert measured == pytest.approx(snr_db, abs=1.0)


def test_add_noise_is_deterministic(clip):
    assert np.array_equal(augment.add_noise(clip, 20.0, SR, seed=7), augment.add_noise(clip, 20.0, SR, seed=7))


def test_add_noise_rejects_an_unknown_colour(clip):
    with pytest.raises(ValueError, match="noise colour"):
        augment.add_noise(clip, 20.0, SR, color="brown")


def test_pink_noise_has_more_low_frequency_energy(clip):
    white = augment.add_noise(np.zeros(SR, dtype=np.float32) + 1e-6, 0.0, SR, "white", seed=1)
    pink = augment.add_noise(np.zeros(SR, dtype=np.float32) + 1e-6, 0.0, SR, "pink", seed=1)
    ratio = lambda w: (  # noqa: E731
        np.abs(np.fft.rfft(w))[: 200].sum() / (np.abs(np.fft.rfft(w))[2000:].sum() + 1e-9)
    )
    assert ratio(pink) > ratio(white)


def test_reverb_preserves_length_and_pitch(clip):
    out = augment.add_reverb(clip, SR, rt60=0.3, seed=0)
    assert out.size == clip.size
    before = float(np.median(estimate_f0(clip, sr=SR).voiced_f0))
    after = float(np.median(estimate_f0(out, sr=SR).voiced_f0))
    assert after == pytest.approx(before, rel=0.1)


def test_band_limit_removes_energy_outside_the_passband(clip):
    out = augment.band_limit(clip, SR, 300.0, 3400.0)
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    freqs = np.fft.rfftfreq(out.size, 1 / SR)
    inside = spectrum[(freqs > 400) & (freqs < 3300)].mean()
    above = spectrum[freqs > 5000].mean()
    assert above < inside * 0.05


def test_soft_clip_bounds_the_waveform(clip):
    assert np.max(np.abs(augment.soft_clip(clip, drive=8.0))) <= 1.0


def test_spectral_tilt_changes_the_balance(clip):
    tilted = augment.spectral_tilt(clip, 3.0, SR)
    balance = lambda w: (  # noqa: E731
        np.abs(np.fft.rfft(w.astype(np.float64)))[1500:].sum()
        / (np.abs(np.fft.rfft(w.astype(np.float64)))[:1500].sum() + 1e-9)
    )
    assert balance(tilted) > balance(clip)


def test_default_suite_produces_usable_audio(clip):
    outputs = augment.augment_all(clip, SR, seed=3)
    assert len(outputs) == len(augment.default_suite())
    for name, wav in outputs.items():
        assert np.all(np.isfinite(wav)), name
        assert np.max(np.abs(wav)) <= 1.0, name
        assert wav.size > SR, name


def test_light_suite_is_a_subset_of_conditions():
    assert {a.name for a in augment.light_suite()} <= {a.name for a in augment.default_suite()}
