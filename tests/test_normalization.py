import numpy as np
import pytest

from voxprint.encoders.base import EncoderSpec
from voxprint.normalization import EmbeddingStandardizer, load_default_reference


def _spec(dim=8):
    return EncoderSpec("test", "1", dim, 16000)


def test_identity_leaves_direction_unchanged():
    std = EmbeddingStandardizer.identity(8)
    vec = np.arange(8, dtype=float)
    assert std.is_identity
    assert np.allclose(std.transform(vec), vec / np.linalg.norm(vec))


def test_fit_refuses_too_few_samples():
    std = EmbeddingStandardizer.fit(np.random.default_rng(0).standard_normal((4, 8)), _spec())
    assert std.is_identity, "fitting a per-dimension variance on 4 points must not be trusted"


def test_fit_standardises_each_dimension():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 8)) * np.array([10, 1, 5, 1, 1, 1, 1, 1]) + 3.0
    std = EmbeddingStandardizer.fit(data, _spec())
    assert not std.is_identity
    assert np.allclose(std.mean, 3.0, atol=0.6)
    transformed = np.stack([std.transform(row) for row in data])
    assert np.allclose(np.linalg.norm(transformed, axis=1), 1.0)


def test_fit_floors_a_degenerate_dimension():
    data = np.random.default_rng(0).standard_normal((100, 6))
    data[:, 2] = 1.0                                  # constant in the reference set
    std = EmbeddingStandardizer.fit(data, _spec(6))
    assert std.scale[2] > 0
    assert np.isfinite(std.transform(np.ones(6))).all()


def test_standardisation_spreads_out_a_clustered_space():
    """The whole point of the module: clustered vectors must separate."""
    rng = np.random.default_rng(3)
    base = np.ones(32)
    population = np.stack([base + 0.02 * rng.standard_normal(32) for _ in range(200)])
    std = EmbeddingStandardizer.fit(population, _spec(32))

    a, b = population[0], population[1]
    raw = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    scaled = float(std.transform(a) @ std.transform(b))
    assert raw > 0.999
    assert abs(scaled) < 0.6


def test_save_and_load_roundtrip(tmp_path):
    data = np.random.default_rng(0).standard_normal((100, 8))
    std = EmbeddingStandardizer.fit(data, _spec())
    std.save(tmp_path / "ref.npz")
    loaded = EmbeddingStandardizer.load(tmp_path / "ref.npz")
    assert np.allclose(loaded.mean, std.mean)
    assert np.allclose(loaded.scale, std.scale)
    assert loaded.n_samples == std.n_samples
    assert loaded.encoder == "test"


def test_transform_rejects_a_wrong_dimension():
    std = EmbeddingStandardizer.fit(np.random.default_rng(0).standard_normal((100, 8)), _spec())
    with pytest.raises(ValueError, match="8-dim"):
        std.transform(np.zeros(5))


def test_packaged_reference_exists_for_the_dsp_encoder(encoder):
    std = load_default_reference(encoder.spec)
    assert not std.is_identity, "the shipped DSP reference statistics are missing"
    assert std.dim == encoder.dim
    assert std.n_samples >= 100


def test_unknown_encoder_falls_back_to_identity():
    assert load_default_reference(EncoderSpec("nope", "1", 16, 16000)).is_identity
