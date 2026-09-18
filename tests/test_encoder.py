import itertools

import numpy as np
import pytest

from voxprint.encoders import available_encoders, get_encoder, l2_normalize
from voxprint.normalization import load_default_reference
from voxprint.synthetic import PHRASES, SR, add_noise


@pytest.fixture(scope="module")
def std(encoder):
    return load_default_reference(encoder.spec)


def _score(std, a, b):
    return float(std.transform(a) @ std.transform(b))


def test_registry_lists_all_backends():
    assert {"dsp", "ecapa", "resemblyzer"} <= set(available_encoders())


def test_unknown_encoder_is_rejected():
    with pytest.raises(KeyError, match="unknown encoder"):
        get_encoder("does-not-exist")


def test_embedding_is_unit_length_and_right_size(encoder, cast):
    vec = encoder.embed(cast["omar"].say("aiueo", 3.0, seed=1))
    assert vec.shape == (encoder.dim,)
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert np.all(np.isfinite(vec))


def test_embedding_is_deterministic(encoder, cast):
    wav = cast["hana"].say("aiueo", 3.0, seed=1)
    assert np.array_equal(encoder.embed(wav), encoder.embed(wav))


def test_too_short_audio_is_refused(encoder):
    with pytest.raises(ValueError, match="speech"):
        encoder.embed(np.zeros(100, dtype=np.float32))


def test_block_dimensions_sum_to_the_embedding_size(encoder, cast):
    blocks = encoder.blocks(cast["omar"].say("aiueo", 2.0, seed=1))
    assert sum(b.size for b in blocks.values()) == encoder.dim


def test_same_speaker_scores_above_every_impostor(encoder, std, cast):
    embeddings = {
        name: [encoder.embed(sp.say(p, 3.0, seed=10 + i)) for i, p in enumerate(PHRASES[:2])]
        for name, sp in cast.items()
    }
    same = [_score(std, v[0], v[1]) for v in embeddings.values()]
    impostor = [
        _score(std, embeddings[a][0], embeddings[b][1]) for a, b in itertools.combinations(cast, 2)
    ]
    assert min(same) > max(impostor), (
        f"no separation: worst same-speaker {min(same):.3f} <= best impostor {max(impostor):.3f}"
    )


def test_identity_survives_a_change_of_content(encoder, std, cast):
    """The encoder must key on the speaker, not on which vowels were spoken."""
    speaker = cast["laila"]
    a = encoder.embed(speaker.say("aiueo", 3.0, seed=1))
    b = encoder.embed(speaker.say("uoeia", 3.0, seed=2))
    other = encoder.embed(cast["khaled"].say("aiueo", 3.0, seed=1))
    assert _score(std, a, b) > _score(std, a, other)


def test_loudness_does_not_change_the_embedding(encoder, cast):
    wav = cast["sami"].say("aiueo", 3.0, seed=1)
    quiet = encoder.embed(wav * 0.05)
    loud = encoder.embed(wav * 0.9)
    assert float(quiet @ loud) > 0.999


def test_moderate_noise_keeps_the_nearest_neighbour_correct(encoder, std, cast):
    refs = {name: encoder.embed(sp.say("aiueo", 3.0, seed=1)) for name, sp in cast.items()}
    query = encoder.embed(add_noise(cast["laila"].say("aiueo", 3.0, seed=1), snr_db=30.0))
    best = max(refs, key=lambda n: _score(std, query, refs[n]))
    assert best == "laila"


def test_embed_windows_returns_one_vector_per_window(encoder, cast):
    long_clip = cast["omar"].say("aiueoaiueoaiueo", 12.0, seed=1)
    windows = encoder.embed_windows(long_clip, SR, window_seconds=3.0, hop_seconds=1.5)
    assert windows.shape[0] >= 5
    assert windows.shape[1] == encoder.dim
    assert np.allclose(np.linalg.norm(windows, axis=1), 1.0)


def test_window_embeddings_of_one_speaker_agree(encoder, cast):
    windows = encoder.embed_windows(cast["hana"].say("aiueoaiueo", 9.0, seed=1), SR)
    sim = l2_normalize(windows, axis=1) @ l2_normalize(windows, axis=1).T
    iu = np.triu_indices(windows.shape[0], k=1)
    assert sim[iu].mean() > 0.9


def test_describe_reports_the_block_layout(encoder):
    info = encoder.describe()
    assert info["dim"] == encoder.dim
    assert set(info["blocks"]) == set(info["weights"])


def test_spec_rejects_a_different_encoder(encoder):
    other = get_encoder("ecapa").spec
    assert not encoder.spec.compatible_with(other)
    assert encoder.spec.compatible_with(encoder.spec)


def test_longer_queries_score_higher(encoder, std, cast):
    """More speech means a better estimate of the same statistics."""
    speaker = cast["omar"]
    reference = encoder.embed(speaker.say("aiueoaiueo", 8.0, seed=1))
    short = _score(std, encoder.embed(speaker.say("aiueo", 1.5, seed=2)), reference)
    long = _score(std, encoder.embed(speaker.say("aiueoaiueo", 6.0, seed=2)), reference)
    assert long > short


def test_noise_lowers_the_score_even_when_the_ranking_holds(encoder, std, cast):
    """Pins the documented weakness: noise shifts scores far more than it shifts ranks."""
    speaker = cast["khaled"]
    reference = encoder.embed(speaker.say("aiueo", 4.0, seed=1))
    clean = _score(std, encoder.embed(speaker.say("oeuia", 4.0, seed=2)), reference)
    noisy = _score(std, encoder.embed(add_noise(speaker.say("oeuia", 4.0, seed=2), 30.0)), reference)
    assert noisy < clean - 0.2, "if this stops holding, the noise caveats in the docs are stale"
