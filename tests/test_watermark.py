import numpy as np
import pytest

from voxprint.audio import resample
from voxprint.synthetic import SR, random_speaker
from voxprint.watermark import (
    BLOCK,
    DEFAULT_Z_THRESHOLD,
    WatermarkDetection,
    detect_watermark,
    embed_watermark,
)


@pytest.fixture(scope="module")
def clip():
    return random_speaker(seed=5).say("aiueoaiueo", seconds=6.0, seed=5)


def test_mark_is_detected(clip):
    detection = detect_watermark(embed_watermark(clip))
    assert detection.present
    assert detection.confidence > DEFAULT_Z_THRESHOLD * 2
    assert detection.tag == "VOX1"


def test_mark_is_quiet(clip):
    marked = embed_watermark(clip)
    noise_power = np.mean((marked - clip).astype(np.float64) ** 2)
    snr_db = 10 * np.log10(np.mean(clip.astype(np.float64) ** 2) / noise_power)
    assert snr_db > 30, "the watermark must stay far below the signal"


def test_unmarked_audio_is_not_flagged(clip):
    assert not detect_watermark(clip).present


@pytest.mark.parametrize("seed", range(8))
def test_no_false_positives_across_voices_and_noise(seed):
    if seed % 2:
        signal = random_speaker(seed=seed).say("aiueo", 5.0, seed=seed)
    else:
        signal = np.random.default_rng(seed).standard_normal(SR * 5).astype(np.float32) * 0.1
    assert not detect_watermark(signal).present


def test_wrong_key_does_not_detect(clip):
    assert not detect_watermark(embed_watermark(clip), key="someone-elses-key").present


def test_wrong_tag_does_not_detect(clip):
    assert not detect_watermark(embed_watermark(clip, tag="AAAA"), tag="BBBB").present


@pytest.mark.parametrize("offset", [1, 777, 1234, 4095])
def test_detection_survives_arbitrary_cropping(clip, offset):
    """A trimmed clip puts the blocks out of phase; the scan must find them."""
    assert detect_watermark(embed_watermark(clip)[offset:]).present


def test_detection_survives_moderate_band_limiting(clip):
    marked = embed_watermark(clip)
    degraded = resample(resample(marked, SR, 12000), 12000, SR)
    assert detect_watermark(degraded).present


def test_telephone_bandwidth_destroys_the_mark(clip):
    """A documented limit, pinned so it cannot be quietly over-claimed later.

    Most of the chip energy lives above 4 kHz, so resampling through 8 kHz
    removes it. Measured across ten clips, only 2/10 still detect.
    """
    marked = embed_watermark(clip)
    degraded = resample(resample(marked, SR, 8000), 8000, SR)
    assert not detect_watermark(degraded).present


def test_detection_survives_added_noise(clip):
    marked = embed_watermark(clip)
    noisy = marked + np.random.default_rng(0).standard_normal(marked.size).astype(np.float32) * 0.01
    assert detect_watermark(noisy).present


def test_short_audio_is_handled():
    short = np.zeros(BLOCK // 2, dtype=np.float32)
    assert embed_watermark(short).size == short.size
    detection = detect_watermark(short)
    assert isinstance(detection, WatermarkDetection)
    assert not detection.present


def test_embedding_does_not_clip(clip):
    assert np.max(np.abs(embed_watermark(clip * 0.99))) <= 1.0


def test_detection_fields_are_json_native():
    """numpy.bool_ serialises as the string "True", which breaks API clients."""
    import json

    clip = random_speaker(seed=2).say("aiueoaiueo", 5.0, seed=2)
    payload = json.loads(json.dumps(detect_watermark(embed_watermark(clip)).as_dict()))
    assert payload["present"] is True
    assert isinstance(payload["confidence"], float)
