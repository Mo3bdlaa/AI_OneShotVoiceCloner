"""Shared fixtures.

All test audio is generated from :mod:`voxprint.synthetic`, so the suite is
deterministic, needs no downloads, and ships no recordings of real people.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxprint.audio import TARGET_SR, save_audio
from voxprint.encoders import get_encoder
from voxprint.synthetic import CAST, PHRASES, cast_by_name

SR = TARGET_SR


@pytest.fixture(scope="session")
def encoder():
    return get_encoder("dsp")


@pytest.fixture(scope="session")
def cast():
    return cast_by_name()


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Four takes per cast member on disk: three for enrolment, one held out."""
    directory = tmp_path_factory.mktemp("clips")
    paths: dict[str, list[str]] = {}
    for speaker in CAST:
        files = []
        for i, phrase in enumerate(PHRASES[:4]):
            path = directory / f"{speaker.name}_{i}.wav"
            save_audio(path, speaker.say(phrase, seconds=3.0, seed=4000 + i), SR)
            files.append(str(path))
        paths[speaker.name] = files
    return paths


@pytest.fixture
def tone():
    """A 2-second, 150 Hz harmonic tone -- a signal with a known exact pitch."""
    t = np.arange(SR * 2) / SR
    wav = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
    return wav.astype(np.float32)
