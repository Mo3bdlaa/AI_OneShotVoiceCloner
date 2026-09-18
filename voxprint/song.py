"""Re-voicing a song: separate the vocal, convert it, put it back.

A recording with music in it cannot be fed to a voice converter directly. The
converter would try to re-voice the drums and the guitar along with the singer,
and the result is unusable. So the vocal has to come out first, be converted on
its own, and be mixed back over the untouched instrumental.

    song  ->  [separator]  ->  vocal  ->  [voice converter]  ->  new vocal
                      \\->  instrumental  ------------------->  +  =  new song

Separation uses Demucs when it is installed. It is an optional dependency for the
same reason the neural backends are: a large model download that most users of
the fingerprinting half will never need.

What to expect
--------------
Speak into a microphone and this pipeline works as well as the converter does.
**Sing, and it does not.** Every zero-shot converter available here was trained
on speech; sustained vowels, vibrato and a two-octave range are outside what they
have seen, and they degrade audibly on singing. Separation adds its own artefacts
on top, which conversion then amplifies.

Systems that actually do singing well -- RVC, so-vits-svc -- are not zero-shot.
They need roughly ten minutes of the target voice and a training run, and that is
the honest price of a convincing sung result. This module is the right pipeline
for that too: swap the converter, keep the separation and the remix.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .audio import TARGET_SR, load_audio, rms_normalize, save_audio


class SeparationError(RuntimeError):
    """Raised when the vocal cannot be separated from the backing track."""


@dataclass
class Stems:
    """A song split into the part to convert and the part to leave alone."""

    vocal: np.ndarray
    instrumental: np.ndarray
    sample_rate: int
    method: str
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.vocal.size) / float(self.sample_rate)


# --------------------------------------------------------------------------- #
# Separation
# --------------------------------------------------------------------------- #

def demucs_available() -> tuple[bool, str]:
    """``(ready, reason)`` for the Demucs separator."""
    try:
        import demucs  # noqa: F401,PLC0415
    except ModuleNotFoundError:
        return False, "demucs not installed (pip install demucs)"
    except ImportError as exc:
        return False, f"demucs is installed but will not import: {exc}"
    return True, "installed (model downloads on first use)"


def separate(
    path: str | os.PathLike[str],
    sr: int = TARGET_SR,
    model: str = "htdemucs",
    device: str | None = None,
) -> Stems:
    """Split a song into vocal and instrumental using Demucs.

    Demucs is invoked as a subprocess rather than through its Python API: the API
    surface has changed repeatedly between releases, while the command-line
    interface has stayed stable, and a separation that takes minutes is not worth
    coupling tightly to a moving import path.
    """
    ready, reason = demucs_available()
    if not ready:
        raise SeparationError(
            f"cannot separate the vocal from the backing track: {reason}\n"
            "Install it with `pip install demucs`, or pass an isolated vocal track "
            "and skip separation."
        )

    workdir = tempfile.mkdtemp(prefix="voxprint_demucs_")
    try:
        command = [
            # sys.executable, not "python": in a virtualenv the two differ, and
            # PATH's python is the one *without* demucs installed.
            sys.executable, "-m", "demucs.separate",
            "--two-stems", "vocals",
            "-n", model,
            "-o", workdir,
            os.fspath(path),
        ]
        if device:
            command[-1:-1] = ["-d", device]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise SeparationError(
                f"demucs failed (exit {result.returncode}):\n{result.stderr.strip()[-600:]}"
            )

        stem_dir = _find_stems(workdir)
        vocal, _ = load_audio(os.path.join(stem_dir, "vocals.wav"), sr=sr)
        instrumental, _ = load_audio(os.path.join(stem_dir, "no_vocals.wav"), sr=sr)
        return Stems(
            vocal=vocal,
            instrumental=instrumental,
            sample_rate=sr,
            method=f"demucs:{model}",
            info={"source": os.fspath(path)},
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _find_stems(workdir: str) -> str:
    """Locate the directory demucs wrote its stems into."""
    for root, _dirs, files in os.walk(workdir):
        if "vocals.wav" in files and "no_vocals.wav" in files:
            return root
    raise SeparationError(f"demucs produced no stems under {workdir}")


# --------------------------------------------------------------------------- #
# Re-voicing
# --------------------------------------------------------------------------- #

def remix(vocal: np.ndarray, instrumental: np.ndarray, vocal_gain_db: float = 0.0) -> np.ndarray:
    """Mix a converted vocal back over the instrumental, matching lengths."""
    length = max(vocal.size, instrumental.size)
    voc = np.pad(vocal, (0, length - vocal.size))
    inst = np.pad(instrumental, (0, length - instrumental.size))
    mixed = inst + voc * (10.0 ** (vocal_gain_db / 20.0))

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def revoice_song(
    song_path: str | os.PathLike[str],
    reference: np.ndarray,
    reference_sr: int,
    *,
    converter,
    separate_vocals: bool = True,
    vocal_gain_db: float = 0.0,
    sr: int = TARGET_SR,
    demucs_model: str = "htdemucs",
) -> tuple[np.ndarray, int, dict]:
    """Re-voice the singing (or speech) in a recording, keeping the backing track.

    ``converter`` is any :class:`voxprint.synth.base.VoiceSynthesizer` supporting
    ``vc``. Returns ``(wav, sample_rate, info)``.
    """
    if not converter.supports("vc"):
        raise ValueError(f"backend {converter.name!r} cannot do voice conversion")

    if separate_vocals:
        stems = separate(song_path, sr=sr, model=demucs_model)
        vocal, instrumental = stems.vocal, stems.instrumental
        method = stems.method
    else:
        vocal, _ = load_audio(song_path, sr=sr)
        instrumental = np.zeros(0, dtype=np.float32)
        method = "none (input treated as an isolated vocal)"

    result = converter.convert(rms_normalize(vocal), reference, sr)
    converted = result.wav
    out_sr = result.sample_rate

    if instrumental.size:
        from .audio import resample  # local import keeps the module graph flat

        backing = resample(instrumental, sr, out_sr) if out_sr != sr else instrumental
        mixed = remix(converted, backing, vocal_gain_db)
    else:
        mixed = converted

    info = {
        "separation": method,
        "backend": result.backend,
        "vocal_seconds": round(float(vocal.size) / sr, 2),
        "vocal_gain_db": vocal_gain_db,
        **result.info,
    }
    return mixed, out_sr, info


def write_stems(stems: Stems, directory: str | os.PathLike[str]) -> dict[str, str]:
    """Write the separated stems, for inspecting a bad separation."""
    os.makedirs(directory, exist_ok=True)
    paths = {
        "vocal": os.path.join(os.fspath(directory), "vocal.wav"),
        "instrumental": os.path.join(os.fspath(directory), "instrumental.wav"),
    }
    save_audio(paths["vocal"], stems.vocal, stems.sample_rate)
    save_audio(paths["instrumental"], stems.instrumental, stems.sample_rate)
    return paths
