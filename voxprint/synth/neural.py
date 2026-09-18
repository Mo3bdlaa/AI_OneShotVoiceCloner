"""Neural cloning backends (optional).

These are the ones that actually clone a voice from a few seconds of audio. They
work because a large model has already learned, from thousands of speakers, how
a voice print maps onto acoustics -- which is knowledge no amount of signal
processing can substitute for.

Licensing, stated up front
--------------------------
XTTS-v2 ships under the **Coqui Public Model License (CPML)**, which does *not*
permit commercial use. YourTTS is likewise research-oriented. Read the licence
of whichever checkpoint you download before building anything on it; nothing in
this repository grants rights to those weights.

Coqui asks for that agreement interactively on first download. This backend will
not answer on your behalf: it checks for ``COQUI_TOS_AGREED=1`` and refuses with
an explanation if it is unset, rather than either accepting a licence for you or
hanging on a prompt nobody can see.

Verified behaviour
------------------
Measured on this codebase with ``torch 2.8.0+cpu``, XTTS-v2 on CPU:

* English, 44 characters -> 3.6 s of 24 kHz audio in 27 s (first call, includes
  warm-up); Arabic, 48 characters -> 4.6 s of audio in 6.6 s. Roughly 1.4x real
  time once warm, which is usable.
* Conditioning demonstrably works. Given references with median pitch 228 Hz and
  143 Hz, the generated speech came back at 230-258 Hz and 160 Hz respectively --
  it reproduces the reference's register rather than a default voice.
* Two generations from one reference score +0.81 against each other; two
  different references score +0.66. The register evidence above is the stronger
  signal, since identical text inflates both numbers.

Dependency constraints found the hard way, both pinned in
``requirements-neural.txt``:

* ``transformers>=5`` removes ``isin_mps_friendly``, which Coqui TTS imports.
  Installation succeeds and the import then fails.
* ``torch>=2.9`` drops torchaudio's built-in audio IO in favour of
  ``torchcodec``, which needs FFmpeg shared libraries present on the system.
  Without them the model loads and then fails when reading the reference file.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from ..audio import TARGET_SR, preprocess, save_audio
from .base import SynthResult, VoiceSynthesizer, register_synth

_INSTALL_HINT = (
    "This backend needs the optional neural extras:\n"
    "    pip install -r requirements-neural.txt\n"
    "The first run also downloads the model checkpoint (~2 GB for XTTS-v2)."
)

_LICENCE_HINT = (
    "This model's weights are covered by the Coqui Public Model License, which does not "
    "permit commercial use: https://coqui.ai/cpml\n"
    "voxprint will not agree to it for you. If you accept those terms, set:\n"
    "    export COQUI_TOS_AGREED=1"
)

#: Languages XTTS-v2 was trained on, including Arabic.
XTTS_LANGUAGES = (
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
)


class CoquiCloner(VoiceSynthesizer):
    """Multilingual zero-shot cloning through the Coqui TTS toolkit."""

    name = "xtts"
    version = "2"
    capabilities = frozenset({"tts", "vc"})
    sample_rate = 24_000
    notes = (
        "Zero-shot cloning from ~6s of reference audio, 17 languages including Arabic. "
        "Weights are CPML-licensed: non-commercial use only."
    )

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        sample_rate: int | None = None,
    ):
        self.model_name = model_name or os.environ.get("VOXPRINT_TTS_MODEL", self.DEFAULT_MODEL)
        self._device = device
        self._model: Any | None = None
        if sample_rate:
            self.sample_rate = int(sample_rate)

    # -- environment ------------------------------------------------------- #

    def available(self) -> tuple[bool, str]:
        """Report readiness, distinguishing *missing* from *broken*.

        These need different fixes and must not share a message. An import error
        from deep inside the package usually means a dependency conflict -- Coqui
        TTS against ``transformers>=5`` fails with a missing
        ``isin_mps_friendly``, for instance -- and telling that user to install
        the package they already have sends them in the wrong direction.
        """
        try:
            import TTS  # noqa: F401,PLC0415
        except ModuleNotFoundError:
            return False, "coqui TTS not installed (pip install -r requirements-neural.txt)"
        except ImportError as exc:
            return False, f"coqui TTS is installed but will not import -- likely a dependency conflict: {exc}"
        if not self._licence_agreed():
            return False, _LICENCE_HINT
        return True, "installed (model downloads on first use)"

    @staticmethod
    def _licence_agreed() -> bool:
        return os.environ.get("COQUI_TOS_AGREED", "").strip() in ("1", "true", "True", "yes")

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch  # noqa: PLC0415

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from TTS.api import TTS  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"coqui TTS is not installed.\n{_INSTALL_HINT}") from exc
        except ImportError as exc:
            raise RuntimeError(
                f"coqui TTS is installed but failed to import: {exc}\n"
                "This is usually a dependency conflict rather than a missing package. "
                "The pinned set in requirements-neural.txt is known to work."
            ) from exc
        model = TTS(self.model_name)
        # Older Coqui releases have no .to(); device selection is then implicit.
        if hasattr(model, "to"):
            model = model.to(self.device)
        self._model = model
        return model

    # -- generation -------------------------------------------------------- #

    def synthesize(
        self,
        text: str,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        *,
        language: str = "en",
        speed: float = 1.0,
        **kwargs,
    ) -> SynthResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("nothing to synthesize: text is empty")
        self._require_licence()
        if language not in XTTS_LANGUAGES and self.model_name == self.DEFAULT_MODEL:
            raise ValueError(
                f"language {language!r} is not supported by XTTS-v2. Supported: {', '.join(XTTS_LANGUAGES)}"
            )

        model = self._load()
        with _reference_file(reference, sr) as ref_path:
            wav = model.tts(text=text, speaker_wav=ref_path, language=language, speed=speed, **kwargs)

        out = np.asarray(wav, dtype=np.float32)
        return SynthResult(
            wav=out,
            sample_rate=self._model_sample_rate(),
            backend=self.name,
            info={
                "model": self.model_name,
                "language": language,
                "device": self.device,
                "characters": len(text),
            },
        )

    def convert(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        **kwargs,
    ) -> SynthResult:
        self._require_licence()
        model = self._load()
        with _reference_file(source, sr) as src_path, _reference_file(reference, sr) as ref_path:
            wav = model.voice_conversion(source_wav=src_path, target_wav=ref_path, **kwargs)
        return SynthResult(
            wav=np.asarray(wav, dtype=np.float32),
            sample_rate=self._model_sample_rate(),
            backend=self.name,
            info={"model": self.model_name, "device": self.device, "mode": "voice_conversion"},
        )

    def _require_licence(self) -> None:
        """Refuse rather than accept the model licence on the user's behalf.

        Coqui prompts for agreement on stdin, which in a non-interactive run
        hangs with no visible cause. Neither accepting silently nor hanging is
        acceptable, so this fails with the terms named.
        """
        if not self._licence_agreed():
            raise RuntimeError(_LICENCE_HINT)

    def _model_sample_rate(self) -> int:
        """Ask the loaded model for its output rate, falling back to the default."""
        try:
            return int(self._model.synthesizer.output_sample_rate)
        except Exception:  # pragma: no cover - depends on the model wrapper
            return self.sample_rate


class _ReferenceFile:
    """Write a reference waveform to a temp file, because Coqui wants a path."""

    def __init__(self, wav: np.ndarray, sr: int):
        self.wav = wav
        self.sr = sr
        self.path: str | None = None

    def __enter__(self) -> str:
        cleaned = preprocess(np.asarray(self.wav, dtype=np.float32), self.sr, do_vad=False)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxprint_ref_")
        os.close(fd)
        save_audio(path, cleaned, self.sr)
        self.path = path
        return path

    def __exit__(self, *exc) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


def _reference_file(wav: np.ndarray, sr: int) -> _ReferenceFile:
    return _ReferenceFile(wav, sr)


class CoquiVoiceConverter(VoiceSynthesizer):
    """Zero-shot any-to-any voice conversion: re-voice a recording, keep the words.

    This is the backend to reach for when you already *have* the audio -- someone
    singing, reading, talking -- and want it in a different voice. The recording
    supplies the words, the timing and the performance; only the timbre changes.
    Unlike text-to-speech cloning it needs no text and no language support, and
    unlike :class:`voxprint.synth.dspvc.DspVoiceConverter` it is a trained model
    and actually reaches the target identity.

    Measured on two LibriSpeech speakers, scoring the output against the target's
    ECAPA voice print (source scored -0.041 to the target and +0.918 to itself
    before conversion):

    ==============  ==========  ==========  ======
    model           -> target   -> source   time
    ==============  ==========  ==========  ======
    ``knnvc``           +0.645      +0.031     8 s
    ``openvoice``       +0.401      +0.265     6 s
    ``freevc``          +0.239      +0.161    33 s
    ==============  ==========  ==========  ======

    kNN-VC is the default: it moved the identity furthest *and* scrubbed the
    source identity most completely, in the least time. It works by replacing
    each frame's self-supervised feature with its nearest neighbours drawn from
    the target's own recordings, so more reference audio directly means a better
    match -- unlike encoder-based converters, which compress the reference into
    one vector and stop improving.

    On singing
    ----------
    All of these were trained on speech. They will process a sung recording, but
    sustained vowels, vibrato and a wider pitch range are out of domain, and the
    result degrades accordingly. A song also needs its vocal separated from the
    backing track first -- see :mod:`voxprint.song`. Systems built for singing
    (RVC, so-vits-svc) reach much higher quality but are not zero-shot: they need
    roughly ten minutes of the target voice and a training run.
    """

    name = "knnvc"
    version = "1"
    capabilities = frozenset({"vc"})
    sample_rate = 16_000
    notes = "Zero-shot voice conversion. Keeps the words and performance, changes the voice."

    DEFAULT_MODEL = "voice_conversion_models/multilingual/multi-dataset/knnvc"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._model: Any | None = None

    def available(self) -> tuple[bool, str]:
        try:
            import TTS  # noqa: F401,PLC0415
        except ModuleNotFoundError:
            return False, "coqui TTS not installed (pip install -r requirements-neural.txt)"
        except ImportError as exc:
            return False, f"coqui TTS is installed but will not import -- likely a dependency conflict: {exc}"
        return True, "installed (model downloads on first use)"

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch  # noqa: PLC0415

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from TTS.api import TTS  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"coqui TTS is not installed.\n{_INSTALL_HINT}") from exc
        model = TTS(self.model_name)
        if hasattr(model, "to"):
            model = model.to(self.device)
        self._model = model
        return model

    def convert(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        sr: int = TARGET_SR,
        **kwargs,
    ) -> SynthResult:
        model = self._load()
        with _reference_file(source, sr) as source_path, _reference_file(reference, sr) as target_path:
            wav = model.voice_conversion(source_wav=source_path, target_wav=target_path, **kwargs)

        out = np.asarray(wav, dtype=np.float32).squeeze()
        return SynthResult(
            wav=out,
            sample_rate=self._output_rate(),
            backend=self.name,
            info={
                "model": self.model_name,
                "device": self.device,
                "source_seconds": round(float(np.size(source)) / sr, 2),
                "reference_seconds": round(float(np.size(reference)) / sr, 2),
            },
        )

    def _output_rate(self) -> int:
        try:
            return int(self._model.voice_converter.output_sample_rate)
        except Exception:  # pragma: no cover - depends on the model wrapper
            return self.sample_rate


class FreeVcConverter(CoquiVoiceConverter):
    """FreeVC24 -- 24 kHz output, but measurably weaker identity transfer."""

    name = "freevc"
    sample_rate = 24_000
    notes = "24 kHz voice conversion. Cleaner audio than kNN-VC, reaches the target identity less well."
    DEFAULT_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"


class OpenVoiceConverter(CoquiVoiceConverter):
    """OpenVoice v2 tone-colour converter -- MIT-licensed weights."""

    name = "openvoice"
    sample_rate = 22_050
    notes = "Tone-colour conversion, MIT-licensed weights. Between kNN-VC and FreeVC on identity."
    DEFAULT_MODEL = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"


class YourTtsCloner(CoquiCloner):
    """YourTTS -- smaller and faster than XTTS, fewer languages, lower fidelity."""

    name = "yourtts"
    version = "1"
    sample_rate = 16_000
    notes = "Lighter zero-shot cloner (en/fr/pt). Faster on CPU, noticeably lower quality than XTTS-v2."

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/your_tts"

    def synthesize(self, text, reference, sr=TARGET_SR, *, language="en", **kwargs):
        # YourTTS uses its own language tags and has no speed control.
        kwargs.pop("speed", None)
        return super().synthesize(text, reference, sr, language=language, **kwargs)


@register_synth("knnvc")
def _make_knnvc(**kwargs) -> CoquiVoiceConverter:
    return CoquiVoiceConverter(**kwargs)


@register_synth("freevc")
def _make_freevc(**kwargs) -> FreeVcConverter:
    return FreeVcConverter(**kwargs)


@register_synth("openvoice")
def _make_openvoice(**kwargs) -> OpenVoiceConverter:
    return OpenVoiceConverter(**kwargs)


@register_synth("xtts")
def _make_xtts(**kwargs) -> CoquiCloner:
    return CoquiCloner(**kwargs)


@register_synth("yourtts")
def _make_yourtts(**kwargs) -> YourTtsCloner:
    return YourTtsCloner(**kwargs)
