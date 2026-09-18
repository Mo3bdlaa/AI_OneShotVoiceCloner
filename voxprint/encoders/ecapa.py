"""Pretrained neural speaker encoders (optional backends).

These are the accurate ones. Unlike the DSP encoder they were trained
discriminatively on thousands of speakers, so the embedding space is organised
*by identity* rather than by whatever acoustics happen to correlate with it.
Published equal-error rates on VoxCeleb1-O are around 1% for ECAPA-TDNN versus
the double-digit range a statistics-based encoder reaches -- an order of
magnitude, not a tweak.

The cost is a model download (~80 MB) and a PyTorch dependency, which is why
they are optional and imported lazily. Nothing else in the package changes:
the gallery, scoring and CLI are identical whichever encoder is selected.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..audio import TARGET_SR
from .base import SpeakerEncoder, register_encoder

_INSTALL_HINT = (
    "This backend needs the optional neural extras:\n"
    "    pip install -r requirements-neural.txt\n"
    "or  pip install 'voxprint[neural]'"
)


class EcapaSpeakerEncoder(SpeakerEncoder):
    """ECAPA-TDNN x-vector encoder (SpeechBrain, trained on VoxCeleb)."""

    name = "ecapa"
    version = "1"
    dim = 192
    sample_rate = TARGET_SR
    min_speech_seconds = 1.0

    DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, source: str | None = None, device: str | None = None, cache_dir: str | None = None):
        self.source = source or os.environ.get("VOXPRINT_ECAPA_SOURCE", self.DEFAULT_SOURCE)
        self.cache_dir = cache_dir or os.environ.get("VOXPRINT_MODEL_DIR", "models/ecapa")
        self._device = device
        self._model: Any | None = None

    # -- lazy model loading ------------------------------------------------ #

    def _torch(self):
        try:
            import torch  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"PyTorch is not installed.\n{_INSTALL_HINT}") from exc
        return torch

    def available(self) -> tuple[bool, str]:
        try:
            import speechbrain  # noqa: F401,PLC0415
            import torch  # noqa: F401,PLC0415
        except ModuleNotFoundError as exc:
            return False, f"{exc.name} not installed (pip install -r requirements-neural.txt)"
        except ImportError as exc:
            return False, f"installed but will not import -- likely a dependency conflict: {exc}"
        return True, "installed (model downloads on first use)"

    @property
    def device(self) -> str:
        if self._device is None:
            torch = self._torch()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load(self):
        """Fetch the pretrained model once, on first use."""
        if self._model is not None:
            return self._model
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415
        except ImportError as primary:
            try:  # SpeechBrain < 1.0 kept it under a different path
                from speechbrain.pretrained import EncoderClassifier  # noqa: PLC0415
            except ModuleNotFoundError as exc:
                raise RuntimeError(f"speechbrain is not installed.\n{_INSTALL_HINT}") from exc
            except ImportError as exc:
                # Installed but unimportable -- a dependency conflict, which needs
                # a different fix from a missing package.
                raise RuntimeError(
                    f"speechbrain is installed but failed to import: {primary}\n"
                    "This is usually a dependency conflict rather than a missing package."
                ) from exc

        self._model = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=self.cache_dir,
            run_opts={"device": self.device},
        )
        return self._model

    # -- embedding --------------------------------------------------------- #

    def _embed_prepared(self, wav: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch()
        model = self._load()
        tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model.encode_batch(tensor)
        return emb.squeeze().detach().cpu().numpy().astype(np.float64)

    def describe(self) -> dict:
        info = super().describe()
        info["source"] = self.source
        info["cache_dir"] = self.cache_dir
        info["loaded"] = self._model is not None
        return info


class ResemblyzerEncoder(SpeakerEncoder):
    """GE2E d-vector encoder (Resemblyzer) -- lighter than ECAPA, CPU-friendly."""

    name = "resemblyzer"
    version = "1"
    dim = 256
    sample_rate = TARGET_SR
    min_speech_seconds = 1.0

    def __init__(self, device: str | None = None):
        self._device = device
        self._model: Any | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from resemblyzer import VoiceEncoder  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(f"resemblyzer is not installed.\n{_INSTALL_HINT}") from exc
        self._model = VoiceEncoder(device=self._device) if self._device else VoiceEncoder()
        return self._model

    def _embed_prepared(self, wav: np.ndarray, sr: int) -> np.ndarray:
        model = self._load()
        return np.asarray(model.embed_utterance(np.asarray(wav, dtype=np.float32)), dtype=np.float64)


@register_encoder("ecapa")
def _make_ecapa(**kwargs) -> EcapaSpeakerEncoder:
    return EcapaSpeakerEncoder(**kwargs)


@register_encoder("resemblyzer")
def _make_resemblyzer(**kwargs) -> ResemblyzerEncoder:
    return ResemblyzerEncoder(**kwargs)
