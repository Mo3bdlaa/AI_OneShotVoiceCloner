"""High-level API: one object that ties the whole system together.

The CLI is a thin shell over this class, and so is any server or notebook that
wants to use the project. Everything that touches the filesystem lives here, so
the layers below stay pure: encoders take arrays, the gallery takes vectors.

Layout on disk
--------------
A :class:`VoiceLab` owns a directory::

    voices/
      gallery.npz          voice prints, metadata, consent, threshold
      audio/<id>.wav       reference audio, only if the speaker allowed it

Reference audio is kept separately and only on request, because the cloning
backends need real audio while identification needs only the embedding -- and an
embedding, unlike a recording, cannot be played back.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .audio import concat_with_gap, duration_seconds, load_audio, preprocess, save_audio
from .encoders import SpeakerEncoder, get_encoder
from .gallery import ConsentError, ConsentRecord, Gallery, GalleryError, VoicePrint
from .scoring import (
    CalibrationResult,
    IdentifyResult,
    PlattScaler,
    VerifyResult,
    calibrate,
    identify,
    verify,
)
from .synth import SynthResult, get_synth
from .watermark import embed_watermark

GALLERY_FILE = "gallery.npz"
AUDIO_DIR = "audio"
#: Reference audio kept per speaker for the cloning and conversion backends.
#: XTTS needs about six seconds and stops improving there, but kNN-VC matches each
#: frame against the target's own recordings, so it keeps improving with more.
#: Measured: 5s of reference converts to +0.53 against the target's voice print,
#: 10s to +0.65, 20s to +0.73, 40s to +0.78, and it plateaus after that.
MAX_REFERENCE_SECONDS = 60.0


@dataclass
class EnrollmentReport:
    """What enrolment produced, and what is wrong with it."""

    speaker_id: str
    files: list[str]
    seconds: float
    n_embeddings: int
    cohesion: float
    warnings: list[str] = field(default_factory=list)
    reference_audio: str | None = None

    def as_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "files": self.files,
            "seconds": round(self.seconds, 2),
            "embeddings": self.n_embeddings,
            "cohesion": None if np.isnan(self.cohesion) else round(self.cohesion, 4),
            "reference_audio": self.reference_audio,
            "warnings": self.warnings,
        }


class VoiceLab:
    """Enrolment, identification and imitation over one gallery directory."""

    def __init__(
        self,
        root: str = "voices",
        encoder: SpeakerEncoder | str = "dsp",
        *,
        require_consent: bool = True,
        auto_save: bool = True,
    ):
        self.root = os.fspath(root)
        self.encoder = get_encoder(encoder) if isinstance(encoder, str) else encoder
        self.auto_save = auto_save
        self.gallery = self._open_gallery(require_consent)
        self.scaler: PlattScaler | None = None

    # -- storage ----------------------------------------------------------- #

    @property
    def gallery_path(self) -> str:
        return os.path.join(self.root, GALLERY_FILE)

    @property
    def audio_dir(self) -> str:
        return os.path.join(self.root, AUDIO_DIR)

    def _open_gallery(self, require_consent: bool) -> Gallery:
        if os.path.exists(self.gallery_path):
            gallery = Gallery.load(self.gallery_path)
            if not gallery.spec.compatible_with(self.encoder.spec):
                raise GalleryError(
                    f"gallery at {self.gallery_path} was built with encoder "
                    f"{gallery.spec.name} v{gallery.spec.version} ({gallery.spec.dim}-dim), "
                    f"but this run uses {self.encoder.spec.name} v{self.encoder.spec.version} "
                    f"({self.encoder.spec.dim}-dim). Embeddings from different encoders are not "
                    "comparable -- re-enrol, or select the original encoder."
                )
            return gallery
        return Gallery(self.encoder.spec, require_consent=require_consent)

    def save(self) -> str:
        self.gallery.save(self.gallery_path)
        return self.gallery_path

    def _maybe_save(self) -> None:
        if self.auto_save:
            self.save()

    def reference_audio_path(self, speaker_id: str) -> str:
        return os.path.join(self.audio_dir, f"{_safe_name(speaker_id)}.wav")

    # -- embedding --------------------------------------------------------- #

    def embed_file(self, path: str, windowed: bool = True) -> tuple[np.ndarray, float]:
        """``(embeddings, seconds)`` for one audio file.

        Long files are embedded in overlapping windows and kept as several
        vectors rather than averaged immediately: the gallery can then measure
        how much they disagree, which is the enrolment quality signal.
        """
        wav, sr = load_audio(path, sr=self.encoder.sample_rate)
        speech = preprocess(wav, sr)
        seconds = duration_seconds(speech, sr)
        if seconds < 0.5:
            raise ValueError(f"{path}: only {seconds:.2f}s of speech detected after silence removal")
        vecs = self.encoder.embed_windows(speech, sr) if windowed else self.encoder.embed(speech, sr)[None, :]
        return np.atleast_2d(vecs), seconds

    # -- enrolment --------------------------------------------------------- #

    def enroll(
        self,
        speaker_id: str,
        paths: Sequence[str],
        *,
        consent: ConsentRecord | None = None,
        display_name: str = "",
        notes: str = "",
        keep_audio: bool = True,
        replace: bool = False,
    ) -> EnrollmentReport:
        """Register or extend a speaker from one or more recordings."""
        if not paths:
            raise ValueError("no audio files given")
        # Check consent before spending time on feature extraction: a refusal
        # after a minute of embedding is a worse experience than an instant one.
        existing = self.gallery.prints.get(speaker_id)
        if self.gallery.require_consent and consent is None and (existing is None or existing.consent is None):
            raise ConsentError(
                f"enrolling {speaker_id!r} requires a consent record. A voice print is biometric "
                "data and can drive the cloning backends; record who agreed and why."
            )

        vectors: list[np.ndarray] = []
        clips: list[np.ndarray] = []
        total = 0.0
        used: list[str] = []
        problems: list[str] = []

        for path in paths:
            try:
                vecs, seconds = self.embed_file(path)
            except Exception as exc:
                problems.append(f"{os.path.basename(path)}: {exc}")
                continue
            vectors.append(vecs)
            total += seconds
            used.append(path)
            if keep_audio:
                wav, sr = load_audio(path, sr=self.encoder.sample_rate)
                clips.append(preprocess(wav, sr, do_vad=False))

        if not vectors:
            raise ValueError("no usable audio; " + "; ".join(problems))

        stacked = np.vstack(vectors)
        print_ = self.gallery.enroll(
            speaker_id,
            stacked,
            display_name=display_name,
            consent=consent,
            seconds=total,
            sources=[os.path.basename(p) for p in used],
            notes=notes,
            replace=replace,
        )

        reference_path = self._store_reference(speaker_id, clips) if clips else None
        report = print_.quality_report()
        self._maybe_save()

        return EnrollmentReport(
            speaker_id=print_.speaker_id,
            files=used,
            seconds=total,
            n_embeddings=int(stacked.shape[0]),
            cohesion=print_.cohesion(),
            warnings=problems + report["warnings"],
            reference_audio=reference_path,
        )

    def _store_reference(self, speaker_id: str, clips: Iterable[np.ndarray]) -> str:
        """Keep a trimmed reference recording for the cloning backends."""
        merged = concat_with_gap(clips, self.encoder.sample_rate)
        limit = int(MAX_REFERENCE_SECONDS * self.encoder.sample_rate)
        path = self.reference_audio_path(speaker_id)
        save_audio(path, merged[:limit], self.encoder.sample_rate)
        return path

    def load_reference(self, speaker_id: str) -> tuple[np.ndarray, int]:
        """Read back a speaker's stored reference audio."""
        path = self.reference_audio_path(speaker_id)
        if not os.path.exists(path):
            raise GalleryError(
                f"no reference audio stored for {speaker_id!r}. Cloning needs real audio, not just "
                "the voice print -- re-enrol with keep_audio=True, or pass a reference file directly."
            )
        return load_audio(path, sr=self.encoder.sample_rate)

    def remove(self, speaker_id: str, delete_audio: bool = True) -> None:
        """Forget a speaker completely, audio included."""
        self.gallery.remove(speaker_id)
        path = self.reference_audio_path(speaker_id)
        if delete_audio and os.path.exists(path):
            os.remove(path)
        self._maybe_save()

    # -- recognition ------------------------------------------------------- #

    def identify_file(self, path: str, *, top_k: int = 5, threshold: float | None = None,
                      min_margin: float = 0.0) -> IdentifyResult:
        vecs, _ = self.embed_file(path)
        query = vecs.mean(axis=0)
        return identify(query, self.gallery, threshold=threshold, top_k=top_k,
                        min_margin=min_margin, scaler=self.scaler)

    def verify_file(self, path: str, speaker_id: str, *, threshold: float | None = None) -> VerifyResult:
        vecs, _ = self.embed_file(path)
        return verify(vecs.mean(axis=0), self.gallery, speaker_id,
                      threshold=threshold, scaler=self.scaler)

    def calibrate(
        self,
        *,
        criterion: str = "eer",
        max_far: float = 0.01,
        robust: bool = False,
        suite: list | None = None,
    ) -> CalibrationResult:
        """Measure score distributions, set the threshold, fit the probability map.

        With ``robust=True`` the trials also include degraded copies of the
        enrolment audio -- noise, reverberation, telephone bandwidth, a different
        microphone response. A clean-only calibration produces a threshold that is
        valid only for queries recorded exactly like the enrolment was; measured
        on synthetic speakers, such a threshold accepts 97% of clean queries and
        3% of the same queries at 40 dB SNR, even though the encoder still ranks
        the right speaker first every time. Showing calibration the mismatch is
        what fixes that.

        It costs accuracy on clean audio -- a lower threshold accepts more
        impostors -- so it is opt-in, and the returned result reports both the
        error rates and the conditions they were measured under.
        """
        self.gallery.refit_standardizer()
        result = calibrate(self.gallery, criterion=criterion, max_far=max_far)

        if robust:
            result = self._widen_calibration(result, criterion=criterion, max_far=max_far, suite=suite)

        # An unusable calibration must not leave a threshold behind: `identify`
        # would then report `calibrated: true` for a number nothing measured.
        if result.usable:
            self.gallery.threshold = result.threshold
            self.gallery.identification_threshold = result.identification_threshold
            self.scaler = PlattScaler.fit(result.target_scores, result.impostor_scores)
            self._maybe_save()
        return result

    def _widen_calibration(
        self,
        clean: CalibrationResult,
        *,
        criterion: str,
        max_far: float,
        suite: list | None = None,
    ) -> CalibrationResult:
        """Re-derive the threshold with degraded copies of the enrolment audio."""
        from .augment import default_suite
        from .scoring import equal_error_rate, threshold_for_far

        suite = default_suite() if suite is None else suite
        ids, centroids = self.gallery.centroid_matrix(standardize=True)
        index = {sid: i for i, sid in enumerate(ids)}

        target = [clean.target_scores]
        impostor = [clean.impostor_scores]
        conditions: list[str] = []
        skipped: list[str] = []

        for sid in ids:
            path = self.reference_audio_path(sid)
            if not os.path.exists(path):
                continue
            wav, sr = load_audio(path, sr=self.encoder.sample_rate)
            # A few seconds is enough, and keeps a robust calibration quick.
            wav = wav[: int(self.encoder.sample_rate * 8.0)]
            for augmentation in suite:
                try:
                    degraded = augmentation(wav, sr, seed=_stable_seed(sid))
                    vec = self.gallery.standardizer.transform(self.encoder.embed(degraded, sr))
                except (ValueError, FloatingPointError) as exc:
                    # A degradation can legitimately destroy a clip -- heavy
                    # band-limiting on a quiet take leaves too little speech to
                    # embed. That is worth recording, not worth failing on.
                    # Anything else is a bug and must not be swallowed: a blanket
                    # handler here once hid a NameError and silently downgraded
                    # every robust calibration to a clean one.
                    skipped.append(f"{sid}/{augmentation.name}: {exc}")
                    continue
                scores = centroids @ vec
                own = index[sid]
                target.append(np.array([scores[own]]))
                impostor.append(np.delete(scores, own))
                if augmentation.name not in conditions:
                    conditions.append(augmentation.name)

        all_target = np.concatenate(target)
        all_impostor = np.concatenate(impostor)
        eer, eer_threshold = equal_error_rate(all_target, all_impostor)
        threshold = eer_threshold if criterion == "eer" else threshold_for_far(all_target, all_impostor, max_far)

        warnings = list(clean.warnings)
        if skipped:
            warnings.append(
                f"{len(skipped)} degraded trial(s) could not be embedded: {'; '.join(skipped[:3])}"
            )
        if not conditions:
            warnings.append(
                "no reference audio stored for any speaker, so no degraded trials could be built; "
                "this is the clean calibration. Re-enrol with keep_audio=True."
            )

        return CalibrationResult(
            threshold=float(threshold),
            eer=eer,
            n_target=int(all_target.size),
            n_impostor=int(all_impostor.size),
            target_scores=all_target,
            impostor_scores=all_impostor,
            criterion=f"{criterion}+robust" if conditions else criterion,
            warnings=warnings,
            conditions=conditions,
        )

    # -- imitation --------------------------------------------------------- #

    def convert_file(
        self,
        source_path: str,
        speaker_id: str | None = None,
        *,
        reference_path: str | None = None,
        backend: str = "dspvc",
        watermark: bool = True,
        **kwargs,
    ) -> SynthResult:
        """Re-voice an existing recording toward an enrolled (or supplied) voice."""
        synth = get_synth(backend)
        if not synth.supports("vc"):
            raise ValueError(f"backend {backend!r} does not support voice conversion")
        _require_ready(synth)

        source, sr = load_audio(source_path, sr=synth.sample_rate)
        reference, _ = self._resolve_reference(speaker_id, reference_path, synth.sample_rate)
        result = synth.convert(source, reference, sr, **kwargs)
        return self._finish(result, watermark)

    def speak(
        self,
        text: str,
        speaker_id: str | None = None,
        *,
        reference_path: str | None = None,
        backend: str = "xtts",
        language: str = "en",
        watermark: bool = True,
        **kwargs,
    ) -> SynthResult:
        """Synthesise ``text`` in an enrolled (or supplied) voice."""
        synth = get_synth(backend)
        if not synth.supports("tts"):
            ready = [n for n in ("xtts", "yourtts") if get_synth(n).available()[0]]
            raise ValueError(
                f"backend {backend!r} cannot generate speech from text. Text-to-speech needs a "
                f"trained model; install the neural extras and use one of: {', '.join(ready) or 'xtts, yourtts'}"
            )
        _require_ready(synth)
        reference, _ = self._resolve_reference(speaker_id, reference_path, synth.sample_rate)
        result = synth.synthesize(text, reference, synth.sample_rate, language=language, **kwargs)
        return self._finish(result, watermark)

    def _resolve_reference(
        self, speaker_id: str | None, reference_path: str | None, sr: int
    ) -> tuple[np.ndarray, int]:
        if reference_path:
            return load_audio(reference_path, sr=sr)
        if speaker_id:
            wav, _ = self.load_reference(speaker_id)
            from .audio import resample  # local import keeps the module graph flat

            return (wav if sr == self.encoder.sample_rate else resample(wav, self.encoder.sample_rate, sr)), sr
        raise ValueError("need either an enrolled speaker_id or a reference audio file")

    def _finish(self, result: SynthResult, watermark: bool) -> SynthResult:
        """Mark generated audio, and measure how close it actually got."""
        wav = embed_watermark(result.wav) if watermark else result.wav
        info = dict(result.info)
        info["watermarked"] = watermark
        return SynthResult(wav=wav, sample_rate=result.sample_rate, backend=result.backend, info=info)

    def similarity_to(self, wav: np.ndarray, sr: int, speaker_id: str) -> float:
        """Score generated audio against an enrolled print.

        Worth calling on anything a cloning backend produces: it answers "would
        this fool the recogniser?" with a number instead of an impression.
        """
        from .audio import resample

        if sr != self.encoder.sample_rate:
            wav = resample(wav, sr, self.encoder.sample_rate)
        vec = self.encoder.embed(wav, self.encoder.sample_rate)
        return verify(vec, self.gallery, speaker_id).score

    # -- reporting --------------------------------------------------------- #

    def summary(self) -> dict:
        info = self.gallery.summary()
        info["root"] = self.root
        info["gallery_file"] = self.gallery_path if os.path.exists(self.gallery_path) else None
        return info

    def speakers(self) -> list[VoicePrint]:
        return [self.gallery.prints[i] for i in self.gallery.ids()]


def _stable_seed(name: str) -> int:
    """A per-speaker seed that survives a restart.

    ``hash()`` on a string is salted per interpreter process unless
    PYTHONHASHSEED is set, so using it here made robust calibration produce a
    different threshold on every run from identical inputs.
    """
    return zlib.crc32(name.encode("utf-8")) % 10_000


def _require_ready(synth) -> None:
    """Fail with the backend's own explanation rather than an import traceback."""
    ready, reason = synth.available()
    if not ready:
        raise RuntimeError(f"backend {synth.name!r} is not usable: {reason}")


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
