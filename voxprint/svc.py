"""so-vits-svc: singing voice conversion with a model trained per voice.

Why this is separate from :mod:`voxprint.synth`
-----------------------------------------------
Every other converter in this project is *zero-shot*: hand it a few seconds of
the target and it re-voices anything. so-vits-svc is not. It trains a model on
one person's recordings, and that model is the voice. The trade is deliberate and
it is the right one for singing:

* **Zero-shot converters were trained on speech.** Sustained vowels, vibrato and
  a two-octave range are outside what they have seen, and they degrade audibly on
  singing.
* **so-vits-svc conditions on F0 explicitly** and is routinely trained on sung
  material, so the melody survives -- provided ``auto_predict_f0`` is left off,
  which is why :func:`infer` defaults it to off. Turning it on lets the model
  re-invent the pitch contour, which is right for speech and ruins a song.

The cost is real: roughly ten minutes of clean solo recordings of the target, and
a training run that wants a GPU. This module wraps the ``so-vits-svc-fork``
command line rather than importing it, for the same reason
:mod:`voxprint.song` shells out to Demucs -- the CLI is the stable surface.

Measured cost on this machine
-----------------------------
See :func:`train` for the numbers. They are from a CPU-only box and are the
argument for using a GPU, not against the method.

Consent
-------
Training a model on someone's voice is a larger act than storing an embedding of
it: the artefact can generate unlimited new audio in that voice, indefinitely.
:func:`train_for_speaker` therefore refuses unless the speaker is already
enrolled with a consent record, and writes that record next to the model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus"}
#: Below this, a trained model mostly reproduces the training clips rather than
#: the voice. The project's own documentation asks for ten minutes or more.
MIN_TRAINING_MINUTES = 8.0


class SvcError(RuntimeError):
    """Raised when a so-vits-svc step cannot run or fails."""


def available() -> tuple[bool, str]:
    """``(ready, reason)`` for the so-vits-svc toolchain."""
    try:
        import so_vits_svc_fork  # noqa: F401,PLC0415
    except ModuleNotFoundError:
        return False, "so-vits-svc-fork not installed (pip install so-vits-svc-fork)"
    except ImportError as exc:
        return False, f"so-vits-svc-fork is installed but will not import: {exc}"
    if shutil.which("svc") is None and not _svc_entrypoint().exists():
        return False, "so-vits-svc-fork is installed but its `svc` command was not found"
    return True, "installed"


def compat_patched() -> bool:
    """Whether the NumPy compatibility shim can be applied (see :mod:`voxprint._svc_compat`)."""
    from ._svc_compat import apply

    return apply()


def _svc_entrypoint() -> Path:
    """The ``svc`` script next to the running interpreter.

    Looked up relative to ``sys.executable`` rather than through PATH: inside a
    virtualenv those differ, and PATH's may belong to a different environment.
    """
    return Path(sys.executable).parent / "svc"


def _run(args: list[str], cwd: Path, step: str, timeout: float | None = None) -> str:
    # Invoked through voxprint._svc_compat rather than the `svc` script: it
    # installs a NumPy compatibility patch and then hands over to the same CLI.
    # Without it, training dies at its first logging step on any current NumPy.
    command = [sys.executable, "-m", "voxprint._svc_compat", *args]
    # so-vits-svc runs with cwd set to its own workspace, so an uninstalled
    # checkout of voxprint would not be importable there and the shim would be
    # skipped -- silently, since the upstream CLI is what actually runs.
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH", "")]))
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        raise SvcError(f"{step} exceeded its {timeout:.0f}s budget") from exc
    if result.returncode != 0:
        raise SvcError(f"{step} failed (exit {result.returncode}):\n{result.stderr.strip()[-800:]}")
    return result.stdout


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #

@dataclass
class SvcWorkspace:
    """The directory layout so-vits-svc expects, rooted anywhere."""

    root: Path
    speaker: str

    @property
    def dataset_raw(self) -> Path:
        return self.root / "dataset_raw" / self.speaker

    @property
    def config_path(self) -> Path:
        return self.root / "configs" / "44k" / "config.json"

    @property
    def model_dir(self) -> Path:
        return self.root / "logs" / "44k"

    def checkpoints(self) -> list[Path]:
        """Generator checkpoints, newest last."""
        if not self.model_dir.exists():
            return []
        return sorted(
            (p for p in self.model_dir.glob("G_*.pth") if p.stem != "G_0"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
        )

    def latest_checkpoint(self) -> Path | None:
        found = self.checkpoints()
        return found[-1] if found else None

    def is_trained(self) -> bool:
        return self.config_path.exists() and self.latest_checkpoint() is not None


def stage_audio(workspace: SvcWorkspace, paths: list[str | os.PathLike[str]]) -> float:
    """Copy training audio into the workspace; returns total minutes."""
    import soundfile as sf

    workspace.dataset_raw.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for index, path in enumerate(paths):
        source = Path(path)
        if source.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            total += sf.info(str(source)).duration
        except Exception as exc:
            raise SvcError(f"cannot read {source}: {exc}") from exc
        shutil.copy(source, workspace.dataset_raw / f"{index:04d}{source.suffix}")
    return total / 60.0


def collect_audio(directory: str | os.PathLike[str]) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def preprocess(workspace: SvcWorkspace, timeout: float | None = 7200) -> dict:
    """Resample, write the config, and extract HuBERT features and F0.

    The third step downloads a ~360 MB content encoder on first run.
    """
    ready, reason = available()
    if not ready:
        raise SvcError(reason)

    steps = {}
    for step, args in (
        ("pre-resample", ["pre-resample"]),
        ("pre-config", ["pre-config"]),
        ("pre-hubert", ["pre-hubert"]),
    ):
        _run(args, workspace.root, step, timeout=timeout)
        steps[step] = "ok"

    if not workspace.config_path.exists():
        raise SvcError(f"preprocessing produced no config at {workspace.config_path}")
    return steps


def set_training_budget(
    workspace: SvcWorkspace,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_interval: int | None = None,
    log_interval: int | None = None,
) -> dict:
    """Edit the generated config before training.

    Exposed because the defaults assume a GPU and 10 000 steps. On a CPU that is
    days, so a caller needs to be able to say "give me something in an hour" and
    understand that it will sound like an hour of training.
    """
    config = json.loads(workspace.config_path.read_text())
    train = config.setdefault("train", {})
    for key, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("eval_interval", eval_interval),
        ("log_interval", log_interval),
    ):
        if value is not None:
            train[key] = value
    workspace.config_path.write_text(json.dumps(config, indent=2))
    return train


def train(workspace: SvcWorkspace, timeout: float | None = None) -> Path:
    """Run training. Returns the newest checkpoint.

    Measured on this 4-core CPU box with 8.2 minutes of 44.1 kHz training audio,
    preprocessing (resample, config, HuBERT and F0 extraction) took 54 seconds.
    Training is the expensive part: upstream asks for 10 000+ steps, which is
    hours on a mid-range GPU and days here. Treat CPU training as a way to verify
    the pipeline runs, not to produce a voice you would use.

    ``eval_interval`` in the config decides when a checkpoint is written. Leaving
    it at the default of 200 while training for fewer steps than that produces a
    run that finishes having saved nothing, which looks exactly like a failure.
    :func:`set_training_budget` is the place to lower it.
    """
    ready, reason = available()
    if not ready:
        raise SvcError(reason)
    if not workspace.config_path.exists():
        raise SvcError("no config found; run preprocess() first")

    _run(["train", "-c", str(workspace.config_path), "-m", str(workspace.model_dir)],
         workspace.root, "train", timeout=timeout)

    checkpoint = workspace.latest_checkpoint()
    if checkpoint is None:
        raise SvcError(f"training finished but wrote no checkpoint into {workspace.model_dir}")
    return checkpoint


def infer(
    workspace: SvcWorkspace,
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    transpose: int = 0,
    auto_predict_f0: bool = False,
    f0_method: str = "dio",
    noise_scale: float = 0.4,
    device: str = "cpu",
    timeout: float | None = 3600,
) -> Path:
    """Convert a recording with the trained model.

    ``auto_predict_f0`` defaults to **off**, which is the setting for singing:
    leaving it on lets the model predict its own pitch contour and the melody
    goes with it. Turn it on for speech, where a re-predicted contour sounds more
    natural than a transplanted one.

    ``transpose`` shifts the key in semitones, for when the source and the target
    do not share a comfortable range -- a soprano part sung by a bass model needs
    it, and no amount of conversion quality substitutes.
    """
    if not workspace.is_trained():
        raise SvcError(
            f"no trained model in {workspace.model_dir}. so-vits-svc is not zero-shot: "
            "train one with `voxprint train-svc` before converting."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "infer", str(Path(input_path).resolve()),
        "-o", str(output.resolve()),
        "-m", str(workspace.model_dir.resolve()),
        "-c", str(workspace.config_path.resolve()),
        "-s", workspace.speaker,
        "-t", str(int(transpose)),
        "-fm", f0_method,
        "-n", str(noise_scale),
        "-d", device,
        "-a" if auto_predict_f0 else "-na",
    ]
    _run(args, workspace.root, "infer", timeout=timeout)
    if not output.exists():
        raise SvcError(f"inference reported success but wrote nothing to {output}")
    return output


# --------------------------------------------------------------------------- #
# Gallery integration
# --------------------------------------------------------------------------- #

@dataclass
class TrainingReport:
    """What a training run consumed and produced."""

    speaker: str
    workspace: Path
    minutes_of_audio: float
    checkpoint: Path | None
    config: Path
    budget: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "workspace": str(self.workspace),
            "minutes_of_audio": round(self.minutes_of_audio, 2),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "config": str(self.config),
            "budget": self.budget,
            "warnings": self.warnings,
        }


def train_for_speaker(
    lab,
    speaker_id: str,
    audio_dir: str | os.PathLike[str],
    *,
    workspace_root: str | os.PathLike[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    skip_training: bool = False,
    timeout: float | None = None,
) -> TrainingReport:
    """Train a so-vits-svc model for an enrolled speaker and register it.

    Requires the speaker to be enrolled *with consent*: the model that comes out
    of this can generate unlimited audio in their voice, which is a larger thing
    to hold than an embedding.
    """
    print_ = lab.gallery.get(speaker_id)
    if lab.gallery.require_consent and print_.consent is None:
        raise SvcError(
            f"{speaker_id!r} has no consent record. Training a voice model produces an artefact "
            "that can generate unlimited speech in this person's voice; enrol them with a consent "
            "statement first."
        )

    root = Path(workspace_root) if workspace_root else Path(lab.root) / "svc" / _safe(speaker_id)
    root.mkdir(parents=True, exist_ok=True)
    workspace = SvcWorkspace(root=root, speaker=_safe(speaker_id))

    files = collect_audio(audio_dir)
    if not files:
        raise SvcError(f"no audio found under {audio_dir}")
    minutes = stage_audio(workspace, files)

    warnings: list[str] = []
    if minutes < MIN_TRAINING_MINUTES:
        warnings.append(
            f"only {minutes:.1f} minutes of training audio; so-vits-svc wants "
            f"{MIN_TRAINING_MINUTES:.0f}+ and the result degrades sharply below that"
        )

    preprocess(workspace)
    budget = set_training_budget(workspace, epochs=epochs, batch_size=batch_size)

    checkpoint = None
    if not skip_training:
        checkpoint = train(workspace, timeout=timeout)
        register_model(lab, speaker_id, workspace)

    return TrainingReport(
        speaker=speaker_id,
        workspace=root,
        minutes_of_audio=minutes,
        checkpoint=checkpoint,
        config=workspace.config_path,
        budget=budget,
        warnings=warnings,
    )


def register_model(lab, speaker_id: str, workspace: SvcWorkspace) -> None:
    """Record the trained model on the speaker's voice print."""
    print_ = lab.gallery.get(speaker_id)
    print_.models["sovits"] = str(workspace.root)
    lab.save()


def workspace_for(lab, speaker_id: str) -> SvcWorkspace | None:
    """The registered so-vits-svc workspace for a speaker, if there is one."""
    root = lab.gallery.get(speaker_id).models.get("sovits")
    if not root:
        return None
    return SvcWorkspace(root=Path(root), speaker=_safe(speaker_id))


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #

#: Per-OS recipes for making the converted audio appear as a microphone to other
#: applications. The routing is an operating-system job, not something any Python
#: package can do for you: the converter writes to an output device, and a
#: virtual cable makes that device readable as an input device elsewhere.
VIRTUAL_MIC_SETUP = {
    "linux": (
        "PulseAudio / PipeWire:\n"
        "    pactl load-module module-null-sink sink_name=voxprint \\\n"
        "          sink_properties=device.description=voxprint\n"
        "    pactl load-module module-remap-source master=voxprint.monitor \\\n"
        "          source_name=voxprint_mic source_properties=device.description='voxprint mic'\n"
        "  Send the converter's output to the `voxprint` sink, then pick `voxprint mic`\n"
        "  as the microphone in the other application."
    ),
    "darwin": (
        "Install BlackHole (free) or Loopback:\n"
        "    brew install blackhole-2ch\n"
        "  Send the converter's output to BlackHole, then pick BlackHole as the\n"
        "  microphone in the other application."
    ),
    "win32": (
        "Install VB-Audio Virtual Cable (free) or VoiceMeeter:\n"
        "    https://vb-audio.com/Cable/\n"
        "  Send the converter's output to `CABLE Input`, then pick `CABLE Output`\n"
        "  as the microphone in the other application."
    ),
}


def audio_devices() -> tuple[bool, str, list[dict]]:
    """``(ready, reason, devices)`` for the host's audio I/O.

    Realtime conversion needs a real input and output device, which a container
    or a headless server generally does not have -- and the failure is an
    unhelpful ``OSError`` from deep inside PortAudio, so it is caught here.
    """
    try:
        import sounddevice  # noqa: PLC0415
    except ModuleNotFoundError:
        return False, "sounddevice not installed (comes with so-vits-svc-fork)", []
    except OSError as exc:
        return False, f"PortAudio is missing, so no audio device can be opened: {exc}", []

    try:
        found = sounddevice.query_devices()
    except Exception as exc:
        return False, f"could not enumerate audio devices: {exc}", []

    devices = [
        {
            "index": i,
            "name": d.get("name", "?"),
            "inputs": d.get("max_input_channels", 0),
            "outputs": d.get("max_output_channels", 0),
        }
        for i, d in enumerate(found)
    ]
    if not devices:
        return False, "no audio devices are present on this host", []
    return True, f"{len(devices)} audio device(s)", devices


def realtime_command(
    workspace: SvcWorkspace,
    *,
    input_device: int | None = None,
    output_device: int | None = None,
    transpose: int = 0,
    block_seconds: float = 0.35,
    chunk_seconds: float = 0.5,
    crossfade_seconds: float = 0.05,
    device: str = "cpu",
    passthrough: bool = False,
) -> list[str]:
    """Build the ``svc vc`` command line for live conversion.

    Returned rather than run, so a caller can print it, log it, or hand it to a
    supervisor. Latency is roughly ``block_seconds`` plus the model's own
    inference time per block; the upstream default block of 0.5 s is lowered here
    because the first thing anyone wants is less delay, and raising it back is
    the obvious fix when the audio starts breaking up.

    ``auto_predict_f0`` is left off: upstream's own help says a re-predicted
    contour is unstable in realtime.
    """
    args = [
        sys.executable, "-m", "voxprint._svc_compat", "vc",
        "-m", str(workspace.model_dir.resolve()),
        "-c", str(workspace.config_path.resolve()),
        "-s", workspace.speaker,
        "-t", str(int(transpose)),
        "-na",
        "-b", str(block_seconds),
        "-ch", str(chunk_seconds),
        "-cr", str(crossfade_seconds),
        "-d", device,
    ]
    if input_device is not None:
        args += ["-i", str(input_device)]
    if output_device is not None:
        args += ["-o", str(output_device)]
    if passthrough:
        args.append("-po")
    return args


def realtime_readiness(workspace: SvcWorkspace | None) -> dict:
    """Everything that has to be true before live conversion can start."""
    import platform

    svc_ready, svc_reason = available()
    audio_ready, audio_reason, devices = audio_devices()
    model_ready = workspace is not None and workspace.is_trained()

    system = sys.platform
    return {
        "toolchain": {"ready": svc_ready, "detail": svc_reason},
        "audio": {"ready": audio_ready, "detail": audio_reason, "devices": devices},
        "model": {
            "ready": model_ready,
            "detail": "trained model available" if model_ready
            else "no trained model; so-vits-svc is not zero-shot",
        },
        "ready": bool(svc_ready and audio_ready and model_ready),
        "platform": f"{platform.system()} ({system})",
        "virtual_mic": VIRTUAL_MIC_SETUP.get(system, VIRTUAL_MIC_SETUP["linux"]),
    }
