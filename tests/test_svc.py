"""Tests for the so-vits-svc integration.

Training needs a GPU to be worth doing and minutes even to fail, so what is
tested here is everything around it: the workspace layout, the consent gate, the
config editing, the readiness reporting, and the compatibility shim. The
subprocess calls themselves are covered by asserting on the command that would
be run, which is where the mistakes actually live -- a wrong interpreter, a
missing flag, a path that only resolves from one directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from voxprint import svc
from voxprint.audio import save_audio
from voxprint.gallery import ConsentRecord
from voxprint.pipeline import VoiceLab
from voxprint.synthetic import SR, distinct_speakers

CONSENT = ConsentRecord(granted_by="fixture", statement="synthetic voice")


@pytest.fixture
def workspace(tmp_path):
    return svc.SvcWorkspace(root=tmp_path / "ws", speaker="omar")


@pytest.fixture
def training_audio(tmp_path):
    """A handful of clips, as a training directory."""
    directory = tmp_path / "recordings"
    directory.mkdir()
    speaker = distinct_speakers(1, seed=5)[0]
    for i in range(4):
        save_audio(directory / f"take{i}.wav", speaker.say("aiueo", 3.0, seed=100 + i), SR)
    return directory


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #

def test_workspace_paths_follow_the_upstream_layout(workspace):
    assert workspace.dataset_raw.parts[-2:] == ("dataset_raw", "omar")
    assert workspace.config_path.parts[-3:] == ("configs", "44k", "config.json")
    assert workspace.model_dir.parts[-2:] == ("logs", "44k")


def test_untrained_workspace_reports_itself_untrained(workspace):
    assert workspace.checkpoints() == []
    assert workspace.latest_checkpoint() is None
    assert not workspace.is_trained()


def test_checkpoints_sort_by_step_not_by_name(workspace):
    workspace.model_dir.mkdir(parents=True)
    for step in (200, 1000, 90):
        (workspace.model_dir / f"G_{step}.pth").write_bytes(b"x")
    (workspace.model_dir / "G_0.pth").write_bytes(b"x")     # the untrained base model

    names = [p.name for p in workspace.checkpoints()]
    assert names == ["G_90.pth", "G_200.pth", "G_1000.pth"], "string sorting would put G_1000 first"
    assert workspace.latest_checkpoint().name == "G_1000.pth"


def test_is_trained_needs_both_a_config_and_a_checkpoint(workspace):
    workspace.model_dir.mkdir(parents=True)
    (workspace.model_dir / "G_500.pth").write_bytes(b"x")
    assert not workspace.is_trained(), "a checkpoint without a config is not usable"

    workspace.config_path.parent.mkdir(parents=True)
    workspace.config_path.write_text("{}")
    assert workspace.is_trained()


# --------------------------------------------------------------------------- #
# Dataset staging
# --------------------------------------------------------------------------- #

def test_collect_audio_finds_clips_recursively(training_audio):
    (training_audio / "nested").mkdir()
    save_audio(training_audio / "nested" / "more.wav", distinct_speakers(1, seed=6)[0].say("a", 2.0), SR)
    assert len(svc.collect_audio(training_audio)) == 5


def test_collect_audio_ignores_non_audio(training_audio):
    (training_audio / "notes.txt").write_text("not audio")
    assert all(p.suffix != ".txt" for p in svc.collect_audio(training_audio))


def test_stage_audio_copies_and_totals_the_duration(workspace, training_audio):
    minutes = svc.stage_audio(workspace, svc.collect_audio(training_audio))
    assert minutes == pytest.approx(4 * 3.0 / 60.0, rel=0.1)
    assert len(list(workspace.dataset_raw.glob("*.wav"))) == 4


def test_stage_audio_reports_an_unreadable_file(workspace, tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(svc.SvcError, match="cannot read"):
        svc.stage_audio(workspace, [bad])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_training_budget_edits_only_what_it_is_given(workspace):
    workspace.config_path.parent.mkdir(parents=True)
    workspace.config_path.write_text(json.dumps({"train": {"epochs": 10000, "batch_size": 16, "seed": 7}}))

    budget = svc.set_training_budget(workspace, epochs=3, batch_size=2)
    assert budget["epochs"] == 3
    assert budget["batch_size"] == 2
    assert budget["seed"] == 7, "unrelated settings must survive"

    written = json.loads(workspace.config_path.read_text())["train"]
    assert written["epochs"] == 3


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #

def test_inference_refuses_without_a_trained_model(workspace, tmp_path):
    with pytest.raises(svc.SvcError, match="not zero-shot"):
        svc.infer(workspace, tmp_path / "in.wav", tmp_path / "out.wav")


def test_training_refuses_without_a_consent_record(tmp_path, clips, training_audio):
    lab = VoiceLab(tmp_path / "voices", require_consent=False)
    lab.enroll("omar", clips["omar"][:1])            # enrolled, but no consent recorded
    lab.gallery.require_consent = True
    with pytest.raises(svc.SvcError, match="consent"):
        svc.train_for_speaker(lab, "omar", training_audio)


def test_training_refuses_an_empty_audio_directory(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:1], consent=CONSENT)
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(svc.SvcError, match="no audio found"):
        svc.train_for_speaker(lab, "omar", empty)


def test_a_trained_model_is_registered_on_the_voice_print(tmp_path, clips):
    lab = VoiceLab(tmp_path / "voices")
    lab.enroll("omar", clips["omar"][:1], consent=CONSENT)
    workspace = svc.SvcWorkspace(root=tmp_path / "ws", speaker="omar")

    assert svc.workspace_for(lab, "omar") is None
    svc.register_model(lab, "omar", workspace)
    assert lab.gallery.get("omar").models["sovits"] == str(workspace.root)

    reopened = VoiceLab(tmp_path / "voices")
    assert svc.workspace_for(reopened, "omar").root == workspace.root


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #

def test_realtime_command_preserves_the_melody_and_resolves_paths(workspace):
    command = svc.realtime_command(workspace, input_device=1, output_device=4, transpose=-2)

    assert command[0] == sys.executable, "must not rely on whichever python is on PATH"
    assert command[1:3] == ["-m", "voxprint._svc_compat"], "the NumPy shim must wrap the CLI"

    # Everything after the subcommand belongs to svc; the python invocation has a
    # `-m` of its own, so the two halves have to be read separately.
    sub = command[command.index("vc"):]
    assert "-na" in sub, "auto-predict-f0 must stay off: it re-invents the pitch contour"
    assert "-a" not in sub
    assert sub[sub.index("-t") + 1] == "-2"
    assert sub[sub.index("-i") + 1] == "1"
    assert sub[sub.index("-o") + 1] == "4"
    assert all(Path(sub[sub.index(flag) + 1]).is_absolute() for flag in ("-m", "-c")), (
        "paths must be absolute: svc runs with its workspace as the working directory"
    )


def test_realtime_command_omits_devices_when_unspecified(workspace):
    sub = svc.realtime_command(workspace)[svc.realtime_command(workspace).index("vc"):]
    assert "-i" not in sub and "-o" not in sub


def test_realtime_readiness_reports_each_prerequisite_separately():
    report = svc.realtime_readiness(None)
    assert set(report) >= {"toolchain", "audio", "model", "ready", "platform", "virtual_mic"}
    assert report["model"]["ready"] is False
    assert report["ready"] is False
    assert isinstance(report["virtual_mic"], str) and report["virtual_mic"]


def test_audio_devices_fails_clearly_when_portaudio_is_absent():
    """Headless hosts raise an opaque OSError from inside PortAudio; catch it."""
    ready, reason, devices = svc.audio_devices()
    assert isinstance(ready, bool)
    assert reason
    if not ready:
        assert devices == []


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_every_platform_has_virtual_mic_instructions(platform):
    assert svc.VIRTUAL_MIC_SETUP[platform].strip()


# --------------------------------------------------------------------------- #
# NumPy compatibility shim
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not svc.available()[0], reason="so-vits-svc-fork not installed")
def test_compat_shim_replaces_the_removed_numpy_call():
    """Upstream uses np.fromstring, which NumPy removed; training dies without this."""
    import numpy as np

    from voxprint._svc_compat import apply

    assert apply() is True

    from so_vits_svc_fork import utils

    image = utils.plot_spectrogram_to_numpy(np.random.default_rng(0).random((40, 80)))
    assert image.ndim == 3
    assert image.shape[2] == 4
    assert image.dtype == np.uint8
