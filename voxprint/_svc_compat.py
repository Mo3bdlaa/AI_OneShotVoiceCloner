"""Compatibility shims for running so-vits-svc-fork on a current NumPy.

so-vits-svc-fork 4.2.x renders its TensorBoard spectrogram previews with
``np.fromstring(fig.canvas.tostring_argb(), ...)``. NumPy removed the binary mode
of ``fromstring``, so training crashes the first time it reaches a logging step:

    ValueError: The binary mode of fromstring is removed, use frombuffer instead

It is a logging path, not a modelling one -- the run is otherwise fine -- so
patching it is preferable to pinning NumPy back for the whole project. Importing
this module installs the patch; :mod:`voxprint.svc` runs the upstream CLI through
it so the fix applies without anyone having to remember.

Delete this module once upstream ships a release that uses ``frombuffer``.
"""

from __future__ import annotations

import numpy as np


def _canvas_to_array(fig) -> np.ndarray:
    """Read a rendered Matplotlib canvas into ``(h, w, 4)`` uint8.

    Tries the modern buffer first and falls back to the ARGB one, since which
    methods a canvas exposes depends on the Matplotlib version.
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    for method in ("buffer_rgba", "tostring_argb"):
        reader = getattr(fig.canvas, method, None)
        if reader is None:
            continue
        data = np.frombuffer(reader(), dtype=np.uint8)
        return data.reshape(height, width, 4)
    raise RuntimeError("this Matplotlib canvas exposes no readable RGBA buffer")


def apply() -> bool:
    """Patch so-vits-svc-fork's plotting helpers. Returns whether it did anything."""
    try:
        from so_vits_svc_fork import utils
    except ImportError:
        return False

    import matplotlib
    import matplotlib.pyplot as plt

    def plot_spectrogram_to_numpy(spectrogram):
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(10, 2))
        image = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(image, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()
        data = _canvas_to_array(fig)
        plt.close(fig)
        return data

    def plot_data_to_numpy(x, y):
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(x)
        ax.plot(y)
        plt.tight_layout()
        data = _canvas_to_array(fig)
        plt.close(fig)
        return data

    utils.plot_spectrogram_to_numpy = plot_spectrogram_to_numpy
    utils.plot_data_to_numpy = plot_data_to_numpy

    # The training module imports the helpers by name at import time.
    try:
        from so_vits_svc_fork import train as train_module

        if hasattr(train_module, "plot_spectrogram_to_numpy"):
            train_module.plot_spectrogram_to_numpy = plot_spectrogram_to_numpy
        if hasattr(train_module, "plot_data_to_numpy"):
            train_module.plot_data_to_numpy = plot_data_to_numpy
    except ImportError:  # pragma: no cover - layout differs between versions
        pass
    return True


def main() -> int:
    """Entry point: patch, then hand over to the upstream ``svc`` CLI."""
    apply()
    from so_vits_svc_fork.__main__ import cli

    cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
