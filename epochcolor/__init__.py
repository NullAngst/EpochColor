"""EpochColor: colorize black and white film and photos."""

# its own file so setuptools can read it without tripping over epochcolor.py,
# the source launcher next to this folder
from ._version import __version__  # noqa: F401


def _activate_torch() -> None:
    # A PyTorch installed on first run lives in its own folder; this adds it
    # to the import path, after everything else, in every process (the GUI,
    # the CLI and the worker).
    try:
        from .torch_setup import activate

        activate()
    except Exception:
        pass


_activate_torch()
