"""EpochColor: colorize black and white film and photos."""

__version__ = "0.5.0"


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
