"""Disabled export shim used when export functionality is removed.

Provides `export_formats`, `Exporter`, and `gd_outputs` used across the codebase
so we can safely remove the heavy export implementation while keeping import
compatibility. Calling `Exporter` will raise a clear error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def export_formats() -> dict[str, list[Any]]:
    """Return a minimal mapping of export formats for compatibility.

    Many files only query keys like 'Format', 'Argument', 'Suffix', 'Arguments'.
    Provide these keys with empty lists to avoid KeyError and allow checks.
    """
    keys = ["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"]
    return dict(zip(keys, [[] for _ in keys]))


class Exporter:
    def __init__(self, overrides: dict | None = None, _callbacks: dict | None = None):
        self.args = SimpleNamespace()
        if overrides:
            for k, v in overrides.items():
                setattr(self.args, k, v)

    def __call__(self, model=None):
        raise RuntimeError(
            "Model export has been removed from this local build. "
            "Install the full 'ultralytics' package to enable export functionality."
        )


def gd_outputs(*_args, **_kwargs) -> list:
    """Fallback for TensorFlow export helper `gd_outputs`.

    Return an empty list — exporting helpers are disabled in this build.
    """
    return []
