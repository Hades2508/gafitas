"""Pinned imports of the borrowed authorities, with provenance.

Three things are imported and never reimplemented:

    programmer.guard         containment. Security code that ~60 measured
                             missions have exercised with zero violations.
                             Rewriting it would be the single worst decision
                             available here.
    programmer.edit_channel  list_symbols(), so this harness and Gafotas'
                             edit path agree on what a symbol is.
    telemetry.TelemetryStore MISSION_TELEMETRY_V1.

Why this module is not ``generalista/_deps.py`` again
-----------------------------------------------------
That module did ``sys.path.insert`` at import time and nothing else, and it
produced a real invalid-results incident: a runner colocated with a target
worktree bound ``import generalista`` to the target's stale copy. The defence
bolted on afterwards (``generalista_bootstrap.py``, 89 lines) had to run before
any other import.

Here the same job is done differently and in one place:

  * roots are resolved from an environment variable or a documented default,
    never from cwd and never from argv;
  * the imported module's ``__file__`` is checked to be under the root we
    pinned, so a module that arrived from somewhere else is detected rather
    than trusted;
  * a missing or shadowed dependency raises ``HarnessInvalid`` at import time.
    It never degrades to "run without containment" -- an unguarded run is not
    a weaker measurement, it is not a measurement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .errors import HarnessInvalid

PROGRAMMER_ROOT = Path(
    os.environ.get("LOCALPROG_PROGRAMMER_ROOT", r"D:\PROGRAMMER-BUILD-ROOT")
)
TELEMETRY_ROOT = Path(
    os.environ.get("LOCALPROG_TELEMETRY_ROOT", r"D:\PROGRAMMING-TELEMETRY-ROOT")
)


def _pin(root: Path) -> None:
    text = str(root)
    if not root.is_dir():
        raise HarnessInvalid(
            f"dependency root does not exist: {text}. Set LOCALPROG_PROGRAMMER_ROOT "
            f"/ LOCALPROG_TELEMETRY_ROOT, or fix the checkout."
        )
    # Drop cwd markers -- the classic shadow vector -- then pin at the front.
    sys.path[:] = [p for p in sys.path if p not in ("", ".", text)]
    sys.path.insert(0, text)


_pin(PROGRAMMER_ROOT)
_pin(TELEMETRY_ROOT)

try:
    from programmer import edit_channel as edit_channel  # noqa: E402
    from programmer import guard as guard  # noqa: E402
    from telemetry import TelemetryStore as TelemetryStore  # noqa: E402
    from telemetry.api import DuplicateMissionError as DuplicateMissionError  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment defect
    raise HarnessInvalid(f"cannot import a pinned dependency: {exc}") from exc


def _verify(module, root: Path, name: str) -> None:
    """Prove the module we got is the one we pinned, not a shadow."""
    path = getattr(module, "__file__", None)
    if path is None:
        raise HarnessInvalid(f"{name} has no __file__; cannot verify provenance")
    resolved = Path(path).resolve()
    if root.resolve() not in resolved.parents:
        raise HarnessInvalid(
            f"DEPENDENCY_SHADOWED: {name} was imported from {resolved}, which is "
            f"not under the pinned root {root}. Something else on sys.path claimed "
            f"the name first."
        )


_verify(guard, PROGRAMMER_ROOT, "programmer.guard")
_verify(edit_channel, PROGRAMMER_ROOT, "programmer.edit_channel")
_verify(sys.modules["telemetry"], TELEMETRY_ROOT, "telemetry")


def provenance() -> dict:
    """Compact record of which code actually ran. Sealed into every run."""
    return {
        "programmer_root": str(PROGRAMMER_ROOT),
        "telemetry_root": str(TELEMETRY_ROOT),
        "guard_file": str(Path(guard.__file__).resolve()),
        "edit_channel_file": str(Path(edit_channel.__file__).resolve()),
        "telemetry_file": str(Path(sys.modules["telemetry"].__file__).resolve()),
        "localprog_file": str(Path(__file__).resolve().parent),
        "cwd": str(Path.cwd()),
    }


__all__ = [
    "PROGRAMMER_ROOT",
    "TELEMETRY_ROOT",
    "guard",
    "edit_channel",
    "TelemetryStore",
    "DuplicateMissionError",
    "provenance",
]
