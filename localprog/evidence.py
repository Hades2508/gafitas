"""Sealing what happened, including -- especially -- when it went wrong.

The unit of evidence here is the TURN TRANSCRIPT, not a plan. Every tool call,
its arguments, its result classification and the diff are written before
anything is cleaned up. That is strictly more informative than the old
per-attempt plan record, and it is the thing that made the only large capability
jump in the project's history possible: replaying a sealed candidate months
later located the exact point where feedback was being dropped.

A run that ends in HARNESS_INVALID gets a notice file of its own, so a
contaminated cohort announces itself instead of being discovered later by
someone puzzled at the numbers.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "LOCAL_PROGRAMMER_EVIDENCE_V0"


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def diff(relative: str, before: str | None, after: str | None) -> str:
    return "".join(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        )
    )


def capture_before(root: Path, paths) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for relative in paths:
        target = root / relative
        try:
            out[relative] = target.read_text(encoding="utf-8") if target.is_file() else None
        except (OSError, UnicodeDecodeError):
            out[relative] = None
    return out


def capture_after(root: Path, before: dict[str, str | None]) -> tuple[dict[str, str | None], str]:
    after: dict[str, str | None] = {}
    text = ""
    for relative in sorted(before):
        target = root / relative
        try:
            after[relative] = target.read_text(encoding="utf-8") if target.is_file() else None
        except (OSError, UnicodeDecodeError):
            after[relative] = None
        text += diff(relative, before[relative], after[relative])
    return after, text


def seal(
    directory: Path,
    *,
    name: str,
    record: dict[str, Any],
    transcript: dict[str, Any] | None = None,
    events: list | None = None,
    diff_text: str = "",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "record": record,
        "transcript": transcript,
        "events": events or [],
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if diff_text:
        (directory / f"{name}.diff").write_text(diff_text, encoding="utf-8")
    return path


def harness_invalid_notice(directory: Path, *, runs: list[dict]) -> Path:
    """Announce contamination in the cohort's own directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "HARNESS_INVALID.md"
    lines = [
        "# HARNESS_INVALID",
        "",
        "One or more runs in this directory ended in a defect of the harness itself,",
        "not of the model and not of the repository under test. **No metric in this",
        "directory may be used.** Fix the harness, then re-run under a new run id.",
        "",
    ]
    for run in runs:
        detail = (run.get("harness_invalid") or {}).get("detail", "(sin detalle)")
        lines.append(f"- `{run.get('label', '?')}`: {detail}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
