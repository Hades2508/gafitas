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
    payloads: dict[str, str] | None = None,
) -> Path:
    """Seal one run. ``payloads`` maps sha256 -> the full text of any tool
    argument too long to sit in an event (A2).

    Kept in a sibling directory rather than inline: the events file stays
    readable, a payload repeated across turns is stored once, and the thing that
    actually got written to disk is recoverable months later. Truncating and
    keeping nothing else is how two analyses in this project ran aground -- the
    only way to tell a correct copy from an over-copy is to read what was
    written.
    """
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
    if payloads:
        store = directory / "payloads"
        store.mkdir(parents=True, exist_ok=True)
        for digest, text in payloads.items():
            target = store / f"{digest}.txt"
            if not target.exists():          # identical content, written once
                target.write_text(text, encoding="utf-8")
    return path


def read_payload(directory: Path, digest: str) -> str | None:
    """The full text behind an event's ``sha256``, or None if it is not here."""
    target = Path(directory) / "payloads" / f"{digest}.txt"
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        return None


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
