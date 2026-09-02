"""Acceptance for DOGFOOD-04: SKIP_DIRS must be defined once.

A real defect, introduced by Claude while building verify.py, and exactly the
kind this repository is organised against: I3 says there is one source of truth
for a thing, and this is a thing defined twice.

    localprog/tools.py:81   SKIP_DIRS = {".git", "__pycache__", ...}
    localprog/verify.py:68  SKIP_DIRS = {".git", "__pycache__", ...}

They are identical today. If they ever drift, the tools skip one set of
directories and the conscience skips another -- so `list_dir` and `grep` would
be ignoring a directory that `_tracked_files` still watches, and changed-file
reporting would quietly disagree with itself. That is a silent-wrong-answer bug
waiting on one careless edit, which is the worst kind to leave lying around.

Written before the fix and proven to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import tools, verify  # noqa: E402


def test_skip_dirs_is_one_object_not_two_equal_ones():
    """Equality is not enough: two equal sets drift the moment one is edited.
    The check is identity, because that is the property that cannot rot."""
    assert verify.SKIP_DIRS is tools.SKIP_DIRS


def test_there_is_exactly_one_definition_in_the_package():
    """Belt and braces: the identity above could be satisfied by an alias while
    a second literal still sat in the file for someone to edit by mistake."""
    root = Path(__file__).resolve().parents[1] / "localprog"
    defining = [
        path.name
        for path in sorted(root.glob("*.py"))
        if any(
            line.lstrip().startswith("SKIP_DIRS") and "=" in line and "{" in line
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert defining == ["tools.py"], f"SKIP_DIRS is defined in {defining}"


def test_the_contents_are_unchanged():
    """Deduplicating must not quietly redefine what gets skipped."""
    assert tools.SKIP_DIRS == {
        ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
    }


def test_both_modules_still_use_it():
    """The point is one definition, not one user. Both sides must still skip."""
    assert ".git" in verify.SKIP_DIRS
    assert "__pycache__" in tools.SKIP_DIRS
