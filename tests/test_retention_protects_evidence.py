"""The retention policy promised two things and delivered one and a half.

F-176. `runstate`'s own header says:

    "NOTHING HERE DELETES EVIDENCE BY DEFAULT. The policy has to be asked for,
     with its bounds, and it refuses to touch anything marked as belonging to a
     contaminated or unfinished run."

THE PROMISE WITH NOTHING BEHIND IT
----------------------------------
Only `pinned` was ever checked, and **nothing in this codebase passes
pinned=True**. `mark_preserved` is called from exactly one place,
`workspace.dispose`, which forwards a parameter no caller sets. So the
protection could not be requested and the sentence describing it was unbacked.

The outcome was already surveyed and already written to the log. It simply was
not consulted. Four outcomes are the harness admitting its own defect, and a
record of this factory producing invalid results is the last thing an age limit
should be free to reclaim.

THE DOUBLE BOOKING
------------------
The size pass filtered with `e not in doomed`, comparing a candidate dict
against doomed dicts carrying an extra "reason" key. Never equal, so
`remaining` held everything the age pass had already condemned.

Reproduced: four workspaces of 1088 bytes, age limit 5 days, budget 2500. Two
directories appeared TWICE in the selection, once per reason, and
selected_bytes reported 4352 -- the entire corpus -- for a deletion that would
free 2176. Three consequences, all in a module whose subject is evidence:

  * the append-only deletion log recorded each path twice, with two reasons;
  * the figure a human reads before running this was double the truth;
  * the second rmtree on an already-removed path landed in `failed`, so a
    clean run reported failures it had not had.
"""

from __future__ import annotations

import json
import time

import pytest

from localprog import runstate

MARKER = ".gafitas_preserved.json"


def workspace(base, name: str, *, age_days: float, outcome: str = "FAIL",
              pinned: bool = False, size: int = 1000):
    target = base / name
    target.mkdir()
    (target / "d.bin").write_bytes(b"x" * size)
    (target / MARKER).write_text(json.dumps({
        "ticket": name, "outcome": outcome, "pinned": pinned,
        "preserved_at": time.time() - age_days * 86400,
    }), encoding="utf-8")
    return target


@pytest.mark.parametrize("outcome", sorted(runstate.NEVER_DELETE))
def test_a_harness_defect_is_never_aged_out(tmp_path, outcome):
    """A record of this factory producing invalid results outlives any limit."""
    workspace(tmp_path, "gafitas_old", age_days=999, outcome=outcome)
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=1,
                             max_total_bytes=1, dry_run=True)
    assert report["selected"] == [], f"{outcome} was scheduled for deletion"
    assert report["protected"] == 1


def test_an_ordinary_failure_is_still_subject_to_the_policy(tmp_path):
    """Preserving everything for ever is the leak this module exists to close."""
    workspace(tmp_path, "gafitas_old", age_days=999, outcome="FAIL")
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=1,
                             dry_run=True)
    assert len(report["selected"]) == 1


def test_an_unmarked_directory_is_never_touched(tmp_path):
    (tmp_path / "gafitas_mystery").mkdir()
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=0,
                             max_total_bytes=0, dry_run=True)
    assert report["selected"] == []


def test_no_path_is_scheduled_twice(tmp_path):
    for n, age in enumerate((20, 20, 1, 1)):
        workspace(tmp_path, f"gafitas_w{n}", age_days=age)
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=5,
                             max_total_bytes=2500, dry_run=True)
    paths = [entry["path"] for entry in report["selected"]]
    assert len(paths) == len(set(paths)), (
        "a path scheduled twice writes two lines to the append-only log and "
        "fails its own second deletion")


def test_the_reported_size_is_what_would_actually_be_freed(tmp_path):
    for n, age in enumerate((20, 20, 1, 1)):
        workspace(tmp_path, f"gafitas_w{n}", age_days=age)
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=5,
                             max_total_bytes=2500, dry_run=True)
    freed = sum(entry["bytes"] for entry in report["selected"])
    assert report["selected_bytes"] == freed
    assert report["selected_bytes"] < report["total_bytes"], (
        "it reported the whole corpus as recoverable when only the two aged "
        "directories were going anywhere")


def test_a_pin_is_still_honoured(tmp_path):
    workspace(tmp_path, "gafitas_pinned", age_days=999, pinned=True)
    report = runstate.retain(tmp_path, prefix="gafitas_", max_age_days=1,
                             dry_run=True)
    assert report["selected"] == []


def test_dry_run_removes_nothing(tmp_path):
    target = workspace(tmp_path, "gafitas_old", age_days=999)
    runstate.retain(tmp_path, prefix="gafitas_", max_age_days=1, dry_run=True)
    assert target.is_dir()
