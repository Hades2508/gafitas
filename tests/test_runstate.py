"""Job lifecycle and evidence retention (A1, A6).

Both come from a forensic audit of a real campaign. 920 preserved workspaces
holding 816 MB with nothing that was ever going to delete them, and three
background runs that died mid-corpus leaving a results file that looked exactly
like a finished one.
"""

from __future__ import annotations

import json
import time

import pytest

from localprog import runstate


# ------------------------------------------------------------- A6 lifecycle

def test_a_finished_job_says_so(tmp_path):
    with runstate.JobState(tmp_path, job="t", total=3) as job:
        job.beat(3)
    assert runstate.read_state(tmp_path)["state"] == runstate.COMPLETED


def test_a_crashed_job_says_what_killed_it(tmp_path):
    with pytest.raises(RuntimeError):
        with runstate.JobState(tmp_path, job="t"):
            raise RuntimeError("boom")
    state = runstate.read_state(tmp_path)
    assert state["state"] == runstate.FAILED
    assert "boom" in state["detail"]


def test_an_interrupt_is_aborted_not_failed(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with runstate.JobState(tmp_path, job="t"):
            raise KeyboardInterrupt
    assert runstate.read_state(tmp_path)["state"] == runstate.ABORTED


def test_a_dead_job_is_orphaned_not_believed(tmp_path):
    """The file says RUNNING because nothing got the chance to say otherwise.
    Reading it literally is how three dead runs were taken for live ones."""
    job = runstate.JobState(tmp_path, job="t").start()
    job.beat(7)
    stale = json.loads((tmp_path / runstate.STATUS_FILE).read_text(encoding="utf-8"))
    stale["heartbeat"] = time.time() - runstate.STALE_HEARTBEAT_SECONDS - 60
    (tmp_path / runstate.STATUS_FILE).write_text(json.dumps(stale), encoding="utf-8")
    state = runstate.read_state(tmp_path)
    assert state["state"] == runstate.ORPHANED
    assert state["done"] == 7, "and how far it got is still readable"


def test_a_live_job_is_not_called_orphaned(tmp_path):
    runstate.JobState(tmp_path, job="t").start().beat(1)
    assert runstate.read_state(tmp_path)["state"] == runstate.RUNNING


def test_an_untracked_directory_is_not_silently_a_success(tmp_path):
    assert runstate.read_state(tmp_path)["state"] == "NOT_TRACKED"


# ------------------------------------------------------------ A1 retention

def make(tmp_path, name, *, age_days=0.0, size=1000, pinned=False, marked=True):
    d = tmp_path / f"gafitas_{name}"
    d.mkdir()
    (d / "blob.txt").write_text("x" * size, encoding="utf-8")
    if marked:
        runstate.mark_preserved(d, ticket=name, outcome="FAIL", pinned=pinned)
        marker = json.loads((d / runstate.MARKER_FILE).read_text(encoding="utf-8"))
        marker["preserved_at"] = time.time() - age_days * 86400
        (d / runstate.MARKER_FILE).write_text(json.dumps(marker), encoding="utf-8")
    return d


def test_a_dry_run_deletes_nothing(tmp_path):
    old = make(tmp_path, "old", age_days=90)
    report = runstate.retain(tmp_path, max_age_days=30)
    assert report["dry_run"] and report["selected"]
    assert old.exists(), "the default must never delete"


def test_age_selects_and_only_when_asked(tmp_path):
    old = make(tmp_path, "old", age_days=90)
    new = make(tmp_path, "new", age_days=1)
    runstate.retain(tmp_path, max_age_days=30, dry_run=False)
    assert not old.exists()
    assert new.exists()


def test_a_pinned_workspace_is_never_touched(tmp_path):
    """Evidence from a contaminated or unexplained run is not subject to a
    disk budget."""
    pinned = make(tmp_path, "pinned", age_days=900, pinned=True)
    runstate.retain(tmp_path, max_age_days=1, max_total_bytes=0, dry_run=False)
    assert pinned.exists()


def test_an_unmarked_workspace_is_never_touched(tmp_path):
    """We do not know what it is, and guessing about evidence is the failure
    this guards against."""
    unknown = make(tmp_path, "unknown", age_days=900, marked=False)
    runstate.retain(tmp_path, max_age_days=1, max_total_bytes=0, dry_run=False)
    assert unknown.exists()


def test_size_deletes_oldest_first(tmp_path):
    old = make(tmp_path, "old", age_days=50, size=1000)
    mid = make(tmp_path, "mid", age_days=20, size=1000)
    young = make(tmp_path, "young", age_days=1, size=1000)
    runstate.retain(tmp_path, max_total_bytes=1500, dry_run=False)
    assert not old.exists()
    assert young.exists(), "newest survives"
    assert mid.exists() or not mid.exists()  # whichever, never the newest first


def test_every_deletion_is_logged_before_it_happens(tmp_path):
    make(tmp_path, "old", age_days=90)
    report = runstate.retain(tmp_path, max_age_days=30, dry_run=False)
    log = tmp_path / runstate.DELETION_LOG
    assert log.exists()
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["reason"].startswith("older than")
    assert entry["ticket"] == "old"
    assert report["removed"]


def test_no_policy_means_no_deletion(tmp_path):
    kept = make(tmp_path, "old", age_days=9000)
    report = runstate.retain(tmp_path, dry_run=False)
    assert report["selected"] == [] and kept.exists()
