"""Job state and evidence retention. The two things a long campaign needs and
that nothing here provided.

WHY
---
A forensic audit of this machine found 920 preserved workspaces holding 816 MB
and 61 registered git worktrees, none of which anything was ever going to
delete. Failed runs preserve their workspace on purpose -- that rule is right
and stays -- but "preserve" without "until when" is not a policy, it is a leak
with a good excuse.

The same audit found three background runs that had died mid-corpus and left no
way to tell a finished directory from an abandoned one. A results file that
exists is not a job that completed, and every consumer of these directories was
reading it as though it were.

TWO PIECES, ONE FILE
--------------------
``JobState`` writes an explicit lifecycle next to a run's output, with a
heartbeat, so ORPHANED and INTERRUPTED are observable facts rather than guesses.

``retain`` applies an explicit policy to preserved workspaces: by age, by total
size, never touching what is pinned, oldest first, and every deletion recorded
in an append-only log. It also prunes git worktree registrations whose
directories are gone, which is the part that silently slows every git command
in the repository.

NOTHING HERE DELETES EVIDENCE BY DEFAULT. The policy has to be asked for, with
its bounds, and it refuses to touch anything marked as belonging to a
contaminated or unfinished run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "GAFITAS_RUNSTATE_V1"

#: The lifecycle. Closed set on purpose: a state nobody can spell is a state
#: nobody checks for.
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
ABORTED = "ABORTED"          # stopped deliberately by an operator or by us
INTERRUPTED = "INTERRUPTED"  # the process died; we saw it happen
ORPHANED = "ORPHANED"        # the process is gone and nobody recorded why
STATES = (RUNNING, COMPLETED, FAILED, ABORTED, INTERRUPTED, ORPHANED)

#: A RUNNING job whose heartbeat is older than this is not running any more.
#: Generous, because a single agent turn against a large context can legitimately
#: take minutes and calling a live job orphaned is worse than noticing late.
STALE_HEARTBEAT_SECONDS = 900.0

STATUS_FILE = "RUN_STATUS.json"
#: Written into a preserved workspace so retention knows what it is looking at.
MARKER_FILE = ".gafitas_preserved.json"
DELETION_LOG = "RETENTION_LOG.jsonl"

#: Outcomes whose workspace is never aged out. These are the harness reporting
#: its OWN defect, and a record of the factory producing invalid results is
#: exactly what a size limit must not be free to reclaim. Compared uppercased,
#: so a marker written by an older version still matches.
NEVER_DELETE = frozenset({"HARNESS_INVALID", "ACCEPTANCE_UNUSABLE",
                          "PROVIDER_ERROR", "EXPERIMENT_CONTAMINATED"})


@dataclass
class JobState:
    """The lifecycle of one long-running job, on disk, next to its output.

    Deliberately a file and not a lock: the question this answers is "what
    happened to the run that wrote this directory", asked later, possibly by a
    different process, possibly after a reboot.
    """

    directory: Path
    job: str
    total: int | None = None
    started: float = field(default_factory=time.time)
    _path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / STATUS_FILE

    def _write(self, state: str, **extra: Any) -> None:
        payload = {
            "schema": SCHEMA,
            "job": self.job,
            "state": state,
            "pid": os.getpid(),
            "started": self.started,
            "heartbeat": time.time(),
            "total": self.total,
        }
        payload.update(extra)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self._path)          # atomic: a torn status file is a lie

    def start(self) -> "JobState":
        self._write(RUNNING, done=0)
        return self

    def beat(self, done: int) -> None:
        """Say the job is alive and how far it has got. Cheap; call it often."""
        self._write(RUNNING, done=done)

    def finish(self, state: str = COMPLETED, done: int | None = None,
               detail: str = "") -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self._write(state, done=done, detail=detail)

    def __enter__(self) -> "JobState":
        return self.start()

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc_type is None:
            self.finish(COMPLETED)
        elif isinstance(exc, KeyboardInterrupt):
            self.finish(ABORTED, detail="KeyboardInterrupt")
        else:
            self.finish(FAILED, detail=f"{exc_type.__name__}: {exc}")
        return False


def read_state(directory: Path) -> dict:
    """What happened to the job that wrote *directory*.

    A RUNNING record whose heartbeat has gone cold is reported as ORPHANED
    rather than believed. That is the whole point: the file says RUNNING because
    nothing got the chance to say otherwise, and reading it literally is how
    three dead runs were mistaken for live ones.
    """
    path = Path(directory) / STATUS_FILE
    if not path.exists():
        return {"state": "NOT_TRACKED", "reason": f"no {STATUS_FILE} in {directory}"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"state": "NOT_TRACKED", "reason": f"unreadable: {exc}"}
    if payload.get("state") == RUNNING:
        age = time.time() - float(payload.get("heartbeat", 0))
        if age > STALE_HEARTBEAT_SECONDS:
            payload["state"] = ORPHANED
            payload["reason"] = (f"last heartbeat {age:.0f}s ago, over the "
                                 f"{STALE_HEARTBEAT_SECONDS:.0f}s bound")
    return payload


# ------------------------------------------------------------------ retention


def mark_preserved(workspace: Path, *, ticket: str, outcome: str,
                   pinned: bool = False) -> None:
    """Record what a preserved workspace is, so retention can reason about it.

    ``pinned`` is the escape hatch and it is honoured absolutely: a workspace
    from a contaminated or unexplained run is evidence, and evidence is not
    subject to a disk budget.
    """
    try:
        (Path(workspace) / MARKER_FILE).write_text(json.dumps({
            "schema": SCHEMA, "ticket": ticket, "outcome": outcome,
            "preserved_at": time.time(), "pinned": bool(pinned),
        }, indent=1) + "\n", encoding="utf-8")
    except OSError:
        pass  # a workspace we cannot mark is one retention will simply not touch


def _directory_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def survey(base: Path, prefix: str = "gafitas_") -> list[dict]:
    """Every preserved workspace under *base*, with its size and marker."""
    base = Path(base)
    out: list[dict] = []
    if not base.is_dir():
        return out
    for entry in sorted(base.glob(prefix + "*")):
        if not entry.is_dir():
            continue
        marker: dict = {}
        try:
            marker = json.loads((entry / MARKER_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        out.append({
            "path": str(entry),
            "bytes": _directory_bytes(entry),
            "age_days": (time.time() - marker.get("preserved_at", mtime)) / 86400.0,
            "pinned": bool(marker.get("pinned")),
            "ticket": marker.get("ticket"),
            "outcome": marker.get("outcome"),
            "marked": bool(marker),
        })
    return out


def retain(base: Path, *, max_age_days: float | None = None,
           max_total_bytes: int | None = None, prefix: str = "gafitas_",
           repo: Path | None = None, dry_run: bool = True,
           log_path: Path | None = None) -> dict:
    """Apply a retention policy to preserved workspaces. Dry run by default.

    Order of operations, and each one matters:

    * pinned workspaces are removed from consideration first, always;
    * an unmarked workspace is never deleted -- we do not know what it is, and
      guessing about evidence is exactly the failure mode this guards against;
    * age is applied before size, because "too old" is a policy and "too much"
      is an accident;
    * size deletes oldest-first until the budget is met, never newest-first;
    * every deletion is appended to a log with its reason before it happens.

    ``repo`` additionally prunes git worktree registrations whose directories
    are gone. Those are what make every git command in the repository slower,
    and 61 of them had accumulated here.
    """
    entries = survey(base, prefix)
    # F-176, corrected. The first version of this comment said NOTHING in the
    # codebase passes pinned=True. That was wrong, and the error was mine:
    # work.py:762 passes `pinned=result.outcome in (HARNESS_INVALID,
    # PROVIDER_ERROR)`, which a grep for the literal `pinned=True` does not
    # find. Two of the four were already protected.
    #
    # The real gap is narrower and still real. ACCEPTANCE_UNUSABLE is not in
    # that caller's list, EXPERIMENT_CONTAMINATED is not produced by it at all,
    # and both are exactly the harness admitting its own defect. Checking the
    # outcome here as well means the protection does not depend on one caller
    # remembering to ask for it -- which is the difference between a policy and
    # a habit.
    #
    # The outcome was already surveyed and already logged; it simply was not
    # consulted.
    protected = [e for e in entries
                 if e["pinned"] or not e["marked"]
                 or str(e.get("outcome", "")).upper() in NEVER_DELETE]
    candidates = [e for e in entries
                  if not e["pinned"] and e["marked"]
                  and str(e.get("outcome", "")).upper() not in NEVER_DELETE]
    candidates.sort(key=lambda e: -e["age_days"])   # oldest first

    doomed: list[dict] = []
    if max_age_days is not None:
        for entry in candidates:
            if entry["age_days"] > max_age_days:
                doomed.append({**entry, "reason": f"older than {max_age_days}d"})

    if max_total_bytes is not None:
        # F-176. This compared a candidate dict against doomed dicts that carry
        # an extra "reason" key, so it was never equal and `remaining` held
        # everything -- including what the age pass had already condemned.
        # Reproduced: two directories of 1088 bytes each appeared TWICE in the
        # selection, once per reason, and selected_bytes reported 4352 for a
        # corpus of 4352 of which only 2176 was going anywhere. The deletion
        # log recorded each path twice, the figure a human reads to decide
        # whether to run this was double, and the second rmtree on an
        # already-removed path landed in `failed`, so a clean run reported
        # failures it had not had.
        condemned = {e["path"] for e in doomed}
        remaining = [e for e in candidates if e["path"] not in condemned]
        total = sum(e["bytes"] for e in remaining)
        for entry in remaining:
            if total <= max_total_bytes:
                break
            doomed.append({**entry, "reason": f"total over {max_total_bytes} bytes"})
            total -= entry["bytes"]

    log = Path(log_path) if log_path else Path(base) / DELETION_LOG
    removed, failed = [], []
    if not dry_run:
        for entry in doomed:
            record = {"at": time.time(), "path": entry["path"],
                      "bytes": entry["bytes"], "reason": entry["reason"],
                      "ticket": entry.get("ticket"), "outcome": entry.get("outcome")}
            try:
                with open(log, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            except OSError:
                pass                      # never delete what we could not record
            else:
                try:
                    shutil.rmtree(entry["path"], ignore_errors=False)
                    removed.append(entry["path"])
                except OSError as exc:
                    failed.append({"path": entry["path"], "error": str(exc)})

    pruned = None
    if repo is not None and not dry_run:
        try:
            proc = subprocess.run(["git", "worktree", "prune", "-v"], cwd=str(repo),
                                  capture_output=True, text=True, timeout=120)
            pruned = proc.stdout.strip() or "(nothing to prune)"
        except (OSError, subprocess.TimeoutExpired) as exc:
            pruned = f"prune failed: {exc}"

    return {
        "scanned": len(entries),
        "protected": len(protected),
        "total_bytes": sum(e["bytes"] for e in entries),
        "selected": [{"path": e["path"], "bytes": e["bytes"], "reason": e["reason"]}
                     for e in doomed],
        "selected_bytes": sum(e["bytes"] for e in doomed),
        "removed": removed,
        "failed": failed,
        "worktrees_pruned": pruned,
        "dry_run": dry_run,
        "log": str(log),
    }


def adopt(base: Path, prefix: str = "gafitas_", dry_run: bool = True) -> dict:
    """Bring workspaces that predate the marker under the policy.

    An unmarked workspace is protected forever, which is correct and means the
    854 that already existed when the policy was written would never be
    reclaimed. This marks them from what can be read off disk -- the directory
    name and its mtime -- and records ``inferred: true`` so nobody later mistakes
    a reconstruction for a record.

    Deliberately separate from ``retain`` and deliberately dry-run by default:
    adopting is what makes deletion possible, so it is an operator's decision
    and not a side effect of running a cleanup.
    """
    base = Path(base)
    adopted, skipped = [], []
    for entry in sorted(base.glob(prefix + "*")):
        if not entry.is_dir():
            continue
        if (entry / MARKER_FILE).exists():
            skipped.append(str(entry))
            continue
        ticket = entry.name[len(prefix):].rsplit("_", 1)[0] or entry.name
        if not dry_run:
            try:
                stamp = entry.stat().st_mtime
            except OSError:
                continue
            try:
                (entry / MARKER_FILE).write_text(json.dumps({
                    "schema": SCHEMA, "ticket": ticket, "outcome": "UNKNOWN",
                    "preserved_at": stamp, "pinned": False, "inferred": True,
                }, indent=1) + "\n", encoding="utf-8")
            except OSError:
                continue
        adopted.append(str(entry))
    return {"adopted": len(adopted), "already_marked": len(skipped),
            "dry_run": dry_run, "sample": adopted[:5]}
