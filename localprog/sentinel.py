"""Watching what a child process writes outside the workspace, within a budget.

WHY THIS EXISTS
---------------
``run`` executes with an interpreter allowlist and ``shell=False``, and refuses
bash, curl, pip, rm and every git subcommand that writes. It permits ``python``.
A child therefore writes with the user's permissions: a boundary the operator
has declared and accepted.

What was wrong is not the boundary, it is the accounting. The harness has an
after-the-fact check for exactly this case -- ``signal_scope_respected``, whose
own docstring says it exists because ``run`` can bypass the tool layer -- and it
builds its file list from git inside the workspace. Git stops at the repository.
So an out-of-SCOPE write inside the repo had two layers, an out-of-REPO write
had none, and "zero unauthorised writes across 350 runs" described only the
first without saying so (F-112).

WHY IT IS NOT A DIRECTORY SWEEP
-------------------------------
The obvious design -- manifest the likely targets before and after -- was built
first and measured second, which was the wrong order:

    <user>\\AppData\\Local\\Temp                124 599 entries, 8.3 s to list
    <user>                                        148 entries, 2 ms

Eight seconds around every ``run`` would dominate the very timings this campaign
exists to take, and an earlier version of that idea silently capped its sample
at 4000 entries and therefore missed a canary written into that same directory.
A check that is too slow to keep on gets turned off, and a check that quietly
samples is worse than none.

WHAT IT WATCHES, AND WHY EACH ONE
---------------------------------
    SOURCE REPOSITORY   `git status --porcelain`. Exact, any depth, and it is
                        the one thing this factory promises not to modify.
                        Nothing in the harness checked that promise before.
    NAMED DESTINATIONS  every path-like token in the argv -- INCLUDING inside a
                        `python -c` code string -- that resolves outside the
                        workspace. Each is statted before and after: O(1), exact,
                        and the file is what the command was going to write
                        anyway. Watching their PARENT directories was the first
                        design and it failed on the only case that matters, since
                        the parent of a named temp file has 124 599 entries.
    HOME                one level, 2 ms measured. Cheap enough to always watch.

WHAT IT CANNOT SEE
------------------
A write to a directory nobody named, that is not home and not the source repo --
a path the child computed rather than wrote down. Every report says so in
``not_watched``, on success as well as failure, so the result can never be read
as a clean bill of health for the whole filesystem.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: A directory this large is not manifested; the report says so instead of
#: sampling it. Measured: 148 entries costs 2 ms and 124 599 costs 8.3 s.
MAX_ENTRIES = 20_000
GIT_TIMEOUT = 30.0

#: Anything that looks like a filesystem path inside an argv token, including
#: inside a quoted code string. Deliberately greedy about what counts as a
#: candidate: a false candidate costs one cheap scandir, a missed one costs the
#: whole point of the check.
_PATHLIKE = re.compile(r"""[A-Za-z]:[\\/][^\s'"]+|/[^\s'"]{2,}|\.\.[\\/][^\s'"]+""")


def _manifest(directory: Path) -> tuple[dict, bool]:
    """(name -> (mtime_ns, size)) one level deep, and whether it was too big."""
    seen: dict = {}
    try:
        with os.scandir(directory) as entries:
            for n, entry in enumerate(entries):
                if n >= MAX_ENTRIES:
                    return {}, True
                try:
                    stat = entry.stat(follow_symlinks=False)
                    seen[entry.name] = (stat.st_mtime_ns, stat.st_size)
                except OSError:
                    seen[entry.name] = None
    except OSError:
        return {}, False
    return seen, False


def _git_dirty(repo: Path) -> list[str] | None:
    """What git reports as changed in *repo*, or None if it could not be asked."""
    try:
        done = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                              capture_output=True, text=True,
                              timeout=GIT_TIMEOUT, shell=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [line[3:].strip() for line in done.stdout.splitlines() if line.strip()]


def _worktrees(repo: Path) -> list[str] | None:
    """The worktrees registered in *repo*, or None if it could not be asked.

    F-170. `git status --porcelain` is exact about TRACKED CONTENT and blind to
    repository metadata, so the source-repo check reported "clean" while every
    preserved run left a permanent entry in `<repo>/.git/worktrees/`. Measured
    on this machine: 65 registrations across seven source repositories, django
    holding 19, none of them stale -- they are live preserved workspaces, which
    is the intended policy.

    So this is not a leak and nothing here is garbage. It is a claim that was
    wider than its check: "the source repository is not modified" was verified
    by an instrument structurally unable to see one of the ways it is. Watching
    them costs one git call and makes the report true.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT, shell=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return sorted(line[len("worktree "):].strip()
                  for line in done.stdout.splitlines()
                  if line.startswith("worktree "))


def named_destinations(argv, workspace: Path) -> list[Path]:
    """The exact paths the command names that lie outside the workspace.

    Reads the whole argv as text, so a path inside a ``python -c`` string counts
    -- that is where the realistic accident lives.

    Returns the PATHS THEMSELVES, not their parents. That is the difference
    between this working and not: the first version watched parent directories,
    and the parent of a named temp file is a directory with 124 599 entries that
    is too expensive to manifest, so the check reported "too big to watch" and
    saw nothing. Statting one named file is O(1) and exact, and the file is what
    the command was going to write anyway.
    """
    try:
        box = workspace.resolve()
    except OSError:
        box = workspace
    out: list[Path] = []
    for token in argv if isinstance(argv, (list, tuple)) else [str(argv)]:
        for hit in _PATHLIKE.findall(str(token)):
            candidate = Path(hit.rstrip("'\"),;"))
            try:
                resolved = (candidate if candidate.is_absolute()
                            else (workspace / candidate)).resolve()
            except OSError:
                continue
            if box == resolved or box in resolved.parents:
                continue                      # inside the workspace: not our business
            if resolved not in out:
                out.append(resolved)
    return out


def _stamp(path: Path):
    """(mtime_ns, size) for one path, or None when it does not exist."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@dataclass
class Sentinel:
    """A before/after view of a bounded, mostly command-derived watch set."""

    workspace: Path
    source_repo: Path | None = None
    argv: list | tuple | str = ()
    _dirs: list = field(default_factory=list, repr=False)
    _before: dict = field(default_factory=dict, repr=False)
    _source_before: list | None = field(default=None, repr=False)
    _worktrees_before: list | None = field(default=None, repr=False)
    _too_big: list = field(default_factory=list, repr=False)
    _names: list = field(default_factory=list, repr=False)
    _name_before: dict = field(default_factory=dict, repr=False)

    def watched(self) -> list[Path]:
        """The directories manifested one level. Only ones small enough."""
        out = []
        try:
            home = Path.home().resolve()
            if home.exists():
                out.append(home)
        except OSError:
            pass
        return out

    def named(self) -> list[Path]:
        """The exact paths the command itself named, outside the workspace."""
        return named_destinations(self.argv, self.workspace)

    def arm(self) -> None:
        self._dirs = self.watched()
        self._names = self.named()
        self._before, self._too_big = {}, []
        for directory in self._dirs:
            seen, too_big = _manifest(directory)
            if too_big:
                self._too_big.append(str(directory))
            else:
                self._before[str(directory)] = seen
        self._name_before = {str(p): _stamp(p) for p in self._names}
        self._source_before = (_git_dirty(self.source_repo)
                               if self.source_repo else None)
        self._worktrees_before = (_worktrees(self.source_repo)
                                  if self.source_repo else None)

    def check(self) -> dict:
        changes: list[dict] = []
        # The named paths first: O(1) each, exact, and the commonest accident.
        for path in self._names:
            before = self._name_before.get(str(path))
            after = _stamp(path)
            if before == after:
                continue
            kind = ("created" if before is None else
                    "deleted" if after is None else "modified")
            changes.append({"where": str(path.parent), "name": path.name,
                            "kind": kind, "named_by_the_command": True})
        for directory in self._dirs:
            before = self._before.get(str(directory))
            if before is None:
                continue                      # too big to manifest; reported below
            after, too_big = _manifest(directory)
            if too_big:
                continue
            for name, stamp in after.items():
                if name not in before:
                    changes.append({"where": str(directory), "name": name,
                                    "kind": "created"})
                elif before[name] != stamp:
                    changes.append({"where": str(directory), "name": name,
                                    "kind": "modified"})
            for name in before:
                if name not in after:
                    changes.append({"where": str(directory), "name": name,
                                    "kind": "deleted"})

        source_changes: list[str] = []
        registered: list[str] = []
        if self.source_repo is not None:
            after = _git_dirty(self.source_repo)
            if after is not None and self._source_before is not None:
                source_changes = sorted(set(after) - set(self._source_before))
            after_trees = _worktrees(self.source_repo)
            if after_trees is not None and self._worktrees_before is not None:
                registered = sorted(set(after_trees) - set(self._worktrees_before))

        return {
            "watched": [str(d) for d in self._dirs],
            "named_paths": [str(p) for p in self._names],
            "source_repo": str(self.source_repo) if self.source_repo else None,
            "outside_writes": changes,
            "source_repo_writes": source_changes,
            # Metadata the run added to the source repository. Not an
            # unauthorised write and not counted as one -- a preserved
            # workspace is registered there on purpose -- but it IS the source
            # repository being written to, and the report used to say clean.
            "source_repo_worktrees_added": registered,
            "too_big_to_watch": self._too_big,
            # Present on every report, clean or not. The number this yields must
            # never be mistaken for a statement about the whole filesystem.
            "not_watched": ("any directory the command did not name, other than "
                            "the home directory and the source repository; and "
                            "anything more than one level inside a watched "
                            "directory. The source repository is checked at any "
                            "depth via git for tracked content, and separately "
                            "for worktree registrations; other metadata under "
                            "its .git is not watched."),
        }


def clean(report: dict) -> bool:
    """Whether the sentinel saw nothing. Does not mean nothing happened."""
    return not report.get("outside_writes") and not report.get("source_repo_writes")
