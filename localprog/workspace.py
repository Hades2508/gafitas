"""The box: an isolated tree the model may edit, and how it is disposed of.

Contract §C.1. Rollback is "delete the box", which is strictly stronger than
restoring the files a plan happened to name -- it also takes with it anything
the model created that nobody declared.

Disposal policy, from experiments.md: a workspace is removed when the run
succeeded and **preserved when it did not**. The previous runner deleted the
tree in a ``finally``, so every failure destroyed its own evidence on the way
out -- including the failures that were the harness's fault.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import HarnessInvalid

GIT_TIMEOUT = 120.0
IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv", "venv")


def _git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True,
            text=True, timeout=GIT_TIMEOUT, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessInvalid(f"git {' '.join(args)} failed: {exc}") from exc


@dataclass
class Workspace:
    path: Path
    kind: str                       # GIT_WORKTREE | COPY | SYNTHETIC
    source: str | None = None
    commit: str | None = None
    disposed: bool = False
    preserved: bool = False
    _origin: Path | None = field(default=None, repr=False)

    def describe(self) -> dict:
        return {
            "kind": self.kind, "path": str(self.path), "source": self.source,
            "commit": self.commit, "preserved": self.preserved, "disposed": self.disposed,
        }

    def dispose(self, *, preserve: bool) -> None:
        """Remove the box, or keep it and say so. Idempotent."""
        if self.disposed:
            return
        if preserve:
            self.preserved = True
            self.disposed = True
            return
        if self.kind == "GIT_WORKTREE" and self._origin is not None:
            _git(["worktree", "remove", "--force", str(self.path)], cwd=self._origin)
            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
        else:
            shutil.rmtree(self.path, ignore_errors=True)
        self.disposed = True


def _is_git_repo(repo: Path) -> bool:
    result = _git(["rev-parse", "--is-inside-work-tree"], cwd=repo)
    return result.returncode == 0 and result.stdout.strip() == "true"


def _mkdtemp(prefix: str, base_dir: Path | None) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir) if base_dir else None))


def open_repo(
    repo: Path, *, commit: str | None = None, prefix: str = "localprog_",
    base_dir: Path | None = None,
) -> Workspace:
    """A worktree at *commit* when git allows it; an isolated copy otherwise.

    The fallback is explicit and recorded rather than silent: a mission whose
    target is not a git repository is still runnable, but the evidence says
    ``COPY`` so nobody later reads it as a reproducible checkout.
    """
    repo = Path(repo)
    if not repo.is_dir():
        raise HarnessInvalid(f"mission repo does not exist: {repo}")

    if _is_git_repo(repo):
        target = _mkdtemp(prefix, base_dir)
        target.rmdir()  # git insists on creating the directory itself
        ref = commit or "HEAD"
        result = _git(["worktree", "add", "--detach", str(target), ref], cwd=repo)
        if result.returncode != 0:
            raise HarnessInvalid(
                f"git worktree add failed for {repo} @ {ref}: {result.stderr.strip()[:400]}"
            )
        head = _git(["rev-parse", "HEAD"], cwd=target)
        return Workspace(
            path=target, kind="GIT_WORKTREE", source=str(repo),
            commit=head.stdout.strip() or None, _origin=repo,
        )

    target = _mkdtemp(prefix, base_dir)
    shutil.copytree(repo, target, dirs_exist_ok=True, ignore=IGNORE)
    return Workspace(path=target, kind="COPY", source=str(repo))


def synthesize(
    files: dict[str, str], *, prefix: str = "localprog_screen_",
    base_dir: Path | None = None,
) -> Workspace:
    """A fresh tree containing exactly *files*. Used by the frozen screen."""
    target = _mkdtemp(prefix, base_dir)
    for relative, content in files.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
    return Workspace(path=target, kind="SYNTHETIC")
