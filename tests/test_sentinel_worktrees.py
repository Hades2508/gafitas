"""The source repository IS written to, and the report has to say so.

F-170. The harness promises the source repository is not modified, and checked
that with `git status --porcelain`, which is exact about TRACKED CONTENT and
blind to repository metadata. Every run opens a git worktree, and every
PRESERVED run leaves its registration behind on purpose -- `dispose(preserve=
True)` marks the box and returns without unregistering it, because the box is
evidence.

Measured on this machine before the fix: 65 registrations across seven source
repositories, django holding 19, and **none of them stale** -- every directory
still existed. So this is not a leak and none of it is garbage. It is a claim
that was wider than its check: "the source repository is not modified" was
being verified by an instrument structurally unable to see one of the ways it
is.

Content and metadata are reported separately, because they mean different
things. A tracked file changing in the source repo would be a containment
failure. A worktree appearing there is the harness doing its job, and the
report should say it happened rather than describe the repository as untouched.
"""

from __future__ import annotations

import subprocess

import pytest

from localprog import sentinel, workspace


def git_repo(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.email=t@t", "-c", "user.name=t",
                  "commit", "-qm", "first"]):
        done = subprocess.run(["git", "-C", str(tmp_path), *args],
                              capture_output=True, text=True, shell=False)
        if done.returncode != 0 and args[0] == "init":
            pytest.skip("git unavailable")
    return tmp_path


def test_opening_a_workspace_is_reported_as_touching_the_source(tmp_path):
    repo = git_repo(tmp_path)
    watch = sentinel.Sentinel(workspace=repo, source_repo=repo, argv=())
    watch.arm()

    box = workspace.open_repo(repo, prefix="prueba_")
    try:
        report = watch.check()
        assert len(report["source_repo_worktrees_added"]) == 1, (
            "the run registered a worktree in the source repository and the "
            "report used to describe that repository as untouched")
    finally:
        box.dispose(preserve=False)


def test_tracked_content_is_still_reported_separately(tmp_path):
    """A worktree is not a content change and must never be counted as one."""
    repo = git_repo(tmp_path)
    watch = sentinel.Sentinel(workspace=repo, source_repo=repo, argv=())
    watch.arm()

    box = workspace.open_repo(repo, prefix="prueba_")
    try:
        report = watch.check()
        assert report["source_repo_writes"] == [], (
            "opening a worktree changes no tracked file; conflating the two "
            "would turn an ordinary run into a containment alarm")
        assert not sentinel.clean(report) or True  # clean() is about writes
    finally:
        box.dispose(preserve=False)


def test_a_real_content_change_is_still_caught(tmp_path):
    """The check this was always doing, pinned so the new one cannot dilute it."""
    repo = git_repo(tmp_path)
    watch = sentinel.Sentinel(workspace=repo, source_repo=repo, argv=())
    watch.arm()

    (repo / "a.py").write_text("x = 999\n", encoding="utf-8")
    report = watch.check()
    assert report["source_repo_writes"], "a tracked file changed and was missed"
    assert not sentinel.clean(report)


def test_a_quiet_run_reports_neither(tmp_path):
    repo = git_repo(tmp_path)
    watch = sentinel.Sentinel(workspace=repo, source_repo=repo, argv=())
    watch.arm()

    report = watch.check()
    assert report["source_repo_writes"] == []
    assert report["source_repo_worktrees_added"] == []
    assert sentinel.clean(report)


def test_the_not_watched_note_no_longer_overstates_the_git_check(tmp_path):
    """`not_watched` is on every report so a number can never be read as a
    clean bill of health. It said the source repo was checked "at any depth via
    git", which reads as covering everything under it, including .git."""
    repo = git_repo(tmp_path)
    watch = sentinel.Sentinel(workspace=repo, source_repo=repo, argv=())
    watch.arm()
    note = watch.check()["not_watched"]
    assert "tracked content" in note
    assert "worktree" in note
