"""F-112: what a child process writes outside the workspace.

``run`` permits ``python``, so a child writes with the user's permissions. That
boundary is declared and accepted. What was wrong is the accounting: the
harness's after-the-fact check builds its file list from git inside the
workspace, so an out-of-repo write was neither prevented nor detected, and
"zero unauthorised writes" described only half the boundary without saying so.

Two design mistakes were made and measured on the way here, and the tests keep
both fixed:

  1. Manifesting the likely target directories. The system temp directory holds
     124 599 entries and takes 8.3 seconds to list. Eight seconds around every
     run would dominate the timings this campaign exists to take.
  2. Capping that manifest at 4000 entries instead. The canary was written into
     exactly that directory and fell outside the sample, so the check reported
     clean and saw nothing.

What works is watching the paths the command NAMES, at file granularity: O(1),
exact, and the named file is what the command was going to write anyway.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from localprog import sentinel, tools


@pytest.fixture()
def box(tmp_path):
    (tmp_path / "src").mkdir()
    return tmp_path


def test_a_named_path_outside_the_workspace_is_watched(box):
    target = Path(tempfile.gettempdir()) / "whatever.txt"
    argv = ["python", "-c", f"open(r'{target}','w').write('x')"]
    named = sentinel.named_destinations(argv, box)
    assert target.resolve() in named


def test_a_path_inside_the_workspace_is_not_watched(box):
    inside = box / "src" / "x.py"
    named = sentinel.named_destinations(["python", "-c", f"open(r'{inside}','w')"], box)
    assert inside.resolve() not in named


def test_a_path_is_found_inside_a_code_string(box):
    """The realistic accident lives inside `python -c`, not in a bare argv."""
    target = Path(tempfile.gettempdir()) / "deep.txt"
    argv = ["python", "-c",
            f"import os\nwith open(r'{target}', 'w') as fh:\n    fh.write('x')"]
    assert target.resolve() in sentinel.named_destinations(argv, box)


def test_a_write_to_a_named_outside_path_is_detected(box):
    target = Path(tempfile.gettempdir()) / "sentinel_test_canary.txt"
    target.unlink(missing_ok=True)
    watch = sentinel.Sentinel(workspace=box, argv=["python", "-c", f"open(r'{target}','w')"])
    watch.arm()
    target.write_text("PWNED", encoding="utf-8")
    report = watch.check()
    target.unlink(missing_ok=True)

    assert not sentinel.clean(report)
    assert any(c["name"] == "sentinel_test_canary.txt" and c["kind"] == "created"
               for c in report["outside_writes"])


def test_a_deletion_outside_is_detected_too(box):
    target = Path(tempfile.gettempdir()) / "sentinel_test_victim.txt"
    target.write_text("original", encoding="utf-8")
    watch = sentinel.Sentinel(workspace=box, argv=["python", "-c", f"os.remove(r'{target}')"])
    watch.arm()
    target.unlink()
    report = watch.check()
    assert any(c["kind"] == "deleted" for c in report["outside_writes"])


def test_a_quiet_command_reports_clean(box):
    watch = sentinel.Sentinel(workspace=box, argv=["python", "-c", "print(1)"])
    watch.arm()
    assert sentinel.clean(watch.check())


def test_every_report_says_what_it_did_not_watch(box):
    """Clean or not. The number must never read as a statement about the whole
    filesystem, and the only way to guarantee that is to say so every time."""
    watch = sentinel.Sentinel(workspace=box, argv=["python", "-c", "print(1)"])
    watch.arm()
    report = watch.check()
    assert report["not_watched"]
    assert "did not name" in report["not_watched"]


def test_a_source_repo_change_is_caught_at_any_depth(box, tmp_path):
    """git, not a one-level manifest: the source repository is the one thing
    this factory promises not to modify, and a manifest cannot see depth."""
    import subprocess
    repo = tmp_path / "source"
    (repo / "deep" / "nested").mkdir(parents=True)
    (repo / "deep" / "nested" / "file.py").write_text("x = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)

    watch = sentinel.Sentinel(workspace=box, source_repo=repo, argv=["python", "-c", "pass"])
    watch.arm()
    (repo / "deep" / "nested" / "file.py").write_text("x = 2\n", encoding="utf-8")
    report = watch.check()
    assert report["source_repo_writes"], "a change deep in the source repo was missed"
    assert not sentinel.clean(report)


def test_run_carries_the_sentinel_report_and_tells_the_agent(box):
    """End to end through the real tool: detected, reported, and said out loud."""
    target = Path(tempfile.gettempdir()) / "sentinel_run_canary.txt"
    target.unlink(missing_ok=True)
    ctx = tools.ToolContext(root=box, write_scope=("src/**/*.py",))
    out = tools.run(ctx, argv=["python", "-c", f"open(r'{target}','w').write('PWNED')"],
                    timeout=30)
    existed = target.exists()
    target.unlink(missing_ok=True)

    assert existed, "the boundary itself is unchanged: the write still happens"
    assert not sentinel.clean(out["sentinel"]), "and it is now detected"
    assert "FUERA del workspace" in (out.get("note") or "")
    assert ctx.outside_writes, "and recorded on the context for the run record"


def test_a_harmless_command_is_not_flagged(box):
    ctx = tools.ToolContext(root=box, write_scope=("src/**/*.py",))
    out = tools.run(ctx, argv=["python", "-c", "print(1)"], timeout=30)
    assert sentinel.clean(out["sentinel"])
    assert "FUERA del workspace" not in (out.get("note") or "")


def test_a_directory_too_large_to_manifest_is_declared_not_sampled():
    """The mistake that made the first version useless: it capped its sample at
    4000 entries in a directory holding 124 599, reported clean, and saw
    nothing. Now it refuses to manifest and says which directory it skipped."""
    huge, too_big = sentinel._manifest(Path(tempfile.gettempdir()))
    if too_big:
        assert huge == {}, "a capped manifest must be empty, never partial"
