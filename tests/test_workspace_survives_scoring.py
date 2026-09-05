"""F-128: a run that SUCCEEDS must still be scoreable by whoever ran it.

The retention policy keeps a workspace when the run FAILED, because that is
where evidence of a defect lives, and deletes it when the run succeeded. Right
for evidence, and silently wrong for measurement: the Phase 5 comparison took
its verdict by running the acceptance suite in the workspace, so it could not
score any GAFITAS run whose agent finished cleanly. pytest was handed a
directory that no longer existed, the pair became INFRA_ERROR, and the cohort
reported GAFITAS 0/8.

The defect drops exactly the runs most likely to be successes, and it is
ONE-SIDED -- the bare arm writes into the runner's own directory, which is never
disposed. In p5r2 it hit ga06, whose recovered suite is 22 passed.

These tests pin it closed in both directions: a caller can demand the box
survives, and the default retention behaviour is unchanged for everyone who
does not ask.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from localprog import work
from localprog.provider import FakeProvider

BROKEN = "def halve(n):\n    return n / 0\n"
FIXED = "def halve(n):\n    return n / 2\n"
TEST = ("from pkg.calc import halve\n\n\ndef test_halve():\n"
        "    assert halve(10) == 5\n")


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "demo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "calc.py").write_text(BROKEN, encoding="utf-8")
    (root / "tests" / "test_calc.py").write_text(TEST, encoding="utf-8")
    git(["init", "-q"], root)
    git(["config", "user.email", "t@t"], root)
    git(["config", "user.name", "t"], root)
    git(["add", "-A"], root)
    git(["commit", "-qm", "init"], root)
    return root


def tc(name, **arguments):
    # A script entry IS the message; FakeProvider wraps it. Wrapping it here as
    # well produced three ERROR_NO_TOOL_CALL turns and a stall, which is the
    # provider faithfully reporting that my script contained no calls.
    return {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}]}


def ticket(repo):
    return work.Ticket(
        ticket_id="f128", repo=repo,
        objective="Arregla halve() para que devuelva la mitad.",
        write_scope=("pkg/",), acceptance_tests=("tests/test_calc.py",),
        allowed_new_files=(), full_suite=False, max_turns=8)


def solve(repo, tmp_path, *, preserve):
    # base_dir must already exist -- the workspace is created INSIDE it, not
    # as it. Learned by writing this test wrong first.
    boxes = tmp_path / "boxes"
    boxes.mkdir(parents=True, exist_ok=True)
    script = [tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
              tc("run_tests"),
              tc("finish", status="DONE", summary="arreglado")]
    return work.run_ticket(
        ticket(repo), "fake",
        provider_factory=lambda _m: FakeProvider(script),
        out_dir=tmp_path / "out", base_dir=boxes,
        preserve_workspace=preserve)


def test_a_successful_run_can_be_asked_to_keep_its_workspace(box, tmp_path):
    """The whole finding: without this a caller cannot score its own run."""
    result = solve(box, tmp_path, preserve=True)
    assert result.outcome in ("PASS", "PASS_UNCONFIRMED", "CANDIDATE"), result.outcome
    where = Path(result.workspace["path"])
    assert where.is_dir(), "the box a scorer is about to test must still exist"


def test_the_preserved_workspace_holds_the_agents_change(box, tmp_path):
    result = solve(box, tmp_path, preserve=True)
    body = (Path(result.workspace["path"]) / "pkg" / "calc.py").read_text(encoding="utf-8")
    assert "n / 2" in body, "and it must still hold what the agent wrote"


def test_the_default_is_unchanged(box, tmp_path):
    """Nobody who did not ask gets a different retention policy. A fix that
    quietly stopped cleaning up would fill the disk instead."""
    result = solve(box, tmp_path, preserve=False)
    assert result.workspace["preserved"] is False
    assert not Path(result.workspace["path"]).exists()


def test_a_failed_run_keeps_its_workspace_either_way(box, tmp_path):
    """The original guarantee, undisturbed: evidence of a defect is never
    deleted, whether or not the caller asked for preservation."""
    boxes = tmp_path / "boxes"
    boxes.mkdir(parents=True, exist_ok=True)
    script = [{"content": "no llamo nada"}] * 40
    result = work.run_ticket(
        ticket(box), "fake",
        provider_factory=lambda _m: FakeProvider(script),
        out_dir=tmp_path / "out", base_dir=boxes)
    assert result.outcome not in ("PASS", "PASS_UNCONFIRMED", "CANDIDATE")
    assert result.workspace["preserved"] is True


@pytest.mark.parametrize("preserve", [True, False])
def test_preservation_cannot_change_the_run(box, tmp_path, preserve):
    """Disposal happens after the loop has ended and the patch is sealed, so
    the agent cannot observe it. Same script, same outcome, same changes."""
    result = solve(box, tmp_path / f"b{preserve}", preserve=preserve)
    assert result.outcome in ("PASS", "PASS_UNCONFIRMED", "CANDIDATE")
    assert result.changed_files == ["pkg/calc.py"]
    assert result.turns_used == 3


def test_the_patch_is_sealed_whether_or_not_the_box_is_kept(box, tmp_path):
    """The independent route. Replay scoring reads the sealed writes, so it must
    not depend on the retention decision either -- two routes that shared a
    single point of failure would not be two routes."""
    for preserve in (True, False):
        result = solve(box, tmp_path / f"seal{preserve}", preserve=preserve)
        assert result.patch_path and Path(result.patch_path).exists()
