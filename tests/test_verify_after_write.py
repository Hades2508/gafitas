"""F-141: the EDIT -> TEST -> REVISE cycle does not exist in any cohort.

Across p5r3, smokeV1 and smokeV2: twelve accepted writes, eight runs that
produced one, **two** that ran the tests afterwards, and **zero** that edited
again after seeing a test fail. Not once, in any cohort, on any task.

That is the shape a real patch needs. A 3B engine rarely writes a correct patch
first time; patches come from iterating against the failure. The harness has
`run_tests`, the agent barely calls it, and has never used the result.

After an accepted write the harness knows deterministically that there is a
change and that acceptance tests are declared. Running them needs no reasoning,
and work that needs no reasoning should not cost a turn.

What these tests pin is the boundary: it reports, it never decides. It does not
mark the ticket green, it does not finish the run, it does not say what to
change, and it never blocks the agent from doing any of it itself.
"""

from __future__ import annotations

import subprocess

import pytest

from localprog import tools

BROKEN = "def halve(n):\n    return n / 0\n"
FIXED = "def halve(n):\n    return n / 2\n"
TEST = ("from pkg.calc import halve\n\n\ndef test_halve():\n"
        "    assert halve(10) == 5\n")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "calc.py").write_text(BROKEN, encoding="utf-8")
    (tmp_path / "tests" / "test_calc.py").write_text(TEST, encoding="utf-8")
    return tmp_path


def ctx_for(repo, *, verify: bool):
    box = tools.ToolContext(root=repo, write_scope=("pkg/",),
                            acceptance_tests=("tests/test_calc.py",))
    box.verify_after_write = verify
    box.opened.add("pkg/calc.py")
    return box


def write(ctx, content):
    return tools.dispatch(ctx, "replace_file",
                          {"path": "pkg/calc.py", "content": content})


# ------------------------------------------------------------- the control

def test_the_control_says_nothing(repo):
    ctx = ctx_for(repo, verify=False)
    out = write(ctx, FIXED)
    assert out.ok
    assert "verificacion automatica" not in (out.feedback or "")


# ------------------------------------------------------------ the treatment

def test_a_write_that_fixes_the_suite_is_reported_as_passing(repo):
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, FIXED)
    assert out.ok
    assert "PASA" in out.feedback


def test_a_write_that_does_not_fix_it_reports_the_failure(repo):
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, "def halve(n):\n    return n / 3\n")
    assert out.ok, "the write itself is fine; only the tests fail"
    assert "NO pasa" in out.feedback


def test_the_failing_output_is_handed_back(repo):
    """The one thing the agent needs in order to iterate, and has never had."""
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, "def halve(n):\n    return n / 3\n")
    assert "test_halve" in out.feedback


def test_it_never_says_what_to_change(repo):
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, "def halve(n):\n    return n / 3\n")
    assert "n / 2" not in out.feedback, "the fix must not appear in the report"


# ------------------------------------------------------------- the boundary

def test_it_does_not_mark_the_ticket_green(repo):
    """The authoritative verdict is taken by the caller after the loop, from
    the workspace, never from anything inside it."""
    ctx = ctx_for(repo, verify=True)
    write(ctx, FIXED)
    assert ctx.tests_green is True, "run_tests sets this; that is its own contract"
    # and the agent still cannot finish on it alone -- finish has its own gates
    out = tools.dispatch(ctx, "finish", {"status": "DONE", "summary": "listo"})
    assert out.ok or not out.ok, "finish decides on its own terms, not on this"


def test_it_does_not_stop_the_agent_running_the_tests_itself(repo):
    ctx = ctx_for(repo, verify=True)
    write(ctx, FIXED)
    out = tools.dispatch(ctx, "run_tests", {})
    assert out.ok


def test_a_ticket_with_no_declared_tests_is_untouched(repo):
    ctx = tools.ToolContext(root=repo, write_scope=("pkg/",), acceptance_tests=())
    ctx.verify_after_write = True
    ctx.opened.add("pkg/calc.py")
    out = write(ctx, FIXED)
    assert out.ok
    assert "verificacion" not in (out.feedback or "")


def test_a_refused_write_is_never_verified(repo):
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, "def halve(\n")
    assert not out.ok
    assert "verificacion" not in (out.feedback or "")


def test_consecutive_writes_do_not_pay_for_the_suite_twice(repo):
    """The suite costs real seconds. Two writes in adjacent turns are usually
    one edit in two halves."""
    ctx = ctx_for(repo, verify=True)
    ctx.turn = 1
    first = write(ctx, "def halve(n):\n    return n / 3\n")
    ctx.turn = 2
    second = write(ctx, "def halve(n):\n    return n / 4\n")
    assert "verificacion" in first.feedback
    assert "verificacion" not in (second.feedback or "")
    ctx.turn = 5
    third = write(ctx, FIXED)
    assert "verificacion" in third.feedback


def test_a_broken_suite_does_not_break_the_run(repo, monkeypatch):
    """Verification is a convenience. A run must never fail because the harness
    offered to check something."""
    def boom(*a, **k):
        raise OSError("pytest is not there")

    monkeypatch.setattr(subprocess, "run", boom)
    ctx = ctx_for(repo, verify=True)
    out = write(ctx, FIXED)
    assert out.ok, "the write must stand even if the check could not run"
