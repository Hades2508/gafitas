"""Acceptance for dogfood06. Frozen BEFORE the agent runs, and not edited after.

run_tests clips pytest's output to the last TEST_OUTPUT_TAIL characters and says
nothing about it. ``run`` clips the same way and does say so. An agent reading a
silently clipped failure report is reading a lie of omission -- which is exactly
the class of defect F-46 was about.
"""

from __future__ import annotations

import pytest

from localprog import tools


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "tests").mkdir()
    return tmp_path


def write_test(repo, body: str) -> None:
    (repo / "tests" / "test_generated.py").write_text(body, encoding="utf-8")


def ctx_for(repo):
    return tools.ToolContext(
        root=repo, write_scope=("**/*.py",),
        acceptance_tests=("tests/test_generated.py",),
    )


def test_a_short_run_says_nothing_about_truncation(repo):
    write_test(repo, "def test_ok():\n    assert True\n")
    out = tools.dispatch(ctx_for(repo), "run_tests", {})
    assert out.ok
    assert "note" not in out.value, "nothing was clipped, so there is nothing to say"


def test_a_clipped_run_says_so(repo):
    """A failure that prints far more than the tail budget."""
    write_test(repo, (
        "def test_noisy():\n"
        "    for i in range(4000):\n"
        "        print('x' * 200)\n"
        "    assert False\n"
    ))
    out = tools.dispatch(ctx_for(repo), "run_tests", {})
    assert out.ok
    assert out.value["passed"] is False
    note = out.value.get("note")
    assert note, "the output was clipped and run_tests did not say so"
    assert "truncada" in note
    assert str(tools.TEST_OUTPUT_TAIL) in note


def test_the_clipping_itself_is_unchanged(repo):
    write_test(repo, (
        "def test_noisy():\n"
        "    for i in range(4000):\n"
        "        print('y' * 200)\n"
        "    assert False\n"
    ))
    out = tools.dispatch(ctx_for(repo), "run_tests", {})
    assert len(out.value["output"]) <= tools.TEST_OUTPUT_TAIL
