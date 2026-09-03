"""Acceptance for dogfood08. Frozen BEFORE the agent runs, and not edited after.

An extract-to-module refactor. The byte-identity assertion is the point: a
behaviour test cannot tell "moved" from "retyped from memory", and those are
different things -- the second is where a refactor quietly changes something.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The body as it stands before the move, so "identical" means something after
#: it. Recorded here rather than read from tools.py at test time, because after
#: a correct move tools.py no longer has it to compare against.
ORIGINAL = '''def _unreachable(source: str) -> list[tuple[int, str]]:
    """Statements that can never execute, as (line, the statement's kind).

    Only the certain case: a statement standing directly after a return, raise,
    break or continue in the SAME block. No flow analysis, no cleverness, no
    opinions about style -- just the one thing that is unreachable under every
    reading of the language.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for previous, statement in zip(block, block[1:]):
                if isinstance(previous, TERMINATORS):
                    found.append((getattr(statement, "lineno", 0),
                                  type(statement).__name__))
    return sorted(set(found))'''


def test_the_new_module_exists_and_defines_both_names():
    from localprog import deadcode
    assert hasattr(deadcode, "TERMINATORS")
    assert hasattr(deadcode, "_unreachable")


def test_tools_no_longer_defines_them_itself():
    source = (ROOT / "localprog" / "tools.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_unreachable" not in defined, "tools.py still defines it"
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert "TERMINATORS" not in assigned, "tools.py still assigns it"


def test_tools_still_reaches_the_same_function():
    from localprog import deadcode, tools
    assert tools._unreachable is deadcode._unreachable


def test_the_moved_code_is_identical_not_rewritten():
    """A move, not a rewrite. One changed space and it is a rewrite."""
    from localprog import deadcode
    got = textwrap.dedent(inspect.getsource(deadcode._unreachable)).rstrip()
    assert got == ORIGINAL, (
        "the function was retyped rather than moved; first difference at "
        f"line {next((i + 1 for i, (a, b) in enumerate(zip(got.splitlines(), ORIGINAL.splitlines())) if a != b), '(length)')}"
    )


def test_the_terminators_are_the_same_four():
    from localprog import deadcode
    assert {t.__name__ for t in deadcode.TERMINATORS} == {
        "Return", "Raise", "Continue", "Break"}


def test_the_behaviour_is_unchanged(tmp_path):
    from localprog import tools
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",))
    out = tools.dispatch(ctx, "edit", {"path": "m.py", "old": "    return 1",
                                       "new": "    return 1\n    return 2"})
    assert out.ok
    assert "ERROR_UNREACHABLE_CODE" in out.value


def test_a_clean_file_still_says_nothing(tmp_path):
    from localprog import tools
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",))
    out = tools.dispatch(ctx, "edit", {"path": "m.py", "old": "    return 1",
                                       "new": "    return 2"})
    assert out.ok and "UNREACHABLE" not in out.value
