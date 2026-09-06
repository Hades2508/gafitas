"""A concrete offer never names a path the mission forbids (F-148).

`_offer_a_primitive` asked applicability of the TREE alone -- does the file
exist, has it been read, do its symbols resolve -- and never consulted the write
scope. Sized against sealed evidence before the treatment had ever run in a
cohort, 12 of the 123 offers it would have made named a forbidden path, and
three of the four reproduced cases named the ACCEPTANCE TEST: the offer read
`tests/test_scope.py`'s own test functions back to a model that had just tried
to edit them.

The scope guard still refused the write, so nothing could land and the oracle
was never actually reachable. What the offer would have spent is a turn on a
call guaranteed to be refused, and what it would have taught is to rewrite the
test that judges the run.

Every test below is written in the STRONG form: it first asserts that the
primitive really is applicable to the tree, so that a silent offer is the scope
rule working rather than the fixture failing to set up the situation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localprog import adoption, tools

MODULE = (
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "\n"
    "def beta():\n"
    "    return 2\n"
)


def sandbox(tmp_path: Path, *, scope=("src/mod.py",), allowed=(), offers=True):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "mod.py").write_text(MODULE, encoding="utf-8")
    (root / "tests" / "test_mod.py").write_text(
        "def test_alpha():\n    assert alpha() == 1\n", encoding="utf-8")
    # allowed_new_files must be passed at construction: the scope object is
    # built from it there, and setting the attribute afterwards leaves the
    # real boundary unaware of the permission.
    ctx = tools.ToolContext(root=root, write_scope=tuple(scope),
                            allowed_new_files=tuple(allowed))
    ctx.concrete_offers = offers
    # Both files read, so `opened` cannot be what makes an offer silent.
    tools.dispatch(ctx, "read_file", {"path": "src/mod.py"})
    tools.dispatch(ctx, "read_file", {"path": "tests/test_mod.py"})
    return root, ctx


def offer(ctx, path):
    return tools._offer_a_primitive(
        ctx, "edit", {"path": path, "old": "x", "new": "y"}, tools.ERROR_NO_MATCH)


# ------------------------------------------------------- the mechanism works

def test_an_in_scope_path_still_gets_a_concrete_offer(tmp_path):
    """The fix must not silence the treatment it is protecting."""
    root, ctx = sandbox(tmp_path)
    can_symbol, _why, _symbols = adoption.symbol_is_replaceable(root, "src/mod.py")
    assert can_symbol
    text = offer(ctx, "src/mod.py")
    assert "replace_symbol_body" in text
    assert "src/mod.py" in text


def test_the_offer_names_symbols_but_never_a_body(tmp_path):
    root, ctx = sandbox(tmp_path)
    text = offer(ctx, "src/mod.py")
    assert "alpha" in text and "beta" in text
    assert "return 1" not in text


# ------------------------------------------------------------- the fix bites

def test_a_path_outside_write_scope_gets_no_offer(tmp_path):
    root, ctx = sandbox(tmp_path)
    # Strong form: the primitive IS applicable to this file. Only the scope
    # rule may make the offer silent.
    can_symbol, _why, symbols = adoption.symbol_is_replaceable(root, "tests/test_mod.py")
    can_file, _why2 = adoption.file_is_replaceable(root, "tests/test_mod.py",
                                                   opened=ctx.opened)
    assert can_symbol and can_file, "fixture failed to make the file offerable"
    assert offer(ctx, "tests/test_mod.py") == ""


def test_the_offer_never_names_the_acceptance_test(tmp_path):
    """The exact shape found in sealed evidence, as a regression."""
    root, ctx = sandbox(tmp_path)
    ctx.acceptance_tests = ("tests/test_mod.py",)
    text = offer(ctx, "tests/test_mod.py")
    assert "tests/test_mod.py" not in text
    assert "test_alpha" not in text


@pytest.mark.parametrize("path", ["tests/test_mod.py", "src/../tests/test_mod.py"])
def test_silence_does_not_depend_on_how_the_path_is_spelled(tmp_path, path):
    _root, ctx = sandbox(tmp_path)
    assert offer(ctx, path) == ""


# ------------------------------------------- what the fix must NOT take away

def test_a_file_this_run_created_is_still_offerable(tmp_path):
    """`_check_writable` lets a run rewrite what it created, and the offer
    inherits that rather than re-deciding it.

    This is why the guard is CALLED instead of copied: a second copy of the
    scope rule would have had to remember this exemption too.
    """
    root, ctx = sandbox(tmp_path, scope=(), allowed=("out.py",))
    wrote = tools.dispatch(ctx, "write_file", {"path": "out.py", "content": MODULE})
    assert wrote.ok, wrote.code
    assert "out.py" in ctx.created_files
    tools.dispatch(ctx, "read_file", {"path": "out.py"})
    text = offer(ctx, "out.py")
    assert "replace_symbol_body" in text and "out.py" in text


def test_the_control_arm_is_unchanged(tmp_path):
    _root, ctx = sandbox(tmp_path, offers=False)
    assert offer(ctx, "src/mod.py") == ""
    assert offer(ctx, "tests/test_mod.py") == ""


def test_the_opportunity_is_still_recorded_when_the_offer_is_silent(tmp_path):
    """Recording is unconditional by design; only OFFERING is the treatment.

    A scope-silenced offer must not also lose the observation, or the A/B
    accounting would count fewer opportunities in the arm that speaks less.
    """
    _root, ctx = sandbox(tmp_path)
    ctx.adoption_ledger = adoption.AdoptionLedger()
    before = len(ctx.adoption_ledger.opportunities)
    assert offer(ctx, "tests/test_mod.py") == ""
    # Two: the ledger records one opportunity per PRIMITIVE, not per call.
    got = ctx.adoption_ledger.opportunities[before:]
    assert {o.primitive for o in got} == {"replace_file", "replace_symbol_body"}
    assert all(o.applicable and not o.called for o in got)
