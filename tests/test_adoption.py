"""ADOPTION_V3: a tool that exists, applies and is available, and is never called.

    replace_file          9 calls out of 85 write calls across two smokes
    replace_symbol_body   0 calls out of 53

Both are in the schema the model receives every turn. A primitive with zero
invocations cannot improve any rate, so the question is not yet whether it is
good.

The chain is kept separate here because collapsing any two stages hides the
actual failure:

    AVAILABLE -> APPLICABLE -> OFFERED -> CONCRETE -> SELECTED -> CALLED
              -> ACCEPTED -> USEFUL

The tests that matter most are the ones about what the treatment must NOT do:
it must not hide an alternative, must not force a selection, must not fill in
the body, and must not claim applicability it cannot compute.
"""

from __future__ import annotations

import pytest

from localprog import adoption, tools

MODULE = '''"""m."""


def helper(a, b):
    total = a + b
    return total


class Box:
    def size(self):
        return 1
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "m.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain text\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def f(\n", encoding="utf-8")
    return tmp_path


def ctx_for(repo, *, treatment: bool):
    box = tools.ToolContext(root=repo, write_scope=("**",),
                            allowed_new_files=("new.py",))
    box.adoption_ledger = adoption.AdoptionLedger()
    box.concrete_offers = treatment
    return box


# ------------------------------------------------ applicability is computed

def test_a_file_that_does_not_exist_is_not_replaceable(repo):
    ok, why = adoption.file_is_replaceable(repo, "nope.py", opened=set())
    assert not ok and "does not exist" in why


def test_an_unread_file_is_not_replaceable(repo):
    """Not an opinion: replace_file itself refuses a blind overwrite, so
    claiming applicability there would be claiming something false."""
    ok, why = adoption.file_is_replaceable(repo, "m.py", opened=set())
    assert not ok and "not been opened" in why
    ok, _ = adoption.file_is_replaceable(repo, "m.py", opened={"m.py"})
    assert ok


def test_symbols_are_found_and_qualified(repo):
    ok, why, symbols = adoption.symbol_is_replaceable(repo, "m.py")
    assert ok
    assert "helper" in symbols and "Box.size" in symbols


def test_a_language_without_a_symbol_index_is_not_applicable(repo):
    ok, why, symbols = adoption.symbol_is_replaceable(repo, "notes.txt")
    assert not ok and "no symbol index" in why and symbols == []


def test_a_file_that_does_not_parse_is_not_applicable(repo):
    ok, why, _ = adoption.symbol_is_replaceable(repo, "broken.py")
    assert not ok and "does not parse" in why


def test_an_ambiguous_bare_name_is_excluded(tmp_path):
    """The primitive refuses an ambiguous name, so applicability must not
    count one. Two classes with the same method name is the ordinary case."""
    (tmp_path / "two.py").write_text(
        "class A:\n    def run(self):\n        return 1\n\n\n"
        "class B:\n    def run(self):\n        return 2\n", encoding="utf-8")
    ok, _why, symbols = adoption.symbol_is_replaceable(tmp_path, "two.py")
    assert not ok or all(not s.endswith(".run") for s in symbols)


def test_a_one_line_definition_is_not_applicable(tmp_path):
    (tmp_path / "one.py").write_text("def f(): return 1\n", encoding="utf-8")
    ok, why, _ = adoption.symbol_is_replaceable(tmp_path, "one.py")
    assert not ok and "own lines" in why


# ------------------------------------------------------ the offer is concrete

def test_the_offer_names_the_path_and_the_real_symbols(repo):
    _ok, _why, symbols = adoption.symbol_is_replaceable(repo, "m.py")
    text = adoption.concrete_offer("replace_symbol_body", "m.py", symbols)
    assert "replace_symbol_body(" in text
    assert "'m.py'" in text
    assert "helper" in text


def test_the_offer_never_fills_in_the_body(repo):
    """The line the treatment must not cross. The path and the symbol are
    things the harness knows; what the code should say is the task."""
    _ok, _why, symbols = adoption.symbol_is_replaceable(repo, "m.py")
    text = adoption.concrete_offer("replace_symbol_body", "m.py", symbols)
    assert "body=<" in text
    assert "return" not in text, "no code may appear in an offer"


# -------------------------------------------- the treatment, through dispatch

def test_the_control_says_nothing_extra(repo):
    ctx = ctx_for(repo, treatment=False)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    out = tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
    assert not out.ok
    assert "replace_symbol_body(" not in out.feedback


def test_the_treatment_names_the_concrete_call(repo):
    ctx = ctx_for(repo, treatment=True)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    out = tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
    assert not out.ok
    assert "replace_symbol_body(path='m.py'" in out.feedback
    assert "replace_file(path='m.py'" in out.feedback


def test_the_treatment_offers_nothing_where_nothing_applies(repo):
    ctx = ctx_for(repo, treatment=True)
    out = tools.dispatch(ctx, "edit", {"path": "nope.py", "old": "a", "new": "b"})
    assert not out.ok
    assert "replace_symbol_body(" not in out.feedback


def test_the_treatment_does_not_hide_the_old_path(repo):
    """Adoption bought by removing an alternative is not adoption. write_file
    and edit must remain exactly as available as before."""
    ctx = ctx_for(repo, treatment=True)
    assert "write_file" in tools.SPECS and "edit" in tools.SPECS
    out = tools.dispatch(ctx, "write_file", {"path": "new.py", "content": "x = 1\n"})
    assert out.ok, "the ordinary path must still work untouched"


def test_the_treatment_does_not_force_the_primitive(repo):
    """An offer is not an obligation. The model may keep using edit and it
    must keep working."""
    ctx = ctx_for(repo, treatment=True)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    out = tools.dispatch(ctx, "edit", {"path": "m.py", "old": "return total",
                                       "new": "return a * b"})
    assert out.ok


# ------------------------------------------------------------- the accounting

def test_every_write_attempt_is_recorded_in_both_arms(repo):
    for treatment in (False, True):
        ctx = ctx_for(repo, treatment=treatment)
        tools.dispatch(ctx, "read_file", {"path": "m.py"})
        tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
        body = ctx.adoption_ledger.to_dict()
        assert body["by_primitive"]["replace_symbol_body"]["applicable"] == 1


def test_the_control_records_the_opportunity_it_declined_to_mention(repo):
    """The whole point of the A/B: the control must count what it did not say,
    or the two arms cannot be compared."""
    ctx = ctx_for(repo, treatment=False)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
    body = ctx.adoption_ledger.to_dict()
    slot = body["by_primitive"]["replace_symbol_body"]
    assert slot["applicable"] == 1 and slot["offered"] == 0


def test_a_call_to_the_primitive_is_recorded_as_selected(repo):
    ctx = ctx_for(repo, treatment=True)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    out = tools.dispatch(ctx, "replace_symbol_body",
                         {"path": "m.py", "name": "helper", "body": "return a * b"})
    assert out.ok
    rows = [r for r in ctx.adoption_ledger.to_dict()["events"]
            if r["primitive"] == "replace_symbol_body"]
    assert any(r["called"] and r["accepted"] for r in rows)


def test_the_fallback_tool_is_named(repo):
    ctx = ctx_for(repo, treatment=False)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
    rows = [r for r in ctx.adoption_ledger.to_dict()["events"]
            if r["primitive"] == "replace_symbol_body"]
    assert rows and rows[-1]["fallback_tool_selected"] == "edit"


def test_the_ledger_survives_serialisation(repo):
    import json

    ctx = ctx_for(repo, treatment=True)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    tools.dispatch(ctx, "edit", {"path": "m.py", "old": "", "new": "x"})
    assert json.dumps(ctx.adoption_ledger.to_dict())


def test_no_accounting_without_a_ledger(repo):
    """A run that did not ask for the accounting must behave exactly as before."""
    ctx = tools.ToolContext(root=repo, write_scope=("**",))
    out = tools.dispatch(ctx, "read_file", {"path": "m.py"})
    assert out.ok
