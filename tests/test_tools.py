"""Every tool, every error path. The point is that none of them raises.

Each of these is a path that, in the previous runner, either crashed the whole
campaign or was silently scored as a model failure.
"""

from __future__ import annotations

import pytest

from conftest import call
from localprog import errors, tools


# ------------------------------------------------------------------ read_file


def test_read_file_returns_numbered_lines(ctx):
    out = call(ctx, "read_file", path="calc.py")
    assert out.ok
    assert out.value.splitlines()[0] == "   1\tdef add(a, b):"


def test_read_file_honours_start_and_end(ctx):
    out = call(ctx, "read_file", path="calc.py", start=1, end=2)
    assert out.ok and len(out.value.splitlines()) == 2


def test_read_file_missing_names_siblings(ctx):
    out = call(ctx, "read_file", path="nope.py")
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_FILE_NOT_FOUND
    assert "calc.py" in out.feedback  # the hint that lets the model recover


def test_read_file_on_directory_is_a_tool_error(ctx):
    """The 'list_dir sobre fichero' bug class, in its surviving form."""
    out = call(ctx, "read_file", path="pkg")
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_IS_DIRECTORY


def test_read_file_rejects_traversal(ctx):
    out = call(ctx, "read_file", path="../secret.py")
    assert not out.ok and out.code == errors.ERROR_PATH_OUTSIDE_REPO


def test_read_file_rejects_absolute_path(ctx):
    out = call(ctx, "read_file", path="C:/Windows/win.ini")
    assert not out.ok and out.code == errors.ERROR_PATH_OUTSIDE_REPO


def test_read_file_rejects_binary(ctx, repo):
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe binary")
    out = call(ctx, "read_file", path="blob.bin")
    assert not out.ok and out.code == errors.ERROR_NOT_TEXT


def test_read_file_truncates_a_huge_file(ctx, repo):
    (repo / "big.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    out = call(ctx, "read_file", path="big.py")
    assert out.ok and "truncado" in out.value
    assert len(out.value.splitlines()) < tools.MAX_READ_LINES


def test_read_file_accepts_backslash_separators(ctx):
    out = call(ctx, "read_file", path="pkg\\mod.py")
    assert out.ok and "VALUE" in out.value


# ----------------------------------------------------------------------- grep


def test_grep_finds_matches(ctx):
    out = call(ctx, "grep", pattern="def divide")
    assert out.ok and out.value["hits"][0]["file"] == "calc.py"


def test_grep_with_no_hits_is_not_an_error(ctx):
    out = call(ctx, "grep", pattern="zzzz_nothing")
    assert out.ok and out.value["hits"] == [] and "0 coincidencias" in out.value["note"]


def test_grep_bad_regex_is_a_tool_error(ctx):
    out = call(ctx, "grep", pattern="([unclosed")
    assert not out.ok and out.code == errors.ERROR_BAD_PATTERN


def test_grep_caps_results(ctx, repo):
    (repo / "many.py").write_text("hit = 1\n" * 200, encoding="utf-8")
    out = call(ctx, "grep", pattern="hit")
    assert out.ok and len(out.value["hits"]) == tools.MAX_GREP_HITS


def test_grep_skips_unreadable_files(ctx, repo):
    (repo / "blob.py").write_bytes(b"\x00\xff")
    out = call(ctx, "grep", pattern="def")
    assert out.ok  # a binary file is not a grep failure


# --------------------------------------------------------------- list_symbols


def test_list_symbols_reports_names_and_lines(ctx):
    out = call(ctx, "list_symbols", path="calc.py")
    assert out.ok and out.value[0].startswith("add (línea 1)")


def test_list_symbols_on_broken_file_is_a_tool_error(ctx, repo):
    """The propagated SyntaxError that used to kill the whole run."""
    (repo / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    out = call(ctx, "list_symbols", path="broken.py")
    assert not out.ok and out.tool_error and out.code == errors.ERROR_SYNTAX
    assert "línea" in out.feedback


def test_list_symbols_on_directory_is_a_tool_error(ctx):
    out = call(ctx, "list_symbols", path="pkg")
    assert not out.ok and out.code == errors.ERROR_IS_DIRECTORY


# ----------------------------------------------------------------------- edit


def test_edit_applies_and_records_the_file(ctx, repo):
    out = call(ctx, "edit", path="calc.py", old="    return a / b", new="    return b and a / b")
    assert out.ok and ctx.changed_files == {"calc.py"}
    assert "return b and a / b" in (repo / "calc.py").read_text(encoding="utf-8")


def test_two_edits_to_one_file_both_survive(ctx, repo):
    """SAME_FILE_MULTI_OP_COLLISION: still live in Generalista, dead here."""
    assert call(ctx, "edit", path="calc.py", old="def add", new="def suma").ok
    assert call(ctx, "edit", path="calc.py", old="def multiply", new="def producto").ok
    text = (repo / "calc.py").read_text(encoding="utf-8")
    assert "def suma" in text and "def producto" in text


def test_edit_no_match_explains_how_to_recover(ctx):
    """A failed edit must leave the agent able to act, not just informed.

    Rewritten for F-27. The message used to say the text was not found and to
    go read the file -- which was exactly the loop the dogfood run was already
    trapped in: read, guess, fail, read again, six times. It now shows the
    nearest region as it actually stands, and names the line-based alternative.
    """
    out = call(ctx, "edit", path="calc.py", old="return a // b", new="x")
    assert not out.ok and out.code == errors.ERROR_NO_MATCH
    assert "indentacion" in out.feedback
    assert "read_file" in out.feedback or "replace_lines" in out.feedback


def test_edit_multiple_matches_lists_lines(ctx):
    out = call(ctx, "edit", path="calc.py", old="    return", new="    return")
    assert not out.ok and out.code == errors.ERROR_MULTIPLE_MATCHES
    assert "líneas" in out.feedback


def test_edit_empty_old_is_refused(ctx):
    out = call(ctx, "edit", path="calc.py", old="", new="x")
    assert not out.ok and out.code == errors.ERROR_EMPTY_OLD


def test_edit_that_breaks_syntax_writes_nothing(ctx, repo):
    before = (repo / "calc.py").read_text(encoding="utf-8")
    out = call(ctx, "edit", path="calc.py", old="def add(a, b):", new="def add(a, b:")
    assert not out.ok and out.code == errors.ERROR_SYNTAX_AFTER_EDIT
    assert (repo / "calc.py").read_text(encoding="utf-8") == before
    assert ctx.changed_files == set()


def test_edit_outside_write_scope_is_refused(ctx, repo):
    out = call(ctx, "edit", path="test_calc.py", old="def test_add", new="def test_suma")
    assert not out.ok and out.code == errors.ERROR_NOT_IN_WRITE_SCOPE
    assert "calc.py" in out.feedback


def test_edit_non_string_arguments_are_invalid_calls(ctx):
    out = call(ctx, "edit", path="calc.py", old=3, new="x")
    assert not out.ok and out.invalid_call and out.code == errors.ERROR_BAD_ARGUMENTS


# ----------------------------------------------------------------- write_file


def test_write_file_creates_an_allowed_new_file(ctx, repo):
    out = call(ctx, "write_file", path="nuevo.py", content="VALUE = 2\n")
    assert out.ok and (repo / "nuevo.py").exists()


def test_write_file_refuses_an_existing_file(ctx):
    ctx.allowed_new_files = ("calc.py",)
    out = call(ctx, "write_file", path="calc.py", content="x = 1\n")
    assert not out.ok and out.code == errors.ERROR_FILE_EXISTS


def test_write_file_outside_scope_is_refused(ctx):
    out = call(ctx, "write_file", path="otro.py", content="x = 1\n")
    assert not out.ok and out.code == errors.ERROR_NOT_IN_WRITE_SCOPE


def test_write_file_with_broken_syntax_writes_nothing(ctx, repo):
    out = call(ctx, "write_file", path="nuevo.py", content="def f(:\n")
    assert not out.ok and out.code == errors.ERROR_SYNTAX_AFTER_EDIT
    assert not (repo / "nuevo.py").exists()


# ------------------------------------------------------------------ run_tests


def test_run_tests_reports_the_failing_suite(ctx):
    out = call(ctx, "run_tests")
    assert out.ok and out.value["passed"] is False and ctx.tests_green is False


def test_run_tests_goes_green_after_the_real_fix(ctx):
    fix = "    if b == 0:\n        raise ValueError(\"division by zero\")\n    return a / b"
    assert call(ctx, "edit", path="calc.py", old="    return a / b", new=fix).ok
    out = call(ctx, "run_tests")
    assert out.ok and out.value["passed"] is True and ctx.tests_green is True


def test_run_tests_green_with_explicit_node_ids(ctx):
    fix = "    if b == 0:\n        raise ValueError(\"division by zero\")\n    return a / b"
    call(ctx, "edit", path="calc.py", old="    return a / b", new=fix)
    out = call(ctx, "run_tests", node_ids=["test_calc.py"])
    assert out.ok and ctx.tests_green is True


def test_run_tests_rejects_targets_outside_the_repo(ctx):
    out = call(ctx, "run_tests", node_ids=["../../evil.py"])
    assert not out.ok and out.code == errors.ERROR_PATH_OUTSIDE_REPO


def test_run_tests_bad_node_ids_is_an_invalid_call(ctx):
    out = call(ctx, "run_tests", node_ids="test_calc.py")
    assert not out.ok and out.invalid_call


# --------------------------------------------------------------------- finish


def test_finish_without_changes_is_refused(ctx):
    out = call(ctx, "finish", summary="listo")
    assert not out.ok and out.code == errors.ERROR_NOTHING_CHANGED


#: The edit that makes the frozen fixture suite green: test_divide_by_zero
#: is the one failing test in conftest's repo, which is the whole point of
#: that fixture. F-29 means a DONE test now has to actually fix something.
GREEN_DIVIDE = (
    "    if b == 0:\n"
    "        raise ValueError('division by zero')\n"
    "    return a / b"
)


def test_finish_after_a_change_is_accepted(ctx):
    """DONE is accepted once the agent has edited AND watched the suite pass.

    The run_tests call is not ceremony: since F-29 an agent that never looked
    cannot claim completion, because the dogfood agent did exactly that on turn
    12 of a 40-turn budget.
    """
    call(ctx, "edit", path="calc.py", old="    return a / b", new=GREEN_DIVIDE)
    call(ctx, "run_tests")
    out = call(ctx, "finish", summary="hecho")
    assert out.ok and out.value == "FINISHED[DONE]"


def test_finish_done_without_running_the_tests_is_refused(ctx):
    """F-29 itself."""
    call(ctx, "edit", path="calc.py", old="a + b", new="a + b + 0")
    out = call(ctx, "finish", summary="hecho")
    assert not out.ok and out.code == errors.ERROR_NOT_VERIFIED
    assert "run_tests" in out.feedback
    assert "BLOCKED" in out.feedback  # and told the honest way out


def test_finish_done_is_refused_while_the_tests_are_red(ctx):
    """An agent that ran the tests, saw them fail, and says DONE anyway."""
    call(ctx, "edit", path="calc.py", old="a + b", new="a - b")
    call(ctx, "run_tests")
    out = call(ctx, "finish", summary="hecho")
    assert not out.ok and out.code == errors.ERROR_NOT_VERIFIED


def test_a_ticket_with_no_declared_tests_can_still_finish(ctx, tmp_path):
    """The rule must not make finishing impossible where there is nothing to
    run -- it would turn every such ticket into a forced BLOCKED."""
    free = tools.ToolContext(root=ctx.root, write_scope=("calc.py",), acceptance_tests=())
    call(free, "edit", path="calc.py", old="a + b", new="a + b + 0")
    out = call(free, "finish", summary="sin tests declarados")
    assert out.ok and out.value == "FINISHED[DONE]"


@pytest.mark.parametrize("name", sorted(tools.SPECS))
def test_no_tool_raises_on_a_totally_empty_repo(tmp_path, name):
    """Totality: every tool called with plausible junk yields a classified
    outcome, never an exception."""
    ctx = tools.ToolContext(root=tmp_path, write_scope=("a.py",), acceptance_tests=("t.py",))
    outcome = tools.dispatch(ctx, name, {
        "path": "a.py", "pattern": "x", "old": "a", "new": "b",
        "content": "x = 1\n", "summary": "s",
    })
    assert outcome.name == name
