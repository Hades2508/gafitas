"""Round 2: moving code without retyping it, and not giving up in silence.

Three changes, each from a measured failure rather than a hunch.

F-66  the tool surface could not put code anywhere without the model retyping
      it through a JSON argument, and that round trip loses text: on the matched
      evaluation the same model reproduced a function byte for byte 72% of the
      time answering in prose and 40% doing it through the tools.
F-67  the give-up question required declared acceptance tests, so a mission that
      declares none -- most of the ones where giving up early IS the failure --
      was never asked anything. 15 of 100 dev runs were shown the right function
      and finished without ever opening it.
F-68  a dogfood candidate passed its acceptance and left a second `return out`
      below the first. Correct behaviour, unpromotable diff, manual cleanup.
"""

from __future__ import annotations

import pytest

from localprog import loop, tools
from localprog.provider import FakeProvider


def tc(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


SRC = (
    "import os\n"
    "\n"
    "\n"
    "def helper(a):\n"
    '    """Add one, and keep the quotes intact."""\n'
    "    return a + 1\n"
    "\n"
    "\n"
    "def other():\n"
    "    pass\n"
)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src.py").write_text(SRC, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def ctx(repo):
    return tools.ToolContext(root=repo, write_scope=("**/*.py",),
                             allowed_new_files=("out.py", "answer.txt"))


def call(ctx, name, **kwargs):
    return tools.dispatch(ctx, name, kwargs)


# ------------------------------------------------------------- F-66 copy_region

def test_a_region_arrives_byte_for_byte(ctx, repo):
    out = call(ctx, "copy_region", source="src.py", start=4, end=6, dest="out.py")
    assert out.ok, out.feedback
    written = (repo / "out.py").read_text(encoding="utf-8")
    assert written == "\n".join(SRC.splitlines()[3:6]) + "\n"
    assert '"""Add one, and keep the quotes intact."""' in written, (
        "the quotes are the thing that kept coming back doubled"
    )


def test_copying_marks_the_destination_changed(ctx):
    call(ctx, "copy_region", source="src.py", start=4, end=6, dest="out.py")
    assert ctx.changed_files == {"out.py"}


def test_it_refuses_to_overwrite_and_says_what_to_do_instead(ctx, repo):
    (repo / "out.py").write_text("already here\n", encoding="utf-8")
    out = call(ctx, "copy_region", source="src.py", start=4, end=6, dest="out.py")
    assert not out.ok and out.code == "ERROR_FILE_EXISTS"
    assert "at=" in out.feedback
    assert (repo / "out.py").read_text(encoding="utf-8") == "already here\n"


def test_at_inserts_into_a_file_that_already_exists(ctx, repo):
    (repo / "dest.py").write_text("# header\n# footer\n", encoding="utf-8")
    out = call(ctx, "copy_region", source="src.py", start=9, end=10,
               dest="dest.py", at=2)
    assert out.ok, out.feedback
    assert (repo / "dest.py").read_text(encoding="utf-8") == (
        "# header\ndef other():\n    pass\n# footer\n"
    )


def test_the_write_scope_still_holds(repo):
    ctx = tools.ToolContext(root=repo, write_scope=("allowed/**",))
    out = tools.dispatch(ctx, "copy_region", {"source": "src.py", "start": 4,
                                              "end": 6, "dest": "elsewhere.py"})
    assert not out.ok and out.code == "ERROR_NOT_IN_WRITE_SCOPE"


def test_a_range_off_the_end_of_the_file_is_refused(ctx):
    out = call(ctx, "copy_region", source="src.py", start=900, end=910, dest="out.py")
    assert not out.ok and out.code == "ERROR_BAD_RANGE"


def test_a_python_destination_that_would_not_parse_is_refused(ctx, repo):
    """Moving a method to module level needs a dedent, and the tool says so
    rather than leaving a file nobody can import."""
    (repo / "cls.py").write_text(
        "class A:\n    def m(self):\n        return 1\n", encoding="utf-8")
    out = call(ctx, "copy_region", source="cls.py", start=2, end=3, dest="out.py")
    assert not out.ok and out.code == "ERROR_SYNTAX_AFTER_EDIT"
    assert "indentacion" in out.feedback
    assert not (repo / "out.py").exists()


def test_a_non_python_destination_takes_anything(ctx, repo):
    """The answer of a search does not have to be importable."""
    out = call(ctx, "copy_region", source="src.py", start=5, end=5, dest="answer.txt")
    assert out.ok, out.feedback
    assert (repo / "answer.txt").read_text(encoding="utf-8").strip().startswith('"""')


# --------------------------------------------------------- F-67 the give-up

def drive(ctx, script, max_turns=20):
    provider = FakeProvider(script)
    result = loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O",
                           protocol_name="A", max_turns=max_turns)
    shown = []
    for c in provider.calls:
        for message in c["messages"]:
            if message.get("role") == "tool":
                shown.append(message.get("content", ""))
    return result, shown


def test_blocked_is_questioned_even_with_no_declared_tests(ctx):
    """The gate used to require a suite. Most missions where giving up early is
    the failure do not declare one."""
    result, shown = drive(ctx, [
        tc("finish", summary="no lo encuentro", status="BLOCKED"),
        tc("finish", summary="sigo sin encontrarlo", status="BLOCKED"),
    ])
    assert result.outcome == loop.FINISHED
    assert any("antes de darte por vencido" in t for t in shown)


def test_the_question_names_candidates_the_agent_never_opened(ctx):
    result, shown = drive(ctx, [
        tc("search_code", query="add one to a number"),
        tc("finish", summary="no lo encuentro", status="BLOCKED"),
        tc("finish", summary="de verdad que no", status="BLOCKED"),
    ])
    assert result.outcome == loop.FINISHED
    joined = "\n".join(shown)
    assert "NO has abierto" in joined
    assert "src.py" in joined


def test_it_does_not_nag_when_the_candidates_were_read(ctx):
    _result, shown = drive(ctx, [
        tc("search_code", query="add one to a number"),
        {"content": "", "tool_calls": [{"function": {
            "name": "read_symbol",
            "arguments": {"path": "src.py", "name": "helper"}}}]},
        tc("finish", summary="mirado y descartado", status="BLOCKED"),
        tc("finish", summary="mirado y descartado", status="BLOCKED"),
    ])
    joined = "\n".join(shown)
    assert "NO has abierto" not in joined
    assert "descartarlos esta justificado" in joined


def test_a_second_blocked_is_still_accepted_without_further_questions(ctx):
    result, _ = drive(ctx, [
        tc("finish", summary="no", status="BLOCKED"),
        tc("finish", summary="no", status="BLOCKED"),
    ])
    assert result.outcome == loop.FINISHED
    assert ctx.finish_status == "BLOCKED"


# ------------------------------------------------------- F-68 unreachable code

def test_an_edit_that_makes_code_unreachable_says_so(ctx):
    out = call(ctx, "edit", path="src.py", old="    return a + 1",
               new="    return a + 1\n    print('never')")
    assert out.ok, "the write succeeds; this is a warning, not a refusal"
    assert "ERROR_UNREACHABLE_CODE" in out.value
    assert "muerto" in out.value


def test_pre_existing_unreachable_code_is_not_blamed_on_this_edit(ctx, repo):
    (repo / "dead.py").write_text(
        "def g():\n    return 1\n    return 2\n", encoding="utf-8")
    out = call(ctx, "edit", path="dead.py", old="def g():", new="def g():  # renamed")
    assert out.ok
    assert "UNREACHABLE" not in out.value, "it was already there"


def test_a_clean_edit_says_nothing(ctx):
    out = call(ctx, "edit", path="src.py", old="    return a + 1", new="    return a + 2")
    assert out.ok and "UNREACHABLE" not in out.value


def test_the_case_it_was_built_for(ctx, repo):
    """A dogfood candidate passed its acceptance and left a duplicated return
    below the first. Tester was right and the diff still was not promotable."""
    (repo / "d.py").write_text(
        "def f(x):\n    out = {'a': x}\n    return out\n", encoding="utf-8")
    out = call(ctx, "edit", path="d.py", old="    return out",
               new="    return out\n    return out")
    assert "ERROR_UNREACHABLE_CODE" in out.value


def test_the_navigation_note_stays_out_of_a_verification_gate(repo):
    """The note is a RETRIEVAL hint and it was being injected into a
    VERIFICATION gate.

    Measured: the full battery scored 11/11 on the pre-round-2 code and 9/11
    with the note unscoped, and repeating the two tickets that moved gave 10/10
    against 4/8 (Fisher p = 0.008). An agent trying to make a failing test pass,
    told at the moment it tries to stop that search_code returned candidates it
    never opened, goes and opens them -- right on a mission whose difficulty is
    finding something, a detour on one that already knows which test is red.
    """
    ctx = tools.ToolContext(root=repo, write_scope=("**/*.py",),
                            acceptance_tests=("tests/test_x.py",))
    _result, shown = drive(ctx, [
        tc("search_code", query="add one to a number"),
        tc("finish", summary="no puedo", status="BLOCKED"),
        tc("finish", summary="no puedo", status="BLOCKED"),
    ])
    joined = "\n".join(shown)
    assert "antes de darte por vencido" in joined, "the gate itself still fires"
    assert "NO has abierto" not in joined, "but not with retrieval advice"


def test_it_still_fires_when_there_is_no_suite_to_point_at(repo):
    ctx = tools.ToolContext(root=repo, write_scope=("**/*.py",), acceptance_tests=())
    _result, shown = drive(ctx, [
        tc("search_code", query="add one to a number"),
        tc("finish", summary="no puedo", status="BLOCKED"),
        tc("finish", summary="no puedo", status="BLOCKED"),
    ])
    assert "NO has abierto" in "\n".join(shown)
