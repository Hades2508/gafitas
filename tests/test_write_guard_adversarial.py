"""The containment promise, written down as tests instead of as a sentence.

The harness makes one claim it is sold on:

    every write is checked against the mission's scope before it lands, and
    across the recorded runs of this campaign there were zero unauthorised
    writes.

That claim was resting on the guard having held, not on anything asserting it
would. This file is the assertion. Every case here was run adversarially first
and every one was refused; they are kept so a refactor cannot quietly open one.

WHAT THIS DOES NOT COVER, AND MUST NOT BE READ AS COVERING
----------------------------------------------------------
`run` executes an interpreter, and a child process writes with the user's
permissions. Nothing in this file constrains that, and no test here should ever
be cited as though it did. That boundary is declared and accepted, it is
watched after the fact by `sentinel`, and `sentinel`'s own report names what it
cannot see. A guard that is described as stronger than it is, is worse than a
weaker one described accurately.

TWO LAYERS, DELIBERATELY SEPARATE
---------------------------------
CONTAINMENT lives in `programmer.guard`, a different authority, verified at
import by `deps`. It answers "is this path inside the repository at all".
SCOPE lives in `localprog.scope` and answers the narrower question "did the
mission authorise writing here". The error codes below distinguish which layer
refused, and that distinction is part of what is being pinned: a case that
starts being refused by the other layer is a change worth noticing.
"""

from __future__ import annotations

import subprocess

import pytest

from localprog import tools
from localprog.errors import (ERROR_NOT_IN_WRITE_SCOPE, ERROR_PATH_OUTSIDE_REPO,
                              ToolError)

SOURCE = "def f():\n    return 1\n"
FORBIDDEN = "def g():\n    return 2\n"


def build(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "secreto").mkdir()
    (tmp_path / "secreto" / "prohibido.py").write_text(FORBIDDEN, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("src/",),
                             allowed_new_files=("src/nuevo.py",))


# --------------------------------------------------------------------------
# The path forms. Each is a way of writing a destination that is not the one
# it appears to be, and Windows supplies most of them.
# --------------------------------------------------------------------------

ESCAPES = [
    ("parent traversal", "../fuera.txt"),
    ("traversal through a real directory", "src/../../fuera.txt"),
    ("windows drive-relative", "C:fuera.txt"),
    ("UNC share", "//servidor/share/x.txt"),
    ("backslash separators", "..\\fuera.txt"),
    # Windows strips a trailing dot when opening, so "a.py." and "a.py" are the
    # same file on disk and a check that compares strings can disagree with the
    # filesystem about which one was named.
    ("trailing dot", "src/a.py."),
    ("reserved device name", "src/CON"),
    ("NTFS alternate data stream", "src/a.py:oculto"),
    ("embedded NUL", "src/a\x00.py"),
]


@pytest.mark.parametrize("label,path", ESCAPES, ids=[e[0] for e in ESCAPES])
def test_a_path_that_leaves_the_repository_is_refused(tmp_path, label, path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.write_file(ctx, path, "INTRUSO\n")
    assert caught.value.code == ERROR_PATH_OUTSIDE_REPO
    assert not (tmp_path.parent / "fuera.txt").exists()


OUT_OF_SCOPE = [
    ("different case", "SRC/A.PY"),
    ("the acceptance test itself", "tests/test_a.py"),
    ("inside .git", ".git/config"),
    ("a sibling directory", "secreto/prohibido.py"),
]


@pytest.mark.parametrize("label,path", OUT_OF_SCOPE,
                         ids=[e[0] for e in OUT_OF_SCOPE])
def test_a_path_the_mission_did_not_authorise_is_refused(tmp_path, label, path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.write_file(ctx, path, "INTRUSO\n")
    assert caught.value.code == ERROR_NOT_IN_WRITE_SCOPE


#: These belong to a FILE scope, not a directory one. Written first against
#: `src/` and they failed, correctly: `src/` authorises `src/a.py.bak` and
#: there is nothing wrong with that. The question actually worth asking is
#: whether naming ONE file quietly authorises its neighbours.
NEIGHBOURS = [
    ("suffix glued on", "src/a.py.bak"),
    ("neighbouring extension", "src/a.pyx"),
    ("same stem, different suffix", "src/a.pyc"),
    ("prefix match", "src/a.python"),
]


@pytest.mark.parametrize("label,path", NEIGHBOURS, ids=[e[0] for e in NEIGHBOURS])
def test_naming_one_file_does_not_authorise_its_neighbours(tmp_path, label, path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(SOURCE, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("src/a.py",),
                            allowed_new_files=())

    with pytest.raises(ToolError) as caught:
        tools.write_file(ctx, path, "INTRUSO")
    assert caught.value.code == ERROR_NOT_IN_WRITE_SCOPE
    assert not (tmp_path / path).exists()


def test_scope_matching_is_case_sensitive_on_purpose(tmp_path):
    """Windows is case-insensitive on disk; the scope is not, and must not be.

    A scope that matched SRC/A.PY for src/a.py would behave differently on two
    machines, and a containment rule that depends on the platform is not one.
    It fails closed, which is the right direction to fail in.
    """
    ctx = build(tmp_path)
    with pytest.raises(ToolError):
        tools.write_file(ctx, "SRC/NUEVO.PY", "x = 1\n")


# --------------------------------------------------------------------------
# The filesystem-level escape. Not a string trick: a real redirection that a
# purely textual check cannot see.
# --------------------------------------------------------------------------

def test_a_junction_inside_the_repo_does_not_become_a_way_out(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_afuera"
    outside.mkdir()
    (outside / "objetivo.txt").write_text("ORIGINAL\n", encoding="utf-8")

    ctx = build(tmp_path)
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(tmp_path / "src" / "puerta"),
         str(outside)],
        capture_output=True, text=True, shell=False)
    if not (tmp_path / "src" / "puerta").exists():
        pytest.skip(f"could not create a junction here: {made.stderr.strip()}")

    # Inside `src/`, which the mission DOES authorise -- so only resolving the
    # junction can catch this. A scope check on the string alone would pass it.
    with pytest.raises(ToolError) as caught:
        tools.write_file(ctx, "src/puerta/nuevo.txt", "FUGA\n")
    assert caught.value.code == ERROR_PATH_OUTSIDE_REPO
    assert not (outside / "nuevo.txt").exists()
    assert (outside / "objetivo.txt").read_text(encoding="utf-8") == "ORIGINAL\n"


# --------------------------------------------------------------------------
# Every write tool, not just the obvious one.
# --------------------------------------------------------------------------

TARGET = "secreto/prohibido.py"

WRITE_CALLS = {
    "edit": lambda c: tools.edit(c, TARGET, "return 2", "return 999"),
    "replace_lines": lambda c: tools.replace_lines(c, TARGET, 2, 2,
                                                   "    return 999"),
    "write_file": lambda c: tools.write_file(c, "secreto/otro.py", "x = 1\n"),
    "replace_file": lambda c: tools.replace_file(c, TARGET, "x = 1\n",
                                                 expected_sha256=None),
    "replace_symbol_body": lambda c: tools.replace_symbol_body(
        c, TARGET, "g", "    return 999"),
    # A NEW destination on purpose: copy_code first refused this case with
    # ERROR_FILE_EXISTS, which is an incidental refusal and would have hidden a
    # real gap if there had been one.
    "copy_code": lambda c: tools.copy_code(c, "src/a.py", "secreto/nuevo.py",
                                           "f"),
}


@pytest.mark.parametrize("name", sorted(WRITE_CALLS))
def test_every_write_tool_refuses_an_unauthorised_path(tmp_path, name):
    ctx = build(tmp_path)
    with pytest.raises(ToolError):
        WRITE_CALLS[name](ctx)
    assert (tmp_path / "secreto" / "prohibido.py").read_text(
        encoding="utf-8") == FORBIDDEN
    assert not (tmp_path / "secreto" / "otro.py").exists()
    assert not (tmp_path / "secreto" / "nuevo.py").exists()


def test_the_authorised_write_still_works(tmp_path):
    """A guard that refuses everything is not a guard, it is a brick."""
    ctx = build(tmp_path)
    tools.write_file(ctx, "src/nuevo.py", "x = 1\n")
    assert (tmp_path / "src" / "nuevo.py").read_text(encoding="utf-8") == "x = 1\n"

    tools.edit(ctx, "src/a.py", "return 1", "return 42")
    assert "return 42" in (tmp_path / "src" / "a.py").read_text(encoding="utf-8")
