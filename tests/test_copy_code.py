"""The harness moves the bytes, so the engine does not have to retype them.

Two measurements motivate this tool, and neither is a preference.

read_symbol's own docstring records the first: the reference engine met the same
file both ways and reproduced the function byte for byte 72% of the time as
plain text against 30% reading it through the tools and writing it back. Six of
the differences were nothing but indentation.

The payload probe recorded the second, and it is harder. granite4.1:3b loses a
payload above 1600 characters through the text protocol and above 800 natively,
while the median function this harness asks about is longer than the first. Some
answers it knows, it cannot emit. Copying is the only way those runs can succeed
at all.

The risk a tool like this carries is that it could move text nobody ever saw, so
what it returns must let a reviewer check the bytes came off disk.
"""

from __future__ import annotations

import hashlib

import pytest

from localprog import tools
from localprog.errors import (
    ERROR_BAD_ARGUMENTS,
    ERROR_FILE_EXISTS,
    ERROR_NOT_IN_WRITE_SCOPE,
    ERROR_NO_MATCH,
    InvalidCall,
    ToolError,
)

SOURCE = '''"""Un modulo."""


def sencilla(x):
    return x + 1


@decorada(1, 2)
def con_decorador(a, b):
    """Con comillas "dobles" y sangrado raro."""
    if a:
            return b
    return a


class Caja:
    def metodo(self):
        return "hola"
'''


def ctx_for(tmp_path, scope=(), new=("salida.py",)):
    (tmp_path / "origen.py").write_text(SOURCE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=scope, allowed_new_files=new)


def test_a_symbol_arrives_byte_for_byte(tmp_path):
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    got = (tmp_path / "salida.py").read_text(encoding="utf-8")
    assert got == ('@decorada(1, 2)\n'
                   'def con_decorador(a, b):\n'
                   '    """Con comillas "dobles" y sangrado raro."""\n'
                   '    if a:\n'
                   '            return b\n'
                   '    return a')
    assert out["symbol"] == "con_decorador"
    assert out["bytes"] == len(got.encode("utf-8"))


def test_the_decorator_comes_with_it(tmp_path):
    """A decorator is part of what the symbol IS -- the same rule read_symbol
    follows, and it follows it because both now ask the same function."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    assert (tmp_path / "salida.py").read_text(encoding="utf-8").startswith("@decorada")


def test_odd_indentation_and_quotes_survive(tmp_path):
    """The exact two things that were lost when the model retyped: indentation
    and quoting."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    got = (tmp_path / "salida.py").read_text(encoding="utf-8")
    assert "            return b" in got
    assert '"""Con comillas "dobles" y sangrado raro."""' in got


def test_a_line_range_works_where_there_is_no_symbol_index(tmp_path):
    """The common path: no parser needed, so it works in every language."""
    (tmp_path / "origen.rs").write_text("uno\ndos\ntres\ncuatro\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=("salida.txt",))
    out = tools.copy_code(ctx, "origen.rs", "salida.txt", start=2, end=3)
    assert (tmp_path / "salida.txt").read_text(encoding="utf-8") == "dos\ntres"
    assert out["lines"] == [2, 3]


def test_a_bare_method_name_resolves_the_way_read_symbol_resolves_it(tmp_path):
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "origen.py", "salida.py", name="metodo")
    assert out["symbol"] == "Caja.metodo"


# --------------------------------------------------------------- auditability

def test_it_reports_enough_to_verify_the_bytes_came_off_disk(tmp_path):
    """A tool that moves text nobody saw would otherwise be the one place a run
    could not be checked."""
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    body = (tmp_path / "salida.py").read_text(encoding="utf-8")
    assert out["sha256_16"] == hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    assert out["copied_from"] == "origen.py"
    assert out["first_line"] == "def sencilla(x):"
    assert out["last_line"] == "    return x + 1"


def test_the_body_does_not_come_back_through_the_transcript(tmp_path):
    """Returning it would put back exactly the weight this tool exists to keep
    out of the transcript."""
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    assert "return b" not in "".join(str(v) for v in out.values())


# ------------------------------------------------------------------- refusals

def test_neither_selector_lists_what_there_is_to_copy(tmp_path):
    """Observed on granite4.1:3b: it reached for copy_code correctly on turn one
    with only the selector missing, was told to go and use list_symbols, did,
    and never came back. The harness had already resolved the path and read the
    file -- it could answer the question the failed call was asking, so it
    does. A refusal whose only content is what you did wrong has thrown away
    the work it just did."""
    ctx = ctx_for(tmp_path)
    with pytest.raises(InvalidCall) as exc:
        tools.copy_code(ctx, "origen.py", "salida.py")
    assert exc.value.code == ERROR_BAD_ARGUMENTS
    detail = exc.value.detail
    assert "sencilla" in detail and "con_decorador" in detail
    assert "copy_code(src='origen.py'" in detail
    # And it must NOT read as a recommendation. The first version showed a
    # worked example using names[0], and granite4.1:3b took the first symbol
    # offered in 12 of 50 runs -- ALWAYS_NO_SPACE when it wanted visit -- and
    # finished DONE. An unranked list presented as a choice is worse than none.
    assert "EN ORDEN DE FICHERO" in detail
    assert "no elijas uno al azar" in detail
    assert "search_code(query=" in detail and "path='origen.py'" in detail
    for symbol in ("sencilla", "con_decorador", "Caja"):
        assert f"name='{symbol}'" not in detail, (
            f"the catalogue must not recommend {symbol}: it does not know which "
            f"one is wanted, and saying one is how the wrong one gets copied")


def test_the_catalogue_falls_back_to_line_counts_without_a_parser(tmp_path):
    """A language with no symbol index still gets something usable."""
    (tmp_path / "a.rs").write_text(chr(10).join(["uno","dos","tres"]) + chr(10), encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=("salida.txt",))
    with pytest.raises(InvalidCall) as exc:
        tools.copy_code(ctx, "a.rs", "salida.txt")
    assert "3 lineas" in exc.value.detail


def test_the_catalogue_cannot_depend_on_the_objective(tmp_path):
    """It lists the file's own declarations in name order. If it could be
    ordered by relevance it would be a retrieval channel hiding in an error
    message, and errors are not where retrieval belongs."""
    ctx = ctx_for(tmp_path)
    with pytest.raises(InvalidCall) as first:
        tools.copy_code(ctx, "origen.py", "salida.py")
    with pytest.raises(InvalidCall) as second:
        tools.copy_code(ctx, "origen.py", "salida.py")
    assert first.value.detail == second.value.detail


def test_both_selectors_at_once_is_refused(tmp_path):
    ctx = ctx_for(tmp_path)
    with pytest.raises(InvalidCall):
        tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla", start=1)


def test_an_unknown_symbol_lists_what_there_is(tmp_path):
    ctx = ctx_for(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "origen.py", "salida.py", name="no_existe")
    assert exc.value.code == ERROR_NO_MATCH
    assert "sencilla" in exc.value.detail


def test_it_will_not_overwrite_a_file_that_was_already_there(tmp_path):
    ctx = ctx_for(tmp_path)
    (tmp_path / "salida.py").write_text("codigo real\n", encoding="utf-8")
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    assert exc.value.code == ERROR_FILE_EXISTS
    assert (tmp_path / "salida.py").read_text(encoding="utf-8") == "codigo real\n"


def test_the_destination_is_inside_the_write_scope_like_any_other_write(tmp_path):
    ctx = ctx_for(tmp_path, scope=(), new=("salida.py",))
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "origen.py", "otro.py", name="sencilla")
    assert exc.value.code == ERROR_NOT_IN_WRITE_SCOPE


def test_a_second_copy_appends_to_a_file_this_run_created(tmp_path):
    """What an extract-to-module refactor needs: several definitions moved into
    one new file."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    out = tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    assert out["appended"] is True
    got = (tmp_path / "salida.py").read_text(encoding="utf-8")
    assert "def sencilla" in got and "def con_decorador" in got


def test_a_line_number_past_the_end_says_how_long_the_file_is(tmp_path):
    ctx = ctx_for(tmp_path)
    with pytest.raises(InvalidCall) as exc:
        tools.copy_code(ctx, "origen.py", "salida.py", start=1, end=9999)
    assert "lineas" in exc.value.detail


def test_a_wrong_path_is_a_wrong_path_not_a_containment_violation(tmp_path):
    """copy_code used _resolve(must_exist=True), and the guard reports "does not
    exist" as a ContainmentError, which becomes ERROR_PATH_OUTSIDE_REPO. So
    granite4.1:3b was told six times that its path was outside the repository
    when the path was merely wrong -- false, alarming, and teaching it nothing.

    Same defect class as A3: a safety error standing in for an ordinary mistake.
    """
    from localprog.errors import ERROR_FILE_NOT_FOUND, ERROR_PATH_OUTSIDE_REPO

    ctx = ctx_for(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "no/existe.py", "salida.py", name="sencilla")
    assert exc.value.code == ERROR_FILE_NOT_FOUND

    # And a REAL containment violation is still exactly that.
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "../fuera.py", "salida.py", name="sencilla")
    assert exc.value.code == ERROR_PATH_OUTSIDE_REPO


def test_the_not_found_message_lists_what_is_actually_there(tmp_path):
    """The same help read_file gives, for the same reason: the next call should
    be able to be right."""
    ctx = ctx_for(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "origen_mal.py", "salida.py", name="sencilla")
    assert "origen.py" in exc.value.detail


def test_an_append_says_it_appended(tmp_path):
    """copy_code appends into a file this run created, which is what an
    extract-to-module refactor needs and what silently concatenates a
    single-answer mission. The reference engine's v7 runs:

        reached the symbol and was RIGHT   30 runs, 2.0 copies, 396 chars
        reached the symbol and was WRONG    8 runs, 4.5 copies, 963 chars

    The failing ones copied the right symbol AND several others. Conversion fell
    92% -> 77% across the versions where turn zero began naming a candidate to
    copy: two changes of mine, each defensible, combining into a defect neither
    had alone.

    A note and not a refusal -- appending is sometimes exactly the intent. What
    was missing is that the agent could not SEE it, because copy_code
    deliberately does not return the body.
    """
    ctx = ctx_for(tmp_path)
    first = tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    assert first["appended"] is False
    assert "ANADIDO" not in first["note"]

    second = tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    assert second["appended"] is True
    assert "ANADIDO" in second["note"]
    assert "2 copias" in second["note"]
    assert "write_file(path='salida.py'" in second["note"]


def test_the_append_count_is_one_number(tmp_path):
    """Counted in copy_code and in write_file alike, so "how many things are in
    this file" cannot be two numbers that disagree."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    tools.write_file(ctx, "salida.py", "reescrito\n")
    third = tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    assert "3 copias" in third["note"]


def test_appending_still_works(tmp_path):
    """The refactor case is why the tool appends at all. Warning about it must
    not break it."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "origen.py", "salida.py", name="sencilla")
    tools.copy_code(ctx, "origen.py", "salida.py", name="con_decorador")
    got = (tmp_path / "salida.py").read_text(encoding="utf-8")
    assert "def sencilla" in got and "def con_decorador" in got
