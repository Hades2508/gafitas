"""Answering "the issue is not in this code" with a fact about this code.

F-173. Six of eight empty SWE-bench runs ended on a variant of the same
sentence:

    "The issue mentions a function 'translate_url()' that is not found in the
     codebase."                                  -- django__django-11477
    "The issue described in the query is not actually present in the code."
                                                 -- django__django-13158

They were not lazy. Seven to thirteen turns each, up to seven reads and five
searches. They looked, did not find it, and concluded it was not there --
generalising "I have not found this" into "this does not exist".

The confirmation question already existed (F-61): NO_CHANGE with an empty diff
is asked once, and the evidence shows it firing in all six. The agent confirmed
every time, which is the part worth fixing. For an EDIT mission the question
said only

    "this mission authorised you to write in: django/"

which is a permission the agent already had and which says nothing about the
belief it is about to seal.

The harness could answer that belief precisely, and did not. `translate_url` is
in django/urls/base.py; the index knew it; the agent had never opened the file.

Deliberately narrow: tokens shaped like identifiers, an exact declaration in
the index, files the agent never opened, source files only, at most three. The
first version matched "arguments" out of the prose -- an ordinary English word
heading a region in docs/releases/2.1.txt -- and offered it beside
`translate_url` as if the two were the same kind of fact. One false row
devalues the true one next to it, which is the entire point of the note.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_NOTHING_CHANGED, ToolError

MODULE = '''def translate_url(url, lang_code):
    """Devuelve la URL equivalente en otro idioma."""
    return url
'''

ELSEWHERE = '''class RegexPattern:
    """Empareja una URL contra una expresion regular."""

    def match(self, path):
        return path
'''

PROSE = "arguments\n---------\n\nAlgo sobre argumentos y su uso.\n"

ISSUE = ("translate_url() creates an incorrect URL when optional named groups "
         "are missing in the URL pattern. The arguments () are dropped.")


def build(tmp_path):
    pkg = tmp_path / "django" / "urls"
    pkg.mkdir(parents=True)
    (pkg / "base.py").write_text(MODULE, encoding="utf-8")
    (pkg / "resolvers.py").write_text(ELSEWHERE, encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "releases.txt").write_text(PROSE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("django/",),
                             objective=ISSUE, turns_left=20, max_turns_hint=40)


def test_the_identifier_the_objective_names_is_found(tmp_path):
    ctx = build(tmp_path)
    ctx.opened.add("django/urls/resolvers.py")      # what the agent did read
    found = dict(tools._named_but_never_opened(ctx))
    assert found.get("translate_url") == "django/urls/base.py"


def test_a_file_the_agent_already_opened_is_not_offered(tmp_path):
    """Telling it to look where it has already looked is noise."""
    ctx = build(tmp_path)
    ctx.opened.add("django/urls/base.py")
    assert "translate_url" not in dict(tools._named_but_never_opened(ctx))


def test_prose_files_are_not_offered_as_places_code_lives(tmp_path):
    ctx = build(tmp_path)
    offered = dict(tools._named_but_never_opened(ctx))
    assert "arguments" not in offered, (
        "an ordinary English word heading a region in a .txt is not a lead, "
        "and one false row devalues the true one beside it")


def test_the_confirmation_carries_the_fact(tmp_path):
    ctx = build(tmp_path)
    ctx.opened.add("django/urls/resolvers.py")
    with pytest.raises(ToolError) as caught:
        tools.finish(ctx, summary="translate_url is not in the codebase.",
                     status="NO_CHANGE")
    assert caught.value.code == ERROR_NOTHING_CHANGED
    message = str(caught.value)
    assert "translate_url" in message
    assert "django/urls/base.py" in message


def test_the_second_no_change_is_still_accepted(tmp_path):
    """F-61's contract: asked once, then taken. This must not become a wall."""
    ctx = build(tmp_path)
    with pytest.raises(ToolError):
        tools.finish(ctx, summary="nada que hacer", status="NO_CHANGE")
    assert tools.finish(ctx, summary="nada que hacer", status="NO_CHANGE")
    assert ctx.finish_status == "NO_CHANGE"


def test_nothing_is_offered_when_the_objective_names_nothing_present(tmp_path):
    ctx = build(tmp_path)
    ctx.objective = "Something is wrong somewhere in the system."
    assert tools._named_but_never_opened(ctx) == []


def test_an_empty_objective_is_harmless(tmp_path):
    ctx = build(tmp_path)
    ctx.objective = ""
    assert tools._named_but_never_opened(ctx) == []
