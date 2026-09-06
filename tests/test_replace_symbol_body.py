"""V2: generate a body, not a file.

smokeV1 left syntax at 20 of 30 refusals, and 30 of p5r3's 35 syntax refusals
were genuinely invalid code -- one run wrote ten invalid whole-module payloads
for a single file. Asking a 3B engine for a whole module is asking it to get
thousands of characters right at once. This asks for tens.

The gate this had to pass, and what each test here is for:

  * safety intact, nothing written outside scope;
  * only an EXISTING, unambiguously located symbol;
  * fail closed on ambiguity or absence;
  * the whole file must parse before anything is written;
  * the primitive must NOT repair content semantically.

That last one is the line. Re-indenting a body to sit under its declaration is
PLACEMENT and it is deterministic -- five of p5r3's syntax refusals were
fragments that parsed alone and broke the file when spliced in at the wrong
indentation. Changing what the code says would be something else entirely, and
these tests pin that it does not.
"""

from __future__ import annotations

import pytest

from localprog import tools

MODULE = '''"""A module."""
import os


class Box:
    def size(self):
        return 1

    def name(self):
        return "box"


def helper(a, b):
    total = a + b
    return total


@property
def decorated(self):
    return 2
'''


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "m.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / "locked.py").write_text(MODULE, encoding="utf-8")
    box = tools.ToolContext(root=tmp_path, write_scope=("m.py",),
                            allowed_new_files=())
    return box


def call(ctx, **kw):
    return tools.dispatch(ctx, "replace_symbol_body", kw)


# ------------------------------------------------------------ the normal path

def test_a_body_replaces_only_the_body(ctx, tmp_path):
    out = call(ctx, path="m.py", name="helper", body="return a * b")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "def helper(a, b):" in body, "the declaration must survive"
    assert "return a * b" in body
    assert "total = a + b" not in body


def test_the_indentation_is_placed_by_the_harness(ctx, tmp_path):
    """The 'valid fragment, wrong location' class. The model sends the body at
    whatever indentation it likes and the harness puts it where it belongs."""
    out = call(ctx, path="m.py", name="helper", body="x = 1\nreturn x")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "\n    x = 1\n    return x\n" in body


def test_an_over_indented_body_is_normalised_not_doubled(ctx, tmp_path):
    out = call(ctx, path="m.py", name="helper", body="        x = 1\n        return x")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "\n    x = 1\n    return x\n" in body


def test_a_method_keeps_its_class_indentation(ctx, tmp_path):
    out = call(ctx, path="m.py", name="size", body="return 99")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "\n        return 99\n" in body
    assert "def name(self):" in body, "the sibling method must be untouched"


def test_a_decorated_symbol_keeps_its_decorator(ctx, tmp_path):
    out = call(ctx, path="m.py", name="decorated", body="return 3")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "@property" in body


def test_a_multiline_body_survives_whole(ctx, tmp_path):
    out = call(ctx, path="m.py", name="helper",
               body="if a > b:\n    return a\nreturn b")
    assert out.ok, out.feedback
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "\n    if a > b:\n        return a\n    return b\n" in body


# ------------------------------------------------------------- it fails closed

def test_a_symbol_that_does_not_exist_is_refused(ctx):
    out = call(ctx, path="m.py", name="nope", body="return 1")
    assert not out.ok and out.code == "ERROR_NO_MATCH"


def test_an_ambiguous_symbol_is_refused_rather_than_guessed(tmp_path):
    source = ("class A:\n    def run(self):\n        return 1\n\n\n"
              "class B:\n    def run(self):\n        return 2\n")
    (tmp_path / "two.py").write_text(source, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**",))
    out = tools.dispatch(ctx, "replace_symbol_body",
                         {"path": "two.py", "name": "run", "body": "return 3"})
    assert not out.ok and out.code == "ERROR_NO_MATCH"
    assert "ambiguo" in out.feedback
    assert (tmp_path / "two.py").read_text(encoding="utf-8") == source


def test_a_body_that_would_not_parse_writes_nothing(ctx, tmp_path):
    before = (tmp_path / "m.py").read_text(encoding="utf-8")
    out = call(ctx, path="m.py", name="helper", body="return (")
    assert not out.ok and out.code == "ERROR_SYNTAX_AFTER_EDIT"
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == before, \
        "a refused edit must leave the file exactly as it was"


def test_an_empty_body_is_refused(ctx):
    out = call(ctx, path="m.py", name="helper", body="   \n  ")
    assert not out.ok, "emptying a symbol's body is not a body replacement"


def test_a_one_line_definition_is_declined_not_guessed(tmp_path):
    """`def f(): return 1` has no span that is only the body. Declining beats
    inventing a range."""
    (tmp_path / "one.py").write_text("def f(): return 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**",))
    out = tools.dispatch(ctx, "replace_symbol_body",
                         {"path": "one.py", "name": "f", "body": "return 2"})
    assert not out.ok
    assert "misma linea" in out.feedback


def test_an_identical_body_is_not_a_change(ctx):
    out = call(ctx, path="m.py", name="helper", body="total = a + b\nreturn total")
    assert not out.ok and out.code == "ERROR_NOTHING_CHANGED"


# ------------------------------------------------------------------- safety

def test_it_respects_the_write_scope(ctx, tmp_path):
    before = (tmp_path / "locked.py").read_text(encoding="utf-8")
    out = call(ctx, path="locked.py", name="helper", body="return 1")
    assert not out.ok and out.code == "ERROR_NOT_IN_WRITE_SCOPE"
    assert (tmp_path / "locked.py").read_text(encoding="utf-8") == before


def test_it_cannot_reach_outside_the_repository(ctx):
    out = call(ctx, path="../escape.py", name="helper", body="return 1")
    assert not out.ok
    assert out.code != "ERROR_NO_MATCH", "containment must refuse before resolution"


def test_it_does_not_repair_the_body_semantically(ctx, tmp_path):
    """The line this must not cross. The harness places the body; it never
    changes what the body says. A body that is wrong but parses is written
    exactly as sent, and the tests are what judge it."""
    call(ctx, path="m.py", name="helper", body="return None")
    body = (tmp_path / "m.py").read_text(encoding="utf-8")
    assert "return None" in body, "written verbatim, right or wrong"


def test_a_language_without_a_symbol_index_declines(tmp_path):
    (tmp_path / "x.ts").write_text("export function f() { return 1; }\n",
                                   encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**",))
    out = tools.dispatch(ctx, "replace_symbol_body",
                         {"path": "x.ts", "name": "f", "body": "return 2;"})
    assert not out.ok
    assert out.code in ("ERROR_LANGUAGE_UNSUPPORTED", "ERROR_NO_MATCH")
