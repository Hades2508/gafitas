"""F-93: name= and start=/end= together, when they say the same thing.

``read_symbol`` prints "lineas 90-94" and then the model calls ``copy_code``
with the name it asked for AND the numbers it was just shown. That used to be
ERROR_BAD_ARGUMENTS, and the retry was always the identical call with the name
deleted -- one wasted turn, 18 times across four engines in the expansion
cohort, ten of them ministral-3:3b.

The rule these tests fix in place: agreement is completed, disagreement is
still refused. The harness may finish a fact it knows for certain; it may not
decide a question the model has actually left open.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import InvalidCall

SOURCE = '''"""Module."""


def alpha():
    return 1


class Box:
    def beta(self):
        return 2

    def gamma(self):
        return 3
'''


@pytest.fixture()
def ctx(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("OUT.txt",),
                             allowed_new_files=("OUT.txt",))


def span_of(name):
    """The span read_symbol would print -- which is what the model echoes back.

    Deliberately the harness's own resolver rather than a reimplementation: the
    contract under test is "the numbers the agent was just shown are accepted",
    and those numbers come from here.
    """
    _n, _node, _found, lo, hi = tools._symbol_span("mod.py", SOURCE, name)
    return lo, hi


def test_agreeing_name_and_lines_are_accepted(ctx):
    lo, hi = span_of("alpha")
    out = tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="alpha",
                          start=lo, end=hi)
    assert out["symbol"] == "alpha"
    assert out["lines"] == [lo, hi]
    assert (ctx.root / "OUT.txt").read_text(encoding="utf-8").startswith("def alpha():")


def test_the_symbol_wins_so_the_copy_is_exact(ctx):
    """Agreement means the span is the symbol's own, not the model's arithmetic."""
    lo, hi = span_of("alpha")
    out = tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="alpha",
                          start=lo, end=hi)
    body = (ctx.root / "OUT.txt").read_text(encoding="utf-8")
    assert "class Box" not in body and "beta" not in body


def test_disagreeing_name_and_lines_are_still_refused(ctx):
    """A class name with the whole file's range is a real ambiguity."""
    with pytest.raises(InvalidCall) as exc:
        tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="Box",
                        start=1, end=len(SOURCE.splitlines()))
    assert not (ctx.root / "OUT.txt").exists()
    assert "no dicen lo mismo" in str(exc.value)


def test_the_refusal_quotes_the_span_the_name_really_has(ctx):
    """So the next call can be right without another read."""
    lo, hi = span_of("Box")
    with pytest.raises(InvalidCall) as exc:
        tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="Box",
                        start=1, end=2)
    assert f"{lo}-{hi}" in str(exc.value)


def test_an_unresolvable_name_with_lines_is_refused_not_silently_dropped(ctx):
    """Falling back to the line range would copy something the model did not
    ask for, under a name the file does not contain."""
    with pytest.raises(InvalidCall):
        tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="nope",
                        start=1, end=2)
    assert not (ctx.root / "OUT.txt").exists()


def test_only_one_of_start_and_end_is_not_agreement(ctx):
    """A half-specified range cannot be checked against the span, so it cannot
    be called agreement."""
    lo, _hi = span_of("alpha")
    with pytest.raises(InvalidCall):
        tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="alpha", start=lo)


def test_name_alone_and_lines_alone_still_work(ctx):
    lo, hi = span_of("alpha")
    tools.copy_code(ctx, src="mod.py", into="OUT.txt", name="alpha")
    first = (ctx.root / "OUT.txt").read_text(encoding="utf-8")
    (ctx.root / "OUT.txt").unlink()
    ctx.created_files.discard("OUT.txt")
    ctx.changed_files.discard("OUT.txt")
    tools.copy_code(ctx, src="mod.py", into="OUT.txt", start=lo, end=hi)
    assert (ctx.root / "OUT.txt").read_text(encoding="utf-8") == first
