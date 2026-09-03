"""A rewrite with nothing learned in between is a guess, and gets said so.

F-75 was a real trap -- a file the mission told you to create could not be
corrected -- and opening that door had a cost visible in the very next run. Of
the reference engine's first eleven post-fix runs, two wrote their answer SIX
and NINE times, at 17 and 20 turns. The rest wrote it once or twice.

What separates them is not the count. It is whether anything was LEARNED in
between: a rewrite after a read, a search or a run is acting on something new;
a rewrite with nothing in between is a different guess at the same question.
The loop can tell those apart exactly, with no judgement -- it knows which tools
return information and it knows what was called.

A note and never a refusal. F-63 is the standing lesson: a refusal that blocks a
correct answer costs more than the mistake it prevents, and the second write is
very often the right one.
"""

from __future__ import annotations

from localprog import tools


def ctx_for(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=("a.txt",))


def test_the_first_writes_are_silent(tmp_path):
    """Self-correction is normal and must not be nagged at."""
    ctx = ctx_for(tmp_path)
    assert "Has escrito" not in tools.write_file(ctx, "a.txt", "uno")
    assert "Has escrito" not in tools.write_file(ctx, "a.txt", "dos")


def test_a_third_blind_rewrite_is_noted(tmp_path):
    ctx = ctx_for(tmp_path)
    for body in ("uno", "dos"):
        tools.write_file(ctx, "a.txt", body)
    note = tools.write_file(ctx, "a.txt", "tres")
    assert "Has escrito" in note and "3 veces" in note


def test_looking_at_something_resets_it(tmp_path):
    """The whole point: the count is not the signal, the absence of new
    information is."""
    ctx = ctx_for(tmp_path)
    for body in ("uno", "dos"):
        tools.write_file(ctx, "a.txt", body)
    tools.dispatch(ctx, "read_file", {"path": "m.py"})
    assert "Has escrito" not in tools.write_file(ctx, "a.txt", "tres")


def test_every_information_tool_resets_it(tmp_path):
    cases = (("read_file", {"path": "m.py"}),
             ("list_symbols", {"path": "m.py"}),
             ("read_symbol", {"path": "m.py", "name": "f"}),
             ("list_dir", {"path": "."}),
             ("grep", {"pattern": "def"}),
             # A query that actually matches: a search that finds nothing is a
             # failed lookup, and the test below is the one that covers those.
             ("search_code", {"query": "def f return"}))
    for i, (name, args) in enumerate(cases):
        # A fresh repository each time. The ToolContext is new but the
        # directory would not be, and a leftover a.txt turns the second write
        # into a different test.
        box = tmp_path / f"caso{i}"
        box.mkdir()
        ctx = ctx_for(box)
        for body in ("uno", "dos"):
            tools.write_file(ctx, "a.txt", body)
        tools.dispatch(ctx, name, args)
        assert "Has escrito" not in tools.write_file(ctx, "a.txt", "tres"), name


def test_a_failed_lookup_does_not_count_as_learning(tmp_path):
    """An error is feedback about the call, not information about the code."""
    ctx = ctx_for(tmp_path)
    for body in ("uno", "dos"):
        tools.write_file(ctx, "a.txt", body)
    tools.dispatch(ctx, "read_file", {"path": "no_existe.py"})
    assert "Has escrito" in tools.write_file(ctx, "a.txt", "tres")


def test_it_never_refuses(tmp_path):
    """The write happens. F-63: a refusal that blocks a correct answer costs
    more than the mistake it prevents."""
    ctx = ctx_for(tmp_path)
    for body in ("uno", "dos", "tres", "cuatro"):
        tools.write_file(ctx, "a.txt", body)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "cuatro"


def test_it_counts_per_file(tmp_path):
    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=("a.txt", "b.txt"))
    for body in ("uno", "dos"):
        tools.write_file(ctx, "a.txt", body)
    assert "Has escrito" not in tools.write_file(ctx, "b.txt", "uno")
