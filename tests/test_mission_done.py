"""The harness says when the thing the mission asked for exists.

The trace that makes this unarguable, reference engine on the frozen sweep:

    t1  search_code   ok
    t2  read_symbol   ok      the right function
    t3  copy_code     ok      the answer is written, and correct
    t4  ERROR_NO_TOOL_CALL
    t5  write_file    ok      overwrites it with the same thing
    t6..t20                   the same two turns, eight more times

Across the fifty runs: 47 produced an answer, 5 called finish, and 325 turns --
six and a half per run -- went to ERROR_NO_TOOL_CALL after the answer was
already on disk.

F-71 fixed a refusal that offered only exits. This is the same correction for
the success case: a success that offered nothing at all. The harness knew at
turn three that the mission's expected file existed and had content in it, and
had simply never been asked to say so.
"""

from __future__ import annotations

from localprog import tools

MODULE = '''def pequena(x):
    return x + 1


class Caja:
    def uno(self):
        return 1

    def dos(self):
        return 2
'''


def ctx_for(tmp_path, new=("r.txt",)):
    (tmp_path / "m.py").write_text(MODULE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=new)


def test_a_write_that_completes_the_mission_says_so(tmp_path):
    ctx = ctx_for(tmp_path)
    note = tools.write_file(ctx, "r.txt", "la respuesta\n")
    assert "Ya existe lo que pedia la mision" in note
    assert "finish(status='DONE'" in note


def test_a_copy_that_completes_the_mission_says_so(tmp_path):
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "m.py", "r.txt", name="pequena")
    assert "finish(status='DONE'" in out["note"]


def test_an_empty_file_is_not_a_completed_mission(tmp_path):
    """Creating the file is not producing the answer, and saying otherwise
    would invite exactly the empty-answer finish this harness spent a campaign
    learning to refuse."""
    ctx = ctx_for(tmp_path)
    assert "Ya existe" not in tools.write_file(ctx, "r.txt", "")


def test_a_mission_with_several_expected_files_waits_for_all_of_them(tmp_path):
    ctx = ctx_for(tmp_path, new=("a.txt", "b.txt"))
    assert "Ya existe" not in tools.write_file(ctx, "a.txt", "uno\n")
    note = tools.write_file(ctx, "b.txt", "dos\n")
    assert "Ya existe" in note and "a.txt" in note and "b.txt" in note


def test_a_mission_that_expects_no_new_file_says_nothing(tmp_path):
    """An ordinary edit ticket has no such completion condition, and inventing
    one would be the harness deciding a mission is over."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("src/",))
    assert "Ya existe" not in tools.edit(ctx, "src/a.py", "x = 1", "x = 2")


def test_it_does_not_end_the_run(tmp_path):
    """The harness knows the file exists; it does not know the contents are
    right, and finish() carries a status the verdict depends on. Deciding the
    mission is over stays the agent's to do."""
    ctx = ctx_for(tmp_path)
    tools.write_file(ctx, "r.txt", "primera\n")
    tools.write_file(ctx, "r.txt", "segunda\n")
    assert (tmp_path / "r.txt").read_text(encoding="utf-8") == "segunda\n"
    assert ctx.finish_status is None


# --------------------------------------------------- copying more than meant

def test_copying_a_class_says_it_was_a_class(tmp_path):
    """copy_code inherited read_symbol's behaviour on a class and not its F-70
    warning -- and it needs the warning more, because it deliberately does not
    return the body, so an over-copy leaves nothing the agent can see.

    Measured: granite4.1:3b's answers on frozen HEAD were median 4982 chars,
    max 34408, and its CORRECT answers were median 442."""
    ctx = ctx_for(tmp_path)
    out = tools.copy_code(ctx, "m.py", "r.txt", name="Caja")
    assert "es una CLASE entera" in out["note"]
    assert "uno, dos" in out["note"]
    assert "name='Caja.uno'" in out["note"]


def test_copying_one_function_says_nothing_about_classes(tmp_path):
    ctx = ctx_for(tmp_path)
    assert "CLASE" not in tools.copy_code(ctx, "m.py", "r.txt", name="pequena")["note"]


def test_copying_most_of_a_file_by_lines_says_so(tmp_path):
    ctx = ctx_for(tmp_path)
    big = "\n".join(f"linea {i}" for i in range(100)) + "\n"
    (tmp_path / "grande.txt").write_text(big, encoding="utf-8")
    out = tools.copy_code(ctx, "grande.txt", "r.txt", start=1, end=100)
    assert "casi el fichero entero" in out["note"]


def test_a_small_line_range_says_nothing(tmp_path):
    ctx = ctx_for(tmp_path)
    big = "\n".join(f"linea {i}" for i in range(100)) + "\n"
    (tmp_path / "grande.txt").write_text(big, encoding="utf-8")
    out = tools.copy_code(ctx, "grande.txt", "r.txt", start=10, end=20)
    assert "casi el fichero entero" not in out["note"]


def test_neither_note_is_a_refusal(tmp_path):
    """Copying a whole class is sometimes exactly right. F-63 is the standing
    lesson about refusing things that are sometimes right."""
    ctx = ctx_for(tmp_path)
    tools.copy_code(ctx, "m.py", "r.txt", name="Caja")
    assert "class Caja" in (tmp_path / "r.txt").read_text(encoding="utf-8")
