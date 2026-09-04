"""Prose meaning "I'm done" is pointed at finish, not at write_file again.

Measured, reference engine, v4, 50 runs: 368 turns produced no tool call -- 7.4
per run -- alternating perfectly with redundant writes:

    t3  copy_code    ok        the answer is written, and correct
    t4  (no tool call)
    t5  copy_code    ok
    t6  (no tool call)
    t7,9,11,13,15    write_file, again, again, again
    t8,10,12,14,16   (no tool call)

finish succeeded in 2 runs of 50. The rest burned twenty turns with a correct
answer already on disk.

The cause was my own correction message: ERROR_NO_TOOL_CALL advised "if what you
wrote was the CONTENT of a file, pass it as write_file(content=...)". The engine
wrote prose meaning "that's it"; the harness told it to use write_file; it wrote
the file again. I built the loop in a message added to fix a different problem.
"""

from __future__ import annotations

from localprog import loop, tools


def ctx_for(tmp_path, new=("r.txt",)):
    return tools.ToolContext(root=tmp_path, allowed_new_files=new)


def test_prose_after_the_answer_exists_is_pointed_at_finish(tmp_path):
    ctx = ctx_for(tmp_path)
    (tmp_path / "r.txt").write_text("la respuesta\n", encoding="utf-8")
    note = loop._no_call_note(ctx)
    assert "finish(status='DONE')" in note
    assert "YA existe" in note


def test_prose_before_anything_exists_keeps_the_old_advice(tmp_path):
    """The content-as-prose case is real -- F-72 exists because of it -- and
    telling an engine to finish before it has produced anything would be the
    same defect with the sign reversed."""
    assert loop._no_call_note(ctx_for(tmp_path)) == ""


def test_an_empty_output_file_is_not_a_finished_mission(tmp_path):
    ctx = ctx_for(tmp_path)
    (tmp_path / "r.txt").write_text("", encoding="utf-8")
    assert loop._no_call_note(ctx) == ""


def test_a_mission_expecting_several_files_waits_for_all_of_them(tmp_path):
    ctx = ctx_for(tmp_path, new=("a.txt", "b.txt"))
    (tmp_path / "a.txt").write_text("uno\n", encoding="utf-8")
    assert loop._no_call_note(ctx) == ""
    (tmp_path / "b.txt").write_text("dos\n", encoding="utf-8")
    assert "finish" in loop._no_call_note(ctx)


def test_an_edit_mission_says_nothing(tmp_path):
    """An ordinary ticket has no expected-output condition, and inventing one
    would be the harness deciding a mission is over."""
    ctx = tools.ToolContext(root=tmp_path, write_scope=("src/",))
    assert loop._no_call_note(ctx) == ""


def test_it_only_fires_for_a_missing_tool_call(tmp_path):
    """It must not attach itself to every error. A bad argument is not the
    engine saying it has finished."""
    import inspect
    source = inspect.getsource(loop.run_loop)
    assert "if exc.code == ERROR_NO_TOOL_CALL:" in source
