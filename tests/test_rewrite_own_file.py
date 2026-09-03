"""A file you created in this run may be corrected. Until now it could not.

    write_file('answer.txt', ...)   ok
    write_file('answer.txt', ...)   ERROR_FILE_EXISTS: "ya existe. Usa edit."
    edit('answer.txt', ...)         ERROR_NOT_IN_WRITE_SCOPE: scope is (nada)

No third door: the harness named a tool the same harness forbade, and the
mission's own output file was unreachable after the first draft. granite4.1:3b
hit ERROR_FILE_EXISTS seven times in twelve runs. The reference engine rarely
hits it because it usually writes once and correctly -- which is how a defect
survives being measured only against the engine it does not happen to.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_FILE_EXISTS, ERROR_NOT_IN_WRITE_SCOPE, ToolError


def ctx_for(tmp_path, scope=(), new=("answer.txt",)):
    return tools.ToolContext(root=tmp_path, write_scope=scope, allowed_new_files=new)


def test_a_file_this_run_created_can_be_rewritten(tmp_path):
    ctx = ctx_for(tmp_path)
    tools.write_file(ctx, "answer.txt", "primer intento\n")
    tools.write_file(ctx, "answer.txt", "segundo intento\n")
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "segundo intento\n"


def test_a_preexisting_file_is_still_protected(tmp_path):
    """The guard exists to stop an accidental whole-file overwrite of real
    source, and it still does."""
    (tmp_path / "answer.txt").write_text("codigo real\n", encoding="utf-8")
    ctx = ctx_for(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.write_file(ctx, "answer.txt", "otra cosa\n")
    assert exc.value.code == ERROR_FILE_EXISTS
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "codigo real\n"


def test_the_refusal_names_a_tool_that_is_actually_available(tmp_path):
    """The trap was a refusal pointing at edit on a mission where edit is
    illegal. It now says why the file is protected, not just what to type."""
    (tmp_path / "answer.txt").write_text("x\n", encoding="utf-8")
    ctx = ctx_for(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.write_file(ctx, "answer.txt", "y\n")
    assert "no lo has creado tu" in exc.value.detail


def test_creating_grants_nothing_else(tmp_path):
    """Containment does not move: creating one file does not make the
    repository writable."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("x = 1\n", encoding="utf-8")
    ctx = ctx_for(tmp_path)
    tools.write_file(ctx, "answer.txt", "hola\n")
    with pytest.raises(ToolError) as exc:
        tools.edit(ctx, "src/real.py", "x = 1", "x = 2")
    assert exc.value.code == ERROR_NOT_IN_WRITE_SCOPE


def test_a_mission_that_names_an_existing_file_grants_nothing(tmp_path):
    """Keyed on what this run actually created, not on allowed_new_files -- so
    a mission author listing an existing source file by mistake cannot hand out
    write access to it."""
    (tmp_path / "answer.txt").write_text("codigo real\n", encoding="utf-8")
    ctx = ctx_for(tmp_path, new=("answer.txt",))
    with pytest.raises(ToolError):
        tools.write_file(ctx, "answer.txt", "otra cosa\n")
    with pytest.raises(ToolError) as exc:
        tools.edit(ctx, "answer.txt", "codigo", "otro")
    assert exc.value.code == ERROR_NOT_IN_WRITE_SCOPE


def test_a_created_file_can_also_be_edited(tmp_path):
    ctx = ctx_for(tmp_path)
    tools.write_file(ctx, "answer.txt", "hola mundo\n")
    tools.edit(ctx, "answer.txt", "mundo", "gente")
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "hola gente\n"
