"""search_code(path=<a file>) must search THAT FILE.

Reproduced against the real corpus, on a case the reference engine failed:

    search_code("returns a string of newlines and optionally a form feed",
                path="src/black/nodes.py")
    -> rank 1: src/black/comments.py

``path`` was resolved as a directory prefix, so a file became the prefix
"src/black/nodes.py/", nothing was indexed under it, F-69's scope note correctly
reported "nothing under that prefix" and widened the search to the whole
repository. Every step behaved as written and their sum discarded the only
constraint the agent had expressed -- on a mission whose objective NAMES the
file. That run spent 15 turns, called search_code three times, and finished
having written nothing.

With the file honoured, the function it was looking for ranks first.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_SEARCH_SCOPE_EMPTY, ToolError

FILE_A = '''def buscar_usuario(nombre):
    """Encuentra un usuario por su nombre."""
    return DB.get(nombre)


def borrar_usuario(nombre):
    """Elimina un usuario del sistema."""
    DB.pop(nombre, None)
'''

FILE_B = '''def buscar_pedido(codigo):
    """Encuentra un pedido por su codigo."""
    return PEDIDOS.get(codigo)
'''


def build(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "usuarios.py").write_text(FILE_A, encoding="utf-8")
    (tmp_path / "pkg" / "pedidos.py").write_text(FILE_B, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=("out.txt",))


def test_a_file_path_restricts_the_search_to_that_file(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "encuentra algo por su identificador",
                            path="pkg/usuarios.py")
    assert out["candidates"], "the query matches; it must return something"
    assert {c["path"] for c in out["candidates"]} == {"pkg/usuarios.py"}


def test_without_the_scope_the_other_file_is_reachable(tmp_path):
    """Proves the previous test is measuring the scope and not the ranking."""
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "encuentra un pedido por su codigo")
    assert any(c["path"] == "pkg/pedidos.py" for c in out["candidates"])


def test_a_directory_path_still_works(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "encuentra algo", path="pkg")
    assert all(c["path"].startswith("pkg/") for c in out["candidates"])


def test_a_prefix_that_names_nothing_still_widens_and_says_so(tmp_path):
    """F-69 is for a prefix that does not exist -- a Maven-shaped path in a
    checkout that has none. A file that DOES exist is a different case and must
    not be swept into it."""
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "encuentra algo", path="src/main/java")
    assert out["candidates"], "widening is the point of the F-69 note"
    assert ERROR_SEARCH_SCOPE_EMPTY in str(out)


def test_a_file_with_nothing_indexable_says_so_rather_than_searching_elsewhere(tmp_path):
    """An extension the index does not carve. Answering from another file would
    be the same silent widening, one layer down."""
    ctx = build(tmp_path)
    (tmp_path / "datos.bin").write_text("solo bytes\n", encoding="utf-8")
    with pytest.raises(ToolError) as exc:
        tools.search_code(ctx, "encuentra algo", path="datos.bin")
    assert exc.value.code == ERROR_SEARCH_SCOPE_EMPTY
    assert "quita path=" in exc.value.detail.lower()


def test_no_match_inside_a_real_file_says_the_file_is_real(tmp_path):
    """The difference between "your scope is wrong" and "your scope is right and
    the thing is not in it" is the difference between one more search and one
    fewer."""
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.search_code(ctx, "zzzqqq zzzqqq", path="pkg/usuarios.py")
    assert "SI existe" in exc.value.detail
