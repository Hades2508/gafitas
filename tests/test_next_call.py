"""search_code shows the next CALL, not the category of next call.

Splitting recall from selection produced the clearest number of the session.
Across fifty matched runs each:

                            4B                  granite4.1:3b
    REACHED_THE_SYMBOL      20 -> 95% correct     2 -> 100% correct
    OPENED_THE_FILE         28 -> 64%            44 ->   7%
    NEVER_OPENED_IT          2 -> 50%             4 ->   0%

"Reached the symbol" means it called read_symbol or copy_code naming the file
AND the symbol. When either engine does that it is essentially always right.
Granite's whole deficit is that it does it twice in fifty where the 4B does it
twenty times: it opens the file, reads it, and retypes from memory instead.

The note already said "lee con read_symbol o read_file(start, end) el que
encaje". That is a category of action. It now shows the call, filled in.
"""

from __future__ import annotations

from localprog import tools

MODULE = '''"""Facturacion."""


def calcular_impuesto(base, region):
    """Devuelve el impuesto aplicable a base en region."""
    return base * TIPOS[region]


def aplicar_descuento(total, pct):
    """Aplica un descuento porcentual al total."""
    return total - total * pct // 100
'''


def build(tmp_path, new=("respuesta.txt",)):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "factura.py").write_text(MODULE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=new)


def test_every_candidate_names_its_symbol(tmp_path):
    """Without it the agent has to parse the identifier back out of the
    declaration line before it can name it in a call, and naming it in a call is
    the whole difference between 95% and 7%."""
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "devuelve el impuesto aplicable")
    assert all("symbol" in c for c in out["candidates"])
    assert out["candidates"][0]["symbol"] == "calcular_impuesto"


def test_the_note_shows_a_read_symbol_call_that_would_work(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "devuelve el impuesto aplicable")
    assert "read_symbol(path='pkg/factura.py', name='calcular_impuesto')" in out["note"]
    # And it is a call that actually works, not a plausible-looking string.
    assert "def calcular_impuesto" in tools.read_symbol(
        ctx, "pkg/factura.py", "calcular_impuesto")


def test_it_offers_copying_only_when_the_mission_is_waiting_for_a_file(tmp_path):
    """Deterministic mission state, the same predicate legal_tools uses. On an
    edit mission copying is not the next action and is not suggested."""
    ctx = build(tmp_path)
    assert "copy_code(" in tools.search_code(ctx, "impuesto aplicable")["note"]

    editing = tools.ToolContext(root=tmp_path, write_scope=("pkg/",),
                                allowed_new_files=())
    assert "copy_code(" not in tools.search_code(editing, "impuesto aplicable")["note"]


def test_the_copy_call_it_shows_is_one_that_would_work(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "devuelve el impuesto aplicable")
    assert ("copy_code(src='pkg/factura.py', into='respuesta.txt', "
            "name='calcular_impuesto')") in out["note"]
    tools.copy_code(ctx, "pkg/factura.py", "respuesta.txt", name="calcular_impuesto")
    written = (tmp_path / "respuesta.txt").read_text(encoding="utf-8")
    assert written.startswith("def calcular_impuesto(base, region):")


def test_a_file_the_mission_already_created_is_not_offered_again(tmp_path):
    ctx = build(tmp_path)
    (tmp_path / "respuesta.txt").write_text("ya escrito\n", encoding="utf-8")
    assert "copy_code(" not in tools.search_code(ctx, "impuesto aplicable")["note"]


def test_the_ranking_caveat_survives(tmp_path):
    """The first candidate is still not promised to be the right one, and the
    note must keep saying so -- an affordance that also became a claim of
    certainty would be worse than the abstraction it replaced."""
    ctx = build(tmp_path)
    note = tools.search_code(ctx, "impuesto aplicable")["note"]
    assert "no por certeza" in note and "no tiene por que ser el bueno" in note
