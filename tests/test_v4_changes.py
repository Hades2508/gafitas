"""Three changes aimed at SYMBOL_REACH, SELECTION and CLEAN FINISH.

Measured on the v3 sweep, three engines, n=50 each:

    engine            MODEL_ONLY  GAFITAS  reach  converts
    qwen3 4B              70%       76%    37/50    89%
    granite4.1:3b         52%       28%    13/50   100%
    qwen2.5-coder:3b      56%       16%    10/50    80%

Conversion once the symbol is named is 80-100% on all three. The score is the
reach rate. granite's 37 non-reaching runs, BY RUN:

    typed the answer by hand              17
    copied a REAL but WRONG symbol        15
    invented a symbol name                 5

so the dominant failure is selection, and the three changes below attack
selection, spelling and closing respectively.
"""

from __future__ import annotations

import pytest

from localprog import orient, tools, work
from localprog.errors import ERROR_NO_MATCH, ToolError

MODULE = '''def calcular_impuesto(base, region):
    """Devuelve el impuesto aplicable a base en region."""
    return base * TIPOS[region]


def is_valid_percentile(value):
    """Comprueba si un percentil es valido."""
    return 0 < value < 1


def guardar_pedido(pedido):
    """Persiste el pedido."""
    DB.save(pedido)
'''


def build(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "factura.py").write_text(MODULE, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------- 1. opening candidates

def test_the_objective_carries_a_ranked_shortlist(tmp_path):
    """Ranking is arithmetic. It needs no inference, the index is already
    built, and leaving it to the agent costs a turn that is often never spent --
    25 of granite's 37 non-reaching runs never searched at all."""
    root = build(tmp_path)
    out = orient.opening_candidates(
        root, "una funcion que devuelve el impuesto aplicable a una base "
              "segun la region del cliente")
    assert "CANDIDATOS" in out
    assert "calcular_impuesto" in out.splitlines()[1]


def test_the_shortlist_says_it_is_a_ranking_and_not_a_verdict(tmp_path):
    """F-85 is the standing lesson: a list that looks like a recommendation
    gets believed. granite took the first name offered 12 times in 50 and
    finished, confident and wrong."""
    root = build(tmp_path)
    out = orient.opening_candidates(root, "el impuesto aplicable a una base segun region")
    assert "NO por certeza" in out
    assert "no tiene por que ser el bueno" in out


def test_a_short_objective_produces_no_shortlist(tmp_path):
    """A three-word objective ranks noise, and a shortlist built from noise is
    worse than none."""
    assert orient.opening_candidates(build(tmp_path), "arregla el bug") == ""


def test_a_broken_index_is_not_silently_indistinguishable_from_no_matches(tmp_path):
    """The first version caught Exception and swallowed a NameError from a
    missing import for an entire debugging cycle -- it returned "no candidates",
    which is exactly what a working index with no matches returns. A handler
    that cannot be told apart from success is not a handler."""
    import inspect
    source = inspect.getsource(orient.opening_candidates)
    assert "except Exception" not in source


def test_the_shortlist_reaches_the_objective(tmp_path):
    root = build(tmp_path)
    ticket = work.Ticket(ticket_id="t", repo=root,
                         objective="localiza la funcion que comprueba si un "
                                   "percentil es valido y copiala",
                         write_scope=(), allowed_new_files=("r.txt",),
                         acceptance_tests=(), full_suite=False, max_turns=20)
    text = work.objective_text(ticket, root)
    assert "CANDIDATOS" in text and "is_valid_percentile" in text


# ------------------------------------------------------------ 2. did you mean

def test_an_invented_symbol_name_gets_the_closest_real_one(tmp_path):
    """granite asked for '_validate_percentile' where the symbol is
    'is_valid_percentile'. The answer was an ALPHABETICAL list of twenty."""
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "pkg/factura.py", "r.txt", name="_validate_percentile")
    assert exc.value.code == ERROR_NO_MATCH
    assert "is_valid_percentile" in exc.value.detail.split("Todos los simbolos")[0]


def test_a_name_like_nothing_in_the_file_suggests_nothing(tmp_path):
    """When nothing is close enough it returns nothing, and the caller gets the
    plain listing. A confident wrong suggestion is the F-85 failure again."""
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    with pytest.raises(ToolError) as exc:
        tools.copy_code(ctx, "pkg/factura.py", "r.txt", name="zzzqqq_xyzzy")
    assert "Lo mas parecido" not in exc.value.detail
    assert "Todos los simbolos" in exc.value.detail


def test_the_full_listing_is_still_there(tmp_path):
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    with pytest.raises(ToolError) as exc:
        tools.read_symbol(ctx, "pkg/factura.py", "_validate_percentile")
    assert "guardar_pedido" in exc.value.detail


def test_the_suggestion_ranks_against_the_guess_not_the_objective(tmp_path):
    """This is a spelling correction, not retrieval. F-85 was about ranking by
    something the caller never asked for; this ranks by the name they typed."""
    import inspect
    # The signature, not the prose: it takes the caller's guess and the file's
    # own symbols, and there is no parameter through which the objective could
    # reach it. That is the guarantee; a docstring is not.
    params = list(inspect.signature(tools._did_you_mean).parameters)
    assert params == ["guess", "known"]


# --------------------------------------------------------- 3. finish, cleanly

def test_finish_accepts_a_status_without_a_summary(tmp_path):
    """qwen2.5-coder:3b called finish(status='DONE') and was refused 57 times in
    50 runs. summary is documentation: signal_agent_reported takes
    finish_status, and I9/F-24 hold -- the agent's claim cannot create a PASS."""
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    tools.write_file(ctx, "r.txt", "la respuesta\n")
    out = tools.finish(ctx, status="DONE")
    assert ctx.finish_status == "DONE"
    assert "sin summary" in out


def test_the_missing_summary_is_recorded_not_invented(tmp_path):
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    tools.write_file(ctx, "r.txt", "x\n")
    tools.finish(ctx, status="DONE")
    assert ctx.finish_summary == ""


def test_a_summary_still_works_and_is_kept(tmp_path):
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    tools.write_file(ctx, "r.txt", "x\n")
    out = tools.finish(ctx, summary="he copiado la funcion", status="DONE")
    assert ctx.finish_summary == "he copiado la funcion"
    assert "sin summary" not in out


def test_done_with_nothing_changed_is_still_refused(tmp_path):
    """The check that mattered stays. An agent that has done nothing and
    believes it is done is the exact confusion finish() exists to catch."""
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    with pytest.raises(ToolError):
        tools.finish(ctx, status="DONE")


def test_a_non_string_summary_is_still_refused(tmp_path):
    from localprog.errors import InvalidCall
    root = build(tmp_path)
    ctx = tools.ToolContext(root=root, allowed_new_files=("r.txt",))
    tools.write_file(ctx, "r.txt", "x\n")
    with pytest.raises(InvalidCall):
        tools.finish(ctx, summary=42, status="DONE")
