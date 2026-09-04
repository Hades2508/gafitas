"""An identical failing call, twenty times, is a stall and must be named one.

qwen2.5-coder:3b, v6, verbatim:

    t1   finish(status='DONE')   ERROR_NOTHING_CHANGED
    t2   finish(status='DONE')   ERROR_NOTHING_CHANGED
    ...
    t20  finish(status='DONE')   ERROR_NOTHING_CHANGED

Twenty turns, one call, the same refusal every time, and the loop allowed all of
it. The run was then sealed BUDGET_EXHAUSTED, which is not what happened to it.

STALLED already means "asking again produces the same fact". It was wired only
to turns that produced NO call at all, so a call failing identically forever was
invisible to it: a ToolError is not a dead turn by the loop's definition.
"""

from __future__ import annotations

from localprog import loop, tools


def test_the_threshold_is_looser_than_the_dead_turn_one():
    """A failing call is at least an attempt, and the reference engine
    legitimately retries an edit two or three times after re-reading. Killing
    those would trade a real capability for a cost saving."""
    assert loop.REPEAT_STALL_THRESHOLD > loop.STALL_THRESHOLD


def test_stalling_on_repeats_is_wired_to_the_error_path():
    import inspect
    source = inspect.getsource(loop.run_loop)
    assert "if error_repeat >= REPEAT_STALL_THRESHOLD:" in source
    assert "result.outcome = STALLED" in source


def test_stalled_is_still_a_scoreable_outcome():
    """A stall is a measurement, not a harness failure. It must stay countable
    or the runs it saves would vanish from the denominator."""
    result = loop.LoopResult(outcome=loop.STALLED)
    assert result.scoreable


def test_the_refusal_names_the_ranked_call_when_there_is_one(tmp_path):
    """F-71 named write_file in prose and qwen2.5-coder:3b called finish twenty
    times against it. The harness ranked a candidate at turn zero; naming that
    call is the difference F-82 measured at 95% against 7%."""
    from localprog.errors import ToolError

    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=("r.txt",),
                            opening_candidate=("pkg/factura.py", "calcular_impuesto"))
    try:
        tools.finish(ctx, summary="listo", status="DONE")
    except ToolError as exc:
        detail = exc.detail
    else:
        raise AssertionError("DONE with nothing changed must still be refused")
    assert "copy_code(src='pkg/factura.py'" in detail
    assert "into='r.txt'" in detail and "name='calcular_impuesto'" in detail
    # And the hand-written route is still offered, because the answer is not
    # always already in the repository.
    assert "write_file(path='r.txt'" in detail


def test_without_a_ranked_candidate_the_message_is_what_it_was(tmp_path):
    from localprog.errors import ToolError

    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=("r.txt",))
    try:
        tools.finish(ctx, summary="listo", status="DONE")
    except ToolError as exc:
        assert "copy_code(" not in exc.detail
        assert "write_file(path='r.txt'" in exc.detail
    else:
        raise AssertionError("DONE with nothing changed must still be refused")
