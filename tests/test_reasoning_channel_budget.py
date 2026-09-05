"""F-95, and F-101 which superseded its rule.

F-95 found that an engine thinking out loud paid for the thought out of the same
budget as the call. That diagnosis holds. Its FIX -- raise the budget only where
a reasoning channel is observed -- did not: measured on the real bare-arm prompt
over eight cases, the engine that truncates worst has no reasoning channel at
all, and the reference engine was being cut off in three cases of eight.

The property is still measured and still recorded. It no longer decides the
number on its own. What follows tests the rule as it now stands, and the first
test exists to record the supersession rather than let it disappear.

The original finding, kept because it is what led here:

granite4.2:3b scored 28% through GAFITAS against 66% answering bare, and 170 of
its 242 turns were ERROR_NO_TOOL_CALL. The measurement that explains it is the
output token usage, not the protocol:

    granite4.2:3b   mean 916   median 974   max 1024
    granite4.1:3b   mean  61   median  46   max  150

1024 is ``CONSERVATIVE_OUTPUT``, the harness's own fallback. granite4.2 returns
its reasoning in a ``thinking`` field, spent the whole allowance there, and
handed back empty ``content`` -- which the harness reported as "respuesta
vacia", blaming the engine for a ceiling we chose.

The property is observed, and it governs one number. No engine is named
anywhere in the path these tests cover.
"""

from __future__ import annotations

import pytest

from localprog import engine


def caps(**kw):
    base = dict(name="anything", context_window=32768, served_context=32768,
                max_output_tokens=engine.UNKNOWN)
    base.update(kw)
    return engine.EngineCapabilities(**base)


def test_the_budget_no_longer_depends_on_the_reasoning_channel():
    """F-101 SUPERSEDED F-95's rule, and this records the supersession.

    F-95 raised the budget only where a reasoning channel was observed. The
    ladder killed that condition -- cap-hits on the real prompt over eight bare
    cases:

        budget          1024   2048   4096   8192
        ministral-3:3b   0/8      -      -      -
        granite4.1:3b    3/8    2/8    1/8    0/8     no reasoning channel
        gemma3:4b        8/8    6/8    5/8    1/8     no reasoning channel

    The engine that truncates worst has no reasoning channel at all, and the
    reference engine was being cut off in three cases of eight. The condition
    was diagnosing how an engine SPENDS its budget when the question is whether
    the budget covers the answer.
    """
    quiet = caps(emits_reasoning_channel=False).working_output()
    loud = caps(emits_reasoning_channel=True).working_output()
    unmeasured = caps().working_output()
    assert quiet == loud == unmeasured == 32768 // 4


def test_the_reasoning_channel_is_still_measured_and_recorded():
    """It stopped deciding the budget. It did not stop being true, and it is
    what explains WHY one engine spends 2175 tokens per call and another 46."""
    assert caps(emits_reasoning_channel=True).emits_reasoning_channel is True
    assert caps(emits_reasoning_channel=False).emits_reasoning_channel is False


def test_a_declared_limit_still_wins_over_the_harness_guess():
    """This replaces our fallback, never the engine's own statement."""
    got = caps(emits_reasoning_channel=True, max_output_tokens=512).working_output()
    assert got == 512


def test_the_context_quarter_is_still_a_hard_ceiling():
    got = caps(emits_reasoning_channel=True, context_window=4096,
               served_context=4096).working_output()
    assert got == 4096 // 4


def test_a_tiny_window_still_gets_a_floor():
    got = caps(emits_reasoning_channel=True, context_window=512,
               served_context=512).working_output()
    assert got == 256


def test_the_probe_records_the_channel_from_what_came_back(monkeypatch):
    """Observed, not inferred. The engine name in this test is deliberately
    meaningless."""
    seen = {"role": "assistant", "content": "", "thinking": "Let me consider..."}

    monkeypatch.setattr(engine, "_context_from_show",
                        lambda *a, **k: (32768, "3.2B", "Q4_K_M"))
    monkeypatch.setattr(engine, "_post",
                        lambda *a, **k: {"message": seen, "eval_count": 7})
    got = engine.probe("some-engine:tag")
    assert got.emits_reasoning_channel is True


def test_the_probe_records_a_quiet_engine_as_quiet(monkeypatch):
    monkeypatch.setattr(engine, "_context_from_show",
                        lambda *a, **k: (32768, "3.2B", "Q4_K_M"))
    monkeypatch.setattr(engine, "_post",
                        lambda *a, **k: {"message": {"role": "assistant",
                                                     "content": "hello"},
                                         "eval_count": 7})
    got = engine.probe("some-engine:tag")
    assert got.emits_reasoning_channel is False


def test_an_empty_reasoning_field_does_not_count(monkeypatch):
    """A key that is present and blank is not a reasoning channel."""
    monkeypatch.setattr(engine, "_context_from_show",
                        lambda *a, **k: (32768, "3.2B", "Q4_K_M"))
    monkeypatch.setattr(engine, "_post",
                        lambda *a, **k: {"message": {"role": "assistant",
                                                     "content": "hi",
                                                     "thinking": "   "},
                                         "eval_count": 7})
    assert engine.probe("some-engine:tag").emits_reasoning_channel is False


def test_the_capability_survives_a_round_trip_through_the_registry(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(engine, "REGISTRY", tmp_path / "ENGINES.json")
    engine.save_registry({"e": caps(emits_reasoning_channel=True)})
    back = engine.load_registry()["e"]
    assert back.emits_reasoning_channel is True
    assert back.working_output() == 32768 // 4
