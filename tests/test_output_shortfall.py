"""F-120: closing F-101 by analysis, and admitting what is left of it.

F-101 removed the CONSERVATIVE_OUTPUT guess of 1024. `working_output()` now
returns a quarter of the served context for any engine that declares no limit
of its own, and that bound is principled: nothing is asked to produce more than
it can hold beside its own prompt.

What F-101 did not settle is whether a quarter is ENOUGH. Measured over
cohort 3's sealed bare answers -- `bare_truncation.py`, offline, no engine
invoked -- six of seven engines end 0-8% of their answers in a state a finished
answer does not reach, and only one of those still scored PASS. One engine ends
16 of 50 that way, at a budget of 8192 tokens, which is far more than any answer
asked for.

So the budget is not too small for the ANSWER. It is too small for the answer
plus what that engine spends before it, and 80 of that same engine's 82 empty
turns carried a reasoning channel (F-119).

The policy these tests pin is deliberately conservative:

  - the budget itself does NOT move. Raising it on the strength of a
    measurement of the symptom is tuning, and this campaign does not tune.
  - the shortfall becomes VISIBLE, so a run the deployment cannot afford says
    so instead of being recorded as the engine failing.
  - nothing keys on a family, a name or a parameter count. The only input is a
    number measured from that engine's own sealed telemetry.
  - "never measured" is its own answer, distinct from "measured and fine".
    Collapsing those two is how F-101 survived two cohorts.
"""

from __future__ import annotations

import pytest

from localprog.engine import EngineCapabilities


def caps(**kw):
    base = dict(name="fake", context_window=262144, served_context=32768,
                supports_native_tools=True)
    base.update(kw)
    return EngineCapabilities(**base)


# ------------------------------------------------------- the budget is unmoved

def test_the_budget_is_a_quarter_of_what_is_served():
    assert caps().working_output() == 32768 // 4


def test_a_declared_limit_still_wins():
    assert caps(max_output_tokens=4096).working_output() == 4096


def test_a_measured_reasoning_cost_does_not_move_the_budget():
    """The whole point. Measuring the symptom does not license raising the
    number: that would be tuning against the observation."""
    quiet = caps().working_output()
    loud = caps(measured_reasoning_tokens=30_000).working_output()
    assert quiet == loud


# ------------------------------------------------------ the shortfall is stated

def test_an_engine_that_fits_reports_no_shortfall():
    engine = caps(measured_reasoning_tokens=2000)
    assert engine.output_shortfall() == 0
    assert engine.output_is_sufficient() is True


def test_an_engine_that_does_not_fit_says_by_how_much():
    engine = caps(measured_reasoning_tokens=10_000)
    # 10 000 of thinking plus room for an answer, against a budget of 8192.
    assert engine.output_shortfall() == 10_000 + engine.ANSWER_ALLOWANCE - 8192
    assert engine.output_is_sufficient() is False


def test_never_measured_is_its_own_answer():
    """Not True. An engine nobody looked at is not an engine that is fine, and
    reporting it as fine is exactly the error F-101 was."""
    assert caps().output_is_sufficient() is None
    assert caps().output_shortfall() == 0


def test_a_shortfall_needs_the_allowance_to_be_real():
    """Thinking that exactly fills the budget still leaves nothing to answer
    with, so it must count as a shortfall."""
    engine = caps(measured_reasoning_tokens=8192)
    assert engine.output_shortfall() == engine.ANSWER_ALLOWANCE


# ------------------------------------------------- the recommendation is arithmetic

def test_the_served_context_that_would_close_it_is_computed_not_guessed():
    engine = caps(measured_reasoning_tokens=10_000)
    wanted = engine.served_context_for()
    assert wanted == (engine.working_output() + engine.output_shortfall()) * 4
    # And the recommendation must actually be sufficient, or it is not one.
    fixed = caps(served_context=wanted, measured_reasoning_tokens=10_000)
    assert fixed.output_shortfall() == 0


def test_no_shortfall_recommends_no_change():
    engine = caps(measured_reasoning_tokens=2000)
    assert engine.served_context_for() == engine.working_context()


# ----------------------------------------------------------------- neutrality

@pytest.mark.parametrize("name", ["qwen3.5:2b-q4_K_M", "granite4.1:3b",
                                  "ministral-3:3b", "phi4-mini:3.8b"])
def test_the_verdict_does_not_depend_on_the_name(name):
    """I15 and the general-correction rule. Two engines with the same measured
    numbers get the same answer whatever they are called."""
    a = caps(name=name, measured_reasoning_tokens=10_000)
    b = caps(name="anonymous", measured_reasoning_tokens=10_000)
    assert a.output_shortfall() == b.output_shortfall()
    assert a.output_is_sufficient() == b.output_is_sufficient()


def test_the_reasoning_flag_does_not_decide_it_either():
    """F-101's lesson kept: `emits_reasoning_channel` describes HOW an engine
    spends the budget, and the question is whether the budget covers the work.
    An engine with no channel that still spends the tokens must be caught."""
    silent = caps(emits_reasoning_channel=False, measured_reasoning_tokens=10_000)
    loud = caps(emits_reasoning_channel=True, measured_reasoning_tokens=10_000)
    assert silent.output_shortfall() == loud.output_shortfall() > 0


def test_the_measurement_travels_with_the_record():
    body = caps(measured_reasoning_tokens=10_000).to_dict()
    assert body["measured_reasoning_tokens"] == 10_000, \
        "a capability that does not survive serialisation cannot be audited"
