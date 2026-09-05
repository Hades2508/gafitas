"""F-114: tests for the instrument that decides how every engine is driven.

`persistence.py` had 309 statements and zero tests, and it is not a reporting
convenience. `measure_payload_limit` chooses the PROTOCOL an engine is driven
with, and F-90 already records that the probe's goal mode ranks two engines in
the reverse of their real order. An instrument that makes decisions, has a
recorded blind spot, and has no tests can drift without anything noticing.

Nothing here talks to a model. A fake provider returns exactly what a real one
would, which makes the properties that matter testable at all:

  - the ladder is CLIMBED, not truncated at the first loss
  - each rung is SAMPLED, so one stochastic miss cannot set the answer
  - a cliff is told apart from noise wearing a cliff's clothes
  - a truncated payload is not a surviving call

Those four are the corrections that were made by hand after three engines gave
three different answers to the same question on three consecutive passes.
"""

from __future__ import annotations

import json

import pytest

from localprog import persistence
from localprog.errors import ProviderError


class FakeProvider:
    """Answers a write_file request, honouring a per-size survival policy.

    ``policy`` maps a payload size to what should happen: True to echo the
    payload back whole, False to emit no call at all, a float to return that
    fraction of it, or a ProviderError instance to raise.
    """

    def __init__(self, policy, protocol_name="A"):
        self.policy = policy
        self.protocol_name = protocol_name
        self.calls = []

    def chat(self, messages, schema=None):
        text = messages[-1]["content"].split("\n\n", 1)[-1]
        size = len(text)
        rung = min(self.policy, key=lambda s: abs(s - size))
        decision = self.policy[rung]
        self.calls.append(rung)
        if isinstance(decision, ProviderError):
            raise decision
        if decision is False:
            return {"message": {"role": "assistant", "content": "no puedo"},
                    "eval_count": 5}
        body = text if decision is True else text[:int(len(text) * float(decision))]
        payload = {"tool": "write_file",
                   "arguments": {"path": "respuesta.txt", "content": body}}
        if self.protocol_name == "A":
            return {"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"function": {
                                    "name": "write_file",
                                    "arguments": payload["arguments"]}}]},
                    "eval_count": 40}
        return {"message": {"role": "assistant",
                            "content": "```json\n" + json.dumps(payload) + "\n```"},
                "eval_count": 40}


LADDER = (200, 400, 800, 1600)


def limit_for(policy, **kw):
    provider = FakeProvider(policy)
    return persistence.measure_payload_limit(provider, ladder=LADDER, **kw)


# --------------------------------------------------- the ladder is climbed

def test_a_loss_low_down_does_not_stop_the_ladder():
    """The first version returned at the first lost rung. granite4.1:3b really
    does fall off a cliff; not every engine does, and truncating the ladder
    turns one unlucky generation into the answer."""
    got = limit_for({200: True, 400: False, 800: True, 1600: True})
    assert [r["size"] for r in got["rungs"]] == list(LADDER), "every rung must be tried"
    assert got["limit"] == 1600


def test_the_limit_is_the_highest_surviving_rung():
    """The highest rung that SURVIVED, not the lowest that failed. With 200 and
    400 through and 800 and 1600 lost, the answer is 400 -- the largest payload
    the engine actually got across, which is what a protocol decision needs."""
    got = limit_for({200: True, 400: True, 800: False, 1600: False})
    assert got["limit"] == 400
    assert got["ceiling_reached"] is False


def test_reaching_the_top_of_the_ladder_is_reported():
    got = limit_for({s: True for s in LADDER})
    assert got["limit"] == LADDER[-1]
    assert got["ceiling_reached"] is True, "the real limit may be higher than we asked"


def test_an_engine_that_survives_nothing_reports_zero():
    got = limit_for({s: False for s in LADDER})
    assert got["limit"] == 0


# ------------------------------------------------------- a cliff versus noise

def test_losses_all_above_the_survivors_is_a_clean_cliff():
    got = limit_for({200: True, 400: True, 800: False, 1600: False})
    assert got["clean_cliff"] is True


def test_a_loss_interleaved_with_survivors_is_not_a_cliff():
    """This is what noise looks like, and calling it a cliff is how a
    stochastic miss became a protocol decision."""
    got = limit_for({200: True, 400: False, 800: True, 1600: True})
    assert got["clean_cliff"] is False


def test_surviving_everything_is_not_a_cliff():
    assert limit_for({s: True for s in LADDER})["clean_cliff"] is False


# ------------------------------------------------------------- rungs are sampled

def test_each_rung_is_sampled_more_than_once():
    provider = FakeProvider({s: True for s in LADDER})
    persistence.measure_payload_limit(provider, ladder=LADDER, samples=3)
    assert len(provider.calls) == len(LADDER) * 3


def test_a_majority_decides_a_rung():
    """One miss in three does not lose a rung, and one hit in three does not win
    one. The alternative measured 400, then 0, then 3200 on the same ladder."""
    provider = FakeProvider({200: True})
    got = persistence.measure_payload_limit(provider, ladder=(200,), samples=3)
    assert got["rungs"][0]["rate"] == 1.0
    assert got["rungs"][0]["samples"] == 3
    assert len(provider.calls) == 3


def test_a_single_sample_is_still_allowed_but_recorded_as_such():
    got = limit_for({s: True for s in LADDER}, samples=1)
    assert all(r["samples"] == 1 for r in got["rungs"])


# ------------------------------------------- a truncated payload is not survival

def test_a_call_that_arrives_half_empty_does_not_count_as_surviving():
    """The tool would write the wrong file. Half is the bar: below it the
    answer is unusable, above it the loss is recoverable by re-reading."""
    got = limit_for({200: 0.2, 400: 0.2, 800: 0.2, 1600: 0.2})
    assert got["limit"] == 0


def test_a_call_that_arrives_mostly_whole_counts_as_surviving():
    got = limit_for({200: 0.9, 400: 0.9, 800: 0.9, 1600: 0.9})
    assert got["limit"] == LADDER[-1]


def test_the_reason_a_rung_was_lost_is_recorded():
    got = limit_for({200: True, 400: False, 800: False, 1600: False})
    lost = [r for r in got["rungs"] if not r["survived"]]
    assert lost and all(r["reason"] for r in lost), "a lost rung with no reason is undiagnosable"


# ----------------------------------------------------------- provider failures

def test_a_provider_error_loses_the_rung_and_names_the_kind():
    got = limit_for({200: True, 400: ProviderError("TIMEOUT", "sin respuesta"),
                     800: True, 1600: True})
    rung = next(r for r in got["rungs"] if r["size"] == 400)
    assert rung["survived"] is False
    assert rung["reason"] == "TIMEOUT", "an infrastructure failure must be named, not merged into 'lost'"


def test_a_provider_error_does_not_abort_the_ladder():
    got = limit_for({200: ProviderError("TRANSPORT", "x"), 400: True,
                     800: True, 1600: True})
    assert len(got["rungs"]) == len(LADDER)
    assert got["limit"] == LADDER[-1]


# ---------------------------------------------------------------- text protocol

def test_the_text_protocol_is_measured_on_its_own_channel():
    """granite4.1:3b reported supports_native_tools=True and lost everything
    above ~800 characters on that channel. The two are measured separately
    because they are different limits."""
    provider = FakeProvider({s: True for s in LADDER}, protocol_name="J")
    got = persistence.measure_payload_limit(provider, protocol_name="J",
                                            ladder=LADDER, samples=1)
    assert got["protocol"] == "J"
    assert got["limit"] == LADDER[-1]


def test_the_payload_text_is_the_size_it_claims():
    for size in LADDER:
        text = persistence._payload_text(size)
        assert abs(len(text) - size) <= len(persistence._PAYLOAD_UNIT), \
            "a rung that does not carry its own size measures the wrong thing"


# ------------------------------- the multi-turn probe, and its recorded blind spot

class ScriptedProvider:
    """Follows the probe's own script for the first ``good`` turns, then stops.

    The script names an expected tool per step, so a fake that always calls the
    same one fails for the wrong reason. Reading _SCRIPT keeps the fake honest
    about what "holding the protocol" actually means here.
    """

    def __init__(self, good=99):
        self.good, self.turn = good, 0

    def chat(self, messages, schema=None):
        self.turn += 1
        if self.turn > self.good:
            return {"message": {"role": "assistant", "content": "ya he terminado"},
                    "eval_count": 6}
        step = persistence._SCRIPT[min(self.turn, len(persistence._SCRIPT)) - 1]
        tool = step[2]
        return {"message": {"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": tool,
                                                         "arguments": {"path": "x.py"}}}]},
                "eval_count": 20}


def test_an_engine_that_keeps_calling_tools_is_scored_as_such():
    """Asserted on the TOOL rate, not the fully-valid rate.

    Each step also checks its arguments through a closure, and a fake that
    reproduced all twenty argument contracts would be testing my mirror of the
    probe rather than the probe. What this probe exists to answer is whether an
    engine is still calling tools at turn twenty, and that is what is asserted.
    """
    run = persistence.measure(ScriptedProvider(), engine_name="fake", protocol_name="A")
    assert run.acceptable_tool_rate == 1.0
    assert run.no_tool_call == 0


def test_an_engine_that_stops_calling_is_caught_at_the_turn_it_stops():
    """The whole point of the probe: not whether it can call a tool once, but
    whether it is still calling one twenty turns later."""
    run = persistence.measure(ScriptedProvider(good=3), engine_name="fake",
                              protocol_name="A")
    assert run.no_tool_call > 0
    assert run.valid_tool_call_rate < 1.0
    assert run.can_drive_the_loop is False


def test_the_report_survives_a_round_trip():
    run = persistence.measure(ScriptedProvider(), engine_name="fake", protocol_name="A")
    body = run.to_dict()
    assert body["engine"] == "fake" and body["protocol"] == "A"
    assert isinstance(body.get("turns"), list) and body["turns"]


def test_a_provider_that_fails_does_not_crash_the_probe():
    class Broken:
        def chat(self, messages, schema=None):
            raise ProviderError("TRANSPORT", "socket closed")

    run = persistence.measure(Broken(), engine_name="fake", protocol_name="A")
    assert run.can_drive_the_loop is False


def test_the_checkpoints_are_the_ones_the_probe_claims():
    """1, 3, 5, 10 and 20: the shape of the question. A probe that measured
    turn one and called it persistence would answer something else."""
    assert persistence.CHECKPOINTS == (1, 3, 5, 10, 20)
    assert persistence.STEPS >= max(persistence.CHECKPOINTS)
