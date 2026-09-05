"""F-98..F-101: what the last cohort could not be asked, and now can.

Seven engines, 350 runs, and six questions the evidence could not answer:

    316 turns on one engine produced no tool call and the text was not kept
    196 on another, likewise
     13 runs died on a 300 s timeout against an 8192-token budget
      -  no per-call usage, so a cap-hit could not be counted
      -  no reasoning/answer split, so thinking cost could not be separated
      -  no per-turn clock, so time-to-first-action did not exist
      -  no bare-arm usage at all, so no token comparison between the arms

None of this changes what the agent sees or how a decision is taken. It records
what was already happening. An instrumentation change and a behaviour change in
the same cohort would make the next result unattributable, which is the whole
reason these are separate.
"""

from __future__ import annotations

import pytest

from localprog import loop, provider


# --------------------------------------------------- F-98: budget vs patience

def test_a_timeout_covers_the_budget_it_granted():
    """13 runs scored zero because we asked for more than we would wait for."""
    got = provider.timeout_for(8192, 300.0)
    assert got > 300.0
    assert got >= 8192 / provider.MIN_TOKENS_PER_SECOND


def test_a_generous_caller_is_never_cut_down():
    """This is a floor, not an override. Waiting longer is a policy."""
    assert provider.timeout_for(1024, 900.0) == 900.0


def test_a_small_budget_keeps_a_small_timeout():
    assert provider.timeout_for(1024, 300.0) == 300.0


def test_a_hung_server_is_still_a_bounded_failure():
    assert provider.timeout_for(10_000_000, 0) == provider.TIMEOUT_CEILING_SECONDS


def test_the_provider_applies_the_floor_and_remembers_the_request():
    client = provider.OllamaProvider("any-engine", num_predict=8192, timeout=300.0)
    assert client.requested_timeout == 300.0
    assert client.timeout > 300.0


# ------------------------------------ F-99: the text behind a dead turn

def test_the_model_s_own_words_are_sealed():
    sealed = loop._seal_text("no puedo llamar a ninguna herramienta")
    assert sealed["text"] == "no puedo llamar a ninguna herramienta"
    assert sealed["chars"] == len("no puedo llamar a ninguna herramienta")
    assert sealed["truncated"] is False


def test_a_long_emission_is_bounded_and_says_so():
    """An engine in a loop can emit a great deal. Bounded, never silently."""
    sealed = loop._seal_text("x" * 10_000)
    assert len(sealed["text"]) == loop.NO_CALL_TEXT_LIMIT
    assert sealed["chars"] == 10_000
    assert sealed["truncated"] is True


def test_a_non_string_emission_is_described_not_dropped():
    sealed = loop._seal_text(None)
    assert sealed["chars"] == 0 and "NoneType" in sealed["note"]


def test_an_empty_emission_is_distinguishable_from_a_missing_one():
    """'It said nothing' and 'it said something we could not parse' are
    different failures and used to look identical."""
    assert loop._seal_text("")["chars"] == 0
    assert loop._seal_text("")["truncated"] is False
    assert "note" not in loop._seal_text("")


# ------------------------------------------- F-100: per-call cost and thinking

def test_per_call_usage_is_this_call_not_the_running_total():
    got = loop._call_usage({"prompt_eval_count": 1200, "eval_count": 850,
                            "total_duration": 3_500_000_000})
    assert got == {"input_tokens": 1200, "output_tokens": 850,
                   "provider_seconds": 3.5}


def test_per_call_usage_of_a_silent_provider_is_empty_not_zero():
    """Zeros would read as 'this call was free', which is a different claim."""
    assert loop._call_usage({"message": {}}) == {}
    assert loop._call_usage(None) == {}


def test_the_reasoning_channel_is_recorded_separately():
    got = loop._reasoning_of({"content": "ok", "thinking": "let me consider..."})
    assert got["reasoning_key"] == "thinking"
    assert got["reasoning_chars"] == len("let me consider...")


@pytest.mark.parametrize("key", ["thinking", "reasoning", "reasoning_content"])
def test_every_known_reasoning_key_is_seen(key):
    assert loop._reasoning_of({key: "pensando"})["reasoning_chars"] == 8


def test_a_quiet_engine_contributes_no_reasoning_fields():
    """Absent, not zero: an engine without the channel is not an engine that
    thought for nothing."""
    assert loop._reasoning_of({"content": "hola"}) == {}
    assert loop._reasoning_of({"thinking": "   "}) == {}


def test_reasoning_is_measured_in_characters_and_says_so():
    """The server reports no token split. Deriving one from a ratio would be a
    guess wearing a number, so the field name carries the unit."""
    got = loop._reasoning_of({"thinking": "abc"})
    assert "reasoning_chars" in got
    assert not any(k.endswith("_tokens") for k in got)


# ------------------------------- F-101: cut off is not the same as wrong

def test_a_generation_that_reached_its_budget_is_marked():
    got = loop._call_usage({"eval_count": 1024, "prompt_eval_count": 300}, budget=1024)
    assert got["hit_budget"] is True
    assert got["budget"] == 1024


def test_a_generation_that_stopped_on_its_own_is_not_marked():
    got = loop._call_usage({"eval_count": 47, "prompt_eval_count": 300}, budget=1024)
    assert got["hit_budget"] is False


def test_without_a_known_budget_no_claim_is_made():
    """Absent, not False. 'We did not check' and 'it did not happen' are
    different, and only one of them belongs in evidence."""
    got = loop._call_usage({"eval_count": 1024}, budget=None)
    assert "hit_budget" not in got and "budget" not in got


def test_a_provider_that_reports_nothing_makes_no_truncation_claim():
    assert loop._call_usage({}, budget=1024) == {}
