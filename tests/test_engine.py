"""The engine contract: what the core may assume about a brain, and what it may not.

Everything this project measured was measured with one engine. An audit for
model-specific logic found none -- no ``if model ==`` anywhere -- and that is the
easy half. The hard half is that the core did not know a model's properties at
all, so it assumed them: ``WORK_NUM_CTX = 32768`` and native tool calls, which
are the reference engine's numbers wearing the costume of universal truth.
"""

from __future__ import annotations

import json

import pytest

from localprog import engine


def caps(**kwargs) -> engine.EngineCapabilities:
    return engine.EngineCapabilities(name=kwargs.pop("name", "e"), **kwargs)


# --------------------------------------------------------------- the contract

def test_an_unknown_capability_is_not_a_default():
    """The core must be able to tell "no" apart from "nobody asked"."""
    c = caps()
    assert c.context_window is engine.UNKNOWN
    assert c.supports_native_tools is engine.UNKNOWN
    assert c.usage_reporting is engine.UNKNOWN


def test_protocol_comes_from_the_engine_not_the_caller():
    assert caps(supports_native_tools=True).protocol == "A"
    assert caps(supports_native_tools=False,
                supports_text_tool_protocol=True).protocol == "J"


def test_an_engine_that_can_drive_neither_protocol_says_so():
    """Rather than failing on turn one with a provider error that reads as the
    engine's fault."""
    with pytest.raises(ValueError, match="neither"):
        _ = caps(supports_native_tools=False, supports_text_tool_protocol=False).protocol


def test_the_working_context_is_the_smaller_of_two_numbers():
    """The architecture's ceiling and what the deployment serves are different
    numbers, and taking the first alone is how a 262144-token ceiling becomes a
    transcript budget on a server configured for 32768."""
    c = caps(context_window=262144, served_context=32768)
    assert c.working_context() == 32768


def test_an_uncertified_engine_gets_a_conservative_context():
    assert caps().working_context() == engine.CONSERVATIVE_CONTEXT


def test_the_output_budget_never_eats_the_window():
    """A model asked to produce more than it can hold alongside its prompt
    spends the run being truncated. Small engines must degrade, not break."""
    small = caps(served_context=4096, max_output_tokens=8192)
    assert small.working_output() <= 4096 // 4
    big = caps(served_context=32768, max_output_tokens=4096)
    assert big.working_output() == 4096


def test_a_declared_capability_is_never_mistaken_for_an_observed_one():
    assert caps().source == "declared"
    assert engine.EngineCapabilities(name="x", source="probed").source == "probed"


# ---------------------------------------------------------------- the registry

def test_a_round_trip_keeps_every_field(tmp_path):
    original = caps(name="m", context_window=131072, served_context=8192,
                    supports_native_tools=True, usage_reporting=True,
                    known_protocol_constraints=("no streaming",), source="probed")
    path = tmp_path / "ENGINES.json"
    engine.save_registry({"m": original}, path)
    back = engine.load_registry(path)["m"]
    assert back == original


def test_an_unregistered_engine_is_unregistered_not_invented(tmp_path):
    c = engine.capabilities_for("never-seen", path=tmp_path / "ENGINES.json")
    assert c.source == "unregistered"
    assert c.context_window is engine.UNKNOWN
    assert c.working_context() == engine.CONSERVATIVE_CONTEXT


def test_a_missing_registry_is_empty_not_an_error(tmp_path):
    assert engine.load_registry(tmp_path / "nope.json") == {}


def test_a_corrupt_registry_is_empty_not_an_error(tmp_path):
    path = tmp_path / "ENGINES.json"
    path.write_text("{not json", encoding="utf-8")
    assert engine.load_registry(path) == {}


def test_an_unknown_field_in_a_stored_record_does_not_break_loading(tmp_path):
    """Forward compatibility: a registry written by a later version must not
    make this one refuse to start."""
    path = tmp_path / "ENGINES.json"
    path.write_text(json.dumps({
        "schema": engine.SCHEMA,
        "engines": {"m": {"name": "m", "context_window": 4096,
                          "some_future_field": True}},
    }), encoding="utf-8")
    assert engine.load_registry(path)["m"].context_window == 4096


# ------------------------------------------------------- the real registry

def test_the_reference_engine_is_certified_in_the_shipped_registry():
    """The engine every number in this project was measured with is recorded,
    with the context and output budget it was measured under -- so a later
    comparison against another engine can be checked rather than trusted."""
    known = engine.load_registry()
    reference = known.get("qwen3:4b-instruct-2507-q4_K_M")
    assert reference is not None, "the reference engine must be in ENGINES.json"
    assert reference.supports_native_tools is True
    assert reference.served_context == 32768
    assert reference.max_output_tokens == 4096
    assert reference.source.startswith("probed")


# ------------------------------------------------- payload limits (the granite lesson)

def test_supports_native_tools_is_necessary_and_not_sufficient():
    """granite4.1:3b declares native tool support, and it is true. It emits a
    flawless call below ~800 characters and, above that, Ollama's parser for its
    template returns content empty AND tool_calls empty -- the generation is
    gone before the harness sees it. Fifty runs were scored against the engine
    for that. The protocol now asks how much it can carry, not whether it can."""
    granite_shaped = caps(supports_native_tools=True, native_payload_limit=800,
                          supports_text_tool_protocol=True)
    assert granite_shaped.protocol == "J"


def test_an_engine_that_carries_a_real_payload_natively_stays_native():
    assert caps(supports_native_tools=True, native_payload_limit=3200,
                supports_text_tool_protocol=True).protocol == "A"


def test_an_unmeasured_payload_limit_changes_nothing():
    """Every engine certified before this existed keeps the protocol it was
    certified with, so their numbers stay comparable."""
    assert caps(supports_native_tools=True).protocol == "A"
    assert caps(supports_native_tools=True).native_payload_limit is engine.UNKNOWN


def test_a_small_native_limit_with_no_text_protocol_stays_native():
    """There is nowhere better to send it. Refusing to drive an engine at all
    because its native channel is narrow would be worse than a narrow channel."""
    assert caps(supports_native_tools=True, native_payload_limit=200,
                supports_text_tool_protocol=False).protocol == "A"


def test_the_floor_is_a_threshold_not_a_per_engine_rule():
    """It is compared against a number every engine is measured for, which is
    what keeps this an engine adapter and not a branch on a model name."""
    assert isinstance(engine.PAYLOAD_FLOOR, int)
    assert engine.PAYLOAD_FLOOR > 0


def test_payload_limits_round_trip_through_the_registry(tmp_path):
    original = caps(name="m", supports_native_tools=True, native_payload_limit=800,
                    text_payload_limit=3200, supports_text_tool_protocol=True)
    path = tmp_path / "ENGINES.json"
    engine.save_registry({"m": original}, path)
    assert engine.load_registry(path)["m"] == original


def test_the_shipped_registry_records_what_was_measured():
    known = engine.load_registry()
    for name in ("granite4.1:3b", "qwen2.5-coder:3b"):
        rec = known.get(name)
        assert rec is not None, f"{name} must stay in the registry"
        assert isinstance(rec.text_payload_limit, int)
    reference = known["qwen3:4b-instruct-2507-q4_K_M"]
    assert reference.protocol == "A", "the reference engine must not move protocol"
