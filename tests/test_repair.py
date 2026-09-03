"""Recovering a call the engine got nearly right -- and never inventing one.

These tests exist because of a specific measurement. qwen2.5-coder:3b scored 0%
through GAFITAS against 52% answering in one prompt, and the payload probe found
that it reproduces three thousand characters of source with perfect line recall
as prose and could not get any of it through ``write_file``. What it emitted was
a correct call whose JSON was broken by the docstring quotes in its own payload.

So the risk this module carries is not "does it repair enough". It is "does it
ever repair something that was not there" -- because a harness that invents a
tool call is worse than one that drops a real one. Most of what follows tests
the refusals.
"""

from __future__ import annotations

import json

import pytest

from localprog import protocol, repair, tools

KNOWN = frozenset(tools.SPECS)


# ------------------------------------------------------------ what it recovers

def test_the_exact_shape_that_was_losing_runs():
    """A native call in `content`, with unescaped docstring quotes and a raw
    newline in the payload. Ollama drops it; the bytes are all there."""
    blob = ('{"name": "write_file", "arguments": {"content": "def f(x):\\n'
            '    """doc."""\n\\n    return x"}}')
    got = repair.recover(blob, known_tools=KNOWN)
    assert got is not None
    name, args, tier = got
    assert name == "write_file"
    assert tier == repair.TIER_TERMINAL_STRING
    # The payload survives verbatim, quotes and all -- that is the whole point.
    assert args["content"] == 'def f(x):\n    """doc."""\n\n    return x'


def test_a_raw_newline_inside_a_string_is_escaped_not_dropped():
    blob = '{"tool": "write_file", "arguments": {"path": "a.txt", "content": "one\ntwo"}}'
    name, args, tier = repair.recover(blob, known_tools=KNOWN)
    assert (name, tier) == ("write_file", repair.TIER_RAW_CONTROLS)
    assert args["content"] == "one\ntwo"


def test_a_generation_cut_off_mid_payload_still_yields_what_arrived():
    """num_predict ran out. The partial payload is worth more than nothing, and
    the loop can see it is short."""
    blob = ('{"tool": "write_file", "arguments": {"path": "a.txt", '
            '"content": "line1\\nline2 "quoted" tail')
    name, args, _ = repair.recover(blob, known_tools=KNOWN)
    assert name == "write_file"
    assert args["content"] == 'line1\nline2 "quoted" tail'


def test_an_openai_shaped_call_with_stringified_arguments():
    blob = '{"function": {"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}}'
    name, args, _ = repair.recover(blob, known_tools=KNOWN)
    assert (name, args) == ("read_file", {"path": "a.py"})


def test_a_call_wrapped_in_a_fence_and_surrounded_by_prose():
    blob = ("Voy a escribirlo.\n\n```json\n"
            '{"tool": "read_file", "arguments": {"path": "src/a.py"}}\n```\n'
            "Ya esta.")
    name, args, _ = repair.recover(blob, known_tools=KNOWN)
    assert (name, args) == ("read_file", {"path": "src/a.py"})


# ------------------------------------------------------------ what it refuses

@pytest.mark.parametrize("content", [
    "",
    "   ",
    "No puedo hacer eso.",
    "Creo que deberias mirar {esto} primero.",
    "El resultado es {1, 2, 3} elementos.",
    '{"clave": "valor"}',                      # an object, but not a call
    '{"name": 42, "arguments": {}}',           # a name that is not a name
    '{"arguments": {"path": "a.py"}}',         # arguments with nothing to call
])
def test_it_refuses_rather_than_inventing(content):
    assert repair.recover(content, known_tools=KNOWN) is None


def test_a_recovered_name_that_is_not_a_tool_is_rejected():
    """Better a clean ERROR_NO_TOOL_CALL than a manufactured ERROR_UNKNOWN_TOOL:
    the first says what happened, the second blames the engine for our guess."""
    blob = '{"name": "delete_everything", "arguments": {"path": "/"}}'
    assert repair.recover(blob, known_tools=KNOWN) is None
    # Without the gate it IS recoverable -- the gate is what refuses it.
    assert repair.recover(blob) is not None


def test_arguments_that_are_a_list_are_not_coerced_into_a_dict():
    assert repair.recover('{"tool": "grep", "arguments": [1, 2]}',
                          known_tools=KNOWN) is None


# ---------------------------------------------------- it never touches a clean call

def test_a_wellformed_native_call_never_reaches_repair():
    """The guarantee that keeps the reference engine's numbers comparable: the
    ordinary parser runs first and unchanged, so repair cannot alter an engine
    that was already working."""
    message = {"tool_calls": [{"function": {"name": "read_file",
                                            "arguments": {"path": "a.py"}}}]}
    call = protocol.parse("A", message)
    assert call.name == "read_file"
    assert call.repaired is None


def test_a_wellformed_json_call_never_reaches_repair():
    message = {"content": '```json\n{"tool": "grep", "arguments": {"pattern": "x"}}\n```'}
    call = protocol.parse("J", message)
    assert call.repaired is None


def test_a_repaired_call_says_so():
    """Not decoration. A run that only worked because the harness reassembled
    the call is a different fact from one that did not need it."""
    message = {"content": '{"name": "write_file", "arguments": {"path": "a", "content": "x\ny"}}'}
    call = protocol.parse("A", message)
    assert call.name == "write_file"
    assert call.repaired == repair.TIER_RAW_CONTROLS


def test_prose_with_no_call_still_raises_the_ordinary_error():
    from localprog.errors import ERROR_NO_TOOL_CALL, InvalidCall
    with pytest.raises(InvalidCall) as exc:
        protocol.parse("A", {"content": "Lo siento, no se como hacerlo."})
    assert exc.value.code == ERROR_NO_TOOL_CALL


# ------------------------------------------------------------------- properties

def test_escaping_controls_cannot_change_a_valid_document():
    """A raw control character inside a JSON string is never legal, so escaping
    one is not a judgement call. Anything that already parses must round-trip."""
    for value in ({"a": "b\nc"}, {"x": [1, 2, {"y": "z"}]}, {"q": 'he said "hi"'}):
        blob = json.dumps(value)
        assert json.loads(repair._escape_raw_controls(blob)) == value


def test_a_huge_blob_is_bounded_not_scanned_forever():
    assert repair.recover("{" + "x" * (repair.MAX_BLOB * 2)) is None
