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


# ------------------------------------------------- red team (Codex), 2026-09-03

def test_an_oversized_payload_is_refused_not_silently_truncated():
    """The finding that mattered. Slicing at MAX_BLOB recovered a write_file
    whose body had been cut at the boundary and returned it as though whole --
    399944 characters of 400000, ending mid-text, with nothing to say so. A
    truncated file written as complete is worse than no call."""
    big = ('{"tool":"write_file","arguments":{"path":"a","content":"BEGIN '
           + "A" * (repair.MAX_BLOB + 1000) + 'TAIL"}}')
    assert repair.recover(big, known_tools=KNOWN) is None


def test_a_duplicate_key_is_refused():
    """json takes the last value, so a call can carry a harmless-looking
    argument and execute a different one. The ordinary parser lives with that;
    this path reassembles text the ordinary parser already rejected, so refusing
    an ambiguous object costs nothing and is plainly right."""
    smuggled = ('{"tool":"run","arguments":{"argv":["python","safe"],'
                '"argv":["python","-c","print(999)"]}}')
    assert repair.recover(smuggled, known_tools=KNOWN) is None


def test_recovery_stays_linear_in_the_size_of_the_input():
    """No catastrophic backtracking: 400k of pathological braces measured at
    ~1.9s, growing linearly, and now bounded by the refusal above."""
    import time
    start = time.perf_counter()
    repair.recover("{}" * 20_000, known_tools=KNOWN)
    assert time.perf_counter() - start < 5.0


# ------------------------------------------- F-92: Python literals, and the
#                                             prose that used to eat an argument

def test_a_python_literal_object_is_recovered_not_mangled():
    """``True``/``False``/``None`` are what a Python-trained model writes.

    json.loads rejects all three, so every one of these calls fell through to
    ``_terminal_string`` -- which is built for a different failure and glued the
    rest of the object onto the last argument it could see. ministral-3:3b did
    this in 28 of 50 runs of the expansion cohort.
    """
    content = ('```json\n{"tool": "grep", "arguments": {"pattern": "def is_valid", '
               '"context": 3, "ignore_case": True}}\n```\n\n*(nota del modelo)*')
    name, args, tier = repair.recover(content, known_tools=KNOWN)
    assert (name, tier) == ("grep", repair.TIER_PYTHON_LITERAL)
    assert args == {"pattern": "def is_valid", "context": 3, "ignore_case": True}


def test_a_python_none_does_not_swallow_the_rest_of_the_call():
    """The read that this reproduces came back with the whole tail of the
    object, closing fence included, glued onto ``path``."""
    content = ('```json\n{"tool": "read_file", "arguments": '
               '{"path": "src/core/vdom.py", "start": 135, "end": None}}\n```')
    name, args, _ = repair.recover(content, known_tools=KNOWN)
    assert name == "read_file"
    assert args["path"] == "src/core/vdom.py"
    assert args["start"] == 135 and args["end"] is None


def test_single_quoted_python_dict_is_recovered():
    content = "{'tool': 'read_symbol', 'arguments': {'path': 'a/b.py', 'name': 'foo'}}"
    name, args, tier = repair.recover(content, known_tools=KNOWN)
    assert (name, tier) == ("read_symbol", repair.TIER_PYTHON_LITERAL)
    assert args == {"path": "a/b.py", "name": "foo"}


def test_text_after_a_closed_object_is_never_taken_for_a_truncated_payload():
    """The guard itself, with no Python literal involved.

    A complete object followed by prose used to hit the cut-off-generation
    branch, because that branch only asked whether the blob ENDED in a brace. A
    truncated generation stops; it does not close its braces and keep writing.
    """
    content = ('{"tool": "finish", "arguments": {"summary": "he dicho \\"listo\\" ya", '
               '"status": "DONE"}}\n```\n\nY ahora explico por que.')
    got = repair.recover(content, known_tools=KNOWN)
    assert got is not None
    _, args, _ = got
    assert args["status"] == "DONE", "status swallowed the tail of the object"


def test_a_genuinely_truncated_payload_is_still_recovered():
    """The branch the guard narrows must keep doing its own job."""
    content = ('{"tool": "write_file", "arguments": {"path": "OUT.txt", '
               '"content": "def f():\n    return 1')
    name, args, tier = repair.recover(content, known_tools=KNOWN)
    assert (name, tier) == ("write_file", repair.TIER_TERMINAL_STRING)
    assert args["content"] == "def f():\n    return 1"


def test_python_literal_tier_cannot_execute_anything():
    """``ast.literal_eval`` is the whole tier precisely because it refuses
    names, calls and imports rather than evaluating them."""
    for hostile in ('{"tool": __import__("os").system("echo no")}',
                    '{"tool": open("/etc/passwd").read()}',
                    '{"tool": 1+1}'):
        assert repair._python_literal(hostile) is None


# ---------------------- F-109: a call written as a call, in plain text

def test_a_python_style_call_is_recovered():
    """qwen3:4b produced 291 dead turns in one cohort and every one was this.

    Not empty, not truncated, not reasoning: a complete call with the right name
    and the right arguments, written as a function call instead of a JSON
    object, and discarded 291 times. The engine scored 80% while closing the
    loop 8% of the time.
    """
    content = ('finish(status="DONE", summary="Copied the get_poller function '
               'from locust/input_events.py to REPOQA_ANSWER.txt as requested.")')
    name, args, tier = repair.recover(content, known_tools=KNOWN)
    assert (name, tier) == ("finish", repair.TIER_TEXT_CALL)
    assert args["status"] == "DONE"
    assert "get_poller" in args["summary"]


def test_the_last_written_call_wins():
    """Models reason and then act, so the final call is the decision."""
    content = ('Primero pense en read_file(path="a.py") pero mejor '
               'finish(status="BLOCKED", summary="no encuentro el simbolo")')
    name, args, _ = repair.recover(content, known_tools=KNOWN)
    assert name == "finish" and args["status"] == "BLOCKED"


def test_prose_with_parentheses_is_not_a_call():
    """The known-tools gate is what stops ordinary text becoming an action."""
    assert repair.recover("La funcion (que ya lei) parece correcta.",
                          known_tools=KNOWN) is None
    assert repair.recover("no_such_tool(path='a.py')", known_tools=KNOWN) is None


def test_a_tool_name_inside_a_longer_word_is_not_a_call():
    assert repair.recover('unfinish(status="DONE")', known_tools=KNOWN) is None
    assert repair.recover('my.finish(status="DONE")', known_tools=KNOWN) is None


def test_without_the_known_tools_gate_nothing_is_recovered():
    """The gate is not an optimisation. Without it this tier is disabled."""
    assert repair._text_call('finish(status="DONE")', None) is None
    assert repair._text_call('finish(status="DONE")', frozenset()) is None


def test_an_unparseable_argument_declines_instead_of_raising():
    """_value raises on a value it cannot read. A repair tier recovers or
    declines; it never raises, because the caller's ordinary error is what the
    model needs to see."""
    content = 'write_file(path="OUT.txt", content="def f():\n  """doc""" \n  return 1")'
    assert repair.recover(content, known_tools=KNOWN) is None


def test_a_wellformed_json_call_never_reaches_the_text_tier():
    """This tier runs only after every object-shaped reading has failed."""
    content = '{"tool": "finish", "arguments": {"status": "DONE", "summary": "ok"}}'
    _name, _args, tier = repair.recover(content, known_tools=KNOWN)
    assert tier != repair.TIER_TEXT_CALL
