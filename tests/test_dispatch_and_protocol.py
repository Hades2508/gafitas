"""Dispatch classification, the tool schema, and both wire protocols."""

from __future__ import annotations

import pytest

from localprog import errors, protocol, tools


# ------------------------------------------------------------------- dispatch


def test_unknown_tool_is_an_invalid_call_not_a_crash(ctx):
    """A model asking for list_dir -- which the contract does not define."""
    out = tools.dispatch(ctx, "list_dir", {"path": "."})
    assert not out.ok and out.invalid_call
    assert out.code == errors.ERROR_UNKNOWN_TOOL
    assert "read_file" in out.feedback  # it is told what does exist


def test_missing_required_argument_is_an_invalid_call(ctx):
    out = tools.dispatch(ctx, "edit", {"path": "calc.py"})
    assert not out.ok and out.invalid_call
    assert out.code == errors.ERROR_MISSING_ARGUMENT
    assert "old" in out.feedback and "new" in out.feedback


def test_arguments_as_json_string_are_accepted(ctx):
    out = tools.dispatch(ctx, "read_file", '{"path": "calc.py"}')
    assert out.ok and "def add" in out.value


def test_arguments_that_are_a_list_are_an_invalid_call(ctx):
    out = tools.dispatch(ctx, "read_file", ["calc.py"])
    assert not out.ok and out.invalid_call and out.code == errors.ERROR_BAD_ARGUMENTS


def test_arguments_that_are_broken_json_are_an_invalid_call(ctx):
    out = tools.dispatch(ctx, "read_file", "{not json")
    assert not out.ok and out.invalid_call


def test_extra_arguments_are_tolerated_and_reported(ctx):
    out = tools.dispatch(ctx, "read_file", {"path": "calc.py", "encoding": "utf-8"})
    assert out.ok and "encoding" in out.feedback


def test_an_unexpected_exception_becomes_harness_invalid(ctx, monkeypatch):
    """The rule that keeps HARNESS_INVALID meaningful."""
    def boom(*a, **k):
        raise ValueError("simulated harness defect")

    monkeypatch.setitem(tools._IMPL, "read_file", boom)
    with pytest.raises(errors.HarnessInvalid) as excinfo:
        tools.dispatch(ctx, "read_file", {"path": "calc.py"})
    assert "UNHANDLED_TOOL_ERROR" in excinfo.value.detail
    assert excinfo.value.traceback_text  # preserved for the post mortem


def test_tool_error_and_invalid_call_are_never_both_set(ctx):
    for name, args in [("read_file", {"path": "nope"}), ("list_dir", {}),
                       ("edit", {"path": "calc.py"}), ("grep", {"pattern": "(["})]:
        out = tools.dispatch(ctx, name, args)
        assert not (out.tool_error and out.invalid_call)


# --------------------------------------------------------------- tool schema


def test_native_schema_covers_every_tool_and_nothing_else():
    """'tool schema incompleto': the previous runner advertised list_dir with
    no implementation, and named the test tool `run` in one file and
    `run_tests` in another."""
    schema = tools.native_schema()
    assert {f["function"]["name"] for f in schema} == set(tools.SPECS)


@pytest.mark.parametrize("entry", tools.native_schema())
def test_schema_required_matches_specs(entry):
    name = entry["function"]["name"]
    required, optional = tools.SPECS[name]
    params = entry["function"]["parameters"]
    assert set(params["required"]) == set(required)
    assert set(params["properties"]) == set(required + optional)


# -------------------------------------------------------------- protocol A


def test_protocol_a_reads_a_tool_call():
    call = protocol.parse("A", {"tool_calls": [
        {"function": {"name": "read_file", "arguments": {"path": "calc.py"}}}
    ]})
    assert call.name == "read_file"


def test_protocol_a_without_tool_calls_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall) as excinfo:
        protocol.parse("A", {"content": "creo que deberíamos leer calc.py"})
    assert excinfo.value.code == errors.ERROR_NO_TOOL_CALL


def test_protocol_a_with_malformed_tool_call_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall):
        protocol.parse("A", {"tool_calls": [{"nope": 1}]})


# -------------------------------------------------------------- protocol B


def test_protocol_b_preserves_equals_inside_strings():
    """The regex bug that made every protocol-B run meaningless.

    re.sub(r'(\\w+)\\s*=', ...) rewrote `x =` INSIDE the value too, so this
    exact call came out corrupted and the model was blamed for it.
    """
    call = protocol.parse_text_call(
        'LLAMADA: edit(path="calc.py", old="x = 1", new="x = 2")'
    )
    assert call.name == "edit"
    assert call.arguments == {"path": "calc.py", "old": "x = 1", "new": "x = 2"}


def test_protocol_b_handles_commas_and_parens_inside_strings():
    call = protocol.parse_text_call(
        'LLAMADA: edit(path="calc.py", old="f(a, b)", new="g(a, b, c)")'
    )
    assert call.arguments["old"] == "f(a, b)"
    assert call.arguments["new"] == "g(a, b, c)"


def test_protocol_b_handles_escaped_quotes_and_newlines():
    call = protocol.parse_text_call(
        'LLAMADA: edit(path="calc.py", old="a", new="raise ValueError(\\"division by zero\\")")'
    )
    assert call.arguments["new"] == 'raise ValueError("division by zero")'


def test_protocol_b_takes_the_last_marker_after_reasoning():
    call = protocol.parse_text_call(
        "Primero pensaba LLAMADA: read_file(path=\"otro.py\")\n"
        "pero mejor esto:\nLLAMADA: read_file(path=\"calc.py\")"
    )
    assert call.arguments["path"] == "calc.py"


def test_protocol_b_accepts_a_bare_json_object():
    call = protocol.parse_text_call('LLAMADA: edit({"path": "a.py", "old": "x", "new": "y"})')
    assert call.arguments["path"] == "a.py"


def test_protocol_b_accepts_null_and_lists():
    call = protocol.parse_text_call('LLAMADA: run_tests(node_ids=["test_calc.py"])')
    assert call.arguments == {"node_ids": ["test_calc.py"]}
    call = protocol.parse_text_call("LLAMADA: read_file(path='calc.py', start=None)")
    assert call.arguments["start"] is None


def test_protocol_b_without_marker_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall) as excinfo:
        protocol.parse_text_call("no sé qué hacer")
    assert excinfo.value.code == errors.ERROR_FORMAT


def test_protocol_b_unbalanced_parenthesis_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall):
        protocol.parse_text_call('LLAMADA: edit(path="calc.py"')


def test_protocol_b_unreadable_value_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall):
        protocol.parse_text_call("LLAMADA: edit(path=<<<)")


def test_unknown_protocol_is_rejected():
    with pytest.raises(errors.InvalidCall):
        protocol.parse("Z", {})
