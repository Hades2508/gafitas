"""The escalation tier: protocol J, the text manual, and the no-edit nudge.

F-13  LOCAL -> LUNA -> CLAUDE. Luna reaches gpt-5.6 through the Codex CLI,
      which is an agent rather than a chat endpoint, so it cannot be handed a
      tool schema. Protocol J is how a text model drives the same loop, the
      same tools, the same containment and the same conscience.
F-34  An agent that explores forever is told so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import errors, loop, protocol, tools, work  # noqa: E402
from localprog.provider import CodexProvider, FakeProvider  # noqa: E402


def say(text):
    return {"role": "assistant", "content": text}


def jcall(name, **arguments):
    payload = json.dumps({"tool": name, "arguments": arguments})
    return say("Voy a hacer esto.\n\n```json\n" + payload + "\n```")


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("calc.py",))


# ------------------------------------------------------------------ protocol J


def test_a_fenced_json_call_is_parsed():
    content = '```json\n{"tool":"read_file","arguments":{"path":"a.py"}}\n```'
    call = protocol.parse("J", {"content": content})
    assert call.name == "read_file" and call.arguments == {"path": "a.py"}


def test_the_last_call_wins_because_models_reason_first():
    content = (
        'Podria leer el fichero:\n'
        '```json\n{"tool":"read_file","arguments":{"path":"a.py"}}\n```\n'
        'Pero mejor edito:\n'
        '```json\n{"tool":"edit","arguments":{"path":"a.py","old":"x","new":"y"}}\n```'
    )
    assert protocol.parse("J", {"content": content}).name == "edit"


def test_an_unfenced_object_is_still_accepted():
    """A model that forgets the fence has communicated perfectly clearly.
    Refusing it would measure fence discipline rather than programming."""
    content = 'Hago esto: {"tool": "finish", "arguments": {"summary": "ya"}}'
    assert protocol.parse("J", {"content": content}).name == "finish"


def test_multiline_code_survives_the_round_trip():
    """The reason protocol J exists rather than reusing B. B's syntax is
    LLAMADA: edit(path=..., old=..., new=...) parsed by a hand-written scanner,
    and arguments full of newlines, quotes and code are exactly what a
    programming agent sends all day."""
    body = 'def f():\n    return "a, b = 1, 2"\n'
    raw = json.dumps({"tool": "write_file", "arguments": {"path": "m.py", "content": body}})
    call = protocol.parse("J", {"content": "```json\n" + raw + "\n```"})
    assert call.arguments["content"] == body


def test_prose_with_no_call_is_an_invalid_call():
    with pytest.raises(errors.InvalidCall) as excinfo:
        protocol.parse("J", {"content": "Creo que deberiamos revisar el modulo primero."})
    assert excinfo.value.code == errors.ERROR_FORMAT
    assert "json" in excinfo.value.feedback().lower()


def test_malformed_json_is_an_invalid_call_not_a_crash():
    with pytest.raises(errors.InvalidCall):
        protocol.parse("J", {"content": '```json\n{"tool": "read_file", "arguments":\n```'})


def test_an_unknown_protocol_is_a_harness_defect(ctx):
    from localprog.errors import HarnessInvalid

    with pytest.raises(HarnessInvalid):
        loop.run_loop(
            provider=FakeProvider([]), ctx=ctx, system="S", objective="O", protocol_name="Z"
        )


def test_the_whole_loop_runs_on_protocol_j(ctx):
    """Same loop, same tools, same containment -- only the wire format differs."""
    result = loop.run_loop(
        provider=FakeProvider([
            jcall("read_file", path="calc.py"),
            jcall("edit", path="calc.py", old="a + b", new="a - b"),
            jcall("finish", summary="hecho", status="DONE"),
        ]),
        ctx=ctx, system="S", objective="O", protocol_name="J", max_turns=5,
    )
    assert result.outcome == loop.FINISHED
    assert result.changed_files == ["calc.py"]


def test_containment_is_identical_on_the_escalation_tier(ctx):
    """An escalation tier that could reach around the sandbox would make every
    safety property above it conditional on which model happened to answer."""
    result = loop.run_loop(
        provider=FakeProvider([
            jcall("write_file", path="../escaped.py", content="X = 1\n"),
            jcall("finish", summary="no pude", status="BLOCKED"),
        ]),
        ctx=ctx, system="S", objective="O", protocol_name="J", max_turns=4,
    )
    assert result.outcome == loop.FINISHED
    assert result.changed_files == []
    assert not (ctx.root.parent / "escaped.py").exists()


# ------------------------------------------------------------------ the manual


def test_the_text_manual_documents_every_tool():
    manual = tools.text_manual()
    for name in tools.SPECS:
        assert name in manual, f"{name} missing from the manual"
        assert tools.TOOL_DOC[name][:40] in manual


def test_the_manual_and_the_schema_cannot_disagree():
    """I3. A hand-written manual for the escalation tier would be a second
    source of truth for what a tool is, and it would drift -- the old runner
    advertised a list_dir it did not implement."""
    manual = tools.text_manual()
    assert {f["function"]["name"] for f in tools.native_schema()} == set(tools.SPECS)
    for name, (required, optional) in tools.SPECS.items():
        for param in required + optional:
            assert param in manual, f"{name}.{param} undocumented in the manual"


def test_the_manual_can_be_restricted_like_the_schema():
    manual = tools.text_manual(("read_file", "finish"))
    assert "read_file(" in manual and "finish(" in manual
    assert "replace_lines(" not in manual


def test_the_json_prompt_is_only_added_for_protocol_j(tmp_path):
    ticket = work.Ticket(
        ticket_id="T", repo=tmp_path, objective="x",
        write_scope=("pkg/",), acceptance_tests=("tests/t.py",),
    )
    assert "```json" not in work.system_prompt(ticket, "A")
    assert "HERRAMIENTAS DISPONIBLES" not in work.system_prompt(ticket, "A")
    rendered = work.system_prompt(ticket, "J")
    assert "```json" in rendered
    assert "HERRAMIENTAS DISPONIBLES" in rendered
    assert "replace_lines(" in rendered


# ------------------------------------------------------------------ F-34


def test_an_agent_that_never_edits_is_told_so(ctx):
    """qwen3.5:9b ran the same ticket twice: once it got five of six tests
    green, once it made 25 reads, 10 greps, 4 runs and not one edit in forty
    turns. The harness could see that and never mentioned it."""
    provider = FakeProvider([jcall("read_file", path="calc.py") for _ in range(20)])
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O",
                  protocol_name="J", max_turns=20)
    assert "no has cambiado nada" in str(provider.calls[-1]["messages"])


def test_the_nudge_stops_once_something_is_edited(ctx):
    provider = FakeProvider(
        [jcall("read_file", path="calc.py") for _ in range(13)]
        + [jcall("edit", path="calc.py", old="a + b", new="a - b")]
        + [jcall("read_file", path="calc.py") for _ in range(6)]
    )
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O",
                  protocol_name="J", max_turns=20)
    assert "no has cambiado nada" not in str(provider.calls[-1]["messages"][-1])


def test_no_nudge_before_the_threshold(ctx):
    provider = FakeProvider([jcall("read_file", path="calc.py") for _ in range(4)])
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O",
                  protocol_name="J", max_turns=4)
    assert "no has cambiado nada" not in str(provider.calls[-1]["messages"])


# ------------------------------------------------------------------ the tier


def test_the_luna_tier_declares_its_own_cost_bucket():
    """LOCAL, LUNA and CLAUDE are different budgets at very different prices,
    and adding them up hides which part of the factory is expensive."""
    described = CodexProvider("gpt-5.6-luna", effort="low").describe()
    assert described["model_class"] == "LUNA"
    assert described["provider"] == "codex"


def test_the_luna_prompt_keeps_roles_apart():
    """Without labels a tool result and the agent's own reasoning become the
    same kind of text, and the model starts re-answering itself."""
    rendered = CodexProvider._render([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "pense algo"},
        {"role": "tool", "name": "read_file", "content": "def f(): pass"},
    ])
    assert "SYS" in rendered
    assert "TAREA:" in rendered and "TASK" in rendered
    assert "RESULTADO DE read_file" in rendered
    assert rendered.rstrip().endswith("llamada.")
