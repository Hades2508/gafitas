"""Regressions for the ga04 wall: F-15, F-16, F-17, F-18.

ga04 is the run that made these necessary. Turns 1-12 were real programming --
it read the spec, wrote the module, ran the tests, edited three times, ran them
again, and got 14 of 16 tests green from an empty file. Turn 13 produced no tool
call. So did turns 14 through 40: twenty-eight consecutive dead turns, 61k output
tokens, thirteen minutes.

It was not the model giving up. Every failed turn appended the model's whole
reply to the transcript, so failing made the prompt bigger, which made failing
likelier. Once the prompt passed num_ctx the server began truncating from the
front -- where the system prompt and the tool definitions are -- and the model
stopped being able to see that tools existed at all. Nothing in the loop was
watching for "the agent has stopped acting", so it asked twenty-eight more times.

Every test below pins one link in that chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import loop, tools, transcript  # noqa: E402
from localprog.provider import FakeProvider  # noqa: E402


def tc(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def prose(text="Voy a pensar en voz alta sobre este problema. " * 40):
    """A reply with no tool call -- exactly what turns 13-40 of ga04 were."""
    return {"content": text}


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("calc.py",))


# ------------------------------------------------------------------ F-15


def test_a_stalled_agent_stops_the_run_instead_of_burning_the_budget(ctx):
    """The headline regression. Three dead turns, not twenty-eight."""
    result = loop.run_loop(
        provider=FakeProvider([prose()] * 40), ctx=ctx,
        system="S", objective="O", max_turns=40,
    )
    assert result.outcome == loop.STALLED
    assert result.turns_used == loop.STALL_THRESHOLD
    assert result.stalled_after == loop.STALL_THRESHOLD


def test_a_single_bad_reply_is_not_a_stall(ctx):
    """One malformed answer is ordinary and the feedback usually fixes it.
    Stopping on the first would throw away runs that were about to recover."""
    result = loop.run_loop(
        provider=FakeProvider([
            prose(),
            tc("read_file", path="calc.py"),
            tc("edit", path="calc.py", old="a + b", new="a - b"),
            tc("finish", summary="hecho"),
        ]),
        ctx=ctx, system="S", objective="O", max_turns=10,
    )
    assert result.outcome == loop.FINISHED
    assert result.invalid_calls == 1


def test_recovering_resets_the_stall_counter(ctx):
    """Two dead turns, a real call, two more dead turns: not a stall."""
    result = loop.run_loop(
        provider=FakeProvider([
            prose(), prose(),
            tc("read_file", path="calc.py"),
            prose(), prose(),
            tc("edit", path="calc.py", old="a + b", new="a - b"),
            tc("finish", summary="hecho"),
        ]),
        ctx=ctx, system="S", objective="O", max_turns=10,
    )
    assert result.outcome == loop.FINISHED


def test_a_stalled_run_is_still_scoreable(ctx):
    """A stall is a RESULT -- the agent really did fail to act. It is not a
    provider fault and not a harness defect, so it stays in the numbers."""
    result = loop.run_loop(
        provider=FakeProvider([prose()] * 10), ctx=ctx,
        system="S", objective="O", max_turns=10,
    )
    assert result.outcome == loop.STALLED
    assert result.scoreable is True


# ------------------------------------------------------------------ F-16


def test_a_dead_turn_does_not_grow_the_context_without_bound():
    """The runaway itself. An unbounded correction turn makes the next failure
    more likely, which is how three dead turns became twenty-eight."""
    huge = "x" * 50_000
    t = transcript.Transcript(system="S", user="U")
    t.add_correction({"role": "assistant", "content": huge}, "ERROR_NO_TOOL_CALL")
    rendered = t.messages()
    assistant = [m for m in rendered if m.get("role") == "assistant"][0]
    assert len(assistant["content"]) < 1200
    assert "elididos" in assistant["content"]


def test_a_short_reply_is_kept_whole():
    t = transcript.Transcript(system="S", user="U")
    t.add_correction(
        {"role": "assistant", "content": "perdona, ahora llamo a la herramienta"}, "E"
    )
    assistant = [m for m in t.messages() if m.get("role") == "assistant"][0]
    assert assistant["content"] == "perdona, ahora llamo a la herramienta"


def test_the_feedback_still_reaches_the_model_after_truncation():
    """Truncating the reply must not truncate the correction itself -- the
    feedback is the only thing that can get the agent unstuck."""
    t = transcript.Transcript(system="S", user="U")
    t.add_correction(
        {"role": "assistant", "content": "y" * 50_000},
        "ERROR_NO_TOOL_CALL: llama a una herramienta",
    )
    text = " ".join(str(m.get("content")) for m in t.messages())
    assert "ERROR_NO_TOOL_CALL: llama a una herramienta" in text


# ------------------------------------------------------------------ F-17


def test_a_huge_tool_call_argument_is_summarised_once_it_is_old():
    """write_file created explorer/scope.py with 5 KB of content, and that
    content was echoed back in the assistant turn on every later turn. The
    CALL must survive; the payload must not."""
    body = "def f():\n    return 1\n" * 400
    t = transcript.Transcript(
        system="S", user="U", keep_turns=2, budget_chars=2000,
    )
    t.add(transcript.Turn(
        number=1,
        assistant={"content": "", "tool_calls": [
            {"function": {"name": "write_file",
                          "arguments": {"path": "pkg/big.py", "content": body}}}
        ]},
        tool_name="write_file", tool_payload="ok",
    ))
    for n in range(2, 8):
        t.add(transcript.Turn(
            number=n,
            assistant={"content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
            ]},
            tool_name="read_file", tool_payload="x" * 100,
        ))
    rendered = str(t.messages())
    assert body not in rendered, "the 5 KB payload is still being echoed"
    assert "write_file" in rendered, "the decision trace must survive"
    assert "pkg/big.py" in rendered, "which file it wrote must survive"


def test_a_small_argument_is_left_alone():
    t = transcript.Transcript(system="S", user="U", budget_chars=100_000)
    t.add(transcript.Turn(
        number=1,
        assistant={"content": "", "tool_calls": [
            {"function": {"name": "edit",
                          "arguments": {"path": "a.py", "old": "x", "new": "y"}}}
        ]},
        tool_name="edit", tool_payload="ok",
    ))
    assert '"old": ' in str(t.messages()).replace("'", '"')


# ------------------------------------------------------------------ F-18


def test_the_transcript_is_held_under_its_character_budget():
    """Age is a poor proxy for size: eight recent turns can be larger than
    twenty old ones. The budget is what actually keeps the prompt inside
    num_ctx, and therefore what keeps the system prompt from being truncated
    away by the server."""
    budget = 6000
    t = transcript.Transcript(
        system="S" * 200, user="U" * 200, keep_turns=3,
        elide_over_chars=100_000,  # age rule disabled, so only the budget acts
        budget_chars=budget,
    )
    for n in range(1, 30):
        t.add(transcript.Turn(
            number=n,
            assistant={"content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": f"f{n}.py"}}}
            ]},
            tool_name="read_file", tool_payload="R" * 3000,
        ))
    rendered = t.messages()
    size = sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
               for m in rendered)

    # The budget is a soft target with a HARD FLOOR: the last keep_turns are
    # never elided, so a transcript whose recent turns alone exceed the budget
    # cannot reach it. That floor is deliberate -- a budget that could delete
    # what the agent just did would cure the context wall by causing amnesia.
    floor = 3 * 3000
    assert size <= budget + floor, f"transcript is {size} chars (budget {budget}, floor {floor})"

    unbounded = transcript.Transcript(
        system="S" * 200, user="U" * 200, keep_turns=3,
        elide_over_chars=100_000, budget_chars=0,
    )
    unbounded.turns = list(t.turns)
    naive = sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
                for m in unbounded.messages())
    assert size < naive / 3, f"budget barely helped: {size} vs {naive} unbounded"
    assert t.elisions > 0


def test_the_last_turns_are_never_elided_by_the_budget():
    """Whatever else goes, the agent must still be able to see what it just did
    -- otherwise the budget cures the context wall by causing amnesia."""
    t = transcript.Transcript(
        system="S", user="U", keep_turns=3, elide_over_chars=100_000, budget_chars=500,
    )
    for n in range(1, 12):
        t.add(transcript.Turn(
            number=n, assistant={"content": ""},
            tool_name="read_file", tool_payload=f"CONTENIDO-{n}-" + "z" * 2000,
        ))
    rendered = str(t.messages())
    assert "CONTENIDO-11-" in rendered
    assert "CONTENIDO-10-" in rendered
    assert "CONTENIDO-1-z" not in rendered


def test_budget_zero_disables_the_second_pass_entirely():
    """The frozen screen must keep behaving exactly as it did (I13)."""
    t = transcript.Transcript(system="S", user="U", keep_turns=3, budget_chars=0)
    for n in range(1, 10):
        t.add(transcript.Turn(
            number=n, assistant={"content": "A" * 5000},
            tool_name="read_file", tool_payload="B" * 300,
        ))
    rendered = str(t.messages())
    assert rendered.count("A" * 5000) == 9


def test_budget_chars_scales_with_the_context_window():
    small = transcript.budget_chars(8192)
    large = transcript.budget_chars(32768)
    assert abs(large - small * 4) <= 4  # integer truncation, nothing more
    # And it must leave real headroom for the schema, the prompt and the reply.
    assert small < 8192 * transcript.CHARS_PER_TOKEN


# --------------------------------------------- frozen instruments (F-20/F-21)


def test_the_frozen_screen_declares_exactly_the_seven_tools_its_prompt_names():
    """A frozen instrument that changes when unrelated code changes is not
    frozen. Adding list_dir and run must not silently alter what SCREEN_V0
    asks, or its numbers stop being comparable with the runs already sealed."""
    from localprog import screen

    declared = {t["function"]["name"] for t in tools.native_schema(screen.FROZEN_TOOLS)}
    assert declared == set(screen.FROZEN_TOOLS)
    assert len(declared) == 7
    assert "list_dir" not in declared and "run" not in declared


def test_the_screen_prompt_and_its_declared_tools_agree():
    """The prompt is contract section D.3 verbatim. Whatever it names is what
    must be declared -- that correspondence is the thing F-20 broke."""
    from localprog import screen

    for name in screen.FROZEN_TOOLS:
        assert name in screen.SYSTEM_PROMPT, f"{name} declared but not in the D.3 prompt"


def test_restricting_the_schema_does_not_restrict_dispatch(tmp_path):
    """The screen declares seven; the harness still knows nine. A model that
    reaches for an undeclared tool gets ERROR_UNKNOWN_TOOL -- the same answer it
    would have received before those tools were written."""
    from localprog import errors, screen

    ctx = tools.ToolContext(root=tmp_path)
    assert "list_dir" not in screen.FROZEN_TOOLS
    out = tools.dispatch(ctx, "list_dir", {})
    assert out.ok or out.code != errors.ERROR_UNKNOWN_TOOL  # dispatch still has it


def test_asking_the_schema_for_a_tool_that_does_not_exist_is_a_harness_defect():
    from localprog.errors import HarnessInvalid

    with pytest.raises(HarnessInvalid):
        tools.native_schema(("read_file", "teleport"))


def test_step1_refuses_a_mission_whose_acceptance_already_passes(tmp_path):
    """F-21. step1's own corpus was entirely non-discriminating and nothing in
    it could tell. It can now, and it refuses instead of minting another
    invalid number."""
    import subprocess

    from localprog import step1
    from localprog.provider import FakeProvider

    root = tmp_path / "solved"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (root / "tests" / "test_m.py").write_text(
        "from pkg.m import f\n\n\ndef test_f():\n    assert f() == 1\n", encoding="utf-8"
    )
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=str(root), capture_output=True)

    mission = step1.Mission(
        mission_id="solved", repo=root, objective="cambia f() para que devuelva 2",
        read_scope=("pkg/m.py",), write_scope=("pkg/m.py",),
        acceptance_tests=("tests/test_m.py",),
    )
    provider = FakeProvider([tc("finish", summary="x")])
    record = step1.run_mission(
        mission, "fake", provider_factory=lambda _m: provider, base_dir=tmp_path
    )
    assert record.outcome == "NON_DISCRIMINATING"
    assert record.discriminating is False
    assert record.tester_pass is None
    assert provider.calls == []
