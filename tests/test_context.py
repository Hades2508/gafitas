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


def test_the_budget_reserves_what_it_measures_and_no_more():
    """F-53. The reservation used to be a flat 45% of the window, chosen when
    that window was 16k and the schema was a large slice of it. At 32k it held
    back ~15k tokens for a schema costing ~2k -- and the 9B then spent 28 of 40
    turns re-reading files that reservation had discarded, while its prompt
    never passed 17k of an available 32k."""
    schema, system, predict = 8000, 2000, 4096
    budget = transcript.budget_chars(
        32768, num_predict=predict, schema_chars=schema, system_chars=system
    )
    reserved = 32768 - budget / transcript.CHARS_PER_TOKEN
    measured = (schema + system) / transcript.CHARS_PER_TOKEN + predict
    # Everything held back is either measured or the declared safety margin.
    assert measured <= reserved <= measured + transcript.SAFETY_MARGIN_TOKENS + 1


def test_a_bigger_window_gives_proportionally_more_than_it_reserves():
    """The reservation is fixed cost; doubling the window should nearly double
    the usable transcript, not scale it by a constant fraction."""
    args = dict(num_predict=4096, schema_chars=8000, system_chars=2000)
    small = transcript.budget_chars(16384, **args)
    large = transcript.budget_chars(32768, **args)
    assert large > small * 2


def test_the_budget_never_collapses_to_nothing():
    """A window too small to hold a conversation must fail loudly at the
    provider, not quietly by giving the agent amnesia."""
    tiny = transcript.budget_chars(2048, num_predict=4096, schema_chars=8000, system_chars=2000)
    assert tiny == int(transcript.MIN_BUDGET_TOKENS * transcript.CHARS_PER_TOKEN)


def test_an_explicit_reserve_overrides_the_measurement():
    assert transcript.budget_chars(10000, reserve_tokens=1000) == int(
        9000 * transcript.CHARS_PER_TOKEN
    )


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


# ------------------------------------------------------------------ F-25


def test_reading_a_very_large_file_cannot_blow_the_context_window(tmp_path):
    """Found by dogfooding. The agent was asked to add an argument to grep, its
    first move was to read the file that defines it, and localprog/tools.py is
    48k characters against a 16k-token window. Turn 2 came back 400
    exceed_context_size_error. The agent did nothing wrong."""
    big = "\n".join(f"# {'x' * 300}" for _ in range(400))
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "big.py"})
    assert out.ok
    assert len(out.value) <= tools.MAX_TOOL_PAYLOAD_CHARS
    assert "truncado" in out.value
    assert "start=" in out.value, "must say how to read the rest"


def test_a_file_under_the_line_limit_but_over_the_character_limit_is_still_capped(tmp_path):
    """Lines were the only limit, and for source with long lines they are a bad
    proxy: under 2000 lines and still far too big."""
    body = "\n".join("y" * 2000 for _ in range(30))
    (tmp_path / "wide.py").write_text(body, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "wide.py"})
    assert out.ok and len(out.value) <= tools.MAX_TOOL_PAYLOAD_CHARS


def test_an_explicit_range_is_capped_too(tmp_path):
    body = "\n".join("z" * 2000 for _ in range(40))
    (tmp_path / "wide.py").write_text(body, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "wide.py", "start": 1, "end": 40})
    assert out.ok and len(out.value) <= tools.MAX_TOOL_PAYLOAD_CHARS + 200


def test_a_small_file_is_returned_whole(tmp_path):
    (tmp_path / "small.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "small.py"})
    assert out.ok and "truncado" not in out.value and "return 1" in out.value


def test_the_transcript_caps_even_the_most_recent_payload():
    """The budget's hard floor never elides recent turns -- which is right, and
    which leaves exactly one oversized recent result unbounded. One is enough."""
    t = transcript.Transcript(system="S", user="U", keep_turns=3, budget_chars=100_000)
    t.add(transcript.Turn(
        number=1, assistant={"role": "assistant", "content": ""},
        tool_name="read_file", tool_payload="Q" * 200_000,
    ))
    tool_message = [m for m in t.messages() if m.get("role") == "tool"][0]
    assert len(tool_message["content"]) <= transcript.MAX_PAYLOAD_CHARS + 200
    assert "recortado" in tool_message["content"]


def test_a_context_overflow_is_named_and_never_scored():
    """It is the harness's defect -- we built the prompt -- so it must not be
    charged to the model. It stays a ProviderError, which is unscoreable."""
    import urllib.error
    from unittest.mock import patch as mock_patch

    from localprog.errors import ProviderError
    from localprog.provider import OllamaProvider

    body = b'{"error":"exceed_context_size_error: request (16877 tokens)"}'
    error = urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
    error.read = lambda: body  # type: ignore[method-assign]

    with mock_patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ProviderError) as excinfo:
            OllamaProvider("m").chat([{"role": "user", "content": "x"}])
    assert excinfo.value.kind == "CONTEXT_OVERFLOW"

    result = loop.LoopResult(outcome=loop.PROVIDER_ERROR)
    assert result.scoreable is False


def test_legal_tools_cannot_reach_the_frozen_screen():
    """A5. legal_tools makes the declared surface a function of mission state,
    which is right for a mission and wrong for a frozen instrument: the screen's
    surface must not move for ANY reason, including a good one.

    The screen passes declare_tools explicitly, and run_loop honours an explicit
    list over the computed one. This asserts that precedence directly, because
    the alternative -- noticing later that a sealed instrument had quietly
    started asking a different question -- is the failure this whole file exists
    to prevent.
    """
    from localprog import loop, screen

    # A screen-shaped scope would compute a DIFFERENT surface if it were asked.
    computed = tools.legal_tools(tuple(screen.WRITE_SCOPE), ())
    assert set(computed) != set(screen.FROZEN_TOOLS), (
        "if these ever coincide this test proves nothing; pick a scope where "
        "they differ")

    import inspect
    source = inspect.getsource(loop.run_loop)
    assert "declare_tools if declare_tools is not None" in source, (
        "run_loop must prefer an explicit declare_tools over the computed one")


def test_copy_code_is_not_visible_to_the_frozen_screen():
    from localprog import screen

    assert "copy_code" not in screen.FROZEN_TOOLS
    declared = {t["function"]["name"] for t in tools.native_schema(screen.FROZEN_TOOLS)}
    assert "copy_code" not in declared
