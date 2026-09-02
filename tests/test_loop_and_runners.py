"""The loop's four outcomes, and the two runners driven by a fake provider.

This file is the reason real models stop being the fuzzer: every branch a model
could push the loop down is reachable from a script here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localprog import errors, evidence, loop, screen, step1, telemetry_bridge, tools, workspace
from localprog.provider import FakeProvider

FIX = '    if b == 0:\n        raise ValueError("division by zero")\n    return a / b'


def tc(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def solve_script():
    return [
        tc("read_file", path="calc.py"),
        tc("list_symbols", path="calc.py"),
        tc("edit", path="calc.py", old="    return a / b", new=FIX),
        tc("run_tests"),
        tc("finish", summary="divide lanza ValueError"),
    ]


def run(ctx, script, protocol="A", max_turns=loop.MAX_TURNS):
    return loop.run_loop(
        provider=FakeProvider(script), ctx=ctx, system="S", objective="O",
        protocol_name=protocol, max_turns=max_turns,
    )


# ------------------------------------------------------------------- outcomes


def test_happy_path_finishes_and_is_scoreable(ctx):
    result = run(ctx, solve_script())
    assert result.outcome == loop.FINISHED
    assert result.scoreable and result.tests_green
    assert result.changed_files == ["calc.py"]
    assert result.invalid_calls == 0 and result.tool_errors == 0


def test_budget_exhausted_is_a_result_not_an_error(ctx):
    result = run(ctx, [tc("read_file", path="calc.py")] * 3, max_turns=3)
    assert result.outcome == loop.BUDGET_EXHAUSTED
    assert result.scoreable and result.turns_used == 3


@pytest.mark.parametrize("error", [
    errors.ProviderError("HTTP_STATUS", "500 Internal Server Error", status=500),
    errors.ProviderError("TIMEOUT", "sin respuesta en 120s"),
    errors.ProviderError("BAD_BODY", "la respuesta no es JSON"),
    errors.ProviderError("TRANSPORT", "connection refused"),
])
def test_provider_failure_is_structured_and_not_scoreable(ctx, error):
    """A provider failure that persists across every retry ends the run.

    The error is scripted PROVIDER_RETRIES + 1 times because since F-30 one bad
    response no longer ends anything -- qwen3.5:9b lost thirty turns of real
    work to a single malformed tool call the server rejected.
    """
    attempts = [error] * (loop.PROVIDER_RETRIES + 1)
    result = run(ctx, [tc("read_file", path="calc.py"), *attempts])
    assert result.outcome == loop.PROVIDER_ERROR
    assert result.scoreable is False
    assert result.provider_error["kind"] == error.kind


def test_a_transient_provider_failure_does_not_end_the_run(ctx):
    """F-30, the case that matters. The server rejects one malformed tool call;
    asking again gets a good one, because the failure was sampling noise in a
    single generation rather than anything wrong with the machine."""
    from localprog.errors import ProviderError

    script = solve_script()
    script.insert(1, ProviderError("HTTP_STATUS", "500: XML syntax error", status=500))
    result = run(ctx, script, max_turns=8)
    assert result.outcome == loop.FINISHED
    assert result.provider_retries == 1
    assert result.tests_green


def test_a_context_overflow_is_never_retried(ctx):
    """It is deterministic: the same prompt overflows the same window every
    time. Retrying only delays a defect that is ours to fix."""
    from localprog.errors import ProviderError

    result = run(ctx, [ProviderError("CONTEXT_OVERFLOW", "400: too big", status=400)])
    assert result.outcome == loop.PROVIDER_ERROR
    assert result.provider_retries == 0


def test_provider_failure_on_the_first_turn_still_records(ctx):
    result = run(ctx, [errors.ProviderError("HTTP_STATUS", "500", status=500)])
    assert result.outcome == loop.PROVIDER_ERROR and result.turns_used == 1


def test_harness_defect_voids_the_run(ctx, monkeypatch):
    def boom(*a, **k):
        raise ZeroDivisionError("simulated")

    monkeypatch.setitem(tools._IMPL, "read_file", boom)
    result = run(ctx, [tc("read_file", path="calc.py")])
    assert result.outcome == loop.HARNESS_INVALID
    assert result.scoreable is False
    assert "UNHANDLED_TOOL_ERROR" in result.harness_invalid["detail"]


# ------------------------------------------------- tool errors keep it alive


def test_a_tool_error_feeds_back_and_the_loop_continues(ctx):
    script = [tc("edit", path="calc.py", old="NO EXISTE", new="x")] + solve_script()
    result = run(ctx, script)
    assert result.outcome == loop.FINISHED
    assert result.tool_errors == 1 and result.invalid_calls == 0
    assert result.tests_green


def test_the_error_text_actually_reaches_the_model(ctx):
    provider = FakeProvider([tc("read_file", path="nope.py"), tc("read_file", path="calc.py")])
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=2)
    delivered = json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
    assert errors.ERROR_FILE_NOT_FOUND in delivered


def test_invalid_calls_are_counted_separately_from_tool_errors(ctx):
    result = run(ctx, [
        {"content": "solo texto, sin llamada"},        # invalid
        tc("delete_file", path="."),                    # invalid (unknown tool)
        tc("read_file", path="nope.py"),                # tool error
        tc("read_file", path="calc.py"),
    ], max_turns=4)
    assert result.invalid_calls == 2 and result.tool_errors == 1


def test_repeated_identical_calls_are_counted_as_a_loop(ctx):
    result = run(ctx, [tc("read_file", path="calc.py")] * 4, max_turns=4)
    assert result.loops == 1


def test_double_finish_without_changes_is_accepted(ctx):
    result = run(ctx, [tc("finish", summary="nada"), tc("finish", summary="nada")], max_turns=2)
    assert result.outcome == loop.FINISHED and result.finished_without_changes


def test_single_finish_without_changes_is_not_accepted(ctx):
    result = run(ctx, [tc("finish", summary="nada")], max_turns=1)
    assert result.outcome == loop.BUDGET_EXHAUSTED and result.tool_errors == 1


def test_protocol_b_drives_the_same_loop(ctx):
    script = [
        {"content": 'LLAMADA: read_file(path="calc.py")'},
        {"content": 'LLAMADA: grep(pattern="divide")'},
        {"content": 'LLAMADA: edit(path="calc.py", old="    return a / b", new=' + json.dumps(FIX) + ')'},
        {"content": "LLAMADA: run_tests()"},
        {"content": 'LLAMADA: finish(summary="hecho")'},
    ]
    result = run(ctx, script, protocol="B")
    assert result.outcome == loop.FINISHED and result.tests_green


# ---------------------------------------------------------------- elision


def test_old_bulky_results_are_elided_but_calls_survive(ctx, repo):
    (repo / "big.py").write_text("# padding\n" * 400, encoding="utf-8")
    provider = FakeProvider([tc("read_file", path="big.py")] * 6)
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=6)
    sent = provider.calls[-1]["messages"]
    blob = json.dumps(sent, ensure_ascii=False)
    assert "elidido" in blob
    assert sum(1 for m in sent if m.get("role") == "tool") == 5


def test_errors_are_never_elided(ctx):
    provider = FakeProvider([tc("read_file", path="nope.py")] + [tc("read_file", path="calc.py")] * 5)
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=6)
    blob = json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
    assert errors.ERROR_FILE_NOT_FOUND in blob


# ------------------------------------------------------------------- screen


def test_screen_run_scores_five_of_five(tmp_path):
    record = screen.run_once(
        "fake", "A", 1,
        provider_factory=lambda name: FakeProvider(solve_script()),
        out_dir=tmp_path, base_dir=tmp_path,
    )
    assert record.steps == [True] * 5 and record.scoreable
    assert (tmp_path / "transcripts" / "fake_A_1.json").is_file()


def test_screen_classification_levels(tmp_path):
    def make(steps, scoreable=True, turns=5, invalid=0):
        r = screen.RunRecord(model="m", protocol="A", run=1)
        r.steps, r.scoreable, r.turnos_usados, r.llamadas_invalidas = steps, scoreable, turns, invalid
        return r

    full = [make([True] * 5) for _ in range(3)]
    assert screen.classify(full) == screen.PASS_CLEAN
    assert screen.classify([make([True] * 5, turns=10) for _ in range(3)]) == screen.PASS_NOISY
    assert screen.classify(full[:2] + [make([True] * 4 + [False])]) == screen.PASS_NOISY
    assert screen.classify([make([True, True, True, False, False])] * 3) == screen.PARTIAL
    assert screen.classify([make([True, True, False, False, False])] * 3) == screen.FAIL
    assert screen.classify([make([False] * 5, invalid=6, turns=6)] * 3) == screen.INCOMPATIBLE
    assert screen.classify([make([False] * 5, scoreable=False)] * 3) == screen.VOID


def test_screen_provider_failure_does_not_score_the_model(tmp_path):
    record = screen.run_once(
        "fake", "A", 1,
        provider_factory=lambda name: FakeProvider([errors.ProviderError("HTTP_STATUS", "500", status=500)]),
        out_dir=tmp_path, base_dir=tmp_path,
    )
    assert record.scoreable is False and record.steps == [False] * 5
    assert record.workspace["preserved"] is True  # evidence kept


def test_screen_prompt_is_the_frozen_text():
    text = screen.system_prompt("A")
    assert "Tienes 12 turnos." in text and 'Solo puedes escribir en: ["calc.py"]' in text
    assert screen.PROTOCOL_B_SUFFIX not in text
    assert screen.PROTOCOL_B_SUFFIX in screen.system_prompt("B")


def test_screen_repo_is_discriminating(tmp_path):
    """PRE fails, the historical fix passes. Verified, not assumed."""
    box = workspace.synthesize(screen.SCREEN_FILES, base_dir=tmp_path)
    try:
        ctx = tools.ToolContext(root=box.path, write_scope=("calc.py",),
                                acceptance_tests=("test_calc.py",))
        assert tools.dispatch(ctx, "run_tests", {}).value["passed"] is False
        tools.dispatch(ctx, "edit", {"path": "calc.py", "old": "    return a / b", "new": FIX})
        assert tools.dispatch(ctx, "run_tests", {}).value["passed"] is True
    finally:
        box.dispose(preserve=False)


# -------------------------------------------------------------------- step1


def test_malformed_mission_is_harness_invalid_not_a_model_failure(tmp_path):
    """The defect that voided four subtasks in the multifile campaign."""
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({
        "mission_id": "x", "repo": str(tmp_path), "objective": "o",
        "read_scope": ["a.py"], "write_scope": ["a.py"], "acceptance_tests": [],
    }), encoding="utf-8")
    with pytest.raises(errors.HarnessInvalid) as excinfo:
        step1.load_mission(bad)
    assert "acceptance_tests" in str(excinfo.value)


def test_mission_absolute_scopes_are_relativised(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({
        "mission_id": "x", "repo": str(tmp_path), "objective": "o",
        "read_scope": [str(tmp_path / "a.py")], "write_scope": ["a.py"],
        "acceptance_tests": ["t.py"],
    }), encoding="utf-8")
    assert step1.load_mission(path).read_scope == ("a.py",)


def test_step1_runs_a_mission_and_takes_its_own_verdict(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(screen.CALC_PY, encoding="utf-8", newline="\n")
    (repo / "test_calc.py").write_text(screen.TEST_CALC_PY, encoding="utf-8", newline="\n")
    mission_path = tmp_path / "m.json"
    mission_path.write_text(json.dumps({
        "mission_id": "toy", "repo": str(repo), "objective": "arregla divide",
        "read_scope": ["calc.py"], "write_scope": ["calc.py"],
        "acceptance_tests": ["test_calc.py"],
    }), encoding="utf-8")

    record = step1.run_mission(
        step1.load_mission(mission_path), "fake",
        provider_factory=lambda name: FakeProvider(solve_script()),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert record.tester_pass is True and record.out_of_scope_writes == 0
    assert record.workspace["kind"] == "COPY"  # not a git repo, recorded honestly
    assert (tmp_path / "out" / "missions" / "toy_fake_A.json").is_file()
    assert (repo / "calc.py").read_text(encoding="utf-8") == screen.CALC_PY  # origin untouched


def test_telemetry_duplicate_is_recorded_not_raised(tmp_path):
    db = tmp_path / "t.sqlite3"
    telemetry = telemetry_bridge.open_telemetry("RUN", db)
    telemetry.start("m1", repo="r", objective="o")
    telemetry.start("m1", repo="r", objective="o")  # would raise DuplicateMissionError
    assert any("DUPLICATE_MISSION_SKIPPED" in note for note in telemetry.notes)
    telemetry.finish("m1", outcome="FINISHED")
    telemetry.close()


def test_telemetry_unavailable_never_stops_a_run(tmp_path):
    telemetry = telemetry_bridge.open_telemetry("RUN", tmp_path / "no" / "such" / "dir" / "t.sqlite3")
    assert telemetry.available is False
    telemetry.start("m", repo="r", objective="o")
    telemetry.finish("m", outcome="FINISHED")


# ----------------------------------------------------------------- evidence


def test_harness_invalid_notice_is_written(tmp_path):
    path = evidence.harness_invalid_notice(
        tmp_path, runs=[{"label": "m/A/run1", "harness_invalid": {"detail": "boom"}}]
    )
    assert "HARNESS_INVALID" in path.read_text(encoding="utf-8")
    assert "boom" in path.read_text(encoding="utf-8")


def test_workspace_is_preserved_on_failure_and_removed_on_success():
    box = workspace.synthesize({"a.py": "x = 1\n"})
    box.dispose(preserve=True)
    assert box.path.exists() and box.preserved
    box2 = workspace.synthesize({"a.py": "x = 1\n"})
    box2.dispose(preserve=False)
    assert not box2.path.exists()
    import shutil
    shutil.rmtree(box.path, ignore_errors=True)


def test_a_repeated_identical_error_is_named(ctx):
    """F-43. ga04 received the same refusal ten times and acted on none of
    them. An identical error is the same fact as an identical result -- and
    more urgent, because it means feedback already arriving is not reaching the
    agent's decisions."""
    provider = FakeProvider([tc("edit", path="calc.py", old="NOPE", new="x")] * 5)
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=5)
    delivered = str(provider.calls[-1]["messages"])
    assert "vez seguida" in delivered
    assert "Cambia de enfoque" in delivered


def test_two_identical_errors_are_not_yet_nagged(ctx):
    """By the third the agent has had the same correction twice. Before that,
    the ordinary feedback deserves a chance to work."""
    provider = FakeProvider([tc("edit", path="calc.py", old="NOPE", new="x")] * 2)
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=2)
    assert "vez seguida" not in str(provider.calls[-1]["messages"])


def test_different_errors_do_not_accumulate(ctx):
    provider = FakeProvider([
        tc("edit", path="calc.py", old="NOPE", new="x"),
        tc("read_file", path="missing.py"),
        tc("edit", path="calc.py", old="NOPE", new="x"),
    ])
    loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O", max_turns=3)
    assert "vez seguida" not in str(provider.calls[-1]["messages"])
