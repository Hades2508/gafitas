"""The escalation ladder's decision rule.

The cheap tier answers in seconds for the price of electricity; Luna costs real
money per call. So the only question this module answers is: given what the
cheap tier just produced, is a more expensive one worth paying for?

The distinction that matters is not PASS versus not-PASS. It is between a model
that tried and failed -- where a better model plausibly helps -- and a failure
no model can fix, where escalating means paying to rediscover the same thing.
That second group is what a naive retry-harder ladder gets wrong, and it is the
expensive mistake: it turns every corpus defect and every crashed server into a
paid API call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import route, work  # noqa: E402


class FakeResult:
    def __init__(self, outcome, *, seconds=1.0, usage=None):
        self.outcome = outcome
        self.wall_seconds = seconds
        self.usage = usage or {"calls": 1, "output_tokens": 100}

    def to_dict(self):
        return {"outcome": self.outcome}


def ticket(tmp_path):
    return work.Ticket(
        ticket_id="T", repo=tmp_path, objective="x",
        write_scope=("pkg/",), acceptance_tests=("tests/t.py",),
    )


def ladder(*outcomes):
    """A runner that yields the given outcomes in order, one per tier."""
    remaining = list(outcomes)
    seen: list[tuple[str, str]] = []

    def runner(tkt, model, **kwargs):
        seen.append((model, kwargs.get("protocol", "A")))
        return FakeResult(remaining.pop(0))

    return runner, seen


TIERS = [
    {"tier": route.LOCAL, "model": "local-4b", "protocol": "A"},
    {"tier": route.LUNA, "model": "luna", "protocol": "J"},
]


# ----------------------------------------------------------- the decision


@pytest.mark.parametrize("outcome", sorted(route.SUCCESSFUL))
def test_usable_work_is_never_escalated(outcome):
    """Paying for a second opinion on work that already passed its acceptance
    and its conscience buys nothing."""
    assert not route.should_escalate(outcome)


@pytest.mark.parametrize("outcome", sorted(route.ESCALATABLE))
def test_a_model_that_tried_and_failed_is_escalated(outcome):
    assert route.should_escalate(outcome)


@pytest.mark.parametrize("outcome", sorted(route.TERMINAL))
def test_a_failure_no_model_can_fix_is_not_escalated(outcome):
    """The expensive mistake. A ticket that was not a task, a crashed server
    and a defect in this harness all look like 'not a PASS', and escalating any
    of them means paying Luna to rediscover the same thing -- repeatedly, while
    hiding the real cause."""
    assert not route.should_escalate(outcome)


def test_every_work_outcome_is_classified():
    """A new outcome that nobody classified would silently fall through to
    'do not escalate', which is the quiet-failure direction."""
    every = {
        work.PASS, work.PASS_UNCONFIRMED, work.FAIL, work.BLOCKED_BY_CONSCIENCE,
        work.NON_DISCRIMINATING, work.PROVIDER_ERROR, work.HARNESS_INVALID,
    }
    covered = route.SUCCESSFUL | route.ESCALATABLE | route.TERMINAL
    assert every == covered, f"unclassified: {every ^ covered}"


def test_the_groups_do_not_overlap():
    assert not (route.SUCCESSFUL & route.ESCALATABLE)
    assert not (route.SUCCESSFUL & route.TERMINAL)
    assert not (route.ESCALATABLE & route.TERMINAL)


@pytest.mark.parametrize(
    "outcome",
    sorted(route.SUCCESSFUL | route.ESCALATABLE | route.TERMINAL),
)
def test_every_outcome_has_a_readable_reason(outcome):
    """A routing decision nobody can explain is one nobody can audit."""
    reason = route.explain(outcome)
    assert reason and "no clasificado" not in reason


# ----------------------------------------------------------- the ladder


def test_a_local_pass_never_reaches_luna(tmp_path):
    runner, seen = ladder(work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.PASS
    assert routed.final_tier == route.LOCAL
    assert routed.escalated is False
    assert seen == [("local-4b", "A")], "Luna must not have been called at all"


def test_a_local_failure_escalates_to_luna(tmp_path):
    runner, seen = ladder(work.FAIL, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.PASS
    assert routed.final_tier == route.LUNA
    assert routed.escalated is True
    assert [m for m, _ in seen] == ["local-4b", "luna"]


def test_the_escalated_tier_gets_its_own_protocol(tmp_path):
    """Luna is driven through protocol J because Codex cannot be handed a tool
    schema. Routing must carry that through, not assume every tier is native."""
    runner, seen = ladder(work.FAIL, work.PASS)
    route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert seen == [("local-4b", "A"), ("luna", "J")]


def test_a_non_discriminating_ticket_stops_at_the_cheap_tier(tmp_path):
    """The ticket was not a task. Escalating means paying to discover that
    twice."""
    runner, seen = ladder(work.NON_DISCRIMINATING, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.NON_DISCRIMINATING
    assert len(seen) == 1


def test_a_provider_error_stops_at_the_cheap_tier(tmp_path):
    runner, seen = ladder(work.PROVIDER_ERROR, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.PROVIDER_ERROR
    assert len(seen) == 1, "a broken server is not a reason to spend money"


def test_a_harness_defect_stops_at_the_cheap_tier(tmp_path):
    runner, seen = ladder(work.HARNESS_INVALID, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.HARNESS_INVALID
    assert len(seen) == 1, "escalating our own bug would hide it"


def test_both_tiers_failing_reports_the_last_attempt(tmp_path):
    runner, _ = ladder(work.FAIL, work.FAIL)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.FAIL
    assert len(routed.attempts) == 2


def test_an_unconfirmed_pass_is_kept_rather_than_escalated(tmp_path):
    """PASS_UNCONFIRMED is real work the agent was merely unsure about. Paying
    a stronger model to redo it would be buying confidence, not capability."""
    runner, seen = ladder(work.PASS_UNCONFIRMED, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    assert routed.outcome == work.PASS_UNCONFIRMED
    assert len(seen) == 1


# ----------------------------------------------------------- the accounting


def test_cost_is_reported_per_tier(tmp_path):
    """LOCAL, LUNA and CLAUDE are different budgets at very different prices,
    and one combined number hides the only question that matters: whether the
    expensive tier is doing a small share of the work."""
    runner, _ = ladder(work.FAIL, work.PASS)
    escalated = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    runner, _ = ladder(work.PASS)
    cheap = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)

    summary = route.summarise([escalated, cheap])
    assert summary["tickets"] == 2
    assert summary["solved"] == 2
    assert summary["escalated"] == 1
    assert summary["solved_without_escalation"] == 1
    assert summary["by_tier"][route.LOCAL]["attempts"] == 2
    assert summary["by_tier"][route.LOCAL]["solved"] == 1
    assert summary["by_tier"][route.LUNA]["attempts"] == 1
    assert summary["by_tier"][route.LUNA]["solved"] == 1


def test_a_tier_that_failed_is_not_credited_with_the_solve(tmp_path):
    runner, _ = ladder(work.FAIL, work.PASS)
    routed = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)
    summary = route.summarise([routed])
    assert summary["by_tier"][route.LOCAL]["solved"] == 0
    assert summary["by_tier"][route.LUNA]["solved"] == 1


def test_a_tier_that_reports_no_tokens_is_marked_not_free(tmp_path):
    """The Codex CLI reports no token counts, so LUNA's row comes back with
    zeros -- for the only tier that costs actual money. Zero reads as free,
    and that is exactly the blind spot this project was told not to repeat:
    perfect accounting for the cheap side and none for the expensive one."""
    def runner(tkt, model, **kwargs):
        return FakeResult(work.PASS, usage={"calls": 4})  # no token fields

    routed = route.run_with_ladder(
        ticket(tmp_path),
        tiers=[{"tier": route.LUNA, "model": "luna", "protocol": "J"}],
        runner=runner,
    )
    bucket = route.summarise([routed])["by_tier"][route.LUNA]
    assert bucket["tokens_reported"] is False
    assert bucket["calls"] == 4, "calls and wall time are the cost signal instead"


def test_a_tier_that_does_report_tokens_is_marked_reported(tmp_path):
    def runner(tkt, model, **kwargs):
        return FakeResult(work.PASS, usage={"calls": 4, "input_tokens": 900, "output_tokens": 80})

    routed = route.run_with_ladder(
        ticket(tmp_path),
        tiers=[{"tier": route.LOCAL, "model": "local", "protocol": "A"}],
        runner=runner,
    )
    bucket = route.summarise([routed])["by_tier"][route.LOCAL]
    assert bucket["tokens_reported"] is True
    assert bucket["input_tokens"] == 900


def test_a_dead_provider_is_reported_before_the_batch_runs():
    """F-41. An eight-ticket sweep once produced eight PROVIDER_ERRORs and
    reported '0/8 PASS' because the Ollama server had stopped. The routing was
    right -- a provider error is terminal, so nothing escalated to a paid tier --
    but the operator got a report that reads like eight failures."""
    import urllib.error
    from unittest.mock import patch as mock_patch

    from localprog.provider import OllamaProvider

    provider = OllamaProvider("qwen3:4b", endpoint="http://127.0.0.1:11434/api/chat")
    with mock_patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        complaint = provider.preflight()
    assert complaint and "ollama serve" in complaint


def test_a_missing_model_is_named_before_the_batch_runs():
    import io
    import json as _json
    from unittest.mock import patch as mock_patch

    from localprog.provider import OllamaProvider

    body = _json.dumps({"models": [{"name": "qwen3:4b"}, {"name": "other:7b"}]}).encode()

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with mock_patch("urllib.request.urlopen", return_value=Response(body)):
        assert OllamaProvider("qwen3:4b").preflight() is None
    with mock_patch("urllib.request.urlopen", return_value=Response(body)):
        complaint = OllamaProvider("nope:1b").preflight()
    assert complaint and "no esta instalado" in complaint


# ------------------------------------------------------------------ F-48


def test_a_free_tier_is_retried_before_anyone_is_paid(tmp_path):
    """Measured per ticket, most local failures are sampling rather than a
    ceiling: seven of eight tickets passed at least once in three, and only one
    never did. A second free attempt beats a paid one nearly every time."""
    runner, seen = ladder(work.FAIL, work.FAIL, work.PASS)
    tiers = [
        {"tier": route.LOCAL, "model": "local-4b", "protocol": "A", "attempts": 3},
        {"tier": route.LUNA, "model": "luna", "protocol": "J"},
    ]
    routed = route.run_with_ladder(ticket(tmp_path), tiers=tiers, runner=runner)
    assert routed.outcome == work.PASS
    assert [m for m, _ in seen] == ["local-4b"] * 3, "Luna was never called"
    assert routed.paid is False
    assert routed.local_attempts == 3


def test_retrying_the_same_free_tier_is_not_escalation(tmp_path):
    """Persistence is not escalation. Conflating them would make the cost
    report claim a ticket had been escalated when nobody was paid."""
    runner, _ = ladder(work.FAIL, work.PASS)
    tiers = [{"tier": route.LOCAL, "model": "local-4b", "protocol": "A", "attempts": 2}]
    routed = route.run_with_ladder(ticket(tmp_path), tiers=tiers, runner=runner)
    assert len(routed.attempts) == 2
    assert routed.escalated is False
    assert routed.paid is False


def test_the_free_tiers_are_exhausted_before_a_paid_one(tmp_path):
    runner, seen = ladder(work.FAIL, work.FAIL, work.FAIL, work.FAIL, work.PASS)
    tiers = [
        {"tier": route.LOCAL, "model": "fast", "protocol": "A", "attempts": 3},
        {"tier": route.LOCAL_STRONG, "model": "strong", "protocol": "A", "attempts": 1},
        {"tier": route.LUNA, "model": "luna", "protocol": "J"},
    ]
    routed = route.run_with_ladder(ticket(tmp_path), tiers=tiers, runner=runner)
    assert [m for m, _ in seen] == ["fast", "fast", "fast", "strong", "luna"]
    assert routed.paid is True
    assert routed.local_attempts == 4


def test_a_terminal_outcome_stops_the_retries_too(tmp_path):
    """A ticket that was not a task, or a crashed server, does not become one
    by being asked three times."""
    runner, seen = ladder(work.NON_DISCRIMINATING, work.PASS, work.PASS)
    tiers = [{"tier": route.LOCAL, "model": "local", "protocol": "A", "attempts": 3}]
    routed = route.run_with_ladder(ticket(tmp_path), tiers=tiers, runner=runner)
    assert routed.outcome == work.NON_DISCRIMINATING
    assert len(seen) == 1


def test_a_first_attempt_pass_costs_exactly_one_attempt(tmp_path):
    runner, seen = ladder(work.PASS, work.PASS, work.PASS)
    tiers = [{"tier": route.LOCAL, "model": "local", "protocol": "A", "attempts": 3}]
    routed = route.run_with_ladder(ticket(tmp_path), tiers=tiers, runner=runner)
    assert len(seen) == 1 and routed.local_attempts == 1


def test_the_summary_reports_what_was_paid_for(tmp_path):
    """The number the whole project is judged on. Anything above zero means the
    normal path still depends on somebody's API."""
    runner, _ = ladder(work.FAIL, work.PASS)
    free = route.run_with_ladder(
        ticket(tmp_path),
        tiers=[{"tier": route.LOCAL, "model": "l", "protocol": "A", "attempts": 2}],
        runner=runner,
    )
    runner, _ = ladder(work.FAIL, work.PASS)
    costly = route.run_with_ladder(ticket(tmp_path), tiers=TIERS, runner=runner)

    summary = route.summarise([free, costly])
    assert summary["solved"] == 2
    assert summary["solved_free"] == 1
    assert summary["paid_tickets"] == 1
    assert summary["paid_share"] == 0.5


def test_luna_is_absent_from_the_free_tier_set():
    assert route.LOCAL in route.FREE_TIERS
    assert route.LOCAL_STRONG in route.FREE_TIERS
    assert route.LUNA not in route.FREE_TIERS
