"""Routing: try the cheap tier, escalate only on evidence that it failed.

The economics the whole project exists for. A local 4B answers in seconds and
costs nothing but electricity; Luna costs real money per call; Claude costs the
most and is needed least often. So the rule is: send everything to the cheapest
tier, and pay for a more expensive one only when the cheap one has demonstrably
failed on that specific ticket.

WHAT COUNTS AS EVIDENCE OF FAILURE
-----------------------------------
Not every non-PASS is worth escalating, and this distinction is the whole
design. Three groups, three different answers:

  ESCALATE      the agent tried and did not manage it
                FAIL, STALLED, BLOCKED_BY_CONSCIENCE
                A better model plausibly does better. This is what the ladder
                is for.

  STOP, KEEP    the work is usable
                PASS, PASS_UNCONFIRMED
                Paying for a second opinion on work that already passed its
                acceptance and its conscience is buying nothing.

  STOP, REFUSE  the failure is not the model's and a better one cannot help
                NON_DISCRIMINATING  the ticket was not a task. Escalating means
                                    paying Luna to also discover there is
                                    nothing to do.
                PROVIDER_ERROR      the machine broke. Fix the machine.
                HARNESS_INVALID     we broke. Fix us.

That last group is the one a naive "retry harder" ladder gets wrong, and it is
the expensive mistake: it turns every corpus defect and every crashed server
into a paid API call, repeatedly, while hiding the actual cause.

WHY THIS IS NOT AN ORCHESTRATOR
--------------------------------
It is one function, and its inputs are outcomes that already exist. There is no
scoring model, no difficulty estimator and no per-task policy, because none of
those can be built honestly yet -- there is not enough evidence about which
tickets need which tier, and inventing a predictor before the data exists is how
you get a system whose routing decisions nobody can explain. Start with rules
you can read, and let the ledger accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import work

LOCAL = "LOCAL"
LOCAL_STRONG = "LOCAL_STRONG"
LUNA = "LUNA"

#: Tiers that cost nothing but electricity. The point of the whole exercise is
#: that the normal path contains only these.
FREE_TIERS = frozenset({LOCAL, LOCAL_STRONG})

#: The agent tried and did not manage it. A stronger model may.
ESCALATABLE = frozenset({work.FAIL, work.BLOCKED_BY_CONSCIENCE})

#: Usable work. Stop and keep it. CANDIDATE is included because retrying
#: it would mean discarding a real change on the grounds that we could
#: not check it -- and we could not check the next one either.
SUCCESSFUL = frozenset({work.PASS, work.PASS_UNCONFIRMED, work.CANDIDATE})

#: Nobody's model can fix these, so spending on a better one is waste.
TERMINAL = frozenset({work.NON_DISCRIMINATING, work.PROVIDER_ERROR,
                      work.HARNESS_INVALID, work.ACCEPTANCE_UNUSABLE})

#: The ticket was not a task. No tier, free or paid, changes that -- asking
#: again is asking the same question.
ABANDON = frozenset({work.NON_DISCRIMINATING})

#: The machine broke, or we did. Never a reason to PAY for a better model --
#: but every reason to try again on a tier that costs nothing (F-51). A
#: transient 500 on the third local attempt once ended a whole run, and the
#: free strong tier below it was never reached. "Do not pay for infrastructure
#: failure" is right; "give up entirely" was not what that meant.
INFRASTRUCTURE = frozenset({work.PROVIDER_ERROR, work.HARNESS_INVALID,
                            # F-168. The acceptance suite could not be run. A
                            # better model cannot start a pytest that will not
                            # start, so this must never buy a paid tier -- but
                            # it is exactly the sort of thing that is transient,
                            # so a free retry is right.
                            work.ACCEPTANCE_UNUSABLE})


@dataclass
class Attempt:
    tier: str
    model: str
    outcome: str
    wall_seconds: float
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier, "model": self.model, "outcome": self.outcome,
            "wall_seconds": round(self.wall_seconds, 2), "usage": self.usage,
        }


@dataclass
class Routed:
    """What the ladder did for one ticket, and what it cost on the way."""

    ticket_id: str
    attempts: list[Attempt] = field(default_factory=list)
    result: Any = None

    @property
    def outcome(self) -> str:
        return self.result.outcome if self.result else "NOT_RUN"

    @property
    def final_tier(self) -> str:
        return self.attempts[-1].tier if self.attempts else "NONE"

    @property
    def escalated(self) -> bool:
        """Whether the work left the tier it started on.

        Not simply 'more than one attempt': retrying the same free tier is not
        escalation, it is persistence, and conflating the two would make the
        cost report say a ticket had been escalated when nobody was paid.
        """
        return len({a.tier for a in self.attempts}) > 1

    @property
    def local_attempts(self) -> int:
        return sum(1 for a in self.attempts if a.tier in FREE_TIERS)

    @property
    def paid(self) -> bool:
        return any(a.tier not in FREE_TIERS for a in self.attempts)

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "outcome": self.outcome,
            "final_tier": self.final_tier,
            "escalated": self.escalated,
            "local_attempts": self.local_attempts,
            "paid": self.paid,
            "attempts": [a.to_dict() for a in self.attempts],
            "record": self.result.to_dict() if self.result else None,
        }


def should_escalate(outcome: str) -> bool:
    """Whether a stronger model is worth paying for, given this outcome."""
    return outcome in ESCALATABLE


def explain(outcome: str) -> str:
    if outcome == work.CANDIDATE:
        return ("hay un candidato real; quien lo juzga es externo, y otro intento "
                "no lo haria mas comprobable.")
    if outcome in SUCCESSFUL:
        return "el trabajo sirve; no hay nada que ganar pagando otro modelo."
    if outcome in ESCALATABLE:
        return "el modelo lo intento y no salio; un modelo mejor puede."
    if outcome == work.NON_DISCRIMINATING:
        return "el ticket no era una tarea; escalar seria pagar por descubrir lo mismo."
    if outcome == work.PROVIDER_ERROR:
        return "fallo la infraestructura, no el modelo; arreglar la maquina, no pagar mas."
    if outcome == work.HARNESS_INVALID:
        return "defecto de este arnes; escalarlo lo esconderia."
    if outcome == work.ACCEPTANCE_UNUSABLE:
        return ("la aceptacion no se pudo ejecutar: no sabemos si el trabajo "
                "estaba bien, asi que no es un fallo del modelo y pagar otro "
                "no responderia la pregunta.")
    return f"resultado no clasificado: {outcome}"


def run_with_ladder(
    ticket: work.Ticket,
    *,
    tiers: list[dict],
    runner: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Routed:
    """Try each tier in order until one succeeds or the failure is terminal.

    ``tiers`` is a list of ``{"tier", "model", "protocol", "provider_factory"}``
    dicts, cheapest first. Each is tried on a FRESH workspace -- the escalated
    tier must not inherit a half-finished edit from the one below it, because
    then its result measures the pair rather than either, and a conscience
    signal firing on the wreckage of attempt one would be charged to attempt
    two.
    """
    runner = runner or work.run_ticket
    routed = Routed(ticket_id=ticket.ticket_id)

    for step in tiers:
        # F-48: a tier may be tried more than once before moving up. Measured
        # per ticket, most local failures are sampling rather than a ceiling --
        # seven of eight tickets passed at least once in three, and only one
        # never did. A second free attempt beats a paid one nearly every time.
        free = step["tier"] in FREE_TIERS
        for _ in range(max(1, int(step.get("attempts", 1)))):
            result = runner(
                ticket, step["model"],
                provider_factory=step.get("provider_factory"),
                protocol=step.get("protocol", "A"),
                **kwargs,
            )
            routed.attempts.append(Attempt(
                tier=step["tier"], model=step["model"], outcome=result.outcome,
                wall_seconds=result.wall_seconds, usage=dict(result.usage),
            ))
            routed.result = result
            if result.outcome in ABANDON:
                return routed
            # F-51: on a free tier an infrastructure failure is worth another
            # go -- it costs nothing and the attempt never really happened.
            if result.outcome in INFRASTRUCTURE and free:
                continue
            if not should_escalate(result.outcome):
                break
        outcome = routed.result.outcome
        if outcome in ABANDON:
            break
        # Moving UP costs money eventually, so infrastructure failures stop
        # here rather than buying a paid opinion on a broken server.
        if outcome in INFRASTRUCTURE and not free:
            break
        if outcome in INFRASTRUCTURE and free:
            # Every free attempt was eaten by the same fault. The next tier is
            # only worth trying if it is also free.
            if all(t["tier"] in FREE_TIERS for t in tiers[tiers.index(step) + 1:][:1]):
                continue
            break
        if not should_escalate(outcome):
            break
    return routed


def summarise(routed_list: list[Routed]) -> dict:
    """Where the work got done and what each tier cost.

    Split by tier on purpose: LOCAL, LUNA and CLAUDE are different budgets at
    very different prices, and one combined number hides the only question that
    matters -- whether the expensive tier is doing a small share of the work.
    """
    by_tier: dict[str, dict] = {}
    for routed in routed_list:
        for attempt in routed.attempts:
            bucket = by_tier.setdefault(attempt.tier, {
                "attempts": 0, "solved": 0, "calls": 0,
                "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0.0,
                #: Set true only once some attempt actually reports tokens.
                #: The Codex CLI reports none, so LUNA stays false and its zeros
                #: must not be read as free -- calls and wall time are its only
                #: cost signal.
                "tokens_reported": False,
            })
            bucket["attempts"] += 1
            bucket["wall_seconds"] = round(bucket["wall_seconds"] + attempt.wall_seconds, 1)
            for key in ("calls", "input_tokens", "output_tokens"):
                bucket[key] += int(attempt.usage.get(key, 0) or 0)
            # A tier whose provider never reports token counts must say so.
            # Zero reads as "this tier was free", which is the opposite of true
            # for the only tier that costs money -- and it is exactly the blind
            # spot this project was told not to reproduce: perfect accounting
            # for the cheap side and none for the expensive one.
            if int(attempt.usage.get("input_tokens", 0) or 0) > 0:
                bucket["tokens_reported"] = True
            if attempt is routed.attempts[-1] and routed.outcome in SUCCESSFUL:
                bucket["solved"] += 1

    solved = [r for r in routed_list if r.outcome in SUCCESSFUL]
    paid = [r for r in routed_list if r.paid]
    return {
        "tickets": len(routed_list),
        "solved": len(solved),
        "escalated": sum(1 for r in routed_list if r.escalated),
        "solved_without_escalation": sum(1 for r in solved if not r.escalated),
        # The number the whole project is judged on. Anything above zero means
        # the normal path still depends on somebody's API.
        "paid_tickets": len(paid),
        "paid_share": round(len(paid) / len(routed_list), 3) if routed_list else 0.0,
        "solved_free": sum(1 for r in solved if not r.paid),
        "by_tier": by_tier,
    }
