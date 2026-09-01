"""The agent loop. One implementation, used by both the screen and Step 1.

There is exactly one loop in this repository on purpose. The previous attempt
had two -- ``screen_runner.py`` and ``step1_runner.py`` -- which had already
drifted apart before either was trusted: one called the test tool ``run_tests``
(the contract's name), the other called it ``run``; one advertised a
``list_dir`` tool that the contract does not define and that crashed when
handed a file. Two loops means two sets of bugs and two sets of fixes.

Outcome is a closed set:

    FINISHED           the model called finish() and the call was accepted
    BUDGET_EXHAUSTED   max_turns reached. A result, not an error.
    PROVIDER_ERROR     the server failed. NOT a model result.
    HARNESS_INVALID    a defect in this code. Voids the run.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import protocol, tools
from .errors import ERROR_NOTHING_CHANGED, HarnessInvalid, InvalidCall, ProviderError
from .transcript import ELIDE_OVER_CHARS, KEEP_TURNS, Transcript, Turn

MAX_TURNS = 12  # contract §C.2 -- the frozen screen's budget, unchanged.

#: The budget for real work (F-06). Twelve turns cannot hold
#: READ -> SEARCH -> EDIT -> RUN -> OBSERVE -> DEBUG -> REPLAN -> EDIT -> TEST:
#: reading three files and running the suite twice spends half of it before any
#: thinking happens, which is why every STEP1 run ended in BUDGET_EXHAUSTED
#: rather than at a decision. This is still a HARD bound (I10) -- exhausting it
#: is a result, not an error -- it is simply a bound at the scale of the task.
WORK_MAX_TURNS = 40

FINISHED = "FINISHED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
#: The agent stopped emitting tool calls and did not start again (F-15). A
#: distinct outcome from BUDGET_EXHAUSTED because the causes and the fixes are
#: different: exhausting the budget means the work was too big, stalling means
#: the agent lost the ability to act -- usually because the transcript pushed
#: the tool definitions out of the context window.
STALLED = "STALLED"
PROVIDER_ERROR = "PROVIDER_ERROR"
HARNESS_INVALID = "HARNESS_INVALID"
OUTCOMES = (FINISHED, BUDGET_EXHAUSTED, STALLED, PROVIDER_ERROR, HARNESS_INVALID)

#: Consecutive turns with no usable tool call before the run is declared
#: stalled. Three is enough to tell a one-off malformed reply (which the
#: feedback usually fixes on the next turn) from a wedged conversation.
STALL_THRESHOLD = 3

#: A call signature seen this many times is a loop, per contract §D.5.
LOOP_THRESHOLD = 3

#: Remind the agent of its remaining budget once this few turns are left, and
#: on every turn after (F-22). Early on the number is noise; near the end it is
#: the difference between finishing something and being cut off mid-thought.
BUDGET_WARNING_TURNS = 8

#: Identical tool output this many times in a row earns an explicit "nothing
#: changed" note (F-23). Two is right: the second identical result is already
#: evidence that whatever happened in between did not matter.
REPEAT_NOTICE_AFTER = 2

#: Extra attempts for one provider call before the run is abandoned (F-30).
PROVIDER_RETRIES = 2

#: Which provider failures are worth asking again about. A malformed tool call
#: rejected by the server's parser (HTTP_STATUS 500), a dropped connection, a
#: timeout or an unparseable body can all differ on the next sample.
#:
#: CONTEXT_OVERFLOW is deliberately absent: the same prompt overflows the same
#: window every time, so a retry only delays a defect that is ours to fix.
RETRYABLE_PROVIDER_KINDS = frozenset({"HTTP_STATUS", "TIMEOUT", "TRANSPORT", "BAD_BODY"})


@dataclass
class LoopResult:
    outcome: str
    turns_used: int = 0
    invalid_calls: int = 0
    tool_errors: int = 0
    loops: int = 0
    tools_used: dict = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    tests_green: bool = False
    finished_without_changes: bool = False
    elisions: int = 0
    wall_seconds: float = 0.0
    #: Cost, split so LOCAL / LUNA / CLAUDE can be compared (F-08). Filled from
    #: whatever the provider reports; a provider that reports nothing leaves
    #: zeros here rather than silently inventing numbers.
    usage: dict = field(default_factory=lambda: {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cached_tokens": 0, "provider_seconds": 0.0,
    })
    #: Turn at which the run was declared stalled, if it was.
    stalled_after: int | None = None
    #: Longest run of byte-identical tool results. A high number on a failed run
    #: means thrashing -- the agent kept measuring instead of changing anything.
    max_repeat: int = 0
    #: Provider calls that failed and were retried successfully. Non-zero means
    #: the run survived something that used to end it (F-30).
    provider_retries: int = 0
    provider_error: dict | None = None
    harness_invalid: dict | None = None
    events: list[dict] = field(default_factory=list)

    @property
    def scoreable(self) -> bool:
        """Whether this run may contribute to any metric.

        Provider failures and harness defects are recorded and then excluded.
        Counting them would attribute an infrastructure fault to a model, which
        is the specific error the whole taxonomy exists to prevent.
        """
        return self.outcome in (FINISHED, BUDGET_EXHAUSTED, STALLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "scoreable": self.scoreable,
            "turns_used": self.turns_used,
            "invalid_calls": self.invalid_calls,
            "tool_errors": self.tool_errors,
            "loops": self.loops,
            "tools_used": dict(self.tools_used),
            "changed_files": list(self.changed_files),
            "tests_green": self.tests_green,
            "finished_without_changes": self.finished_without_changes,
            "elisions": self.elisions,
            "wall_seconds": round(self.wall_seconds, 3),
            "usage": dict(self.usage),
            "stalled_after": self.stalled_after,
            "max_repeat": self.max_repeat,
            "provider_retries": self.provider_retries,
            "provider_error": self.provider_error,
            "harness_invalid": self.harness_invalid,
        }


#: Where each provider family puts its token counts. Ollama and the
#: OpenAI-shaped APIs disagree about every single name, and a provider that
#: reports nothing must leave the counters alone rather than contribute zeros
#: that later read as "this call was free".
_USAGE_KEYS = (
    ("input_tokens", ("prompt_eval_count", "prompt_tokens", "input_tokens")),
    ("output_tokens", ("eval_count", "completion_tokens", "output_tokens")),
    ("cached_tokens", ("cached_tokens", "cache_read_input_tokens")),
)


def _accumulate_usage(usage: dict, raw: Any) -> None:
    """Fold one provider response's reported cost into the running total."""
    usage["calls"] += 1
    if not isinstance(raw, dict):
        return
    source = raw.get("usage") if isinstance(raw.get("usage"), dict) else raw
    for target, names in _USAGE_KEYS:
        for name in names:
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[target] += value
                break
    nanoseconds = raw.get("total_duration")
    if isinstance(nanoseconds, (int, float)) and not isinstance(nanoseconds, bool):
        usage["provider_seconds"] = round(usage["provider_seconds"] + nanoseconds / 1e9, 3)


def _assistant_message(message: dict) -> dict:
    """The assistant turn to echo back, keeping what the server needs."""
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content", "") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def _payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # a tool returned something unserialisable
        raise HarnessInvalid(f"tool result is not serialisable: {type(value).__name__}: {exc}") from exc


def _budget_note(turn: int, max_turns: int) -> str:
    """A deadline the agent can actually act on (F-22).

    Said only near the end. A countdown on every turn from the first would be
    noise the model learns to skip, and the information is worthless until it
    starts constraining choices.
    """
    left = max_turns - turn
    if left > BUDGET_WARNING_TURNS:
        return ""
    if left <= 0:
        return ""
    if left == 1:
        return ("\n[ULTIMO TURNO. Llama a finish ahora: status='DONE' si los tests "
                "pasan, o status='BLOCKED' explicando que falta.]")
    return (f"\n[Te quedan {left} turnos. Si no vas a llegar, termina con "
            f"finish(status='BLOCKED', summary=...) explicando donde te has quedado.]")


def _repeat_note(count: int, tool_name: str) -> str:
    """Say that nothing changed, when nothing changed (F-23).

    The most useful fact in a debugging loop is that the last edit made no
    difference to what is being measured, and it is invisible to a model
    reading one result at a time.
    """
    if count < REPEAT_NOTICE_AFTER:
        return ""
    return (f"\n[Este resultado de {tool_name} es IDENTICO al anterior ({count} veces "
            f"seguidas). Lo que has hecho entre medias no ha cambiado nada de lo que "
            f"esto mide. Vuelve a leer el codigo o prueba otra cosa: repetirlo dara "
            f"lo mismo.]")


def _signature(name: str, arguments: Any) -> str:
    try:
        return name + "|" + json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + "|<unserialisable>"


def run_loop(
    *,
    provider,
    ctx: tools.ToolContext,
    system: str,
    objective: str,
    protocol_name: str = "A",
    max_turns: int = MAX_TURNS,
    keep_turns: int | None = None,
    elide_over_chars: int | None = None,
    budget_chars: int = 0,
    declare_tools: tuple[str, ...] | None = None,
) -> LoopResult:
    """Drive *provider* against *ctx* until it finishes or runs out of budget."""
    if protocol_name not in ("A", "B"):
        raise HarnessInvalid(f"unknown protocol {protocol_name!r}")

    transcript = Transcript(
        system=system,
        user=objective,
        keep_turns=KEEP_TURNS if keep_turns is None else keep_turns,
        elide_over_chars=(
            ELIDE_OVER_CHARS if elide_over_chars is None else elide_over_chars
        ),
        budget_chars=budget_chars,
    )
    schema = tools.native_schema(declare_tools) if protocol_name == "A" else None
    result = LoopResult(outcome=BUDGET_EXHAUSTED)
    used: Counter = Counter()
    signatures: Counter = Counter()
    events: list[dict] = []
    started = time.perf_counter()
    last_call_was_finish = False
    consecutive_dead = 0
    last_payload: tuple[str, str] | None = None
    repeat_count = 0

    try:
        for turn in range(1, max_turns + 1):
            result.turns_used = turn
            messages = transcript.messages()
            result.elisions = transcript.elisions

            raw = None
            last_error: ProviderError | None = None
            for attempt in range(PROVIDER_RETRIES + 1):
                try:
                    raw = provider.chat(messages, schema)
                    last_error = None
                    break
                except ProviderError as exc:
                    last_error = exc
                    retryable = exc.kind in RETRYABLE_PROVIDER_KINDS
                    events.append({
                        "turn": turn, "provider_error": exc.to_dict(),
                        "attempt": attempt + 1, "retryable": retryable,
                    })
                    if not retryable or attempt == PROVIDER_RETRIES:
                        break
                    result.provider_retries += 1
            if last_error is not None:
                result.outcome = PROVIDER_ERROR
                result.provider_error = last_error.to_dict()
                break

            before_input = result.usage["input_tokens"]
            _accumulate_usage(result.usage, raw)
            turn_input = result.usage["input_tokens"] - before_input
            message = raw.get("message", {}) if isinstance(raw, dict) else {}
            assistant = _assistant_message(message)

            try:
                call = protocol.parse(protocol_name, message)
            except InvalidCall as exc:
                result.invalid_calls += 1
                consecutive_dead += 1
                transcript.add_correction(assistant, exc.feedback())
                events.append({"turn": turn, "invalid_call": exc.code,
                               "feedback": exc.feedback(),
                               "consecutive_dead": consecutive_dead,
                               "prompt_tokens": turn_input})
                last_call_was_finish = False
                if consecutive_dead >= STALL_THRESHOLD:
                    # Asking a 29th time is not persistence, it is spending the
                    # budget to re-observe the same fact. Stop and say so.
                    result.outcome = STALLED
                    result.stalled_after = turn
                    events.append({"turn": turn, "stalled": consecutive_dead})
                    break
                continue

            consecutive_dead = 0
            signature = _signature(call.name, call.arguments)
            signatures[signature] += 1
            used[call.name if isinstance(call.name, str) else "(no-string)"] += 1

            outcome = tools.dispatch(ctx, call.name, call.arguments)

            if outcome.ok:
                payload = _payload(outcome.value)
                if outcome.feedback:
                    payload = payload + "\n" + outcome.feedback

                # F-23: identical output twice running means the work in between
                # did not touch what this measures. Say so.
                key = (outcome.name, payload)
                repeat_count = repeat_count + 1 if key == last_payload else 1
                last_payload = key
                annotated = payload + _repeat_note(repeat_count, outcome.name)
                # F-22: and how much budget is left to act on it.
                annotated += _budget_note(turn, max_turns)

                transcript.add(Turn(number=turn, assistant=assistant,
                                    tool_name=outcome.name, tool_payload=annotated))
                events.append({"turn": turn, "tool": outcome.name, "ok": True,
                               "result_chars": len(payload),
                               "repeat_count": repeat_count,
                               "prompt_tokens": turn_input})
                if outcome.name == "finish":
                    result.outcome = FINISHED
                    break
                last_call_was_finish = False
                continue

            # Not ok: exactly one of the two counters moves.
            if outcome.invalid_call:
                result.invalid_calls += 1
            else:
                result.tool_errors += 1

            # Contract §7: a second consecutive finish() is the legitimate way
            # to say "this mission needs no change".
            if (
                outcome.name == "finish"
                and outcome.code == ERROR_NOTHING_CHANGED
                and last_call_was_finish
            ):
                result.outcome = FINISHED
                result.finished_without_changes = True
                transcript.add(Turn(number=turn, assistant=assistant, tool_name="finish",
                                    tool_payload="FINISHED (sin cambios, confirmado)"))
                events.append({"turn": turn, "tool": "finish", "ok": True, "no_changes": True})
                break

            transcript.add(Turn(
                number=turn, assistant=assistant, tool_name=outcome.name,
                tool_payload=(outcome.feedback or "") + _budget_note(turn, max_turns),
                is_error=True,
            ))
            events.append({"turn": turn, "tool": outcome.name, "ok": False,
                           "code": outcome.code, "invalid_call": outcome.invalid_call,
                           "prompt_tokens": turn_input})
            last_call_was_finish = outcome.name == "finish"

    except HarnessInvalid as exc:
        result.outcome = HARNESS_INVALID
        result.harness_invalid = exc.to_dict()
        events.append({"turn": result.turns_used, "harness_invalid": exc.detail})

    result.wall_seconds = time.perf_counter() - started
    result.tools_used = dict(used)
    result.loops = sum(1 for count in signatures.values() if count >= LOOP_THRESHOLD)
    result.max_repeat = repeat_count
    result.changed_files = sorted(ctx.changed_files)
    result.tests_green = ctx.tests_green
    result.elisions = max(result.elisions, transcript.elisions)
    result.events = events
    return result
