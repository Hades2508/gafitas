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
from .transcript import Transcript, Turn

MAX_TURNS = 12  # contract §C.2

FINISHED = "FINISHED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
PROVIDER_ERROR = "PROVIDER_ERROR"
HARNESS_INVALID = "HARNESS_INVALID"
OUTCOMES = (FINISHED, BUDGET_EXHAUSTED, PROVIDER_ERROR, HARNESS_INVALID)

#: A call signature seen this many times is a loop, per contract §D.5.
LOOP_THRESHOLD = 3


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
        return self.outcome in (FINISHED, BUDGET_EXHAUSTED)

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
            "provider_error": self.provider_error,
            "harness_invalid": self.harness_invalid,
        }


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
) -> LoopResult:
    """Drive *provider* against *ctx* until it finishes or runs out of budget."""
    if protocol_name not in ("A", "B"):
        raise HarnessInvalid(f"unknown protocol {protocol_name!r}")

    transcript = Transcript(system=system, user=objective)
    schema = tools.native_schema() if protocol_name == "A" else None
    result = LoopResult(outcome=BUDGET_EXHAUSTED)
    used: Counter = Counter()
    signatures: Counter = Counter()
    events: list[dict] = []
    started = time.perf_counter()
    last_call_was_finish = False

    try:
        for turn in range(1, max_turns + 1):
            result.turns_used = turn
            messages = transcript.messages()
            result.elisions = transcript.elisions

            try:
                raw = provider.chat(messages, schema)
            except ProviderError as exc:
                result.outcome = PROVIDER_ERROR
                result.provider_error = exc.to_dict()
                events.append({"turn": turn, "provider_error": exc.to_dict()})
                break

            message = raw.get("message", {}) if isinstance(raw, dict) else {}
            assistant = _assistant_message(message)

            try:
                call = protocol.parse(protocol_name, message)
            except InvalidCall as exc:
                result.invalid_calls += 1
                transcript.add_correction(assistant, exc.feedback())
                events.append({"turn": turn, "invalid_call": exc.code, "feedback": exc.feedback()})
                last_call_was_finish = False
                continue

            signature = _signature(call.name, call.arguments)
            signatures[signature] += 1
            used[call.name if isinstance(call.name, str) else "(no-string)"] += 1

            outcome = tools.dispatch(ctx, call.name, call.arguments)

            if outcome.ok:
                payload = _payload(outcome.value)
                if outcome.feedback:
                    payload = payload + "\n" + outcome.feedback
                transcript.add(Turn(number=turn, assistant=assistant,
                                    tool_name=outcome.name, tool_payload=payload))
                events.append({"turn": turn, "tool": outcome.name, "ok": True,
                               "result_chars": len(payload)})
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

            transcript.add(Turn(number=turn, assistant=assistant, tool_name=outcome.name,
                                tool_payload=outcome.feedback or "", is_error=True))
            events.append({"turn": turn, "tool": outcome.name, "ok": False,
                           "code": outcome.code, "invalid_call": outcome.invalid_call})
            last_call_was_finish = outcome.name == "finish"

    except HarnessInvalid as exc:
        result.outcome = HARNESS_INVALID
        result.harness_invalid = exc.to_dict()
        events.append({"turn": result.turns_used, "harness_invalid": exc.detail})

    result.wall_seconds = time.perf_counter() - started
    result.tools_used = dict(used)
    result.loops = sum(1 for count in signatures.values() if count >= LOOP_THRESHOLD)
    result.changed_files = sorted(ctx.changed_files)
    result.tests_green = ctx.tests_green
    result.elisions = max(result.elisions, transcript.elisions)
    result.events = events
    return result
