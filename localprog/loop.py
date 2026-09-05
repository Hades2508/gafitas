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

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import protocol, tools
from .errors import ERROR_NO_TOOL_CALL, ERROR_NOTHING_CHANGED, HarnessInvalid, InvalidCall, ProviderError
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

#: Identical FAILING calls before the run is declared stalled. Deliberately
#: looser than the dead-turn threshold: three dead turns is a stall because
#: nothing happened at all, whereas a failing call is at least an attempt, and
#: the reference engine legitimately retries an edit two or three times after
#: re-reading. Six identical failures is not a retry -- qwen2.5-coder:3b spent
#: TWENTY turns on one refused finish() and the run was sealed BUDGET_EXHAUSTED,
#: which is not what happened to it.
REPEAT_STALL_THRESHOLD = 6

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

#: Turns of pure exploration before the agent is told it has not changed
#: anything yet (F-34). qwen3.5:9b ran the same dogfood ticket twice: once it
#: got five of six tests green, once it made 25 reads, 10 greps, 4 runs and not
#: one edit in forty turns. The harness could see that and never said it.
NO_EDIT_NOTICE_AFTER = 12

#: Fruitless searches remembered before the agent is shown the whole list
#: (F-60). Two is a coincidence; the third is a pattern, and by then the agent
#: has usually forgotten the first one it tried.
FRUITLESS_SEARCH_NOTICE_AFTER = 3

#: Which tools count as "looking for something". A call that returns nothing
#: from one of these is a dead end worth remembering; a failed edit is not.
SEARCH_TOOLS = ("grep", "search_code", "list_symbols", "read_symbol")

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
    #: Searches that came back with nothing, in order. A long list on a failed
    #: run is the signature of a retrieval problem rather than a coding one.
    dead_ends: list[str] = field(default_factory=list)
    provider_error: dict | None = None
    harness_invalid: dict | None = None
    events: list[dict] = field(default_factory=list)
    #: sha256 -> the full text of any tool argument too long for the event (A2).
    payloads: dict = field(default_factory=dict)

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
            "dead_ends": list(self.dead_ends),
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


#: How much of a model's own words to seal when no tool call could be found.
#: The whole point is to make those turns diagnosable, and 316 of them on one
#: engine went unexplained for a whole cohort -- but an engine that loops can
#: emit a great deal, so it is bounded and the truncation is announced rather
#: than silent.
NO_CALL_TEXT_LIMIT = 4000

#: How much of a reasoning channel is sealed. One turn of qwen3.5:2b reached
#: 41 223 characters, so this cannot be unbounded -- a cohort's evidence would
#: be mostly deliberation. Head and tail both, and the tail is the larger of
#: the two: a model that decided on a call decided at the end.
REASONING_HEAD = 400
REASONING_TAIL = 1200


def _seal_text(text: Any, limit: int = NO_CALL_TEXT_LIMIT) -> dict:
    """The model's own output, bounded, with the bound stated."""
    if not isinstance(text, str):
        return {"chars": 0, "text": "", "truncated": False,
                "note": f"no era texto sino {type(text).__name__}"}
    return {"chars": len(text),
            "text": text[:limit],
            "truncated": len(text) > limit}


def _reasoning_of(message: Any) -> dict:
    """The reasoning channel, separately from the answer, when there is one.

    Recorded as its own field because ``eval_count`` counts the reasoning and
    the answer together, so the cost of thinking could not be told apart from
    the cost of answering. Characters, not tokens: the server reports no token
    split, and inventing one from a ratio would be a guess wearing a number.
    """
    if not isinstance(message, dict):
        return {}
    for key in ("thinking", "reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            sealed = {"reasoning_key": key, "reasoning_chars": len(value)}
            # F-119. Eighty-two of qwen3.5:2b's ninety-one dead turns emitted
            # EMPTY content, and eighty of those carried a reasoning channel:
            # the model thought and then closed the turn without answering.
            # Whether the call it had already decided on was sitting in that
            # channel is the question, and the previous record -- a character
            # count and nothing else -- could not be asked it. The TAIL is
            # sealed as well as the head because a decision comes last.
            head = value[:REASONING_HEAD]
            tail = value[-REASONING_TAIL:] if len(value) > REASONING_HEAD else ""
            sealed["reasoning_head"] = head
            if tail and tail != head:
                sealed["reasoning_tail"] = tail
            sealed["reasoning_elided"] = max(
                len(value) - len(head) - len(tail), 0)
            return sealed
    return {}


def _call_usage(raw: Any, budget: int | None = None) -> dict:
    """This ONE call's reported cost, not the running total.

    The ticket total could not answer "how many generations hit the cap", which
    is the difference between an engine that cannot answer and one that was cut
    off mid-answer. Per call, it can -- and with ``budget`` it also says so
    directly, because a generation that reached the budget it was given did not
    finish, whatever it managed to say first (F-101).

    Recorded, never acted on here. A budget raised on a hunch is how the
    previous version of this rule ended up covering only half the engines it
    needed to.
    """
    if not isinstance(raw, dict):
        return {}
    source = raw.get("usage") if isinstance(raw.get("usage"), dict) else raw
    out: dict[str, Any] = {}
    for target, names in _USAGE_KEYS:
        for name in names:
            value = source.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                out[target] = value
                break
    nanoseconds = raw.get("total_duration")
    if isinstance(nanoseconds, (int, float)) and not isinstance(nanoseconds, bool):
        out["provider_seconds"] = round(nanoseconds / 1e9, 3)
    produced = out.get("output_tokens")
    if isinstance(budget, int) and budget > 0 and isinstance(produced, int):
        out["budget"] = budget
        out["hit_budget"] = produced >= budget
    return out


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


def _no_edit_note(turn: int, changed: int, max_turns: int) -> str:
    """Say that nothing has been changed yet, when nothing has (F-34).

    Same rule as the other two notes: state a fact about the agent's own run
    and let it decide. Exploring is legitimate and sometimes long; exploring
    for forty turns and finishing with an empty diff is not a plan, and by then
    it is too late to say so.
    """
    if changed or turn < NO_EDIT_NOTICE_AFTER or turn >= max_turns:
        return ""
    return (f"\n[Llevas {turn} turnos y todavia no has cambiado nada. Explorar esta "
            f"bien, pero el trabajo es la edicion. Si ya sabes que hay que tocar, "
            f"hazlo ahora con edit o replace_lines.]")


def _error_repeat_note(count: int, tool_name: str, code: str | None) -> str:
    """Name a repeated identical failure (F-43).

    Deliberately blunt at the third occurrence. By then the agent has received
    the same correction twice and acted on neither, so a gentler phrasing has
    already been tried -- twice.
    """
    if count < 3:
        return ""
    return (f"\n[Es la {count}a vez seguida que {tool_name} falla con {code}. "
            f"Lo que estas intentando NO esta funcionando y repetirlo dara lo mismo. "
            f"Cambia de enfoque: vuelve a LEER el fichero para ver como esta de "
            f"verdad, o usa otra herramienta.]")


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


#: How much of one argument value is sealed into the evidence (F-58). Enough to
#: read back the query the agent actually ran; not the whole file it wrote.
EVENT_ARG_CHARS = 300


def _event_args(arguments: Any, sidecar: dict | None = None) -> Any:
    """The call's arguments, small enough to keep for every turn of every run.

    A long value is truncated in the event AND kept whole in *sidecar*, keyed by
    the sha256 of its content (A2). Truncation alone made ``write_file`` bodies
    unrecoverable, which blocked two analyses in the campaign that found it: the
    only way to tell a correct copy from an over-copy is to read what was
    written, and the evidence had thrown it away to save space. A hash plus a
    sidecar costs the same space per DISTINCT payload and nothing per repeat.

    ``evidence.py`` has said since it was written that every tool call and *its
    arguments* are sealed. The arguments were never actually recorded, and the
    gap only became visible when a retrieval failure had to be diagnosed from
    the sealed runs: 46 of 100 cases ended with the agent saying it had searched
    and found nothing, and there was no way to see WHAT it searched for. A
    transcript without the queries cannot answer the one question a retrieval
    audit asks.
    """
    def keep(text: str) -> Any:
        if len(text) <= EVENT_ARG_CHARS:
            return text
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        if sidecar is not None:
            sidecar[digest] = text
        return {"truncated": text[:EVENT_ARG_CHARS], "chars": len(text),
                "sha256": digest}

    if not isinstance(arguments, dict):
        return keep(str(arguments))
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
            continue
        out[key] = keep(value if isinstance(value, str) else repr(value))
    return out


def _seen_before_note(previous_turn: int | None, count: int) -> str:
    """Say that this exact call has already been made, and when (F-60).

    ``_repeat_note`` catches an identical RESULT twice in a row. This catches
    the other shape, which the sealed runs are full of: the same search
    repeated five or ten turns later, after the agent has forgotten it already
    tried it. The transcript window means it genuinely cannot see the earlier
    attempt, so the harness -- which can -- has to be the one that remembers.
    """
    if previous_turn is None or count < 2:
        return ""
    return (f"\n[Ya hiciste EXACTAMENTE esta llamada en el turno {previous_turn} "
            f"(van {count}). El repositorio no ha cambiado desde entonces, asi que "
            f"la respuesta es la misma. Prueba otra cosa: otros terminos, otro "
            f"fichero, u otra herramienta.]")


def _fruitless_note(dead_ends: list[str]) -> str:
    """List the searches that have already come back empty (F-60).

    Negative results are information and they are the first thing to fall out
    of a truncated transcript. An agent that cannot see its own dead ends walks
    back into them, which is exactly what 46 of 100 failed RepoQA runs did
    before finishing BLOCKED.
    """
    if len(dead_ends) < FRUITLESS_SEARCH_NOTICE_AFTER:
        return ""
    shown = ", ".join(dead_ends[-6:])
    return (f"\n[Llevas {len(dead_ends)} busquedas que no han encontrado nada: "
            f"{shown}. Ese camino no esta dando resultado. Si buscas por literal, "
            f"prueba search_code(query=...) describiendo con palabras lo que hace "
            f"el codigo; si ya lo haces, cambia los terminos.]")


def _found_nothing(name: str, value: Any) -> bool:
    """Whether a SUCCESSFUL search call actually returned any result."""
    if name not in SEARCH_TOOLS:
        return False
    if isinstance(value, dict):
        if "hits" in value:
            return not value["hits"]
        if "candidates" in value:
            return not value["candidates"]
    if isinstance(value, list):
        return not value
    return False


def _search_label(name: str, arguments: Any) -> str:
    """A short, readable name for one search, for the dead-end list."""
    if isinstance(arguments, dict):
        for key in ("pattern", "query", "name", "path"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return f"{name}({value.strip()[:40]!r})"
    return name + "(...)"


def _signature(name: str, arguments: Any) -> str:
    try:
        return name + "|" + json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + "|<unserialisable>"


def _no_call_note(ctx: tools.ToolContext) -> str:
    """Point prose at the call that expresses it.

    ERROR_NO_TOOL_CALL's standing advice is "if what you wrote was the CONTENT
    of a file, pass it as write_file(content=...)". That is right when nothing
    has been produced yet and it is a trap once something has: the reference
    engine spent 368 turns of 50 runs alternating prose and a redundant
    write_file, with a correct answer already on disk and finish called in 2
    runs out of 50.

    So when the mission's expected output exists and is not empty, the
    correction leads with finish. Otherwise it says nothing and the ordinary
    message stands.
    """
    if not ctx.allowed_new_files:
        return ""
    for name in ctx.allowed_new_files:
        target = ctx.root / name
        try:
            if not target.is_file() or target.stat().st_size <= 0:
                return ""
        except OSError:
            return ""
    return ("\n  OJO: " + ", ".join(ctx.allowed_new_files[:3]) + " YA existe y "
            "tiene contenido. Si lo que querias decir es que has terminado, eso "
            "NO se dice escribiendo texto ni volviendo a escribir el fichero: se "
            "dice con finish(status='DONE'). Volver a escribir lo mismo gasta "
            "turnos y no cambia nada.")


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
    tool_role_supported: bool | None = None,
) -> LoopResult:
    """Drive *provider* against *ctx* until it finishes or runs out of budget."""
    if protocol_name not in ("A", "B", "J"):
        raise HarnessInvalid(f"unknown protocol {protocol_name!r}")

    transcript = Transcript(
        system=system,
        user=objective,
        keep_turns=KEEP_TURNS if keep_turns is None else keep_turns,
        elide_over_chars=(
            ELIDE_OVER_CHARS if elide_over_chars is None else elide_over_chars
        ),
        budget_chars=budget_chars,
        tool_role_supported=tool_role_supported,
    )
    # An explicit declare_tools still wins: the frozen screen names its seven
    # and must keep naming exactly those, or it stops being comparable with the
    # runs already on disk. Otherwise the surface is whatever this mission's
    # scope can actually satisfy.
    declared = (declare_tools if declare_tools is not None
                else tools.legal_tools(tuple(ctx.write_scope),
                                       tuple(ctx.allowed_new_files)))
    schema = tools.native_schema(declared) if protocol_name == "A" else None
    result = LoopResult(outcome=BUDGET_EXHAUSTED)
    used: Counter = Counter()
    signatures: Counter = Counter()
    events: list[dict] = []
    payloads: dict[str, str] = {}
    started = time.perf_counter()
    last_call_was_finish = False
    consecutive_dead = 0
    last_payload: tuple[str, str] | None = None
    repeat_count = 0
    last_tool_error: tuple[str, str] | None = None
    error_repeat = 0
    first_seen: dict[str, int] = {}
    dead_ends: list[str] = []

    try:
        for turn in range(1, max_turns + 1):
            result.turns_used = turn
            # finish() weighs 'I am stuck' against 'I have barely
            # started', so it needs to know where we are (F-45).
            ctx.turns_left = max_turns - turn
            ctx.max_turns_hint = max_turns
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
                feedback = exc.feedback()
                if exc.code == ERROR_NO_TOOL_CALL:
                    feedback += _no_call_note(ctx)
                transcript.add_correction(assistant, feedback)
                events.append({"turn": turn, "invalid_call": exc.code,
                               "feedback": feedback,
                               "consecutive_dead": consecutive_dead,
                               "prompt_tokens": turn_input,
                               # What the model ACTUALLY said. Without this a
                               # dead turn is a fact with no cause attached, and
                               # a whole cohort's worth of them stayed
                               # undiagnosable.
                               "emitted": _seal_text(message.get("content")),
                               "call_usage": _call_usage(raw, getattr(provider, 'num_predict', None)),
                               "at_seconds": round(time.perf_counter() - started, 3),
                               **_reasoning_of(message)})
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
            seen_at = first_seen.get(signature)
            signatures[signature] += 1
            if seen_at is None:
                first_seen[signature] = turn
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
                if _found_nothing(outcome.name, outcome.value):
                    label = _search_label(outcome.name, call.arguments)
                    if label not in dead_ends:
                        dead_ends.append(label)

                annotated = payload + _repeat_note(repeat_count, outcome.name)
                annotated += _seen_before_note(seen_at, signatures[signature])
                if _found_nothing(outcome.name, outcome.value):
                    annotated += _fruitless_note(dead_ends)
                # F-22: and how much budget is left to act on it.
                annotated += _no_edit_note(turn, len(ctx.changed_files), max_turns)
                annotated += _budget_note(turn, max_turns)

                transcript.add(Turn(number=turn, assistant=assistant,
                                    tool_name=outcome.name, tool_payload=annotated))
                events.append({"turn": turn, "tool": outcome.name, "ok": True,
                               "args": _event_args(call.arguments, payloads),
                               "seen_at": seen_at,
                               "result_chars": len(payload),
                               "repeat_count": repeat_count,
                               "prompt_tokens": turn_input,
                               "call_usage": _call_usage(raw, getattr(provider, 'num_predict', None)),
                               "at_seconds": round(time.perf_counter() - started, 3),
                               **_reasoning_of(message)})
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

            # F-61: the double-finish escape used to live here. It predates
            # finish(status='NO_CHANGE') (F-10), which is now the explicit and
            # only way to say a mission needs no change -- and while both
            # existed, ANY refused finish could be turned into an accepted one
            # simply by making the same call again.
            #
            # That is not a theoretical objection. Seven RepoQA runs ended this
            # way: the agent reported in its own summary that it had written the
            # answer file, had in fact never called write_file once, was refused
            # for having changed nothing, repeated itself, and was recorded as
            # having deliberately finished. A rule that converts "you have not
            # done the work" into "confirmed, there was no work" on the second
            # attempt is a rule that launders failure into completion.

            # F-43: the same error, again. F-23 says an identical RESULT twice
            # running means nothing changed; an identical ERROR is the same fact
            # and more urgent, because it means feedback the agent is already
            # receiving is not reaching its decisions. ga04 was refused the same
            # way ten times while sitting on 14 of 16 tests green.
            if outcome.name in SEARCH_TOOLS and outcome.code in (
                "ERROR_NO_MATCH", "ERROR_SEARCH_SCOPE_EMPTY", "ERROR_FILE_NOT_FOUND"
            ):
                label = _search_label(outcome.name, call.arguments)
                if label not in dead_ends:
                    dead_ends.append(label)

            error_key = (outcome.name, outcome.code or "")
            error_repeat = error_repeat + 1 if error_key == last_tool_error else 1
            last_tool_error = error_key
            feedback = (outcome.feedback or "") + _error_repeat_note(
                error_repeat, outcome.name, outcome.code
            ) + _seen_before_note(seen_at, signatures[signature])
            if outcome.name in SEARCH_TOOLS:
                feedback += _fruitless_note(dead_ends)
            transcript.add(Turn(
                number=turn, assistant=assistant, tool_name=outcome.name,
                tool_payload=feedback + _budget_note(turn, max_turns),
                is_error=True,
            ))
            events.append({"turn": turn, "tool": outcome.name, "ok": False,
                           "args": _event_args(call.arguments, payloads),
                           "code": outcome.code, "invalid_call": outcome.invalid_call,
                           "prompt_tokens": turn_input,
                           "call_usage": _call_usage(raw, getattr(provider, 'num_predict', None)),
                           "at_seconds": round(time.perf_counter() - started, 3),
                           **_reasoning_of(message)})
            last_call_was_finish = outcome.name == "finish"
            if error_repeat >= REPEAT_STALL_THRESHOLD:
                # STALLED already means "asking again produces the same fact",
                # and it was wired only to turns that produced NO call at all --
                # so a call failing identically forever was invisible to it. A
                # ToolError is not a dead turn by the loop's definition, and
                # qwen2.5-coder:3b spent TWENTY turns on one refused finish()
                # while the run was sealed as BUDGET_EXHAUSTED, which is not
                # what happened to it.
                result.outcome = STALLED
                result.stalled_after = turn
                events.append({"turn": turn, "stalled_on_repeat": error_repeat,
                               "tool": outcome.name, "code": outcome.code})
                break

    except HarnessInvalid as exc:
        result.outcome = HARNESS_INVALID
        result.harness_invalid = exc.to_dict()
        events.append({"turn": result.turns_used, "harness_invalid": exc.detail})

    result.wall_seconds = time.perf_counter() - started
    result.tools_used = dict(used)
    result.loops = sum(1 for count in signatures.values() if count >= LOOP_THRESHOLD)
    result.max_repeat = repeat_count
    result.dead_ends = dead_ends
    result.changed_files = sorted(ctx.changed_files)
    result.tests_green = ctx.tests_green
    result.elisions = max(result.elisions, transcript.elisions)
    result.events = events
    result.payloads = payloads
    return result
