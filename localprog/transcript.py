"""Conversation state, with the elision rule from contract §C.3.

The old architecture's context problem was "fit all the source in one prompt",
and it cost 482 lines of budget arithmetic. The new one's is different and much
cheaper to solve: the transcript grows. Six turns with three file reads will
pass 8192 tokens, and past 8192 the model stops fitting on the GPU -- 59 tok/s
becomes 17 tok/s (measured, PROGRAMMER_HARDWARE_PROFILE_V1). At 12 turns that
is the difference between a five-minute mission and an eighteen-minute one.

The rule: a tool RESULT older than ``KEEP_TURNS`` turns and longer than
``ELIDE_OVER_CHARS`` is replaced by a one-line placeholder. Never elided:

  * the tool CALL itself (name and arguments) -- the decision trace is what
    gives the model continuity, and it is small;
  * anything that was an error -- errors are the feedback the design bets on;
  * the last ``KEEP_TURNS`` turns, whole.

The model can always read a file again. That is what a person does too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Screen defaults. Correct for a five-call task inside an 8192-token window,
#: and far too aggressive for debugging, where the output you need to compare
#: against is by definition several turns old (F-07). ``Transcript`` takes both
#: as instance fields now; these remain the values the frozen screen uses.
KEEP_TURNS = 3
ELIDE_OVER_CHARS = 500

#: Working defaults, for a real context window. Elision still exists -- a
#: transcript that grows without bound falls off the GPU and the run slows by
#: 3.5x (measured, PROGRAMMER_HARDWARE_PROFILE_V1) -- but it now keeps enough
#: history for an agent to compare a failure against the edit that caused it.
WORK_KEEP_TURNS = 8
WORK_ELIDE_OVER_CHARS = 2500


@dataclass
class Turn:
    number: int
    assistant: dict
    tool_name: str | None = None
    tool_payload: str = ""
    is_error: bool = False


@dataclass
class Transcript:
    system: str
    user: str
    turns: list[Turn] = field(default_factory=list)
    elisions: int = 0
    keep_turns: int = KEEP_TURNS
    elide_over_chars: int = ELIDE_OVER_CHARS
    #: Soft character ceiling for the whole rendered transcript, with a hard
    #: floor: the last ``keep_turns`` are never elided, so a transcript whose
    #: recent turns alone exceed the budget will not reach it. That is the
    #: intended trade -- a budget allowed to delete what the agent just did
    #: would cure the context wall by causing amnesia. 0 disables it,
    #: which is what the frozen screen uses -- its transcripts are tiny and its
    #: behaviour must not change. See budget_chars().
    budget_chars: int = 0

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def _placeholder(self, turn: Turn) -> str:
        return f"[resultado de {turn.tool_name} elidido — vuelve a leerlo si lo necesitas]"

    def _size(self, messages) -> int:
        total = 0
        for message in messages:
            content = message.get("content") or ""
            total += len(content) if isinstance(content, str) else 0
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                total += sum(len(str(c)) for c in calls)
        return total

    def messages(self) -> list[dict]:
        """The message list to send, elided to fit the context budget.

        Two passes, and the order matters. First the age rule: results older
        than ``keep_turns`` and larger than ``elide_over_chars`` become a
        placeholder. Then, if the transcript is still over budget, turns are
        shrunk oldest-first -- assistant payloads included -- until it fits.

        The second pass is the one that keeps the run alive. Without it a
        transcript can exceed num_ctx while every individual turn looks
        reasonable, and the server responds by truncating from the front, where
        the system prompt and the tool definitions live. That failure is
        invisible from here: the model simply stops calling tools.
        """
        out: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
        self.elisions = 0
        last = len(self.turns)
        rendered: list[tuple[int, dict, dict | None]] = []
        for turn in self.turns:
            assistant = turn.assistant
            payload_message = None
            if turn.tool_name is not None:
                recent = turn.number > last - self.keep_turns
                payload = turn.tool_payload
                if not recent and not turn.is_error and len(payload) > self.elide_over_chars:
                    payload = self._placeholder(turn)
                    self.elisions += 1
                payload_message = {"role": "tool", "name": turn.tool_name, "content": payload}
            rendered.append((turn.number, assistant, payload_message))

        def assemble() -> list[dict]:
            messages = list(out)
            for _, assistant, payload_message in rendered:
                messages.append(assistant)
                if payload_message is not None:
                    messages.append(payload_message)
            return messages

        if self.budget_chars:
            index = 0
            while self._size(assemble()) > self.budget_chars and index < len(rendered) - self.keep_turns:
                number, assistant, payload_message = rendered[index]
                rendered[index] = (
                    number,
                    _elide_assistant(assistant, ARGUMENT_ELIDE_OVER_CHARS),
                    (
                        {**payload_message, "content": self._placeholder_text(payload_message["name"])}
                        if payload_message is not None
                        and len(payload_message["content"]) > ARGUMENT_ELIDE_OVER_CHARS
                        else payload_message
                    ),
                )
                self.elisions += 1
                index += 1

        return assemble()

    def _placeholder_text(self, name: str) -> str:
        return f"[resultado de {name} elidido - vuelve a leerlo si lo necesitas]"

    #: How much of a no-tool-call reply to keep. F-16: the whole reply used to
    #: be appended, so a turn that failed made the context bigger, which made
    #: the next turn likelier to fail. ga04 rode that runaway for 28 turns until
    #: the server was truncating the system prompt away. An excerpt is enough
    #: for the model to see what it did wrong; the rest is what poisons the well.
    CORRECTION_KEEP_CHARS = 600

    def add_correction(self, assistant: dict, feedback: str) -> None:
        """Record an answer that carried no usable call, plus the nudge back.

        The reply is truncated deliberately. This is not tidiness: an unbounded
        correction turn is a positive feedback loop into the context ceiling.
        """
        content = assistant.get("content") or ""
        if isinstance(content, str) and len(content) > self.CORRECTION_KEEP_CHARS:
            content = (
                content[: self.CORRECTION_KEEP_CHARS]
                + f"\n[... {len(content) - self.CORRECTION_KEEP_CHARS} caracteres mas, elididos]"
            )
        self.turns.append(
            Turn(number=len(self.turns) + 1,
                 assistant={**assistant, "content": content},
                 tool_name="(sin llamada)", tool_payload=feedback, is_error=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "keep_turns": self.keep_turns,
            "elide_over_chars": self.elide_over_chars,
            "budget_chars": self.budget_chars,
            "elisions_last_render": self.elisions,
            "turns": [
                {
                    "number": t.number, "tool": t.tool_name, "is_error": t.is_error,
                    "payload_chars": len(t.tool_payload),
                }
                for t in self.turns
            ],
        }


# ---------------------------------------------------------------- budgeting

#: Characters per token, for budgeting only. Deliberately pessimistic: code and
#: JSON tokenise worse than prose, and the cost of over-estimating is one extra
#: elision while the cost of under-estimating is silent truncation of the
#: system prompt by the server.
CHARS_PER_TOKEN = 3.0

#: How much of the context window the transcript may occupy. The rest is
#: headroom for the tool schema (about 1.5 k tokens for nine documented tools),
#: the system prompt, and the reply the model is about to generate.
CONTEXT_FRACTION = 0.55

#: A tool-call ARGUMENT longer than this is summarised once it is old. The call
#: itself -- name, and the shape of its arguments -- is never removed.
ARGUMENT_ELIDE_OVER_CHARS = 400


def budget_chars(num_ctx: int) -> int:
    return int(num_ctx * CONTEXT_FRACTION * CHARS_PER_TOKEN)


def _shrink_arguments(arguments, limit: int):
    """Replace oversized argument VALUES with a description of what was there.

    Keeps the model's decision legible -- it can still see that it called
    write_file on that path -- without carrying the payload forever.
    """
    if not isinstance(arguments, dict):
        if isinstance(arguments, str) and len(arguments) > limit:
            return f"[{len(arguments)} caracteres elididos]"
        return arguments
    out = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > limit:
            lines = value.count(chr(10)) + 1
            out[key] = f"[elidido: {len(value)} caracteres, {lines} lineas]"
        else:
            out[key] = value
    return out


def _elide_assistant(message: dict, limit: int) -> dict:
    """A past assistant turn, shrunk to its decision rather than its payload."""
    out = dict(message)
    content = out.get("content") or ""
    if isinstance(content, str) and len(content) > limit:
        out["content"] = content[:limit] + f"\n[... {len(content) - limit} caracteres elididos]"
    calls = out.get("tool_calls")
    if isinstance(calls, list):
        shrunk = []
        for call in calls:
            if not isinstance(call, dict):
                shrunk.append(call)
                continue
            copy = dict(call)
            function = copy.get("function")
            if isinstance(function, dict):
                fcopy = dict(function)
                fcopy["arguments"] = _shrink_arguments(
                    fcopy.get("arguments"), ARGUMENT_ELIDE_OVER_CHARS
                )
                copy["function"] = fcopy
            shrunk.append(copy)
        out["tool_calls"] = shrunk
    return out
