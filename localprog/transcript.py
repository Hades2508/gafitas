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

KEEP_TURNS = 3
ELIDE_OVER_CHARS = 500


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

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def _placeholder(self, turn: Turn) -> str:
        return f"[resultado de {turn.tool_name} elidido — vuelve a leerlo si lo necesitas]"

    def messages(self) -> list[dict]:
        """The message list to send, with old bulky results elided."""
        out: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]
        self.elisions = 0
        last = len(self.turns)
        for turn in self.turns:
            out.append(turn.assistant)
            if turn.tool_name is None:
                continue
            recent = turn.number > last - KEEP_TURNS
            payload = turn.tool_payload
            if not recent and not turn.is_error and len(payload) > ELIDE_OVER_CHARS:
                payload = self._placeholder(turn)
                self.elisions += 1
            out.append({"role": "tool", "name": turn.tool_name, "content": payload})
        return out

    def add_correction(self, assistant: dict, feedback: str) -> None:
        """Record an answer that carried no usable call, plus the nudge back."""
        self.turns.append(
            Turn(number=len(self.turns) + 1, assistant=assistant,
                 tool_name="(sin llamada)", tool_payload=feedback, is_error=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "user": self.user,
            "keep_turns": KEEP_TURNS,
            "elide_over_chars": ELIDE_OVER_CHARS,
            "elisions_last_render": self.elisions,
            "turns": [
                {
                    "number": t.number, "tool": t.tool_name, "is_error": t.is_error,
                    "payload_chars": len(t.tool_payload),
                }
                for t in self.turns
            ],
        }
