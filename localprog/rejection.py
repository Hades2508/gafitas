"""Refusing the same edit twice costs a turn and buys nothing.

MEASURED
--------
Phase 5 p5r3, nine runs of granite4.1:3b on real tasks: 69 edit attempts, 5
accepted, 64 refused -- and **31 of the 64 were byte-identical repeats of a
payload already refused in the same run against the same file**. One run sent
the same bytes fourteen times. Another sent one 195-character payload at turns
6, 7, 9 and 19.

Each of those cost a turn: the harness re-ran the whole operation, failed in
exactly the same way, and returned exactly the same refusal. It had no memory of
what it had already said no to.

WHAT THIS IS NOT
----------------
It is not a limit on retrying. A retry is how an agent recovers, and most of the
64 refusals were *different* payloads -- that is the loop working. This refuses
only the case where nothing has changed: same tool, same target, same normalised
arguments, same payload, and the same file underneath.

VERSIONED BY THE TARGET'S STATE
-------------------------------
The same edit refused against a file that has since changed may now be perfectly
valid, so the memory is keyed on a digest of the file as it is at the moment of
the attempt. Edit the file, and every past refusal against it stops applying.
That is what makes this safe to apply automatically: a legitimate retry cannot
be blocked, because a legitimate retry follows a change.

WHAT IT SAYS BACK
-----------------
The original reason, how many times it has been sent, and the fact that the file
has not moved. Never what to write instead: the harness does not know the answer
and saying so would be inventing one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

#: Arguments that carry the edit's PAYLOAD rather than its address. Normalised
#: separately because whitespace at the very end of a payload is not a different
#: edit, while whitespace inside one is.
#: What the attempt WRITES. A trailing newline here is cosmetic, so it is
#: stripped before hashing and a body resent with one is the same body.
_WRITTEN_KEYS = ("content", "new", "argv")

#: F-178. `old` is deliberately NOT in that list. It is a pattern, matched
#: byte for byte by `edit`, so "hello " and "hello" are different patterns and
#: only one of them can match. Stripping it made a failed attempt and its own
#: correction hash identically, and the agent that fixed its mistake was told
#: it had already tried that.
_PAYLOAD_KEYS = _WRITTEN_KEYS + ("old",)


def _normalise(value) -> str:
    """One stable string for an argument value, whatever shape it arrives in."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def attempt_signature(tool: str, args: dict) -> str:
    """A stable identity for one edit attempt, address and payload together.

    Trailing whitespace is stripped from a payload that is WRITTEN -- resending
    the same body with a newline added is the same body.

    F-178: it used to be stripped from ``old`` as well, and ``old`` is not a
    payload. It is a pattern, matched byte for byte by ``edit``, so "hello " and
    "hello" are different patterns and exactly one of them can succeed.

    The cost of that was the worst behaviour a memory like this can have. An
    agent tried old="hello " against a file containing "hello", was correctly
    told the text was not found, CORRECTED IT to old="hello" -- and was refused
    with ERROR_REPEATED_REJECTED_EDIT, because after stripping the two attempts
    hashed the same. It had done precisely what the error asked of it and was
    told it had already tried that. The file could not be edited again for the
    rest of the run.

    A memory that punishes correction is worse than no memory.
    """
    parts = [str(tool)]
    for key in sorted(args or {}):
        value = args[key]
        text = _normalise(value)
        if key in _WRITTEN_KEYS and isinstance(value, str):
            text = text.rstrip()
        parts.append(f"{key}={text}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def target_state(root: Path, rel: str | None) -> str:
    """A digest of the file this attempt is aimed at, right now.

    ``ABSENT`` when there is no such file, which is itself a state: an edit
    refused because the file did not exist must be reconsidered once it does.
    ``UNREADABLE`` never matches anything, so an unreadable target disables the
    memory rather than blocking on a state nobody can establish.
    """
    if not rel:
        return "NO_TARGET"
    try:
        path = root / rel
        if not path.is_file():
            return "ABSENT"
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "UNREADABLE"


@dataclass
class Rejection:
    code: str
    detail: str
    count: int = 1
    first_turn: int | None = None


@dataclass
class RejectionMemory:
    """What has already been refused, and against which state of which file."""

    entries: dict = field(default_factory=dict)

    @staticmethod
    def key(signature: str, state: str) -> str:
        return f"{signature}@{state}"

    def remember(self, signature: str, state: str, *, code: str, detail: str,
                 turn: int | None = None) -> None:
        if state == "UNREADABLE":
            # A state nobody can establish cannot be compared against later.
            return
        slot = self.key(signature, state)
        found = self.entries.get(slot)
        if found is None:
            self.entries[slot] = Rejection(code=code, detail=detail, first_turn=turn)
        else:
            found.count += 1

    def seen(self, signature: str, state: str) -> Rejection | None:
        if state == "UNREADABLE":
            return None
        return self.entries.get(self.key(signature, state))

    def to_dict(self) -> dict:
        return {"remembered": len(self.entries),
                "repeats_refused": sum(max(e.count - 1, 0) for e in self.entries.values())}


def message(found: Rejection, tool: str, rel: str | None) -> str:
    """Why this is being refused without being run again.

    States the fact and the constraint. It does not say what to write: the
    harness does not know, and a guess dressed as guidance is worse than
    silence.
    """
    where = f" sobre {rel!r}" if rel else ""
    times = ("una vez" if found.count == 1 else f"{found.count} veces")
    return (
        f"este mismo {tool}{where} ya fue rechazado {times} por {found.code}, y "
        f"el fichero NO ha cambiado desde entonces, asi que el resultado seria "
        f"identico." + ("\n  Motivo original: " + found.detail.strip()
                        if found.detail else "")
        + "\n  Para que el intento sea distinto tiene que cambiar el CONTENIDO "
          "que envias o el SITIO al que apunta. Reenviarlo igual no puede dar "
          "otro resultado."
    )
