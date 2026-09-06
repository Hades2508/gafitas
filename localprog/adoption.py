"""Why a tool that exists, applies and is available still never gets called.

MEASURED
--------
Two primitives were added to the surface and neither moved anything, for
different reasons:

    replace_file          9 calls out of 85 write calls across two smokes
    replace_symbol_body   0 calls out of 53

Both are in the native schema the model receives every turn. A primitive with
zero invocations cannot improve any rate, so "is it good" is not yet a question
that can be asked of it.

THE CHAIN, KEPT SEPARATE
------------------------
    AVAILABLE    the harness has it at all
    APPLICABLE   its contract could execute here, by objective conditions
    OFFERED      it was named at a point the model was reading
    CONCRETE     the offer carried a call with real arguments filled in
    SELECTED     the model chose it
    CALLED       the call reached dispatch
    ACCEPTED     the call succeeded
    USEFUL       the result changed the outcome

Collapsing any two of these hides the actual failure. "The tool was applicable
but never presented" and "it was presented and not chosen" need different fixes,
and until now the record could not tell them apart.

APPLICABILITY IS NOT A WISH
---------------------------
It is computed from the tree, never asserted because we would like the model to
use something. `replace_symbol_body` is applicable when a symbol in the target
file resolves unambiguously -- that is a fact about the file, checkable without
knowing what the model intended. It does not claim the model SHOULD have used
it; only that the contract could have run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Calls whose INTENT is "this file should now contain something else", which is
#: what `replace_file` exists to express. Derived from the call, not from a
#: guess about the model's plan.
WHOLE_FILE_INTENT = ("write_file", "replace_file")


@dataclass
class Opportunity:
    """One moment where a primitive could have run, and what happened instead."""

    turn: int
    primitive: str
    available: bool
    applicable: bool
    applicable_because: str
    offered: bool = False
    concrete_call_offered: bool = False
    selected: bool = False
    called: bool = False
    accepted: bool = False
    useful: bool | None = None
    fallback_tool_selected: str | None = None
    reason_not_applicable: str = ""
    reason_rejected: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def file_is_replaceable(root: Path, rel: str | None, *, opened: set) -> tuple[bool, str]:
    """Could `replace_file` run here? A fact about the tree, not about intent."""
    if not rel:
        return False, "the call names no path"
    path = root / rel
    try:
        if not path.is_file():
            return False, "the target does not exist, so there is nothing to replace"
    except OSError:
        return False, "the target could not be read"
    if rel not in opened:
        return False, ("the file has not been opened in this run, so replacing it "
                       "would be a blind overwrite")
    return True, "an existing, already-read file"


def symbol_is_replaceable(root: Path, rel: str | None) -> tuple[bool, str, list]:
    """Could `replace_symbol_body` run here, and on which symbols?

    Objective: the file exists, the language has a symbol index, and at least
    one symbol resolves unambiguously with a body on its own lines. Nothing
    here guesses which symbol the model wanted.
    """
    if not rel:
        return False, "the call names no path", []
    path = root / rel
    try:
        if not path.is_file():
            return False, "the target does not exist", []
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, "the target could not be read", []
    if not rel.endswith((".py", ".pyi")):
        return False, "no symbol index for this language", []

    import ast

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return False, "the file does not parse, so no symbol resolves", []

    names: dict = {}
    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = ".".join(chain + [child.name])
                names[qualified] = child
                walk(child, chain + [child.name])
    walk(tree, [])
    if not names:
        return False, "the file defines no symbols", []

    usable = []
    for qualified, node in names.items():
        inner = getattr(node, "body", None)
        if not inner:
            continue
        # The declaration and the body must be on different lines, or there is
        # no span that is only the body -- the same condition the primitive
        # itself declines on.
        if inner[0].lineno > node.lineno:
            usable.append(qualified)
    if not usable:
        return False, "no symbol has a body on its own lines", []
    # Ambiguity is a property of the bare name, and the primitive refuses it.
    bare: dict = {}
    for qualified in usable:
        bare.setdefault(qualified.rsplit(".", 1)[-1], []).append(qualified)
    unambiguous = [q for q in usable if len(bare[q.rsplit(".", 1)[-1]]) == 1]
    if not unambiguous:
        return False, "every symbol name in the file is ambiguous", []
    return True, f"{len(unambiguous)} symbols resolve unambiguously", sorted(unambiguous)


def concrete_offer(primitive: str, rel: str, symbols: list | None = None) -> str:
    """A call the model could paste, built only from what the harness knows.

    F-82 measured, on a different surface, that naming a CONCRETE call beats
    naming a kind of call: 95% against 7%. This is that mechanism applied to
    tool selection -- and it stops exactly where knowledge stops. The path is
    known. The symbol is known. **The body is not, and is never filled in**:
    deciding what the code should say is the model's work and writing it here
    would be answering the task.
    """
    if primitive == "replace_file":
        return f"replace_file(path={rel!r}, content=<el fichero entero, ya cambiado>)"
    if primitive == "replace_symbol_body":
        if symbols:
            shown = ", ".join(repr(s) for s in symbols[:6])
            more = f" (+{len(symbols) - 6} mas)" if len(symbols) > 6 else ""
            return (f"replace_symbol_body(path={rel!r}, name=<uno de: {shown}{more}>, "
                    f"body=<solo el cuerpo, sin la linea del def>)")
        return f"replace_symbol_body(path={rel!r}, name=..., body=...)"
    return ""


@dataclass
class AdoptionLedger:
    """Every opportunity, in order. Sealed with the run."""

    opportunities: list = field(default_factory=list)
    enabled: bool = True

    def record(self, opportunity: Opportunity) -> None:
        if self.enabled:
            self.opportunities.append(opportunity)

    def to_dict(self) -> dict:
        rows = [o.to_dict() for o in self.opportunities]
        by_primitive: dict = {}
        for row in rows:
            slot = by_primitive.setdefault(row["primitive"], {
                "opportunities": 0, "applicable": 0, "offered": 0,
                "concrete": 0, "called": 0, "accepted": 0})
            slot["opportunities"] += 1
            slot["applicable"] += bool(row["applicable"])
            slot["offered"] += bool(row["offered"])
            slot["concrete"] += bool(row["concrete_call_offered"])
            slot["called"] += bool(row["called"])
            slot["accepted"] += bool(row["accepted"])
        return {"events": rows, "by_primitive": by_primitive}


def observe(ledger: AdoptionLedger, *, turn: int, tool: str, root: Path,
            rel: str | None, opened: set, accepted: bool,
            offers: dict | None = None) -> list:
    """Record what was applicable at this call, and what was chosen instead.

    Returns the primitives that were applicable, so a caller can decide whether
    to offer them. Recording happens either way: an opportunity that is never
    offered is exactly the case this exists to make visible.
    """
    offers = offers or {}
    applicable: list = []

    can_file, why_file = file_is_replaceable(root, rel, opened=opened)
    can_symbol, why_symbol, symbols = symbol_is_replaceable(root, rel)

    for primitive, ok, why in (("replace_file", can_file, why_file),
                               ("replace_symbol_body", can_symbol, why_symbol)):
        chosen = tool == primitive
        offer = offers.get(primitive) or {}
        ledger.record(Opportunity(
            turn=turn, primitive=primitive, available=True,
            applicable=ok, applicable_because=why if ok else "",
            reason_not_applicable="" if ok else why,
            offered=bool(offer.get("offered")),
            concrete_call_offered=bool(offer.get("concrete")),
            selected=chosen, called=chosen, accepted=chosen and accepted,
            fallback_tool_selected=None if chosen else tool))
        if ok:
            applicable.append(primitive)
    return applicable
