"""Recovering a tool call the engine got nearly right, deterministically.

WHAT THIS IS FOR
----------------
qwen2.5-coder:3b was recorded as ``supports_native_tools=False`` and scored 0%
through GAFITAS against 52% answering in one prompt. The payload probe found the
reason, and it was not the engine:

    prose channel   line_recall 1.00 at 400, 1200 and 3000 characters
    tool  channel   0 of 3, "no tool call at all"

It reproduces three thousand characters of source perfectly. What it actually
emitted when asked to do it through ``write_file`` was this::

    {"name": "write_file", "arguments": {"content": "def calcular_impuesto_01(...):
        \"\"\"Devuelve el impuesto...\"\"\"<RAW NEWLINE>\\n    if exento or base <= 0:

That is a correct tool call, with the correct name, the correct argument and the
correct payload -- containing two things JSON does not allow inside a string: a
raw newline, and the docstring's unescaped quotes. Ollama's own parser rejected
it, so ``message.tool_calls`` came back empty, so the harness reported
ERROR_NO_TOOL_CALL and threw the whole thing away.

We were not measuring an engine that cannot call tools. We were measuring a
harness that discards a call for a quoting mistake, fifty times, and then
concluded the engine was too small.

THE RULES THIS FOLLOWS
----------------------
1. **Never invent.** Every tier here either recovers bytes the engine actually
   emitted or fails. Nothing guesses a path, a tool name or an argument.
2. **Never repair a call that parsed.** These tiers run only after the ordinary
   parser has failed, so a well-formed engine can never take this path and can
   never be changed by it.
3. **Announce.** A recovered call carries ``repaired`` and the tier that fired.
   The loop counts them and evidence seals them, because a run that only worked
   because of repairs is a different fact from a run that did not need any.
4. **Bounded.** Five tiers, each a fixed deterministic transformation. No
   model, no heuristic search, no retry.

WHY NOT JUST ASK THE MODEL AGAIN
--------------------------------
It was asked again -- that is what the loop's correction feedback does, and it
is why granite spent 98 turns on this. A retry costs a full generation and
usually reproduces the same quoting mistake, because the mistake is in how the
engine escapes, not in what it intended. Repair costs microseconds and is
deterministic, which also means it can be tested.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

#: Ordered, cheapest first. The name of the tier that succeeded is reported.
TIER_RAW_CONTROLS = "raw_control_chars"
TIER_TERMINAL_STRING = "terminal_string"
TIER_NATIVE_IN_CONTENT = "native_call_in_content"
TIER_PYTHON_LITERAL = "python_literal"
TIER_TEXT_CALL = "text_call"
TIERS = (TIER_NATIVE_IN_CONTENT, TIER_RAW_CONTROLS, TIER_TERMINAL_STRING)

#: A tool call, whatever the provider called the fields. Both spellings are in
#: the wild: OpenAI-shaped emitters say "name", our own text protocol says
#: "tool", and an engine copying an example from its own training says either.
_NAME_KEYS = ("name", "tool", "function", "tool_name")
_ARG_KEYS = ("arguments", "args", "parameters", "input")

_CONTROL = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

#: Guard against pathological input. A tool call is small; a megabyte of text
#: that happens to contain a brace is not one, and scanning it is wasted time.
MAX_BLOB = 400_000


def _escape_raw_controls(blob: str) -> str:
    """Escape literal newlines/tabs that appear INSIDE a JSON string.

    Tracks string state exactly, so control characters in the JSON's own
    formatting -- the newlines between keys -- are left alone. This is the
    common half of the defect and it is unambiguous: a raw newline inside a
    JSON string is never legal, so escaping it cannot change a valid document.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in blob:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if in_string and ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in _CONTROL:
            out.append(_CONTROL[ch])
            continue
        out.append(ch)
    return "".join(out)


def _python_literal(blob: str) -> Any:
    """Recover an object written as a PYTHON literal instead of JSON.

    ``True``, ``False``, ``None`` and single-quoted strings are what a model
    trained on Python writes when asked for "a JSON object", and json.loads
    rejects all four. The call is otherwise perfect: right tool, right
    arguments, right payload.

    Measured on the expansion cohort: ministral-3:3b emitted one of these in 28
    of 50 runs. Every one of them fell through to ``_terminal_string``, which is
    built for a different failure and mangled them -- so a Python ``True`` in an
    unrelated argument was destroying the argument next to it.

    ``ast.literal_eval`` is the whole implementation because it is exactly the
    right tool: it parses literals and refuses everything else. It cannot call,
    import, or evaluate a name, so a blob that is really code is a SyntaxError
    or ValueError here, not an execution.
    """
    try:
        tree = ast.parse(blob.strip(), mode="eval")
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        del exc
        return None
    # ``literal_eval`` keeps the LAST of a duplicated key, silently. That is the
    # argument-smuggling shape the JSON path already refuses -- {"argv": [safe],
    # "argv": [hostile]} -- and it would have walked straight through this tier.
    # Caught by the suite's own red-team test, not by me.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = []
        for key in node.keys:
            if isinstance(key, ast.Constant):
                if key.value in seen:
                    return None
                seen.append(key.value)
    try:
        parsed = ast.literal_eval(tree)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _closes_its_object(blob: str) -> bool:
    """Whether the blob's braces balance out before the text ends.

    Cheap and deliberately naive: a brace inside a string payload counts too.
    That errs toward "this object closed", which makes ``_terminal_string``
    REFUSE rather than guess -- the safe direction, because the cost of refusing
    is an ordinary recoverable error and the cost of guessing wrong is a
    silently corrupted argument.
    """
    depth = 0
    for ch in blob:
        if ch == "{":
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                return True
    return False


def _text_call(blob: str, known_tools: frozenset[str] | None) -> tuple[str, dict] | None:
    """A call written as ``name(key=value, ...)`` in plain text.

    Gated on ``known_tools`` and unusable without it: the gate is what stops
    ordinary prose containing parentheses from becoming a call. The LAST
    occurrence wins, because a model reasons and then acts.

    Argument parsing is protocol B's, imported here rather than duplicated, so
    the two forms of the same syntax cannot drift apart. The import is local
    because protocol imports this module.
    """
    if not known_tools:
        return None
    from .protocol import _scan_balanced, _split_top_level, _value

    best = None
    for name in known_tools:
        start = 0
        while True:
            at = blob.find(name + "(", start)
            if at == -1:
                at = blob.find(name + " (", start)
                if at == -1:
                    break
            # The name must stand on its own: `finish(` is a call and
            # `unfinish(` is not.
            if at > 0 and (blob[at - 1].isalnum() or blob[at - 1] in "_."):
                start = at + 1
                continue
            if best is None or at > best[0]:
                best = (at, name)
            start = at + 1
    if best is None:
        return None
    at, name = best

    open_at = blob.index("(", at)
    end = _scan_balanced(blob, open_at)
    if end == -1:
        return None
    body = blob[open_at + 1:end - 1].strip()

    arguments: dict = {}
    if body:
        for part in _split_top_level(body, ","):
            pieces = _split_top_level(part, "=")
            if len(pieces) < 2:
                return None                 # not keyword form; refuse, do not guess
            key = pieces[0].strip().strip("\"'")
            if not key.isidentifier():
                return None
            try:
                arguments[key] = _value(part[len(pieces[0]) + 1:].strip())
            except Exception:               # noqa: BLE001
                # _value raises InvalidCall on a value it cannot read. A repair
                # tier must never raise: it recovers or it declines, and the
                # caller's ordinary error is what the model should see.
                return None
    return name, arguments


def _terminal_string(blob: str) -> Any:
    """Recover an object whose LAST string value contains unescaped quotes.

    Unescaped quotes are ambiguous in general: nothing distinguishes a quote
    that ends a string from one that belongs inside it. This tier does not try
    to resolve that in general. It resolves the one case where the structure
    settles it -- the value is the last thing in the object, so its closing
    quote is the last quote before the closing braces, and everything between
    the two is the payload.

    That is exactly the shape a tool call takes when the payload is source code
    with a docstring in it, which is the case that was destroying whole runs. If
    the blob is any other shape this returns None rather than guessing.

    The payload is spliced back in re-escaped, rather than the object being
    reassembled around it: the value is usually nested inside ``arguments``, and
    rebuilding the object would have to know where it belonged. Splicing does
    not need to know.
    """
    tail = re.search(r'"[\s}\]]*$', blob)
    if tail:
        close_at = tail.start()
    elif (blob.rstrip() and blob.rstrip()[-1] not in "}]"
          and not _closes_its_object(blob)):
        # No closing quote anywhere AND the object never closed: the generation
        # was cut off mid-payload (num_predict ran out), so the payload runs to
        # the end and there is no trailing prose to mistake for it.
        #
        # The second half of that condition is not decoration. Without it, ANY
        # text after a complete object -- a closing ``` fence, a note, an
        # explanation -- made this branch swallow the rest of the object into
        # the last argument. Reproduced exactly: a read_file whose ``path`` came
        # back as the file path with the REST OF THE OBJECT glued onto it,
        # closing fence included, and a grep whose ``pattern`` swallowed its
        # own ``context`` and
        # ``ignore_case``. A truncated generation STOPS; it does not close its
        # braces and then keep writing.
        close_at = len(blob)
    else:
        return None

    opens = [m for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"', blob)
             if m.end() <= close_at]
    if not opens:
        return None
    last = opens[-1]

    value = _unescape(blob[last.end():close_at])
    # When the payload ran to the end there is no closing quote to keep, so one
    # is supplied along with the braces below. json.dumps gives the escaping;
    # its own surrounding quotes are stripped because the opening one is already
    # in `blob[:last.end()]`.
    closing = blob[close_at:] if tail else '"'
    spliced = blob[:last.end()] + json.dumps(value)[1:-1] + closing
    spliced = _escape_raw_controls(spliced)
    # A generation cut off mid-object leaves braces unclosed, and the quote that
    # swallowed them is precisely what we just repaired. Closing up to three is
    # enough for {"name":..,"arguments":{..}} and small enough that it cannot
    # turn unrelated text into an object.
    for extra in range(4):
        parsed = _try(_loads, spliced + "}" * extra)
        if isinstance(parsed, dict):
            return parsed
    return None


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\",
            "/": "/", "b": "\b", "f": "\f"}


def _unescape(raw: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            if nxt == "u" and i + 5 < len(raw):
                try:
                    out.append(chr(int(raw[i + 2:i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
        out.append(ch)
        i += 1
    return "".join(out)


def _balanced_objects(text: str) -> list[str]:
    """Every top-level ``{...}`` in *text*, brace-balanced outside strings."""
    blobs: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blobs.append(text[start:i + 1])
                start = -1
            elif depth < 0:
                depth = 0
    if depth > 0 and start >= 0:
        # An unterminated object: the generation was cut off, or a quote inside
        # the payload swallowed the closing brace. Take the rest and let the
        # tiers decide -- _terminal_string will reject it if it is not a tool
        # call, and nothing downstream trusts it before it parses.
        blobs.append(text[start:])
    return blobs


def _as_call(obj: Any) -> tuple[str, dict] | None:
    if not isinstance(obj, dict):
        return None
    # An OpenAI-shaped call nests the real thing under "function".
    fn = obj.get("function")
    if isinstance(fn, dict) and any(k in fn for k in _NAME_KEYS):
        obj = fn
    name = next((obj[k] for k in _NAME_KEYS
                 if isinstance(obj.get(k), str) and obj[k].strip()), None)
    if not name:
        return None
    args: Any = next((obj[k] for k in _ARG_KEYS if k in obj), {})
    if isinstance(args, str):
        # Some emitters put the arguments back into a JSON string. Two tiers
        # deep is where this stops: if that string is also broken, it stays
        # broken.
        try:
            args = json.loads(args)
        except ValueError:
            try:
                args = json.loads(_escape_raw_controls(args))
            except ValueError:
                return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return name.strip(), args


def recover(content: str, *, known_tools: frozenset[str] | None = None
            ) -> tuple[str, dict, str] | None:
    """Best effort at the call *content* was trying to be, or None.

    Returns ``(name, arguments, tier)``. ``known_tools``, when given, is a
    correctness gate and not a rescue: a recovered name that is not a real tool
    is rejected, so this can never manufacture a call the harness would then
    have to reject anyway with a worse error message.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    if len(content) > MAX_BLOB:
        # RED TEAM (Codex): slicing here recovered a write_file whose payload
        # had been cut at the boundary and returned it as if it were whole --
        # 399944 characters of a 400000-character body, ending mid-text, with
        # nothing to say so. A truncated file written as though complete is
        # worse than no call at all, and "never silently corrupt a payload" is
        # the promise this module is built on. So: refuse. The caller gets an
        # ordinary ERROR_NO_TOOL_CALL, which is true and recoverable.
        return None

    candidates = _balanced_objects(content)
    # Unescaped quotes inside the payload desynchronise the brace scanner: it
    # can mistake `"arguments": {` for the start of a new top-level object and
    # hand back a fragment. So the whole text from the first brace is tried too,
    # last and least precise. It costs one more parse attempt and it is the only
    # candidate that survives a payload whose quoting broke the scan itself.
    first = content.find("{")
    if first >= 0 and content[first:] not in candidates:
        candidates.append(content[first:])

    for blob in candidates:
        for tier, candidate in (
            (TIER_NATIVE_IN_CONTENT, _try(_loads, blob)),
            (TIER_RAW_CONTROLS, _try(_loads, _escape_raw_controls(blob))),
            (TIER_PYTHON_LITERAL, _python_literal(blob)),
            (TIER_TERMINAL_STRING, _terminal_string(blob)),
        ):
            if candidate is None:
                continue
            call = _as_call(candidate)
            if call is None:
                continue
            name, args = call
            if known_tools is not None and name not in known_tools:
                continue
            return name, args, tier

    # Last: a call that is not an object at all. The brace scanner produces no
    # candidate for `finish(status="DONE")`, so this runs on the whole text and
    # only once every object-shaped reading has failed.
    written = _text_call(content, known_tools)
    if written is not None:
        return written[0], written[1], TIER_TEXT_CALL
    return None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    """Refuse an object that names the same key twice.

    RED TEAM (Codex): ``{"argv": ["python", "safe"], "argv": ["python", "-c",
    "print(999)"]}`` parses, and json takes the LAST value -- so a call can
    carry a harmless-looking argument and execute a different one. That is
    ordinary json behaviour and the ordinary parser lives with it; repair does
    not have to. This path reassembles text the normal parser already rejected,
    so it is the one place where being stricter costs nothing and refusing an
    ambiguous object is plainly right.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r} in a recovered call")
        seen[key] = value
    return seen


def _loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_no_duplicate_keys)


def _try(fn, *args):
    try:
        return fn(*args)
    except ValueError:
        return None
