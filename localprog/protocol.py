"""Protocol A (native tool-calling) and Protocol B (text), per contract §D.4.

Both exist because the evidence does not yet say whether the known local-model
failures are the models or the way we call them: three Qwen3-family models
returned an empty string, and gpt-oss:20b failed 10/10 -- all of them under
``"format": "json"`` on the raw completions endpoint. Protocol B is how that
question gets answered without changing the frozen prompt.

Protocol B's parser is written by hand rather than with a regex, because the
previous one was
    re.sub(r'(\\w+)\\s*=', r'"\\1":', args)
which rewrites every ``key =`` it finds -- including the ones inside string
values. ``edit(path="a.py", old="x = 1", new="x = 2")`` came out corrupted, so
every protocol-B run failed for a reason that had nothing to do with the model.
The scanner below tracks string state, so text inside quotes is never touched.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from .errors import ERROR_BAD_ARGUMENTS, ERROR_FORMAT, ERROR_NO_TOOL_CALL, InvalidCall

MARKER = "LLAMADA:"
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: Any  # dict, or the raw form for tools.normalise_arguments to coerce


# ------------------------------------------------------------------ scanning


def _scan_balanced(text: str, start: int) -> int:
    """Index just past the ``)`` that closes the ``(`` at *start*.

    String-aware: quotes suspend bracket counting, backslash escapes the next
    character. Returns -1 if the parenthesis never closes.
    """
    depth = 0
    quote = ""
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _split_top_level(blob: str, sep: str) -> list[str]:
    """Split on *sep*, ignoring separators inside strings or brackets."""
    parts: list[str] = []
    depth = 0
    quote = ""
    escape = False
    current: list[str] = []
    for ch in blob:
        if quote:
            current.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
        elif ch in _OPEN:
            depth += 1
            current.append(ch)
        elif ch in _CLOSE:
            depth -= 1
            current.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p for p in (x.strip() for x in parts) if p]


def _value(raw: str) -> Any:
    """JSON first, then Python literals. Never evaluates code."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            f"no pude leer el valor {raw[:80]!r}. Usa comillas dobles y valores JSON.",
        ) from None


def parse_text_call(content: str) -> ParsedCall:
    """Parse ``LLAMADA: nombre(k=v, ...)`` out of a free-text answer."""
    if not isinstance(content, str) or MARKER not in content:
        raise InvalidCall(
            ERROR_FORMAT,
            'usa LLAMADA: herramienta(arg="valor") -- exactamente una por turno.',
        )
    # The LAST marker: models commonly reason first and act last.
    head = content.rindex(MARKER) + len(MARKER)
    rest = content[head:].lstrip()

    name_chars = []
    i = 0
    while i < len(rest) and (rest[i].isalnum() or rest[i] == "_"):
        name_chars.append(rest[i])
        i += 1
    name = "".join(name_chars)
    while i < len(rest) and rest[i].isspace():
        i += 1
    if not name or i >= len(rest) or rest[i] != "(":
        raise InvalidCall(ERROR_FORMAT, 'esperaba LLAMADA: nombre(...) tras el marcador.')

    end = _scan_balanced(rest, i)
    if end == -1:
        raise InvalidCall(ERROR_FORMAT, "el paréntesis de la llamada no se cierra.")
    blob = rest[i + 1 : end - 1].strip()

    arguments: dict[str, Any] = {}
    if blob:
        # A bare JSON object is also accepted: some models answer
        # LLAMADA: edit({"path": ...}) instead of keyword form.
        if blob.startswith("{") and _split_top_level(blob, ",") and blob.endswith("}"):
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    return ParsedCall(name=name, arguments=parsed)
            except ValueError:
                pass
        for part in _split_top_level(blob, ","):
            pieces = _split_top_level(part, "=")
            if len(pieces) < 2:
                raise InvalidCall(
                    ERROR_BAD_ARGUMENTS,
                    f"argumento {part[:60]!r} no tiene forma clave=valor.",
                )
            key = pieces[0].strip().strip("\"'")
            value_text = part[len(pieces[0]) + 1 :].strip()
            arguments[key] = _value(value_text)
    return ParsedCall(name=name, arguments=arguments)


def parse_native_call(message: dict) -> ParsedCall:
    """Pull the first tool call out of a native-protocol message."""
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not calls or not isinstance(calls, list):
        raise InvalidCall(
            ERROR_NO_TOOL_CALL,
            "no emitiste ninguna llamada de herramienta, solo texto. El texto de "
            "una respuesta se descarta: lo unico que hace algo es una llamada.\n"
            "  Si lo que escribiste era el CONTENIDO de un fichero, pasalo como "
            "argumento: write_file(path=..., content=<ese texto>).\n"
            "  Si querias mirar algo, llama a la herramienta que lo mira.",
        )
    first = calls[0]
    if not isinstance(first, dict):
        raise InvalidCall(ERROR_NO_TOOL_CALL, "tool_calls[0] no es un objeto")
    fn = first.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        raise InvalidCall(ERROR_NO_TOOL_CALL, "tool_calls[0].function.name ausente")
    # arguments stays raw: tools.normalise_arguments owns the dict/str coercion,
    # so there is exactly one place that knows what providers actually send.
    return ParsedCall(name=fn["name"], arguments=fn.get("arguments"))


#: Protocol J: one JSON object per turn, optionally inside a fenced block.
#: For models reached through a CLI rather than a tool-calling API (F-13).
JSON_INSTRUCTIONS = """
COMO LLAMAR A UNA HERRAMIENTA

Responde con UN solo objeto JSON, dentro de un bloque ```json, y nada mas
despues de el. Formato exacto:

```json
{"tool": "<nombre>", "arguments": {"<arg>": <valor>}}
```

Ejemplos:

```json
{"tool": "list_dir", "arguments": {"path": "."}}
```

```json
{"tool": "grep", "arguments": {"pattern": "def parse", "context": 3}}
```

```json
{"tool": "edit", "arguments": {"path": "pkg/m.py", "old": "return 1", "new": "return 2"}}
```

Una sola llamada por turno. Puedes razonar antes del bloque; lo que cuenta es
el ultimo bloque JSON de tu respuesta.
"""


def _json_blocks(content: str) -> list[str]:
    """Every ```json fenced block, plus any bare {...} span, last first.

    Last first because models reason and then act, so the final object is the
    decision. Bare spans are accepted because a model that forgets the fence has
    still communicated perfectly clearly, and refusing it would measure fence
    discipline rather than programming.
    """
    blocks: list[str] = []
    lowered = content.lower()
    cursor = 0
    while True:
        start = lowered.find("```json", cursor)
        if start == -1:
            break
        body = content.index("\n", start) + 1 if "\n" in content[start:] else start + 7
        end = content.find("```", body)
        if end == -1:
            blocks.append(content[body:])
            break
        blocks.append(content[body:end])
        cursor = end + 3
    depth = 0
    span_start = None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                span_start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and span_start is not None:
                blocks.append(content[span_start : i + 1])
    return list(reversed(blocks))


def parse_json_call(content: str) -> ParsedCall:
    """Protocol J: ``{"tool": name, "arguments": {...}}``."""
    if not isinstance(content, str) or not content.strip():
        raise InvalidCall(ERROR_NO_TOOL_CALL, "respuesta vacia; llama a una herramienta.")
    for block in _json_blocks(content):
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("tool") or parsed.get("name") or parsed.get("function")
        if isinstance(name, dict):
            name = name.get("name")
        if not isinstance(name, str):
            continue
        arguments = parsed.get("arguments")
        if arguments is None:
            arguments = parsed.get("args")
        if arguments is None:
            arguments = {k: v for k, v in parsed.items()
                         if k not in ("tool", "name", "function", "args", "arguments")}
        return ParsedCall(name=name, arguments=arguments)
    raise InvalidCall(
        ERROR_FORMAT,
        'no encontre una llamada valida. Responde con un bloque ```json que '
        'contenga {"tool": "<nombre>", "arguments": {...}} y nada mas.',
    )


def parse(protocol: str, message: dict) -> ParsedCall:
    if protocol == "A":
        return parse_native_call(message)
    if protocol == "B":
        return parse_text_call(message.get("content", "") if isinstance(message, dict) else "")
    if protocol == "J":
        return parse_json_call(message.get("content", "") if isinstance(message, dict) else "")
    raise InvalidCall(ERROR_FORMAT, f"protocolo desconocido {protocol!r}")
