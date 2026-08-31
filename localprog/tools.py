"""The seven tools of LOCAL_PROGRAMMER_V0_CONTRACT.md, as total functions.

"Total" is the design requirement, not a style note. Every tool here returns a
value or raises ``ToolError``/``InvalidCall`` for every input it can receive --
a path that is a directory, a file that is not utf-8, a regex that does not
compile, an argument that is absent, a symbol in a file that does not parse.
Nothing is left to propagate.

The reason is the defect this harness exists to remove. In the previous
runner, ``list_symbols`` on a file with a syntax error raised ``SyntaxError``
out of the tool, out of the loop, and out of ``main()`` -- past a ``finally``
that deleted the workspace on its way. One bad file ended the campaign and
destroyed its own evidence. Every tool below is written so that cannot happen,
and ``dispatch`` converts anything that still escapes into ``HarnessInvalid``
so the harness names its own bug instead of scoring it as a model failure.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import deps
from .errors import (
    ERROR_BAD_ARGUMENTS,
    ERROR_BAD_PATTERN,
    ERROR_EMPTY_OLD,
    ERROR_FILE_EXISTS,
    ERROR_FILE_NOT_FOUND,
    ERROR_IS_DIRECTORY,
    ERROR_MISSING_ARGUMENT,
    ERROR_MULTIPLE_MATCHES,
    ERROR_NO_MATCH,
    ERROR_NOT_IN_WRITE_SCOPE,
    ERROR_NOT_TEXT,
    ERROR_NOTHING_CHANGED,
    ERROR_PATH_OUTSIDE_REPO,
    ERROR_SYNTAX,
    ERROR_SYNTAX_AFTER_EDIT,
    ERROR_UNKNOWN_TOOL,
    HarnessInvalid,
    InvalidCall,
    ToolError,
)

MAX_READ_LINES = 2000
HEAD_LINES = 200
MAX_GREP_HITS = 50
TEST_TIMEOUT_SECONDS = 120.0
TEST_OUTPUT_TAIL = 3000
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}


@dataclass
class ToolContext:
    """Everything a tool may touch. Nothing global, so tests are cheap."""

    root: Path
    write_scope: tuple[str, ...] = ()
    allowed_new_files: tuple[str, ...] = ()
    acceptance_tests: tuple[str, ...] = ()
    changed_files: set[str] = field(default_factory=set)
    test_timeout: float = TEST_TIMEOUT_SECONDS
    #: Set once a run_tests call reported every declared test green. Read by
    #: ``finish`` only for reporting; the authoritative verdict is taken by the
    #: caller after the loop, never from the model's own claim.
    tests_green: bool = False

    def guard(self):
        try:
            return deps.guard.MissionGuard(build_root=self.root)
        except Exception as exc:  # root vanished mid-run: a harness/env defect
            raise HarnessInvalid(f"cannot build MissionGuard on {self.root}: {exc}") from exc


# --------------------------------------------------------------- path helpers


def _normalise(relative: Any) -> str:
    """Accept what a model actually emits, without weakening containment.

    ``guard.validate_relative_path`` accepts forward slashes only, so a model
    that writes ``pkg\\mod.py`` on Windows would be refused for a reason that
    has nothing to do with safety. Translating separators is a kindness;
    ``..``, absolute paths and device names are still rejected downstream,
    which is where the safety actually lives.
    """
    if not isinstance(relative, str):
        raise ToolError(ERROR_PATH_OUTSIDE_REPO, f"path must be a string, got {type(relative).__name__}")
    return relative.replace("\\", "/").strip()


def _resolve(ctx: ToolContext, relative: Any, *, must_exist: bool = False) -> tuple[str, Path]:
    rel = _normalise(relative)
    if not rel:
        raise ToolError(ERROR_PATH_OUTSIDE_REPO, "path is empty")
    try:
        target = ctx.guard().resolve_inside(ctx.root, rel, must_exist=must_exist)
    except deps.guard.ContainmentError as exc:
        raise ToolError(ERROR_PATH_OUTSIDE_REPO, f"{rel!r}: {exc}") from exc
    except OSError as exc:
        raise ToolError(ERROR_PATH_OUTSIDE_REPO, f"{rel!r}: {exc}") from exc
    return rel, target


def _siblings(target: Path, limit: int = 12) -> str:
    try:
        names = sorted(p.name for p in target.parent.iterdir() if p.name not in SKIP_DIRS)
    except OSError:
        return "(no se pudo listar el directorio)"
    if not names:
        return "(directorio vacío)"
    shown = ", ".join(names[:limit])
    return shown + (f", ... (+{len(names) - limit} más)" if len(names) > limit else "")


def _read_text(rel: str, target: Path) -> str:
    """The bytes at *target* as text, or the right ToolError. Never raises OSError."""
    if not target.exists():
        raise ToolError(
            ERROR_FILE_NOT_FOUND,
            f"{rel!r} no existe. Ficheros en ese directorio: {_siblings(target)}",
        )
    if target.is_dir():
        raise ToolError(ERROR_IS_DIRECTORY, f"{rel!r} es un directorio. Contiene: {_siblings(target / 'x')}")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(ERROR_NOT_TEXT, f"{rel!r} no es texto utf-8") from None
    except OSError as exc:
        raise ToolError(ERROR_FILE_NOT_FOUND, f"{rel!r}: {exc}") from None


def _write_text(target: Path, text: str) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".localprog.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(target)
    except OSError as exc:  # disk full, permissions: not the model's doing
        raise HarnessInvalid(f"cannot write {target}: {exc}") from exc


def _check_writable(ctx: ToolContext, rel: str, *, creating: bool) -> None:
    allowed = set(ctx.write_scope) | (set(ctx.allowed_new_files) if creating else set())
    if rel not in allowed:
        raise ToolError(
            ERROR_NOT_IN_WRITE_SCOPE,
            f"{rel!r} no está en write_scope.\n  Puedes escribir en: {', '.join(sorted(allowed)) or '(nada)'}",
        )


# ---------------------------------------------------------------- the 7 tools


def read_file(ctx: ToolContext, path: Any, start: Any = None, end: Any = None) -> str:
    rel, target = _resolve(ctx, path)
    lines = _read_text(rel, target).splitlines()
    total = len(lines)

    if start is None and end is None and total > MAX_READ_LINES:
        body = "\n".join(f"{i:4}\t{l}" for i, l in enumerate(lines[:HEAD_LINES], 1))
        return (
            body
            + f"\n[fichero de {total} líneas truncado. Usa start/end, o "
            f"list_symbols('{rel}') para ver su estructura]"
        )

    lo = 1 if start is None else start
    hi = total if end is None else end
    if not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool) or isinstance(hi, bool):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "start y end deben ser enteros o null")
    lo = max(1, lo)
    hi = min(total, hi)
    if lo > hi or total == 0:
        return f"[rango vacío: el fichero tiene {total} líneas]"
    return "\n".join(f"{i:4}\t{lines[i - 1]}" for i in range(lo, hi + 1))


def grep(ctx: ToolContext, pattern: Any, glob: Any = "**/*.py") -> Any:
    if not isinstance(pattern, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "pattern debe ser una cadena")
    glob = glob if isinstance(glob, str) and glob else "**/*.py"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(ERROR_BAD_PATTERN, str(exc)) from None

    hits: list[dict] = []
    truncated = False
    try:
        candidates = sorted(ctx.root.glob(glob))
    except (OSError, ValueError, IndexError) as exc:
        # An invalid glob (e.g. "**") is the model's mistake, not a crash.
        raise ToolError(ERROR_BAD_PATTERN, f"glob {glob!r} inválido: {exc}") from None

    for p in candidates:
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # a binary or unreadable file is not a grep failure
        rel = p.relative_to(ctx.root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                if len(hits) >= MAX_GREP_HITS:
                    truncated = True
                    break
                hits.append({"file": rel, "line": i, "text": line[:400]})
        if truncated:
            break

    if truncated:
        return {"hits": hits, "note": f"[más de {MAX_GREP_HITS} coincidencias, mostrando {MAX_GREP_HITS}. Afina el patrón o restringe el glob]"}
    if not hits:
        return {"hits": [], "note": f"[0 coincidencias para {pattern!r} en {glob!r}]"}
    return {"hits": hits}


def list_symbols(ctx: ToolContext, path: Any) -> list[str]:
    rel, target = _resolve(ctx, path)
    source = _read_text(rel, target)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolError(ERROR_SYNTAX, f"{rel!r} línea {exc.lineno}: {exc.msg}") from None
    except (ValueError, RecursionError) as exc:  # null bytes, pathological nesting
        raise ToolError(ERROR_SYNTAX, f"{rel!r}: {exc}") from None

    lineno: dict[str, int] = {}

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                dotted = child.name if not prefix else f"{prefix}.{child.name}"
                lineno.setdefault(dotted, child.lineno)
                walk(child, dotted)

    walk(tree, "")
    # edit_channel decides what a symbol IS, so this harness and Gafotas' edit
    # path can never disagree about a name. Line numbers are ours.
    names = deps.edit_channel.list_symbols(source)
    return [f"{n} (línea {lineno[n]})" if n in lineno else n for n in names]


def edit(ctx: ToolContext, path: Any, old: Any, new: Any) -> str:
    rel, target = _resolve(ctx, path)
    _check_writable(ctx, rel, creating=False)
    if not isinstance(old, str) or not isinstance(new, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "old y new deben ser cadenas")
    if not old:
        raise ToolError(ERROR_EMPTY_OLD, "old no puede estar vacío. Para un fichero nuevo usa write_file.")

    text = _read_text(rel, target)
    count = text.count(old)
    if count == 0:
        preview = "\n".join("    " + l for l in old.splitlines()[:5])
        raise ToolError(
            ERROR_NO_MATCH,
            f"no se encontró el texto en {rel!r}.\n  buscado ({len(old.splitlines())} líneas):\n{preview}\n"
            f"  El fichero tiene {len(text.splitlines())} líneas. Usa read_file({rel!r}) para ver "
            f"el texto exacto, incluida la indentación.",
        )
    if count > 1:
        at = []
        idx = text.find(old)
        while idx != -1:
            at.append(text.count("\n", 0, idx) + 1)
            idx = text.find(old, idx + 1)
        raise ToolError(
            ERROR_MULTIPLE_MATCHES,
            f"el texto aparece {count} veces en {rel!r} (líneas {', '.join(map(str, at))}).\n"
            f"  Añade líneas de contexto alrededor para que sea único.",
        )

    candidate = text.replace(old, new, 1)
    if rel.endswith(".py"):
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            raise ToolError(
                ERROR_SYNTAX_AFTER_EDIT,
                f"la edición dejaría {rel!r} sin poder parsearse:\n  línea {exc.lineno}: {exc.msg}\n"
                f"  NO se ha escrito nada. El fichero sigue como estaba.",
            ) from None
        except ValueError as exc:
            raise ToolError(ERROR_SYNTAX_AFTER_EDIT, f"{rel!r}: {exc}. NO se ha escrito nada.") from None

    _write_text(target, candidate)
    ctx.changed_files.add(rel)
    return f"edit aplicada en {rel}"


def write_file(ctx: ToolContext, path: Any, content: Any) -> str:
    rel, target = _resolve(ctx, path)
    _check_writable(ctx, rel, creating=True)
    if not isinstance(content, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "content debe ser una cadena")
    if target.exists():
        raise ToolError(ERROR_FILE_EXISTS, f"{rel!r} ya existe. Usa edit.")
    if rel.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as exc:
            raise ToolError(
                ERROR_SYNTAX_AFTER_EDIT,
                f"{rel!r} no parsea: línea {exc.lineno}: {exc.msg}. NO se ha escrito nada.",
            ) from None
        except ValueError as exc:
            raise ToolError(ERROR_SYNTAX_AFTER_EDIT, f"{rel!r}: {exc}. NO se ha escrito nada.") from None
    _write_text(target, content)
    ctx.changed_files.add(rel)
    return f"write_file aplicada en {rel}"


def run_tests(ctx: ToolContext, node_ids: Any = None) -> dict:
    if node_ids is None:
        targets = list(ctx.acceptance_tests)
    elif isinstance(node_ids, list) and all(isinstance(x, str) for x in node_ids):
        targets = node_ids
    else:
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "node_ids debe ser una lista de cadenas o null")
    if not targets:
        raise ToolError(ERROR_FILE_NOT_FOUND, "no hay tests declarados y no diste node_ids")

    # A node id is "<path>::<selector>"; only the path part is a filesystem
    # claim, and only that part is checked. argv is built here and never taken
    # from the model.
    for t in targets:
        head = _normalise(t).split("::", 1)[0]
        _resolve(ctx, head)

    argv = [sys.executable, "-B", "-m", "pytest", "-q", *targets]
    try:
        proc = subprocess.run(
            argv, cwd=str(ctx.root), capture_output=True, text=True,
            timeout=ctx.test_timeout, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + \
              ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        return {"passed": False, "exit_code": None, "timed_out": True,
                "output": (out[-TEST_OUTPUT_TAIL:] + f"\nTIMEOUT tras {ctx.test_timeout:g}s")}
    except OSError as exc:
        raise HarnessInvalid(f"cannot run pytest: {exc}") from exc

    passed = proc.returncode == 0
    # Green means the DECLARED suite passed. A model that names those tests
    # explicitly has done the same work as one that passed null, so compare
    # the target sets rather than the argument shape.
    if passed and set(ctx.acceptance_tests) <= {_normalise(t) for t in targets}:
        ctx.tests_green = True
    return {
        "passed": passed,
        "exit_code": proc.returncode,
        "timed_out": False,
        "output": (proc.stdout + proc.stderr)[-TEST_OUTPUT_TAIL:],
    }


def finish(ctx: ToolContext, summary: Any) -> str:
    if not isinstance(summary, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "summary debe ser una cadena")
    if not ctx.changed_files:
        raise ToolError(
            ERROR_NOTHING_CHANGED,
            "no has hecho ninguna edición. Termina solo cuando hayas cambiado algo, o si de "
            "verdad no hay nada que hacer explica por qué.",
        )
    return "FINISHED"


# ------------------------------------------------------------------ dispatch

#: name -> (required, optional). The single source of truth for both the
#: native tool schema sent to the provider and the argument check below, so
#: the two can never drift -- the previous runner advertised ``list_dir`` in
#: its schema and had no such tool, and advertised ``run`` while the contract
#: says ``run_tests``.
SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "read_file": (("path",), ("start", "end")),
    "grep": (("pattern",), ("glob",)),
    "list_symbols": (("path",), ()),
    "edit": (("path", "old", "new"), ()),
    "write_file": (("path", "content"), ()),
    "run_tests": ((), ("node_ids",)),
    "finish": (("summary",), ()),
}

_IMPL = {
    "read_file": read_file,
    "grep": grep,
    "list_symbols": list_symbols,
    "edit": edit,
    "write_file": write_file,
    "run_tests": run_tests,
    "finish": finish,
}

assert set(SPECS) == set(_IMPL), "SPECS and _IMPL must describe the same tools"


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call produced, classified.

    ``ok`` -> ``value`` is the result. Otherwise ``feedback`` goes to the model
    and exactly one of ``tool_error`` / ``invalid_call`` is true, which is what
    the frozen scoring counts.
    """

    name: str
    ok: bool
    value: Any = None
    feedback: str | None = None
    code: str | None = None
    tool_error: bool = False
    invalid_call: bool = False


def normalise_arguments(raw: Any) -> dict:
    """Coerce what providers actually send into a dict, or say why not.

    Ollama returns ``arguments`` as an object for some models and as a JSON
    string for others. Both are accepted; anything else is an invalid call
    rather than an ``AttributeError`` three frames deeper.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json

        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise InvalidCall(ERROR_BAD_ARGUMENTS, f"arguments no es JSON válido: {exc}") from None
        if not isinstance(parsed, dict):
            raise InvalidCall(ERROR_BAD_ARGUMENTS, "arguments debe ser un objeto")
        return parsed
    raise InvalidCall(ERROR_BAD_ARGUMENTS, f"arguments debe ser un objeto, no {type(raw).__name__}")


def dispatch(ctx: ToolContext, name: Any, raw_args: Any) -> ToolOutcome:
    """Run one tool call. The only place tool exceptions are allowed to end.

    Contract, and the reason this function exists:

      * ToolError      -> ToolOutcome(ok=False, tool_error=True). Loop continues.
      * InvalidCall    -> ToolOutcome(ok=False, invalid_call=True). Loop continues.
      * anything else  -> HarnessInvalid. The run is voided.

    That last line is the whole design. A tool that hits a condition its author
    did not foresee stops the measurement and names the defect, instead of
    being silently recorded as the model failing.
    """
    try:
        if not isinstance(name, str) or name not in SPECS:
            known = ", ".join(sorted(SPECS))
            raise InvalidCall(ERROR_UNKNOWN_TOOL, f"{name!r} no existe. Herramientas: {known}")
        args = normalise_arguments(raw_args)
        required, optional = SPECS[name]
        missing = [k for k in required if k not in args]
        if missing:
            raise InvalidCall(
                ERROR_MISSING_ARGUMENT,
                f"{name} necesita {', '.join(missing)}. Recibido: {sorted(args) or '(nada)'}",
            )
        unknown = [k for k in args if k not in required + optional]
        call = {k: v for k, v in args.items() if k in required + optional}
        value = _IMPL[name](ctx, **call)
        if unknown:
            # Extra keys are tolerated but reported; refusing here would fail
            # a call that is otherwise perfectly good.
            return ToolOutcome(name=name, ok=True, value=value,
                               feedback=f"[aviso: argumentos ignorados: {', '.join(sorted(unknown))}]")
        return ToolOutcome(name=name, ok=True, value=value)
    except ToolError as exc:
        return ToolOutcome(name=str(name), ok=False, feedback=exc.feedback(), code=exc.code, tool_error=True)
    except InvalidCall as exc:
        return ToolOutcome(name=str(name), ok=False, feedback=exc.feedback(), code=exc.code, invalid_call=True)
    except HarnessInvalid:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all, see docstring
        raise HarnessInvalid(
            f"UNHANDLED_TOOL_ERROR in {name!r}: {type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        ) from exc


def native_schema() -> list[dict]:
    """The provider-side tool schema, generated from SPECS so it cannot drift."""
    types = {
        "path": {"type": "string"},
        "start": {"type": ["integer", "null"]},
        "end": {"type": ["integer", "null"]},
        "pattern": {"type": "string"},
        "glob": {"type": "string"},
        "old": {"type": "string"},
        "new": {"type": "string"},
        "content": {"type": "string"},
        "node_ids": {"type": ["array", "null"], "items": {"type": "string"}},
        "summary": {"type": "string"},
    }
    out = []
    for name, (required, optional) in SPECS.items():
        props = {k: types[k] for k in required + optional}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(required),
                },
            },
        })
    return out
