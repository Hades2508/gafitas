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
from .scope import WriteScope
from .errors import (
    ERROR_BAD_ARGUMENTS,
    ERROR_BAD_PATTERN,
    ERROR_COMMAND_NOT_ALLOWED,
    ERROR_EMPTY_DIRECTORY,
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
    ERROR_TOO_MANY_ENTRIES,
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
FINISH_STATUSES = ("DONE", "NO_CHANGE", "BLOCKED")
MAX_DIR_ENTRIES = 200
RUN_TIMEOUT_SECONDS = 120.0
RUN_OUTPUT_TAIL = 4000

#: What ``run`` may execute. An allowlist and not a denylist, because a
#: denylist on a general-purpose machine is a wish, not a boundary. Every
#: entry is resolved against sys.executable or found on PATH; nothing is ever
#: passed through a shell, so operators like ``&&``, ``|`` and ``>`` are inert
#: characters in an argv element rather than syntax.
#:
#: ``python`` covers the overwhelming majority of what a Python agent needs to
#: observe: run a script, import a module, print a value, drive pytest with
#: flags this harness does not model. ``git`` is read-only in practice here
#: (the workspace is a detached worktree and nothing is pushed) and is how an
#: agent inspects what it has actually changed.
ALLOWED_COMMANDS: dict[str, tuple[str, ...]] = {
    "python": (),
    "pytest": ("-m", "pytest"),
    "git": (),
}

#: git subcommands ``run`` will pass through. Read-only inspection only: the
#: workspace IS the rollback (INVARIANTS I7), so an agent that could commit or
#: check out would be able to defeat the evidence trail without leaving the
#: sandbox.
GIT_READONLY = frozenset({
    "status", "diff", "log", "show", "ls-files", "blame", "grep", "rev-parse", "cat-file",
})


@dataclass
class ToolContext:
    """Everything a tool may touch. Nothing global, so tests are cheap."""

    root: Path
    write_scope: tuple[str, ...] = ()
    allowed_new_files: tuple[str, ...] = ()
    acceptance_tests: tuple[str, ...] = ()
    changed_files: set[str] = field(default_factory=set)
    test_timeout: float = TEST_TIMEOUT_SECONDS
    run_timeout: float = RUN_TIMEOUT_SECONDS
    #: Set once a run_tests call reported every declared test green. Read by
    #: ``finish`` only for reporting; the authoritative verdict is taken by the
    #: caller after the loop, never from the model's own claim.
    tests_green: bool = False
    #: How the agent said it was ending, if it ended deliberately. ``finish``
    #: writes it and the caller reads it, which is what lets 'I am done' be
    #: told apart from 'I cannot do this' -- a distinction the old
    #: changed-something-or-refuse finish could not express (F-10).
    finish_status: str | None = None
    finish_summary: str = ''
    #: Every command ``run`` executed, in order. Cheap to keep, and it is
    #: what makes a debugging session reproducible after the fact.
    commands_run: list[dict] = field(default_factory=list)
    _scope: WriteScope | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._scope = WriteScope(tuple(self.write_scope), tuple(self.allowed_new_files))

    @property
    def scope(self) -> WriteScope:
        # Defensive: a ToolContext built without __init__ (object.__new__, or
        # a test that bypasses it) would otherwise hand None to
        # _check_writable and surface as an AttributeError, which dispatch
        # would correctly but unhelpfully report as HARNESS_INVALID.
        if self._scope is None:
            self._scope = WriteScope(tuple(self.write_scope), tuple(self.allowed_new_files))
        return self._scope

    def guard(self):
        try:
            return deps.guard.MissionGuard(build_root=self.root)
        except Exception as exc:  # root vanished mid-run: a harness/env defect
            raise HarnessInvalid(f"cannot build MissionGuard on {self.root}: {exc}") from exc


# --------------------------------------------------------------- path helpers


def _normalise(relative: Any) -> str:
    """Accept what a model actually emits, without weakening containment.

    Two separate jobs live here, and only the first is cosmetic.

    Separators. ``guard.validate_relative_path`` accepts forward slashes only,
    so a model writing ``pkg\\mod.py`` on Windows would be refused for a reason
    with nothing to do with safety.

    Redundant prefixes. The guard also rejects any path containing a ``.``
    component, which is correct for ``..`` and needlessly hostile for ``./``.
    A model writing ``./pkg/mod.py`` or ``list_dir(".")`` means something
    perfectly ordinary, and answering ERROR_PATH_OUTSIDE_REPO teaches it that
    the repository root is outside the repository. So ``./`` prefixes and
    duplicate slashes are folded away here, and a path that reduces to nothing
    becomes ``""`` -- which callers read as "the root itself".

    What is NOT relaxed: ``..`` in any position, absolute paths, drive letters,
    device names and symlink escapes. All of those are still decided by
    ``programmer.guard``, which is where the safety actually lives. This
    function makes the guard reachable; it never speaks for it.
    """
    if not isinstance(relative, str):
        raise ToolError(
            ERROR_PATH_OUTSIDE_REPO, f"path must be a string, got {type(relative).__name__}"
        )
    text = relative.replace("\\", "/").strip()
    while "//" in text:
        text = text.replace("//", "/")
    while "/./" in text:
        text = text.replace("/./", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.endswith("/") and len(text) > 1:
        text = text.rstrip("/")
    if text in (".", "/"):
        text = ""
    return text


def _resolve_dir(ctx: ToolContext, relative: Any) -> tuple[str, Path]:
    """Like ``_resolve``, but ``""`` (the repository root) is a legal answer.

    The root is trivially inside the root, and the guard has no vocabulary for
    saying so -- it validates path COMPONENTS, and the root has none. Rather
    than invent a component for it, the root is handled here and every other
    path goes through the normal, guarded route untouched.
    """
    rel = _normalise(relative)
    if not rel:
        return ".", ctx.root
    return _resolve(ctx, rel)


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
    """Refuse a write the mission did not authorise.

    Containment is NOT this function's job. ``_resolve`` already put the
    path through ``programmer.guard``, so by the time we arrive ``rel`` is
    known to be inside the repository. What is decided here is the narrower,
    mission-level question of which in-repo paths the agent was told it may
    touch -- and since F-05 that is answered by a pattern language instead of
    string equality against a list the mission author had to guess in advance.
    """
    if not ctx.scope.allows(rel, creating=creating):
        verb = "crear" if creating else "modificar"
        raise ToolError(
            ERROR_NOT_IN_WRITE_SCOPE,
            f"no puedes {verb} {rel!r}: esta fuera de write_scope." \
            f"\n  Ambito de escritura: {ctx.scope.describe(creating=creating)}" \
            f"\n  (un ambito que acaba en '/' incluye todo lo que hay debajo)",
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


def list_dir(ctx: ToolContext, path: Any = ".") -> dict:
    """What is in a directory. The tool whose absence crippled exploration.

    F-03: there was no way to enumerate a directory at all, so an agent facing
    an unfamiliar repository had to guess filenames or grep blindly for them.
    The tool did exist in an earlier runner, where it raised NotADirectoryError
    when handed a file and took the whole campaign down with it -- which is why
    the contract deleted it instead of fixing it. It is reinstated here as a
    total function: every path it can be handed yields a value or a ToolError.

    A file is not an error. Being pointed at ``pkg/mod.py`` when you meant
    ``pkg/`` is an ordinary slip, and the useful answer is that file's own
    directory plus a note saying so, not a refusal.
    """
    rel, target = _resolve_dir(ctx, path if path is not None else "")

    note = None
    if target.exists() and target.is_file():
        note = f"{rel!r} es un fichero; te muestro el directorio que lo contiene."
        target = target.parent
        rel = target.relative_to(ctx.root).as_posix() if target != ctx.root else "."
    if not target.exists():
        raise ToolError(
            ERROR_FILE_NOT_FOUND,
            f"{rel!r} no existe. En su directorio padre hay: {_siblings(target)}",
        )

    try:
        entries = sorted(target.iterdir(), key=lambda q: (q.is_file(), q.name.lower()))
    except OSError as exc:
        raise ToolError(ERROR_FILE_NOT_FOUND, f"no se pudo listar {rel!r}: {exc}") from None

    dirs: list[str] = []
    files: list[dict] = []
    for entry in entries:
        if entry.name in SKIP_DIRS:
            continue
        try:
            if entry.is_dir():
                dirs.append(entry.name + "/")
            else:
                files.append({"name": entry.name, "bytes": entry.stat().st_size})
        except OSError:
            continue  # vanished or unreadable between iterdir and stat

    total = len(dirs) + len(files)
    if total == 0:
        raise ToolError(ERROR_EMPTY_DIRECTORY, f"{rel!r} esta vacio.")

    out: dict = {
        "path": rel,
        "dirs": dirs[:MAX_DIR_ENTRIES],
        "files": files[: max(0, MAX_DIR_ENTRIES - len(dirs))],
    }
    shown = len(out["dirs"]) + len(out["files"])
    notes = []
    if shown < total:
        notes.append(
            f"[{ERROR_TOO_MANY_ENTRIES}: {total} entradas, mostrando {shown}. "
            f"Entra en un subdirectorio para ver el resto]"
        )
    if note:
        notes.append(note)
    if notes:
        out["note"] = " ".join(notes)
    return out


def _resolve_command(argv: list[str]) -> tuple[list[str], str]:
    """Turn the model's argv into a real, allowlisted command line.

    Returns the argv actually executed plus the allowlist key it matched.
    Anything not on the list raises ToolError -- see ALLOWED_COMMANDS for why
    this is an allowlist and not a denylist.
    """
    head = argv[0].replace("\\", "/").rsplit("/", 1)[-1]
    if head.lower().endswith(".exe"):
        head = head[:-4]
    head = head.lower()
    own = Path(sys.executable).stem.lower()
    if head in ("python", "python3", "py", own):
        head = "python"
    if head not in ALLOWED_COMMANDS:
        raise ToolError(
            ERROR_COMMAND_NOT_ALLOWED,
            f"{argv[0]!r} no se puede ejecutar aqui. Permitidos: "
            f"{', '.join(sorted(ALLOWED_COMMANDS))}. Para ejecutar codigo Python usa "
            f'["python", "-c", "..."] o ["python", "ruta/al/script.py"].',
        )
    if head == "git":
        subcommand = next((a for a in argv[1:] if not a.startswith("-")), None)
        if subcommand not in GIT_READONLY:
            raise ToolError(
                ERROR_COMMAND_NOT_ALLOWED,
                f"git {subcommand!r} no esta permitido; solo inspeccion: "
                f"{', '.join(sorted(GIT_READONLY))}.",
            )
        return ["git", *argv[1:]], "git"
    if head == "pytest":
        return [sys.executable, "-B", "-m", "pytest", *argv[1:]], "pytest"
    return [sys.executable, "-B", *argv[1:]], "python"


def run(ctx: ToolContext, argv: Any, timeout: Any = None) -> dict:
    """Execute one allowlisted command in the workspace and report what it did.

    F-04: before this, the only thing that could execute was ``run_tests``, and
    only against test node ids the mission had declared in advance. That removes
    the entire RUN -> OBSERVE -> DEBUG half of the loop. The agent could not
    reproduce a bug, print an intermediate value, check that an import resolves,
    or run a test it had just written. It could only ask "is the declared suite
    green yet" over and over, with no way to find out why it was not.

    Safety here is structural, not advisory:

      * argv is a LIST and ``shell=False``. There is no shell, so ``&&``, ``|``
        and ``>`` are ordinary characters inside a single argument, not syntax.
      * the executable must match ALLOWED_COMMANDS, and git is further narrowed
        to read-only subcommands because the workspace IS the rollback (I7).
      * cwd is the workspace root. The real repository is never mounted here.
      * a hard wall-clock timeout, with output truncated before it is returned.

    One residual risk, named rather than papered over: ``python -c`` can write
    anywhere the OS user can write, and this allowlist does not close that. What
    it does close is the accidental case -- an agent meaning to fix a bug that
    reaches outside its scope -- and the post-loop write verification detects any
    file the tools did not author. A deliberate escape would require the model to
    target an absolute path outside the box on purpose, which is a threat model
    for process isolation, not for a tool surface.
    """
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            "argv debe ser una lista de cadenas no vacia, p.ej. "
            '["python", "-c", "import pkg; print(pkg.f(1))"]',
        )
    if timeout is None:
        limit = ctx.run_timeout
    elif isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and 0 < timeout <= 600:
        limit = float(timeout)
    else:
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            "timeout debe ser un numero de segundos entre 0 y 600, o null",
        )

    real_argv, kind = _resolve_command(argv)
    try:
        proc = subprocess.run(
            real_argv, cwd=str(ctx.root), capture_output=True, text=True,
            timeout=limit, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        head = exc.stdout if isinstance(exc.stdout, str) else ""
        tail = exc.stderr if isinstance(exc.stderr, str) else ""
        ctx.commands_run.append({"argv": argv, "exit_code": None, "timed_out": True})
        return {
            "argv": argv, "kind": kind, "exit_code": None, "timed_out": True,
            "output": (head + tail)[-RUN_OUTPUT_TAIL:] + f"\n[TIMEOUT tras {limit:g}s]",
        }
    except OSError as exc:
        # Allowlisted but not installed. That is a fact about this machine the
        # agent can act on, not a defect in the harness.
        raise ToolError(
            ERROR_COMMAND_NOT_ALLOWED, f"no se pudo ejecutar {argv[0]!r}: {exc}"
        ) from None

    output = proc.stdout + proc.stderr
    ctx.commands_run.append({"argv": argv, "exit_code": proc.returncode, "timed_out": False})
    result = {
        "argv": argv, "kind": kind, "exit_code": proc.returncode,
        "timed_out": False, "output": output[-RUN_OUTPUT_TAIL:],
    }
    if len(output) > RUN_OUTPUT_TAIL:
        result["note"] = f"[salida truncada a los ultimos {RUN_OUTPUT_TAIL} caracteres]"
    return result


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


def finish(ctx: ToolContext, summary: Any, status: Any = "DONE") -> str:
    """End the turn deliberately, saying which kind of ending this is.

    F-10: ``finish`` used to refuse unless something had been edited, and the
    only way to say "there is genuinely nothing to do" was to call it twice and
    have the loop recognise the repeat. That left no way at all to say the third
    thing, which is the one that matters most in real work: *I cannot do this,
    and here is what stopped me.* A programmer that cannot report being blocked
    reports being finished instead, which is worse than failing.

    Three statuses, and the harness treats them very differently:

      DONE      the agent believes the work is complete. This is a CLAIM, not a
                verdict -- I9 still holds and the caller decides after the loop.
      NO_CHANGE the agent examined the task and concluded no change is needed.
      BLOCKED   the agent cannot proceed. Never a pass, under any circumstances.

    DONE with nothing changed is still refused, because that is the specific
    confusion the original check existed to catch: an agent that has done
    nothing and believes it is done. It is told to use NO_CHANGE or BLOCKED,
    both of which are available and neither of which requires an edit.
    """
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "summary debe ser una cadena no vacia")
    if not isinstance(status, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "status debe ser una cadena")
    normalised = status.strip().upper() or "DONE"
    if normalised not in FINISH_STATUSES:
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            f"status debe ser uno de {', '.join(FINISH_STATUSES)}; recibido {status!r}",
        )
    if normalised == "DONE" and not ctx.changed_files:
        raise ToolError(
            ERROR_NOTHING_CHANGED,
            "no has editado nada, asi que no puedes terminar con status='DONE'.\n"
            "  Si de verdad no hace falta ningun cambio: finish(status='NO_CHANGE', "
            "summary='<por que>').\n"
            "  Si no puedes continuar: finish(status='BLOCKED', summary='<que te lo impide>').",
        )
    ctx.finish_status = normalised
    ctx.finish_summary = summary.strip()
    return f"FINISHED[{normalised}]"


# ------------------------------------------------------------------ dispatch

#: name -> (required, optional). The single source of truth for both the
#: native tool schema sent to the provider and the argument check below, so
#: the two can never drift -- the previous runner advertised ``list_dir`` in
#: its schema and had no such tool, and advertised ``run`` while the contract
#: says ``run_tests``.
SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "read_file": (("path",), ("start", "end")),
    "list_dir": ((), ("path",)),
    "grep": (("pattern",), ("glob",)),
    "list_symbols": (("path",), ()),
    "edit": (("path", "old", "new"), ()),
    "write_file": (("path", "content"), ()),
    "run": (("argv",), ("timeout",)),
    "run_tests": ((), ("node_ids",)),
    "finish": (("summary",), ("status",)),
}

_IMPL = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "list_symbols": list_symbols,
    "edit": edit,
    "write_file": write_file,
    "run": run,
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


# ------------------------------------------------------------------- schema

#: Prose for the provider-side schema. Kept next to SPECS and checked against
#: it by the assert below, so a tool can never be advertised without an
#: explanation or explained without existing.
#:
#: Why this exists (audit finding F-02). ``native_schema`` used to emit
#: ``"description": name`` for every tool and nothing at all for parameters.
#: The model was handed nine function signatures with no semantics: no way to
#: learn that ``edit`` needs ``old`` to appear exactly once, that ``write_file``
#: refuses to overwrite, or that ``run`` takes an argv list rather than a
#: command string. qwen2.5-coder:7b then scored 0/5 with 12 invalid calls out of
#: 12 in every protocol-A run -- a number that was read as a verdict on the
#: model. It is a verdict on this dictionary being empty.
#:
#: These strings are production behaviour, exactly like the ToolError texts:
#: they are the entire interface specification the model ever receives.
TOOL_DOC: dict[str, str] = {
    "read_file": (
        "Lee un fichero de texto del repositorio y devuelve sus lineas numeradas. "
        "Usa start/end para leer solo un tramo de un fichero grande. "
        "Puedes leer CUALQUIER fichero del repositorio, no solo los que puedes editar."
    ),
    "list_dir": (
        "Lista lo que hay en un directorio: subdirectorios y ficheros con su tamano. "
        "Empieza por aqui cuando no conozcas la estructura del repositorio. "
        "Sin argumentos lista la raiz."
    ),
    "grep": (
        "Busca una expresion regular de Python en los ficheros del repositorio y "
        "devuelve fichero, linea y texto de cada coincidencia. Es la forma de "
        "encontrar donde se define o se usa algo cuando no sabes en que fichero esta."
    ),
    "list_symbols": (
        "Devuelve las funciones y clases definidas en un fichero Python, con su "
        "numero de linea, en forma 'Clase.metodo'. Mas barato que leer el fichero "
        "entero cuando solo quieres saber que hay dentro."
    ),
    "edit": (
        "Sustituye un fragmento de texto exacto por otro dentro de un fichero que "
        "ya existe. El fragmento 'old' debe aparecer EXACTAMENTE UNA VEZ en el "
        "fichero, con su indentacion original; si aparece varias veces, anade "
        "lineas de contexto alrededor hasta que sea unico. Si el resultado no "
        "seria Python valido, no se escribe nada y te lo digo. Puedes hacer varias "
        "ediciones seguidas sobre el mismo fichero."
    ),
    "write_file": (
        "Crea un fichero NUEVO con el contenido dado. Falla si el fichero ya "
        "existe: para modificar uno existente usa edit."
    ),
    "run": (
        "Ejecuta un comando y devuelve su codigo de salida y su salida combinada. "
        'argv es una LISTA de cadenas, no una cadena: ["python", "-c", "print(1)"]. '
        "No hay shell, asi que no funcionan pipes ni redirecciones. Permitidos: "
        "python (ejecutar un script o -c para una expresion), pytest, y git de solo "
        "lectura (status, diff, log, ls-files...). Usalo para reproducir un fallo, "
        "imprimir un valor intermedio o comprobar que un import funciona."
    ),
    "run_tests": (
        "Ejecuta los tests de aceptacion de la mision y devuelve si pasaron junto "
        "con la salida de pytest. Sin argumentos ejecuta los tests declarados; "
        "pasa node_ids para ejecutar solo algunos. Leer la salida cuando falla es "
        "como averiguas que arreglar."
    ),
    "finish": (
        "Termina tu trabajo. status='DONE' si crees que esta completo, "
        "status='NO_CHANGE' si has comprobado que no hace falta ningun cambio, "
        "status='BLOCKED' si no puedes continuar (explica en summary que te lo "
        "impide). No llames a finish con DONE sin haber editado nada."
    ),
}

#: Per-parameter prose. Same reasoning as TOOL_DOC: a parameter whose meaning
#: is not stated is a parameter the model has to guess.
PARAM_DOC: dict[str, str] = {
    "path": "Ruta relativa a la raiz del repositorio, con barras normales: 'pkg/mod.py'.",
    "start": "Primera linea a leer, empezando en 1. null para leer desde el principio.",
    "end": "Ultima linea a leer, incluida. null para leer hasta el final.",
    "pattern": "Expresion regular de Python. Se busca linea a linea.",
    "glob": "Que ficheros mirar, p.ej. '**/*.py' (por defecto) o 'tests/**/*.py'.",
    "old": "El texto exacto que hay ahora en el fichero, incluida su indentacion. Debe ser unico.",
    "new": "El texto que lo sustituye. Cadena vacia para borrar el fragmento.",
    "content": "Contenido completo del fichero nuevo.",
    "argv": 'Comando como lista de cadenas: ["python", "-m", "pytest", "-x", "tests/test_a.py"].',
    "timeout": "Segundos maximos de ejecucion (1-600). null usa el limite por defecto.",
    "node_ids": "Lista de tests concretos, p.ej. ['tests/test_a.py::test_b']. null ejecuta los declarados.",
    "summary": "Una o dos frases sobre lo que has hecho, o sobre lo que te impide continuar.",
    "status": "Uno de: 'DONE', 'NO_CHANGE', 'BLOCKED'.",
}

assert set(TOOL_DOC) == set(SPECS), "TOOL_DOC and SPECS must describe the same tools"
assert set(PARAM_DOC) >= {p for req, opt in SPECS.values() for p in req + opt}, (
    "every parameter in SPECS needs an entry in PARAM_DOC"
)


def native_schema() -> list[dict]:
    """The provider-side tool schema, generated from SPECS so it cannot drift.

    Both the shape and the prose come from module-level dicts that asserts tie
    to SPECS, so the schema, the argument validation and the documentation are
    one source of truth (INVARIANTS I3). The previous runner advertised a
    ``list_dir`` it did not implement and a ``run`` when the contract said
    ``run_tests``; neither could happen here without failing at import.
    """
    types = {
        "path": {"type": "string"},
        "start": {"type": ["integer", "null"]},
        "end": {"type": ["integer", "null"]},
        "pattern": {"type": "string"},
        "glob": {"type": "string"},
        "old": {"type": "string"},
        "new": {"type": "string"},
        "content": {"type": "string"},
        "argv": {"type": "array", "items": {"type": "string"}},
        "timeout": {"type": ["number", "null"]},
        "node_ids": {"type": ["array", "null"], "items": {"type": "string"}},
        "summary": {"type": "string"},
        "status": {"type": "string", "enum": list(FINISH_STATUSES)},
    }
    out = []
    for name, (required, optional) in SPECS.items():
        props = {}
        for key in required + optional:
            spec = dict(types[key])
            spec["description"] = PARAM_DOC[key]
            props[key] = spec
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DOC[name],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(required),
                },
            },
        })
    return out
