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
import difflib
import hashlib
import re
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import deps, retrieval
from .scope import WriteScope
from . import sentinel as _sentinel
from .errors import (
    ERROR_BAD_ARGUMENTS,
    ERROR_BAD_PATTERN,
    ERROR_BAD_RANGE,
    ERROR_COMMAND_NOT_ALLOWED,
    ERROR_EMPTY_DIRECTORY,
    ERROR_EMPTY_OLD,
    ERROR_ESCAPED_CONTENT,
    ERROR_FILE_EXISTS,
    ERROR_FILE_NOT_FOUND,
    ERROR_IS_DIRECTORY,
    ERROR_LANGUAGE_UNSUPPORTED,
    ERROR_MISSING_ARGUMENT,
    ERROR_MULTIPLE_MATCHES,
    ERROR_NO_MATCH,
    ERROR_NOT_IN_WRITE_SCOPE,
    ERROR_NOT_TEXT,
    ERROR_NOT_VERIFIED,
    ERROR_NOTHING_CHANGED,
    ERROR_PATH_OUTSIDE_REPO,
    ERROR_RESULT_TRUNCATED,
    ERROR_SEARCH_SCOPE_EMPTY,
    ERROR_SYNTAX,
    ERROR_SYNTAX_AFTER_EDIT,
    ERROR_TOO_MANY_ENTRIES,
    ERROR_UNREACHABLE_CODE,
    ERROR_UNKNOWN_TOOL,
    HarnessInvalid,
    InvalidCall,
    ToolError,
)

MAX_READ_LINES = 2000
#: Spelled out rather than written inline, so a literal backslash in this
#: file's own source can never be mistaken for one in the model's payload.
NEWLINE = chr(10)
BACKSLASH_N = chr(92) + "n"
HEAD_LINES = 200

#: Hard ceiling on what any single tool may return, in characters (F-25).
#: Raised 12000 -> 20000 with the window (F-32). The old value was sized
#: against a 16k context; that is 32k since F-28 and this was never
#: revisited. A truncated read now hands back 10k instead of 6k, nearly
#: halving the calls needed to cross a large file -- which is what an agent
#: spent twenty-four of forty turns doing. A tool result that would blow the
#: window is not a bigger answer, it is a 400 and no answer at all.
MAX_TOOL_PAYLOAD_CHARS = 20000
MAX_GREP_HITS = 50
TEST_TIMEOUT_SECONDS = 120.0
#: How much pytest output reaches the model (F-46).
#:
#: This was 3000, sized against the original 8192-token context and never
#: revisited -- so the single most important feedback channel in the loop was
#: also the most aggressively truncated thing in the system. Every run_tests
#: result in the ga04/ga07/ga08 failures came back at 3131-3171 characters:
#: clipped, every time, on exactly the runs where the agent was one test short
#: and needed to see why.
TEST_OUTPUT_TAIL = 9000
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}
FINISH_STATUSES = ("DONE", "NO_CHANGE", "BLOCKED")
MAX_DIR_ENTRIES = 200

#: How many ranked candidates search_code returns by default (F-59). Fifteen
#: is about 3k characters of path + declaration + preview: small enough to
#: read in one turn, wide enough that offline the needle is inside it for
#: 82 of 100 RepoQA python cases.
SEARCH_DEFAULT_LIMIT = 15
#: How many of each search's candidates are remembered for the give-up question.
TRACKED_CANDIDATES = 3
MAX_SEARCH_LIMIT = 50

#: How many files a failed grep names when it reports where matches DO live.
GREP_DISTRIBUTION_FILES = 12
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
    #: (path, symbol) of the harness's own top-ranked candidate for this
    #: mission, computed deterministically at turn zero from the same index the
    #: objective's shortlist uses. Carried so a refusal can name the exact call
    #: rather than a kind of call -- the difference F-82 measured at 95% against
    #: 7%, and the reason qwen2.5-coder:3b called finish twenty times against a
    #: message that described write_file in prose.
    opening_candidate: tuple[str, str] | None = None
    #: How many times each path has been written, and whether anything was
    #: learned since the last one. Two of the reference engine's first eleven
    #: post-F-75 runs wrote their answer six and nine times, at 17 and 20 turns;
    #: what separates that from ordinary self-correction is not the count but
    #: whether a read, a search or a run happened in between.
    write_counts: dict = field(default_factory=dict)
    learned_since_write: bool = True
    #: Paths THIS RUN brought into existence. Separate from changed_files
    #: because the permission it carries is different: a file the agent created
    #: may be rewritten, and a file that was already in the repository may not
    #: become writable just because it was edited once. Only write_file adds to
    #: it, and only after _check_writable has already allowed the creation.
    created_files: set[str] = field(default_factory=set)
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
    #: The repository this workspace was made from, when there is one. The
    #: factory promises not to modify it, and until F-112 nothing in the
    #: harness checked that promise -- it was verified by hand, externally,
    #: once. The sentinel checks it around every child process.
    source_repo: Path | None = None
    #: What the sentinel saw around each ``run``. Kept even when empty: a
    #: reader needs to see that the check happened and what it did not cover.
    outside_writes: list = field(default_factory=list)
    #: Per-test outcomes of the WHOLE suite before the agent touched
    #: anything. Empty when the ticket disabled the full-suite check.
    #: This is what makes it possible to tell the agent, during the run,
    #: what the conscience will tell it afterwards (F-38).
    baseline_outcomes: dict = field(default_factory=dict)
    #: Callable returning per-test outcomes of the whole suite now.
    #: Injected so tools.py does not import verify.py, which imports work.
    full_suite_runner: Any = None
    #: Tests that passed before and fail now, as of the last check. The
    #: agent is told, and finish(DONE) refuses while this is non-empty.
    known_regressions: list = field(default_factory=list)
    collateral_checked: bool = False
    #: The tail of the last failing run_tests output. Kept so a refusal
    #: can quote what is actually wrong instead of telling the agent to
    #: go and look at something it has already looked at (F-42).
    last_test_failure: str = ''
    #: Node ids reported failing by the last run_tests.
    last_failing_tests: list = field(default_factory=list)
    #: Whether a premature BLOCKED has already been questioned once.
    blocked_once: bool = False
    #: Whether an empty-diff NO_CHANGE has already been questioned once (F-61).
    no_change_once: bool = False
    #: The best-ranked candidates search_code has shown, in the order it showed
    #: them, and every file or symbol the agent has actually opened (F-67). Both
    #: are facts about the agent's own session, and the harness holding them
    #: while the agent gives up on turn six is the same defect as F-45.
    top_candidates: list = field(default_factory=list)
    opened: set = field(default_factory=set)
    #: (path, resolved symbol) of every symbol the agent has itself reached,
    #: oldest first. Distinct from ``opening_candidate``, which is the HARNESS's
    #: turn-zero guess: this is the agent's own decision, already made and
    #: already validated against the file. F-94 -- a refusal that names the
    #: harness's guess while the agent is holding a resolved symbol of its own
    #: is arguing with it, and two runs spent twenty turns on that argument.
    reached_symbols: list = field(default_factory=list)
    #: Budget state, written by the loop each turn so finish can weigh
    #: 'I am stuck' against 'I have barely started'.
    turns_left: int | None = None
    max_turns_hint: int = 0
    _scope: WriteScope | None = field(default=None, repr=False)
    #: Lazily built retrieval index, and the state of the tree it was built
    #: from. Rebuilding on every call would cost a second per search on a
    #: large repository; never rebuilding would serve results from a tree the
    #: agent has since edited. Keyed on what the agent changed, which is the
    #: only thing that can invalidate it inside one run.
    _index: Any = field(default=None, repr=False)
    _index_key: Any = field(default=None, repr=False)

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

    def _tree_fingerprint(self) -> tuple:
        """A cheap stamp of what the tree looks like right now.

        Keying the cache on ``changed_files`` alone was wrong, and wrong in the
        direction that matters: ``run(["python", "-c", ...])`` can write
        anything, and nothing outside the edit tools updates that set. A search
        served out of an index built before those writes describes a repository
        that no longer exists, and says so with the same confidence as a correct
        one.

        Counted, not hashed: file count, newest mtime and total size. One stat
        per file, no reads, and it moves whenever a write does.
        """
        count = 0
        newest = 0.0
        total = 0
        for path in retrieval.walk_files(
            self.root, frozenset(retrieval.DEFAULT_SKIP_DIRS | SKIP_DIRS)
        ):
            try:
                info = path.stat()
            except OSError:
                continue
            count += 1
            total += info.st_size
            newest = max(newest, info.st_mtime)
        return (count, total, round(newest, 3), frozenset(self.changed_files))

    def index(self) -> "retrieval.Index":
        """The retrieval index for this repository, rebuilt when the tree moves."""
        key = self._tree_fingerprint()
        if self._index is None or self._index_key != key:
            self._index = retrieval.Index(self.root, skip_dirs=frozenset(
                retrieval.DEFAULT_SKIP_DIRS | SKIP_DIRS
            ))
            self._index_key = key
        return self._index

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


#: How many literal backslash-n sequences make a one-line payload conclusive.
#: One could be a string constant in a genuine one-liner. Two, with no real
#: newline anywhere, is a multi-line body that lost its newlines in transit.
ESCAPED_NEWLINE_THRESHOLD = 2

#: And how long the payload has to be before that is conclusive. A one-line
#: assignment holding two newline escapes is real code; a function body that
#: lost its newlines in transit is never this short. The six observed cases ran
#: from 240 to 6600 characters.
ESCAPED_MIN_CHARS = 120


def _unescape_if_flattened(argument: str, text: Any) -> tuple[Any, str]:
    """Repair content whose newlines arrived escaped, and say so (F-63).

    A tool call carries its arguments as JSON, and a small model writing a
    multi-line body into a JSON string sometimes escapes it twice. What arrives
    is one physical line of literal backslash-n -- and the harness wrote it out
    exactly like that, without a word.

    Measured on the matched evaluation: 6 of 50 answers came out as a single
    line of backslashes, against 1 of 50 when the same model on the same task
    emitted the same code as plain text instead of through a tool argument.
    That difference is the channel, which makes it ours.

    This first refused, on the principle that a write tool which edits what it
    is handed is worse than one that says no. Measured, that principle cost
    more than it saved: of 50 matched cases, refusing removed five of the six
    corrupted writes and turned one of them into a run that spent five turns
    being told no and finished BLOCKED holding the correct answer. The model
    could not produce the unescaped form, so "try again" was not a route.

    So it is repaired -- and announced in the tool's own result, which is the
    part that matters. The harness saying what it changed is not the same thing
    as the harness changing it quietly, and the agent can read the file back.
    The signal stays deliberately narrow: no real newline anywhere, at least two
    escapes, and long enough that it can only be a flattened body.
    """
    if not isinstance(text, str) or NEWLINE in text:
        return text, ""
    count = text.count(BACKSLASH_N)
    # Two escapes in a SHORT one-liner is ordinary code: ``sep = "\n\n"`` is a
    # legitimate single-line replacement and must go through untouched. What is
    # never ordinary is a whole body flattened onto one line, and that is always
    # long. The bar keeps the guard on the case it was built for.
    if count < ESCAPED_NEWLINE_THRESHOLD or len(text) < ESCAPED_MIN_CHARS:
        return text, ""
    repaired = (text.replace(BACKSLASH_N, NEWLINE)
                    .replace(chr(92) + chr(92) + '"', '"')
                    .replace(chr(92) + '"', '"')
                    .replace(chr(92) + "t", chr(9)))
    return repaired, (
        f"{NEWLINE}[{ERROR_ESCAPED_CONTENT}: {argument} llego en UNA SOLA LINEA "
        f"con {count} secuencias literales barra-n y ningun salto real, asi que "
        f"lo he DESESCAPADO antes de escribirlo: ahora tiene "
        f"{repaired.count(NEWLINE) + 1} lineas. En los argumentos de una "
        f"herramienta el texto va TAL CUAL, sin escapar. Si querias barras "
        f"invertidas de verdad, leelo y corrigelo.]"
    )


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
    if rel in ctx.created_files:
        # This run created it, so this run may rewrite it. Without this the
        # mission's own output file is unreachable after the first draft:
        # write_file says "ya existe, usa edit" and edit says "fuera de
        # write_scope", and there is no third door. Deliberately keyed on what
        # was actually created rather than on allowed_new_files, so a mission
        # that names an existing source file by mistake grants nothing.
        return
    if not ctx.scope.allows(rel, creating=creating):
        verb = "crear" if creating else "modificar"
        # A3. Scope matching is case-SENSITIVE on purpose: Windows is not, and a
        # containment rule that behaves differently on two machines is not a
        # containment rule. scope.py argues that at length and it stays.
        #
        # What was missing is that the refusal was undiagnosable. On Windows
        # 'PKG/a.py' and 'pkg/a.py' are the same file, so an agent that varied
        # the case got "outside write_scope" for a path that is plainly inside
        # it, with no way to see why. The rule does not move; it now says which
        # of the two things went wrong.
        hint = ""
        if ctx.scope.allows(rel.lower(), creating=creating) or \
                ctx.scope.allows(rel.upper(), creating=creating):
            hint = (f"\n  OJO: {rel.lower()!r} SI estaria dentro del ambito. La "
                    f"diferencia es solo de mayusculas/minusculas, y el ambito "
                    f"las distingue a proposito para que se comporte igual en "
                    f"Windows y en Linux. Escribe la ruta como esta en el disco.")
        # The work first, the boundary second -- the F-71 correction applied to
        # the other refusal that only ever said no. A mission waiting for a file
        # that does not exist knows exactly what it wants; describing the scope
        # and stopping there left 18 of 23 granite runs to quit at turn three.
        pending = [name for name in ctx.allowed_new_files
                   if not (ctx.root / name).exists()]
        wanted = ""
        if pending:
            wanted = (f"\n  ESTA MISION ESPERA QUE CREES: {', '.join(pending[:4])}. "
                      f"Todavia no existe, asi que no hay nada que editar ahi.\n"
                      f"  Para crearlo: write_file(path='{pending[0]}', "
                      f"content=<el texto entero>).")
        raise ToolError(
            ERROR_NOT_IN_WRITE_SCOPE,
            f"no puedes {verb} {rel!r}: esta fuera de write_scope."
            + wanted
            + f"\n  Ambito de escritura: {ctx.scope.describe(creating=creating)}"
            + "\n  (un ambito que acaba en '/' incluye todo lo que hay debajo)" + hint,
        )


# ---------------------------------------------------------------- the 7 tools


#: Tools that bring something back the agent did not already have. Writing is
#: not here, and neither is finish: repeating a write after one of THOSE is the
#: pattern this watches for.
INFORMATION_TOOLS = frozenset({
    "read_file", "read_symbol", "list_symbols", "list_dir",
    "grep", "search_code", "run", "run_tests",
})


def _rewrite_note(ctx: ToolContext, rel: str) -> str:
    """Note a rewrite that had nothing new behind it.

    A note and never a refusal. F-63 is the standing lesson: a refusal that
    blocks a correct answer costs more than the mistake it prevents, and the
    second write is very often the right one.
    """
    count = ctx.write_counts.get(rel, 0) + 1
    ctx.write_counts[rel] = count
    learned = ctx.learned_since_write
    ctx.learned_since_write = False
    if count < 3 or learned:
        return ""
    return (f"{NEWLINE}[Has escrito {rel} {count} veces, y desde la anterior no "
            f"has leido, buscado ni ejecutado nada: es la misma pregunta con "
            f"otra respuesta, no informacion nueva. Si no estas seguro de cual "
            f"es el codigo correcto, MIRALO (read_symbol, search_code) en vez de "
            f"volver a escribirlo; si ya lo tienes en el repositorio, copy_code "
            f"lo pone tal cual.]")


def _range_note(rel: str, text: str, lo: int, hi: int) -> str:
    """Say what a line range cuts through (F-65).

    An agent that asks for lines 348-370 is guessing at where a definition
    starts and stops, and when the guess is wrong the harness can see exactly
    how: one run copied its target function plus the opening lines of the next
    one and scored 0.38 on an answer it believed was exact. The boundaries were
    one parse away and nobody mentioned them.

    Language-agnostic, because it reuses the same declaration splitter the
    retrieval index is built on -- a file we cannot carve simply gets no note
    rather than a wrong one.
    """
    try:
        family = retrieval._FAMILY.get("." + rel.rsplit(".", 1)[-1].lower())
        regions = retrieval._split_regions(text, family)
    except Exception:      # a note is a courtesy; it never breaks the read
        return ""
    if not regions or family is None:
        return ""
    lines = text.splitlines()

    def declared(at: int) -> str:
        return lines[at - 1].strip()[:70] if 1 <= at <= len(lines) else ""

    covered = [(a, b) for a, b in regions if a <= hi and b >= lo]
    if not covered:
        return ""
    # Speak only when the aim is unambiguously wrong: the range takes in more
    # than one declaration (so copying it drags a neighbour along, which is the
    # failure this exists for), or it begins inside one (so what comes back has
    # no head). A range that merely stops short of the end is what was asked
    # for, and saying so on every ordinary read is noise the agent learns to
    # skip -- which is how a useful note stops being read at all.
    starts_inside = covered[0][0] < lo
    if len(covered) <= 1 and not starts_inside:
        return ""
    pieces = []
    for a, b in covered[:6]:
        mark = "" if a >= lo and b <= hi else "  (INCOMPLETA en este rango)"
        pieces.append(f"    lineas {a}-{b}: {declared(a)}{mark}")
    tail = f"{NEWLINE}    ... y {len(covered) - 6} mas" if len(covered) > 6 else ""
    return (
        f"{NEWLINE}[el rango {lo}-{hi} toca {len(covered)} definicion(es):{NEWLINE}"
        + NEWLINE.join(pieces) + tail
        + f"{NEWLINE}  Si querias UNA entera, read_symbol({rel!r}, '<nombre>') te da "
          f"sus limites exactos.]"
    )


def read_file(ctx: ToolContext, path: Any, start: Any = None, end: Any = None) -> str:
    rel, target = _resolve(ctx, path)
    text = _read_text(rel, target)
    ctx.opened.add(rel)
    lines = text.splitlines()
    total = len(lines)

    # F-25: too many lines OR too many characters. The second condition is the
    # one that matters and the one that was missing: localprog/tools.py is about
    # a thousand lines and 48k characters, which is more than a 16k-token window
    # can hold. Reading it returned a 400 and ended the run on turn 2.
    if start is None and end is None and (total > MAX_READ_LINES or len(text) > MAX_TOOL_PAYLOAD_CHARS):
        shown = []
        size = 0
        for i, line in enumerate(lines[:MAX_READ_LINES], 1):
            entry = f"{i:4}\t{line}"
            if size + len(entry) > MAX_TOOL_PAYLOAD_CHARS // 2:
                break
            shown.append(entry)
            size += len(entry) + 1
        return (
            "\n".join(shown)
            + f"\n[fichero de {total} lineas / {len(text)} caracteres, truncado en la "
            f"linea {len(shown)}. Para mirar algo concreto NO hace falta leerlo "
            f"entero: usa read_symbol('{rel}', '<nombre>') si sabes que buscas, "
            f"list_symbols('{rel}') para ver que hay, o read_file('{rel}', "
            f"start=, end=) para otro tramo]"
        )

    lo = 1 if start is None else start
    hi = total if end is None else end
    if not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool) or isinstance(hi, bool):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "start y end deben ser enteros o null")
    lo = max(1, lo)
    hi = min(total, hi)
    if lo > hi or total == 0:
        return f"[rango vacío: el fichero tiene {total} líneas]"
    body = "\n".join(f"{i:4}\t{lines[i - 1]}" for i in range(lo, hi + 1))
    if len(body) > MAX_TOOL_PAYLOAD_CHARS:
        body = (body[:MAX_TOOL_PAYLOAD_CHARS]
                + f"\n[truncado a {MAX_TOOL_PAYLOAD_CHARS} caracteres; pide un rango mas corto]")
    if start is not None or end is not None:
        body += _range_note(rel, text, lo, hi)
    return body


def grep(ctx: ToolContext, pattern: Any, glob: Any = "**/*.py", context: Any = 0, ignore_case: Any = False) -> Any:
    """Search the repository, optionally returning lines around each hit.

    ``context`` (F-33) is the difference between one call and two. Without it,
    finding a symbol means grep to learn the line number and then read_file to
    see the code, and the read returns a whole window of mostly irrelevant
    lines. The measured cost was real: an agent looking for one function in a
    48k-character file spent twenty-four of its forty turns reading.
    """
    if not isinstance(pattern, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "pattern debe ser una cadena")
    glob = glob if isinstance(glob, str) and glob else "**/*.py"
    if context is None:
        context = 0
    if not isinstance(context, int) or isinstance(context, bool) or not 0 <= context <= 20:
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "context debe ser un entero entre 0 y 20")
    if not isinstance(ignore_case, bool):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "ignore_case debe ser booleano")
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ToolError(ERROR_BAD_PATTERN, str(exc)) from None

    hits: list[dict] = []
    truncated = False
    budget = MAX_TOOL_PAYLOAD_CHARS
    spent = 0
    searched = 0
    try:
        candidates = sorted(ctx.root.glob(glob))
    except (OSError, ValueError, IndexError) as exc:
        # An invalid glob (e.g. "**") is the model's mistake, not a crash.
        raise ToolError(ERROR_BAD_PATTERN, f"glob {glob!r} invalido: {exc}") from None

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
        lines = text.splitlines()
        searched += 1
        for i, line in enumerate(lines, 1):
            if not rx.search(line):
                continue
            if len(hits) >= MAX_GREP_HITS or spent >= budget:
                truncated = True
                break
            hit: dict = {"file": rel, "line": i, "text": line[:400]}
            if context:
                lo = max(1, i - context)
                hi = min(len(lines), i + context)
                block = "\n".join(f"{n:4}\t{lines[n - 1]}" for n in range(lo, hi + 1))
                hit["context"] = block[: budget - spent]
                spent += len(hit["context"])
            spent += len(hit["text"])
            hits.append(hit)
        if truncated:
            break

    if truncated:
        # Where the REST of the matches are (F-59). Cutting at 50 and breaking
        # out of the file loop means the agent sees only whatever sorted first
        # and cannot tell whether that is the whole story or a tenth of it. The
        # remaining count is one more cheap pass and turns a blind truncation
        # into a distribution it can act on.
        rest = _match_distribution(ctx, rx, candidates, seen=len(hits))
        return {"hits": hits, "searched": searched, "note": (
            f"[{ERROR_RESULT_TRUNCATED}: te muestro {len(hits)} de "
            f"{rest['total']} coincidencias en {rest['files']} ficheros. "
            f"Mas coincidencias en: {rest['summary']}. "
            f"Afina el patron, restringe el glob, o baja context]"
        )}
    if not hits:
        return {"hits": [], "searched": searched, "matched_files": 0,
                "note": _empty_grep_note(ctx, pattern, glob, searched, ignore_case)}
    matched_files = len(set(hit['file'] for hit in hits))
    return {"hits": hits, "searched": searched, "matched_files": matched_files}


def _match_distribution(ctx: ToolContext, rx, candidates, *, seen: int) -> dict:
    """How many matches there are in total, and which files hold them."""
    per_file: list[tuple[str, int]] = []
    total = 0
    for p in candidates:
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        count = sum(1 for line in text.splitlines() if rx.search(line))
        if count:
            per_file.append((p.relative_to(ctx.root).as_posix(), count))
            total += count
    per_file.sort(key=lambda pair: (-pair[1], pair[0]))
    summary = ", ".join(f"{name} ({count})"
                        for name, count in per_file[:GREP_DISTRIBUTION_FILES])
    if len(per_file) > GREP_DISTRIBUTION_FILES:
        summary += f", y {len(per_file) - GREP_DISTRIBUTION_FILES} ficheros mas"
    return {"total": max(total, seen), "files": len(per_file), "summary": summary or "-"}


def _empty_grep_note(ctx: ToolContext, pattern: str, glob: str,
                     searched: int, ignore_case: bool) -> str:
    """Say what the system already knows about why a search found nothing.

    Not hypotheses: facts it can check for free. Whether the glob matched any
    file at all, which extensions the repository actually contains, whether the
    same pattern matches when case is ignored, and which words of a multi-word
    pattern appear anywhere. Answering only 'no matches' when all of that is
    one pass away is the harness withholding what it knows -- and it is what 46
    of 100 agents were told, over and over, before giving up.
    """
    if searched == 0:
        present = Counter(
            p.suffix.lower() for p in ctx.root.rglob("*")
            if p.is_file() and p.suffix and not any(part in SKIP_DIRS for part in p.parts)
        )
        top = ", ".join(f"{ext} ({count})" for ext, count in present.most_common(8))
        return (f"[{ERROR_SEARCH_SCOPE_EMPTY}: el glob {glob!r} no encaja con NINGUN "
                f"fichero, asi que no se ha buscado en ninguna parte. El repositorio "
                f"tiene: {top or '(ningun fichero con extension)'}. Cambia el glob.]")

    extra: list[str] = []
    if not ignore_case:
        try:
            insensitive = re.compile(pattern, re.IGNORECASE)
        except re.error:
            insensitive = None
        if insensitive is not None and _any_match(ctx, insensitive, glob):
            extra.append("SI hay coincidencias si ignoras mayusculas: "
                         "repite con ignore_case=True")

    words = [w for w in re.split(r"[^A-Za-z0-9_]+", pattern) if len(w) > 2]
    if len(words) > 1:
        found = [w for w in dict.fromkeys(words)
                 if _any_match(ctx, re.compile(re.escape(w), re.IGNORECASE), glob)]
        missing = [w for w in dict.fromkeys(words) if w not in found]
        if found:
            extra.append("de tu patron SI aparecen por separado: " + ", ".join(found[:6]))
        if missing:
            extra.append("no aparecen en ninguna parte: " + ", ".join(missing[:6]))

    tail = (" " + ". ".join(extra) + ".") if extra else (
        " Prueba search_code(query=...) con una descripcion en palabras: "
        "busca por significado y no exige el literal exacto."
    )
    return (f"[{ERROR_NO_MATCH}: 0 coincidencias para {pattern!r} en {glob!r} "
            f"(buscado en {searched} ficheros).{tail}]")


def _any_match(ctx: ToolContext, rx, glob: str) -> bool:
    try:
        candidates = sorted(ctx.root.glob(glob))
    except (OSError, ValueError, IndexError):
        return False
    for p in candidates:
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if rx.search(text):
            return True
    return False


def search_code(ctx: ToolContext, query: Any, limit: Any = None, path: Any = None) -> dict:
    """Find the code a description is talking about, by meaning rather than literal.

    The organ that was missing (F-59). ``grep`` answers "where does this exact
    string appear"; every other tool needs you to already know the file. Nothing
    answered "which of these four thousand functions is the one being
    described", so an agent given a paraphrase had to guess literal search terms
    out of it -- and when the guess missed, guess again.

    Measured on the RepoQA python split: of 100 failures, 46 ended with the
    agent reporting it had searched and found nothing, for a function that was
    on disk every time, and 33 more answered with a different function. Ranking
    the repository's declarations against the description puts the right one in
    the top 15 for 82 of those 100 cases.
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidCall(ERROR_BAD_ARGUMENTS,
                          "query debe ser texto: describe lo que buscas, con palabras")
    if limit is None:
        limit = SEARCH_DEFAULT_LIMIT
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise InvalidCall(ERROR_BAD_ARGUMENTS,
                          f"limit debe ser un entero entre 1 y {MAX_SEARCH_LIMIT}")
    prefix = ""
    only_file = ""
    if path is not None and str(path).strip() not in ("", ".", "/"):
        prefix, target = _resolve_dir(ctx, path)
        if target.is_file():
            # A file is the narrowest scope there is, and it is exactly what an
            # agent asks for when the mission has already named the file.
            # Treating it as a directory prefix turned it into
            # "src/black/nodes.py/", which nothing is indexed under, so F-69
            # widened the search to the whole repository and answered from
            # another file entirely. Every step was right and the sum ignored
            # the only constraint the agent had expressed.
            only_file, prefix = prefix, ""
        else:
            prefix = "" if prefix == "." else prefix.rstrip("/") + "/"

    index = ctx.index()
    if only_file and not any(r.path == only_file for r in index.regions):
        # The file exists but carved no regions -- not a source file, or empty.
        # Saying so beats silently searching somewhere else.
        raise ToolError(
            ERROR_SEARCH_SCOPE_EMPTY,
            f"{only_file!r} existe pero no tiene nada indexable (no es un fichero "
            f"de codigo, o esta vacio). Quita path= para buscar en todo el "
            f"repositorio, o usa grep si buscas un literal.")
    scope_note = ""
    if prefix and not any(r.path.startswith(prefix) for r in index.regions):
        # F-69: the prefix names nothing indexed. Answering "no match" here is
        # false -- regions matched, none of them are under a directory that does
        # not exist -- and it reads as "the function is not in this repository".
        real = sorted({r.path.split("/")[0] + "/" for r in index.regions if "/" in r.path})
        scope_note = (
            f"{NEWLINE}[{ERROR_SEARCH_SCOPE_EMPTY}: no hay ningun fichero indexado "
            f"bajo {prefix!r}, asi que he buscado en TODO el repositorio en vez de "
            f"no devolverte nada. Los directorios de primer nivel que si existen "
            f"son: {', '.join(real[:10]) or '(ninguno, todo esta en la raiz)'}.]"
        )
        prefix = ""
    if not len(index):
        raise ToolError(
            ERROR_SEARCH_SCOPE_EMPTY,
            "no hay ningun fichero de codigo indexable en este repositorio.",
        )

    # Over-fetch when filtering by path, so a subtree still yields `limit` rows.
    narrowed = prefix or only_file
    raw = index.search(query, limit=limit if not narrowed else min(limit * 20, 1000))
    if only_file:
        raw = [pair for pair in raw if pair[0].path == only_file][:limit]
    elif prefix:
        raw = [pair for pair in raw if pair[0].path.startswith(prefix)][:limit]

    if not raw:
        known, unknown = index.matched_terms(query)
        detail = f"ninguna region coincide con {query!r}"
        if only_file:
            detail += (f" dentro de {only_file!r} (ese fichero SI existe y esta "
                       f"indexado). Quita path= para buscar en todo el repositorio")
        elif prefix:
            detail += f" bajo {prefix!r} (ese directorio SI existe)"
        if unknown:
            detail += (". Estas palabras no aparecen en NINGUN sitio del repositorio: "
                       + ", ".join(unknown[:8]))
        if known:
            detail += ". Si aparecen: " + ", ".join(known[:8])
        raise ToolError(ERROR_NO_MATCH, detail + ".")

    candidates = [{
        "rank": n,
        "path": region.path,
        "lines": f"{region.start}-{region.end}",
        # The identifier, spelled out. Without it the agent has to parse it back
        # out of the declaration line before it can name it in a call, and
        # naming it in a call is the whole difference between 95% and 7%.
        "symbol": region.name,
        "declares": region.header,
        "preview": region.preview,
        "score": round(score, 2),
    } for n, (region, score) in enumerate(raw, 1)]

    for row in candidates[:TRACKED_CANDIDATES]:
        entry = (row["path"], row["lines"], row["declares"])
        if entry not in ctx.top_candidates:
            ctx.top_candidates.append(entry)

    # The next call, filled in, rather than the name of a kind of call.
    # Measured: an agent that NAMES the symbol answers correctly 95% of the time
    # (reference engine, 20 of 50 runs); one that only opens the file, 64%, and
    # for granite4.1:3b 7%. The gap is not comprehension, it is which call gets
    # made next, so the result says which call to make next.
    top = candidates[0]
    pending = [n for n in ctx.allowed_new_files if not (ctx.root / n).exists()]
    next_call = (f"{NEWLINE}  Para ver uno entero, tal cual esta: "
                 f"read_symbol(path={top['path']!r}, name={top['symbol']!r}).")
    if pending:
        next_call += (f"{NEWLINE}  Para ponerlo en {pending[0]} sin reescribirlo: "
                      f"copy_code(src={top['path']!r}, into={pending[0]!r}, "
                      f"name={top['symbol']!r}).")
    return {
        "query": query,
        "candidates": candidates,
        "indexed": {"regions": len(index), "files": index.files_indexed},
        "note": ("[candidatos ordenados por parecido con tu descripcion, no por "
                 "certeza: el primero no tiene por que ser el bueno. Mira las "
                 "declaraciones y quedate con el que encaje."
                 + next_call + "]") + scope_note,
    }


def _require_python(rel: str) -> None:
    """Refuse a file this tool cannot parse, and say what to use instead.

    ``list_symbols`` and ``read_symbol`` are built on ``ast``, so they only ever
    worked for Python. Handed a TypeScript, Java or Rust file they reported
    ERROR_SYNTAX -- "linea 1: invalid syntax" -- about a perfectly valid file.
    That is the harness asserting something false about the repository, and an
    agent that believes it goes looking for a bug that does not exist.

    The limitation stays; the lie does not. There is a language-agnostic path
    for exactly this (search_code returns a line range, read_file reads it), so
    the error names it.
    """
    if rel.rsplit(".", 1)[-1].lower() in ("py", "pyi"):
        return
    raise ToolError(
        ERROR_LANGUAGE_UNSUPPORTED,
        f"{rel!r} no es Python, y esta herramienta solo entiende Python. "
        f"El fichero NO esta mal: es esta herramienta la que no lo lee.\n"
        f"  Para cualquier lenguaje: search_code(query=...) te da fichero y "
        f"rango de lineas, y read_file(path, start=, end=) te da el codigo.",
    )


def list_symbols(ctx: ToolContext, path: Any) -> list[str]:
    rel, target = _resolve(ctx, path)
    source = _read_text(rel, target)
    _require_python(rel)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolError(ERROR_SYNTAX, f"{rel!r} línea {exc.lineno}: {exc.msg}") from None
    except (ValueError, RecursionError) as exc:  # null bytes, pathological nesting
        raise ToolError(ERROR_SYNTAX, f"{rel!r}: {exc}") from None

    collected = _collect_symbols(tree)
    lineno = {name: node.lineno for name, node in collected.items()}

    # edit_channel decides what a CALLABLE is, so this harness and Gafotas' edit
    # path can never disagree about those names. Module-level data is ours to
    # add (F-54): edit_channel has no opinion about it, and an agent that cannot
    # see SPECS or PARAM_DOC cannot change them.
    names = list(deps.edit_channel.list_symbols(source))
    for name in collected:
        if name not in names:
            names.append(name)
    names.sort(key=lambda n: (lineno.get(n, 0), n))
    return [f"{n} (linea {lineno[n]})" if n in lineno else n for n in names]


def _assignment_targets(node) -> list[str]:
    """Names bound by an assignment statement, if it binds simple names.

    Tuple unpacking and subscript targets are skipped rather than guessed at:
    "A, B = f()" does not have a source range that means "A", and offering one
    would be worse than saying nothing.
    """
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    return []


def _collect_symbols(tree) -> dict:
    """Every named definition in a module: callables, classes AND data (F-54).

    The last of those is the one that was missing, and it cost whole runs.
    PARAM_DOC, SPECS, TOOL_DOC, EXIT_CODES, EXCLUDED_DIR_NAMES -- module-level
    tables are what a ticket most often asks to be changed, and a navigation
    tool that only walks def and class cannot see any of them.
    """
    found: dict = {}

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                dotted = child.name if not prefix else f"{prefix}.{child.name}"
                found.setdefault(dotted, child)
                walk(child, dotted)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                for name in _assignment_targets(child):
                    dotted = name if not prefix else f"{prefix}.{name}"
                    found.setdefault(dotted, child)

    walk(tree, "")
    return found


def _syntax_report(source: str, exc: SyntaxError, rel: str) -> str:
    """Show the model the line it got wrong, in the text it just sent (F-39).

    ga06 failed with twenty-five consecutive ERROR_SYNTAX_AFTER_EDIT. Each
    rejection said "line 23: unexpected indent" and nothing else -- and line 23
    of WHAT? The content is the model's own past output, already elided from the
    transcript, so it has no way to look. It rewrote the whole module from
    scratch each time and made a different mistake.

    Every byte needed to answer that was in this process. Quoting it back turns
    an unactionable refusal into a one-line fix.
    """
    lines = source.splitlines()
    lineno = exc.lineno or 0
    if not (1 <= lineno <= len(lines)):
        return f"  linea {lineno}: {exc.msg}"
    lo = max(1, lineno - 3)
    hi = min(len(lines), lineno + 2)
    out = [f"  linea {lineno}: {exc.msg}", "  Esto es lo que enviaste:"]
    for n in range(lo, hi + 1):
        marker = ">>" if n == lineno else "  "
        out.append(f"  {marker}{n:4}| {lines[n - 1]}")
        if n == lineno and exc.offset and 0 < exc.offset <= len(lines[n - 1]) + 1:
            out.append("       | " + " " * (exc.offset - 1) + "^")
    return "\n".join(out)


def _nearest_region(text: str, old: str, width: int = 12) -> str:
    """The part of *text* that most resembles *old*, for a failed edit (F-27).

    "Not found, go and read the file" was the old advice, and it was the very
    loop the agent was already trapped in: read, guess, fail, read again. What
    it needed was to SEE the difference -- almost always an indent that is four
    spaces instead of eight, or a line that has moved on since the read.
    """
    old_lines = [l for l in old.splitlines() if l.strip()]
    if not old_lines:
        return ""
    lines = text.splitlines()
    anchor = max(old_lines, key=len).strip()
    best_index, best_score = None, 0.0
    for i, line in enumerate(lines):
        score = difflib.SequenceMatcher(None, anchor, line.strip()).ratio()
        if score > best_score:
            best_index, best_score = i, score
    if best_index is None or best_score < 0.5:
        return ""
    lo = max(0, best_index - width // 2)
    hi = min(len(lines), best_index + width // 2 + 1)
    body = "\n".join(f"{n:4}\t{lines[n - 1]}" for n in range(lo + 1, hi + 1))
    return (f"\n  Lo mas parecido esta por la linea {best_index + 1} "
            f"(parecido {best_score:.0%}). Asi esta AHORA en el fichero:\n{body}\n"
            f"  Copia el texto exacto de ahi (con su indentacion), o usa "
            f"replace_lines('{{rel}}', start, end, content) con esos numeros de linea.")


#: How close a guess has to be before it is worth offering as a correction.
#: 0.35 is loose enough for '_validate_percentile' -> 'is_valid_percentile',
#: which is the real case this exists for, and tight enough that an unrelated
#: name produces nothing rather than a confident wrong suggestion.
_NAME_SIMILARITY = 0.35


def _did_you_mean(guess: str, known) -> list[str]:
    """The symbols closest to a name the caller invented.

    Ranked against the CALLER'S OWN GUESS, never against the objective. That is
    what keeps this a spelling correction rather than retrieval hiding in an
    error message -- the distinction F-85 was about. When nothing is close
    enough it returns nothing, and the caller gets the plain listing.
    """
    return difflib.get_close_matches(guess, list(known), n=4,
                                     cutoff=_NAME_SIMILARITY)


def _symbol_span(rel: str, source: str, name: str
                 ) -> tuple[str, Any, dict, int, int]:
    """Which lines a named symbol occupies, decorators included.

    Shared by read_symbol and copy_code so the two cannot come to different
    conclusions about what a symbol is. Returns the RESOLVED name -- a bare
    method name that matched exactly one qualified symbol comes back qualified,
    which is what the caller should report.
    """
    _require_python(rel)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolError(ERROR_SYNTAX, f"{rel!r} linea {exc.lineno}: {exc.msg}") from None
    except (ValueError, RecursionError) as exc:
        raise ToolError(ERROR_SYNTAX, f"{rel!r}: {exc}") from None

    found = _collect_symbols(tree)
    node = found.get(name)
    if node is None:
        # A bare method name is what a model usually types, and refusing it
        # while knowing exactly which symbol was meant would be pedantry.
        matches = [k for k in found if k.rsplit(".", 1)[-1] == name]
        if len(matches) == 1:
            node = found[matches[0]]
            name = matches[0]
        elif matches:
            raise ToolError(
                ERROR_NO_MATCH,
                f"{name!r} es ambiguo en {rel!r}: {', '.join(sorted(matches))}. "
                f"Usa el nombre completo.",
            )
        else:
            # A guess built out of the objective's words is the commonest way
            # this fails -- granite4.1:3b asked for '_validate_percentile' 21
            # times in 50 runs where the symbol is 'is_valid_percentile'. An
            # alphabetical list of twenty is not an answer to that; the closest
            # names are.
            close = _did_you_mean(name, found)
            near = ", ".join(sorted(found)[:20]) or "(ninguno)"
            detail = f"no hay ningun simbolo {name!r} en {rel!r}."
            if close:
                detail += (f"\n  Lo mas parecido que SI existe: "
                           f"{', '.join(repr(c) for c in close)}.")
            detail += (f"\n  Todos los simbolos del fichero: {near}"
                       + (" ..." if len(found) > 20 else ""))
            raise ToolError(ERROR_NO_MATCH, detail)

    # Decorators sit above the def and are part of what the symbol IS.
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = getattr(node, "end_lineno", None) or node.lineno
    return name, node, found, start, min(end, len(source.splitlines()))


def read_symbol(ctx: ToolContext, path: Any, name: Any) -> str:
    """The source of one function or class, by name (F-50).

    Reading a named definition out of a large file is the commonest navigation
    a programmer does, and it was three calls: list_symbols for the line
    number, read_file with a guessed range, and usually another read because
    the guess was short. Measured, both local models failed at it -- one spent
    thirty of forty turns searching a file it had already identified and never
    reached the tests.

    ``name`` accepts "funcion" or "Clase.metodo". Line numbers are the file's
    own, so the result feeds straight into replace_lines.
    """
    rel, target = _resolve(ctx, path)
    if not isinstance(name, str) or not name.strip():
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "name debe ser el nombre de una funcion o clase")
    name = name.strip()

    source = _read_text(rel, target)
    ctx.opened.add(rel)
    name, node, found, start, end = _symbol_span(rel, source, name)
    # The agent reached this by itself and the file agreed. Kept so a later
    # refusal can complete the call rather than propose a different one (F-94).
    if (rel, name) in ctx.reached_symbols:
        ctx.reached_symbols.remove((rel, name))
    ctx.reached_symbols.append((rel, name))
    lines = source.splitlines()
    # VERBATIM, with no per-line numbering (F-64). read_symbol exists to hand
    # back a symbol's source, and the commonest thing done with that source is
    # to reproduce it exactly -- into edit's `old`, or into a new file. A
    # right-aligned number and a tab on every line move the code's own
    # indentation away from the left margin and turn 'copy this' into
    # 'transcribe this, stripping a prefix' -- a job the harness invented and
    # then made the model do.
    #
    # Measured on the matched evaluation, where the same model met the same
    # file both ways: asked for the function as plain text it reproduced it
    # byte for byte 72% of the time; reading it through the tools and writing
    # it back, 30% -- and six of the differences were nothing but indentation.
    #
    # The line numbers are not lost. They are stated once in the header, which
    # is the shape replace_lines actually consumes: a start and an end, not a
    # number per line.
    body = NEWLINE.join(lines[n - 1] for n in range(start, end + 1))
    if len(body) > MAX_TOOL_PAYLOAD_CHARS:
        kept = body[:MAX_TOOL_PAYLOAD_CHARS]
        shown = kept.count(chr(10)) + 1
        body = kept + (
            f"\n[{name} ocupa las lineas {start}-{end}; te muestro hasta la "
            f"{start + shown - 1}. Usa read_file('{rel}', start=, end=) para el resto]"
        )
    # F-70: say when what came back is a container rather than a behaviour.
    # Ten dev failures were the same shape -- the right place, read_symbol, and
    # then an answer three to forty times longer than the function wanted,
    # because read_symbol on a class returns the whole class and never said so.
    inside = ""
    if isinstance(node, ast.ClassDef):
        methods = [child.name for child in node.body
                   if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if methods:
            shown = ", ".join(methods[:12]) + (" ..." if len(methods) > 12 else "")
            inside = (
                f"{NEWLINE}[OJO: {name!r} es una CLASE entera de "
                f"{end - start + 1} lineas, no una funcion. Contiene "
                f"{len(methods)} metodos: {shown}."
                f"{NEWLINE}  Si lo que buscabas es UNO de ellos, pidelo por su "
                f"nombre: read_symbol({rel!r}, '{name}.{methods[0]}'). "
                f"Copiar la clase entera cuando querias un metodo es el error "
                f"mas caro que se puede cometer aqui.]"
            )
    return (
        f"{rel}::{name}  (lineas {start}-{end}; la primera de abajo es la "
        f"{start}). El codigo va TAL CUAL esta en el fichero, sin numerar: "
        f"puedes copiarlo literalmente." + inside + NEWLINE + body
    )


#: Statements after which nothing in the same block can run.
TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _unreachable(source: str) -> list[tuple[int, str]]:
    """Statements that can never execute, as (line, the statement's kind).

    Only the certain case: a statement standing directly after a return, raise,
    break or continue in the SAME block. No flow analysis, no cleverness, no
    opinions about style -- just the one thing that is unreachable under every
    reading of the language.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for previous, statement in zip(block, block[1:]):
                if isinstance(previous, TERMINATORS):
                    found.append((getattr(statement, "lineno", 0),
                                  type(statement).__name__))
    return sorted(set(found))


def _unreachable_note(rel: str, before: str | None, after: str) -> str:
    """Report code the edit made unreachable, without nagging about the rest.

    A dogfood candidate passed its acceptance and left a second ``return out``
    below the first. Tester was right -- the behaviour was correct -- and the
    diff still was not promotable, which cost a manual cleanup pass. The harness
    can see it for the price of one parse, and only complains about what THIS
    edit introduced.
    """
    if not rel.endswith(".py"):
        return ""
    new = _unreachable(after)
    if not new:
        return ""
    old = set(_unreachable(before)) if before is not None else set()
    # compare by kind and count rather than by line, because an edit above moves
    # every line below it and would otherwise look like a new defect.
    introduced = [x for x in new if x not in old]
    if before is not None and len(new) <= len(old):
        return ""
    if not introduced:
        return ""
    where = ", ".join(f"linea {line} ({kind})" for line, kind in introduced[:4])
    return (
        f"{NEWLINE}[{ERROR_UNREACHABLE_CODE}: el fichero ha quedado con codigo "
        f"que no se puede ejecutar nunca -- {where} viene justo despues de un "
        f"return/raise/break/continue en el mismo bloque. Los tests pueden pasar "
        f"igual; el codigo sigue estando muerto. Borralo.]"
    )


def edit(ctx: ToolContext, path: Any, old: Any, new: Any) -> str:
    rel, target = _resolve(ctx, path)
    _check_writable(ctx, rel, creating=False)
    if not isinstance(old, str) or not isinstance(new, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "old y new deben ser cadenas")
    old, old_note = _unescape_if_flattened("old", old)
    new, new_note = _unescape_if_flattened("new", new)
    escape_note = old_note + new_note
    if not old:
        raise ToolError(ERROR_EMPTY_OLD, "old no puede estar vacío. Para un fichero nuevo usa write_file.")

    text = _read_text(rel, target)
    count = text.count(old)
    if count == 0:
        preview = "\n".join("    " + l for l in old.splitlines()[:5])
        hint = _nearest_region(text, old).replace("{rel}", rel)
        raise ToolError(
            ERROR_NO_MATCH,
            f"no se encontro el texto en {rel!r}.\n  buscado ({len(old.splitlines())} lineas):\n"
            f"{preview}\n  El fichero tiene {len(text.splitlines())} lineas." + (
                hint or f"\n  Usa read_file({rel!r}, start=, end=) para ver el texto exacto, "
                        f"incluida la indentacion."
            ),
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
                f"la edicion dejaria {rel!r} sin poder parsearse:\n"
                + _syntax_report(candidate, exc, rel)
                + "\n  NO se ha escrito nada. El fichero sigue como estaba.",
            ) from None
        except ValueError as exc:
            raise ToolError(ERROR_SYNTAX_AFTER_EDIT, f"{rel!r}: {exc}. NO se ha escrito nada.") from None

    _write_text(target, candidate)
    ctx.changed_files.add(rel)
    return (f"edit aplicada en {rel}" + escape_note
            + _unreachable_note(rel, text, candidate))


def replace_lines(ctx: ToolContext, path: Any, start: Any, end: Any, content: Any) -> str:
    """Replace lines *start*..*end* (inclusive, 1-based) with *content*.

    The companion to ``edit``, added because ``edit`` alone is unusable on a
    large file inside a finite context (F-26). ``edit`` requires the old text
    reproduced byte for byte; the only source for those bytes is a read whose
    result is subject to elision, so on a 48k-character file the agent ends up
    quoting from a document it was not allowed to keep. The dogfood run failed
    six times running that way, re-reading between every attempt.

    Line numbers do not have that problem. They come free with every read_file
    and every list_symbols, they are four characters long so elision never
    touches them, and they stay valid as long as nothing above them has moved.

    ``edit`` is still the better tool when it works: matching text is
    self-verifying, whereas a line number is only as good as the agent's memory
    of what was on it. This is the fallback for when that self-verification is
    not affordable -- so the two coexist rather than one replacing the other.

    Same protections as ``edit``: write scope, and a Python file that would not
    parse afterwards is refused with nothing written.
    """
    rel, target = _resolve(ctx, path)
    _check_writable(ctx, rel, creating=False)
    if not isinstance(content, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "content debe ser una cadena")
    content, escape_note = _unescape_if_flattened("content", content)
    for name, value in (("start", start), ("end", end)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidCall(ERROR_BAD_ARGUMENTS, f"{name} debe ser un entero (linea, empezando en 1)")

    text = _read_text(rel, target)
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if start < 1 or end < start:
        raise ToolError(
            ERROR_BAD_RANGE,
            f"rango invalido: start={start}, end={end}. start empieza en 1 y end debe ser >= start.",
        )
    if start > total:
        raise ToolError(
            ERROR_BAD_RANGE,
            f"start={start} pero {rel!r} solo tiene {total} lineas. "
            f"Para anadir al final usa start={total} y end={total}.",
        )
    stop = min(end, total)

    body = content if content.endswith("\n") or not content else content + "\n"
    candidate = "".join(lines[: start - 1]) + body + "".join(lines[stop:])

    if rel.endswith(".py"):
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            raise ToolError(
                ERROR_SYNTAX_AFTER_EDIT,
                f"el reemplazo dejaria {rel!r} sin poder parsearse:\n"
                + _syntax_report(candidate, exc, rel)
                + f"\n  NO se ha escrito nada. Comprueba la indentacion de content y "
                f"que el rango {start}-{stop} empieza y acaba donde crees.",
            ) from None
        except ValueError as exc:
            raise ToolError(ERROR_SYNTAX_AFTER_EDIT, f"{rel!r}: {exc}. NO se ha escrito nada.") from None

    _write_text(target, candidate)
    ctx.changed_files.add(rel)
    replaced = stop - start + 1
    written = body.count("\n")
    return (f"replace_lines: {rel} lineas {start}-{stop} ({replaced}) "
            f"sustituidas por {written} lineas"
            + escape_note + _unreachable_note(rel, text, candidate))


def write_file(ctx: ToolContext, path: Any, content: Any) -> str:
    rel, target = _resolve(ctx, path)
    _check_writable(ctx, rel, creating=True)
    if not isinstance(content, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "content debe ser una cadena")
    content, escape_note = _unescape_if_flattened("content", content)
    rewriting = rel in ctx.created_files
    if target.exists() and not rewriting:
        # Still refused for a file that was already in the repository: an
        # accidental whole-file overwrite of real source is the thing this
        # guard exists for, and edit/replace_lines are the right tools there.
        raise ToolError(
            ERROR_FILE_EXISTS,
            f"{rel!r} ya existe y no lo has creado tu en esta mision. "
            f"Para cambiarlo usa edit(path={rel!r}, old=..., new=...) o "
            f"replace_lines.")
    if rel.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as exc:
            raise ToolError(
                ERROR_SYNTAX_AFTER_EDIT,
                f"{rel!r} no parsea:\n"
                + _syntax_report(content, exc, rel)
                + "\n  NO se ha escrito nada.",
            ) from None
        except ValueError as exc:
            raise ToolError(ERROR_SYNTAX_AFTER_EDIT, f"{rel!r}: {exc}. NO se ha escrito nada.") from None
    _write_text(target, content)
    ctx.changed_files.add(rel)
    ctx.created_files.add(rel)
    return (f"write_file aplicada en {rel}" + escape_note
            + _unreachable_note(rel, None, content)
            + _rewrite_note(ctx, rel)
            + _mission_output_ready(ctx))


def _mission_output_ready(ctx: ToolContext) -> str:
    """Say that what the mission asked to be created now exists.

    Measured on the reference engine: of fifty runs, 47 produced an answer and
    5 called finish, and 325 turns -- six and a half per run -- went to
    ERROR_NO_TOOL_CALL after the answer was already on disk. One run had the
    correct answer at turn 3 and spent seventeen more rewriting it.

    The harness knew. It had simply never been asked to say so.

    Not a refusal and not an ending: the harness knows the file exists, not that
    its contents are right, and finish() carries a status the verdict depends
    on. Deciding the mission is over is the agent's to do.
    """
    if not ctx.allowed_new_files:
        return ""
    sizes = []
    for name in ctx.allowed_new_files:
        target = ctx.root / name
        try:
            if not target.is_file():
                return ""
            size = target.stat().st_size
        except OSError:
            return ""
        if size <= 0:
            return ""
        sizes.append((name, size))
    listed = ", ".join(f"{n} ({b} caracteres)" for n, b in sizes[:3])
    return (f"{NEWLINE}[Ya existe lo que pedia la mision: {listed}. Si el "
            f"contenido es el que hacia falta, CIERRA con "
            f"finish(status='DONE', summary='<que has hecho>'). Volver a "
            f"escribir lo mismo no cambia nada y gasta turnos.]")


def _copied_too_much(node, name: str, src_rel: str, into_rel: str,
                     lo: int, hi: int, total_lines: int) -> str:
    """Say when what came back is bigger than the caller probably meant.

    F-70 made read_symbol say this because a run scored 0.38 on an answer it
    believed was exact. copy_code needs it more, not less: it does not return
    the body, so an over-copy leaves no trace the agent can see.
    """
    copied = hi - lo + 1
    if node is not None and isinstance(node, ast.ClassDef):
        methods = [n.name for n in node.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        shown = ", ".join(methods[:6]) + (" ..." if len(methods) > 6 else "")
        return (f"{NEWLINE}[OJO: {name!r} es una CLASE entera de {copied} lineas, "
                f"no una funcion, y se ha copiado completa en {into_rel}. "
                f"Contiene {len(methods)} metodos: {shown}."
                f"{NEWLINE}  Si querias UNO de ellos, pidelo por su nombre: "
                f"copy_code(src={src_rel!r}, into={into_rel!r}, "
                f"name='{name}.{methods[0]}') -- pero borra antes lo que acabas "
                f"de escribir o quedaran los dos.]" if methods else "")
    if total_lines and copied >= max(40, int(total_lines * 0.6)):
        return (f"{NEWLINE}[OJO: has copiado {copied} de las {total_lines} lineas "
                f"de {src_rel}, es decir casi el fichero entero. Si lo que hacia "
                f"falta era una sola definicion, pidela por su nombre con "
                f"name= en vez de por lineas.]")
    return ""


def copy_code(ctx: ToolContext, src: Any, into: Any, name: Any = None,
              start: Any = None, end: Any = None) -> dict:
    """Copy a region of one file into another, byte for byte, without retyping it.

    Either ``name`` (a symbol, where the language has a symbol index) or
    ``start``/``end`` (line numbers, anywhere). The bytes come off disk; the
    engine never carries them.

    This is not a convenience. Carrying a payload is a measured limit:
    granite4.1:3b loses one above 1600 characters through the text protocol and
    above 800 natively, while the median function this harness is asked about is
    longer than the first of those -- so there are answers it knows and cannot
    emit. And read_symbol's own docstring records the cost for the reference
    engine: the same model met the same file both ways and reproduced the
    function byte for byte 72% of the time as plain text against 30% reading it
    through the tools and writing it back.

    The same act -- moving a definition verbatim -- is what an extract-to-module
    refactor is made of, which is why dogfood08's frozen acceptance asserts
    byte-identity rather than equivalence.

    Appends when the destination exists AND this run created it. It will not
    overwrite a file that was already in the repository: that is what edit and
    replace_lines are for.
    """
    # NOT must_exist=True. The guard reports "does not exist" as a
    # ContainmentError, which _resolve turns into ERROR_PATH_OUTSIDE_REPO -- so a
    # simple wrong path was reported to granite4.1:3b six times as a containment
    # violation, which is both false and undiagnosable. _read_text below raises
    # ERROR_FILE_NOT_FOUND with the directory's actual contents, exactly as
    # read_file does. Same defect class as A3: a safety error standing in for an
    # ordinary mistake teaches the agent nothing and looks alarming.
    src_rel, src_path = _resolve(ctx, src)
    into_rel, into_path = _resolve(ctx, into)
    source = _read_text(src_rel, src_path)
    ctx.opened.add(src_rel)

    has_name = isinstance(name, str) and name.strip()
    has_lines = start is not None or end is not None
    if has_name and has_lines:
        # F-93. Sending both is not a contradiction by itself -- it is what a
        # model does after read_symbol PRINTS the span: it repeats the name it
        # asked for and the line numbers it was just shown. When the two say the
        # same thing there is exactly one objective answer, and refusing it cost
        # a turn spent deleting an argument that was correct. Measured on the
        # expansion cohort: 18 runs across four engines, 10 of them
        # ministral-3:3b, every retry identical but for the dropped name.
        #
        # When they DISAGREE the intent really is ambiguous -- a class name with
        # one method's lines, a symbol with the whole file's range -- and that
        # stays a refusal, now quoting the span the name actually has so the
        # next call can be right. The harness completes a fact it knows; it does
        # not decide a question the model has left open.
        total = len(source.splitlines())
        span = None
        try:
            _n, _node, _found, _lo, _hi = _symbol_span(src_rel, source, name.strip())
            span = (_lo, _hi)
        except (ToolError, InvalidCall):
            span = None
        agreed = (span is not None and start is not None and end is not None
                  and _line_number("start", start, total) == span[0]
                  and _line_number("end", end, total) == span[1])
        if not agreed:
            where = ""
            if span is not None:
                where = (NEWLINE + f"  {name.strip()!r} ocupa las lineas "
                         f"{span[0]}-{span[1]} en {src_rel!r}, que no es lo que "
                         f"has pedido.")
            raise InvalidCall(
                ERROR_BAD_ARGUMENTS,
                "copy_code toma name= O start=/end=, no las dos, porque en esta "
                "llamada no dicen lo mismo." + where
                + NEWLINE + "  Copia el simbolo entero con name=, o el rango "
                  "exacto con start=/end=, pero no los dos.")
        has_lines = False
        start = end = None
    if not has_name and not has_lines:
        # The path is already resolved and the file is already read, so the
        # question this call was asking -- "what can I copy out of here?" -- can
        # be answered right now instead of being handed back as homework.
        # granite4.1:3b reached for copy_code correctly on turn one with only
        # the selector missing, was told to go and use list_symbols, did, and
        # never came back. Where the harness can answer, it answers.
        catalogue = ""
        try:
            names = sorted(_collect_symbols(ast.parse(source)))
        except (SyntaxError, ValueError, RecursionError):
            names = []
        except ToolError:
            names = []
        if names:
            # In FILE order, and it now says so. The first version listed
            # them and showed a worked example using names[0], which turned
            # an error message into a menu: granite4.1:3b took the first
            # symbol offered in 12 of 50 runs -- ALWAYS_NO_SPACE when it
            # wanted visit, LazyProxy when it wanted __getattr__ -- and
            # finished DONE, confident. An unranked list presented as a
            # choice is worse than no list.
            #
            # The ranking still does not happen here; errors are not where
            # retrieval belongs. It points at the tool that ranks, with the
            # call filled in, which F-79 made possible by letting
            # search_code scope to a single file.
            shown = ", ".join(names[:12])
            catalogue = (
                f"\n  En {src_rel!r} hay {len(names)} simbolos, EN ORDEN DE "
                f"FICHERO y no por lo que buscas: {shown}"
                + (f" (+{len(names) - 12} mas)" if len(names) > 12 else "")
                + f".\n  Si NO sabes cual necesitas, no elijas uno al azar: "
                  f"search_code(query='<la descripcion de lo que buscas>', "
                  f"path={src_rel!r}) te los ordena por parecido."
                  f"\n  Cuando sepas cual es: copy_code(src={src_rel!r}, "
                  f"into={into_rel!r}, name='<ese>')")
        else:
            catalogue = (f"\n  {src_rel!r} tiene {len(source.splitlines())} lineas; "
                         f"copialas con start= y end=.")
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            "copy_code necesita name=<simbolo> o start=/end=<lineas>." + catalogue)

    lines = source.splitlines()
    node = None
    if has_name:
        name, node, _found, lo, hi = _symbol_span(src_rel, source, name.strip())
    else:
        lo = _line_number("start", start if start is not None else 1, len(lines))
        hi = _line_number("end", end, len(lines)) if end is not None else len(lines)
        if hi < lo:
            raise InvalidCall(ERROR_BAD_ARGUMENTS,
                              f"end ({hi}) es menor que start ({lo})")
    body = NEWLINE.join(lines[lo - 1:hi])

    existing = ""
    if into_path.exists():
        if into_rel not in ctx.created_files:
            raise ToolError(
                ERROR_FILE_EXISTS,
                f"{into_rel!r} ya existe y no lo has creado tu en esta mision. "
                f"copy_code no sobrescribe codigo que ya estaba en el repositorio; "
                f"para eso estan edit y replace_lines.")
        _check_writable(ctx, into_rel, creating=False)
        existing = _read_text(into_rel, into_path)
    else:
        _check_writable(ctx, into_rel, creating=True)

    separator = "" if not existing or existing.endswith(NEWLINE) else NEWLINE
    _write_text(into_path, existing + separator + body)
    ctx.changed_files.add(into_rel)
    ctx.created_files.add(into_rel)
    # Counted here as well as in write_file, so "how many things are in this
    # file" is one number rather than two that disagree.
    ctx.write_counts[into_rel] = ctx.write_counts.get(into_rel, 0) + 1
    return {
        # Everything a reviewer needs to check the bytes came off disk and not
        # out of the model. A tool that moves text nobody saw would otherwise be
        # the one place a run could not be audited.
        "copied_from": src_rel,
        "symbol": name if has_name else None,
        "lines": [lo, hi],
        "into": into_rel,
        "bytes": len(body.encode("utf-8")),
        "sha256_16": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
        "appended": bool(existing),
        # The ends only. Returning the body would put back into the transcript
        # exactly the weight this tool exists to keep out of it.
        "first_line": lines[lo - 1] if lines[lo - 1:hi] else "",
        "last_line": lines[hi - 1] if lines[lo - 1:hi] else "",
        # The body never comes back -- that is the point of the tool -- so an
        # over-copy would otherwise be invisible. F-70 for read_symbol, and it
        # matters more here.
        "note": (_copied_too_much(node, name if has_name else "", src_rel,
                                  into_rel, lo, hi, len(lines))
                 + _appended_note(ctx, into_rel, bool(existing))
                 + _mission_output_ready(ctx)),
    }


def _appended_note(ctx: ToolContext, into_rel: str, appended: bool) -> str:
    """Say that this copy was ADDED to what was already there.

    copy_code appends into a file this run created, which is what an
    extract-to-module refactor needs: several definitions moved into one new
    file. On a mission whose output is a single answer it silently concatenates,
    and the reference engine's v7 runs show exactly that:

        reached the symbol and was RIGHT   30 runs, 2.0 copies, 396 chars
        reached the symbol and was WRONG    8 runs, 4.5 copies, 963 chars

    The failing ones copied the right symbol AND several others. Conversion fell
    92% -> 86% -> 82% -> 77% across the versions where the turn-zero shortlist
    started naming a candidate to copy. Two of my own changes, each defensible,
    combining into a defect neither had alone.

    A note and not a refusal: appending is sometimes precisely the intent. What
    was missing is that the agent could not see it had happened -- copy_code
    deliberately does not return the body, so a concatenation leaves no trace in
    the transcript.
    """
    if not appended:
        return ""
    total = ctx.write_counts.get(into_rel, 0)
    return (f"{NEWLINE}[OJO: {into_rel} YA tenia contenido y esto se ha ANADIDO "
            f"al final, no lo ha sustituido. El fichero lleva ya {total} "
            f"copias. Si la respuesta es UNA sola definicion, ahora hay de mas: "
            f"reescribelo con write_file(path='{into_rel}', content=<solo la "
            f"que quieres>) antes de terminar.]")


def _line_number(field: str, value: Any, total: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(str(value).strip())
        except (TypeError, ValueError):
            raise InvalidCall(ERROR_BAD_ARGUMENTS,
                              f"{field} debe ser un numero de linea") from None
    if value < 1:
        raise InvalidCall(ERROR_BAD_ARGUMENTS, f"{field} empieza en 1, no en {value}")
    if value > total:
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            f"{field}={value} pasa del final del fichero ({total} lineas)")
    return value



def list_dir(ctx: ToolContext, path: Any = ".", recursive: Any = False) -> dict:
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
    if not isinstance(recursive, bool):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "recursive debe ser booleano")
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
    if recursive:
        pending = [(target, "")]
        while pending:
            current, prefix = pending.pop(0)
            try:
                children = sorted(current.iterdir(), key=lambda q: (q.is_file(), q.name.lower()))
            except OSError:
                continue
            for entry in children:
                if entry.name in SKIP_DIRS:
                    continue
                child_name = f"{prefix}{entry.name}"
                try:
                    if entry.is_dir():
                        dirs.append(child_name + "/")
                        pending.append((entry, child_name + "/"))
                    else:
                        files.append({"name": child_name, "bytes": entry.stat().st_size})
                except OSError:
                    continue
    else:
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
        # Every file in this directory, not only the ones that fitted. A caller
        # cannot add up the sizes it was not shown, and the truncated listing is
        # exactly when it most wants the number (dogfood07).
        "total_bytes": sum(entry["bytes"] for entry in files),
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
    # F-112. The tools cannot stop a child writing outside the repository and
    # this does not pretend to: it looks before and after, at a bounded watch
    # set, and reports what it saw AND what it could not see. An honest partial
    # check that runs every time beats a thorough one that is too slow to keep.
    watch = _sentinel.Sentinel(workspace=ctx.root, source_repo=ctx.source_repo,
                               argv=real_argv)
    watch.arm()
    try:
        proc = subprocess.run(
            real_argv, cwd=str(ctx.root), capture_output=True, text=True,
            timeout=limit, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        head = exc.stdout if isinstance(exc.stdout, str) else ""
        tail = exc.stderr if isinstance(exc.stderr, str) else ""
        seen = watch.check()
        ctx.outside_writes.append(seen)
        ctx.commands_run.append({"argv": argv, "exit_code": None, "timed_out": True,
                                 "sentinel": seen})
        return {
            "argv": argv, "kind": kind, "exit_code": None, "timed_out": True,
            "output": (head + tail)[-RUN_OUTPUT_TAIL:] + f"\n[TIMEOUT tras {limit:g}s]",
            "sentinel": seen,
        }
    except OSError as exc:
        # Allowlisted but not installed. That is a fact about this machine the
        # agent can act on, not a defect in the harness.
        raise ToolError(
            ERROR_COMMAND_NOT_ALLOWED, f"no se pudo ejecutar {argv[0]!r}: {exc}"
        ) from None

    output = proc.stdout + proc.stderr
    seen = watch.check()
    ctx.outside_writes.append(seen)
    ctx.commands_run.append({"argv": argv, "exit_code": proc.returncode,
                             "timed_out": False, "sentinel": seen})
    result = {
        "argv": argv, "kind": kind, "exit_code": proc.returncode,
        "timed_out": False, "output": output[-RUN_OUTPUT_TAIL:],
        "sentinel": seen,
    }
    if not _sentinel.clean(seen):
        # Told to the agent, not hidden in the record. A command that wrote
        # outside the workspace is something it needs to know it did.
        result["note"] = (result.get("note", "") +
                          "\n[OJO: este comando escribio FUERA del workspace: "
                          + ", ".join(f"{c['kind']} {c['name']}"
                                      for c in seen["outside_writes"][:5])
                          + ". El workspace es el unico sitio donde puedes "
                            "trabajar.]")
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

    argv = [
        sys.executable, "-B", "-m", "pytest", "-q",
        # Short tracebacks and a failure summary. The default traceback style
        # spends most of the output budget on frames from pytest's own
        # internals, which tells the agent nothing about its code (F-47).
        "--tb=short", "-rf", "-p", "no:cacheprovider",
        "--continue-on-collection-errors",
        *targets,
    ]
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
    covers_acceptance = set(ctx.acceptance_tests) <= {_normalise(t) for t in targets}
    if passed and covers_acceptance:
        ctx.tests_green = True
    output = (proc.stdout + proc.stderr)[-TEST_OUTPUT_TAIL:]
    failing = sorted({
        line.split(" - ")[0].removeprefix("FAILED ").removeprefix("ERROR ").strip()
        for line in output.splitlines()
        if line.startswith("FAILED ") or line.startswith("ERROR ")
    })
    result = {
        "passed": passed,
        "exit_code": proc.returncode,
        "timed_out": False,
        # Structured, so the names survive even if the text is clipped. The
        # output tail is the first thing to be cut and the failing test names
        # are the last thing that should be lost (F-47).
        "failing": failing,
        "output": output,
    }
    # F-42: remember what failed. Several turns later this output has been
    # elided, and an agent being refused a finish cannot be expected to recall
    # it -- but the harness can simply keep it.
    if not passed:
        ctx.last_test_failure = output[-1500:]
        ctx.last_failing_tests = sorted({
            line.split(" - ")[0].removeprefix("FAILED ").removeprefix("ERROR ").strip()
            for line in output.splitlines()
            if line.startswith("FAILED ") or line.startswith("ERROR ")
        })
    else:
        ctx.last_test_failure = ""
        ctx.last_failing_tests = []

    # F-38. The acceptance suite is a sample. The conscience judges the whole
    # repository, and until now the agent could not see the difference: it ran
    # the declared tests, saw green, and stopped -- while a test it was supposed
    # to have updated sat red somewhere it was never shown. tests01 failed
    # exactly that way in both sweeps and was escalated to a paid tier for it.
    #
    # So the moment the acceptance goes green, the collateral check runs here
    # and the answer is appended to this same result. No new argument: a small
    # model does not need another decision, it needs the fact.
    if passed and covers_acceptance and ctx.baseline_outcomes and ctx.full_suite_runner:
        try:
            now = ctx.full_suite_runner(ctx.root)
        except Exception as exc:  # noqa: BLE001 - never let the check break the run
            result["collateral"] = {"checked": False, "error": f"{type(exc).__name__}: {exc}"}
            return result
        broke = sorted(
            node for node, was in ctx.baseline_outcomes.items()
            if was == "passed" and now.get(node) in ("failed", "error")
        )
        ctx.known_regressions = broke
        ctx.collateral_checked = True
        if broke:
            shown = ", ".join(broke[:8]) + (" ..." if len(broke) > 8 else "")
            result["collateral"] = {
                "checked": True, "broken": broke,
                "note": (
                    f"[ATENCION: los tests de aceptacion pasan, pero has ROTO "
                    f"{len(broke)} test(s) que antes pasaban: {shown}. "
                    f"Eso cuenta como fallo. Arreglalo antes de terminar: "
                    f"leelos con read_file y ejecutalos con "
                    f"run_tests(node_ids=[...]) para ver el error.]"
                ),
            }
        else:
            result["collateral"] = {
                "checked": True, "broken": [],
                "note": "[aceptacion en verde y no has roto ningun otro test.]",
            }
    return result


def _navigation_note(ctx: ToolContext) -> str:
    """What the agent's own session says about where it has and has not looked.

    Facts, not hints: how many candidates a search returned, and which of the
    best-ranked ones were never opened. Measured on the dev set, 15 of 100 runs
    were shown the right function by search_code and then finished without ever
    reading it, or read it and declared nothing matched. The harness held both
    halves of that and said neither.
    """
    if not ctx.top_candidates:
        return ""
    if ctx.acceptance_tests:
        # There is a suite, so the open question is not "where is it" -- it is
        # "why is that test red", and the gate answers that a few lines above.
        # Pointing at unopened search candidates here sends an agent that knows
        # exactly what is failing off to read files instead.
        return ""
    unopened = [c for c in ctx.top_candidates if c[0] not in ctx.opened]
    if not unopened:
        return (f"{NEWLINE}  Has abierto los {len(ctx.top_candidates)} mejores "
                f"candidatos que te devolvio search_code, asi que descartarlos "
                f"esta justificado.")
    shown = NEWLINE.join(f"    {path} (lineas {span}) {decl[:70]}"
                         for path, span, decl in unopened[:4])
    return (f"{NEWLINE}  search_code te devolvio estos candidatos bien puntuados "
            f"y NO has abierto ninguno:{NEWLINE}{shown}{NEWLINE}"
            f"  Mirarlos cuesta una llamada.")


def finish(ctx: ToolContext, summary: Any = None, status: Any = "DONE") -> str:
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
    # summary is documentation, not evidence. signal_agent_reported takes
    # finish_status for the signal and only quotes the summary in its
    # message, and I9/F-24 hold: the agent's claim cannot create a PASS and,
    # since ga06, cannot destroy one either.
    #
    # It used to be REQUIRED, and qwen2.5-coder:3b hit that 57 times in 50
    # runs with finish(status='DONE') and nothing else. Refusing there does
    # not protect the record, it damages it: a run that never reaches finish
    # seals as finish_status=None -- 'agoto el presupuesto', INCONCLUSIVE --
    # which says less than DONE with an empty summary. Absence is recorded.
    if summary is None:
        summary = ""
    if not isinstance(summary, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "summary debe ser una cadena")
    missing_summary = not summary.strip()
    if not isinstance(status, str):
        raise InvalidCall(ERROR_BAD_ARGUMENTS, "status debe ser una cadena")
    normalised = status.strip().upper() or "DONE"
    if normalised not in FINISH_STATUSES:
        raise InvalidCall(
            ERROR_BAD_ARGUMENTS,
            f"status debe ser uno de {', '.join(FINISH_STATUSES)}; recibido {status!r}",
        )
    if normalised == "DONE" and not ctx.changed_files:
        # F-71. The work first, the exits second. This message used to offer
        # only NO_CHANGE and BLOCKED, and a weaker engine reads the options it
        # is given: 17 of 50 granite4.1:3b runs took the NO_CHANGE door while
        # the file the mission was waiting for had never been written.
        pending = [name for name in ctx.allowed_new_files
                   if not (ctx.root / name).exists()]
        wanted = ""
        if pending:
            wanted = (
                f"\n  ESTA MISION ESPERA QUE CREES: {', '.join(pending[:4])}. "
                f"Todavia no existe."
            )
            # The agent's OWN resolved symbol first, the harness's turn-zero
            # guess only as a fallback (F-94). Both name a concrete call --
            # which is what matters, since qwen2.5-coder:3b called finish twenty
            # times against a message that described write_file in prose, and
            # the ablation showed the turn-zero shortlist is worth 60% vs 20%.
            # But when the agent has already reached a symbol, proposing a
            # different file is the harness overriding a decision that was made
            # correctly, and two runs spent their whole budget on that.
            reached = ctx.reached_symbols[-1] if ctx.reached_symbols else None
            if reached:
                src, symbol = reached
                wanted += (f"\n  Ya has leido {symbol!r} en {src!r} y NO lo has "
                           f"copiado a ningun sitio. Si es la respuesta: "
                           f"copy_code(src='{src}', into='{pending[0]}', "
                           f"name='{symbol}').")
            elif ctx.opening_candidate:
                src, symbol = ctx.opening_candidate
                wanted += (f"\n  Si la respuesta ya esta en el repositorio, "
                           f"copiala sin reescribirla: copy_code(src='{src}', "
                           f"into='{pending[0]}', name='{symbol}').")
            wanted += (
                f"\n  Si tienes que escribirla tu: write_file(path='{pending[0]}', "
                f"content=<el texto entero>). El contenido va en el argumento, "
                f"no en tu respuesta."
            )
        elif ctx.write_scope:
            wanted = (f"\n  Puedes escribir en: {', '.join(ctx.write_scope[:4])}. "
                      f"Usa edit o replace_lines sobre lo que haya que cambiar.")
        raise ToolError(
            ERROR_NOTHING_CHANGED,
            "no has editado nada, asi que no puedes terminar con status='DONE'."
            + wanted
            + "\n  Si de verdad no hace falta ningun cambio: finish(status='NO_CHANGE', "
              "summary='<por que>').\n"
              "  Si no puedes continuar: finish(status='BLOCKED', summary='<que te lo impide>').",
        )
    # F-29: DONE means "I did it and I checked", not "I did something".
    #
    # The dogfood agent edited grep() correctly, never called run_tests, and
    # declared DONE on turn 12 of a 40-turn budget. It had not noticed that
    # adding a tool argument in this repository also means declaring it in
    # SPECS and documenting it in PARAM_DOC -- which the acceptance suite says
    # in its first line of output. It never asked.
    #
    # The prompt already told it to run the tests first. Saying a thing is not
    # enforcing it. This is not the harness deciding whether the work is good:
    # I9 still holds and the real verdict is taken after the loop, against a
    # suite the agent cannot influence. It is the harness declining to accept a
    # claim of completion from an agent that never looked.
    # F-38: refuse DONE while a collateral regression is known to stand. The
    # conscience fails exactly this, so stopping the agent one line behind it --
    # letting it claim success and only then failing it -- wastes the run and
    # escalates a problem it was perfectly capable of fixing.
    if normalised == "DONE" and ctx.known_regressions:
        shown = ", ".join(ctx.known_regressions[:8])
        raise ToolError(
            ERROR_NOT_VERIFIED,
            f"no puedes terminar con status='DONE': has roto {len(ctx.known_regressions)} "
            f"test(s) que antes pasaban.\n  {shown}\n"
            f"  Arreglalos y vuelve a ejecutar run_tests. Si crees que ese fallo es "
            f"correcto y esperado, explicalo con finish(status='BLOCKED').",
        )
    if normalised == "DONE" and ctx.acceptance_tests and not ctx.tests_green:
        # F-42. Telling an agent to go and read output it has already read, and
        # which has since been elided, is not feedback. ga04 was refused ten
        # times this way while sitting on 14 of 16 tests green. What it needed
        # was the name of the one that was not.
        detail = ""
        if ctx.last_failing_tests:
            detail = (
                "\n  Lo que falla ahora mismo: "
                + ", ".join(ctx.last_failing_tests[:6])
                + ("" if len(ctx.last_failing_tests) <= 6 else " ...")
            )
        if ctx.last_test_failure:
            detail += "\n  Ultimo error:\n" + "\n".join(
                "    " + l for l in ctx.last_test_failure.strip().splitlines()[-12:]
            )
        raise ToolError(
            ERROR_NOT_VERIFIED,
            "no puedes terminar con status='DONE': los tests de aceptacion no "
            "estan en verde." + (detail or "\n  Ejecuta run_tests() y lee el resultado.")
            + "\n  Arregla eso y vuelve a ejecutar run_tests. Si de verdad no puedes: "
            "finish(status='BLOCKED', summary='<que te lo impide>').",
        )
    # F-61. NO_CHANGE with an empty diff gets the same single question BLOCKED
    # gets, and for the same reason: the harness cannot tell "I examined this
    # and there is genuinely nothing to do" from "I think I already did it"
    # without asking, and only one of those is true.
    #
    # It is asked with FACTS, not suspicion: nothing was created or modified,
    # and this is what the mission said you could create. Seven runs ended in
    # NO_CHANGE while their own summary said the deliverable had been written;
    # the harness knew the file did not exist and said nothing. A second
    # NO_CHANGE is accepted immediately, always.
    if (
        normalised == "NO_CHANGE"
        and not ctx.changed_files
        and not ctx.no_change_once
        and ctx.turns_left is not None
        and ctx.turns_left >= 2
    ):
        ctx.no_change_once = True
        expected = ""
        if ctx.allowed_new_files:
            expected = ("\n  Esta mision te autorizaba a CREAR: "
                        + ", ".join(ctx.allowed_new_files[:6])
                        + ". Ninguno existe todavia.")
        elif ctx.write_scope:
            expected = ("\n  Esta mision te autorizaba a escribir en: "
                        + ", ".join(ctx.write_scope[:6]) + ".")
        raise ToolError(
            ERROR_NOTHING_CHANGED,
            "antes de aceptar 'no hace falta ningun cambio': NO has creado ni "
            "modificado NINGUN fichero en todo el run." + expected
            + _navigation_note(ctx)
            + "\n  Si crees que ya escribiste algo, no llego a disco: compruebalo "
            "con read_file o list_dir y escribelo ahora si falta.\n"
            "  Si de verdad no hay nada que hacer, vuelve a llamar a "
            "finish(status='NO_CHANGE') y lo acepto sin mas preguntas.",
        )
    # F-45. BLOCKED is a good outcome and must stay cheap to reach -- but
    # "I tried once and it did not work" is not "I cannot do this", and the
    # harness cannot tell them apart without asking. ga07 gave up on turn 10 of
    # 40 after a single failed import fix.
    #
    # So the FIRST blocked, while the acceptance has never once been green and
    # a third of the budget is unspent, is answered with the last failure and
    # one question. A second BLOCKED is accepted immediately, always -- the same
    # shape as the double-finish that already confirms NO_CHANGE.
    # F-67: the acceptance_tests condition used to be here, and it meant the
    # question never fired for a mission that declares none -- which is most of
    # the ones where giving up early is the whole failure. What justifies asking
    # is unspent budget and a first refusal, not whether a suite exists.
    if (
        normalised == "BLOCKED"
        and not ctx.blocked_once
        and ctx.turns_left is not None
        # A floor as well as a fraction: on a 4-turn budget a third is one
        # turn, and spending it on a question leaves nothing to act on.
        and ctx.turns_left >= max(5, ctx.max_turns_hint // 3)
        and not ctx.tests_green
    ):
        ctx.blocked_once = True
        detail = ""
        if ctx.last_failing_tests:
            detail = "\n  Lo que falla: " + ", ".join(ctx.last_failing_tests[:6])
        if ctx.last_test_failure:
            detail += "\n  Ultimo error:\n" + "\n".join(
                "    " + l for l in ctx.last_test_failure.strip().splitlines()[-10:]
            )
        detail += _navigation_note(ctx)
        raise ToolError(
            ERROR_NOT_VERIFIED,
            f"antes de darte por vencido: todavia te quedan {ctx.turns_left} turnos."
            + detail
            + "\n  Si se te ocurre algo mas que probar, hazlo: lee el fichero otra vez, "
            "ejecuta algo con run, o prueba otro enfoque.\n"
            "  Si de verdad estas atascado, vuelve a llamar a finish(status='BLOCKED') "
            "y lo acepto sin mas preguntas.",
        )
    ctx.finish_status = normalised
    ctx.finish_summary = summary.strip()
    if missing_summary:
        # Recorded, not papered over: a reader can tell 'said nothing' from
        # 'said this', and the sealed record keeps that difference.
        return (f"FINISHED[{normalised}] (sin summary: no has dicho que has "
                f"hecho, y eso queda asi en el registro)")
    return f"FINISHED[{normalised}]"


# ------------------------------------------------------------------ dispatch

#: name -> (required, optional). The single source of truth for both the
#: native tool schema sent to the provider and the argument check below, so
#: the two can never drift -- the previous runner advertised ``list_dir`` in
#: its schema and had no such tool, and advertised ``run`` while the contract
#: says ``run_tests``.
SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "read_file": (("path",), ("start", "end")),
    "list_dir": ((), ("path", "recursive")),
    "grep": (("pattern",), ("glob", "context", "ignore_case")),
    "search_code": (("query",), ("limit", "path")),
    "list_symbols": (("path",), ()),
    "read_symbol": (("path", "name"), ()),
    "edit": (("path", "old", "new"), ()),
    "replace_lines": (("path", "start", "end", "content"), ()),
    "write_file": (("path", "content"), ()),
    "copy_code": (("src", "into"), ("name", "start", "end")),
    "run": (("argv",), ("timeout",)),
    "run_tests": ((), ("node_ids",)),
    "finish": ((), ("summary", "status")),
}

_IMPL = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "search_code": search_code,
    "list_symbols": list_symbols,
    "read_symbol": read_symbol,
    "edit": edit,
    "replace_lines": replace_lines,
    "write_file": write_file,
    "copy_code": copy_code,
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
        if name in INFORMATION_TOOLS:
            # Something came back that the agent did not already have, so the
            # next write is acting on it rather than re-guessing.
            ctx.learned_since_write = True
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
        "Sin argumentos lista la raiz. Con recursive=True incluye tambien el "
        "contenido de los subdirectorios, usando rutas relativas."
    ),
    "grep": (
        "Busca una expresion regular de Python en los ficheros del repositorio y "
        "devuelve fichero, linea y texto de cada coincidencia. Es la forma de "
        "encontrar donde se define o se usa algo cuando no sabes en que fichero "
        "esta. Con context=N te devuelve ademas N lineas antes y despues de cada "
        "coincidencia, asi que muchas veces te ahorra el read_file siguiente. "
        "Con ignore_case=True busca sin distinguir mayusculas y minusculas."
    ),
    "search_code": (
        "Busca por SIGNIFICADO, no por texto literal: le das una descripcion en "
        "palabras de lo que hace el codigo que buscas y te devuelve los sitios "
        "del repositorio que mas se le parecen, ordenados, con fichero, rango de "
        "lineas y la linea de declaracion. Es la herramienta para empezar cuando "
        "NO sabes como se llama ni donde esta lo que buscas -- grep necesita que "
        "aciertes el literal exacto, esto no. Funciona en cualquier lenguaje. "
        "Despues lee el candidato que encaje con read_symbol o read_file."
    ),
    "list_symbols": (
        "Devuelve lo que define un fichero Python -- funciones, clases y tambien "
        "las constantes y tablas de modulo -- con su numero de linea, en forma "
        "'Clase.metodo'. Mas barato que leer el fichero entero cuando solo "
        "quieres saber que hay dentro."
    ),
    "read_symbol": (
        "Devuelve el codigo de UNA funcion, clase o constante por su nombre, "
        "con sus "
        "numeros de linea reales. Es la forma barata de mirar algo concreto "
        "dentro de un fichero grande: no tienes que adivinar el rango ni leerlo "
        "entero. Acepta 'funcion' o 'Clase.metodo'. Los numeros de linea que "
        "devuelve sirven tal cual para replace_lines."
    ),
    "edit": (
        "Sustituye un fragmento de texto exacto por otro dentro de un fichero que "
        "ya existe. El fragmento 'old' debe aparecer EXACTAMENTE UNA VEZ en el "
        "fichero, con su indentacion original; si aparece varias veces, anade "
        "lineas de contexto alrededor hasta que sea unico. Si el resultado no "
        "seria Python valido, no se escribe nada y te lo digo. Puedes hacer varias "
        "ediciones seguidas sobre el mismo fichero."
    ),
    "replace_lines": (
        "Sustituye un rango de lineas de un fichero por un texto nuevo. "
        "Alternativa a edit cuando no puedes reproducir el texto original "
        "exactamente: aqui solo necesitas los numeros de linea, que te dan "
        "read_file y list_symbols. Las lineas se cuentan desde 1 y el rango "
        "incluye start y end. Cuidado: si editas antes, los numeros de despues "
        "se desplazan; vuelve a leer si tienes dudas."
    ),
    "write_file": (
        "Crea un fichero NUEVO con el contenido dado. Falla si el fichero ya "
        "existe y no lo has creado tu en esta mision: para modificar uno que ya "
        "estaba en el repositorio usa edit o replace_lines."
    ),
    "copy_code": (
        "Copia codigo de un fichero a otro TAL CUAL, sin que tengas que "
        "reescribirlo. Dile de donde (src) y a donde (into), y luego O BIEN "
        "name=<nombre de la funcion o clase> O BIEN start=/end=<lineas>. El "
        "texto lo mueve la herramienta leyendolo del disco, asi que sale "
        "identico byte a byte aunque sea largo. Usala siempre que tengas que "
        "reproducir codigo que ya existe en el repositorio: copiarlo a mano es "
        "mas lento y se pierden espacios, sangrados y comillas. Si el destino "
        "ya lo creaste tu, anade al final."
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
        "como averiguas que arreglar. Cuando la aceptacion pasa, ademas te digo "
        "si has roto algun otro test del repositorio que antes pasaba."
    ),
    "finish": (
        "Termina tu trabajo. status='DONE' SOLO despues de ejecutar run_tests y "
        "ver que pasan, "
        "status='NO_CHANGE' si has comprobado que no hace falta ningun cambio, "
        "status='BLOCKED' si no puedes continuar (explica en summary que te lo "
        "impide). No llames a finish con DONE sin haber editado nada."
    ),
}

#: Per-parameter prose. Same reasoning as TOOL_DOC: a parameter whose meaning
#: is not stated is a parameter the model has to guess.
PARAM_DOC: dict[str, str] = {
    "path": "Ruta relativa a la raiz del repositorio, con barras normales: 'pkg/mod.py'.",
    "src": "Fichero del que se copia, relativo a la raiz del repositorio.",
    "into": "Fichero al que se copia. Si no existe se crea; si lo creaste tu en esta mision, se anade al final.",
    "recursive": "Booleano; si es True, lista tambien el contenido de los subdirectorios con rutas relativas. Por defecto False.",
    "start": "Primera linea, empezando en 1. En read_file, null lee desde el principio.",
    "end": "Ultima linea, incluida. En read_file, null lee hasta el final.",
    "pattern": "Expresion regular de Python. Se busca linea a linea.",
    "query": "Que buscas, en palabras. PEGA EL TEXTO DEL OBJETIVO TAL CUAL, sin resumirlo: cuantas mas palabras le des, mejor ordena. Resumir la descripcion en cuatro palabras empeora el resultado.",
    "limit": "Cuantos candidatos devolver (1-50). Por defecto 15.",
    "glob": "Que ficheros mirar, p.ej. '**/*.py' (por defecto) o 'tests/**/*.py'.",
    "context": "Lineas de contexto alrededor de cada coincidencia (0-20). 0 solo da la linea.",
    "ignore_case": "Booleano; si es True, busca sin distinguir mayusculas y minusculas. Por defecto False.",
    "name": "Nombre del simbolo: 'mi_funcion', 'MiClase.mi_metodo' o 'MI_CONSTANTE'.",
    "old": "El texto exacto que hay ahora en el fichero, incluida su indentacion. Debe ser unico.",
    "new": "El texto que lo sustituye. Cadena vacia para borrar el fragmento.",
    "content": "Contenido nuevo: el fichero entero en write_file, o el texto que sustituye al rango en replace_lines.",

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


#: Tools that need an EXISTING file inside the write scope. On a mission whose
#: scope is empty -- only new files may be created -- no argument can satisfy
#: them, so offering them is offering a dead end.
_NEEDS_EXISTING_WRITABLE = ("edit", "replace_lines")

#: Tools that need somewhere to write at all.
_NEEDS_ANY_WRITE = ("write_file",)


def legal_tools(write_scope: tuple[str, ...],
                allowed_new_files: tuple[str, ...]) -> tuple[str, ...]:
    """The tools this mission's scope can actually satisfy, in SPECS order.

    A pure function of deterministic mission state, so the schema (protocol A)
    and the manual (protocol J) can be built from it without either one holding
    a ToolContext.

    Read-only tools are never withheld: a mission always permits looking, and an
    agent that cannot look cannot decide. ``finish`` is never withheld either --
    removing the way out is how a loop turns a wrong turn into a hung run.
    """
    can_edit_existing = bool(write_scope)
    can_write_new = bool(write_scope) or bool(allowed_new_files)
    out = []
    for name in SPECS:
        if name in _NEEDS_EXISTING_WRITABLE and not can_edit_existing:
            continue
        if name in _NEEDS_ANY_WRITE and not can_write_new:
            continue
        out.append(name)
    return tuple(out)


def native_schema(only: tuple[str, ...] | None = None) -> list[dict]:
    """The provider-side tool schema, generated from SPECS so it cannot drift.

    ``only`` restricts which tools are DECLARED to the model, without touching
    what ``dispatch`` can execute. It exists for the frozen screen (F-20):
    that instrument's contract names seven tools, and once list_dir and run
    were added it would otherwise have started declaring nine -- measuring
    something other than what its own prompt describes, and producing numbers
    that could not be compared with the runs already on disk. A tool outside
    the declared set is an ordinary ERROR_UNKNOWN_TOOL, which is precisely how
    it behaved before those tools existed.

    Both the shape and the prose come from module-level dicts that asserts tie
    to SPECS, so the schema, the argument validation and the documentation are
    one source of truth (INVARIANTS I3). The previous runner advertised a
    ``list_dir`` it did not implement and a ``run`` when the contract said
    ``run_tests``; neither could happen here without failing at import.
    """
    types = {
        "path": {"type": "string"},
        "recursive": {"type": "boolean"},
        "start": {"type": ["integer", "null"]},
        "end": {"type": ["integer", "null"]},
        "pattern": {"type": "string"},
        "query": {"type": "string"},
        "limit": {"type": ["integer", "null"]},
        "glob": {"type": "string"},
        "context": {"type": ["integer", "null"]},
        "ignore_case": {"type": "boolean"},
        "old": {"type": "string"},
        "new": {"type": "string"},
        "content": {"type": "string"},
    "src": {"type": "string"},
    "into": {"type": "string"},
        "name": {"type": "string"},
        "argv": {"type": "array", "items": {"type": "string"}},
        "timeout": {"type": ["number", "null"]},
        "node_ids": {"type": ["array", "null"], "items": {"type": "string"}},
        "summary": {"type": "string"},
        "status": {"type": "string", "enum": list(FINISH_STATUSES)},
    }
    if only is not None:
        unknown = [n for n in only if n not in SPECS]
        if unknown:
            # A caller asking for a tool that does not exist is the "schema
            # advertised list_dir with no implementation" defect coming back in
            # a new shape. Fail loudly rather than silently declare eight.
            raise HarnessInvalid(f"native_schema asked for unknown tools: {unknown}")
    out = []
    for name, (required, optional) in SPECS.items():
        if only is not None and name not in only:
            continue
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


def text_manual(only: tuple[str, ...] | None = None) -> str:
    """The tool documentation as prose, for protocols with no schema channel.

    Generated from SPECS, TOOL_DOC and PARAM_DOC -- the same three dicts that
    build native_schema, tied together by the asserts above. A hand-written
    manual for the text protocols would be a second source of truth for what a
    tool is, and it would drift the way the old runner's schema drifted from
    its implementation.
    """
    lines = ["HERRAMIENTAS DISPONIBLES", ""]
    for name, (required, optional) in SPECS.items():
        if only is not None and name not in only:
            continue
        signature = ", ".join(list(required) + [f"{o}=null" for o in optional])
        lines.append(f"{name}({signature})")
        lines.append(f"    {TOOL_DOC[name]}")
        for key in required + optional:
            flag = "" if key in required else " (opcional)"
            lines.append(f"      - {key}{flag}: {PARAM_DOC[key]}")
        lines.append("")
    return "\n".join(lines)
