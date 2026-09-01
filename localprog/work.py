"""The productive path: hand it a ticket, get back a reviewed change.

Everything else in this package is a measuring instrument. ``screen`` asks
whether a model can chain five tool calls; ``step1`` runs a frozen corpus
against a frozen gate. Both are Gate-A apparatus, both are deliberately
unchangeable, and neither can be handed a piece of real work (F-11). This
module is the other thing -- the one that exists to get something done.

The pipeline, and why each stage is where it is:

    open the box              an isolated worktree; the source repo is never touched
    PRE characterisation      what the tree does BEFORE anyone edits it
    discrimination gate       is this task even unsolved? if not, STOP HERE
    the loop                  the model programs, with real budget
    write verification        did anything change that no tool authorised?
    POST characterisation     what the tree does now
    the conscience            refuse-only signals over PRE vs POST
    verdict + evidence        sealed, with a patch a human can read

The discrimination gate sits before the model on purpose, and that ordering is
the single most valuable line in the file. A mission whose acceptance already
passes cannot distinguish success from doing nothing, so running the model
against it burns time to produce a number that means nothing. All five STEP1
missions were in that state, which is how a report came to read 5/5 while every
run in it was BUDGET_EXHAUSTED. Checking first costs one pytest invocation and
makes that class of result impossible rather than merely detectable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import deps, evidence, loop, telemetry_bridge, tools, verify, workspace
from .errors import HarnessInvalid
from .provider import WORK_NUM_CTX, WORK_NUM_PREDICT, OllamaProvider  # noqa: F401  (re-exported for the CLI)
from .scope import WriteScope
from .transcript import WORK_ELIDE_OVER_CHARS, WORK_KEEP_TURNS, budget_chars

SCHEMA = "GAFITAS_WORK_V1"

REQUIRED = ("ticket_id", "repo", "objective", "write_scope", "acceptance_tests")

# Outcomes of a whole ticket. Distinct from loop outcomes: the loop reports how
# the conversation ended, this reports what the work is worth.
PASS = "PASS"                              # did the job, conscience clear, agent confirmed
#: Evidence is clean but the agent did not confirm -- it declared BLOCKED, or
#: the budget ran out before it could call finish. Real, usable work with a
#: flag on it, NOT a failure (F-24). ga06 is the case that earned this.
PASS_UNCONFIRMED = "PASS_UNCONFIRMED"
BLOCKED_BY_CONSCIENCE = "BLOCKED_BY_CONSCIENCE"   # suite green, a signal refused it
FAIL = "FAIL"                              # honest miss: PRE failed, POST still fails
NON_DISCRIMINATING = "NON_DISCRIMINATING"  # the ticket was not a task
PROVIDER_ERROR = "PROVIDER_ERROR"          # infrastructure, never a model result
HARNESS_INVALID = "HARNESS_INVALID"        # our own defect


SYSTEM_PROMPT = """Eres un programador trabajando dentro de un repositorio real.

Trabajas llamando a herramientas, UNA POR TURNO, y esperando su resultado antes
de decidir la siguiente. Las descripciones de cada herramienta y de cada
argumento las tienes en su definicion; leelas.

COMO TRABAJAR

1. ORIENTATE. Si no conoces el repositorio, empieza por list_dir y grep. No
   adivines nombres de ficheros.
2. LEE ANTES DE EDITAR. edit necesita el texto EXACTO que hay ahora, con su
   indentacion. Leelo con read_file y copialo.
3. EJECUTA PARA VER. run te deja reproducir un fallo, imprimir un valor o
   comprobar un import. Cuando algo no funcione como esperas, MIRA lo que pasa
   en vez de suponerlo.
4. LEE LOS ERRORES. Un fallo de test te dice que arreglar. Un error de
   herramienta te dice como llamarla bien. Los dos son informacion util, no
   castigos: cuando recibas uno, corrige y sigue.
5. CAMBIA DE PLAN SI HACE FALTA. Si dos intentos parecidos fallan igual, el
   problema es tu diagnostico, no tu redaccion. Vuelve a leer el codigo.
6. NO ROMPAS LO QUE NO TE PIDEN. Otros tests, que no ves, dependen de este
   codigo. No borres funciones publicas ni cambies firmas si no es lo que se
   pide.
7. NO TOQUES LOS TESTS DE ACEPTACION. Son quien te juzga.

CUANDO TERMINAR

  finish(status="DONE", summary=...)       cuando run_tests pase.
  finish(status="NO_CHANGE", summary=...)  si compruebas que no hay nada que hacer.
  finish(status="BLOCKED", summary=...)    si no puedes continuar; explica que te
                                           lo impide. Decirlo es correcto y util.

LIMITES

  Puedes LEER cualquier fichero del repositorio.
  Solo puedes ESCRIBIR en: {write_scope}
  Tienes {max_turns} turnos.
"""


# ------------------------------------------------------------------- ticket


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    repo: Path
    objective: str
    write_scope: tuple[str, ...]
    acceptance_tests: tuple[str, ...]
    allowed_new_files: tuple[str, ...] = ()
    commit: str | None = None
    #: Whether to run the whole suite before and after for the collateral
    #: regression check. Default on: it is the only signal that catches
    #: "satisfied the acceptance sample, broke the repository", and it needs no
    #: hand-written probe. Turn it off for a repository whose suite is too slow
    #: to run twice, and accept the weaker verdict that follows.
    full_suite: bool = True
    max_turns: int = loop.WORK_MAX_TURNS

    @property
    def scope(self) -> WriteScope:
        return WriteScope(self.write_scope, self.allowed_new_files)


def load_ticket(path: Path) -> Ticket:
    """Read a ticket, or say exactly what is wrong with it.

    A malformed ticket is a defect in the TICKET and is raised as
    HarnessInvalid, never recorded as a model failure. That distinction voided
    four subtasks of the multifile campaign once already: an empty acceptance
    list produced INVALID_SPEC, the mission was scored as a failure, and only a
    later investigation established that no model had ever been called.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarnessInvalid(f"cannot read ticket {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HarnessInvalid(f"ticket {path} is not a JSON object")
    missing = [f for f in REQUIRED if not payload.get(f)]
    if missing:
        raise HarnessInvalid(f"ticket {path} is missing/empty: {missing}")

    repo = Path(payload["repo"])

    def relatives(key: str) -> tuple[str, ...]:
        raw = payload.get(key) or []
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise HarnessInvalid(f"ticket {path}: {key} must be a list of strings")
        out = []
        for item in raw:
            candidate = Path(item)
            if candidate.is_absolute():
                try:
                    item = candidate.relative_to(repo).as_posix()
                except ValueError as exc:
                    raise HarnessInvalid(f"ticket {path}: {item!r} is outside repo") from exc
            out.append(item.replace("\\", "/"))
        return tuple(out)

    return Ticket(
        ticket_id=str(payload["ticket_id"]),
        repo=repo,
        objective=str(payload["objective"]),
        write_scope=relatives("write_scope"),
        acceptance_tests=relatives("acceptance_tests"),
        allowed_new_files=relatives("allowed_new_files"),
        commit=payload.get("commit"),
        full_suite=bool(payload.get("full_suite", True)),
        max_turns=int(payload.get("max_turns", loop.WORK_MAX_TURNS)),
    )


# ------------------------------------------------------------------- result


@dataclass
class WorkResult:
    ticket_id: str
    model: str
    model_class: str = "LOCAL"
    outcome: str = ""
    loop_outcome: str = ""
    scoreable: bool = False
    discrimination: dict = field(default_factory=dict)
    conscience: dict = field(default_factory=dict)
    #: The agent's own account of how it ended. Reported, never allowed to
    #: decide anything on its own (F-24).
    agent_report: dict = field(default_factory=dict)
    turns_used: int = 0
    invalid_calls: int = 0
    tool_errors: int = 0
    loops: int = 0
    max_repeat: int = 0
    tools_used: dict = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    unauthorised_writes: list[str] = field(default_factory=list)
    commands_run: list[dict] = field(default_factory=list)
    finish_status: str | None = None
    finish_summary: str = ""
    usage: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    workspace: dict = field(default_factory=dict)
    patch_path: str | None = None
    provider_error: dict | None = None
    harness_invalid: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.ticket_id}/{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "ticket_id": self.ticket_id, "model": self.model,
            "model_class": self.model_class,
            "outcome": self.outcome, "loop_outcome": self.loop_outcome,
            "scoreable": self.scoreable,
            "discrimination": self.discrimination,
            "conscience": self.conscience,
            "agent_report": self.agent_report,
            "turns_used": self.turns_used, "invalid_calls": self.invalid_calls,
            "tool_errors": self.tool_errors, "loops": self.loops,
            "max_repeat": self.max_repeat,
            "tools_used": self.tools_used,
            "changed_files": self.changed_files,
            "unauthorised_writes": self.unauthorised_writes,
            "commands_run": self.commands_run,
            "finish_status": self.finish_status, "finish_summary": self.finish_summary,
            "usage": self.usage,
            "wall_seconds": round(self.wall_seconds, 3),
            "workspace": self.workspace, "patch": self.patch_path,
            "provider_error": self.provider_error,
            "harness_invalid": self.harness_invalid,
            "notes": self.notes,
        }


# ------------------------------------------------------------------- helpers


def _tracked_files(root: Path) -> dict[str, float]:
    """Path -> (size, mtime) fingerprint of every text-ish file in the tree.

    Used to detect writes no tool performed. Cheap on purpose: a hash of every
    file would be more precise and would also make the check expensive enough
    that somebody would eventually turn it off.
    """
    out: dict[str, float] = {}
    for path in root.rglob("*"):
        if any(part in verify.SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.is_file():
                stat = path.stat()
                out[path.relative_to(root).as_posix()] = stat.st_size + stat.st_mtime
        except OSError:
            continue
    return out


def _changed_since(before: dict[str, float], after: dict[str, float]) -> set[str]:
    return {rel for rel, mark in after.items() if before.get(rel) != mark} | (
        set(before) - set(after)
    )


def _python_files(root: Path) -> list[str]:
    return [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if not any(part in verify.SKIP_DIRS for part in p.parts)
    ]


def system_prompt(ticket: Ticket) -> str:
    return SYSTEM_PROMPT.format(
        write_scope=", ".join(ticket.scope.to_list()) or "(nada)",
        max_turns=ticket.max_turns,
    )


def objective_text(ticket: Ticket) -> str:
    return (
        f"{ticket.objective}\n\n"
        f"Puedes escribir en: {', '.join(ticket.scope.to_list())}\n"
        f"Tests de aceptacion: {', '.join(ticket.acceptance_tests)}\n"
        f"(un ambito que acaba en '/' incluye todo lo que hay debajo)"
    )


# --------------------------------------------------------------------- run


def run_ticket(
    ticket: Ticket,
    model: str,
    *,
    provider_factory: Callable[[str], Any] | None = None,
    out_dir: Path | None = None,
    telemetry: Any | None = None,
    base_dir: Path | None = None,
    num_ctx: int = WORK_NUM_CTX,
) -> WorkResult:
    """Do the ticket. One model, one workspace, one verdict."""
    result = WorkResult(ticket_id=ticket.ticket_id, model=model)
    factory = provider_factory or (
        lambda name: OllamaProvider(name, num_ctx=num_ctx, num_predict=WORK_NUM_PREDICT)
    )
    telemetry = telemetry or telemetry_bridge.NullTelemetry("no telemetry supplied")
    started = time.perf_counter()

    box = workspace.open_repo(
        ticket.repo, commit=ticket.commit,
        prefix=f"gafitas_{ticket.ticket_id}_", base_dir=base_dir,
    )
    result.workspace = box.describe()
    preserve = True
    try:
        # ---------------- PRE: what does this tree do right now? ----------
        pre_disc = verify.check_pre(box.path, ticket.acceptance_tests)
        pre_full = (
            verify.run_suite(box.path) if ticket.full_suite
            else verify.SuiteResult(False, None, False, "(full_suite disabled)", 0.0)
        )
        pre_surface = verify.snapshot_surfaces(box.path, _python_files(box.path))
        pre_files = _tracked_files(box.path)

        # ---------------- the gate. Before spending a single token. -------
        if not pre_disc.measurable:
            result.outcome = NON_DISCRIMINATING
            result.loop_outcome = "(not run)"
            result.scoreable = False
            result.discrimination = pre_disc.to_dict()
            result.notes.append(
                "el modelo NO fue invocado: la aceptacion ya pasaba antes de empezar, "
                "asi que este ticket no puede medir nada."
            )
            result.wall_seconds = time.perf_counter() - started
            _seal(out_dir, ticket, result, events=[], patch="")
            return result

        # ---------------- the loop ----------------------------------------
        ctx = tools.ToolContext(
            root=box.path,
            write_scope=ticket.write_scope,
            allowed_new_files=ticket.allowed_new_files,
            acceptance_tests=ticket.acceptance_tests,
        )
        telemetry.start(ticket.ticket_id, repo=str(ticket.repo), objective=ticket.objective)
        provider = factory(model)
        result.model_class = str(provider.describe().get("model_class", "LOCAL"))

        outcome = loop.run_loop(
            provider=provider, ctx=ctx,
            system=system_prompt(ticket), objective=objective_text(ticket),
            protocol_name="A", max_turns=ticket.max_turns,
            keep_turns=WORK_KEEP_TURNS, elide_over_chars=WORK_ELIDE_OVER_CHARS,
            budget_chars=budget_chars(num_ctx),
        )

        result.loop_outcome = outcome.outcome
        result.turns_used = outcome.turns_used
        result.invalid_calls = outcome.invalid_calls
        result.tool_errors = outcome.tool_errors
        result.loops = outcome.loops
        result.max_repeat = outcome.max_repeat
        result.tools_used = outcome.tools_used
        result.usage = dict(outcome.usage)
        result.commands_run = list(ctx.commands_run)
        result.finish_status = ctx.finish_status
        result.finish_summary = ctx.finish_summary
        result.provider_error = outcome.provider_error
        result.harness_invalid = outcome.harness_invalid

        if outcome.outcome == loop.STALLED:
            result.notes.append(
                f"el agente dejo de emitir llamadas en el turno {outcome.stalled_after} "
                f"y no se recupero; la run se corto en vez de gastar el resto del "
                f"presupuesto repitiendo el mismo fallo."
            )
        if outcome.outcome == loop.PROVIDER_ERROR:
            result.outcome = PROVIDER_ERROR
            result.scoreable = False
            result.discrimination = pre_disc.to_dict()
            result.wall_seconds = time.perf_counter() - started
            _seal(out_dir, ticket, result, outcome.events, "")
            return result
        if outcome.outcome == loop.HARNESS_INVALID:
            result.outcome = HARNESS_INVALID
            result.scoreable = False
            result.discrimination = pre_disc.to_dict()
            result.wall_seconds = time.perf_counter() - started
            _seal(out_dir, ticket, result, outcome.events, "")
            return result

        # ---------------- what actually changed on disk -------------------
        # Not what the tools believe they changed. ``run`` can execute
        # python -c, so the only trustworthy answer comes from the filesystem.
        post_files = _tracked_files(box.path)
        changed = _changed_since(pre_files, post_files)
        result.changed_files = sorted(changed)
        result.unauthorised_writes = sorted(
            rel for rel in changed if not ticket.scope.allows(rel, creating=True)
        )

        # ---------------- POST: what does it do now? ----------------------
        post_disc = verify.check_post(pre_disc, box.path, ticket.acceptance_tests)
        post_full = (
            verify.run_suite(box.path) if ticket.full_suite
            else verify.SuiteResult(False, None, False, "(full_suite disabled)", 0.0)
        )
        post_surface = verify.snapshot_surfaces(box.path, _python_files(box.path))
        result.discrimination = post_disc.to_dict()

        # ---------------- the conscience ----------------------------------
        # Only deterministic evidence about the CODE goes in here. What the
        # agent believes about its own work is reported separately (F-24).
        conscience = verify.Conscience([
            verify.signal_discrimination(post_disc),
            verify.signal_acceptance_untouched(ticket.acceptance_tests, changed),
            verify.signal_scope_respected(changed, ticket.scope),
            verify.signal_public_surface(pre_surface, post_surface),
        ])
        agent_signal = verify.signal_agent_reported(ctx.finish_status, ctx.finish_summary)
        result.agent_report = agent_signal.to_dict()
        if ticket.full_suite:
            conscience.signals.append(
                verify.signal_no_collateral_regression(
                    pre_full, post_full, ticket.acceptance_tests
                )
            )
        result.conscience = conscience.to_dict()

        # ---------------- the verdict -------------------------------------
        # I9: taken here, by the harness, after the loop. The agent's own claim
        # is one signal among several and cannot carry the decision.
        result.scoreable = True
        confirmed = ctx.finish_status in ("DONE", "NO_CHANGE")
        if post_disc.status != verify.DISCRIMINATED:
            result.outcome = FAIL
        elif conscience.verdict != verify.PASS:
            result.outcome = BLOCKED_BY_CONSCIENCE
            result.notes += [f"{s.name}: {s.detail}" for s in conscience.blocking]
        elif confirmed:
            result.outcome = PASS
        else:
            result.outcome = PASS_UNCONFIRMED
            result.notes.append(
                f"la evidencia es limpia pero el agente no lo confirmo "
                f"({agent_signal.detail}). El cambio sirve; conviene una lectura humana."
            )

        telemetry.attempt(ticket.ticket_id, latency_ms=outcome.wall_seconds * 1000)
        telemetry.finish(ticket.ticket_id, outcome=result.outcome)
        result.notes += list(getattr(telemetry, "notes", []))

        patch = _make_patch(box.path, changed)
        result.wall_seconds = time.perf_counter() - started
        result.patch_path = _seal(out_dir, ticket, result, outcome.events, patch)
        preserve = result.outcome not in (PASS, PASS_UNCONFIRMED)
        return result
    finally:
        box.dispose(preserve=preserve)
        result.workspace = box.describe()


def _make_patch(root: Path, changed) -> str:
    """A unified diff of everything that changed, for a human to read.

    Built from the git worktree when there is one -- git already knows how to
    do this correctly, including renames and deletions -- and reconstructed
    file by file otherwise.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "diff", "--no-color", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=60, shell=False,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=str(root),
            capture_output=True, text=True, timeout=60, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "(no se pudo generar el patch)"

    parts = [proc.stdout] if proc.returncode == 0 else []
    for rel in (untracked.stdout or "").splitlines():
        rel = rel.strip()
        if not rel or any(part in verify.SKIP_DIRS for part in Path(rel).parts):
            continue
        try:
            body = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        parts.append(evidence.diff(rel, None, body))
    return "".join(parts) or "(sin cambios)"


def _seal(out_dir: Path | None, ticket: Ticket, result: WorkResult,
          events: list, patch: str) -> str | None:
    if out_dir is None:
        return None
    target = Path(out_dir) / "tickets"
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{ticket.ticket_id}_{result.model.replace('/', '_').replace(':', '_')}"
    (target / f"{stem}.json").write_text(
        json.dumps(
            {"provenance": deps.provenance(), "record": result.to_dict(), "events": events},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    patch_path = None
    if patch and patch != "(sin cambios)":
        patch_file = target / f"{stem}.patch"
        patch_file.write_text(patch, encoding="utf-8")
        patch_path = str(patch_file)
    return patch_path
