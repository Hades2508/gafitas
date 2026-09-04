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

from . import (deps, engine, evidence, loop, orient, protocol as protocol_mod,
               telemetry_bridge, tools, verify, workspace)
from .errors import HarnessInvalid
from .provider import (  # noqa: F401  (WORK_* re-exported for the CLI)
    WORK_NUM_CTX,
    WORK_NUM_PREDICT,
    CodexProvider,
    OllamaProvider,
)
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
#: The ticket declared no acceptance tests, so nothing here can say whether
#: the work is right -- something outside decides (a benchmark harness, a
#: reviewer, CI). A real change was produced and was not shown to be
#: correct. Deliberately NOT a pass (F-57).
CANDIDATE = "CANDIDATE"
PROVIDER_ERROR = "PROVIDER_ERROR"          # infrastructure, never a model result
HARNESS_INVALID = "HARNESS_INVALID"        # our own defect


SYSTEM_PROMPT = """Eres un programador trabajando dentro de un repositorio real.

Trabajas llamando a herramientas, UNA POR TURNO, y esperando su resultado antes
de decidir la siguiente. Las descripciones de cada herramienta y de cada
argumento las tienes en su definicion; leelas.

COMO TRABAJAR

1. ORIENTATE. Tienes la estructura del repositorio en el objetivo: usala en
   vez de recorrer directorios. list_dir es para mirar un directorio
   concreto que no aparezca ahi. No adivines nombres de ficheros.
   Si NO SABES donde esta lo que buscas ni como se llama, usa search_code y
   pegale el texto del objetivo tal cual: busca por significado y te devuelve
   los sitios que mas se le parecen. grep es para cuando ya sabes el literal
   exacto que quieres encontrar; si le das una frase descriptiva no encontrara
   nada, porque compara texto, no ideas.
   En un fichero GRANDE no lo leas entero: list_symbols te dice que hay dentro
   y read_symbol te da una funcion o clase concreta con sus lineas. Leer el
   mismo fichero una y otra vez gasta turnos y no averigua nada nuevo.
2. LEE ANTES DE EDITAR. edit necesita el texto EXACTO que hay ahora, con su
   indentacion. Leelo con read_file y copialo.
   Si el codigo que tienes que producir YA EXISTE en el repositorio, no lo
   reescribas a mano: copy_code lo mueve tal cual, byte a byte, y no se pierden
   sangrados ni comillas. Reescribir a mano codigo largo es la forma mas comun
   de estropearlo.
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

  finish(status="DONE", summary=...)       SOLO tras ejecutar run_tests y
                                           ver que pasan. Si no lo has hecho,
                                           te lo recordare.
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
    protocol: str = "A"
    #: The EngineCapabilities this run was planned against (context, output
    #: budget, protocol). Sealed so a comparison between engines can be checked
    #: rather than trusted.
    engine: dict = field(default_factory=dict)
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
    provider_retries: int = 0
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
            "protocol": self.protocol,
            "engine": dict(self.engine),
            "outcome": self.outcome, "loop_outcome": self.loop_outcome,
            "scoreable": self.scoreable,
            "discrimination": self.discrimination,
            "conscience": self.conscience,
            "agent_report": self.agent_report,
            "turns_used": self.turns_used, "invalid_calls": self.invalid_calls,
            "tool_errors": self.tool_errors, "loops": self.loops,
            "max_repeat": self.max_repeat,
            "provider_retries": self.provider_retries,
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


def system_prompt(ticket: Ticket, protocol: str = "A") -> str:
    """The system prompt, plus whatever the protocol needs to be usable.

    Protocol A gets the tool schema through the provider's own channel and needs
    nothing extra. Protocol J has no such channel, so the manual and the call
    format travel in the prompt -- both generated from the same dicts that build
    the schema, so the two tiers cannot disagree about what a tool is.
    """
    text = SYSTEM_PROMPT.format(
        write_scope=", ".join(ticket.scope.to_list()) or "(nada)",
        max_turns=ticket.max_turns,
    )
    if protocol == "J":
        # The same surface protocol A gets through the schema channel. If the
        # manual advertised a tool the schema withholds, the two tiers would
        # disagree about what this mission permits -- and protocol J is exactly
        # where the small engines run.
        legal = tools.legal_tools(tuple(ticket.write_scope),
                                  tuple(ticket.allowed_new_files))
        text += "\n\n" + tools.text_manual(legal) + "\n" + protocol_mod.JSON_INSTRUCTIONS
    return text


def objective_text(ticket: Ticket, root: Path | None = None) -> str:
    """The mission, plus the repository's shape.

    The map is handed over rather than left to the agent because walking a
    directory needs no reasoning and cost three inferences. Measured over fifty
    matched runs of the reference engine: list_dir was called 140 times -- on
    turn one in 49 of 50 runs and still on turn three in 24 of them, on missions
    that NAME the file -- and the first write did not land until turn 7.

    ``root`` is optional so a caller with no repository on disk gets the
    objective exactly as it was before.
    """
    parts = [ticket.objective, ""]
    if root is not None:
        picture = orient.repo_map(Path(root))
        if picture:
            parts += [picture, ""]
        opening = orient.opening_candidates(Path(root), ticket.objective)
        if opening:
            parts += [opening, ""]
    parts += [
        f"Puedes escribir en: {', '.join(ticket.scope.to_list())}",
        f"Tests de aceptacion: {', '.join(ticket.acceptance_tests)}",
        "(un ambito que acaba en '/' incluye todo lo que hay debajo)",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------- run


def run_ticket(
    ticket: Ticket,
    model: str,
    *,
    provider_factory: Callable[[str], Any] | None = None,
    out_dir: Path | None = None,
    telemetry: Any | None = None,
    base_dir: Path | None = None,
    num_ctx: int | None = None,
    protocol: str | None = None,
    capabilities: Any | None = None,
) -> WorkResult:
    """Do the ticket. One model, one workspace, one verdict.

    ``num_ctx`` and ``protocol`` come FROM THE ENGINE unless a caller overrides
    them. Before this they were module constants -- 32768 and native tool calls
    -- which are the reference engine's properties wearing the costume of
    universal truth. Point the harness at a model with an 8k window and it did
    not adapt, it overflowed, and the overflow was recorded as a PROVIDER_ERROR:
    the harness reporting its own misconfiguration as the engine's fault. That
    makes engine comparison impossible, which is precisely what we want to do.

    An unregistered engine still runs. It gets a conservative context and no
    assumptions, which is the honest treatment of a model nobody has certified.
    """
    caps = capabilities or engine.capabilities_for(model)
    if num_ctx is None:
        num_ctx = caps.working_context()
    if protocol is None:
        try:
            protocol = caps.protocol
        except ValueError:
            # Nobody established how to drive it. Native first is the same
            # choice the code made before, but now it is a recorded fallback
            # rather than an invisible assumption.
            protocol = "A"
    num_predict = caps.working_output()

    result = WorkResult(ticket_id=ticket.ticket_id, model=model, protocol=protocol)
    result.engine = caps.to_dict()
    factory = provider_factory or (
        lambda name: OllamaProvider(name, num_ctx=num_ctx, num_predict=num_predict)
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
        # F-57: with no declared acceptance, check_pre would run pytest with no
        # node ids -- the WHOLE repository suite, treated as the acceptance. On
        # a project whose suite is green at the base commit that returns
        # NON_DISCRIMINATING and the model is never invoked at all. There is
        # nothing to gate on here, so the gate does not run.
        judged_here = bool(ticket.acceptance_tests)
        pre_disc = (
            verify.check_pre(box.path, ticket.acceptance_tests)
            if judged_here
            else verify.Discrimination(
                verify.UNKNOWN,
                verify.SuiteResult(False, None, False, "(no acceptance declared)", 0.0),
                None,
                "el ticket no declara tests de aceptacion: lo juzga algo externo.",
            )
        )
        pre_full = (
            verify.run_suite(box.path) if ticket.full_suite
            else verify.SuiteResult(False, None, False, "(full_suite disabled)", 0.0)
        )
        pre_surface = verify.snapshot_surfaces(box.path, _python_files(box.path))
        pre_files = _tracked_files(box.path)

        # ---------------- the gate. Before spending a single token. -------
        if judged_here and not pre_disc.measurable:
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
            # F-38: hand the agent the same baseline the conscience will use,
            # so "you broke something" can be said during the run rather than
            # discovered afterwards and charged to it.
            baseline_outcomes=dict(pre_full.outcomes) if ticket.full_suite else {},
            full_suite_runner=(
                (lambda root: verify.run_suite(root).outcomes) if ticket.full_suite else None
            ),
        )
        telemetry.start(ticket.ticket_id, repo=str(ticket.repo), objective=ticket.objective)
        provider = factory(model)
        result.model_class = str(provider.describe().get("model_class", "LOCAL"))

        outcome = loop.run_loop(
            provider=provider, ctx=ctx,
            system=system_prompt(ticket, protocol),
            objective=objective_text(ticket, ctx.root),
            protocol_name=protocol, max_turns=ticket.max_turns,
            keep_turns=WORK_KEEP_TURNS, elide_over_chars=WORK_ELIDE_OVER_CHARS,
            budget_chars=budget_chars(
                num_ctx,
                num_predict=WORK_NUM_PREDICT,
                schema_chars=len(json.dumps(tools.native_schema(
                    tools.legal_tools(tuple(ticket.write_scope),
                                      tuple(ticket.allowed_new_files)))))
                if protocol == "A" else 0,
                system_chars=len(system_prompt(ticket, protocol))
                + len(objective_text(ticket, ctx.root)),
            ),
        )

        result.loop_outcome = outcome.outcome
        result.turns_used = outcome.turns_used
        result.invalid_calls = outcome.invalid_calls
        result.tool_errors = outcome.tool_errors
        result.loops = outcome.loops
        result.max_repeat = outcome.max_repeat
        result.provider_retries = outcome.provider_retries
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
        if outcome.outcome in (loop.PROVIDER_ERROR, loop.HARNESS_INVALID):
            # F-31: record what the agent actually did before abandoning the
            # run. This used to return immediately, so a run that made five
            # edits before the server rejected one malformed tool call sealed a
            # record saying it had changed nothing -- a failure destroying its
            # own evidence on the way out, which experiments.md forbids.
            # Whatever those edits were, somebody may want to look at them.
            aborted = _changed_since(pre_files, _tracked_files(box.path))
            result.changed_files = sorted(aborted)
            result.unauthorised_writes = sorted(
                rel for rel in aborted if not ticket.scope.allows(rel, creating=True)
            )
            result.commands_run = list(ctx.commands_run)
            result.outcome = (
                PROVIDER_ERROR if outcome.outcome == loop.PROVIDER_ERROR else HARNESS_INVALID
            )
            result.scoreable = False
            result.discrimination = pre_disc.to_dict()
            result.notes.append(
                f"run abandonada tras {outcome.turns_used} turnos; "
                f"{len(aborted)} ficheros ya modificados quedan en el workspace preservado."
            )
            result.wall_seconds = time.perf_counter() - started
            result.patch_path = _seal(
                out_dir, ticket, result, outcome.events, _make_patch(box.path, aborted)
            )
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
        post_disc = (
            verify.check_post(pre_disc, box.path, ticket.acceptance_tests)
            if judged_here
            else pre_disc
        )
        post_full = (
            verify.run_suite(box.path) if ticket.full_suite
            else verify.SuiteResult(False, None, False, "(full_suite disabled)", 0.0)
        )
        post_surface = verify.snapshot_surfaces(box.path, _python_files(box.path))
        result.discrimination = post_disc.to_dict()

        # ---------------- the conscience ----------------------------------
        # Only deterministic evidence about the CODE goes in here. What the
        # agent believes about its own work is reported separately (F-24).
        signals = [
            verify.signal_acceptance_untouched(ticket.acceptance_tests, changed),
            verify.signal_scope_respected(changed, ticket.scope),
            verify.signal_public_surface(pre_surface, post_surface),
        ]
        if judged_here:
            # Without a declared acceptance there is nothing to discriminate,
            # and a signal that always returns INCONCLUSIVE would block every
            # externally-judged ticket for the crime of being one.
            signals.insert(0, verify.signal_discrimination(post_disc))
        conscience = verify.Conscience(signals)
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
        if not judged_here:
            # F-57. Nothing here has shown the work is correct. The signals that
            # did run only show it did not obviously break anything, and calling
            # that a pass would be "tests green = success" with the tests
            # missing too.
            result.outcome = CANDIDATE if changed else FAIL
            result.notes.append(
                "sin tests de aceptacion declarados: GAFITAS entrega un CANDIDATO "
                "y NO afirma que sea correcto. Lo juzga el evaluador externo."
                if changed else
                "sin tests de aceptacion declarados y sin ningun cambio: no hay "
                "candidato que entregar."
            )
        elif post_disc.status != verify.DISCRIMINATED:
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
        result.patch_path = _seal(out_dir, ticket, result, outcome.events, patch,
                                  payloads=getattr(outcome, "payloads", None))
        preserve = result.outcome not in (PASS, PASS_UNCONFIRMED, CANDIDATE)
        return result
    finally:
        # A1: a preserved box records WHAT it is, so a retention policy can tell
        # an ordinary failure from evidence that must never be deleted. Anything
        # our own defect produced is pinned and exempt from every budget.
        box.dispose(preserve=preserve, ticket=ticket.ticket_id,
                    outcome=result.outcome,
                    pinned=result.outcome in (HARNESS_INVALID, PROVIDER_ERROR))
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
          events: list, patch: str,
          payloads: dict[str, str] | None = None) -> str | None:
    if out_dir is None:
        return None
    target = Path(out_dir) / "tickets"
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{ticket.ticket_id}_{result.model.replace('/', '_').replace(':', '_')}"
    # F-49: since F-48 a tier may be retried, and every attempt used to seal to
    # the same filename -- so three local attempts left one file and the first
    # two vanished. Losing the evidence of the attempts that FAILED is exactly
    # backwards: those are the ones worth reading, and experiments.md forbids a
    # failure destroying its own record.
    if (target / f"{stem}.json").exists():
        attempt = 2
        while (target / f"{stem}__try{attempt}.json").exists():
            attempt += 1
        stem = f"{stem}__try{attempt}"
    (target / f"{stem}.json").write_text(
        json.dumps(
            {"provenance": deps.provenance(), "record": result.to_dict(), "events": events},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # A2: any tool argument too long for its event is kept whole here, keyed by
    # the sha256 the event records. One file per DISTINCT payload, so a body
    # written twice costs nothing the second time, and what actually reached the
    # disk stays readable long after the workspace is gone.
    if payloads:
        store = target / "payloads"
        store.mkdir(parents=True, exist_ok=True)
        for digest, text in payloads.items():
            body = store / f"{digest}.txt"
            if not body.exists():
                body.write_text(text, encoding="utf-8")

    patch_path = None
    if patch and patch != "(sin cambios)":
        patch_file = target / f"{stem}.patch"
        patch_file.write_text(patch, encoding="utf-8")
        patch_path = str(patch_file)
    return patch_path
