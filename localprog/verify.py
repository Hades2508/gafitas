"""Verification. What has to be true before a change may be called a PASS.

Two jobs, and they are different enough to be worth naming separately.

DISCRIMINATION (F-01) asks whether the question was a question at all. Running
the acceptance suite after the agent works tells you nothing unless you also
know it failed before. This is not a hypothetical worry: every one of the five
STEP1 mission repos was snapshotted in its SOLVED state, so the target file
already existed and the suite already passed. The report says 5/5 tester_pass.
Every run in it is BUDGET_EXHAUSTED, because the model spent twelve turns
finding nothing to do. That measurement is void, and nothing in the harness
could have noticed -- it only ever looked at the POST state.

CONSCIENCE (F-09) asks whether a green suite is telling the truth. A suite is
a sample of behaviour, and an agent optimising against it can satisfy the
sample while breaking everything around it. So a handful of deterministic
signals compare the before and after states directly.

THE RULE THAT MAKES THIS SAFE TO BUILD INCREMENTALLY
----------------------------------------------------
Every signal here is REFUSE-ONLY. It may downgrade a verdict; it may never
upgrade one. ``combine`` takes the worst verdict present and there is no code
path that turns a non-PASS into a PASS. That property is what lets the
conscience be grown one signal at a time without ever becoming a way to
manufacture a false green: the worst a wrong signal can do is block work that
was fine, which is visible and annoying, rather than approve work that was
broken, which is invisible and permanent.

The vocabulary is the Behavioral Oracle's, deliberately -- same four verdicts,
same severity order -- so the two can be folded together later without
translating between two sets of words that mean nearly but not quite the same.

    PASS          proved to have done the job without breaking anything checked
    REGRESSION    proved to have broken something
    INCONCLUSIVE  could not be proved either way
    INFRA_ERROR   the harness could not carry out the check
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PASS = "PASS"
REGRESSION = "REGRESSION"
INCONCLUSIVE = "INCONCLUSIVE"
INFRA_ERROR = "INFRA_ERROR"

_SEVERITY = {PASS: 0, INCONCLUSIVE: 1, REGRESSION: 2, INFRA_ERROR: 3}

#: Discrimination outcomes. Kept apart from the verdicts because "this mission
#: is broken" and "this change is broken" are different findings with different
#: owners, and merging them is how a corpus defect gets recorded as a model
#: failure.
DISCRIMINATED = "DISCRIMINATED"              # PRE failed, POST passed: real signal
NOT_SOLVED = "NOT_SOLVED"                    # PRE failed, POST failed: honest miss
NON_DISCRIMINATING = "NON_DISCRIMINATING"    # PRE already passed: void as a measurement
BROKE_IT = "BROKE_IT"                        # PRE passed, POST fails: a regression
UNKNOWN = "UNKNOWN"                          # could not be established

PYTEST_TIMEOUT = 300.0
OUTPUT_TAIL = 4000
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}


def combine(verdicts) -> str:
    """The worst verdict present. The only combinator, and it never upgrades."""
    verdicts = [v for v in verdicts if v in _SEVERITY]
    if not verdicts:
        return INCONCLUSIVE  # nothing was checked, so nothing was proved
    return max(verdicts, key=lambda v: _SEVERITY[v])


# --------------------------------------------------------------- test running


@dataclass(frozen=True)
class SuiteResult:
    """One pytest invocation against one state of the tree."""

    passed: bool
    exit_code: int | None
    timed_out: bool
    output: str
    seconds: float
    node_ids: tuple[str, ...] = ()
    #: Per-test outcomes, when we asked for them. The map is what makes a
    #: whole-suite comparison possible: an aggregate pass/fail cannot tell
    #: "fixed three, broke one" apart from "fixed two".
    outcomes: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether this result can support a conclusion.

        pytest exit code 5 is "no tests collected" and 2/3/4 are internal or
        usage errors. None of them mean "the code is wrong", and treating them
        as failures is how an infrastructure problem gets recorded as a model
        one -- the same confusion the four-class error taxonomy exists to stop.
        """
        if self.timed_out:
            return False
        if self.exit_code in (0, 1):
            return True
        # 2 is "interrupted". Almost always a collection error, which is the
        # code under test failing to import -- a real failure of the code, and
        # for a create-this-module task the very failure we need PRE to show.
        # It is only unusable when pytest reported no per-file errors either,
        # which means something stopped the run rather than the code being bad.
        return self.exit_code == 2 and any(v == "error" for v in self.outcomes.values())

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "exit_code": self.exit_code,
            "timed_out": self.timed_out, "seconds": round(self.seconds, 2),
            "node_ids": list(self.node_ids), "usable": self.usable,
            "counts": _counts(self.outcomes),
            "output_tail": self.output[-1200:],
        }


def _counts(outcomes: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for status in outcomes.values():
        out[status] = out.get(status, 0) + 1
    return out


def _parse_report(text: str) -> dict[str, str]:
    """Per-test outcomes from ``-rA`` short-summary lines.

    Parsed from pytest's own text rather than a plugin, because requiring
    pytest-json-report would make verification depend on what happens to be
    installed in the workspace -- and the workspace is a checkout of somebody
    else's repository, so that dependency would fail exactly when it matters.
    """
    outcomes: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        for prefix, status in (
            ("PASSED ", "passed"), ("FAILED ", "failed"), ("ERROR ", "error"),
            ("XFAIL ", "xfail"), ("XPASS ", "xpass"), ("SKIPPED ", "skipped"),
        ):
            if line.startswith(prefix):
                node = line[len(prefix):].split(" - ")[0].strip()
                if node:
                    outcomes[node] = status
                break
    return outcomes


def run_suite(
    root: Path,
    node_ids: tuple[str, ...] = (),
    *,
    timeout: float = PYTEST_TIMEOUT,
    detailed: bool = True,
) -> SuiteResult:
    """Run pytest in *root* and report what happened. Never raises."""
    argv = [
        sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        # Without this, one un-importable test module aborts the entire run
        # ("Interrupted: 1 error during collection") and pytest exits 2, so the
        # PRE snapshot of a "create this module" task collects nothing at all
        # and there is no baseline left to compare POST against. With it, every
        # module that CAN be collected still runs, the exit code stays in the
        # ordinary 0/1 range, and the un-importable one is reported as an ERROR
        # outcome -- which is exactly what a missing module should look like.
        "--continue-on-collection-errors",
    ]
    if detailed:
        argv += ["-rA", "--tb=line"]
    argv += list(node_ids)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(root), capture_output=True, text=True,
            timeout=timeout, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        head = exc.stdout if isinstance(exc.stdout, str) else ""
        tail = exc.stderr if isinstance(exc.stderr, str) else ""
        return SuiteResult(
            passed=False, exit_code=None, timed_out=True,
            output=(head + tail)[-OUTPUT_TAIL:], seconds=time.perf_counter() - started,
            node_ids=tuple(node_ids),
        )
    except OSError as exc:
        return SuiteResult(
            passed=False, exit_code=None, timed_out=False,
            output=f"no se pudo ejecutar pytest: {exc}",
            seconds=time.perf_counter() - started, node_ids=tuple(node_ids),
        )
    text = proc.stdout + proc.stderr
    return SuiteResult(
        passed=proc.returncode == 0,
        exit_code=proc.returncode,
        timed_out=False,
        output=text[-OUTPUT_TAIL:],
        seconds=time.perf_counter() - started,
        node_ids=tuple(node_ids),
        outcomes=_parse_report(text) if detailed else {},
    )


# ------------------------------------------------------- discrimination (F-01)


@dataclass(frozen=True)
class Discrimination:
    status: str
    pre: SuiteResult
    post: SuiteResult | None
    detail: str

    @property
    def measurable(self) -> bool:
        """Whether a run against this mission can be scored at all.

        NON_DISCRIMINATING is the case that voided STEP1: the mission's own
        acceptance already passed before the agent was invited to do anything.
        Whatever the agent then did, the suite passing afterwards proves
        nothing, so the run is excluded rather than counted -- exactly as a
        PROVIDER_ERROR is excluded, and for the same reason.
        """
        return self.status not in (NON_DISCRIMINATING, UNKNOWN)

    def to_dict(self) -> dict:
        return {
            "status": self.status, "measurable": self.measurable, "detail": self.detail,
            "pre": self.pre.to_dict(), "post": self.post.to_dict() if self.post else None,
        }


def check_pre(root: Path, acceptance: tuple[str, ...], *, timeout: float = PYTEST_TIMEOUT) -> Discrimination:
    """Establish that the task is not already done, BEFORE the agent runs.

    Called on the pristine workspace. A mission whose acceptance is already
    green is not a hard mission or an easy one; it is not a mission.
    """
    pre = run_suite(root, acceptance, timeout=timeout)
    if not pre.usable:
        return Discrimination(
            UNKNOWN, pre, None,
            f"la aceptacion PRE no se pudo ejecutar (exit={pre.exit_code}, "
            f"timeout={pre.timed_out}). Sin PRE no hay medida.",
        )
    if pre.passed:
        return Discrimination(
            NON_DISCRIMINATING, pre, None,
            "la aceptacion YA PASA antes de tocar nada: esta mision no puede "
            "distinguir un exito de un no-hacer-nada. No es un resultado del modelo.",
        )
    failing = sorted(n for n, s in pre.outcomes.items() if s in ("failed", "error"))
    return Discrimination(
        NOT_SOLVED, pre, None,
        f"PRE falla como debe ({len(failing) or '?'} tests en rojo). La mision discrimina.",
    )


def check_post(
    pre: Discrimination, root: Path, acceptance: tuple[str, ...],
    *, timeout: float = PYTEST_TIMEOUT,
) -> Discrimination:
    """Complete the PRE/POST pair after the agent has finished."""
    if pre.status == NON_DISCRIMINATING:
        return pre  # nothing measured before, nothing measurable now
    post = run_suite(root, acceptance, timeout=timeout)
    if not post.usable:
        return Discrimination(
            UNKNOWN, pre.pre, post,
            f"la aceptacion POST no se pudo ejecutar (exit={post.exit_code}, "
            f"timeout={post.timed_out}).",
        )
    if pre.status == UNKNOWN:
        return Discrimination(UNKNOWN, pre.pre, post, pre.detail)
    if post.passed:
        return Discrimination(
            DISCRIMINATED, pre.pre, post,
            "PRE fallaba y POST pasa: el cambio resuelve la aceptacion declarada.",
        )
    return Discrimination(
        NOT_SOLVED, pre.pre, post,
        "PRE fallaba y POST sigue fallando: la aceptacion no esta resuelta.",
    )


# ----------------------------------------------------------- surface snapshots


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str          # function | class
    params: tuple[str, ...]


def public_surface(source: str) -> dict[str, Symbol]:
    """Module-level public API of one Python file, by name.

    "Public" is the leading-underscore convention, and nested functions are
    ignored: what matters for a preservation check is what other modules can
    reach. Parameters are recorded because renaming or reordering them breaks
    every keyword call site while leaving the name present, which a
    name-only comparison would score as unchanged.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return {}

    out: dict[str, Symbol] = {}

    def params_of(node) -> tuple[str, ...]:
        a = node.args
        names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        return tuple(names)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                out[node.name] = Symbol(node.name, "function", params_of(node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            out[node.name] = Symbol(node.name, "class", ())
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") and child.name != "__init__":
                        continue
                    dotted = f"{node.name}.{child.name}"
                    out[dotted] = Symbol(dotted, "function", params_of(child))
    return out


def snapshot_surfaces(root: Path, relatives) -> dict[str, dict[str, Symbol]]:
    out: dict[str, dict[str, Symbol]] = {}
    for rel in relatives:
        if not str(rel).endswith(".py"):
            continue
        target = root / rel
        try:
            out[str(rel)] = public_surface(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            out[str(rel)] = {}
    return out


# ------------------------------------------------------------- the conscience


@dataclass
class Signal:
    """One deterministic check, its verdict, and why."""

    name: str
    verdict: str
    detail: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "verdict": self.verdict,
                "detail": self.detail, "data": self.data}


EXPECTED_CHANGE = "EXPECTED_CHANGE"
PRESERVED_BEHAVIOR = "PRESERVED_BEHAVIOR"
UNEXPLAINED_CHANGE = "UNEXPLAINED_CHANGE"


@dataclass
class Conscience:
    signals: list[Signal] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return combine(s.verdict for s in self.signals)

    @property
    def blocking(self) -> list[Signal]:
        worst = self.verdict
        return [s for s in self.signals if s.verdict == worst and worst != PASS]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "signals": [s.to_dict() for s in self.signals],
            "blocking": [s.name for s in self.blocking],
        }


def signal_discrimination(disc: Discrimination) -> Signal:
    """The acceptance suite proved the change did the job. Or it did not."""
    mapping = {
        DISCRIMINATED: (PASS, EXPECTED_CHANGE),
        NOT_SOLVED: (REGRESSION, UNEXPLAINED_CHANGE),
        BROKE_IT: (REGRESSION, UNEXPLAINED_CHANGE),
        NON_DISCRIMINATING: (INCONCLUSIVE, UNEXPLAINED_CHANGE),
        UNKNOWN: (INCONCLUSIVE, UNEXPLAINED_CHANGE),
    }
    verdict, classification = mapping.get(disc.status, (INCONCLUSIVE, UNEXPLAINED_CHANGE))
    return Signal("acceptance_discriminates", verdict, disc.detail,
                  {"status": disc.status, "classification": classification})


def signal_acceptance_untouched(acceptance: tuple[str, ...], changed: set[str]) -> Signal:
    """Editing the test that judges you invalidates your own result.

    Tester already refuses a validation step that mutates tracked source, for
    the same reason. Here it is the sharpest single signal available: a green
    suite the agent rewrote is not evidence of anything.
    """
    touched = sorted(set(acceptance) & changed)
    if touched:
        return Signal(
            "acceptance_untouched", REGRESSION,
            f"el agente modifico su propia aceptacion: {', '.join(touched)}. "
            f"Un verde que se ha reescrito no prueba nada.",
            {"touched": touched, "classification": UNEXPLAINED_CHANGE},
        )
    return Signal("acceptance_untouched", PASS,
                  "la aceptacion no fue modificada por el agente.",
                  {"classification": PRESERVED_BEHAVIOR})


def signal_public_surface(
    before: dict[str, dict[str, Symbol]],
    after: dict[str, dict[str, Symbol]],
) -> Signal:
    """Nothing that other code could call has silently disappeared or changed shape.

    Additions are EXPECTED_CHANGE -- that is what building a feature looks
    like. Removals and signature changes are UNEXPLAINED_CHANGE: they break
    call sites the acceptance suite may well not cover, which is precisely the
    blind spot a green suite leaves.
    """
    removed: list[str] = []
    resigned: list[str] = []
    added: list[str] = []
    for rel, old in before.items():
        new = after.get(rel, {})
        for name, symbol in old.items():
            if name not in new:
                removed.append(f"{rel}::{name}")
            elif symbol.kind == "function" and new[name].params != symbol.params:
                resigned.append(
                    f"{rel}::{name}({', '.join(symbol.params)}) -> ({', '.join(new[name].params)})"
                )
        added += [f"{rel}::{n}" for n in new if n not in old]
    for rel, new in after.items():
        if rel not in before:
            added += [f"{rel}::{n}" for n in new]

    data = {"removed": removed, "signature_changed": resigned, "added": added}
    if removed or resigned:
        parts = []
        if removed:
            parts.append(f"desaparecieron: {', '.join(removed[:8])}")
        if resigned:
            parts.append(f"cambio de firma: {', '.join(resigned[:8])}")
        return Signal(
            "public_surface_preserved", INCONCLUSIVE,
            "la superficie publica cambio de forma no solicitada -- " + "; ".join(parts)
            + ". Puede ser correcto, pero no esta probado por la aceptacion.",
            data | {"classification": UNEXPLAINED_CHANGE},
        )
    return Signal(
        "public_surface_preserved", PASS,
        f"ninguna funcion o clase publica desaparecio ni cambio de firma"
        + (f"; {len(added)} anadidas" if added else "") + ".",
        data | {"classification": EXPECTED_CHANGE if added else PRESERVED_BEHAVIOR},
    )


def signal_no_collateral_regression(
    before: SuiteResult, after: SuiteResult, acceptance: tuple[str, ...],
) -> Signal:
    """Tests that passed before must still pass, even ones nobody mentioned.

    This is the signal that catches the failure mode a targeted acceptance
    suite structurally cannot: satisfying the sample while breaking the rest of
    the repository. It costs two whole-suite runs, and it is the best value in
    this module -- it needs no hand-written probe, no per-task setup, and it
    generalises to any repository with tests.
    """
    if not before.usable or not after.usable:
        return Signal(
            "no_collateral_regression", INCONCLUSIVE,
            f"la suite completa no se pudo comparar (PRE exit={before.exit_code}, "
            f"POST exit={after.exit_code}).",
            {"classification": UNEXPLAINED_CHANGE},
        )
    acceptance_set = set(acceptance)

    def outside_acceptance(node: str) -> bool:
        return not any(node == a or node.startswith(a) for a in acceptance_set)

    broke = sorted(
        node for node, status in before.outcomes.items()
        if status == "passed"
        and after.outcomes.get(node) in ("failed", "error")
        and outside_acceptance(node)
    )
    vanished = sorted(
        node for node, status in before.outcomes.items()
        if status == "passed" and node not in after.outcomes and outside_acceptance(node)
    )
    data = {"broke": broke, "disappeared": vanished,
            "pre_passed": sum(1 for s in before.outcomes.values() if s == "passed"),
            "post_passed": sum(1 for s in after.outcomes.values() if s == "passed")}
    if broke:
        return Signal(
            "no_collateral_regression", REGRESSION,
            f"{len(broke)} tests que pasaban antes ahora fallan: {', '.join(broke[:6])}"
            + (" ..." if len(broke) > 6 else ""),
            data | {"classification": UNEXPLAINED_CHANGE},
        )
    if vanished:
        return Signal(
            "no_collateral_regression", INCONCLUSIVE,
            f"{len(vanished)} tests que pasaban antes ya no se recogen: "
            f"{', '.join(vanished[:6])}. Un test que desaparece no es un test que pasa.",
            data | {"classification": UNEXPLAINED_CHANGE},
        )
    if data["pre_passed"] == 0:
        # Nothing passed before, so nothing can have gone from green to red.
        # That is a true statement rather than a dodge -- but it must be said
        # out loud, because "0 tests preserved" and "400 tests preserved" are
        # very different amounts of assurance behind the same PASS.
        return Signal(
            "no_collateral_regression", PASS,
            f"no habia ningun test en verde antes del cambio (la suite PRE no "
            f"llegaba a importarse), asi que no habia nada que romper. "
            f"Ahora pasan {data['post_passed']}. Comprobacion vacia, no fuerte.",
            data | {"classification": EXPECTED_CHANGE, "empty_baseline": True},
        )
    return Signal(
        "no_collateral_regression", PASS,
        f"los {data['pre_passed']} tests que pasaban antes siguen pasando.",
        data | {"classification": PRESERVED_BEHAVIOR, "empty_baseline": False},
    )


def signal_scope_respected(changed: set[str], scope) -> Signal:
    """Every file that differs was one the mission authorised.

    The tools refuse an out-of-scope write, so in principle this can only fire
    through ``run`` -- an agent that reaches for ``python -c`` with an open
    file. That is exactly why the check is done against the filesystem after
    the fact rather than trusted from the tool layer: a containment property
    verified only by the component that enforces it is not verified.
    """
    stray = sorted(rel for rel in changed if not scope.allows(rel, creating=True))
    if stray:
        return Signal(
            "scope_respected", REGRESSION,
            f"ficheros modificados fuera de write_scope: {', '.join(stray)}. "
            f"Ninguna herramienta lo permite, asi que se escribieron por otra via.",
            {"stray": stray, "classification": UNEXPLAINED_CHANGE},
        )
    return Signal("scope_respected", PASS,
                  f"los {len(changed)} ficheros modificados estan dentro del ambito.",
                  {"changed": sorted(changed), "classification": PRESERVED_BEHAVIOR})


def signal_agent_reported(finish_status: str | None, summary: str) -> Signal:
    """The agent's own claim about its work. NOT part of the conscience verdict.

    F-24. This used to be folded in with the deterministic signals, and it cost
    a real result: ga06 produced a module that took its acceptance suite from a
    collection error to 22 passed, broke nothing, and stayed in scope -- and was
    refused because the agent had called finish(status='BLOCKED'), unsure it had
    succeeded. Six pieces of evidence agreeing with each other were overruled by
    the model's confidence.

    I9 says the harness owns the verdict and the model's claim can never create
    a PASS. Letting that same claim destroy one makes the model the judge again
    with the sign reversed. So this is computed, reported and shown to whoever
    reads the evidence -- and ``work`` turns it into PASS_UNCONFIRMED rather
    than a refusal.
    """
    if finish_status == "BLOCKED":
        return Signal("agent_not_blocked", INCONCLUSIVE,
                      f"el agente declaro BLOCKED: {summary[:300]}",
                      {"finish_status": finish_status, "classification": UNEXPLAINED_CHANGE})
    if finish_status is None:
        return Signal("agent_not_blocked", INCONCLUSIVE,
                      "el agente nunca llamo a finish (agoto el presupuesto).",
                      {"finish_status": None, "classification": UNEXPLAINED_CHANGE})
    return Signal("agent_not_blocked", PASS, f"el agente termino con status={finish_status}.",
                  {"finish_status": finish_status, "classification": PRESERVED_BEHAVIOR})
