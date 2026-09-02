"""SWE-bench instance -> GAFITAS ticket -> model_patch. The minimum adapter.

One translation layer, permanent, with no per-repo and no per-instance special
cases. GAFITAS works with its ordinary tools; nothing here teaches it anything
about SWE-bench.

WHAT THE AGENT IS ALLOWED TO SEE
--------------------------------
Only what a legitimate agent could know at the moment the issue was filed:

    problem_statement, repo, base_commit, and the repo AT base_commit.

Never ``patch`` (the human fix), never ``test_patch`` (the tests that judge
it), never FAIL_TO_PASS or PASS_TO_PASS (their names alone would point
straight at the answer), never ``hints_text`` (it quotes the real discussion,
sometimes including the fix). ``build_ticket`` reads exactly four fields off
the instance and there is a test asserting it touches no others -- an
adapter that leaks the answer measures nothing, and it would leak quietly.

THE SHAPE MISMATCH, STATED PLAINLY
----------------------------------
GAFITAS is built around a mission that hands over a FAILING TEST: the
discrimination gate proves the task is unsolved before the model is invoked
(F-01), ``finish(DONE)`` refuses until the agent has watched that test pass
(F-29), and the strongest conscience signal is PRE-fail-to-POST-pass.

SWE-bench deliberately withholds the tests. So on this benchmark:

  * ``acceptance_tests`` is EMPTY. There is nothing to declare.
  * the discrimination gate cannot run -- it has nothing to discriminate on.
  * ``finish(DONE)`` falls back to its no-declared-tests behaviour, which
    accepts the agent's word that it is done.
  * ``acceptance_discriminates`` is absent from the conscience, and the
    remaining signals check only that nothing was broken, not that anything
    was fixed.

That is not a defect in the adapter and it must not be papered over. It means
GAFITAS runs here with its main verification organ removed, judged afterwards
by a harness it cannot see -- a harder task than any it has been measured on,
and a different one. The agent can still run the repository's own existing
tests through ``run`` and ``run_tests(node_ids=...)``; it simply has no
declared target to aim at.

The official SWE-bench harness is the only judge of the score. Our conscience
signals are recorded alongside as internal telemetry and decide nothing.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import work

SCHEMA = "GAFITAS_SWEBENCH_ADAPTER_V1"

#: The only fields of an instance the agent may ever be exposed to. Anything
#: else on a SWE-bench row is either the answer or a pointer to it.
VISIBLE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")

#: Fields that must never reach the agent, named so the guard is readable and
#: so a future reader can see WHY each one is excluded.
FORBIDDEN_FIELDS = (
    "patch",           # the human fix
    "test_patch",      # the tests that judge it
    "FAIL_TO_PASS",    # their names point at the answer
    "PASS_TO_PASS",
    "hints_text",      # quotes the real discussion, sometimes with the fix
)

GIT_TIMEOUT = 900.0


@dataclass
class Instance:
    """One SWE-bench row, reduced to what an agent may legitimately know."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    @classmethod
    def from_row(cls, row: dict) -> "Instance":
        missing = [f for f in VISIBLE_FIELDS if not row.get(f)]
        if missing:
            raise ValueError(f"instance is missing {missing}")
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row["problem_statement"]),
        )

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id, "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement_chars": len(self.problem_statement),
        }


OBJECTIVE = """{problem_statement}

---
Este es un issue real del repositorio {repo}. Arreglalo en el codigo.

No hay tests de aceptacion declarados: nadie te va a decir cual es el test que
te juzga. Tendras que decidir tu que hay que cambiar a partir de lo que dice el
issue, y comprobarlo tu mismo.

Como trabajar aqui:
  - El repositorio es GRANDE. No lo leas entero. Usa grep para encontrar donde
    esta lo que menciona el issue, y read_symbol para mirar una funcion o clase
    concreta.
  - Los tests que YA existen en el repositorio si puedes ejecutarlos:
    run_tests(node_ids=["ruta/al/test.py::test_lo_que_sea"]). Usalos para
    comprobar que no rompes nada.
  - Haz el cambio MINIMO que arregla lo que describe el issue. No refactorices
    de paso, no arregles otras cosas, no toques los tests.
  - Si no consigues localizar el problema, finish(status='BLOCKED') explicando
    que has mirado. Es mejor eso que un cambio al azar.
"""


def build_ticket(
    instance: Instance,
    repo_path: Path,
    *,
    max_turns: int = 40,
    write_scope: tuple[str, ...] = ("**",),
) -> work.Ticket:
    """A GAFITAS ticket carrying only what the agent is allowed to know.

    ``write_scope`` is the whole checkout, because a real issue does not come
    with a list of files to change and narrowing it would be handing over part
    of the answer. Containment is unaffected: ``programmer.guard`` still
    refuses anything outside the workspace, and the post-run write verification
    still reports every file touched.

    ``acceptance_tests`` is empty. See the module docstring -- that is the
    benchmark's design, not an oversight, and it removes GAFITAS's main
    verification organ for the duration.
    """
    return work.Ticket(
        ticket_id=instance.instance_id,
        repo=repo_path,
        objective=OBJECTIVE.format(
            problem_statement=instance.problem_statement.strip(),
            repo=instance.repo,
        ),
        write_scope=write_scope,
        acceptance_tests=(),
        allowed_new_files=(),
        # The whole suite of a 300k-line repository, twice, is not affordable
        # per instance -- and with no acceptance test there is no PRE baseline
        # worth comparing against anyway.
        full_suite=False,
        max_turns=max_turns,
    )


def assert_no_leak(row: dict, ticket: work.Ticket) -> None:
    """Prove the answer did not reach the ticket. Cheap, and worth doing.

    A leaking adapter does not fail loudly; it produces a good score and a
    worthless measurement. This runs on every instance.
    """
    haystack = f"{ticket.objective}\n{ticket.ticket_id}"
    for field_name in FORBIDDEN_FIELDS:
        value = row.get(field_name)
        if not value:
            continue
        text = value if isinstance(value, str) else json.dumps(value)
        # A short fragment can coincide innocently; a long one cannot.
        for chunk in _significant_chunks(text):
            if chunk in haystack:
                raise ValueError(
                    f"LEAK: {field_name} content reached the ticket for "
                    f"{ticket.ticket_id}: {chunk[:80]!r}"
                )


def _significant_chunks(text: str, size: int = 120) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 40]
    return [l[:size] for l in lines[:200]]


# ------------------------------------------------------------------ checkout


def prepare_repo(instance: Instance, cache_dir: Path) -> Path:
    """A checkout of *instance.repo* at *base_commit*, cloned once per repo.

    The clone is a cache shared across every instance of the same repository --
    django alone is 231 of the 500 instances and cloning it 231 times would be
    most of the wall time. GAFITAS still runs in its own disposable worktree
    off this checkout, so instances cannot see each other's edits.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    mirror = cache_dir / instance.repo.replace("/", "__")
    if not (mirror / ".git").is_dir():
        result = _git(
            ["clone", f"https://github.com/{instance.repo}.git", str(mirror)],
            cwd=cache_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(f"clone failed for {instance.repo}: {result.stderr[-400:]}")

    fetch = _git(["fetch", "--quiet", "origin", instance.base_commit], cwd=mirror)
    if fetch.returncode != 0:
        _git(["fetch", "--quiet", "--all"], cwd=mirror)
    checkout = _git(["checkout", "--force", "--quiet", instance.base_commit], cwd=mirror)
    if checkout.returncode != 0:
        raise RuntimeError(
            f"checkout {instance.base_commit} failed for {instance.repo}: "
            f"{checkout.stderr[-400:]}"
        )
    _git(["clean", "-qfdx"], cwd=mirror)
    return mirror


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        timeout=GIT_TIMEOUT, shell=False,
    )


# ------------------------------------------------------------- predictions


@dataclass
class Prediction:
    instance_id: str
    model_patch: str
    model_name_or_path: str
    #: Internal telemetry. Recorded beside the prediction and never part of it:
    #: the official harness is the only judge of the score.
    internal: dict = field(default_factory=dict)

    def official(self) -> dict:
        """Exactly the three fields the official harness reads."""
        return {
            "instance_id": self.instance_id,
            "model_patch": self.model_patch,
            "model_name_or_path": self.model_name_or_path,
        }


def extract_patch(workspace: Path) -> str:
    """The agent's change as a unified diff against base_commit.

    Untracked files are added to the index first, so a new module the agent
    created appears in the diff. Nothing is committed -- the index is a
    scratch surface here, and the workspace is thrown away afterwards.
    """
    _git(["add", "-A"], cwd=workspace)
    result = _git(["diff", "--cached", "--no-color"], cwd=workspace)
    return result.stdout if result.returncode == 0 else ""


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.official() for p in predictions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
