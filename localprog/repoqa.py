"""RepoQA Search-Needle-Function as a GAFITAS ticket. The minimum adapter.

WHAT REPOQA ASKS
----------------
Given a natural-language description of a function -- deliberately obfuscated so
it never names the function -- find that function in a repository and reproduce
it. The official evaluator parses the answer, extracts a function, and compares
it to the reference with a fuzzy similarity threshold.

TWO DIFFERENT MEASUREMENTS, AND THEY MUST NOT BE MIXED
------------------------------------------------------
MODEL_NATIVE is RepoQA's own protocol: the harness flattens ~16k tokens of
truncated repository source into one prompt and the model answers in one shot.
That measures long-context understanding, and it is what the leaderboard ranks.

GAFITAS_AGENT is this adapter: the repository is written to disk and the agent
searches it with grep, list_symbols and read_symbol over many turns. That
measures NAVIGATION -- which is the half of GAFITAS's job this benchmark is
actually relevant to, and which the single-prompt protocol cannot see.

They answer different questions and the numbers are not interchangeable. Any
GAFITAS_AGENT result must be reported as AGENT_ADAPTED and NOT compared to the
leaderboard. Claiming otherwise would be the same error as scoring a solved
STEP1 corpus.

WHAT THE AGENT MAY SEE
----------------------
The description, and the repository. Never ``name``, never ``path``, never the
line or byte offsets -- any one of them is the answer. ``build_ticket`` reads
exactly one field off the needle and ``assert_no_leak`` checks the rest stayed
out, on every case.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import work

SCHEMA = "GAFITAS_REPOQA_ADAPTER_V1"

#: The only needle field the agent may be shown.
VISIBLE_NEEDLE_FIELDS = ("description",)

#: Each of these identifies the answer outright.
FORBIDDEN_NEEDLE_FIELDS = (
    "name", "path", "start_line", "end_line", "start_byte", "end_byte",
    "global_start_line", "global_end_line", "global_start_byte", "global_end_byte",
)

#: Where the agent is told to put its answer. A file, because writing one is
#: something GAFITAS already knows how to do -- no new tool, no new protocol.
ANSWER_FILE = "REPOQA_ANSWER.txt"

EXTENSION = {
    "python": ".py", "cpp": ".cpp", "java": ".java",
    "typescript": ".ts", "rust": ".rs",
}


OBJECTIVE = """Busca UNA funcion concreta dentro de este repositorio.

Esto es lo unico que se sabe de ella -- una descripcion de lo que hace. No
tienes su nombre ni su fichero: encontrarlos es la tarea.

--- DESCRIPCION ---
{description}
--- FIN DE LA DESCRIPCION ---

Cuando la encuentres, escribe su codigo COMPLETO Y LITERAL en el fichero
{answer_file}: desde la linea de la declaracion hasta su ultima linea, tal cual
esta en el repositorio, sin cambiar nada y sin anadir explicaciones.

Como buscarla:
  - El repositorio esta en {language}. Empieza con list_dir para ver su forma.
  - grep con context es tu mejor herramienta: busca terminos de la descripcion
    (tipos, palabras clave, mensajes de error, nombres de campos).
  - list_symbols te dice que define un fichero; read_symbol te da una funcion
    concreta con sus lineas exactas, que es justo lo que hay que copiar.
  - NO leas el repositorio entero. Es grande y no te hace falta.

Cuando hayas escrito {answer_file}, termina con finish(status='DONE').
Si no consigues encontrarla, finish(status='BLOCKED') diciendo que buscaste.
"""


@dataclass
class Case:
    """One needle, reduced to what the agent may legitimately know."""

    language: str
    repo: str
    description: str
    #: Kept for bookkeeping and for the evaluator. NEVER put into the ticket.
    needle_name: str

    @property
    def case_id(self) -> str:
        return f"{self.language}::{self.repo}::{self.needle_name}"


def build_ticket(case: Case, repo_path: Path, *, max_turns: int = 40) -> work.Ticket:
    """A ticket carrying the description and nothing else identifying."""
    return work.Ticket(
        ticket_id=f"{case.language}__{case.repo.replace('/', '_')}__{case.needle_name}",
        repo=repo_path,
        objective=OBJECTIVE.format(
            description=case.description.strip(),
            answer_file=ANSWER_FILE,
            language=case.language,
        ),
        # The agent may create only its answer file. It has no reason to modify
        # the repository, and forbidding it means a stray edit is refused by the
        # tools rather than found afterwards in the diff.
        write_scope=(),
        allowed_new_files=(ANSWER_FILE,),
        acceptance_tests=(),
        full_suite=False,
        max_turns=max_turns,
    )


def assert_no_leak(needle: dict, ticket: work.Ticket) -> None:
    """Prove WE did not add the answer to the ticket. Runs on every case.

    Checked against the objective MINUS the description, and that exclusion is
    the whole point. The description is sanctioned content: RepoQA obfuscates it
    on purpose so the function cannot be identified from it, and whatever words
    it happens to contain are the benchmark's decision, not our contamination.

    Checking the description too produced a false positive that killed a run 19
    cases in -- the needle was named ``base``, and a description about a base
    directory naturally contains the word. A guard that fires on ordinary
    English is a guard that gets switched off, which is worse than no guard.
    What this must catch is the adapter putting the name or path into the
    template, and that is exactly what remains once the description is removed.
    """
    description = needle.get("description") or "￿-no-description-￿"
    surround = ticket.objective.replace(description, "")
    for field in FORBIDDEN_NEEDLE_FIELDS:
        value = needle.get(field)
        if value is None:
            continue
        text = str(value)
        if field in ("name", "path") and text and text in surround:
            raise ValueError(
                f"LEAK: needle {field}={text!r} was added to the ticket by the adapter"
            )


def materialise_repo(repo_record: dict, target: Path) -> Path:
    """Write the repository's files to disk and make it a git repo.

    GAFITAS opens a git worktree when it can, so a real repository here means
    every case runs in the same isolation as ordinary work -- and the answer
    file lands in a throwaway tree rather than anywhere persistent.
    """
    target.mkdir(parents=True, exist_ok=True)
    for relative, content in repo_record["content"].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    (target / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "repoqa@gafitas"],
        ["config", "user.name", "repoqa"],
        ["add", "-A"],
        ["commit", "-qm", "repoqa snapshot"],
    ):
        subprocess.run(["git", *args], cwd=str(target), capture_output=True, timeout=300)
    return target


def read_answer(workspace: Path) -> str:
    """Whatever the agent wrote, verbatim. Empty when it wrote nothing."""
    answer = workspace / ANSWER_FILE
    try:
        return answer.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def answer_from_patch(patch_text: str) -> str:
    """Recover the answer file's contents out of the sealed patch.

    ``run_ticket`` disposes the workspace once it has produced a CANDIDATE, so
    by the time the caller gets the result the answer file is gone. The patch
    survives, and the answer is a NEW file in it, which means every one of its
    lines is an addition. Reading it back from there beats keeping the whole
    workspace alive just to fish one file out of it.
    """
    if not patch_text:
        return ""
    lines = patch_text.splitlines()
    collecting = False
    body: list[str] = []
    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            # +++ b/<path> tells us which file the following hunk belongs to.
            if line.startswith("+++ "):
                collecting = line.endswith(ANSWER_FILE)
            continue
        if line.startswith("diff --git") or line.startswith("index "):
            collecting = False
            continue
        if line.startswith("@@"):
            continue
        if collecting and line.startswith("+"):
            body.append(line[1:])
    return chr(10).join(body)


def to_official_output(case: Case, answer: str, *, model: str) -> dict:
    """One row in the shape repoqa.compute_score consumes.

    The evaluator reads ``output`` and parses a function out of it, so the raw
    answer goes in unmodified. Nothing here tries to help it parse: cleaning up
    the model's text would be scoring our post-processing instead of the model.
    """
    return {
        "repo": case.repo,
        "name": case.needle_name,
        "language": case.language,
        "output": [answer],
        "model": model,
        # compute_score copies these three into its per-case record for its
        # needle-position analysis. They describe WHERE the needle sat inside
        # the truncated 16k context of the native protocol -- and in the agent
        # setting nothing is truncated, because the whole repository is on
        # disk. So they are explicitly null rather than invented: the verdict
        # itself is computed only from the answer, the ground-truth name and
        # the repo, so scoring is unaffected either way.
        "position_ratio": None,
        "needle_token_start": None,
        "needle_token_end": None,
    }


def write_outputs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
