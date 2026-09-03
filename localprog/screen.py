"""Paso 0 -- the frozen screen of LOCAL_PROGRAMMER_V0_CONTRACT.md §D.

Question: *can this model chain five tool calls without breaking?* It does not
measure programming ability; the fix is two lines. A model that cannot drive
the loop is unusable here however clever it is, and that is the cheap thing to
discover first.

Everything in this module that the contract froze is reproduced verbatim: the
repo bytes (§D.1), the objective (§D.2), the system prompt (§D.3), the two
protocols (§D.4) and the scoring (§D.5). None of it may be tuned per model.

ONE DEVIATION, DELIBERATE AND FLAGGED
-------------------------------------
§D.3's system prompt never tells the model protocol B's wire format, while
§D.4 defines that format. A protocol whose syntax is never communicated cannot
be answered, and that is part of why every protocol-B run so far scored 0/5 --
the model had to guess ``LLAMADA:`` and only learned it from an error message
after already failing.

``PROTOCOL_B_SUFFIX`` below states the format taken verbatim from §D.4. It is
appended only for protocol B, identically for every model, and it is recorded
in the results as ``protocol_b_suffix_applied``. The §D.3 text itself is
untouched, byte for byte. If the operator prefers strict silence, delete the
suffix -- but then protocol B measures guessing, not the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import evidence, loop, tools, workspace
from .provider import OllamaProvider

SCREEN_ID = "SCREEN_V0"

#: The seven tools section D.3's prompt names, and the only ones this frozen
#: instrument declares (F-20). Everything since -- list_dir, run, search_code,
#: read_symbol, replace_lines, copy_code -- was added to the harness after this
#: screen was recorded; declaring any of them would change the question the
#: screen asks and make its results incomparable with the runs already sealed in
#: GATE-A-WS. dispatch still knows all thirteen; a model that reaches for one of
#: the other six gets ERROR_UNKNOWN_TOOL, exactly as it would have before they
#: were written.
#:
#: Passing declare_tools explicitly is also what keeps the screen outside
#: legal_tools: the surface a mission's scope permits is the right rule for a
#: mission and the wrong rule for a frozen instrument, whose surface must not
#: move for any reason at all.
FROZEN_TOOLS = (
    "read_file", "grep", "list_symbols", "edit", "write_file", "run_tests", "finish",
)

# ------------------------------------------------------------- frozen §D.1

CALC_PY = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def multiply(a, b):\n"
    "    return a * b\n"
    "\n"
    "\n"
    "def divide(a, b):\n"
    "    return a / b\n"
)

TEST_CALC_PY = (
    "import pytest\n"
    "\n"
    "from calc import add, divide, multiply\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_multiply():\n"
    "    assert multiply(2, 3) == 6\n"
    "\n"
    "\n"
    "def test_divide():\n"
    "    assert divide(6, 3) == 2\n"
    "\n"
    "\n"
    "def test_divide_by_zero():\n"
    "    with pytest.raises(ValueError):\n"
    "        divide(1, 0)\n"
)

SCREEN_FILES = {"calc.py": CALC_PY, "test_calc.py": TEST_CALC_PY}
WRITE_SCOPE = ("calc.py",)
ACCEPTANCE_TESTS = ("test_calc.py",)

# ------------------------------------------------------------- frozen §D.2

OBJECTIVE = (
    'El test test_divide_by_zero falla. Haz que divide() lance ValueError con el '
    'mensaje "division by zero" cuando b sea 0. No modifiques los tests.'
)

# ------------------------------------------------------------- frozen §D.3

SYSTEM_PROMPT = """Eres un programador. Trabajas llamando a herramientas, de una en una.

Herramientas:
  read_file(path, start=null, end=null)
  grep(pattern, glob="**/*.py")
  list_symbols(path)
  edit(path, old, new)          -- 'old' debe aparecer exactamente una vez
  write_file(path, content)     -- solo ficheros nuevos
  run_tests(node_ids=null)
  finish(summary)

Reglas:
  - Una llamada por turno. Espera el resultado antes de la siguiente.
  - Puedes leer cualquier fichero. Solo puedes escribir en: {write_scope}
  - Antes de editar, lee el fichero para copiar el texto exacto.
  - Cuando los tests pasen, llama a finish().
  - Tienes 12 turnos."""

#: §D.4's stated wire format, and nothing else. See the module docstring.
PROTOCOL_B_SUFFIX = """

Formato de llamada (protocolo de texto):
  LLAMADA: edit(path="calc.py", old="...", new="...")"""


def system_prompt(protocol: str, write_scope=WRITE_SCOPE) -> str:
    text = SYSTEM_PROMPT.format(write_scope=json.dumps(list(write_scope)))
    return text + PROTOCOL_B_SUFFIX if protocol == "B" else text


# ------------------------------------------------------------- frozen §D.6

DEFAULT_MODELS = (
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "deepcoder:14b-preview-q4_K_M",
    "hf.co/unsloth/Seed-Coder-8B-Instruct-GGUF:Q4_K_M",
    "opencoder:8b-instruct-q4_K_M",
    "qwen3.5:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
)

RUNS_PER_MODEL = 3

# ------------------------------------------------------------- frozen §D.5

PASS_CLEAN = "PASS_CLEAN"
PASS_NOISY = "PASS_NOISY"
PARTIAL = "PARTIAL"
FAIL = "FAIL"
INCOMPATIBLE = "INCOMPATIBLE"
VOID = "VOID"  # not a §D.5 level: no scoreable run existed

STEP_NAMES = ("S1_read_file", "S2_search", "S3_edit_applied", "S4_run_tests", "S5_finish_green")


@dataclass
class RunRecord:
    model: str
    protocol: str
    run: int
    steps: list[bool] = field(default_factory=lambda: [False] * 5)
    turnos_usados: int = 0
    llamadas_invalidas: int = 0
    bucles: int = 0
    tool_errors: int = 0
    outcome: str = ""
    scoreable: bool = False
    workspace: dict = field(default_factory=dict)
    provider_error: dict | None = None
    harness_invalid: dict | None = None
    wall_seconds: float = 0.0
    protocol_b_suffix_applied: bool = False

    @property
    def label(self) -> str:
        return f"{self.model}/{self.protocol}/run{self.run}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "protocol": self.protocol, "run": self.run,
            "steps": self.steps, "steps_named": dict(zip(STEP_NAMES, self.steps)),
            "score": sum(self.steps),
            "turnos_usados": self.turnos_usados,
            "llamadas_invalidas": self.llamadas_invalidas,
            "bucles": self.bucles, "tool_errors": self.tool_errors,
            "outcome": self.outcome, "scoreable": self.scoreable,
            "workspace": self.workspace,
            "provider_error": self.provider_error,
            "harness_invalid": self.harness_invalid,
            "wall_seconds": round(self.wall_seconds, 3),
            "protocol_b_suffix_applied": self.protocol_b_suffix_applied,
        }


def run_once(
    model: str,
    protocol: str,
    number: int,
    *,
    provider_factory: Callable[[str], Any] | None = None,
    out_dir: Path | None = None,
    base_dir: Path | None = None,
) -> RunRecord:
    """One screen run. Never raises for a model, provider or repo condition."""
    record = RunRecord(model=model, protocol=protocol, run=number)
    record.protocol_b_suffix_applied = protocol == "B"
    factory = provider_factory or (lambda name: OllamaProvider(name))
    box = workspace.synthesize(SCREEN_FILES, base_dir=base_dir)
    record.workspace = box.describe()

    try:
        ctx = tools.ToolContext(
            root=box.path, write_scope=WRITE_SCOPE,
            acceptance_tests=ACCEPTANCE_TESTS,
        )
        before = evidence.capture_before(box.path, SCREEN_FILES)
        result = loop.run_loop(
            provider=factory(model), ctx=ctx,
            system=system_prompt(protocol), objective=OBJECTIVE,
            protocol_name=protocol,
        declare_tools=FROZEN_TOOLS,
        )

        record.outcome = result.outcome
        record.scoreable = result.scoreable
        record.turnos_usados = result.turns_used
        record.llamadas_invalidas = result.invalid_calls
        record.bucles = result.loops
        record.tool_errors = result.tool_errors
        record.wall_seconds = result.wall_seconds
        record.provider_error = result.provider_error
        record.harness_invalid = result.harness_invalid

        if result.scoreable:
            used = result.tools_used
            ok_by_tool = {e.get("tool") for e in result.events if e.get("ok")}
            record.steps[0] = "read_file" in ok_by_tool
            record.steps[1] = bool({"grep", "list_symbols"} & ok_by_tool)
            record.steps[2] = bool(result.changed_files)
            record.steps[3] = "run_tests" in used
            # S5 is decided by the harness re-running the suite, never by the
            # model's own claim that it is done.
            final_green = False
            if result.outcome == loop.FINISHED:
                verdict = tools.dispatch(ctx, "run_tests", {})
                final_green = bool(verdict.ok and verdict.value.get("passed"))
            record.steps[4] = result.outcome == loop.FINISHED and final_green

        after, diff_text = evidence.capture_after(box.path, before)
        if out_dir is not None:
            evidence.seal(
                out_dir / "transcripts",
                name=f"{model.replace('/', '_').replace(':', '_')}_{protocol}_{number}",
                record=record.to_dict(),
                events=result.events,
                diff_text=diff_text,
            )
        return record
    finally:
        # Preserve anything that was not a clean, scoreable run.
        box.dispose(preserve=not record.scoreable)
        record.workspace = box.describe()


def classify(runs: list[RunRecord]) -> str:
    """§D.5, applied to one model's runs. Scoring is frozen; this only reads it."""
    scoreable = [r for r in runs if r.scoreable]
    if not scoreable:
        return VOID
    full = [r for r in scoreable if all(r.steps)]
    reached_edit = [r for r in scoreable if r.steps[2]]
    any_valid_call = any(r.turnos_usados > r.llamadas_invalidas for r in scoreable)

    if len(full) == len(scoreable) == RUNS_PER_MODEL:
        average_turns = sum(r.turnos_usados for r in scoreable) / len(scoreable)
        if average_turns <= 8 and all(r.llamadas_invalidas == 0 for r in scoreable):
            return PASS_CLEAN
    if len(full) >= 2:
        return PASS_NOISY
    if len(reached_edit) >= 2:
        return PARTIAL
    if not any_valid_call:
        return INCOMPATIBLE
    return FAIL


ADVANCING = (PASS_CLEAN, PASS_NOISY)
