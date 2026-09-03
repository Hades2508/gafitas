"""Can this engine actually drive GAFITAS? Seven questions, one small repository.

WHAT THIS IS FOR
----------------
Adding an engine must cost an adapter and a certification, not a fork of the
harness. If a new brain needs a new runner, new tools, a new Tester or a new
evidence pipeline, the abstraction is wrong and this file is where that shows up
first: certification uses the ordinary ``work.run_ticket`` path, the ordinary
tools and the ordinary verdict. Nothing here is special-cased for anybody.

THE SEVEN
---------
    TOOL_USE   emits a well-formed call the dispatcher accepts
    SEARCH     finds a symbol it was not told the name of
    READ       opens what it found
    EDIT       changes it, and the change is the one asked for
    RUN        executes the suite and reads the result
    RECOVERY   is refused once, and recovers instead of repeating
    FINISH     ends deliberately, with the verdict the harness reaches

They are ordered because they compose: an engine that cannot emit a call cannot
search, and one that cannot read cannot edit. A failure at step N says the
engine is unusable from N onwards, which is more useful than a score.

WHAT A PASS MEANS AND DOES NOT
------------------------------
Certification says an engine can be driven. It says nothing about how well it
programs -- that is what the ticket batteries and the benchmarks are for, and
conflating the two would let "it answered" stand in for "it worked".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import engine, work

SCHEMA = "GAFITAS_ENGINE_CERTIFICATION_V1"

CHECKS = ("TOOL_USE", "SEARCH", "READ", "EDIT", "RUN", "RECOVERY", "FINISH")

#: The repository every engine is certified against. Deliberately tiny and
#: deliberately NOT a benchmark case: the bug is obvious, the fix is one line,
#: and the only thing being measured is whether the engine can be driven through
#: the loop at all. Anything harder would confound "cannot drive the harness"
#: with "cannot do the task".
FIXTURE = {
    "app/__init__.py": "",
    "app/rates.py": (
        "def apply_discount(total_cents, percent):\n"
        '    """Take a percentage off a total, in whole cents.\n'
        "\n"
        "    The result must never be negative and must never exceed the total.\n"
        '    """\n'
        "    return total_cents - (total_cents * percent)\n"
    ),
    "tests/__init__.py": "",
    "tests/test_rates.py": (
        "from app.rates import apply_discount\n"
        "\n"
        "\n"
        "def test_ten_percent_off_a_thousand():\n"
        "    assert apply_discount(1000, 10) == 900\n"
        "\n"
        "\n"
        "def test_nothing_off_is_the_whole_total():\n"
        "    assert apply_discount(1000, 0) == 1000\n"
    ),
}

OBJECTIVE = """En este repositorio hay una funcion que resta un porcentaje a un
total en centimos, y esta mal: trata el porcentaje como si fuera una fraccion ya
dividida, asi que descuenta muchisimo de mas.

Arreglala para que un 10 por ciento de 1000 sean 900.

No sabes como se llama ni en que fichero esta: encuentrala. Cuando la hayas
arreglado, ejecuta los tests y termina con finish(status='DONE').
"""


@dataclass
class Certification:
    engine: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    outcome: str = ""
    turns: int = 0
    invalid_calls: int = 0
    tool_errors: int = 0
    wall_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tools_used: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    detail: str = ""

    @property
    def certified(self) -> bool:
        """Every check, in order. A partial pass is not a certification."""
        return not self.failed and len(self.passed) == len(CHECKS)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema"] = SCHEMA
        out["certified"] = self.certified
        return out


def _materialise(root: Path) -> Path:
    for relative, body in FIXTURE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def certify(model: str, *, base_dir: Path, out_dir: Path | None = None,
            max_turns: int = 25, capabilities: Any | None = None,
            provider_factory: Any | None = None) -> Certification:
    """Drive *model* through the seven checks on the ordinary path."""
    caps = capabilities or engine.capabilities_for(model, probe_if_missing=True)
    repo = _materialise(Path(base_dir) / f"certify_{model.replace(':', '_').replace('/', '_')}")

    ticket = work.Ticket(
        ticket_id=f"certify__{model.replace(':', '_').replace('/', '_')}",
        repo=repo,
        objective=OBJECTIVE,
        write_scope=("app/",),
        acceptance_tests=("tests/test_rates.py",),
        full_suite=False,
        max_turns=max_turns,
    )

    started = time.perf_counter()
    result = work.run_ticket(ticket, model, out_dir=out_dir, capabilities=caps,
                             base_dir=Path(base_dir), provider_factory=provider_factory)
    report = Certification(
        engine=model, outcome=result.outcome, turns=result.turns_used,
        invalid_calls=result.invalid_calls, tool_errors=result.tool_errors,
        wall_seconds=round(time.perf_counter() - started, 1),
        input_tokens=result.usage.get("input_tokens", 0),
        output_tokens=result.usage.get("output_tokens", 0),
        tools_used=dict(result.tools_used), capabilities=caps.to_dict(),
    )

    used = result.tools_used
    # Any navigation tool counts. The capability is "explored to locate a target
    # it was not told the position of", not "used the tool I happen to prefer":
    # on a four-file fixture list_dir IS searching, and the first version of
    # this check failed a run that had found and fixed the bug in six turns.
    searched = sum(used.get(name, 0) for name in
                   ("search_code", "grep", "list_symbols", "list_dir"))
    read = used.get("read_symbol", 0) + used.get("read_file", 0)
    edited = used.get("edit", 0) + used.get("replace_lines", 0) + used.get("write_file", 0)

    evidence = {
        # A call the dispatcher accepted at all. invalid_calls counts the ones
        # it did not, so "some turns produced a usable call" is the question.
        "TOOL_USE": result.turns_used > result.invalid_calls and bool(used),
        "SEARCH": searched > 0,
        "READ": read > 0,
        "EDIT": edited > 0 and bool(result.changed_files),
        "RUN": used.get("run_tests", 0) + used.get("run", 0) > 0,
        # Being refused and continuing anyway. An engine that never gets refused
        # passes this trivially and correctly: it did not need to recover.
        "RECOVERY": (result.tool_errors + result.invalid_calls) == 0
                    or result.turns_used > (result.tool_errors + result.invalid_calls),
        # The harness's verdict, not the engine's claim about itself.
        "FINISH": result.outcome in (work.PASS, work.PASS_UNCONFIRMED),
    }
    for check in CHECKS:
        (report.passed if evidence[check] else report.failed).append(check)
    report.detail = "; ".join(result.notes[:3])
    return report


def certify_many(models: list[str], *, base_dir: Path,
                 out_dir: Path | None = None) -> dict:
    reports = {}
    for model in models:
        try:
            reports[model] = certify(model, base_dir=base_dir, out_dir=out_dir).to_dict()
        except Exception as exc:            # noqa: BLE001 - one engine must not
            reports[model] = {              # take the certification run down
                "engine": model, "certified": False,
                "failed": list(CHECKS),
                "detail": f"{type(exc).__name__}: {exc}",
            }
    return {"schema": SCHEMA, "at": time.time(), "engines": reports}


def write_report(report: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return Path(path)
