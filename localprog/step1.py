"""Paso 1 -- the bare loop over a mission corpus (contract §E).

Same loop, same tools, same protocols as the screen. Only the workspace and the
objective change: a real repository at a real commit instead of a synthesised
three-file toy.

The gate (§E) is 3/5 on the frozen corpus. This module does not evaluate the
gate and does not decide anything -- it measures and seals. Reading the number
is the operator's job, and running it is explicitly NOT part of the harness
build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import evidence, loop, telemetry_bridge, tools, workspace
from .errors import HarnessInvalid
from .provider import OllamaProvider
from .screen import PROTOCOL_B_SUFFIX, SYSTEM_PROMPT

STEP1_ID = "STEP1"
REQUIRED_FIELDS = ("mission_id", "repo", "objective", "read_scope", "write_scope", "acceptance_tests")


@dataclass(frozen=True)
class Mission:
    mission_id: str
    repo: Path
    objective: str
    read_scope: tuple[str, ...]
    write_scope: tuple[str, ...]
    acceptance_tests: tuple[str, ...]
    allowed_new_files: tuple[str, ...] = ()
    commit: str | None = None

    @property
    def writable(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.write_scope + self.allowed_new_files))


def load_mission(path: Path) -> Mission:
    """Read a mission, or say precisely what is wrong with it.

    A malformed mission is a defect in the CORPUS, and it is raised as
    HarnessInvalid so it can never be recorded as a model failure. That is the
    exact confusion that voided four subtasks in the multifile campaign: an
    empty acceptance list produced INVALID_SPEC, the mission was scored as a
    failure, and only a later investigation found no model had ever been called.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarnessInvalid(f"cannot read mission {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HarnessInvalid(f"mission {path} is not a JSON object")
    missing = [f for f in REQUIRED_FIELDS if not payload.get(f)]
    if missing:
        raise HarnessInvalid(f"mission {path} is missing/empty: {missing}")

    def relatives(key: str) -> tuple[str, ...]:
        raw = payload.get(key) or []
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise HarnessInvalid(f"mission {path}: {key} must be a list of strings")
        repo_root = Path(payload["repo"])
        out = []
        for item in raw:
            candidate = Path(item)
            if candidate.is_absolute():
                try:
                    item = candidate.relative_to(repo_root).as_posix()
                except ValueError as exc:
                    raise HarnessInvalid(f"mission {path}: {item!r} is outside repo") from exc
            out.append(item.replace("\\", "/"))
        return tuple(out)

    return Mission(
        mission_id=str(payload["mission_id"]),
        repo=Path(payload["repo"]),
        objective=str(payload["objective"]),
        read_scope=relatives("read_scope"),
        write_scope=relatives("write_scope"),
        acceptance_tests=relatives("acceptance_tests"),
        allowed_new_files=relatives("allowed_new_files"),
        commit=payload.get("commit"),
    )


@dataclass
class MissionRecord:
    mission_id: str
    model: str
    protocol: str
    outcome: str = ""
    scoreable: bool = False
    tester_pass: bool | None = None
    turns_used: int = 0
    invalid_calls: int = 0
    tool_errors: int = 0
    loops: int = 0
    tools_used: dict = field(default_factory=dict)
    changed_files: list[str] = field(default_factory=list)
    out_of_scope_writes: int = 0
    wall_seconds: float = 0.0
    workspace: dict = field(default_factory=dict)
    provider_error: dict | None = None
    harness_invalid: dict | None = None
    telemetry_notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.mission_id}/{self.model}/{self.protocol}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id, "model": self.model, "protocol": self.protocol,
            "outcome": self.outcome, "scoreable": self.scoreable,
            "tester_pass": self.tester_pass,
            "turns_used": self.turns_used, "invalid_calls": self.invalid_calls,
            "tool_errors": self.tool_errors, "loops": self.loops,
            "tools_used": self.tools_used, "changed_files": self.changed_files,
            "out_of_scope_writes": self.out_of_scope_writes,
            "wall_seconds": round(self.wall_seconds, 3),
            "workspace": self.workspace,
            "provider_error": self.provider_error,
            "harness_invalid": self.harness_invalid,
            "telemetry_notes": self.telemetry_notes,
        }


def system_prompt(mission: Mission, protocol: str) -> str:
    text = SYSTEM_PROMPT.format(write_scope=json.dumps(list(mission.writable)))
    return text + PROTOCOL_B_SUFFIX if protocol == "B" else text


def objective_text(mission: Mission) -> str:
    return (
        f"{mission.objective}\n\n"
        f"write_scope = {json.dumps(list(mission.writable))}\n"
        f"read_scope = {json.dumps(list(mission.read_scope))}\n"
        f"acceptance_tests = {json.dumps(list(mission.acceptance_tests))}"
    )


def run_mission(
    mission: Mission,
    model: str,
    *,
    protocol: str = "A",
    provider_factory: Callable[[str], Any] | None = None,
    out_dir: Path | None = None,
    telemetry: Any | None = None,
    base_dir: Path | None = None,
) -> MissionRecord:
    record = MissionRecord(mission_id=mission.mission_id, model=model, protocol=protocol)
    factory = provider_factory or (lambda name: OllamaProvider(name))
    telemetry = telemetry or telemetry_bridge.NullTelemetry("no telemetry supplied")

    box = workspace.open_repo(
        mission.repo, commit=mission.commit,
        prefix=f"lp_{mission.mission_id}_", base_dir=base_dir,
    )
    record.workspace = box.describe()
    telemetry.start(mission.mission_id, repo=str(mission.repo), objective=mission.objective)

    try:
        ctx = tools.ToolContext(
            root=box.path,
            write_scope=mission.write_scope,
            allowed_new_files=mission.allowed_new_files,
            acceptance_tests=mission.acceptance_tests,
        )
        watched = set(mission.writable) | set(mission.read_scope)
        before = evidence.capture_before(box.path, watched)

        result = loop.run_loop(
            provider=factory(model), ctx=ctx,
            system=system_prompt(mission, protocol),
            objective=objective_text(mission),
            protocol_name=protocol,
        )

        record.outcome = result.outcome
        record.scoreable = result.scoreable
        record.turns_used = result.turns_used
        record.invalid_calls = result.invalid_calls
        record.tool_errors = result.tool_errors
        record.loops = result.loops
        record.tools_used = result.tools_used
        record.changed_files = result.changed_files
        record.wall_seconds = result.wall_seconds
        record.provider_error = result.provider_error
        record.harness_invalid = result.harness_invalid
        record.out_of_scope_writes = sum(
            1 for f in result.changed_files if f not in set(mission.writable)
        )

        # The verdict is the harness's, taken after the loop, on the declared
        # suite -- never the model's claim and never a mid-run reading.
        if result.scoreable:
            verdict = tools.dispatch(ctx, "run_tests", {})
            record.tester_pass = bool(verdict.ok and verdict.value.get("passed"))

        after, diff_text = evidence.capture_after(box.path, before)
        telemetry.attempt(mission.mission_id, latency_ms=result.wall_seconds * 1000)
        telemetry.finish(mission.mission_id, outcome=result.outcome)
        record.telemetry_notes = list(getattr(telemetry, "notes", []))

        if out_dir is not None:
            evidence.seal(
                out_dir / "missions",
                name=f"{mission.mission_id}_{model.replace('/', '_').replace(':', '_')}_{protocol}",
                record=record.to_dict(), events=result.events, diff_text=diff_text,
            )
        return record
    finally:
        box.dispose(preserve=not record.scoreable or record.tester_pass is not True)
        record.workspace = box.describe()
