"""MISSION_TELEMETRY_V1 lifecycle, with the duplicate guard handled, not hit.

Telemetry refuses to start a mission id twice under the same run id -- a
deliberate anti-double-counting rule. In the multifile campaign that refusal
arrived as an uncaught ``DuplicateMissionError`` and killed a re-run mid-cohort.
The guard is right; crashing on it is not. Here it becomes a structured,
recorded outcome, and the run continues without telemetry rather than dying.

Telemetry is never allowed to change a measurement. If the store cannot be
opened at all, ``NullTelemetry`` takes over and the fact is recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import deps

TASK_TYPE = "LOCAL_PROGRAMMER_LOOP"

#: Loop outcome -> (telemetry result, failure category). Categories are
#: Telemetry's own closed enum; nothing new is invented here.
OUTCOME_MAP = {
    "FINISHED": ("SUCCESS", None),
    "BUDGET_EXHAUSTED": ("FAILURE", "IMPLEMENTATION_FAILURE"),
    "PROVIDER_ERROR": ("ERROR", "TRANSPORT_ERROR"),
    "HARNESS_INVALID": ("ERROR", "UNKNOWN"),
}


class NullTelemetry:
    """Used when the store is unavailable. Records nothing, breaks nothing."""

    available = False

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.notes: list[str] = [reason]

    def start(self, *a, **k) -> None: ...
    def attempt(self, *a, **k) -> None: ...
    def finish(self, *a, **k) -> None: ...
    def close(self) -> None: ...


class Telemetry:
    available = True

    def __init__(self, run_id: str, db_path: Path) -> None:
        self.run_id = run_id
        self.notes: list[str] = []
        self._skip: set[str] = set()
        self._store = deps.TelemetryStore(str(db_path), run_id=run_id)

    def start(self, mission_id: str, *, repo: str, objective: str, language: str = "python") -> None:
        try:
            self._store.start_mission(
                mission_id, task_type=TASK_TYPE, repo=repo, language=language,
                is_hermetic=True, objective=objective,
            )
        except deps.DuplicateMissionError as exc:
            # Recorded and skipped: the run itself is still valid evidence, it
            # simply will not be double-counted in the store.
            self._skip.add(mission_id)
            self.notes.append(f"DUPLICATE_MISSION_SKIPPED: {exc}")

    def attempt(self, mission_id: str, *, latency_ms: float) -> None:
        if mission_id in self._skip:
            return
        try:
            self._store.record_attempt(mission_id, deterministic_steps=1, latency_ms=latency_ms)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            self.notes.append(f"ATTEMPT_NOT_RECORDED: {exc}")

    def finish(self, mission_id: str, *, outcome: str, failure_code: str | None = None) -> None:
        if mission_id in self._skip:
            return
        result, category = OUTCOME_MAP.get(outcome, ("UNKNOWN", "UNKNOWN"))
        try:
            if category is not None:
                self._store.record_failure(
                    mission_id, failure_code=failure_code or outcome, failure_category=category
                )
            self._store.finish_mission(mission_id, result=result, rollback_used=False)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"FINISH_NOT_RECORDED: {exc}")

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:  # noqa: BLE001
            pass


def open_telemetry(run_id: str, db_path: Path | None) -> Any:
    if db_path is None:
        return NullTelemetry("telemetry disabled for this run")
    try:
        return Telemetry(run_id, db_path)
    except Exception as exc:  # noqa: BLE001 - a broken store must not stop work
        return NullTelemetry(f"TELEMETRY_UNAVAILABLE: {type(exc).__name__}: {exc}")
