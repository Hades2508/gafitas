"""End-to-end regressions for the productive path (F-01, F-09, F-11).

These run the whole ``run_ticket`` pipeline against real repositories on disk,
with a scripted provider standing in for the model. That combination is the
point: the workspace, pytest, the filesystem, git and the conscience are all
genuinely exercised, while the model's answers are fixed so a failure here is
always the harness's and never a model's bad day.

The headline test is ``test_a_solved_ticket_is_refused_before_the_model_runs``.
It encodes the finding that voided every capability measurement this project
has ever taken.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import verify, work  # noqa: E402
from localprog.provider import FakeProvider  # noqa: E402

BROKEN = "def halve(n):\n    return n / 0\n"
FIXED = "def halve(n):\n    return n / 2\n"
TEST = (
    "from pkg.calc import halve\n"
    "\n"
    "\n"
    "def test_halve():\n"
    "    assert halve(10) == 5\n"
)
OTHER_SRC = "def triple(n):\n    return n * 3\n"
OTHER_TEST = (
    "from pkg.other import triple\n"
    "\n"
    "\n"
    "def test_triple():\n"
    "    assert triple(2) == 6\n"
)


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A small git repository whose acceptance test genuinely fails."""
    root = tmp_path / "demo"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "calc.py").write_text(BROKEN, encoding="utf-8")
    (root / "pkg" / "other.py").write_text(OTHER_SRC, encoding="utf-8")
    (root / "tests" / "test_calc.py").write_text(TEST, encoding="utf-8")
    (root / "tests" / "test_other.py").write_text(OTHER_TEST, encoding="utf-8")
    git(["init", "-q"], root)
    git(["config", "user.email", "t@t"], root)
    git(["config", "user.name", "t"], root)
    git(["add", "-A"], root)
    git(["commit", "-qm", "init"], root)
    return root


def ticket(repo, **overrides):
    base = dict(
        ticket_id="T1",
        repo=repo,
        objective="Arregla halve() para que devuelva la mitad.",
        write_scope=("pkg/",),
        acceptance_tests=("tests/test_calc.py",),
        allowed_new_files=(),
        full_suite=True,
        max_turns=8,
    )
    base.update(overrides)
    return work.Ticket(**base)


def tc(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def scripted(*steps):
    return lambda _model: FakeProvider(list(steps))


# ------------------------------------------------------------------ F-01


def test_a_solved_ticket_is_refused_before_the_model_runs(repo, tmp_path):
    """The finding that voided STEP1, pinned.

    All five STEP1 mission repos were snapshotted in their solved state, so the
    acceptance suite passed before the agent was invited to do anything. The
    report says 5/5 tester_pass; every run in it is BUDGET_EXHAUSTED. Here the
    same situation is detected, the model is never called, and the outcome is
    explicitly not a model result.
    """
    (repo / "pkg" / "calc.py").write_text(FIXED, encoding="utf-8")
    git(["commit", "-qam", "already fixed"], repo)

    provider = FakeProvider([tc("finish", summary="x")])
    result = work.run_ticket(
        ticket(repo), "fake", provider_factory=lambda _m: provider,
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )

    assert result.outcome == work.NON_DISCRIMINATING
    assert result.scoreable is False
    assert provider.calls == [], "the model must not be invoked on a solved ticket"
    assert result.discrimination["status"] == verify.NON_DISCRIMINATING
    assert result.discrimination["pre"]["passed"] is True


def test_a_real_ticket_is_measurable(repo, tmp_path):
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("run_tests"),
            tc("finish", summary="arreglado", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.outcome == work.PASS
    assert result.scoreable is True
    assert result.discrimination["status"] == verify.DISCRIMINATED
    assert result.discrimination["pre"]["passed"] is False
    assert result.discrimination["post"]["passed"] is True
    assert result.changed_files == ["pkg/calc.py"]


def test_an_honest_miss_is_a_fail_not_a_pass(repo, tmp_path):
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 3"),
            tc("finish", summary="creo que ya", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.outcome == work.FAIL
    assert result.finish_status == "DONE"  # the claim is recorded...
    # ...and does not decide anything. I9.


# ------------------------------------------------------------------ F-09


def test_rewriting_the_acceptance_test_cannot_buy_a_pass(repo, tmp_path):
    """The most direct way to fake a green suite, refused.

    The agent is given write access to the tests directory and uses it to
    delete the assertion instead of fixing the code. pytest goes green. The
    conscience does not.
    """
    result = work.run_ticket(
        ticket(repo, write_scope=("pkg/", "tests/")), "fake",
        provider_factory=scripted(
            tc("edit", path="tests/test_calc.py",
               old="assert halve(10) == 5", new="assert True"),
            tc("run_tests"),
            tc("finish", summary="verde", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.discrimination["post"]["passed"] is True, "pytest really did go green"
    assert result.outcome == work.BLOCKED_BY_CONSCIENCE
    signals = {s["name"]: s for s in result.conscience["signals"]}
    assert signals["acceptance_untouched"]["verdict"] == verify.REGRESSION


def test_breaking_an_unrelated_test_is_caught_by_the_whole_suite(repo, tmp_path):
    """The failure mode a targeted acceptance suite structurally cannot see.

    The agent fixes what it was asked to fix and breaks something nobody
    mentioned. The acceptance suite passes -- it never looked at pkg/other.py.
    """
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("edit", path="pkg/other.py", old="n * 3", new="n * 4"),
            tc("run_tests"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.discrimination["post"]["passed"] is True
    assert result.outcome == work.BLOCKED_BY_CONSCIENCE
    signals = {s["name"]: s for s in result.conscience["signals"]}
    collateral = signals["no_collateral_regression"]
    assert collateral["verdict"] == verify.REGRESSION
    assert any("test_triple" in n for n in collateral["data"]["broke"])


def test_deleting_a_public_function_is_flagged_even_when_tests_pass(repo, tmp_path):
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("write_file", path="pkg/extra.py", content="def kept():\n    return 1\n"),
            tc("run_tests"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    # Adding a public function is EXPECTED_CHANGE and must not block.
    signals = {s["name"]: s for s in result.conscience["signals"]}
    assert signals["public_surface_preserved"]["verdict"] == verify.PASS
    assert result.outcome == work.PASS


def test_removing_a_public_function_blocks(repo, tmp_path):
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old=BROKEN, new=FIXED + "\n"),
            tc("edit", path="pkg/other.py", old=OTHER_SRC, new="TRIPLE = 3\n"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    signals = {s["name"]: s for s in result.conscience["signals"]}
    assert "pkg/other.py::triple" in signals["public_surface_preserved"]["data"]["removed"]
    assert result.outcome == work.BLOCKED_BY_CONSCIENCE


def test_a_blocked_agent_with_clean_evidence_is_unconfirmed_not_refused(repo, tmp_path):
    """F-24, and the run that earned it.

    ga06 built the module it was asked for, took its acceptance suite from a
    collection error to 22 passed, broke none of the 11 tests that already
    passed, and stayed in scope -- and was refused, because it had called
    finish(status='BLOCKED'), unsure it had succeeded.

    I9 says the harness owns the verdict and the model's claim can never create
    a PASS. Letting the same claim destroy one puts the model back in the
    judge's chair with the sign reversed. The claim is reported instead, and the
    work keeps the outcome its evidence supports -- with a flag on it.
    """
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="no estoy seguro de haberlo resuelto", status="BLOCKED"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.finish_status == "BLOCKED"
    assert result.outcome == work.PASS_UNCONFIRMED
    assert result.agent_report["data"]["finish_status"] == "BLOCKED"
    assert any("no lo confirmo" in n for n in result.notes)


def test_the_agent_report_is_not_a_conscience_signal(repo, tmp_path):
    """The boundary itself: the conscience holds deterministic evidence about
    the CODE. What the agent believes is metadata about the run."""
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="dudo", status="BLOCKED"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    names = {s["name"] for s in result.conscience["signals"]}
    assert "agent_not_blocked" not in names
    assert result.conscience["verdict"] == verify.PASS


def test_a_blocked_agent_whose_work_is_actually_broken_still_fails(repo, tmp_path):
    """PASS_UNCONFIRMED must not become a way through for work that is wrong.
    The evidence still decides; the flag only records the agent's doubt."""
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 7"),
            tc("finish", summary="no puedo", status="BLOCKED"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.outcome == work.FAIL


def test_a_conscience_refusal_still_beats_a_confident_agent(repo, tmp_path):
    """And the other direction: DONE does not buy anything either."""
    result = work.run_ticket(
        ticket(repo, write_scope=("pkg/", "tests/")), "fake",
        provider_factory=scripted(
            tc("edit", path="tests/test_calc.py",
               old="assert halve(10) == 5", new="assert True"),
            tc("finish", summary="verde", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.finish_status == "DONE"
    assert result.outcome == work.BLOCKED_BY_CONSCIENCE


def test_the_conscience_can_never_manufacture_a_pass():
    """The invariant the whole module rests on: refuse-only.

    Combining verdicts always yields the worst one present, so no arrangement
    of signals can turn a REGRESSION into a PASS. If this ever stops holding,
    every green verdict downstream becomes worthless.
    """
    assert verify.combine([verify.PASS, verify.REGRESSION]) == verify.REGRESSION
    assert verify.combine([verify.PASS, verify.INCONCLUSIVE]) == verify.INCONCLUSIVE
    assert verify.combine([verify.REGRESSION, verify.INFRA_ERROR]) == verify.INFRA_ERROR
    assert verify.combine([]) == verify.INCONCLUSIVE  # nothing checked proves nothing
    assert verify.combine([verify.PASS, verify.PASS]) == verify.PASS


# ------------------------------------------------------------------ F-11


def test_provider_failure_is_never_scored_against_the_model(repo, tmp_path):
    from localprog.errors import ProviderError

    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=lambda _m: FakeProvider([ProviderError("HTTP_STATUS", "500")]),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.outcome == work.PROVIDER_ERROR
    assert result.scoreable is False
    assert result.provider_error["kind"] == "HTTP_STATUS"


def test_the_source_repository_is_never_modified(repo, tmp_path):
    before = (repo / "pkg" / "calc.py").read_text(encoding="utf-8")
    work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert (repo / "pkg" / "calc.py").read_text(encoding="utf-8") == before


def test_a_pass_seals_a_patch_a_human_can_read(repo, tmp_path):
    out = tmp_path / "out"
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=out, base_dir=tmp_path,
    )
    assert result.patch_path and Path(result.patch_path).exists()
    patch = Path(result.patch_path).read_text(encoding="utf-8")
    assert "pkg/calc.py" in patch and "n / 2" in patch

    sealed = json.loads((out / "tickets" / "T1_fake.json").read_text(encoding="utf-8"))
    assert sealed["record"]["outcome"] == work.PASS
    assert sealed["provenance"]["guard_file"]
    assert sealed["events"], "the turn-by-turn trace must be preserved"


def test_a_new_file_can_be_created_inside_a_directory_scope(repo, tmp_path):
    """F-05 through the whole pipeline: the module nobody named in advance."""
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("write_file", path="pkg/helper.py", content="def helper():\n    return 2\n"),
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.outcome == work.PASS
    assert "pkg/helper.py" in result.changed_files


def test_usage_is_accounted_for(repo, tmp_path):
    result = work.run_ticket(
        ticket(repo), "fake",
        provider_factory=scripted(
            tc("edit", path="pkg/calc.py", old="n / 0", new="n / 2"),
            tc("finish", summary="hecho", status="DONE"),
        ),
        out_dir=tmp_path / "out", base_dir=tmp_path,
    )
    assert result.usage["calls"] == 2
    assert result.model_class == "FAKE"


def test_a_malformed_ticket_is_a_harness_defect_not_a_model_failure(tmp_path):
    from localprog.errors import HarnessInvalid

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"ticket_id": "x", "repo": str(tmp_path)}), encoding="utf-8")
    with pytest.raises(HarnessInvalid) as excinfo:
        work.load_ticket(bad)
    assert "objective" in excinfo.value.detail
