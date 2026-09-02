"""The SWE-bench adapter, and above all the anti-contamination guarantee.

A leaking adapter does not fail loudly. It produces a good score and a
worthless measurement, and nobody notices until someone reruns it. So the
leak tests here are the important ones and they run on shapes taken from the
real dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import swebench  # noqa: E402

ROW = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "base_commit": "d26b2424437dabeeca94d7900b37d2df4410da0c",
    "problem_statement": (
        "UsernameValidator allows trailing newline in usernames\n"
        "Description\n"
        "ASCIIUsernameValidator and UnicodeUsernameValidator use the regex "
        r"r'^[\w.@+-]+$'. The problem is that \A and \Z should be used instead."
    ),
    "patch": (
        "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n"
        "--- a/django/contrib/auth/validators.py\n"
        "+++ b/django/contrib/auth/validators.py\n"
        "@@ -7,7 +7,7 @@\n"
        "-    regex = r'^[\\w.@+-]+$'\n"
        "+    regex = r'\\A[\\w.@+-]+\\Z'\n"
    ),
    "test_patch": (
        "diff --git a/tests/auth_tests/test_validators.py b/tests/auth_tests/test_validators.py\n"
        "+    def test_ascii_validator_rejects_trailing_newline(self):\n"
        "+        with self.assertRaises(ValidationError):\n"
        "+            v('trailing_newline\\n')\n"
    ),
    "FAIL_TO_PASS": '["test_ascii_validator_rejects_trailing_newline"]',
    "PASS_TO_PASS": '["test_help_text", "test_username_validators"]',
    "hints_text": "The fix is to replace ^ and $ with \\A and \\Z in the regex.",
    "difficulty": "<15 min fix",
}


@pytest.fixture
def ticket(tmp_path):
    instance = swebench.Instance.from_row(ROW)
    return instance, swebench.build_ticket(instance, tmp_path)


# ------------------------------------------------------- anti-contamination


def test_the_ticket_carries_the_problem_statement(ticket):
    _, tkt = ticket
    assert "UsernameValidator allows trailing newline" in tkt.objective


@pytest.mark.parametrize("forbidden", swebench.FORBIDDEN_FIELDS)
def test_no_forbidden_field_reaches_the_ticket(ticket, forbidden):
    """The answer, the tests that judge it, their names, and the discussion
    that contains the fix. Each excluded for its own reason, each checked."""
    _, tkt = ticket
    swebench.assert_no_leak(ROW, tkt)
    assert forbidden in swebench.FORBIDDEN_FIELDS


def test_the_gold_patch_is_not_in_the_objective(ticket):
    _, tkt = ticket
    assert "\\A[\\w.@+-]+\\Z" not in tkt.objective
    assert "validators.py" not in tkt.objective


def test_the_test_names_are_not_in_the_objective(ticket):
    """FAIL_TO_PASS names alone point straight at the answer."""
    _, tkt = ticket
    assert "test_ascii_validator_rejects_trailing_newline" not in tkt.objective


def test_the_hints_are_not_in_the_objective(ticket):
    _, tkt = ticket
    assert "replace ^ and $" not in tkt.objective


def test_a_leak_is_detected_rather_than_ignored(tmp_path):
    """The guard has to actually fire, or it is decoration."""
    instance = swebench.Instance.from_row(ROW)
    tkt = swebench.build_ticket(instance, tmp_path)
    leaked = type(tkt)(
        ticket_id=tkt.ticket_id, repo=tkt.repo,
        objective=tkt.objective + "\n" + ROW["hints_text"],
        write_scope=tkt.write_scope, acceptance_tests=(),
    )
    with pytest.raises(ValueError, match="LEAK"):
        swebench.assert_no_leak(ROW, leaked)


def test_only_the_visible_fields_are_read(tmp_path):
    """A row containing ONLY the four permitted fields must still build. If the
    adapter ever starts depending on a fifth, this fails rather than the
    dependency going unnoticed."""
    minimal = {k: ROW[k] for k in swebench.VISIBLE_FIELDS}
    instance = swebench.Instance.from_row(minimal)
    tkt = swebench.build_ticket(instance, tmp_path)
    assert tkt.ticket_id == "django__django-11099"


# ------------------------------------------------------------- ticket shape


def test_acceptance_is_empty_because_the_benchmark_hides_the_tests(ticket):
    """Not an oversight. SWE-bench withholds the tests by design, so the
    discrimination gate has nothing to gate on and GAFITAS runs here with its
    main verification organ removed. Recorded, not papered over."""
    _, tkt = ticket
    assert tkt.acceptance_tests == ()
    assert tkt.full_suite is False


def test_the_write_scope_is_the_whole_checkout(ticket):
    """A real issue does not arrive with a list of files to change, and
    narrowing the scope would hand over part of the answer. Containment is
    unaffected -- guard still refuses anything outside the workspace."""
    _, tkt = ticket
    assert tkt.scope.allows("django/contrib/auth/validators.py", creating=False)
    assert tkt.scope.allows("anything/at/all.py", creating=True)


def test_the_objective_tells_the_agent_there_is_no_declared_test(ticket):
    _, tkt = ticket
    assert "No hay tests de aceptacion declarados" in tkt.objective


def test_a_row_missing_a_visible_field_is_rejected():
    broken = {k: v for k, v in ROW.items() if k != "base_commit"}
    with pytest.raises(ValueError, match="base_commit"):
        swebench.Instance.from_row(broken)


# --------------------------------------------------------------- prediction


def test_a_prediction_carries_exactly_the_three_official_fields():
    p = swebench.Prediction(
        instance_id="x", model_patch="diff", model_name_or_path="gafitas",
        internal={"turns": 12},
    )
    assert set(p.official()) == {"instance_id", "model_patch", "model_name_or_path"}
    assert "internal" not in p.official(), "our telemetry never enters the score"


def test_predictions_are_written_in_the_official_shape(tmp_path):
    out = tmp_path / "predictions.json"
    swebench.write_predictions(out, [
        swebench.Prediction("a", "diff-a", "gafitas-local-qwen3-4b"),
        swebench.Prediction("b", "", "gafitas-local-qwen3-4b"),
    ])
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["instance_id"] for r in rows] == ["a", "b"]
    assert all(set(r) == {"instance_id", "model_patch", "model_name_or_path"} for r in rows)


def test_extract_patch_returns_a_unified_diff(tmp_path):
    repo = tmp_path / "r"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "m.py").write_text("X = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=str(repo), capture_output=True)

    (repo / "pkg" / "m.py").write_text("X = 2\n", encoding="utf-8")
    patch = swebench.extract_patch(repo)
    assert "pkg/m.py" in patch and "-X = 1" in patch and "+X = 2" in patch


def test_extract_patch_includes_files_the_agent_created(tmp_path):
    """A new module must appear in the diff, or the fix is invisible to the
    harness even though it exists on disk."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("A = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=str(repo), capture_output=True)

    (repo / "nuevo.py").write_text("B = 2\n", encoding="utf-8")
    assert "nuevo.py" in swebench.extract_patch(repo)


def test_extract_patch_on_an_untouched_repo_is_empty(tmp_path):
    """An agent that changed nothing must produce an empty patch, not a
    spurious one -- the harness would score a spurious diff as a real attempt."""
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("A = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=str(repo), capture_output=True)
    assert swebench.extract_patch(repo).strip() == ""
