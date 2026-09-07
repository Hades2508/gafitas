"""A suite that could not RUN is not a model that failed.

F-168, found by an independent adversarial audit of the verdict path.

`verify.SuiteResult.usable` gets this exactly right, and says why in its own
docstring: pytest exit codes 2 through 5 are internal errors, usage errors and
"no tests collected", and

    "None of them mean 'the code is wrong', and treating them as failures is
     how an infrastructure problem gets recorded as a model one."

`verify.check_post` honours that and returns UNKNOWN. Then `work.run_ticket`
collapsed everything that was not DISCRIMINATED into FAIL, with
`scoreable=True` — so a pytest that never started and a model that wrote the
wrong patch landed in the same bucket, and every rate computed downstream
carried the mixture.

The same collapse happened before the loop, where an unrunnable PRE was
reported as NON_DISCRIMINATING under the note "the acceptance already passed
before we touched anything" — a sentence that is false when the acceptance
never ran.

The layer that knew the distinction was careful about it. The layer that
assigned the verdict threw it away. `tester.md` states the rule this broke:
INFRA_ERROR is not FAIL, and conflating them corrupts every metric downstream.
"""

from __future__ import annotations

from localprog import route, verify, work


def test_the_outcome_exists_and_is_distinct():
    assert work.ACCEPTANCE_UNUSABLE not in (work.FAIL, work.NON_DISCRIMINATING)


def test_it_is_never_escalatable():
    """A better model cannot start a pytest that will not start."""
    assert not route.should_escalate(work.ACCEPTANCE_UNUSABLE)
    assert work.ACCEPTANCE_UNUSABLE not in route.ESCALATABLE


def test_it_is_infrastructure_not_a_result():
    assert work.ACCEPTANCE_UNUSABLE in route.INFRASTRUCTURE
    assert work.ACCEPTANCE_UNUSABLE in route.TERMINAL
    assert work.ACCEPTANCE_UNUSABLE not in route.SUCCESSFUL


def test_it_is_not_counted_as_a_success():
    assert work.ACCEPTANCE_UNUSABLE not in route.SUCCESSFUL


def test_routing_explains_it_without_blaming_the_model():
    said = route.explain(work.ACCEPTANCE_UNUSABLE)
    assert "no clasificado" not in said, "it must not fall through"
    assert "modelo" in said


def test_an_unusable_suite_result_is_not_usable():
    """The premise the whole finding rests on, pinned so it cannot drift."""
    timed_out = verify.SuiteResult(False, None, True, "(timeout)", 1.0)
    assert not timed_out.usable

    # exit 5 is "no tests collected": nothing was measured, and that is not the
    # code being wrong.
    nothing_collected = verify.SuiteResult(False, 5, False, "", 0.1)
    assert not nothing_collected.usable

    # exit 1 is real failures. That IS a measurement.
    real_failures = verify.SuiteResult(False, 1, False, "", 0.1)
    assert real_failures.usable


def test_check_post_reports_unknown_when_the_suite_cannot_run(tmp_path):
    """UNKNOWN is the only status that means "we could not measure"."""
    # A PRE that measured something, so the pair is not short-circuited.
    pre = verify.Discrimination(
        verify.DISCRIMINATED,
        verify.SuiteResult(False, 1, False, "", 0.1), None, "pre failed")
    # An empty directory with a named acceptance file that does not exist:
    # pytest collects nothing, exit 4/5, unusable.
    got = verify.check_post(pre, tmp_path, ("tests/test_missing.py",))
    assert got.status == verify.UNKNOWN, (
        "if this stops being UNKNOWN the verdict mapping in work.py is keyed "
        "on the wrong thing and F-168 comes back")
