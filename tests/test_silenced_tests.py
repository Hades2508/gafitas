"""Marking a broken test is not fixing it, and the signal has to notice.

F-169, from an independent adversarial audit. `signal_no_collateral_regression`
exists to catch the failure a targeted acceptance suite structurally cannot:
satisfying the sample while damaging the rest of the repository. It caught

    passed -> failed / error      (broke)
    passed -> not collected       (disappeared)

and missed the cleanest cheat there is:

    passed -> xfail

The test is still collected, so `disappeared` does not see it, and it does not
fail, so `broke` does not either. Reproduced before fixing -- two green tests,
one then given @pytest.mark.xfail with its body changed to `assert False`:

    verdict PASS
    detail  "los 2 tests que pasaban antes siguen pasando"
    data    pre_passed=2, post_passed=1

The signal contradicted its own data inside the same object.

`skipped` happened to be caught already, but by accident rather than by
design: pytest's SKIPPED summary line carries a different shape, the node id
does not match, and the test fell into `disappeared`. Relying on the format of
someone else's summary line is not a check.
"""

from __future__ import annotations

import pytest

from localprog import verify

TWO_GREEN = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"

XFAILED = ("import pytest\n\n\ndef test_a():\n    assert True\n\n\n"
           "@pytest.mark.xfail\ndef test_b():\n    assert False\n")

SKIPPED = ("import pytest\n\n\ndef test_a():\n    assert True\n\n\n"
           '@pytest.mark.skip(reason="flaky")\ndef test_b():\n    assert False\n')

BROKEN = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert False\n"


def run_pair(tmp_path, after_source: str):
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests" / "test_cosas.py"
    target.write_text(TWO_GREEN, encoding="utf-8")
    before = verify.run_suite(tmp_path)
    target.write_text(after_source, encoding="utf-8")
    after = verify.run_suite(tmp_path)
    return before, after


def test_the_baseline_really_is_two_green(tmp_path):
    """The premise. Without it nothing below means anything."""
    before, _ = run_pair(tmp_path, TWO_GREEN)
    assert sorted(before.outcomes.values()) == ["passed", "passed"]


def test_xfailing_a_broken_test_is_a_regression(tmp_path):
    before, after = run_pair(tmp_path, XFAILED)
    signal = verify.signal_no_collateral_regression(before, after, acceptance=())
    assert signal.verdict == verify.REGRESSION, (
        "this returned PASS, with the sentence 'the 2 tests that passed before "
        "still pass', while carrying post_passed=1")
    assert signal.data["silenced"] == ["tests/test_cosas.py::test_b"]


def test_skipping_a_broken_test_is_not_a_pass(tmp_path):
    before, after = run_pair(tmp_path, SKIPPED)
    signal = verify.signal_no_collateral_regression(before, after, acceptance=())
    assert signal.verdict != verify.PASS


def test_an_honestly_broken_test_is_still_a_regression(tmp_path):
    """The case that always worked, pinned so the new branch cannot shadow it."""
    before, after = run_pair(tmp_path, BROKEN)
    signal = verify.signal_no_collateral_regression(before, after, acceptance=())
    assert signal.verdict == verify.REGRESSION
    assert signal.data["broke"] == ["tests/test_cosas.py::test_b"]


def test_an_untouched_suite_still_passes(tmp_path):
    """The fix must not make every run look like a regression."""
    before, after = run_pair(tmp_path, TWO_GREEN)
    signal = verify.signal_no_collateral_regression(before, after, acceptance=())
    assert signal.verdict == verify.PASS
    assert signal.data["silenced"] == []


def test_the_signal_never_claims_preservation_against_its_own_counts(tmp_path):
    """The guard for the NEXT gap of this shape, whatever it turns out to be.

    Every specific check can pass while fewer tests are green than were green
    before. Claiming preserved behaviour there is the signal contradicting its
    own data, which is exactly what it did for xfail before xfail was named.
    """
    before, after = run_pair(tmp_path, XFAILED)
    signal = verify.signal_no_collateral_regression(before, after, acceptance=())
    if signal.verdict == verify.PASS:
        assert signal.data["post_passed"] >= signal.data["pre_passed"], (
            "PASS while post_passed < pre_passed is a false claim about the "
            "repository, whichever branch produced it")
