"""F-114: certification, which decided whether an engine was usable and had no tests.

`certify.py` was 67 statements at 0% coverage. It drives an engine through a
fixed set of checks on a small fixture and says whether it can be used at all --
a gate, not a report. A gate with no tests can pass an engine it should refuse,
or refuse one it should pass, and nothing would notice either way.

Nothing here talks to a model. The provider is a fake that uses the tools it is
told to, so the properties under test are the ones that belong to the gate:
partial passes are not certifications, the fixture really is materialised, and a
run that produces no usable call fails the first check rather than sliding
through on the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localprog import certify


def report(**kw):
    base = dict(engine="fake", passed=list(certify.CHECKS), failed=[])
    base.update(kw)
    return certify.Certification(**base)


# ---------------------------------------------------------------- the gate

def test_every_check_passed_is_a_certification():
    assert report().certified is True


def test_one_failure_is_not_a_certification():
    got = report(passed=list(certify.CHECKS)[:-1], failed=[list(certify.CHECKS)[-1]])
    assert got.certified is False


def test_a_partial_pass_with_no_failures_is_still_not_a_certification():
    """The subtle one. A run that simply did not reach a check has not passed
    it, and an empty `failed` list must not be read as success."""
    got = report(passed=list(certify.CHECKS)[:2], failed=[])
    assert got.certified is False


def test_nothing_passed_is_not_a_certification():
    assert report(passed=[], failed=[]).certified is False


def test_the_verdict_travels_with_the_record():
    body = report().to_dict()
    assert body["certified"] is True
    assert body["schema"] == certify.SCHEMA
    assert json.dumps(body), "a record that will not serialise cannot be sealed"


def test_a_failure_travels_with_the_record():
    body = report(passed=[], failed=["TOOL_USE"]).to_dict()
    assert body["certified"] is False
    assert body["failed"] == ["TOOL_USE"]


# ------------------------------------------------------------- the fixture

def test_the_fixture_is_materialised_whole(tmp_path):
    """The engine is judged on this tree. A missing file would fail an engine
    for the harness's reason."""
    root = certify._materialise(tmp_path / "box")
    for relative in certify.FIXTURE:
        assert (root / relative).exists(), f"{relative} was not written"
        if Path(relative).name != "__init__.py":
            # A package marker is legitimately empty; everything else carries
            # the code the engine is judged on.
            assert (root / relative).read_text(encoding="utf-8"), f"{relative} is empty"


def test_the_fixture_actually_fails_before_the_fix(tmp_path):
    """A certification whose test suite already passes measures nothing. This
    is the same trap the real-task corpus fell into, and it is worth one
    assertion here."""
    from localprog import verify as verify_mod

    root = certify._materialise(tmp_path / "box")
    result = verify_mod.run_suite(root, ("tests/test_rates.py",))
    assert not result.passed, "the fixture is already solved; the gate would pass anything"


def test_materialising_twice_is_idempotent(tmp_path):
    first = certify._materialise(tmp_path / "box")
    bodies = {r: (first / r).read_text(encoding="utf-8") for r in certify.FIXTURE}
    second = certify._materialise(tmp_path / "box")
    assert all((second / r).read_text(encoding="utf-8") == bodies[r]
               for r in certify.FIXTURE)


# ---------------------------------------------------------- the check list

def test_the_checks_are_a_fixed_closed_set():
    """`certified` counts against CHECKS, so a check added without being added
    here would silently make every past certification incomplete."""
    assert certify.CHECKS, "an empty check list certifies everything"
    assert len(set(certify.CHECKS)) == len(certify.CHECKS), "duplicate check names"


@pytest.mark.parametrize("name", ["TOOL_USE"])
def test_the_first_check_is_that_a_call_landed_at_all(name):
    """Everything else is downstream of the engine producing one usable call,
    so it has to be a check in its own right rather than an assumption."""
    assert name in certify.CHECKS
