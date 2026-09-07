"""Deleting the explanation is not part of fixing the bug.

F-171, found by reading the twenty SWE-bench Verified patches this harness
actually produced rather than by counting them. Three of the thirteen non-empty
ones deleted more documentation than they added, and one of those was the whole
patch:

    django__django-10999   +0 / -7, every removed line a docstring line
                           about parse_duration. No functional change at all.

    sympy__sympy-13877     removed the docstring off the Bareiss determinant
                           algorithm, its TODO and its paper reference included.

    django__django-16569   RESOLVED the issue -- and deleted a docstring and
                           rewrote _("Order") as _('Order') three times on the
                           way past.

No existing signal could see any of it. `signal_public_surface` compares names
and parameter lists, so a function that keeps both and loses its explanation is
"unchanged", and the acceptance suite does not read prose. A patch could strip
a repository of its documentation and be reported as having preserved
everything that mattered.

WHERE IT BITES, AND WHERE IT DOES NOT
-------------------------------------
The conscience only gates a ticket that DECLARES acceptance tests. SWE-bench
tickets deliberately declare none -- a real issue does not arrive with the test
that judges it -- so there the signal is recorded in the evidence and blocks
nothing. It refuses only where this harness is itself the judge, which is the
right division: an external evaluator can decide for itself what it will accept.
"""

from __future__ import annotations

from localprog import verify

WITH_DOC = '''def parse_duration(value):
    """Parse a duration string and return a datetime.timedelta."""
    return value


class Matrix:
    """A matrix."""

    def det(self):
        """Compute the determinant using the Bareiss algorithm."""
        return 0
'''

WITHOUT_DOC = '''def parse_duration(value):
    return value


class Matrix:
    """A matrix."""

    def det(self):
        return 0
'''


def surfaces(source: str):
    return {"m.py": verify.public_surface(source)}


def test_a_surviving_symbol_that_loses_its_docstring_is_a_regression():
    signal = verify.signal_documentation_preserved(
        surfaces(WITH_DOC), surfaces(WITHOUT_DOC))
    assert signal.verdict == verify.REGRESSION
    assert set(signal.data["lost_docstrings"]) == {
        "m.py::parse_duration", "m.py::Matrix.det"}


def test_an_untouched_file_passes():
    signal = verify.signal_documentation_preserved(
        surfaces(WITH_DOC), surfaces(WITH_DOC))
    assert signal.verdict == verify.PASS
    assert signal.data["lost_docstrings"] == []


def test_adding_documentation_is_not_a_regression():
    signal = verify.signal_documentation_preserved(
        surfaces(WITHOUT_DOC), surfaces(WITH_DOC))
    assert signal.verdict == verify.PASS
    assert signal.data["gained_docstrings"] == 2


def test_deleting_the_whole_function_is_not_reported_here():
    """That is signal_public_surface's finding, and one act is not two."""
    gone = "def otra():\n    return 1\n"
    signal = verify.signal_documentation_preserved(
        surfaces(WITH_DOC), surfaces(gone))
    assert signal.verdict == verify.PASS, (
        "a removed symbol is a removal, and counting it here as well would "
        "inflate one act into two findings")


def test_the_public_surface_signal_still_catches_the_removal():
    """The other half, so the pair covers the ground between them."""
    gone = "def otra():\n    return 1\n"
    signal = verify.signal_public_surface(surfaces(WITH_DOC), surfaces(gone))
    assert signal.verdict != verify.PASS


# ---------------------------------------------------------------- F-171b ----

def test_losing_a_default_breaks_callers_and_is_caught():
    """`def f(x=1)` -> `def f(x)` keeps every name, so the prefix test passed.

    Every existing `f()` call site now raises TypeError. It was the one shape
    of break this signal could not see.
    """
    before = surfaces("def f(x=1):\n    return x\n")
    after = surfaces("def f(x):\n    return x\n")
    signal = verify.signal_public_surface(before, after)
    assert signal.verdict != verify.PASS


def test_gaining_a_default_is_fine():
    before = surfaces("def f(x):\n    return x\n")
    after = surfaces("def f(x=1):\n    return x\n")
    assert verify.signal_public_surface(before, after).verdict == verify.PASS


def test_an_added_optional_parameter_is_still_fine():
    """The case the prefix rule exists for; pinned so the fix cannot break it."""
    before = surfaces("def f(x):\n    return x\n")
    after = surfaces("def f(x, y=2):\n    return x\n")
    assert verify.signal_public_surface(before, after).verdict == verify.PASS
