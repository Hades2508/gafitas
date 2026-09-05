"""F-113: the write scope is safe on its own, and its glob path is tested.

The forensic audit found that `WriteScope.allows()` matched a path out of the
tree it was guarding:

    scope src/**/*.py     matched     src/../etc/passwd.py

It was never reachable. `_resolve` rejects any `..` before the matcher is
consulted, and an end-to-end attack with four escape forms was refused four
times. Defence in depth held.

What these tests fix is where the guarantee lives. `allows()` is public, had no
documented precondition, and the entire glob translation -- `*`, `?`, `[...]`,
lines 78 to 93 -- was covered by no test at all. Its other call sites feed it
paths from git, which never emits `..`: true by accident rather than by
contract, and a future caller that skips `_resolve` would reopen it silently.

The two halves below are deliberate. The escapes must be refused whatever the
pattern says, and the ordinary globbing must keep working -- a scope that
refuses too much is a different way to break a run.
"""

from __future__ import annotations

import pytest

from localprog.scope import WriteScope, escapes_upward


def allows(scope: str, path: str, *, creating: bool = True) -> bool:
    return WriteScope((scope,), ()).allows(path, creating=creating)


# ------------------------------------------------- escapes, whatever the pattern

@pytest.mark.parametrize("path", [
    "src/../etc/passwd.py",
    "src/a/../../x.py",
    "../outside.py",
    "..",
    "a/../../b.py",
])
def test_an_upward_escape_is_never_in_scope(path):
    assert allows("src/**/*.py", path) is False
    assert allows("**", path) is False, "not even the widest pattern"


@pytest.mark.parametrize("path", ["/etc/passwd", "C:/Windows/System32/x.py",
                                  "D:/other/repo/file.py"])
def test_an_absolute_path_is_never_in_scope(path):
    assert allows("**", path) is False


def test_a_backslash_escape_is_normalised_before_the_check():
    assert allows("src/**/*.py", "src\\..\\secret\\keys.py") is False


def test_escapes_upward_is_exported_and_says_what_it_means():
    assert escapes_upward("../x") is True
    assert escapes_upward("a/../b") is True
    assert escapes_upward("/abs") is True
    assert escapes_upward("C:/x") is True
    assert escapes_upward("src/a.py") is False
    assert escapes_upward("") is False


def test_a_filename_containing_dots_is_not_an_escape():
    """Checked on SEGMENTS, not by substring. A file honestly called `a..b.py`
    or `..hidden` is a legitimate name and refusing it would be a different
    kind of breakage."""
    assert escapes_upward("a..b.py") is False
    assert escapes_upward("..hidden.py") is False
    assert allows("*.py", "a..b.py") is True
    assert allows("..hidden.py", "..hidden.py") is True


# --------------------------------------------- the glob path, previously untested

@pytest.mark.parametrize("scope,path,expected", [
    ("src/**/*.py", "src/a.py", True),
    ("src/**/*.py", "src/a/b.py", True),
    ("src/**/*.py", "src/a/b/c/d.py", True),
    ("src/**/*.py", "src/a.txt", False),
    ("src/**/*.py", "other/a.py", False),
    # a single star must not cross a directory separator
    ("src/*.py", "src/a.py", True),
    ("src/*.py", "src/deep/a.py", False),
    ("*.py", "a.py", True),
    ("*.py", "sub/a.py", False),
    # ? is one character, and not a separator
    ("a?.py", "ab.py", True),
    ("a?.py", "abc.py", False),
    ("a?.py", "a/b.py", False),
    # character classes, including negation
    ("a[bc].py", "ab.py", True),
    ("a[bc].py", "ad.py", False),
    ("a[!b].py", "ac.py", True),
    ("a[!b].py", "ab.py", False),
    # an unclosed bracket is a literal, not a crash
    ("a[b.py", "a[b.py", True),
])
def test_glob_translation(scope, path, expected):
    assert allows(scope, path) is expected


def test_a_directory_scope_respects_the_separator():
    assert allows("src/", "src/deep/x.py") is True
    assert allows("src/", "srcevil/x.py") is False, "prefix match must stop at /"


def test_creating_consults_the_new_file_entries_and_writing_does_not():
    scope = WriteScope(("src/**/*.py",), ("ANSWER.txt",))
    assert scope.allows("ANSWER.txt", creating=True) is True
    assert scope.allows("ANSWER.txt", creating=False) is False


def test_an_escape_is_refused_even_for_an_allowed_new_file():
    scope = WriteScope((), ("ANSWER.txt",))
    assert scope.allows("../ANSWER.txt", creating=True) is False
