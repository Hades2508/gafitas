"""Acceptance for DOGFOOD-05: grep must say how many files it looked in.

A real diagnostic gap, seen repeatedly in the traces. When grep returns zero
hits the agent cannot tell these apart:

    the pattern is wrong          (searched 40 files, matched nothing)
    the glob matched no files     (searched 0 files, so of course nothing)

Both come back as "[0 coincidencias para ... en ...]", and models handle the
two very differently -- one means rethink the regex, the other means fix the
glob. Runs have been lost re-running a fine pattern against a glob that never
matched anything.

Written before the fix and proven to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import tools  # noqa: E402


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("pkg/",))


def call(ctx, **kwargs):
    return tools.dispatch(ctx, "grep", kwargs)


def test_a_hit_reports_how_many_files_were_searched(ctx):
    out = call(ctx, pattern="def alpha")
    assert out.ok, out.feedback
    assert out.value["searched"] == 2, "both .py files under the default glob"


def test_zero_hits_over_real_files_reports_the_count(ctx):
    """The pattern is wrong: files were searched and none matched."""
    out = call(ctx, pattern="def gamma")
    assert out.ok
    assert out.value["hits"] == []
    assert out.value["searched"] == 2


def test_a_glob_that_matches_nothing_reports_zero_searched(ctx):
    """The glob is wrong: nothing was searched at all. This is the case the
    agent could not distinguish, and it is a different fix."""
    out = call(ctx, pattern="def alpha", glob="**/*.rs")
    assert out.ok
    assert out.value["hits"] == []
    assert out.value["searched"] == 0


def test_the_note_says_when_nothing_was_searched(ctx):
    """A count in a field is useless if the model only reads the prose."""
    out = call(ctx, pattern="def alpha", glob="**/*.rs")
    assert "0 ficheros" in out.value["note"] or "ningun fichero" in out.value["note"]


def test_a_narrower_glob_reports_a_smaller_count(ctx):
    assert call(ctx, pattern="def", glob="pkg/a.py").value["searched"] == 1


def test_non_python_files_are_searched_when_the_glob_asks(ctx):
    out = call(ctx, pattern="alpha", glob="*.txt")
    assert out.value["searched"] == 1
    assert out.value["hits"], "notes.txt contains alpha"


def test_skipped_directories_are_not_counted(ctx, tmp_path):
    """__pycache__ is never searched, so counting it would overstate the work
    and mislead exactly when the count matters."""
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("def alpha(): pass\n", encoding="utf-8")
    assert call(ctx, pattern="def alpha").value["searched"] == 2


def test_existing_behaviour_is_unchanged(ctx):
    """Adding a field must not disturb what was already there."""
    out = call(ctx, pattern="def alpha")
    hit = out.value["hits"][0]
    assert hit["file"] == "pkg/a.py"
    assert hit["line"] == 1
    assert "def alpha" in hit["text"]
