"""Acceptance for dogfood09. Frozen BEFORE the agent runs, and not edited after.

grep says how many files it looked in and never how many it found something in,
which is the number that tells you whether a symbol lives in one place or is
smeared across twenty.
"""

from __future__ import annotations

import pytest

from localprog import tools


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def alpha():\n    return 'x'\n\n\ndef alpha_again():\n    return 'x'\n",
        encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("def beta():\n    return 'x'\n",
                                           encoding="utf-8")
    (tmp_path / "pkg" / "c.py").write_text("def gamma():\n    return 'y'\n",
                                           encoding="utf-8")
    return tmp_path


@pytest.fixture()
def ctx(repo):
    return tools.ToolContext(root=repo)


def grep(ctx, **kwargs):
    return tools.dispatch(ctx, "grep", kwargs)


def test_it_counts_distinct_files_not_hits(ctx):
    """Three hits for 'x', in two files."""
    out = grep(ctx, pattern="return 'x'")
    assert out.ok, out.feedback
    assert len(out.value["hits"]) == 3
    assert out.value["matched_files"] == 2


def test_one_file_is_one_file(ctx):
    out = grep(ctx, pattern="gamma")
    assert out.ok
    assert out.value["matched_files"] == 1


def test_nothing_found_is_zero_not_missing(ctx):
    out = grep(ctx, pattern="quaternion")
    assert out.ok
    assert out.value["matched_files"] == 0, (
        "the key has to be there when the answer is zero, or a caller has to "
        "special-case exactly the result it most wants to reason about"
    )


def test_the_keys_that_were_already_there_are_untouched(ctx):
    out = grep(ctx, pattern="return 'x'")
    assert set(out.value) >= {"hits", "searched", "matched_files"}
    assert out.value["searched"] == 3
    assert {h["file"] for h in out.value["hits"]} == {"pkg/a.py", "pkg/b.py"}


def test_a_narrower_glob_narrows_the_count(ctx):
    out = grep(ctx, pattern="return 'x'", glob="pkg/a.py")
    assert out.ok
    assert out.value["matched_files"] == 1
    assert out.value["searched"] == 1
