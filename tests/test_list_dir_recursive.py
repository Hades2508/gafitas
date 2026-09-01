"""Acceptance for DOGFOOD-02: list_dir gains a recursive option.

Written by the auditor before any implementation exists, and proven to fail
against the current tree. Whoever implements it does not get to touch this file.

The task is real work from the backlog. Orienting in an unfamiliar repository
currently costs one list_dir per directory, and the agent has to guess which
directories are worth opening. A bounded recursive listing answers "what is the
shape of this project" in one call, which is the question every run opens with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import errors, tools  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg" / "deep").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "deep" / "inner.py").write_text("Y = 2\n", encoding="utf-8")
    (tmp_path / "tests" / "test_mod.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    return tmp_path


@pytest.fixture
def ctx(repo):
    return tools.ToolContext(root=repo, write_scope=("pkg/",))


def call(ctx, **kwargs):
    return tools.dispatch(ctx, "list_dir", kwargs)


def test_recursive_reaches_nested_files(ctx):
    out = call(ctx, recursive=True)
    assert out.ok, out.feedback
    listed = " ".join(str(v) for v in out.value.values())
    assert "pkg/mod.py" in listed
    assert "pkg/deep/inner.py" in listed
    assert "tests/test_mod.py" in listed


def test_recursive_paths_are_relative_to_the_listed_directory(ctx):
    out = call(ctx, path="pkg", recursive=True)
    assert out.ok
    listed = " ".join(str(v) for v in out.value.values())
    assert "deep/inner.py" in listed
    assert "test_mod.py" not in listed, "must not escape the directory it was given"


def test_recursive_still_skips_noise_directories(ctx):
    out = call(ctx, recursive=True)
    assert out.ok
    assert "__pycache__" not in " ".join(str(v) for v in out.value.values())


def test_recursive_is_off_by_default(ctx):
    """The default must not change: existing callers get one level."""
    out = call(ctx)
    assert out.ok
    listed = " ".join(str(v) for v in out.value.values())
    assert "pkg/deep/inner.py" not in listed
    assert "pkg/" in listed


def test_recursive_false_behaves_exactly_as_before(ctx):
    assert call(ctx, recursive=False).value == call(ctx).value


def test_a_non_boolean_recursive_is_an_invalid_call(ctx):
    out = call(ctx, recursive="yes")
    assert not out.ok and out.invalid_call
    assert out.code == errors.ERROR_BAD_ARGUMENTS


def test_recursive_output_stays_bounded(ctx, repo):
    """A recursive listing multiplies output by the size of the tree, so it
    needs the same ceiling every other tool has -- otherwise it becomes the
    thing that blew the context window in F-25."""
    for i in range(60):
        d = repo / "pkg" / f"sub{i}"
        d.mkdir()
        for j in range(20):
            (d / f"f{j}.py").write_text("Z = 1\n", encoding="utf-8")
    out = call(ctx, recursive=True)
    assert out.ok
    entries = len(out.value.get("dirs", [])) + len(out.value.get("files", []))
    assert entries <= tools.MAX_DIR_ENTRIES
    assert out.value.get("note"), "a truncated listing must say it was truncated"


def test_recursive_is_declared_in_the_tool_schema():
    """I3: schema, argument check and documentation are one source of truth.
    A parameter that works but is not declared is invisible to the model."""
    entry = next(f for f in tools.native_schema() if f["function"]["name"] == "list_dir")
    properties = entry["function"]["parameters"]["properties"]
    assert "recursive" in properties
    assert properties["recursive"].get("description")
    assert "recursive" not in entry["function"]["parameters"]["required"]


def test_recursive_is_in_the_text_manual():
    """Protocol J has no schema channel, so the escalation tier learns about
    tools from the manual. A parameter missing there is invisible to Luna."""
    assert "recursive" in tools.text_manual(("list_dir",))
