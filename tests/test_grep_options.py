"""Acceptance for the grep case-insensitivity ticket (DOGFOOD-01).

Written BEFORE the implementation, by the auditor, and proven to fail against
the current tree. Whoever implements it -- Claude, a local model, Luna -- does
not get to touch this file.

The task itself is real. ``grep`` is how the agent finds code it has not read,
and searching for ``ValueError`` when the source says ``valueerror`` currently
returns nothing at all, with no hint that case was the problem. An agent that
gets zero hits concludes the symbol does not exist and goes looking somewhere
else, which is the most expensive kind of wrong answer a search tool can give.
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
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(
        "class Widget:\n"
        "    def Build(self):\n"
        "        raise ValueError('nope')\n"
        "\n"
        "\n"
        "def build_widget():\n"
        "    return Widget()\n",
        encoding="utf-8",
    )
    return tools.ToolContext(root=tmp_path, write_scope=("pkg/",))


def call(ctx, **kwargs):
    return tools.dispatch(ctx, "grep", kwargs)


def test_ignore_case_finds_a_differently_cased_match(ctx):
    out = call(ctx, pattern="widget", ignore_case=True)
    assert out.ok, out.feedback
    files = {h["file"] for h in out.value["hits"]}
    assert "pkg/mod.py" in files
    assert any("Widget" in h["text"] for h in out.value["hits"])


def test_ignore_case_is_off_by_default(ctx):
    """The default must not change. A search that silently became fuzzy would
    make every existing precise query noisier."""
    out = call(ctx, pattern="widget")
    assert out.ok
    assert all("Widget" not in h["text"] or "widget" in h["text"]
               for h in out.value["hits"])
    assert not any(h["text"].strip().startswith("class Widget") for h in out.value["hits"])


def test_ignore_case_false_behaves_exactly_as_before(ctx):
    explicit = call(ctx, pattern="Widget", ignore_case=False)
    implicit = call(ctx, pattern="Widget")
    assert explicit.ok and implicit.ok
    assert explicit.value == implicit.value


def test_ignore_case_still_returns_line_numbers_and_files(ctx):
    out = call(ctx, pattern="BUILD", ignore_case=True)
    assert out.ok
    assert out.value["hits"], "should have found Build and build_widget"
    for hit in out.value["hits"]:
        assert isinstance(hit["line"], int) and hit["line"] > 0
        assert hit["file"] == "pkg/mod.py"


def test_a_non_boolean_ignore_case_is_an_invalid_call(ctx):
    out = call(ctx, pattern="widget", ignore_case="yes")
    assert not out.ok and out.invalid_call


def test_ignore_case_is_declared_in_the_tool_schema(ctx):
    """I3: the schema, the argument check and the documentation are one source
    of truth. A parameter that works but is not declared is invisible to the
    model, which makes it useless."""
    entry = next(f for f in tools.native_schema() if f["function"]["name"] == "grep")
    properties = entry["function"]["parameters"]["properties"]
    assert "ignore_case" in properties
    assert properties["ignore_case"].get("description")
    assert "ignore_case" not in entry["function"]["parameters"]["required"]
