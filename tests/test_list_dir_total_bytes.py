"""Acceptance for dogfood07. Frozen BEFORE the agent runs, and not edited after.

The directory listing reports a size per file and no total, so a caller has to
add them up -- and when the listing is truncated it cannot, because the sizes it
would need are the ones that were cut.
"""

from __future__ import annotations

import pytest

from localprog import tools


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x" * 100, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("y" * 250, encoding="utf-8")
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "c.py").write_text("z" * 9000, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def ctx(repo):
    return tools.ToolContext(root=repo)


def test_the_listing_reports_the_total_size_of_its_files(ctx):
    out = tools.dispatch(ctx, "list_dir", {"path": "pkg"})
    assert out.ok, out.feedback
    assert out.value["total_bytes"] == 350, out.value


def test_subdirectories_do_not_count_towards_the_total(ctx):
    """9000 bytes live under pkg/sub and must not appear in pkg's own total."""
    out = tools.dispatch(ctx, "list_dir", {"path": "pkg"})
    assert out.value["total_bytes"] == 350


def test_a_recursive_listing_counts_everything_it_lists(ctx):
    out = tools.dispatch(ctx, "list_dir", {"path": "pkg", "recursive": True})
    assert out.ok, out.feedback
    assert out.value["total_bytes"] == 9350, out.value


def test_the_keys_that_were_already_there_are_untouched(ctx):
    out = tools.dispatch(ctx, "list_dir", {"path": "pkg"})
    assert set(out.value) >= {"path", "dirs", "files"}
    assert out.value["path"] == "pkg"
    assert [f["name"] for f in out.value["files"]] == ["a.py", "b.py"]
    assert out.value["dirs"] == ["sub/"]


def test_the_truncation_note_still_counts_what_it_showed(ctx, repo):
    """Added after an audit, and for a reason worth writing down.

    The first candidate passed every case above and, in passing, changed the
    count of entries actually shown from dirs+files to dirs alone -- which
    makes the listing announce ERROR_TOO_MANY_ENTRIES on any directory that
    contains a file. Nothing in the acceptance covered that line and nothing in
    the suite did either, so a real regression scored PASS. This is the hole,
    closed.
    """
    for i in range(5):
        (repo / "pkg" / f"extra{i}.py").write_text("k", encoding="utf-8")
    out = tools.dispatch(ctx, "list_dir", {"path": "pkg"})
    assert out.ok
    assert "note" not in out.value or "TOO_MANY" not in out.value["note"], (
        "nothing was truncated here, so nothing may claim it was"
    )


def test_the_total_survives_a_truncated_listing(tmp_path):
    """The case a caller cannot work around: the sizes it would have to add up
    are exactly the ones that were cut out of the response."""
    big = tmp_path / "many"
    big.mkdir()
    for i in range(tools.MAX_DIR_ENTRIES + 20):
        (big / f"f{i:04}.txt").write_text("q" * 10, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "list_dir", {"path": "many"})
    assert out.ok
    assert len(out.value["files"]) < tools.MAX_DIR_ENTRIES + 20, "this must truncate"
    assert out.value["total_bytes"] == (tools.MAX_DIR_ENTRIES + 20) * 10
