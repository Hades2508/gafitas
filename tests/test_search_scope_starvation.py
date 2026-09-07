"""A filter must not hide a match that the ranking found.

F-167, found by an independent adversarial review of the same night's work,
and reproduced before it was believed.

`search_code` used to over-fetch `limit * 20` rows and filter the result. That
is exact only while the thing being filtered out is rarer than the multiple.
Constructed: thirty test files that match the query strongly, one source file
that matches it weakly, source at rank 31.

    search_code(query, limit=1, exclude_tests=True)
    -> ERROR_NO_MATCH: "las 20 coincidencias ... estan TODAS en ficheros de test"

There were thirty-one matches and one of them was source. The instrument
asserted something false, which is the single thing it must never do -- and it
was introduced by the change that added `exclude_tests` the same night.

The fix is not a larger multiple, which is the same bug at another scale. The
whole ranking is computed inside the index and only sliced at the end, so the
predicate belongs there: filtering before the cut is exact and free, and it
also removes the identical starvation for path scoping, which predates all of
this and had the same shape.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_NO_MATCH, ToolError

QUERY = "limpia una cabecera http quitando espacios sobrantes del principio"


def build(tmp_path, tests: int = 30, with_source: bool = True):
    (tmp_path / "tests").mkdir()
    for i in range(tests):
        # Repeating the terms makes these outrank the source file, which is
        # what a real test file does anyway: it describes the behaviour in
        # prose, and an issue title is prose about behaviour.
        (tmp_path / "tests" / f"test_{i:02d}.py").write_text(
            f'def test_{i}():\n    """{QUERY}. {QUERY}. {QUERY}."""\n'
            f'    assert True\n', encoding="utf-8")
    if with_source:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "headers.py").write_text(
            'def limpia(raw):\n    """cabecera espacios"""\n'
            '    return raw.strip()\n', encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=())


def test_the_source_file_is_buried_far_below_the_window(tmp_path):
    """The premise. If this stops holding, the tests below prove nothing."""
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, limit=50)
    ranks = [n for n, c in enumerate(out["candidates"], 1)
             if not c["path"].startswith("tests/")]
    assert ranks and ranks[0] > 20, (
        "the source must sit past any plausible over-fetch for this to be the "
        "starvation case at all")


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_excluding_tests_finds_the_deeply_buried_source(tmp_path, limit):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, limit=limit, exclude_tests=True)
    assert [c["path"] for c in out["candidates"]] == ["src/headers.py"]


def test_the_all_tests_message_is_only_used_when_it_is_true(tmp_path):
    """The error may say every match was a test only when every match was."""
    ctx = build(tmp_path, with_source=False)
    with pytest.raises(ToolError) as caught:
        tools.search_code(ctx, QUERY, limit=1, exclude_tests=True)
    assert caught.value.code == ERROR_NO_MATCH
    message = str(caught.value)
    assert "TODAS" in message
    # The count has to be the real one, not the size of some internal fetch.
    assert "30" in message, (
        "the number in the message must be counted over the whole ranking; "
        "reporting a fetch window as the match count is how it came to claim "
        "twenty matches when there were thirty-one")


def test_path_scoping_is_not_starved_either(tmp_path):
    """The same shape, and it predates exclude_tests."""
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, limit=1, path="src")
    assert [c["path"] for c in out["candidates"]] == ["src/headers.py"]


def test_an_unfiltered_search_is_unchanged(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, limit=5)
    assert len(out["candidates"]) == 5
    assert all(c["path"].startswith("tests/") for c in out["candidates"]), (
        "without a filter the strong matches still win; the fix must not "
        "quietly reorder an ordinary search")
