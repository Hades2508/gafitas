"""The agent chooses whether tests count as results; the index never guesses.

F-163. Demoting test files by DEFAULT was implemented, measured and rejected:

    SWE-bench file-finding, 245 DEV cases   64.1% -> 73.9%   (+9.8)
    RepoQA python, 100 needles              74.0% -> 68.0%   (-6.0)

The loss is not incidental. 8 of RepoQA's 100 python needles LIVE in test
files, and 5 of 100 in typescript; the three languages with no test-resident
needles moved by exactly zero. A blanket demotion pushes the answer down
whenever the answer is a test, and the ranking has no way to know which case it
is in.

The agent always does. So the choice is an argument it passes, and the fact --
how many of these results are tests -- is reported rather than acted on. That
is the same shape as F-161, which discloses the search scope instead of
overriding it, and it is the only shape that has survived measurement here.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_BAD_ARGUMENTS, ERROR_NO_MATCH, ToolError

SOURCE = '''def normalise_header(raw):
    """Limpia una cabecera http quitando espacios."""
    return raw.strip().lower()
'''

TEST_FILE = '''def test_normalise_header():
    """Comprueba que limpia una cabecera http quitando espacios."""
    assert normalise_header("  X ") == "x"


def test_normalise_header_empty():
    """Comprueba que una cabecera http vacia sigue vacia."""
    assert normalise_header("") == ""
'''

UNRELATED = '''def open_socket(host):
    """Abre una conexion de red."""
    return host
'''


def build(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "headers.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "pkg" / "net.py").write_text(UNRELATED, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_headers.py").write_text(TEST_FILE,
                                                        encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=("out.txt",))


QUERY = "limpia una cabecera http quitando espacios"


def test_by_default_tests_are_returned(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY)
    paths = {c["path"] for c in out["candidates"]}
    assert "tests/test_headers.py" in paths, (
        "the default must not silently drop tests -- that was measured and it "
        "costs RepoQA python 6 points")


def test_excluding_tests_removes_them(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, exclude_tests=True)
    paths = {c["path"] for c in out["candidates"]}
    assert paths, "the source file still matches"
    assert "pkg/headers.py" in paths
    assert not any("test" in p for p in paths)


def test_the_result_says_how_many_are_tests(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY)
    note = out["note"]
    assert "TEST" in note, (
        "the index knows the mix and the agent cannot see it without reading "
        "every path; saying so costs one line")
    assert "exclude_tests=true" in note, "it must say how to act on the fact"


def test_no_notice_when_the_repository_has_no_tests(tmp_path):
    """A notice that fires when it does not apply stops being read."""
    # Written first against a tree that HAS tests, asking for something
    # unrelated -- and it failed, correctly: BM25 still returned the test file
    # third, because a partial match is a match. The notice was right and the
    # fixture was wrong. The honest way to test the negative is a repository
    # with no test files in it at all.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "net.py").write_text(UNRELATED, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=())

    out = tools.search_code(ctx, "abre una conexion de red")
    assert out["candidates"]
    assert "TEST" not in out["note"]


def test_no_notice_when_the_agent_already_excluded_them(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, QUERY, exclude_tests=True)
    assert "exclude_tests=true" not in out["note"], (
        "telling the agent to do the thing it just did is noise")


def test_excluding_everything_says_so_rather_than_no_match(tmp_path):
    """All the matches were tests. That is a different fact from 'no match'."""
    tmp_path.joinpath("tests").mkdir()
    tmp_path.joinpath("tests", "test_only.py").write_text(
        'def test_thing():\n    """Comprueba una cosa muy concreta."""\n'
        '    assert True\n', encoding="utf-8")
    tmp_path.joinpath("pkg").mkdir()
    # No shared vocabulary with the query at all. The first version of this
    # fixture reused the Spanish UNRELATED source and the test did not raise,
    # because STOPWORDS is English: "una" is an ordinary term to this index and
    # it matched both files.
    tmp_path.joinpath("pkg", "other.py").write_text(
        'def zephyr(qq):\n    """Xylophone bracket."""\n    return qq\n',
        encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, allowed_new_files=())

    with pytest.raises(ToolError) as caught:
        tools.search_code(ctx, "comprueba una cosa muy concreta",
                          exclude_tests=True)
    assert caught.value.code == ERROR_NO_MATCH
    message = str(caught.value)
    assert "TODAS" in message or "todas" in message
    assert "exclude_tests" in message, (
        "the agent has to be told that its own filter, not the repository, is "
        "why the list is empty")


def test_exclude_tests_must_be_a_boolean(tmp_path):
    ctx = build(tmp_path)
    with pytest.raises(tools.InvalidCall) as caught:
        tools.search_code(ctx, QUERY, exclude_tests="yes")
    assert caught.value.code == ERROR_BAD_ARGUMENTS


def test_the_parameter_is_advertised_to_the_model(tmp_path):
    """A capability the schema does not mention does not exist."""
    assert "exclude_tests" in tools.SPECS["search_code"][1]
    assert "exclude_tests" in tools.PARAM_DOC
    schema = tools.native_schema(only=("search_code",))
    props = schema[0]["function"]["parameters"]["properties"]
    assert props["exclude_tests"]["type"] == "boolean"
    assert props["exclude_tests"]["description"]
