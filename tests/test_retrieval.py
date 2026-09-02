"""The retrieval index, and the search_code tool built on it (F-59).

The language cases matter more than they look. ``list_symbols`` and
``read_symbol`` parse Python with ``ast`` and do nothing at all for anything
else, and a retrieval fix that inherited that limit would be a Python trick
wearing the word "general". Every snippet below is written here by hand, so
these assertions are about the mechanism and owe nothing to any benchmark.
"""

from __future__ import annotations

import pytest

from localprog import retrieval, tools
from localprog.errors import ERROR_NO_MATCH, ERROR_SEARCH_SCOPE_EMPTY


# ------------------------------------------------------------------ tokenizer

def test_identifiers_split_the_way_a_programmer_reads_them():
    assert retrieval.tokenize("parse_http_header") == ["parse", "http", "header"]
    assert retrieval.tokenize("parseHttpHeader") == ["parse", "http", "header"]


def test_stopwords_and_one_letter_noise_go():
    assert retrieval.tokenize("the value of i and x") == ["value"]


# ------------------------------------------------------- regions per language

PY = '''\
import os


def outer(a):
    """Add two numbers together."""
    def helper(b):
        return b + 1
    return helper(a)


class Thing:
    def method(self):
        return 1
'''

TS = '''\
import { readFile } from "fs";

export function parseHeader(raw: string): Header {
  return split(raw);
}

export const toAscii = (bytes: Uint8Array): string => {
  return decoder.decode(bytes);
};

export class Client {
  private send(payload: string): void {
    this.socket.write(payload);
  }
}
'''

JAVA = '''\
package app;

public class Formatter {
    private static final int WIDTH = 80;

    public String wrapLine(String text) {
        return text.substring(0, WIDTH);
    }
}
'''

RUST = '''\
use std::io;

pub fn read_config(path: &str) -> io::Result<Config> {
    Config::load(path)
}

pub struct Config {
    pub width: usize,
}
'''

CPP = '''\
#include <string>

namespace fmt {

class Buffer {
public:
    void append(const std::string& text) {
        data_ += text;
    }
};

}
'''


@pytest.mark.parametrize("suffix,source,expected", [
    (".py", PY, {"outer", "helper", "Thing", "method"}),
    (".ts", TS, {"parseHeader", "toAscii", "Client", "send"}),
    (".java", JAVA, {"Formatter", "wrapLine"}),
    (".rs", RUST, {"read_config", "Config"}),
    (".cpp", CPP, {"Buffer", "append"}),
])
def test_declarations_are_found_in_every_supported_language(tmp_path, suffix, source, expected):
    (tmp_path / ("sample" + suffix)).write_text(source, encoding="utf-8")
    index = retrieval.Index(tmp_path)
    found = {region.name for region in index.regions}
    missing = expected - found
    assert not missing, f"{suffix}: never saw {sorted(missing)} (found {sorted(found)})"


def test_a_python_function_keeps_its_nested_helper(tmp_path):
    """A region runs to the next declaration at the same or shallower indent,
    so a function is not truncated at its first inner def."""
    (tmp_path / "m.py").write_text(PY, encoding="utf-8")
    index = retrieval.Index(tmp_path)
    outer = next(r for r in index.regions if r.name == "outer")
    assert outer.end - outer.start >= 4, "outer swallowed at its nested def"


def test_an_unknown_language_still_gets_indexed(tmp_path):
    """Degrade, never vanish: a file we cannot carve is indexed as chunks."""
    (tmp_path / "notes.md").write_text("# Title\n\nhow the widget cache expires\n",
                                       encoding="utf-8")
    index = retrieval.Index(tmp_path)
    assert len(index) >= 1
    assert index.search("widget cache expiry")


def test_binary_and_skipped_directories_are_left_alone(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("function hidden(){}", encoding="utf-8")
    (tmp_path / "app.py").write_text("def visible():\n    pass\n", encoding="utf-8")
    index = retrieval.Index(tmp_path)
    assert {r.path for r in index.regions} == {"app.py"}


# ------------------------------------------------------------------- ranking

REPO = {
    "src/audio.py": (
        "def transcribe_audio(stream):\n"
        '    """Turn spoken words in a recording into written text."""\n'
        "    return model.decode(stream)\n"
    ),
    "src/colours.py": (
        "def blend(a, b):\n"
        '    """Mix two colours in equal parts."""\n'
        "    return (a + b) / 2\n"
    ),
    "src/net.py": (
        "def retry_request(fn, times=3):\n"
        '    """Call fn again after a failure, up to a number of attempts."""\n'
        "    for _ in range(times):\n"
        "        try:\n"
        "            return fn()\n"
        "        except OSError:\n"
        "            continue\n"
    ),
}


@pytest.fixture()
def repo(tmp_path):
    for relative, body in REPO.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def test_a_description_finds_the_function_it_describes(repo):
    index = retrieval.Index(repo)
    best = index.search("converts speech in a recording into text", limit=1)
    assert best and best[0][0].path == "src/audio.py"


def test_the_query_never_has_to_contain_the_identifier(repo):
    """The whole point: grep needs the literal, this does not."""
    index = retrieval.Index(repo)
    best = index.search("try an operation several times when it fails", limit=1)
    assert best and best[0][0].path == "src/net.py"


def test_unknown_words_are_reported_as_unknown(repo):
    index = retrieval.Index(repo)
    known, unknown = index.matched_terms("blend quaternion holography")
    assert "blend" in known
    assert {"quaternion", "holography"} <= set(unknown)


# ---------------------------------------------------------------- the tool

def call(ctx, name="search_code", **kwargs):
    return tools.dispatch(ctx, name, kwargs)


def test_search_code_returns_ranked_candidates(repo):
    ctx = tools.ToolContext(root=repo)
    out = call(ctx, query="turn a recording of speech into written text")
    assert out.ok
    assert out.value["candidates"][0]["path"] == "src/audio.py"
    assert out.value["candidates"][0]["rank"] == 1
    assert out.value["indexed"]["files"] == 3


def test_search_code_says_which_words_are_nowhere(repo):
    ctx = tools.ToolContext(root=repo)
    out = call(ctx, query="quaternion holography")
    assert not out.ok and out.code == ERROR_NO_MATCH
    assert "quaternion" in out.feedback


def test_search_code_can_be_restricted_to_a_subtree(repo):
    (repo / "other").mkdir()
    (repo / "other" / "x.py").write_text("def blend_two_colours():\n    pass\n",
                                         encoding="utf-8")
    ctx = tools.ToolContext(root=repo)
    out = call(ctx, query="mix two colours", path="other")
    assert out.ok
    assert all(c["path"].startswith("other/") for c in out.value["candidates"])


def test_search_code_refuses_a_limit_outside_the_range(repo):
    ctx = tools.ToolContext(root=repo)
    out = call(ctx, query="anything", limit=0)
    assert not out.ok and out.invalid_call


def test_search_code_on_an_empty_repository_says_so(tmp_path):
    ctx = tools.ToolContext(root=tmp_path)
    out = call(ctx, query="anything at all")
    assert not out.ok and out.code == ERROR_SEARCH_SCOPE_EMPTY


def test_the_index_is_rebuilt_after_an_edit(repo):
    """A search served from a tree the agent has already changed is a lie."""
    ctx = tools.ToolContext(root=repo, write_scope=("src/**/*.py",))
    first = ctx.index()
    assert ctx.index() is first, "rebuilt with nothing changed"
    tools.dispatch(ctx, "edit", {"path": "src/colours.py", "old": "Mix two colours",
                                 "new": "Average two colours"})
    assert ctx.index() is not first


# ------------------------------------------------- grep tells what it knows

def test_grep_reports_where_the_rest_of_the_matches_are(tmp_path):
    """Truncation used to be blind: 50 hits from whatever sorted first, and no
    way to tell that from 50 hits total."""
    for n in range(4):
        (tmp_path / f"m{n}.py").write_text("x = 1\n" * 40, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "grep", {"pattern": "x = 1"})
    assert out.ok
    note = out.value["note"]
    assert "TRUNCATED" in note
    assert "160" in note, note        # 4 files x 40 lines, all of them counted
    assert "m3.py" in note, "the files past the cut are exactly what was invisible"


def test_grep_offers_the_case_insensitive_fact_when_it_is_true(tmp_path):
    (tmp_path / "a.py").write_text("def Handle_Request():\n    pass\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    note = tools.dispatch(ctx, "grep", {"pattern": "handle_request"}).value["note"]
    assert "ignore_case=True" in note


def test_grep_says_which_words_of_a_phrase_exist(tmp_path):
    (tmp_path / "a.py").write_text("def widget():\n    pass\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    note = tools.dispatch(ctx, "grep", {"pattern": "widget quaternion"}).value["note"]
    assert "widget" in note and "quaternion" in note


# ------------------------------------------- the symbol tools tell the truth

def test_a_valid_typescript_file_is_not_called_invalid_syntax(tmp_path):
    """``list_symbols`` and ``read_symbol`` are built on ``ast``, so they only
    ever worked for Python -- and handed anything else they said "linea 1:
    invalid syntax" about a perfectly good file. The limitation is fine; the
    harness asserting something false about the repository is not."""
    (tmp_path / "a.ts").write_text(
        "export function parseHeader(raw: string): Header {\n  return split(raw);\n}\n",
        encoding="utf-8",
    )
    ctx = tools.ToolContext(root=tmp_path)
    for name, args in (("list_symbols", {"path": "a.ts"}),
                       ("read_symbol", {"path": "a.ts", "name": "parseHeader"})):
        out = tools.dispatch(ctx, name, args)
        assert not out.ok
        assert out.code == "ERROR_LANGUAGE_UNSUPPORTED"
        assert "search_code" in out.feedback, "an alternative that does work"
        assert "NO esta mal" in out.feedback


def test_a_directory_is_still_reported_as_a_directory(tmp_path):
    """The language check must not pre-empt the better answer."""
    (tmp_path / "pkg").mkdir()
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "list_symbols", {"path": "pkg"})
    assert out.code == "ERROR_IS_DIRECTORY"


def test_broken_python_is_still_a_syntax_error(tmp_path):
    (tmp_path / "b.py").write_text("def broken(:\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "list_symbols", {"path": "b.py"})
    assert out.code == "ERROR_SYNTAX"
