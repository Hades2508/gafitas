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


# --------------------------------------- the write channel does not corrupt

#: A real function body with its newlines escaped, at the size the six observed
#: cases actually had. The guard deliberately ignores short payloads.
ESCAPED_BODY = (
    "def parse_header(raw, strict=False):" + chr(92) + "n"
    "    if not raw:" + chr(92) + "n"
    "        raise ValueError('empty header')" + chr(92) + "n"
    "    name, _, value = raw.partition(':')" + chr(92) + "n"
    "    if strict and not value:" + chr(92) + "n"
    "        raise ValueError('header without a value')" + chr(92) + "n"
    "    return name.strip(), value.strip()" + chr(92) + "n"
)

def test_escaped_newlines_are_repaired_and_the_repair_is_announced(tmp_path):
    """F-63. A tool call carries its arguments as JSON, and a small model
    writing a multi-line body into a JSON string sometimes escapes it twice, so
    what arrives is one physical line of literal backslash-n. The harness used
    to write that out exactly as given, without a word.

    The first fix refused it, on the principle that a write tool which edits
    what it is handed is worse than one that says no. Measured, that cost more
    than it saved: one matched run spent five turns being refused and finished
    BLOCKED holding the correct answer, because the model could not produce the
    unescaped form at all. So it is repaired -- and the result says so, which
    is the part that keeps it honest.
    """
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",),
                            allowed_new_files=("a.txt",))
    out = tools.dispatch(ctx, "write_file", {"path": "a.txt", "content": ESCAPED_BODY})
    assert out.ok
    written = (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert written.count("\n") >= 6, "the body came back as real lines"
    assert chr(92) + "n" not in written
    assert "DESESCAPADO" in out.value, "a silent repair is the thing to avoid"


def test_real_newlines_are_written_untouched(tmp_path):
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",))
    body = "def f():\n    return 1\n"
    assert tools.dispatch(ctx, "write_file", {"path": "a.py", "content": body}).ok
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == body


def test_a_genuine_one_liner_with_escapes_still_goes_through(tmp_path):
    """Refuse the conclusive case only.

    ``sep = "\\n\\n"`` is real code: one line, no newline in it, two
    escape sequences. An audit raised it as a false positive and it was, so the
    guard now also requires the payload to be long enough that it can only be a
    flattened body. A write tool that rewrites what it is handed is worse than
    one that says no, and one that refuses valid code is worse than both.
    """
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",))
    ok = 'sep = "' + chr(92) + 'n' + chr(92) + 'n"'
    assert tools.dispatch(ctx, "write_file", {"path": "c.py", "content": ok}).ok
    assert (tmp_path / "c.py").read_text(encoding="utf-8").startswith("sep =")


def test_edit_repairs_an_escaped_needle_so_it_can_match(tmp_path):
    """An escaped `old` never matches, and 'no se encontro el texto' sends the
    agent looking for a problem in the file instead of in its own call."""
    body = ESCAPED_BODY.replace(chr(92) + "n", "\n")
    (tmp_path / "m.py").write_text(body, encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**/*.py",))
    out = tools.dispatch(ctx, "edit", {
        "path": "m.py", "old": ESCAPED_BODY, "new": "def parse_header():\n    pass\n",
    })
    assert out.ok, out.feedback
    assert "DESESCAPADO" in out.value
    assert "def parse_header():" in (tmp_path / "m.py").read_text(encoding="utf-8")


# ------------------------------------------------ the index tells the truth

def test_a_write_outside_the_edit_tools_still_invalidates_the_index(tmp_path):
    """The cache used to be keyed on ``changed_files`` alone, and nothing but
    the edit tools updates that set -- so a file written by
    ``run(["python", "-c", ...])`` left search_code answering out of a
    repository that no longer existed, with the same confidence as a correct
    answer."""
    (tmp_path / "a.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    first = ctx.index()
    assert ctx.index() is first
    (tmp_path / "b.py").write_text("def beta():\n    pass\n", encoding="utf-8")
    assert ctx.index() is not first, "a file appeared and the index never noticed"
    assert any(r.path == "b.py" for r in ctx.index().regions)


def test_the_index_does_not_follow_a_symlink_out_of_the_repository(tmp_path):
    """Containment is a property of this harness, and rglob follows symlinked
    directories. A link pointing outside would pull foreign files into the
    index and hand their contents back through search_code."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("def confidential():\n    pass\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "own.py").write_text("def mine():\n    pass\n", encoding="utf-8")
    try:
        (repo / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not create symlinks without privileges")
    paths = {r.path for r in retrieval.Index(repo).regions}
    assert paths == {"own.py"}, paths


def test_a_decorated_function_is_one_region_not_two(tmp_path):
    """A decorator is the start of the thing it decorates. Left standalone it
    becomes a one-line document saying '@property' while the function it
    belongs to sits in a second document that no longer mentions it."""
    (tmp_path / "m.py").write_text(
        "@property\n"
        "def cached_width(self):\n"
        '    """The width of the rendered column."""\n'
        "    return self._width\n",
        encoding="utf-8",
    )
    regions = retrieval.Index(tmp_path).regions
    owning = [r for r in regions if r.start <= 2 <= r.end]
    assert len(owning) == 1, [(r.start, r.end, r.header) for r in regions]
    assert owning[0].start == 1, "the decorator belongs to the function below it"


def test_a_ranged_read_says_when_it_spans_two_definitions(tmp_path):
    """F-65: a run copied its target function plus the opening lines of the
    next one, because it guessed a line range instead of naming the symbol, and
    scored 0.38 on an answer it believed was exact. The boundary was one parse
    away and nobody mentioned it."""
    (tmp_path / "m.py").write_text(
        "def first(a):\n    return a\n\n\ndef second(b):\n    return b\n",
        encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "m.py", "start": 1, "end": 6})
    assert "definicion(es)" in out.value and "read_symbol" in out.value


def test_an_ordinary_ranged_read_is_left_alone(tmp_path):
    """A note on every read is noise the agent learns to skip, which is how a
    useful note stops being read at all."""
    (tmp_path / "m.py").write_text(
        "def only(a):\n    return a\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "read_file", {"path": "m.py", "start": 1, "end": 2})
    assert "definicion(es)" not in out.value


# ------------------------------- a scope that names nothing is not a result

def test_a_prefix_that_names_nothing_searches_everywhere_and_says_so(tmp_path):
    """F-69, from the java holdout.

    Java repositories have deep conventional layouts and the model knows the
    convention: of 219 search_code calls there, 166 carried a path and 75 came
    back empty, on prefixes like 'src/main/java/org/apache/flink/ml/common/' --
    exactly right for a Maven project and not where these files live. The tool
    answered "ninguna region coincide" every time, which is false: regions
    matched, none of them were under a directory that does not exist. The agent
    read that as "not in this repository" and searched elsewhere.
    """
    deep = tmp_path / "java" / "org" / "x"
    deep.mkdir(parents=True)
    (deep / "Fmt.java").write_text(
        "public class Fmt {\n    public String wrapLine(String t) { return t; }\n}\n",
        encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "search_code",
                         {"query": "wrap a line of text", "path": "src/main/java/org/x"})
    assert out.ok, "an imaginary directory must not cost the whole call"
    assert out.value["candidates"], "it searched the repository instead"
    note = out.value["note"]
    assert "ERROR_SEARCH_SCOPE_EMPTY" in note
    assert "java/" in note, "and it names the directories that do exist"


def test_a_real_prefix_with_no_matches_still_says_no_matches(tmp_path):
    """The two cases must stay distinguishable in the other direction too."""
    (tmp_path / "java").mkdir()
    (tmp_path / "java" / "A.java").write_text(
        "public class A {\n    public int size() { return 1; }\n}\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "search_code",
                         {"query": "quaternion holography", "path": "java"})
    assert not out.ok and out.code == ERROR_NO_MATCH
    assert "SI existe" in out.feedback


def test_a_real_prefix_still_restricts_the_search(tmp_path):
    for name in ("kept", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "m.py").write_text("def widget():\n    pass\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path)
    out = tools.dispatch(ctx, "search_code", {"query": "widget", "path": "kept"})
    assert out.ok
    assert {c["path"] for c in out.value["candidates"]} == {"kept/m.py"}
