"""A search that succeeded inside a narrow scope must SAY the scope was narrow.

F-161, reproduced from SWE-bench Verified django__django-11477. The agent made
one call:

    search_code(query="translate_url creates an incorrect URL when optional "
                      "named groups are missing in the URL pattern",
                path="django/urls/resolvers.py")

got 4829 characters of plausible results out of that one file, read three
symbols from it, and finished BLOCKED with:

    "The issue mentions a function 'translate_url()' that is not found in the
     codebase. After analyzing the relevant files in django/urls/resolvers.py,
     no such function exists."

`translate_url` is in django/urls/base.py at line 160 and has been for years.

The failure is not that the agent narrowed the search -- narrowing is a
legitimate thing to do, and honouring it is what test_search_scope.py exists to
protect. The failure is that the RESULT did not disclose the narrowing, while
`indexed` went on reporting the repository totals. Plausible hits sitting next
to "regions: 2431" read as a repository-wide search, and the agent generalised
"not in this file" into "not in this repository".

The asymmetry was the giveaway: a scoped search that finds NOTHING already told
the agent to drop path=. Only the successful case was silent, and the
successful case is the dangerous one, because it produces a confident wrong
conclusion instead of an error.
"""

from __future__ import annotations

from localprog import tools

# The target lives here. The agent never looks.
BASE = '''def translate_url(url, lang_code):
    """Devuelve la URL equivalente en otro idioma."""
    parsed = urlsplit(url)
    return parsed.geturl()
'''

# The agent scopes its search to this file, which is real, indexed, and wrong.
RESOLVERS = '''class RegexPattern:
    """Empareja una URL contra una expresion regular."""

    def match(self, path):
        return self.regex.search(path)


class RoutePattern:
    """Empareja una URL contra una ruta con parametros con nombre."""

    def match(self, path):
        return self.regex.search(path)
'''

OTHER = '''def reverse_lazy(viewname):
    """Resuelve un nombre de vista de forma perezosa."""
    return viewname
'''


def build(tmp_path):
    pkg = tmp_path / "django" / "urls"
    pkg.mkdir(parents=True)
    (pkg / "base.py").write_text(BASE, encoding="utf-8")
    (pkg / "resolvers.py").write_text(RESOLVERS, encoding="utf-8")
    (pkg / "utils.py").write_text(OTHER, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, allowed_new_files=("out.txt",))


def test_a_successful_file_scoped_search_says_it_was_scoped(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "empareja una url contra un patron",
                            path="django/urls/resolvers.py")

    assert out["candidates"], "the query matches inside that file"
    assert {c["path"] for c in out["candidates"]} == {"django/urls/resolvers.py"}

    note = out["note"]
    assert "AMBITO" in note, (
        "the search covered one file and the result did not say so -- this is "
        "the exact silence that produced 'the function is not found in the "
        "codebase' for a function that was in the next file along")
    assert "%" in note, (
        "the fraction is what makes the notice proportionate: 0.2% has to read "
        "differently from 28.7% without a threshold deciding it")
    assert "django/urls/resolvers.py" in note
    assert "sin path=" in note, "it must say how to widen the search"


def test_the_counts_distinguish_scope_from_repository(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "empareja una url contra un patron",
                            path="django/urls/resolvers.py")

    indexed = out["indexed"]
    assert indexed["scope"] == "django/urls/resolvers.py"
    assert indexed["searched_regions"] < indexed["regions"], (
        "reporting the repository total alone is what made the silence "
        "misleading rather than merely incomplete")
    # resolvers.py carves two classes and their two methods; the point is only
    # that the scoped count is real and smaller, not its exact value.
    assert indexed["searched_regions"] >= 1


def test_a_directory_scoped_search_says_it_was_scoped(tmp_path):
    ctx = build(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_urls.py").write_text(
        "def test_algo():\n    assert True\n", encoding="utf-8")

    out = tools.search_code(ctx, "empareja una url contra un patron",
                            path="django/urls")
    assert "AMBITO" in out["note"]
    assert out["indexed"]["searched_regions"] < out["indexed"]["regions"]


def test_an_unscoped_search_does_not_claim_to_be_scoped(tmp_path):
    ctx = build(tmp_path)
    out = tools.search_code(ctx, "devuelve la url equivalente en otro idioma")

    assert "AMBITO" not in out["note"], (
        "a repository-wide search must not carry a narrowing warning; crying "
        "wolf on every search is how a real warning stops being read")
    assert out["indexed"]["scope"] == "(todo el repositorio)"
    assert out["indexed"]["searched_regions"] == out["indexed"]["regions"]


def test_the_unscoped_search_finds_what_the_scoped_one_could_not(tmp_path):
    """The half that makes the warning worth acting on."""
    ctx = build(tmp_path)

    scoped = tools.search_code(ctx, "devuelve la url equivalente en otro idioma",
                               path="django/urls/resolvers.py")
    assert "django/urls/base.py" not in {c["path"] for c in scoped["candidates"]}

    wide = tools.search_code(ctx, "devuelve la url equivalente en otro idioma")
    assert "django/urls/base.py" in {c["path"] for c in wide["candidates"]}, (
        "dropping path= is the advice the note gives; it has to actually work")
