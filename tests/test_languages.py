"""The language capability table, and the property TypeScript proved by accident.

read_symbol and list_symbols are built on Python's ast and do nothing for
anything else. On the TypeScript holdout the agent lost 202 calls to
ERROR_LANGUAGE_UNSUPPORTED and still went from 17% to 51%, because search_code
and ranged reads do not need a parser. That is the design rule this module
writes down: AST tooling is an amplifier, never a prerequisite.
"""

from __future__ import annotations

from localprog import languages


def test_the_common_path_makes_a_language_usable_without_a_parser():
    ts = languages.LANGUAGES["typescript"]
    assert ts.usable
    assert not ts.symbol_index, "no AST tooling at all"
    assert "symbol_index" in ts.missing()


def test_python_has_the_amplifiers_and_typescript_does_not():
    py = languages.LANGUAGES["python"]
    assert py.symbol_index and py.syntax_validation
    assert "symbol_index" not in py.missing()


def test_a_language_nobody_measured_says_so_rather_than_claiming_support():
    for name in ("go", "csharp", "ruby"):
        caps = languages.LANGUAGES[name]
        assert caps.retrieval_status == languages.NOT_MEASURED
        assert caps.usable, "usable is about the common path, not about evidence"


def test_every_status_is_from_the_closed_set():
    for caps in languages.LANGUAGES.values():
        for status in (caps.retrieval_status, caps.editing_status,
                       caps.verification_status):
            assert status in languages.STATUSES


def test_a_supported_status_must_point_at_evidence():
    """A language whose syntax the model can produce is not a supported
    language. Every claim here has to name a run."""
    for name, caps in languages.LANGUAGES.items():
        claimed = {caps.retrieval_status, caps.editing_status,
                   caps.verification_status}
        if claimed & {languages.SUPPORTED, languages.PARTIAL}:
            assert caps.evidence, f"{name} claims a status with no evidence"


def test_extensions_resolve_to_their_language():
    assert languages.for_extension(".ts").name == "typescript"
    assert languages.for_extension(".rs").name == "rust"
    assert languages.for_extension(".py").name == "python"


def test_an_unclaimed_extension_is_none_and_that_is_not_a_failure():
    """None means no amplifier applies. The index still chunks it and lexical
    retrieval still works, which is why .md and .toml are searchable."""
    assert languages.for_extension(".zig") is None


def test_no_extension_is_claimed_by_two_languages():
    seen: dict[str, str] = {}
    for name, caps in languages.LANGUAGES.items():
        for ext in caps.extensions:
            assert ext not in seen, f"{ext} claimed by {seen.get(ext)} and {name}"
            seen[ext] = name


def test_the_matrix_is_reportable():
    m = languages.matrix()
    assert m["typescript"]["retrieval"] == languages.SUPPORTED
    assert "symbol_index" in m["typescript"]["amplifiers_missing"]
    assert m["go"]["retrieval"] == languages.NOT_MEASURED


def test_every_language_the_index_carves_has_a_row():
    """The declaration patterns and this table must not drift apart."""
    from localprog import retrieval
    carved = {suffix for suffix in retrieval._FAMILY}
    claimed = {ext for caps in languages.LANGUAGES.values() for ext in caps.extensions}
    missing = carved - claimed
    assert not missing, f"the index carves {sorted(missing)} with no capability row"
