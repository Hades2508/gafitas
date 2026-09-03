"""What GAFITAS can actually do with a language, stated rather than assumed.

WHY THIS SHAPE
--------------
TypeScript already proved the important property, and it proved it by accident.
``read_symbol`` and ``list_symbols`` are built on Python's ``ast`` and do nothing
at all for anything else, so on the TypeScript holdout the agent lost 202 calls
to ERROR_LANGUAGE_UNSUPPORTED -- and still went from 17% to 51%, because
``search_code`` and ranged reads do not need a parser.

That is the design rule this module writes down: **AST tooling is an
amplifier, never a prerequisite.** The common path -- lexical retrieval, ranged
reads, grep, edit, run -- must work everywhere, and a language earns extra
capabilities without the core learning its name.

WHAT IS NOT HERE
----------------
No ``if python: ... elif rust: ...`` in the core. A capability is data about a
language, and the tools ask for it. When a language eventually needs a real
parser or a compiler, it becomes a row in this table and an adapter behind it,
not a branch in ``tools.py``.

HONESTY ABOUT STATUS
--------------------
SUPPORTED / PARTIAL / UNSUPPORTED are claims about MEASURED behaviour, and the
default is NOT_MEASURED. A language whose syntax the model can produce is not a
supported language; every status here points at a number somebody ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Measured, not hoped for.
SUPPORTED = "SUPPORTED"        # end-to-end evidence on a real corpus
PARTIAL = "PARTIAL"            # retrieval and reading work; some organ is missing
UNSUPPORTED = "UNSUPPORTED"    # measured and found wanting
NOT_MEASURED = "NOT_MEASURED"  # the only honest default
STATUSES = (SUPPORTED, PARTIAL, UNSUPPORTED, NOT_MEASURED)


@dataclass(frozen=True)
class LanguageCapabilities:
    """Which organs exist for one language, and what has been measured.

    ``file_detection``, ``lexical_retrieval`` and ``ranged_reads`` are true for
    every language the index will look at, because they are the common path.
    Everything else is an amplifier that a language may or may not have.
    """

    name: str
    extensions: tuple[str, ...]
    #: The common path. True everywhere the index indexes.
    lexical_retrieval: bool = True
    ranged_reads: bool = True
    #: Amplifiers.
    symbol_index: bool = False          # list_symbols / read_symbol
    syntax_validation: bool = False     # refuse an edit that would not parse
    declaration_regions: bool = False   # the index can carve declarations
    imports: bool = False
    references: bool = False
    formatter: bool = False
    compiler: bool = False
    test_runner: bool = False
    package_manager: bool = False
    #: Status per class of work, each pointing at evidence.
    retrieval_status: str = NOT_MEASURED
    editing_status: str = NOT_MEASURED
    verification_status: str = NOT_MEASURED
    evidence: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Can GAFITAS work in this language at all?

        The bar is the common path, not the amplifiers. TypeScript scores 51%
        with no symbol index at all, so requiring one would declare a language
        unusable that demonstrably is not.
        """
        return self.lexical_retrieval and self.ranged_reads

    def missing(self) -> tuple[str, ...]:
        """Amplifiers this language does not have. Not failures -- gaps."""
        return tuple(name for name in (
            "symbol_index", "syntax_validation", "declaration_regions",
            "imports", "references", "formatter", "compiler",
            "test_runner", "package_manager",
        ) if not getattr(self, name))


#: The table. Every status here points at a run; nothing is aspirational.
#:
#: retrieval_status is the one with real evidence today: four languages measured
#: against a pre-change control on the same corpus, each with its own frozen
#: holdout. editing_status and verification_status are honest about the fact
#: that the ticket batteries are Python-only.
LANGUAGES: dict[str, LanguageCapabilities] = {
    "python": LanguageCapabilities(
        name="python", extensions=(".py", ".pyi"),
        symbol_index=True, syntax_validation=True, declaration_regions=True,
        imports=True, test_runner=True,
        retrieval_status=SUPPORTED,
        editing_status=SUPPORTED,
        verification_status=SUPPORTED,
        evidence=("RepoQA agent 19.00% -> 43/46% (p=0.00027)",
                  "ticket battery 10/11, zero paid",
                  "self-maintenance: dogfood07, dogfood09 promoted",
                  "candidate recall@15, no model in the loop: 74.0% over the whole repo, "
                  "89.0% scoped to the file the mission names (F-79)",
                  ),
    ),
    "typescript": LanguageCapabilities(
        name="typescript", extensions=(".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
        declaration_regions=True,
        retrieval_status=SUPPORTED,
        editing_status=NOT_MEASURED,
        verification_status=NOT_MEASURED,
        evidence=("RepoQA agent 17.00% -> 51.00% (p=1.4e-07), holdout frozen first",
                  "no symbol index: 202 calls lost to ERROR_LANGUAGE_UNSUPPORTED "
                  "and it still tripled. This is the proof that AST tooling is an "
                  "amplifier and not a prerequisite.",
                  "candidate recall@15, no model in the loop: 80.0% over the whole repo, "
                  "98.0% scoped to the file the mission names (F-79)",
                  ),
    ),
    "java": LanguageCapabilities(
        name="java", extensions=(".java",),
        declaration_regions=True,
        retrieval_status=SUPPORTED,
        editing_status=NOT_MEASURED,
        verification_status=NOT_MEASURED,
        evidence=("RepoQA agent 14.00% -> 36.00% (p=0.00068), holdout frozen first",
                  "F-69 found here: 75 of 219 searches died on Maven-shaped paths "
                  "that do not exist in these checkouts",
                  "candidate recall@15, no model in the loop: 54.0% over the whole repo, "
                  "83.0% scoped to the file the mission names (F-79)",
                  ),
    ),
    "rust": LanguageCapabilities(
        name="rust", extensions=(".rs",),
        declaration_regions=True,
        retrieval_status=SUPPORTED,
        editing_status=NOT_MEASURED,
        verification_status=NOT_MEASURED,
        evidence=("RepoQA agent 11.00% -> 32.00% (p=0.00019), holdout frozen "
                  "before the fix it validates",
                  "candidate recall@15, no model in the loop: 64.0% over the whole repo, "
                  "86.0% scoped to the file the mission names (F-79)",
                  ),
    ),
    "cpp": LanguageCapabilities(
        name="cpp", extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c"),
        declaration_regions=True,
        retrieval_status=PARTIAL,
        editing_status=NOT_MEASURED,
        verification_status=NOT_MEASURED,
        evidence=("RepoQA agent 22.00% post-retrieval; control measured "
                  "separately. The weakest language for the reference engine "
                  "natively too (MODEL_NATIVE 71.00 against python's 85.00).",
                  "candidate recall@15, no model in the loop: 58.0% over the whole repo, "
                  "81.0% scoped to the file the mission names (F-79)",
                  ),
    ),
    # Carved by the index and never measured. Rows exist so the table and the
    # declaration patterns cannot drift apart -- a test asserts exactly that,
    # and it is what found these five missing.
    "kotlin": LanguageCapabilities(
        name="kotlin", extensions=(".kt",), declaration_regions=True),
    "scala": LanguageCapabilities(
        name="scala", extensions=(".scala",), declaration_regions=True),
    "swift": LanguageCapabilities(
        name="swift", extensions=(".swift",), declaration_regions=True),
    "php": LanguageCapabilities(
        name="php", extensions=(".php",), declaration_regions=True),
    "objc": LanguageCapabilities(
        name="objc", extensions=(".m",), declaration_regions=True),
    "go": LanguageCapabilities(
        name="go", extensions=(".go",), declaration_regions=True),
    "csharp": LanguageCapabilities(
        name="csharp", extensions=(".cs",), declaration_regions=True),
    "ruby": LanguageCapabilities(
        name="ruby", extensions=(".rb",), declaration_regions=True),
}


def for_extension(suffix: str) -> LanguageCapabilities | None:
    """Which language owns a file extension, or None if nobody claims it.

    None does NOT mean unusable: the index falls back to chunking, and lexical
    retrieval works on anything that is text. It means no amplifier applies.
    """
    suffix = suffix.lower()
    for caps in LANGUAGES.values():
        if suffix in caps.extensions:
            return caps
    return None


def matrix() -> dict:
    """The capability table, for a report or a status page."""
    return {
        name: {
            "usable": caps.usable,
            "retrieval": caps.retrieval_status,
            "editing": caps.editing_status,
            "verification": caps.verification_status,
            "amplifiers_present": [n for n in (
                "symbol_index", "syntax_validation", "declaration_regions",
                "imports", "references", "formatter", "compiler",
                "test_runner", "package_manager") if getattr(caps, n)],
            "amplifiers_missing": list(caps.missing()),
            "evidence": list(caps.evidence),
        }
        for name, caps in LANGUAGES.items()
    }
