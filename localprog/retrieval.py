"""Finding the code a description is talking about. Deterministic, local, stdlib.

WHY THIS EXISTS
---------------
Measured, not assumed. On RepoQA's python split the same model scores 85% when
the function it must identify is handed to it inside the prompt, and 18% when
it has to find that function in the repository first. Reading the 100 sealed
runs, 46 ended with the agent saying it had searched and found nothing -- for a
function that was on disk the whole time, every time.

The tools it had were ``list_dir``, ``grep`` and ``read_symbol``. All three
require you to already suspect where to look. ``grep`` answers "where does this
exact string appear"; nothing answered "which of these four thousand functions
is the one being described". So the agent guessed literal search terms out of a
paraphrase, got nothing, guessed again, and gave up. That is a missing organ,
not a stupid model.

WHAT IT DOES
------------
One BM25 index over definition-delimited regions of the repository. Ask it a
question in words and it returns ranked candidates: path, line range, the
declaration line, and the first lines of the body. The model then recognises
which one it wants -- which is the thing this model is measurably good at.

Offline on the 100 python needles, ranking by description alone puts the right
function in the top 10 of ~4500 candidates about two thirds of the time.

LANGUAGE-AGNOSTIC ON PURPOSE
----------------------------
``list_symbols`` and ``read_symbol`` parse Python with ``ast`` and therefore do
nothing at all for TypeScript, Java, Rust or C++. A retrieval fix that inherited
that limit would be a Python trick dressed as a general capability. So regions
here are found with per-language declaration patterns and a plain indentation
rule, and a file whose language is unknown still gets indexed as chunks. Worse
than a parser, available everywhere, and it degrades instead of vanishing.

NO SEMANTIC MODEL, NO EMBEDDINGS, NO NETWORK
--------------------------------------------
BM25 with textbook constants (k1=1.2, b=0.75). It is lexical: it matches the
words of the query against the words of the code and its comments. That is
enough to put a short list in front of a model that can read, and it costs one
pass over the repository with no third-party dependency and no GPU.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: Files worth indexing. Extension-driven and deliberately broad: retrieval
#: that only looks at one language is the defect this module exists to remove.
SOURCE_EXTENSIONS = frozenset({
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".scala",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".rs", ".go", ".rb", ".php", ".cs", ".swift", ".m",
    ".sh", ".sql", ".lua", ".pl", ".r",
    ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json",
})

#: Which of those actually get carved into declarations. The rest are indexed
#: whole (a README is one document; splitting it on a regex would be noise).
DECLARATION_PATTERNS: dict[str, str] = {
    "py": r"^\s*(?:@\w|(?:async\s+)?def\s+\w|class\s+\w)",
    "ts": (r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
           r"(?:function\s*\*?\s*\w|class\s+\w|interface\s+\w|type\s+\w+\s*=|enum\s+\w|"
           r"(?:const|let|var)\s+\w+\s*(?::[^=]+?)?=\s*(?:async\s*)?"
           r"(?:\([^)]*\)|\w+)\s*(?::[^=;]+?)?=>|"
           r"(?:public|private|protected|static|readonly|abstract|get|set)\s+\w|"
           # a bare method, but never a control-flow header: "if (x) {" is not
           # a declaration and treating it as one shreds every function into
           # one region per branch.
           r"(?!(?:if|for|while|switch|catch|do|else|return|typeof|new|await)\b)"
           r"\w+\s*\([^;{)]*\)\s*(?::\s*[\w<>\[\]|., ]+)?\s*\{)"),
    "java": (r"^\s*(?:@\w+[\w.()\"', ]*\s*$|"
             r"(?:public|private|protected|static|final|abstract|synchronized|native|default)"
             r"[\w<>\[\], ]*\s+\w+\s*\(|"
             r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?"
             r"(?:class|interface|enum|record)\s+\w)"),
    "rs": (r"^\s*(?:#\[|(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"\w+\"\s+)?"
           r"(?:fn|struct|enum|trait|impl|mod|macro_rules!|type|const|static)\s)"),
    "c": (r"^\s*(?:(?:template\s*<[^>]*>\s*)?(?:class|struct|union|enum|namespace)\s+\w|"
          r"[\w:<>~&*\]\[ ]+\s+[\w:~]+\s*\([^;]*\)\s*(?:const)?\s*\{?\s*$|"
          r"#define\s+\w)"),
    "go": r"^\s*(?:func\s|type\s+\w+\s+(?:struct|interface)\b)",
    "rb": r"^\s*(?:def\s+\w|class\s+\w|module\s+\w)",
    "php": r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*(?:function\s+\w|class\s+\w)",
    "cs": (r"^\s*(?:\[[\w(). \"]+\]\s*$|"
           r"(?:public|private|protected|internal|static|abstract|sealed|override|virtual|async)"
           r"[\w<>\[\], ?]*\s+\w+\s*[({])"),
}

#: extension -> which pattern family carves it.
_FAMILY = {
    ".py": "py", ".pyi": "py",
    ".ts": "ts", ".tsx": "ts", ".js": "ts", ".jsx": "ts", ".mjs": "ts", ".cjs": "ts",
    ".java": "java", ".kt": "java", ".scala": "java", ".swift": "java",
    ".rs": "rs",
    ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".cxx": "c", ".hpp": "c", ".hh": "c",
    ".m": "c",
    ".go": "go", ".rb": "rb", ".php": "php", ".cs": "cs",
}

_COMPILED = {family: re.compile(pattern) for family, pattern in DECLARATION_PATTERNS.items()}

#: A region longer than this is cut. A 400-line function is one document either
#: way; the cap stops a file with no detectable declarations from becoming a
#: single enormous document that BM25's length normalisation then buries.
MAX_REGION_LINES = 400

#: Files above this are skipped whole. Generated bundles and vendored blobs are
#: not what anyone is searching for, and they dominate the term statistics.
MAX_FILE_BYTES = 2_000_000

#: Directories never indexed. Kept in sync with tools.SKIP_DIRS by the caller,
#: with the build-output directories that only matter to a repo-wide walk.
DEFAULT_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".tox", ".eggs", "target",
    ".next", ".idea", ".vscode", "site-packages",
})

#: Ordinary English and boilerplate identifiers. A query is a description in
#: prose, so the usual stopwords matter more here than in a code-only index.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can cannot could did do does doing done down during each few for
from further had has have having he her here hers him his how i if in into is it its itself
just me more most my no nor not now of off on once only or other ought our ours out over own
same she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom why with
would you your yours use used using return returns given based one two also may must shall
""".split())

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")

#: Directory components that make a path a test path.
_TEST_DIRS = frozenset({"test", "tests", "testing", "_test", "__tests__",
                        "spec", "specs"})


def is_test_path(path: str) -> bool:
    """Whether this looks like a test file. Conservative on purpose.

    A directory component that says so, or a basename that announces itself.
    Anything cleverer would be guessing, and a guess here quietly changes what
    the agent is shown.

    THIS DOES NOT RANK ANYTHING. It was written for a change that DID -- test
    regions pushed below non-test ones -- and that change was measured and
    rejected: RepoQA python fell 6.0 points, because 8 of its 100 needles live
    in test files and demoting tests pushes the answer down whenever the answer
    is one. The ranking cannot know which case it is in; only the agent can.
    So this is used to REPORT the mix, and to honour an exclusion the agent
    asks for by name. See TESTS_LAST_RESULT.md.
    """
    parts = path.split("/")
    if any(part.lower() in _TEST_DIRS for part in parts[:-1]):
        return True
    base = parts[-1].lower()
    return base.startswith("test_") or base.endswith(
        ("_test.py", "_tests.py", ".test.ts", ".test.js", ".spec.ts",
         ".spec.js", "test.java"))


def tokenize(text: str) -> list[str]:
    """Words, with identifiers split the way a programmer reads them.

    ``parse_http_header`` and ``parseHttpHeader`` both become
    ``parse http header``, so a description written in prose can match code
    written in either convention. Single and two-letter tokens go: they are
    loop variables and noise, and they blow up the postings for nothing.
    """
    out: list[str] = []
    for raw in _WORD.findall(text or ""):
        for word in _CAMEL.sub(r"\1 \2", raw).replace("_", " ").lower().split():
            if len(word) > 2 and word not in STOPWORDS:
                out.append(word)
    return out


@dataclass(frozen=True)
class Region:
    """One indexed unit: a declaration and its body, or a chunk of a file."""

    path: str
    start: int          # 1-based, inclusive
    end: int            # 1-based, inclusive
    header: str         # the declaration line, stripped
    preview: str        # first content lines under it, for the model to judge by

    @property
    def name(self) -> str:
        """A best-effort identifier for the declaration. Display only."""
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*[(<:={]", self.header)
        if match:
            return match.group(1)
        words = _WORD.findall(self.header)
        return words[-1] if words else ""


def _split_regions(text: str, family: str | None) -> list[tuple[int, int]]:
    """Line ranges for one file, as (start, end) 1-based inclusive.

    A declaration owns everything up to the next declaration at the same or
    shallower indentation, so a function keeps its nested helpers instead of
    being truncated at the first one. Anything before the first declaration --
    imports, module docstring, constants -- is its own region, because that is
    frequently the part that says what the file is for.
    """
    lines = text.splitlines()
    if not lines:
        return []
    rx = _COMPILED.get(family or "")
    if rx is None:
        return [(i + 1, min(i + MAX_REGION_LINES, len(lines)))
                for i in range(0, len(lines), MAX_REGION_LINES)]

    starts: list[tuple[int, int]] = []   # (line index, indent)
    for i, line in enumerate(lines):
        if line.strip() and rx.match(line):
            starts.append((i, len(line) - len(line.lstrip())))
    if not starts:
        return [(i + 1, min(i + MAX_REGION_LINES, len(lines)))
                for i in range(0, len(lines), MAX_REGION_LINES)]

    # A decorator line is the start of the thing it decorates, not a region of
    # its own: leaving it standalone produces a one-line document consisting of
    # "@property" and pushes the function it belongs to into a second document
    # that no longer mentions it.
    merged: list[tuple[int, int]] = []
    for n, (idx, indent) in enumerate(starts):
        decorator = lines[idx].lstrip().startswith(("@", "#["))
        if decorator and n + 1 < len(starts) and starts[n + 1][1] == indent:
            continue
        if merged and decorator:
            continue
        merged.append((idx, indent))
    if merged:
        # keep the decorator's own line as the start of the merged region
        rebuilt = []
        for idx, indent in merged:
            back = idx
            while back > 0 and lines[back - 1].lstrip().startswith(("@", "#[")) \
                    and len(lines[back - 1]) - len(lines[back - 1].lstrip()) == indent:
                back -= 1
            rebuilt.append((back, indent))
        starts = rebuilt

    regions: list[tuple[int, int]] = []
    if starts[0][0] > 0:
        regions.append((1, starts[0][0]))
    for n, (idx, indent) in enumerate(starts):
        end = len(lines)
        for later_idx, later_indent in starts[n + 1:]:
            if later_indent <= indent:
                end = later_idx
                break
        end = min(end, idx + MAX_REGION_LINES)
        if end > idx:
            regions.append((idx + 1, end))
    return regions


def _preview(lines: list[str], start: int, end: int, limit: int = 3) -> str:
    """The first few non-empty body lines: docstring, comment or first statement."""
    out: list[str] = []
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped:
            continue
        out.append(stripped[:160])
        if len(out) >= limit:
            break
    return " ".join(out)[:320]


def walk_files(root: Path, skip_dirs=DEFAULT_SKIP_DIRS):
    """Every ordinary file under *root*, pruning as it goes. Sorted, so the
    index is byte-identical between runs on the same tree.

    ``os.walk`` rather than ``rglob`` for two reasons, both of which cost real
    time or real safety:

    * it does not descend into symlinked directories, and a link pointing out of
      the repository would otherwise pull foreign files into the index and hand
      their contents back through ``search_code``. Containment is a property of
      this harness, not a detail of the walk.
    * it lets a skipped directory be pruned instead of walked and then filtered.
      Checking every ancestor of every file for a symlink cost six times the
      build time of the whole index; pruning costs nothing.
    """
    for current, dirs, names in os.walk(root, followlinks=False):
        here = Path(current)
        dirs[:] = sorted(d for d in dirs
                         if d not in skip_dirs and not (here / d).is_symlink())
        for name in sorted(names):
            path = here / name
            if path.is_symlink():
                continue
            yield path


class Index:
    """A BM25 index over one repository's code regions.

    Built in one pass and then reused. ``stale_for`` lets the caller drop it
    when the agent has edited the tree, which is cheaper and more honest than
    rebuilding on every call or serving results from a repository that no
    longer looks like that.
    """

    K1 = 1.2
    B = 0.75

    def __init__(self, root: Path, *, skip_dirs=DEFAULT_SKIP_DIRS,
                 extensions=SOURCE_EXTENSIONS):
        self.root = root
        self.regions: list[Region] = []
        self.files_indexed = 0
        self.extensions_seen: Counter = Counter()
        docs: list[list[str]] = []

        for path in walk_files(root, skip_dirs):
            suffix = path.suffix.lower()
            self.extensions_seen[suffix] += 1
            if suffix not in extensions:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            relative = path.relative_to(root).as_posix()
            lines = text.splitlines()
            path_tokens = tokenize(relative)
            self.files_indexed += 1
            for start, end in _split_regions(text, _FAMILY.get(suffix)):
                body = "\n".join(lines[start - 1:end])
                header = lines[start - 1].strip() if start <= len(lines) else ""
                self.regions.append(Region(
                    path=relative, start=start, end=end, header=header[:200],
                    preview=_preview(lines, start, end),
                ))
                # The path is part of what a region is about: a symbol under
                # audio/transcriptions.py is about audio whether or not it says
                # so. Counted once, so it informs without dominating the body.
                docs.append(path_tokens + tokenize(body))

        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) or 1 for d in docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        total = len(docs)
        self._idf = {word: math.log(1 + (total - count + 0.5) / (count + 0.5))
                     for word, count in df.items()}
        # postings: word -> [(doc index, term frequency)], so scoring touches
        # only the documents that contain a query word.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self._tf):
            for word, freq in tf.items():
                self._postings.setdefault(word, []).append((i, freq))

    def __len__(self) -> int:
        return len(self.regions)

    def search(self, query: str, limit: int = 15) -> list[tuple[Region, float]]:
        """The best-scoring regions for *query*, best first. Ties break by path."""
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = {}
        for word in set(terms):
            idf = self._idf.get(word)
            if idf is None:
                continue
            for i, freq in self._postings.get(word, ()):
                dl = self._len[i]
                scores[i] = scores.get(i, 0.0) + idf * (freq * (self.K1 + 1)) / (
                    freq + self.K1 * (1 - self.B + self.B * dl / self._avg)
                )
        ranked = sorted(scores.items(),
                        key=lambda kv: (-kv[1], self.regions[kv[0]].path,
                                        self.regions[kv[0]].start))
        return [(self.regions[i], score) for i, score in ranked[:limit]]

    def matched_terms(self, query: str) -> tuple[list[str], list[str]]:
        """Which query words exist anywhere in the index, and which do not.

        A word that appears nowhere is the single most useful thing to say back
        about a search that found nothing: it is a fact, it is cheap, and it
        tells the agent which part of its query to abandon.
        """
        known, unknown = [], []
        for word in dict.fromkeys(tokenize(query)):
            (known if word in self._idf else unknown).append(word)
        return known, unknown
