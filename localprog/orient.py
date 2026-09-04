"""The repository's shape, handed over for free instead of walked for.

MEASURED
--------
Fifty matched runs of the reference engine, 426 tool calls:

    list_dir      140      turn 1: 49 of 50 runs.  turn 2: 38.  turn 3: 24.
    first write   median turn 7

The mission NAMES the file. The agent still spent its first three or four turns
walking the directory tree, because the working instructions opened with
"ORIENTATE. Si no conoces el repositorio, empieza por list_dir" and it followed
them. A quarter of every run went on establishing something the harness already
knew and could have said in the objective.

That is the deterministic-offload rule applied to the very first thing an agent
does: work that needs no reasoning should not cost a turn. Walking a directory
is os.walk. It costs the harness milliseconds and the engine three inferences,
and the engine's three inferences are the expensive ones.

WHAT THIS IS NOT
----------------
It is not a summary and not a hint. It is the same listing ``list_dir`` would
have returned, chosen deterministically and bounded, so an agent that reads it
knows exactly what an agent that walked would have known. Nothing is ranked
toward the answer -- the map cannot depend on the objective, or it would be a
retrieval channel pretending to be orientation.

BOUNDS
------
A repository map that grows with the repository would eat the context it is
meant to save. Directories come first by file count, files within them in path
order, and the whole thing is truncated to a fixed character budget with a line
saying what was left out. A truncated map is still an orientation; an unbounded
one is a second copy of the repo.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import retrieval
from .retrieval import walk_files

#: Roughly 400 tokens. Three list_dir results cost several times this, and the
#: engine had to spend a turn on each.
MAX_CHARS = 1600

#: Per directory. Enough to recognise a package, not enough to enumerate one.
MAX_FILES_PER_DIR = 6
MAX_DIRS = 14


def repo_map(root: Path, *, max_chars: int = MAX_CHARS,
             max_files_per_dir: int = MAX_FILES_PER_DIR,
             max_dirs: int = MAX_DIRS) -> str:
    """A compact, deterministic picture of what is in *root*.

    Ordering is by file count then by name, so the same repository always
    produces the same map -- a map that varied between runs would make two runs
    of the same mission incomparable for no reason.
    """
    by_dir: dict[str, list[str]] = defaultdict(list)
    total = 0
    for path in walk_files(root):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:                      # a symlink out of the tree
            continue
        total += 1
        parent = rel.rsplit("/", 1)[0] if "/" in rel else "."
        by_dir[parent].append(rel.rsplit("/", 1)[-1])

    if not by_dir:
        return ""

    ordered = sorted(by_dir.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    lines = [f"ESTRUCTURA DEL REPOSITORIO ({total} ficheros de codigo):"]
    shown_dirs = 0
    for directory, names in ordered:
        if shown_dirs >= max_dirs:
            break
        names.sort()
        head = ", ".join(names[:max_files_per_dir])
        more = len(names) - max_files_per_dir
        line = f"  {directory}/  ({len(names)}) {head}" + (f", +{more} mas" if more > 0 else "")
        if sum(len(x) + 1 for x in lines) + len(line) > max_chars:
            break
        lines.append(line)
        shown_dirs += 1

    hidden = len(ordered) - shown_dirs
    if hidden > 0:
        # Saying what is missing is the difference between a bounded map and a
        # misleading one: an agent that believes it has seen everything will not
        # go looking for the rest.
        lines.append(f"  (+{hidden} directorios mas; list_dir los enumera)")
    return "\n".join(lines)

#: How many regions to rank at turn zero. Five is the shape of a shortlist: the
#: true needle's median rank is 2 and recall@15 is 74-98% depending on language,
#: so five buys most of the reach for a fraction of the tokens fifteen would
#: cost in a prompt that is sent on every turn.
OPENING_CANDIDATES = 5


def opening_candidates(root: Path, objective: str, *,
                       limit: int = OPENING_CANDIDATES,
                       pending: tuple[str, ...] = ()) -> str:
    """The repository's own ranking of the objective, computed once, for free.

    Reaching the right symbol is what decides these runs -- conversion is
    88-100% once it is named, across a 4B and two 3Bs -- and the reach rate is
    what separates them: 34, 13 and 3 out of 50. Of granite4.1:3b's 37
    non-reaching runs, 25 never ran a search at all, although the instructions
    tell it to. Prose describing an action is not the action.

    Ranking is arithmetic. It needs no inference, the index is already built,
    and leaving it to the agent costs a turn that is often never spent.

    Presented as a ranking and never as an answer. F-85 is the standing lesson:
    a list that looks like a recommendation gets taken as one. This differs from
    that catalogue in the way that matters -- it is ordered by similarity to the
    objective rather than by position in a file -- and it still says so.
    """
    text = (objective or "").strip()
    if len(text) < 40:
        # Too short to rank against. A three-word objective produces noise, and
        # a shortlist built from noise is worse than none.
        return ""
    try:
        index = retrieval.Index(root)
        rows = index.search(text, limit=limit)
    except (OSError, ValueError, RecursionError):
        # A shortlist is a convenience and a run must never fail because
        # building one did -- but the catch is narrow on purpose. The first
        # version caught Exception, and swallowed a NameError from a missing
        # import for a whole debugging cycle: it returned "no candidates",
        # which is exactly what a working index with no matches returns.
        # A handler that cannot be told apart from success is not a handler.
        return ""
    if not rows:
        return ""

    lines = [f"CANDIDATOS (los {len(rows)} sitios del repositorio que mas se "
             f"parecen a lo que pide el objetivo, ordenados por parecido y NO "
             f"por certeza: el primero no tiene por que ser el bueno):"]
    for n, (region, _score) in enumerate(rows, 1):
        symbol = region.name or "?"
        lines.append(f"  {n}. {region.path}:{region.start}-{region.end}  "
                     f"{symbol}  |  {region.header[:70]}")
    top, _score = rows[0]
    lines.append(f"  Para ver uno entero: read_symbol(path={top.path!r}, "
                 f"name={top.name!r}).")
    if pending:
        # The omission this fixes. granite4.1:3b's 21 remaining failures all
        # WRITE the answer by hand instead of copying it, and its measured
        # payload limit is 1600 characters, so retyping a long function
        # corrupts it. The shortlist named read_symbol and search_code and
        # never named the one call that moves bytes without retyping --
        # search_code's own note has named it since F-82, and turn zero did not.
        lines.append(f"  Para ponerlo en {pending[0]} sin reescribirlo: "
                     f"copy_code(src={top.path!r}, into={pending[0]!r}, "
                     f"name={top.name!r}).")
    lines.append("  Para buscar otra cosa: search_code(query=...).")
    return "\n".join(lines)
