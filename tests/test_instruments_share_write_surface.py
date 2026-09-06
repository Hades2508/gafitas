"""No instrument re-declares the write surface (F-145, the other half).

``test_write_surface`` pins the harness's own list against the harness's own
code. It cannot see the benchmark workspace, and the benchmark workspace is
where the damage happened: four instruments each carried a hand-written copy of
the write-tool list, and the replay scorer's copy went stale for two cohorts
without failing anything.

So this checks the property that would have caught it: an instrument may IMPORT
the write surface, and may not RESTATE it.

HOW IT LOOKS RATHER THAN IMPORTS
--------------------------------
The instruments live outside this package and need two ``sys.path`` entries to
import at all. Importing them here to inspect a constant would make the harness
suite depend on the benchmark workspace's import health -- and a copy
reintroduced in a file that happens not to import cleanly would then be reported
as an import error rather than as the drift it is. Reading the source is enough:
a restated list is a literal collection of tool names, and that is visible in the
syntax tree.

SKIPPED WHEN THE WORKSPACE IS ABSENT
------------------------------------
This is a campaign-machine check. Elsewhere it skips, loudly, rather than
pretending to have verified something it could not see.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from localprog import tools

WORKSPACE = Path(r"D:\GATE-A-WS\GAFITAS_EXTERNAL_BENCHMARKS")

#: The four that carried a copy. Named individually rather than globbed: this
#: test should fail if one of them is renamed or deleted, because the reason it
#: exists would then need re-deciding rather than silently passing over an empty
#: match.
INSTRUMENTS = ("replay_state.py", "no_edit_autopsy.py",
               "historical_replay_audit.py", "counterfactual_synonyms.py")

#: A literal naming this many tools is a restated surface, not a coincidence.
#: Two would fire on an ordinary pair like the ``("path", "src")`` key lists
#: these files legitimately use.
RESTATEMENT = 3


def literals(tree: ast.AST):
    """Every literal collection of strings in the file, with its line."""
    for node in ast.walk(tree):
        # No ast.FrozenSet: there is no frozenset literal in the grammar, so a
        # frozen copy would arrive here as the Set or Tuple passed to the call.
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            names = {e.value for e in node.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if names:
                yield node, names


@pytest.mark.skipif(not WORKSPACE.is_dir(),
                    reason=f"benchmark workspace not on this machine: {WORKSPACE}")
@pytest.mark.parametrize("filename", INSTRUMENTS)
def test_instrument_imports_the_write_surface_and_does_not_restate_it(filename):
    path = WORKSPACE / filename
    assert path.is_file(), f"{filename} is gone; this check needs re-deciding"
    source = path.read_text(encoding="utf-8")

    assert "from localprog.tools import WRITE_TOOLS" in source, (
        f"{filename} no longer imports the harness write surface")

    surface = set(tools.WRITE_TOOLS)
    restated = [(node.lineno, sorted(names & surface))
                for node, names in literals(ast.parse(source))
                if len(names & surface) >= RESTATEMENT]
    assert not restated, (
        f"{filename} restates the write surface at {restated}. A copy is how "
        f"F-145 happened: it went stale against replace_file and "
        f"replace_symbol_body, and the replay scorer rebuilt trees without "
        f"them for two cohorts. Import tools.WRITE_TOOLS instead.")


@pytest.mark.skipif(not WORKSPACE.is_dir(),
                    reason=f"benchmark workspace not on this machine: {WORKSPACE}")
def test_the_check_can_still_see_a_copy():
    """The scan above proves nothing unless it fires on the thing it hunts.

    The exact literal that was in ``replay_state.py`` before the fix, checked
    against the detector rather than against my confidence in it.
    """
    stale = '("write_file", "copy_code", "edit", "replace_lines")'
    hits = [names & set(tools.WRITE_TOOLS)
            for _node, names in literals(ast.parse(f"WRITES = {stale}"))]
    assert hits and len(hits[0]) >= RESTATEMENT
