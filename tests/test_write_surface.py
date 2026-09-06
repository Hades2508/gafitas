"""WRITE_TOOLS is derived from the code, not maintained by hand (F-145).

WHY THIS FILE EXISTS
--------------------
The list of tools that can change the tree was written out by hand in five
places. Four of them were copies. When ``replace_file`` and
``replace_symbol_body`` joined the write surface, the copies were not updated,
and nobody noticed for two cohorts -- because a stale copy is not a crash, it is
a silence.

The cost was not theoretical. The replay scorer walks a run's write events to
rebuild the tree it produced. Blind to those two tools, it rebuilt a tree
WITHOUT the change, the acceptance suite failed there, and it reported a
disagreement with the direct score. A cohort was invalidated by a verification
route that could not see the thing it was verifying -- the worst failure mode a
second opinion can have, because it looks like evidence.

The copies are gone; every instrument imports ``tools.WRITE_TOOLS`` now. That
fixes the copies, not the original. This file fixes the original, by refusing to
let the one remaining list be maintained by memory.

THE ANCHOR
----------
Every tool that mutates a tracked file in this module goes through exactly one
function, ``_write_text``. That single door is what makes the write surface a
fact about the code instead of a claim about it: the set of tools that can reach
it is computable, so ``WRITE_TOOLS`` can be checked against the program rather
than against whoever last remembered to edit it.

A seventh write tool added tomorrow and left out of the list fails here, in the
same commit that adds it.

WHAT THIS DOES NOT CLAIM
------------------------
``run`` and ``run_tests`` are NOT in ``WRITE_TOOLS`` and are deliberately left
out of the sandbox check below. They spawn subprocesses, and a subprocess writes
with the user's permissions -- that frontier is real, it is known, and it is not
closed by anything here. ``WRITE_TOOLS`` is the set of tools that write DIRECTLY,
which is exactly what a replay of tool calls can reconstruct. Test
``test_subprocess_frontier_is_recorded_not_hidden`` pins that distinction so no
future reader mistakes this file for a containment proof.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from localprog import tools

SOURCE = Path(tools.__file__)
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCS = {node.name: node for node in TREE.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

#: The single door to disk. Asserted to be exactly that by
#: ``test_write_text_is_the_only_door``.
DOOR = "_write_text"

#: Filesystem mutators whose NAME alone is unambiguous: no builtin container or
#: string carries these methods, so an attribute call spelling one of them is
#: touching a path.
MUTATORS = frozenset({
    "write_text", "write_bytes", "unlink", "touch", "mkdir", "rmdir",
    "rename", "symlink_to", "hardlink_to", "chmod",
})

#: Mutators whose names ARE ambiguous, matched only in dotted ``os.x`` /
#: ``shutil.x`` form.
#:
#: ``remove`` and ``replace`` are both list/str methods and both appear in this
#: module doing ordinary work -- ``ctx.reached_symbols.remove(...)`` at
#: tools.py:1250 is a list, not a file. Matching them by bare name reports the
#: harness as having a second door to disk when it does not, and a check that
#: cries wolf gets deleted by the next person to see it fail.
#:
#: Ambiguity is not tolerated here so much as moved: what the name cannot decide,
#: the sandbox tests below decide by watching the TREE, so a write through any
#: route -- including a ``Path.replace`` this scan cannot see -- still shows up.
QUALIFIED = {"os": frozenset({"remove", "unlink", "rename", "replace", "rmdir",
                              "removedirs", "renames", "truncate", "mkdir",
                              "makedirs"}),
             "shutil": frozenset({"copyfile", "copytree", "copy", "copy2",
                                  "move", "rmtree"})}


def called_names(node) -> set:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
    return out


def reaches(start: str, target: str) -> bool:
    """Whether *start* can reach *target* through module-level functions."""
    seen, stack = set(), [start]
    while stack:
        name = stack.pop()
        if name == target:
            return True
        if name in seen or name not in FUNCS:
            continue
        seen.add(name)
        stack.extend(called_names(FUNCS[name]))
    return False


# --------------------------------------------------------------- the list

def test_write_tools_is_exactly_what_can_reach_the_door():
    """The declared list equals the tools that can actually write.

    Derived from the call graph, so it fails in both directions: a tool that
    writes and is missing from the list, and a name in the list that cannot
    write and would make a replay reconstruct a change nobody made.
    """
    derived = {tool for tool, impl in tools._IMPL.items()
               if reaches(impl.__name__, DOOR)}
    assert derived == set(tools.WRITE_TOOLS), (
        f"write surface drifted: only in code {sorted(derived - set(tools.WRITE_TOOLS))}, "
        f"only in WRITE_TOOLS {sorted(set(tools.WRITE_TOOLS) - derived)}")


def test_write_tools_are_real_tools_and_listed_once():
    assert set(tools.WRITE_TOOLS) <= set(tools.SPECS)
    assert len(tools.WRITE_TOOLS) == len(set(tools.WRITE_TOOLS))


def test_write_text_is_the_only_door():
    """No second route to disk, so the anchor above stays an anchor."""
    strays = []
    for name, node in FUNCS.items():
        if name == DOOR:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if isinstance(func, ast.Attribute) and func.attr in MUTATORS:
                strays.append((name, func.attr, sub.lineno))
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.attr in QUALIFIED.get(func.value.id, ())):
                strays.append((name, f"{func.value.id}.{func.attr}", sub.lineno))
            if isinstance(func, ast.Name) and func.id == "open":
                mode = ""
                if len(sub.args) > 1 and isinstance(sub.args[1], ast.Constant):
                    mode = str(sub.args[1].value)
                for kw in sub.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if any(ch in mode for ch in "wax+"):
                    strays.append((name, f"open({mode!r})", sub.lineno))
    assert not strays, (
        f"a second route to disk exists, so WRITE_TOOLS can no longer be "
        f"derived from {DOOR}: {strays}")


# ------------------------------------------------------- the sandbox check

MODULE = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"


def snapshot(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return out


def sandbox(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mod.py").write_text(MODULE, encoding="utf-8")
    ctx = tools.ToolContext(root=root, write_scope=("mod.py",),
                            allowed_new_files=("new.py", "out.py"))
    return root, ctx


#: One valid call per tool. The write half must change the tree; the read half
#: must not. ``run`` and ``run_tests`` are excluded -- see the module docstring.
READS = [
    ("read_file", {"path": "mod.py"}),
    ("list_dir", {}),
    ("grep", {"pattern": "alpha"}),
    ("search_code", {"query": "alpha"}),
    ("list_symbols", {"path": "mod.py"}),
    ("read_symbol", {"path": "mod.py", "name": "alpha"}),
    ("finish", {"summary": "nothing", "status": "GAVE_UP"}),
]

WRITES = [
    ("edit", {"path": "mod.py", "old": "return 1", "new": "return 9"}),
    ("replace_lines", {"path": "mod.py", "start": 2, "end": 2,
                       "content": "    return 9"}),
    ("write_file", {"path": "new.py", "content": "x = 1\n"}),
    ("replace_file", {"path": "mod.py", "content": "def alpha():\n    return 9\n"}),
    ("replace_symbol_body", {"path": "mod.py", "name": "alpha",
                             "body": "return 9"}),
    ("copy_code", {"src": "mod.py", "into": "out.py", "name": "beta"}),
]


@pytest.mark.parametrize("name, args", READS)
def test_a_tool_outside_the_list_leaves_the_tree_alone(tmp_path, name, args):
    root, ctx = sandbox(tmp_path)
    assert name not in tools.WRITE_TOOLS
    before = snapshot(root)
    tools.dispatch(ctx, name, args)
    assert snapshot(root) == before, f"{name} changed the tree and is not in WRITE_TOOLS"


@pytest.mark.parametrize("name, args", WRITES)
def test_every_listed_write_tool_can_actually_write(tmp_path, name, args):
    """A name in the list that cannot write would be as wrong as a missing one.

    ``replace_file`` and ``replace_symbol_body`` require the file to have been
    opened this run, so the read that earns that right is part of the call, not
    a way around the check.
    """
    root, ctx = sandbox(tmp_path)
    assert name in tools.WRITE_TOOLS
    if name in ("replace_file", "replace_symbol_body"):
        tools.dispatch(ctx, "read_file", {"path": "mod.py"})
    before = snapshot(root)
    outcome = tools.dispatch(ctx, name, args)
    assert outcome.ok, f"{name} refused a call this test expects to work: {outcome.code}"
    assert snapshot(root) != before, f"{name} is in WRITE_TOOLS and changed nothing"


def test_the_sandbox_check_covers_the_whole_surface():
    """No tool is quietly left untested except the two named in the docstring."""
    covered = {name for name, _ in READS} | {name for name, _ in WRITES}
    assert covered == set(tools.SPECS) - {"run", "run_tests"}


def test_subprocess_frontier_is_recorded_not_hidden():
    """``run``/``run_tests`` are outside WRITE_TOOLS, and that is not a claim
    that they cannot write.

    They execute subprocesses with the user's permissions. Nothing in this file
    constrains that, and calling ``WRITE_TOOLS`` 'the tools that can write'
    without this note would overstate it.
    """
    assert "run" not in tools.WRITE_TOOLS
    assert "run_tests" not in tools.WRITE_TOOLS
    assert not reaches(tools._IMPL["run"].__name__, DOOR)
    assert not reaches(tools._IMPL["run_tests"].__name__, DOOR)
