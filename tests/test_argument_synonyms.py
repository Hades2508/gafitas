"""F-116: the harness spells one argument two ways, and charged the model.

Six of the seven file tools call the file `path`. `copy_code` alone calls it
`src`. In cohort 3 phi4-mini emitted

    copy_code(path='mlc_llm/transform/lift_tir_global_buffer_alloc.py',
              name='contain_symbolic_var', start=69, end=...)

-- the right tool, the right file, the right symbol, the right lines -- and was
refused with ERROR_MISSING_ARGUMENT. It repeated the identical call on turns 4,
8 and 9 of the same run and was refused each time. That run is a loss recorded
against the model for a name the harness itself is inconsistent about.

The rule is deliberately narrow, and the narrowness is what these tests pin:

  - a synonym is taken only when the real name is ABSENT;
  - a call that supplies both is left exactly as written, never merged;
  - the rename is REPORTED, because an agent that is not told the right name
    will spell it the same way next turn;
  - the table cannot drift from SPECS, checked at import.
"""

from __future__ import annotations

import pytest

from localprog import tools


# ------------------------------------------------------------- the mapping

def test_the_call_phi_kept_making_now_works():
    got, renamed = tools.apply_synonyms(
        "copy_code", {"path": "a/b.py", "into": "ANSWER.txt", "name": "f",
                      "start": 69, "end": 80})
    assert got["src"] == "a/b.py"
    assert "path" not in got
    assert renamed == ["path -> src"]
    assert got["name"] == "f" and got["start"] == 69, "nothing else may move"


def test_a_correct_call_is_untouched():
    args = {"src": "a.py", "into": "b.txt", "name": "f"}
    got, renamed = tools.apply_synonyms("copy_code", dict(args))
    assert got == args and renamed == []


def test_both_names_present_is_left_alone():
    """Two different values for one parameter is ambiguity, not a misspelling.
    Picking one would be a guess, and a wrong guess here copies the wrong file
    while reporting success."""
    args = {"src": "right.py", "path": "wrong.py", "into": "b.txt"}
    got, renamed = tools.apply_synonyms("copy_code", dict(args))
    assert got["src"] == "right.py" and renamed == []


def test_a_tool_with_no_table_is_untouched():
    args = {"path": "a.py", "name": "f"}
    got, renamed = tools.apply_synonyms("read_symbol", dict(args))
    assert got == args and renamed == []


@pytest.mark.parametrize("alias", ["path", "file", "source", "from"])
def test_every_declared_alias_for_src_resolves(alias):
    got, renamed = tools.apply_synonyms("copy_code", {alias: "a.py", "into": "b"})
    assert got["src"] == "a.py" and renamed == [f"{alias} -> src"]


def test_the_destination_has_aliases_too():
    got, _ = tools.apply_synonyms("copy_code", {"src": "a.py", "dest": "b.txt"})
    assert got["into"] == "b.txt"


# --------------------------------------------------------------- the guard

def test_the_table_cannot_drift_from_the_specs():
    """Asserted at import, and asserted again here so the reason is written
    down: a synonym pointing at a parameter that no longer exists would rename
    a good argument into oblivion and fail for a reason nobody could find."""
    for tool, table in tools.SYNONYMS.items():
        assert tool in tools.SPECS
        params = set(tools.SPECS[tool][0]) | set(tools.SPECS[tool][1])
        for real, aliases in table.items():
            assert real in params
            assert not (set(aliases) & params), (
                f"{tool}: an alias may not also be a real parameter")


def test_no_alias_is_claimed_by_two_parameters_of_one_tool():
    for tool, table in tools.SYNONYMS.items():
        seen: set = set()
        for aliases in table.values():
            assert not (seen & set(aliases)), f"{tool}: ambiguous alias"
            seen.update(aliases)


# ------------------------------------------------------- through dispatch

def box(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("**",),
                             allowed_new_files=("ANSWER.txt",))


def test_dispatch_accepts_the_synonym_and_says_so(tmp_path):
    ctx = box(tmp_path)
    out = tools.dispatch(ctx, "copy_code",
                         {"path": "a.py", "into": "ANSWER.txt", "name": "f"})
    assert out.ok, out.feedback
    assert (tmp_path / "ANSWER.txt").read_text(encoding="utf-8").strip()
    assert "src" in (out.feedback or ""), \
        "an agent that is not told the right name will use the wrong one again"


def test_dispatch_still_refuses_a_call_with_no_file_at_all(tmp_path):
    ctx = box(tmp_path)
    out = tools.dispatch(ctx, "copy_code", {"into": "ANSWER.txt", "name": "f"})
    assert not out.ok and out.invalid_call
