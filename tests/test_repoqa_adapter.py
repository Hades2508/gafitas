"""The RepoQA adapter, and above all that the needle's identity stays hidden.

The needle's name and path ARE the answer. An adapter that lets either reach
the ticket scores well and measures nothing, so those tests come first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import repoqa, work  # noqa: E402

NEEDLE = {
    "name": "_merge_string_group",
    "path": "src/black/trans.py",
    "description": (
        "1. **Purpose**: To combine adjacent strings into a single string within a "
        "line of code, ensuring that the merged result is syntactically correct. "
        "2. **Input**: A line of code and a list of indices."
    ),
    "start_line": 412,
    "end_line": 480,
    "start_byte": 15000,
    "end_byte": 17800,
    "global_start_line": 1412,
    "global_end_line": 1480,
    "global_start_byte": 55000,
    "global_end_byte": 57800,
}

CASE = repoqa.Case(
    language="python",
    repo="psf/black",
    description=NEEDLE["description"],
    needle_name=NEEDLE["name"],
)


@pytest.fixture
def ticket(tmp_path):
    return repoqa.build_ticket(CASE, tmp_path)


# --------------------------------------------------------- anti-contamination


def test_the_description_reaches_the_agent(ticket):
    assert "combine adjacent strings" in ticket.objective


def test_the_needle_name_never_reaches_the_agent(ticket):
    """The name IS the answer. RepoQA obfuscates the description precisely so
    the function cannot be identified by name."""
    assert "_merge_string_group" not in ticket.objective
    repoqa.assert_no_leak(NEEDLE, ticket)


def test_the_needle_path_never_reaches_the_agent(ticket):
    """Being told the file collapses the search to one small step."""
    assert "src/black/trans.py" not in ticket.objective
    assert "trans.py" not in ticket.objective


def test_a_leak_is_detected_rather_than_ignored(tmp_path):
    """The guard must actually fire, or it is decoration."""
    leaked = repoqa.build_ticket(CASE, tmp_path)
    leaked = type(leaked)(
        ticket_id=leaked.ticket_id, repo=leaked.repo,
        objective=leaked.objective + "\nHint: _merge_string_group",
        write_scope=(), allowed_new_files=(repoqa.ANSWER_FILE,), acceptance_tests=(),
    )
    with pytest.raises(ValueError, match="LEAK"):
        repoqa.assert_no_leak(NEEDLE, leaked)


def test_only_the_description_is_read_from_the_needle():
    assert repoqa.VISIBLE_NEEDLE_FIELDS == ("description",)
    for field in ("name", "path", "start_line", "end_line"):
        assert field in repoqa.FORBIDDEN_NEEDLE_FIELDS


# ------------------------------------------------------------- ticket shape


def test_the_agent_may_only_create_the_answer_file(ticket):
    """It has no reason to modify the repository, and forbidding it means a
    stray edit is refused by the tools rather than discovered in the diff."""
    assert ticket.scope.allows(repoqa.ANSWER_FILE, creating=True)
    assert not ticket.scope.allows("src/black/trans.py", creating=False)
    assert not ticket.scope.allows("setup.py", creating=True)


def test_there_are_no_acceptance_tests(ticket):
    """RepoQA is judged by its own evaluator, not by us."""
    assert ticket.acceptance_tests == ()
    assert ticket.full_suite is False


def test_the_objective_names_the_answer_file(ticket):
    assert repoqa.ANSWER_FILE in ticket.objective


def test_the_objective_names_the_language(ticket):
    assert "python" in ticket.objective


# ------------------------------------------------------------- materialising


def test_a_repository_is_written_and_committed(tmp_path):
    record = {"content": {
        "src/pkg/mod.py": "def alpha():\n    return 1\n",
        "README.md": "hello\n",
    }}
    root = repoqa.materialise_repo(record, tmp_path / "r")
    assert (root / "src" / "pkg" / "mod.py").read_text(encoding="utf-8").startswith("def alpha")
    assert (root / ".git").is_dir(), "a real git repo, so GAFITAS gets a worktree"


def test_reading_a_missing_answer_is_empty_not_an_error(tmp_path):
    assert repoqa.read_answer(tmp_path) == ""


def test_reading_the_answer_returns_it_verbatim(tmp_path):
    body = "def alpha():\n    return 1\n"
    (tmp_path / repoqa.ANSWER_FILE).write_text(body, encoding="utf-8")
    assert repoqa.read_answer(tmp_path) == body


# ------------------------------------------------------------ official shape


def test_the_output_row_matches_what_compute_score_consumes():
    row = repoqa.to_official_output(CASE, "def x(): pass", model="gafitas")
    assert row["repo"] == "psf/black"
    assert row["name"] == "_merge_string_group"
    assert row["language"] == "python"
    assert row["output"] == ["def x(): pass"]


def test_the_answer_is_passed_through_unmodified():
    """Cleaning up the answer would score our post-processing, not the agent."""
    messy = "Here you go:\n```python\ndef x():\n    pass\n```\nHope that helps."
    row = repoqa.to_official_output(CASE, messy, model="gafitas")
    assert row["output"][0] == messy


def test_outputs_are_written_as_jsonl(tmp_path):
    out = tmp_path / "o.jsonl"
    repoqa.write_outputs(out, [
        repoqa.to_official_output(CASE, "a", model="m"),
        repoqa.to_official_output(CASE, "b", model="m"),
    ])
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[1]["output"] == ["b"]


# --------------------------------------------------- recovering the answer


def test_the_answer_is_recovered_from_the_sealed_patch():
    """run_ticket disposes the workspace once it has a CANDIDATE, so the answer
    file is gone by the time the caller sees the result. It survives in the
    patch, where a new file is entirely additions."""
    patch = (
        "diff --git a/REPOQA_ANSWER.txt b/REPOQA_ANSWER.txt\n"
        "--- a/REPOQA_ANSWER.txt\n"
        "+++ b/REPOQA_ANSWER.txt\n"
        "@@ -0,0 +1,3 @@\n"
        "+def alpha(x):\n"
        "+    return x + 1\n"
        "+\n"
    )
    assert repoqa.answer_from_patch(patch) == "def alpha(x):\n    return x + 1\n"


def test_only_the_answer_file_is_recovered():
    """A patch touching other files must not contaminate the answer."""
    patch = (
        "diff --git a/src/other.py b/src/other.py\n"
        "--- a/src/other.py\n"
        "+++ b/src/other.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+CONTAMINATION = 1\n"
        "diff --git a/REPOQA_ANSWER.txt b/REPOQA_ANSWER.txt\n"
        "--- a/REPOQA_ANSWER.txt\n"
        "+++ b/REPOQA_ANSWER.txt\n"
        "@@ -0,0 +1,1 @@\n"
        "+def alpha(): pass\n"
    )
    recovered = repoqa.answer_from_patch(patch)
    assert "CONTAMINATION" not in recovered
    assert recovered == "def alpha(): pass"


def test_an_empty_patch_yields_an_empty_answer():
    assert repoqa.answer_from_patch("") == ""
    assert repoqa.answer_from_patch("(sin cambios)") == ""


def test_a_needle_named_after_a_common_word_is_not_a_false_leak():
    """A run died 19 cases in on a needle named `base`, because a description
    about a base directory naturally contains the word. A guard that fires on
    ordinary English gets switched off, which is worse than no guard.

    The description is sanctioned content -- RepoQA obfuscates it on purpose --
    so only what the ADAPTER adds is checked."""
    needle = {"name": "base", "path": "src/pkg/base.py",
              "description": "Resolves the base directory used as the base for all lookups."}
    case = repoqa.Case(language="python", repo="x/y",
                       description=needle["description"], needle_name="base")
    ticket = repoqa.build_ticket(case, Path("."))
    repoqa.assert_no_leak(needle, ticket)  # must not raise


def test_a_real_leak_is_still_caught_when_the_name_is_common():
    """Narrowing to the adapter's own text must not blind the guard."""
    needle = {"name": "base", "path": "src/pkg/base.py", "description": "Some description."}
    case = repoqa.Case(language="python", repo="x/y",
                       description=needle["description"], needle_name="base")
    good = repoqa.build_ticket(case, Path("."))
    leaked = type(good)(
        ticket_id=good.ticket_id, repo=good.repo,
        objective=good.objective + "\nThe function is called base.",
        write_scope=(), allowed_new_files=(repoqa.ANSWER_FILE,), acceptance_tests=(),
    )
    with pytest.raises(ValueError, match="LEAK"):
        repoqa.assert_no_leak(needle, leaked)


def test_a_path_added_by_the_adapter_is_still_caught():
    needle = {"name": "zzz_unique", "path": "src/pkg/base.py", "description": "Desc."}
    case = repoqa.Case(language="python", repo="x/y",
                       description="Desc.", needle_name="zzz_unique")
    good = repoqa.build_ticket(case, Path("."))
    leaked = type(good)(
        ticket_id=good.ticket_id, repo=good.repo,
        objective=good.objective + "\nLook in src/pkg/base.py",
        write_scope=(), allowed_new_files=(repoqa.ANSWER_FILE,), acceptance_tests=(),
    )
    with pytest.raises(ValueError, match="LEAK"):
        repoqa.assert_no_leak(needle, leaked)


def test_a_short_needle_name_is_not_matched_inside_a_tool_name():
    """The guard has now killed two runs, both the same mistake.

    A needle named 'is' occurs inside 'list_dir' and 'list_symbols' in the
    objective's own tool guidance, so a substring search finds the "leak" in
    the adapter's boilerplate and stops the benchmark at case 59. Before that,
    a needle named 'base' was matched inside a description about a base
    directory. Any short name is a substring of something.
    """
    case = repoqa.Case(language="typescript", repo="o/r",
                       description="devuelve verdadero cuando el valor es de ese tipo",
                       needle_name="is")
    ticket = repoqa.build_ticket(case, Path("."))
    repoqa.assert_no_leak({"name": "is", "path": "src/guards.ts",
                           "description": case.description}, ticket)


def test_a_real_leak_is_still_caught():
    """The boundary rule must not turn the guard off."""
    case = repoqa.Case(language="python", repo="o/r", description="hace algo",
                       needle_name="parse_header")
    ticket = repoqa.build_ticket(case, Path("."))
    leaked = work.Ticket(
        ticket_id=ticket.ticket_id, repo=ticket.repo,
        objective=ticket.objective + "\n(la funcion se llama parse_header)",
        write_scope=(), allowed_new_files=(repoqa.ANSWER_FILE,),
        acceptance_tests=(), full_suite=False, max_turns=10,
    )
    with pytest.raises(ValueError, match="LEAK"):
        repoqa.assert_no_leak({"name": "parse_header", "description": "hace algo"}, leaked)


def test_a_leaked_path_is_still_caught():
    case = repoqa.Case(language="python", repo="o/r", description="hace algo",
                       needle_name="f")
    ticket = repoqa.build_ticket(case, Path("."))
    leaked = work.Ticket(
        ticket_id=ticket.ticket_id, repo=ticket.repo,
        objective=ticket.objective + "\nMira en pkg/mod.py",
        write_scope=(), allowed_new_files=(repoqa.ANSWER_FILE,),
        acceptance_tests=(), full_suite=False, max_turns=10,
    )
    with pytest.raises(ValueError, match="LEAK"):
        repoqa.assert_no_leak({"name": "f", "path": "pkg/mod.py",
                               "description": "hace algo"}, leaked)
