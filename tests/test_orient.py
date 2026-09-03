"""The repository's shape, handed over instead of walked for.

Fifty matched runs of the reference engine spent 140 tool calls on list_dir --
turn one in 49 of 50 runs, still turn three in 24 of them -- on missions that
NAME the file they concern, and did not write until median turn 7. The working
instructions opened with "empieza por list_dir" and it did as it was told.

Walking a directory needs no reasoning. It costs the harness milliseconds and
the engine three inferences.
"""

from __future__ import annotations

from pathlib import Path

from localprog import orient, work


def build(tmp_path: Path, layout: dict[str, str]) -> Path:
    for rel, body in layout.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


def test_it_names_the_directories_and_the_files(tmp_path):
    build(tmp_path, {"pkg/a.py": "x = 1\n", "pkg/b.py": "y = 2\n",
                     "tests/test_a.py": "pass\n"})
    got = orient.repo_map(tmp_path)
    assert "pkg/" in got and "a.py" in got and "b.py" in got
    assert "tests/" in got and "test_a.py" in got


def test_it_is_deterministic(tmp_path):
    """Two runs of the same mission must see the same map, or they are not two
    runs of the same mission."""
    build(tmp_path, {f"pkg/m{i}.py": "x\n" for i in range(12)})
    assert orient.repo_map(tmp_path) == orient.repo_map(tmp_path)


def test_it_cannot_depend_on_the_objective(tmp_path):
    """The map is orientation, not retrieval. If it could be steered toward the
    answer it would be a search channel wearing orientation's clothes -- and it
    would make every run's context depend on what was being asked."""
    import inspect
    signature = inspect.signature(orient.repo_map)
    assert "objective" not in signature.parameters
    assert "query" not in signature.parameters


def test_it_stays_inside_its_budget(tmp_path):
    build(tmp_path, {f"d{d}/f{i}.py": "x\n" for d in range(40) for i in range(20)})
    got = orient.repo_map(tmp_path)
    assert len(got) <= orient.MAX_CHARS + 200, len(got)


def test_a_truncated_map_says_it_is_truncated(tmp_path):
    """An agent that believes it has seen everything will not go looking for the
    rest. Bounded is fine; misleading is not."""
    build(tmp_path, {f"d{d}/f{i}.py": "x\n" for d in range(40) for i in range(3)})
    got = orient.repo_map(tmp_path)
    assert "directorios mas" in got and "list_dir" in got


def test_files_beyond_the_per_directory_cap_are_counted_not_hidden(tmp_path):
    build(tmp_path, {f"pkg/m{i:02d}.py": "x\n" for i in range(20)})
    got = orient.repo_map(tmp_path)
    assert "(20)" in got and "mas" in got


def test_an_empty_repository_produces_nothing_rather_than_a_heading(tmp_path):
    assert orient.repo_map(tmp_path) == ""


def test_it_skips_what_the_index_skips(tmp_path):
    """One definition of "what is in this repository", shared with the index --
    a map that advertised .git or a virtualenv would send the agent into them."""
    build(tmp_path, {"pkg/a.py": "x\n", ".git/config": "x\n",
                     "__pycache__/a.pyc": "x\n"})
    got = orient.repo_map(tmp_path)
    assert ".git" not in got and "__pycache__" not in got


# ------------------------------------------------------------ it reaches the run

def test_the_objective_carries_the_map(tmp_path):
    build(tmp_path, {"pkg/a.py": "x = 1\n"})
    ticket = work.Ticket(ticket_id="t", repo=tmp_path, objective="arregla algo",
                         write_scope=("pkg/",), allowed_new_files=(),
                         acceptance_tests=(), full_suite=False, max_turns=5)
    text = work.objective_text(ticket, tmp_path)
    assert "arregla algo" in text
    assert "ESTRUCTURA DEL REPOSITORIO" in text and "a.py" in text


def test_without_a_root_the_objective_is_exactly_what_it_was(tmp_path):
    ticket = work.Ticket(ticket_id="t", repo=tmp_path, objective="arregla algo",
                         write_scope=("pkg/",), allowed_new_files=(),
                         acceptance_tests=(), full_suite=False, max_turns=5)
    assert "ESTRUCTURA" not in work.objective_text(ticket)


def test_the_instructions_no_longer_open_with_a_directory_walk(tmp_path):
    ticket = work.Ticket(ticket_id="t", repo=tmp_path, objective="o",
                         write_scope=("pkg/",), allowed_new_files=(),
                         acceptance_tests=(), full_suite=False, max_turns=5)
    prompt = work.system_prompt(ticket, "A")
    assert "empieza por list_dir" not in prompt
    assert "estructura del repositorio" in prompt.lower()
