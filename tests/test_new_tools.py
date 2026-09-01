"""Regressions for the tool surface added after GAFITAS_INITIAL_AUDIT.

Every test here is tied to a finding. The point is not coverage for its own
sake: each one pins a defect that was found by reading the code, so that a
later refactor cannot quietly reintroduce it.

    F-02  the provider schema documented nothing
    F-03  no way to enumerate a directory
    F-04  no way to execute anything but the declared test suite
    F-05  write scope was an exact-path allowlist
    F-10  no way to report being blocked
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import errors, tools  # noqa: E402
from localprog.scope import WriteScope  # noqa: E402


def call(ctx, name, **kwargs):
    return tools.dispatch(ctx, name, kwargs)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text(
        "def double(n):\n    return n * 2\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text(
        "from pkg.mod import double\n\n\ndef test_double():\n    assert double(3) == 6\n",
        encoding="utf-8",
    )
    (tmp_path / "empty").mkdir()
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    return tmp_path


@pytest.fixture
def ctx(repo):
    return tools.ToolContext(
        root=repo,
        write_scope=("pkg/",),
        allowed_new_files=("tests/test_*.py",),
        acceptance_tests=("tests/test_mod.py",),
    )


# ------------------------------------------------------------------ F-03


def test_list_dir_lists_the_repository_root(ctx):
    out = call(ctx, "list_dir")
    assert out.ok
    assert "pkg/" in out.value["dirs"] and "tests/" in out.value["dirs"]
    assert any(f["name"] == "README.md" for f in out.value["files"])


def test_list_dir_hides_noise_directories(ctx):
    """__pycache__ and friends are never useful and crowd out real entries."""
    out = call(ctx, "list_dir")
    assert "__pycache__/" not in out.value["dirs"]


def test_list_dir_on_a_file_shows_its_directory_instead_of_crashing(ctx):
    """The exact bug that got list_dir deleted from the contract.

    In the previous runner this raised NotADirectoryError out of the tool, out
    of the loop, and out of main(). Here it is an ordinary, useful answer.
    """
    out = call(ctx, "list_dir", path="pkg/mod.py")
    assert out.ok
    assert out.value["path"] == "pkg"
    assert any(f["name"] == "mod.py" for f in out.value["files"])
    assert "fichero" in out.value["note"]


def test_list_dir_on_an_empty_directory_says_so(ctx):
    out = call(ctx, "list_dir", path="empty")
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_EMPTY_DIRECTORY


def test_list_dir_on_a_missing_path_is_a_tool_error(ctx):
    out = call(ctx, "list_dir", path="nope")
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_FILE_NOT_FOUND


def test_list_dir_cannot_escape_the_repository(ctx):
    out = call(ctx, "list_dir", path="../..")
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_PATH_OUTSIDE_REPO


def test_list_dir_truncates_a_huge_directory_and_says_how_many(ctx, repo):
    big = repo / "many"
    big.mkdir()
    for i in range(tools.MAX_DIR_ENTRIES + 25):
        (big / f"f{i:04}.txt").write_text("x", encoding="utf-8")
    out = call(ctx, "list_dir", path="many")
    assert out.ok
    shown = len(out.value["dirs"]) + len(out.value["files"])
    assert shown == tools.MAX_DIR_ENTRIES
    assert errors.ERROR_TOO_MANY_ENTRIES in out.value["note"]


# ------------------------------------------------------------------ F-04


def test_run_executes_python_and_returns_its_output(ctx):
    out = call(ctx, "run", argv=["python", "-c", "print(6 * 7)"])
    assert out.ok
    assert out.value["exit_code"] == 0
    assert "42" in out.value["output"]


def test_run_reports_a_nonzero_exit_without_failing_the_loop(ctx):
    out = call(ctx, "run", argv=["python", "-c", "raise SystemExit(3)"])
    assert out.ok  # the CALL succeeded; the COMMAND failed, and that is data
    assert out.value["exit_code"] == 3


def test_run_can_import_the_repository_under_test(ctx):
    """cwd is the workspace, which is what makes observation possible at all."""
    out = call(ctx, "run", argv=["python", "-c", "from pkg.mod import double; print(double(21))"])
    assert out.ok and "42" in out.value["output"]


def test_run_refuses_an_executable_outside_the_allowlist(ctx):
    out = call(ctx, "run", argv=["curl", "https://example.com"])
    assert not out.ok and out.tool_error
    assert out.code == errors.ERROR_COMMAND_NOT_ALLOWED
    assert "python" in out.feedback  # told what it may use instead


def test_run_refuses_a_git_subcommand_that_mutates(ctx):
    """The workspace is the rollback, so an agent that could commit or check
    out could defeat the evidence trail without leaving the sandbox."""
    out = call(ctx, "run", argv=["git", "commit", "-m", "x"])
    assert not out.ok and out.code == errors.ERROR_COMMAND_NOT_ALLOWED
    assert call(ctx, "run", argv=["git", "checkout", "."]).code == errors.ERROR_COMMAND_NOT_ALLOWED
    assert call(ctx, "run", argv=["git", "push"]).code == errors.ERROR_COMMAND_NOT_ALLOWED


def test_run_has_no_shell_so_metacharacters_are_inert(ctx):
    """A model writing shell syntax gets it back as a literal string, not a
    second command. This is the property that makes an argv list safe."""
    out = call(ctx, "run", argv=["python", "-c", "print('a && rm -rf / ; b')"])
    assert out.ok and "a && rm -rf / ; b" in out.value["output"]


def test_run_rejects_a_command_string_instead_of_a_list(ctx):
    out = call(ctx, "run", argv="python -c 'print(1)'")
    assert not out.ok and out.invalid_call
    assert out.code == errors.ERROR_BAD_ARGUMENTS
    assert "lista" in out.feedback


def test_run_rejects_an_empty_argv(ctx):
    assert call(ctx, "run", argv=[]).invalid_call


def test_run_times_out_without_hanging_the_harness(ctx):
    out = call(ctx, "run", argv=["python", "-c", "import time; time.sleep(30)"], timeout=1)
    assert out.ok
    assert out.value["timed_out"] is True
    assert out.value["exit_code"] is None


def test_run_records_every_command_for_reproduction(ctx):
    call(ctx, "run", argv=["python", "-c", "print(1)"])
    call(ctx, "run", argv=["python", "-c", "print(2)"])
    assert [c["argv"][-1] for c in ctx.commands_run] == ["print(1)", "print(2)"]


def test_run_output_is_bounded(ctx):
    out = call(ctx, "run", argv=["python", "-c", "print('x' * 100000)"])
    assert out.ok and len(out.value["output"]) <= tools.RUN_OUTPUT_TAIL + 200
    assert "truncada" in out.value.get("note", "")


def test_run_pytest_works_and_is_not_confused_with_python(ctx):
    out = call(ctx, "run", argv=["pytest", "-q", "tests/test_mod.py"])
    assert out.ok and out.value["exit_code"] == 0


# ------------------------------------------------------------------ F-05


def test_a_directory_scope_permits_a_file_the_mission_never_named(ctx):
    """The whole point of F-05: creating a module nobody predicted."""
    out = call(ctx, "write_file", path="pkg/helper.py", content="X = 1\n")
    assert out.ok
    assert (ctx.root / "pkg" / "helper.py").exists()


def test_a_directory_scope_reaches_into_subdirectories(ctx):
    out = call(ctx, "write_file", path="pkg/deep/inner.py", content="Y = 2\n")
    assert out.ok


def test_a_glob_scope_permits_only_matching_names(ctx):
    assert call(ctx, "write_file", path="tests/test_new.py", content="def test_x():\n    pass\n").ok
    denied = call(ctx, "write_file", path="tests/helper.py", content="Z = 3\n")
    assert not denied.ok and denied.code == errors.ERROR_NOT_IN_WRITE_SCOPE


def test_out_of_scope_write_is_still_refused(ctx):
    out = call(ctx, "write_file", path="README2.md", content="no\n")
    assert not out.ok and out.code == errors.ERROR_NOT_IN_WRITE_SCOPE
    assert "pkg/" in out.feedback  # shown the scope in the mission's own words


def test_allowed_new_files_does_not_widen_editing(ctx, repo):
    """A mission that says 'you may ADD a test' must not thereby permit
    rewriting an existing one."""
    out = call(ctx, "edit", path="tests/test_mod.py", old="== 6", new="== 7")
    assert not out.ok and out.code == errors.ERROR_NOT_IN_WRITE_SCOPE


def test_containment_still_beats_a_permissive_scope(repo):
    """A scope of 'everything' does not disable the guard: containment and
    write scope are different layers, and only the outer one is security."""
    ctx = tools.ToolContext(root=repo, write_scope=("**",), allowed_new_files=("**",))
    out = call(ctx, "write_file", path="../escaped.py", content="X = 1\n")
    assert not out.ok
    assert out.code == errors.ERROR_PATH_OUTSIDE_REPO
    assert not (repo.parent / "escaped.py").exists()


def test_scope_matching_does_not_cross_directories_on_a_single_star():
    s = WriteScope(("src/*.py",))
    assert s.allows("src/a.py", creating=False)
    assert not s.allows("src/deep/a.py", creating=False)


def test_scope_double_star_does_cross_directories():
    s = WriteScope(("src/**/*.py",))
    assert s.allows("src/a.py", creating=False)
    assert s.allows("src/deep/a.py", creating=False)


def test_backslash_paths_are_accepted_in_scope_and_in_calls(ctx):
    out = call(ctx, "write_file", path="pkg\\windows.py", content="W = 1\n")
    assert out.ok and (ctx.root / "pkg" / "windows.py").exists()


# ------------------------------------------------------------------ F-10


def test_finish_blocked_is_accepted_with_no_edits(ctx):
    """Requirement 14: detect when it cannot continue, and say so."""
    out = call(ctx, "finish", summary="falta la dependencia foo", status="BLOCKED")
    assert out.ok and out.value == "FINISHED[BLOCKED]"
    assert ctx.finish_status == "BLOCKED"
    assert ctx.finish_summary == "falta la dependencia foo"


def test_finish_no_change_is_accepted_with_no_edits(ctx):
    out = call(ctx, "finish", summary="ya estaba hecho", status="NO_CHANGE")
    assert out.ok and ctx.finish_status == "NO_CHANGE"


def test_finish_done_without_edits_is_still_refused(ctx):
    """The original check earned its keep: an agent that has done nothing and
    believes it is done must not be able to say DONE."""
    out = call(ctx, "finish", summary="listo")
    assert not out.ok and out.code == errors.ERROR_NOTHING_CHANGED
    assert "NO_CHANGE" in out.feedback and "BLOCKED" in out.feedback


def test_finish_done_after_an_edit_records_the_status(ctx):
    call(ctx, "edit", path="pkg/mod.py", old="n * 2", new="n + n")
    call(ctx, "run_tests")  # F-29: DONE requires having looked
    out = call(ctx, "finish", summary="hecho")
    assert out.ok and ctx.finish_status == "DONE"


def test_finish_rejects_an_unknown_status(ctx):
    out = call(ctx, "finish", summary="x", status="MAYBE")
    assert not out.ok and out.invalid_call
    assert "DONE" in out.feedback


def test_finish_rejects_an_empty_summary(ctx):
    assert call(ctx, "finish", summary="   ", status="BLOCKED").invalid_call


# ------------------------------------------------------------------ F-02


def test_every_tool_is_documented_and_every_parameter_too():
    schema = tools.native_schema()
    assert {f["function"]["name"] for f in schema} == set(tools.SPECS)
    for entry in schema:
        fn = entry["function"]
        assert fn["description"] != fn["name"], f"{fn['name']} is undocumented"
        assert len(fn["description"]) > 40, f"{fn['name']} description is a stub"
        for param, spec in fn["parameters"]["properties"].items():
            assert spec.get("description"), f"{fn['name']}.{param} is undocumented"


def test_documentation_cannot_drift_from_the_tool_set():
    """I3, extended to the prose: TOOL_DOC and SPECS are asserted equal at
    import time, so a tool added without documentation fails to import."""
    assert set(tools.TOOL_DOC) == set(tools.SPECS)
    every_param = {p for req, opt in tools.SPECS.values() for p in req + opt}
    assert every_param <= set(tools.PARAM_DOC)


def test_finish_status_enum_is_published_in_the_schema():
    entry = next(f for f in tools.native_schema() if f["function"]["name"] == "finish")
    assert entry["function"]["parameters"]["properties"]["status"]["enum"] == list(
        tools.FINISH_STATUSES
    )


# ------------------------------------------------- path normalisation (F-03)


@pytest.mark.parametrize(
    "given",
    [".", "", "./", "/", "pkg/.."],
)
def test_list_dir_understands_every_spelling_of_the_root(ctx, given):
    """The guard rejects a '.' path component, which is right for '..' and
    wrong for the root. A model told the repository root is outside the
    repository has been taught something false about its own sandbox."""
    if given == "pkg/..":
        pytest.skip("'..' is containment's business, checked separately below")
    out = call(ctx, "list_dir", path=given)
    assert out.ok and out.value["path"] == "."


def test_a_leading_dot_slash_is_not_an_escape_attempt(ctx):
    out = call(ctx, "read_file", path="./pkg/mod.py")
    assert out.ok and "double" in out.value


def test_an_interior_dot_segment_is_folded_away(ctx):
    out = call(ctx, "read_file", path="pkg/./mod.py")
    assert out.ok and "double" in out.value


def test_duplicate_slashes_are_folded_away(ctx):
    assert call(ctx, "read_file", path="pkg//mod.py").ok


@pytest.mark.parametrize(
    "escape",
    ["..", "../x.py", "./../x.py", "pkg/../../x.py", "/etc/passwd", "C:/Windows/x.py"],
)
def test_normalisation_never_opens_a_hole_in_containment(ctx, escape):
    """Everything the normaliser relaxes must still hit the guard."""
    out = call(ctx, "read_file", path=escape)
    assert not out.ok
    assert out.code in (errors.ERROR_PATH_OUTSIDE_REPO, errors.ERROR_FILE_NOT_FOUND)


# ------------------------------------------------- F-26/F-27: editing at scale


def test_replace_lines_replaces_a_range(ctx):
    out = call(ctx, "replace_lines", path="pkg/mod.py", start=1, end=2,
               content="def double(n):\n    return n + n")
    assert out.ok, out.feedback
    assert (ctx.root / "pkg" / "mod.py").read_text(encoding="utf-8") == "def double(n):\n    return n + n\n"


def test_replace_lines_needs_no_exact_text(ctx, repo):
    """The whole reason it exists. The dogfood run failed six times running
    because edit needs the old bytes reproduced exactly, and on a 48k-character
    file the read that showed them had already been elided. Line numbers come
    free with read_file and list_symbols and survive elision."""
    body = "\n".join(f"LINE{i}" for i in range(1, 101)) + "\n"
    (repo / "pkg" / "long.py").write_text(body, encoding="utf-8")
    out = call(ctx, "replace_lines", path="pkg/long.py", start=50, end=50, content="CAMBIADA")
    assert out.ok
    lines = (repo / "pkg" / "long.py").read_text(encoding="utf-8").splitlines()
    assert lines[48] == "LINE49" and lines[49] == "CAMBIADA" and lines[50] == "LINE51"
    assert len(lines) == 100


def test_replace_lines_refuses_to_break_syntax(ctx):
    before = (ctx.root / "pkg" / "mod.py").read_text(encoding="utf-8")
    out = call(ctx, "replace_lines", path="pkg/mod.py", start=1, end=1, content="def broken(")
    assert not out.ok and out.code == errors.ERROR_SYNTAX_AFTER_EDIT
    assert (ctx.root / "pkg" / "mod.py").read_text(encoding="utf-8") == before


def test_replace_lines_respects_write_scope(ctx):
    out = call(ctx, "replace_lines", path="tests/test_mod.py", start=1, end=1, content="# no")
    assert not out.ok and out.code == errors.ERROR_NOT_IN_WRITE_SCOPE


def test_replace_lines_rejects_an_impossible_range(ctx):
    out = call(ctx, "replace_lines", path="pkg/mod.py", start=999, end=1000, content="x = 1")
    assert not out.ok and out.code == errors.ERROR_BAD_RANGE
    assert "2 lineas" in out.feedback or "lineas" in out.feedback


def test_replace_lines_rejects_non_integer_lines(ctx):
    out = call(ctx, "replace_lines", path="pkg/mod.py", start="1", end=2, content="x = 1")
    assert not out.ok and out.invalid_call


def test_replace_lines_clamps_an_end_past_the_file(ctx):
    out = call(ctx, "replace_lines", path="pkg/mod.py", start=1, end=9999,
               content="def double(n):\n    return 0")
    assert out.ok
    assert (ctx.root / "pkg" / "mod.py").read_text(encoding="utf-8").endswith("return 0\n")


def test_a_failed_edit_shows_the_text_as_it_actually_is(ctx):
    """F-27. 'Not found, go read the file' was the advice that produced the
    read-guess-fail loop. Showing the nearest region makes a wrong indent
    visible immediately."""
    out = call(ctx, "edit", path="pkg/mod.py",
               old="def double(n):\n        return n * 2", new="x")  # wrong indent
    assert not out.ok and out.code == errors.ERROR_NO_MATCH
    assert "return n * 2" in out.feedback, "must show the real text"
    assert "replace_lines" in out.feedback, "must name the line-based way out"


def test_the_nearest_hint_is_omitted_when_nothing_is_close(ctx):
    """A confident pointer at unrelated code would be worse than none."""
    out = call(ctx, "edit", path="pkg/mod.py", old="zzzzzz_qqqqq_wwwww", new="x")
    assert not out.ok and out.code == errors.ERROR_NO_MATCH
    assert "read_file" in out.feedback


# ------------------------------------------------------- F-33: grep context


def test_grep_returns_surrounding_lines_when_asked(ctx):
    """One call instead of two. An agent looking for one function in a 48k
    file spent 24 of its 40 turns on read_file, because grep told it where the
    code was and nothing about what the code said."""
    out = call(ctx, "grep", pattern="def double", glob="**/*.py", context=2)
    assert out.ok
    hit = out.value["hits"][0]
    assert "context" in hit
    assert "return n * 2" in hit["context"], "the body, not just the signature"
    assert "\t" in hit["context"], "numbered, so it can be fed straight to replace_lines"


def test_grep_without_context_is_unchanged(ctx):
    out = call(ctx, "grep", pattern="def double", glob="**/*.py")
    assert out.ok and "context" not in out.value["hits"][0]


def test_grep_context_is_bounded(ctx, repo):
    """Context multiplies output by 2N+1, so it needs its own ceiling or it
    becomes the very thing that blew the window in F-25."""
    body = "\n".join(f"def f{i}():\n    return {i}" for i in range(400))
    (repo / "pkg" / "many.py").write_text(body, encoding="utf-8")
    out = call(ctx, "grep", pattern="return", glob="pkg/many.py", context=20)
    assert out.ok
    size = sum(len(h.get("context", "")) + len(h["text"]) for h in out.value["hits"])
    assert size <= tools.MAX_TOOL_PAYLOAD_CHARS * 1.1


def test_grep_rejects_an_absurd_context(ctx):
    assert call(ctx, "grep", pattern="x", context=999).invalid_call
    assert call(ctx, "grep", pattern="x", context="two").invalid_call


def test_grep_context_null_is_accepted(ctx):
    """Providers send null for an omitted optional argument."""
    out = call(ctx, "grep", pattern="def double", context=None)
    assert out.ok
