"""A memory that punishes correction is worse than no memory.

F-178. `attempt_signature` stripped trailing whitespace from every payload key,
including `old` -- and `old` is not a payload. It is a PATTERN, matched byte
for byte by `edit`, so "hello " and "hello" are different patterns and exactly
one of them can succeed against a file containing "hello".

Reproduced end to end:

    edit(old="hello ", new="adios")   -> not found, correctly
    edit(old="hello ", new="adios")   -> ERROR_REPEATED_REJECTED_EDIT, correctly
    edit(old="hello",  new="adios")   -> ERROR_REPEATED_REJECTED_EDIT

The third is the agent doing exactly what the first error asked of it: it
dropped the stray space, and the pattern it sent DOES occur in the file. It was
refused on the grounds that it had already tried that, and the file could not
be edited for the rest of the run.

The stripping is right for what an attempt WRITES -- resending the same body
with a newline added is the same body -- and wrong for what it MATCHES. The two
are now separate lists.
"""

from __future__ import annotations

from localprog import rejection, tools


def build(tmp_path, body: str = "hello\n"):
    (tmp_path / "a.py").write_text(body, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("a.py",),
                             rejection_memory=rejection.RejectionMemory())


def test_correcting_the_pattern_is_allowed(tmp_path):
    ctx = build(tmp_path)
    tools.dispatch(ctx, "edit", {"path": "a.py", "old": "hello ", "new": "adios"})
    tools.dispatch(ctx, "edit", {"path": "a.py", "old": "hello ", "new": "adios"})

    corrected = tools.dispatch(ctx, "edit",
                               {"path": "a.py", "old": "hello", "new": "adios"})
    assert corrected.ok, (
        "the agent removed the stray space the error complained about and was "
        "told it had already tried that")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "adios\n"


def test_the_same_pattern_twice_is_still_remembered(tmp_path):
    """The memory has to keep working; this is what it is for."""
    ctx = build(tmp_path)
    tools.dispatch(ctx, "edit", {"path": "a.py", "old": "nope", "new": "x"})
    tools.dispatch(ctx, "edit", {"path": "a.py", "old": "nope", "new": "x"})
    third = tools.dispatch(ctx, "edit", {"path": "a.py", "old": "nope", "new": "x"})
    assert not third.ok
    assert third.code == "ERROR_REPEATED_REJECTED_EDIT"


def test_a_cosmetic_resend_of_the_written_body_is_still_caught(tmp_path):
    """Stripping is right for what is WRITTEN, and that half must survive."""
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("b.py",),
                            rejection_memory=rejection.RejectionMemory())
    tools.dispatch(ctx, "write_file", {"path": "b.py", "content": "y = 2"})
    tools.dispatch(ctx, "write_file", {"path": "b.py", "content": "y = 2"})
    again = tools.dispatch(ctx, "write_file",
                           {"path": "b.py", "content": "y = 2\n"})
    assert not again.ok, "the same body with a newline added is the same body"


def test_the_signature_separates_pattern_from_payload():
    """Directly, so the two lists cannot silently merge again."""
    pattern_a = rejection.attempt_signature("edit", {"path": "a", "old": "x ",
                                                     "new": "y"})
    pattern_b = rejection.attempt_signature("edit", {"path": "a", "old": "x",
                                                     "new": "y"})
    assert pattern_a != pattern_b, "old is matched exactly; do not normalise it"

    body_a = rejection.attempt_signature("write_file", {"path": "a",
                                                        "content": "y"})
    body_b = rejection.attempt_signature("write_file", {"path": "a",
                                                        "content": "y\n"})
    assert body_a == body_b, "a trailing newline on a written body is cosmetic"
