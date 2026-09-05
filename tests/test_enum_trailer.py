"""F-117: a correct finish, refused 38 times, for the punctuation around it.

ministral-3:3b is the engine that scores 92% bare and 68% through GAFITAS. Of
its 56 invalid calls in cohort 3, 44 were ERROR_BAD_ARGUMENTS, and 38 of those
44 -- 86% -- were the same string:

    status='DONE"}\\n```'

It wrote a well-formed ``finish`` inside a fenced JSON block, and the argument
value carried the block's closers out with it. The status was DONE. The harness
called it an unknown status and refused, over and over, in runs that had already
produced the right answer.

The repair is narrow on purpose, and the narrowness is the whole design:

  - the value must START with exactly one member of a CLOSED enum;
  - everything after it must be structural -- quotes, braces, fences, commas,
    whitespace -- and nothing else;
  - one word character after the member and this declines, because at that
    point the model may have meant something the harness has not thought of.

It is never applied to free-text arguments. A file's content may legitimately
end in a brace, and "repairing" that would corrupt the file.
"""

from __future__ import annotations

import pytest

from localprog import tools


# ----------------------------------------------------- the repair in isolation

@pytest.mark.parametrize("raw", [
    'DONE"}\n```',          # the sealed string, verbatim
    'DONE"}',
    "DONE'",
    "DONE`",
    "DONE  \n",
    'DONE",',
    "DONE)]}",
])
def test_a_status_wearing_its_wrapper_is_still_that_status(raw):
    assert tools._enum_member(raw.upper(), tools.FINISH_STATUSES) == "DONE"


@pytest.mark.parametrize("member", tools.FINISH_STATUSES)
def test_every_member_of_the_enum_is_repairable(member):
    assert tools._enum_member(member + '"}', tools.FINISH_STATUSES) == member


@pytest.mark.parametrize("raw", ["DONE_MAYBE", "DONEISH", "DONE OR BLOCKED",
                                 "ALMOST_DONE", "FINISHED", ""])
def test_a_word_character_after_the_member_declines(raw):
    """The line between a repair and a rescue. `DONE_MAYBE` is not DONE with
    punctuation on it; it is a different word, and guessing would be inventing
    an outcome the model did not claim."""
    assert tools._enum_member(raw, tools.FINISH_STATUSES) is None


def test_a_non_string_declines_rather_than_raising():
    assert tools._enum_member(None, tools.FINISH_STATUSES) is None
    assert tools._enum_member(7, tools.FINISH_STATUSES) is None


def test_a_longer_member_is_not_swallowed_by_a_shorter_prefix():
    """NO_CHANGE must not be read as some prefix of another member. Checked
    because the matcher walks the tuple in order."""
    assert tools._enum_member("NO_CHANGE", tools.FINISH_STATUSES) == "NO_CHANGE"


# -------------------------------------------------------------- through finish

def ctx_with_a_change(tmp_path):
    (tmp_path / "ANSWER.txt").write_text("x\n", encoding="utf-8")
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**",),
                            allowed_new_files=("ANSWER.txt",))
    ctx.changed_files.add("ANSWER.txt")
    return ctx


def test_the_exact_ministral_call_is_now_accepted(tmp_path):
    ctx = ctx_with_a_change(tmp_path)
    out = tools.dispatch(ctx, "finish",
                         {"summary": "copie la funcion",
                          "status": 'DONE"}\n```'})
    assert out.ok, out.feedback
    assert ctx.finish_status == "DONE"


def test_an_unrecognisable_status_is_still_refused(tmp_path):
    ctx = ctx_with_a_change(tmp_path)
    out = tools.dispatch(ctx, "finish",
                         {"summary": "s", "status": "MOSTLY_DONE"})
    assert not out.ok and out.code == "ERROR_BAD_ARGUMENTS"


def test_the_repair_does_not_reach_the_summary(tmp_path):
    """The summary is free text. A summary that happens to start with the word
    DONE must arrive as written."""
    ctx = ctx_with_a_change(tmp_path)
    out = tools.dispatch(ctx, "finish",
                         {"summary": 'DONE"}```', "status": "DONE"})
    assert out.ok
    assert ctx.finish_summary == 'DONE"}```'


def test_a_repaired_done_still_faces_every_other_gate(tmp_path):
    """The repair fixes the spelling of the status and nothing else. DONE with
    nothing changed is still refused -- that gate exists for a different
    reason and this must not open it."""
    ctx = tools.ToolContext(root=tmp_path, write_scope=("**",),
                            allowed_new_files=("ANSWER.txt",))
    out = tools.dispatch(ctx, "finish",
                         {"summary": "s", "status": 'DONE"}\n```'})
    assert not out.ok, "a repaired DONE with an empty diff must not pass"
