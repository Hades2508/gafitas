"""F-135: refusing the same edit twice costs a turn and buys nothing.

Phase 5 p5r3, nine runs: 69 edit attempts, 5 accepted, 64 refused -- and **31 of
the 64 were byte-identical repeats of a payload already refused in the same run
against the same file**. One run sent the same bytes fourteen times; another sent
one 195-character payload at turns 6, 7, 9 and 19. Each cost a turn to be told
the same thing.

The danger in fixing this is the opposite error: blocking a legitimate retry.
Most of the 64 refusals were DIFFERENT payloads, and that is the loop working.
So the memory is keyed on the attempt AND on a digest of the file underneath,
and every test below that matters is about something being ALLOWED through.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_REPEATED_REJECTED_EDIT
from localprog.rejection import RejectionMemory, attempt_signature, target_state

GOOD = "def f():\n    return 1\n"
BROKEN = "def f(\n"


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "a.py").write_text(GOOD, encoding="utf-8")
    (tmp_path / "b.py").write_text(GOOD, encoding="utf-8")
    box = tools.ToolContext(root=tmp_path, write_scope=("**",),
                            allowed_new_files=("new.py",))
    box.rejection_memory = RejectionMemory()
    box.opened.update({"a.py", "b.py"})
    return box


def send(ctx, **kw):
    return tools.dispatch(ctx, "replace_file", kw)


# --------------------------------------------------------- the thing it stops

def test_an_identical_refused_edit_is_not_run_again(ctx):
    first = send(ctx, path="a.py", content=BROKEN)
    assert not first.ok and first.code == "ERROR_SYNTAX_AFTER_EDIT"

    second = send(ctx, path="a.py", content=BROKEN)
    assert not second.ok
    assert second.code == ERROR_REPEATED_REJECTED_EDIT
    assert "ERROR_SYNTAX_AFTER_EDIT" in second.feedback, \
        "the original reason must travel with the refusal"
    assert "NO ha cambiado" in second.feedback


def test_the_refusal_counts_the_repeats(ctx):
    for _ in range(4):
        send(ctx, path="a.py", content=BROKEN)
    last = send(ctx, path="a.py", content=BROKEN)
    assert "veces" in last.feedback


def test_it_never_says_what_to_write_instead(ctx):
    send(ctx, path="a.py", content=BROKEN)
    again = send(ctx, path="a.py", content=BROKEN)
    # The harness does not know the answer. A guess dressed as guidance would be
    # worse than silence, and would be a hint the model did not earn.
    assert "CONTENIDO" in again.feedback or "SITIO" in again.feedback

    # The only code in the message is the model's OWN payload, quoted back by
    # the syntax report. Asserting "no code at all" was wrong and would have
    # forbidden echoing the broken line, which is the one genuinely useful
    # thing the refusal carries.
    quoted = {line.split("|", 1)[-1].strip()
              for line in again.feedback.splitlines() if "|" in line}
    # The caret that points at the column is part of the report's shape, not
    # code. Written out rather than filtered silently, because the first version
    # of this assertion failed on it and the failure was the test's, not the
    # message's.
    quoted -= {"", "^"}
    sent = {line.strip() for line in BROKEN.splitlines() if line.strip()}
    assert quoted <= sent, \
        "every line of code shown must be one the model itself sent"


# ------------------------------------------------- everything it must NOT stop

def test_a_different_payload_is_allowed_through(ctx):
    send(ctx, path="a.py", content=BROKEN)
    other = send(ctx, path="a.py", content="def g(\n")
    assert other.code == "ERROR_SYNTAX_AFTER_EDIT", \
        "a different attempt must be judged on its own merits"


def test_a_corrected_payload_is_allowed_through(ctx):
    send(ctx, path="a.py", content=BROKEN)
    fixed = send(ctx, path="a.py", content="def f():\n    return 2\n")
    assert fixed.ok, fixed.feedback


def test_a_different_target_is_allowed_through(ctx):
    send(ctx, path="a.py", content=BROKEN)
    elsewhere = send(ctx, path="b.py", content=BROKEN)
    assert elsewhere.code == "ERROR_SYNTAX_AFTER_EDIT"


def test_the_same_edit_is_reconsidered_once_the_file_changes(ctx, tmp_path):
    """The property that makes this safe to apply automatically.

    A retry that follows a change is a legitimate retry, and the memory is
    keyed on the file's digest so it stops applying the moment the file moves.
    """
    payload = "import os\ndef f():\n    return os.sep\n"
    (tmp_path / "a.py").write_text("def f(\n", encoding="utf-8")   # unparseable
    first = send(ctx, path="a.py", content=payload)
    assert first.ok

    # put it back to a state where that same payload was never tried
    (tmp_path / "a.py").write_text(GOOD, encoding="utf-8")
    again = send(ctx, path="a.py", content=payload)
    assert again.ok, "the file changed, so the attempt is new"


def test_a_reads_and_searches_are_never_remembered(ctx):
    """Only writes. Re-reading a file is cheap and sometimes exactly right."""
    for _ in range(3):
        out = tools.dispatch(ctx, "read_file", {"path": "a.py"})
        assert out.ok


def test_trailing_whitespace_is_the_same_attempt(ctx):
    send(ctx, path="a.py", content=BROKEN)
    padded = send(ctx, path="a.py", content=BROKEN + "\n\n")
    assert padded.code == ERROR_REPEATED_REJECTED_EDIT, \
        "the same body with a newline added is the same body"


def test_an_accepted_edit_is_not_remembered_as_a_refusal(ctx):
    good = send(ctx, path="a.py", content="def f():\n    return 3\n")
    assert good.ok
    # the same content is now identical to the file, which is a different
    # refusal, not the remembered one
    repeat = send(ctx, path="a.py", content="def f():\n    return 3\n")
    assert repeat.code == "ERROR_NOTHING_CHANGED"


# ------------------------------------------------------------- the signature

def test_the_signature_separates_address_from_payload():
    a = attempt_signature("edit", {"path": "x.py", "old": "a", "new": "b"})
    b = attempt_signature("edit", {"path": "y.py", "old": "a", "new": "b"})
    c = attempt_signature("edit", {"path": "x.py", "old": "a", "new": "c"})
    assert a != b and a != c


def test_the_signature_ignores_argument_order():
    a = attempt_signature("edit", {"path": "x.py", "old": "a", "new": "b"})
    b = attempt_signature("edit", {"new": "b", "old": "a", "path": "x.py"})
    assert a == b


def test_an_unreadable_target_disables_the_memory(tmp_path):
    """A state nobody can establish must not become a state everything matches.
    Disabling is the safe direction: an extra turn beats a wrong block."""
    memory = RejectionMemory()
    memory.remember("sig", "UNREADABLE", code="X", detail="")
    assert memory.seen("sig", "UNREADABLE") is None


def test_an_absent_file_is_a_state_of_its_own(tmp_path):
    assert target_state(tmp_path, "nope.py") == "ABSENT"
    (tmp_path / "nope.py").write_text("x", encoding="utf-8")
    assert target_state(tmp_path, "nope.py") != "ABSENT"
