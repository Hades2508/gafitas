"""F-94: a refusal completes the agent's own decision before proposing its own.

Two ministral-3:3b runs spent their entire twenty-turn budget here.
``_write_lock_file`` resolved ``Installer._write_lock_file`` on turn 3 and then
called finish(DONE) six times; every refusal named the harness's turn-zero
candidate, which was a different file, and the copy was never made.

The distinction these tests hold in place:

    reached_symbols     what the AGENT resolved, against the real file
    opening_candidate   what the HARNESS ranked before the run started

The first outranks the second, because overriding a correct decision the agent
already made is the harness putting its own ceiling on the model. The turn-zero
candidate is not removed -- its ablation is worth 60% against 20% -- it just
stops speaking over the agent.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ToolError

SOURCE = '''"""Installer."""


class Installer:
    def update(self):
        return 1

    def _write_lock_file(self, force=False):
        return force
'''

OTHER = '''def unrelated():
    return 0
'''


@pytest.fixture()
def ctx(tmp_path):
    (tmp_path / "installer.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "other.py").write_text(OTHER, encoding="utf-8")
    return tools.ToolContext(
        root=tmp_path,
        write_scope=("ANSWER.txt",),
        allowed_new_files=("ANSWER.txt",),
        opening_candidate=("other.py", "unrelated"),
    )


def refusal(ctx):
    with pytest.raises(ToolError) as exc:
        tools.finish(ctx, summary="ya esta", status="DONE")
    return str(exc.value)


def test_without_a_reach_the_turn_zero_candidate_still_speaks(ctx):
    message = refusal(ctx)
    assert "other.py" in message and "unrelated" in message


def test_a_reached_symbol_is_what_the_refusal_names(ctx):
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    message = refusal(ctx)
    assert "Installer._write_lock_file" in message
    assert "installer.py" in message
    assert "unrelated" not in message, "the harness talked over the agent"


def test_the_refusal_names_a_complete_call_not_a_kind_of_call(ctx):
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    message = refusal(ctx)
    assert "copy_code(src='installer.py'" in message
    assert "into='ANSWER.txt'" in message
    assert "name='Installer._write_lock_file')" in message


def test_writing_it_yourself_is_still_offered(ctx):
    """Additive, not a funnel: copying is named first, authoring stays open."""
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    message = refusal(ctx)
    assert "write_file(path='ANSWER.txt'" in message


def test_the_most_recent_reach_wins(ctx):
    tools.read_symbol(ctx, path="installer.py", name="Installer.update")
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    message = refusal(ctx)
    assert "Installer._write_lock_file" in message
    assert "Installer.update" not in message


def test_re_reading_an_earlier_symbol_moves_it_back_to_the_front(ctx):
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    tools.read_symbol(ctx, path="installer.py", name="Installer.update")
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    assert ctx.reached_symbols == [("installer.py", "Installer.update"),
                                   ("installer.py", "Installer._write_lock_file")]


def test_a_failed_read_is_not_a_reach(ctx):
    with pytest.raises(ToolError):
        tools.read_symbol(ctx, path="installer.py", name="does_not_exist")
    assert ctx.reached_symbols == []


def test_a_bare_name_is_recorded_resolved_so_the_copy_call_works(ctx):
    """The agent typed a bare method name; the refusal must quote the qualified
    one, because that is what copy_code will accept."""
    tools.read_symbol(ctx, path="installer.py", name="_write_lock_file")
    assert ctx.reached_symbols == [("installer.py", "Installer._write_lock_file")]


def test_the_named_call_is_one_the_harness_would_actually_accept(ctx):
    """The whole point is that the next turn works. So run it."""
    tools.read_symbol(ctx, path="installer.py", name="Installer._write_lock_file")
    refusal(ctx)
    out = tools.copy_code(ctx, src="installer.py", into="ANSWER.txt",
                          name="Installer._write_lock_file")
    body = (ctx.root / "ANSWER.txt").read_text(encoding="utf-8")
    assert out["symbol"] == "Installer._write_lock_file"
    assert "def _write_lock_file" in body and "def update" not in body
