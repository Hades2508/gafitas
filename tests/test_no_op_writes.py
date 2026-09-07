"""A write that changes nothing is not a success, and must not be recorded as one.

F-166, found in sealed evidence. `psf__requests-2931`, turn 9:

    edit(path="requests/utils.py", old=<203 chars>, new=<the same 203 chars>)
    -> ok: True

`old` and `new` were byte-identical. The file was untouched, the turn was
spent, the agent was told it had worked, and `changed_files` gained a path the
tree did not have -- so the record disagreed with the repository, which is the
one thing the record exists not to do.

The gap was inconsistency rather than oversight. Audited across the six write
tools, `replace_file` and `replace_symbol_body` already refused a no-op with
ERROR_NOTHING_CHANGED; `edit` and `replace_lines` did not. There is no reason
for the write surface to disagree with itself about what "nothing changed"
means, so they now all say the same thing.

`write_file` is not in this list on purpose: it creates a file, and writing to
one that exists is already refused as ERROR_FILE_EXISTS.
"""

from __future__ import annotations

import pytest

from localprog import tools
from localprog.errors import ERROR_NOTHING_CHANGED, ToolError

BODY = "def suma(a, b):\n    return a + b\n"


def build(tmp_path):
    (tmp_path / "m.py").write_text(BODY, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("m.py",),
                             allowed_new_files=())


def test_edit_refuses_old_identical_to_new(tmp_path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.edit(ctx, "m.py", "return a + b", "return a + b")
    assert caught.value.code == ERROR_NOTHING_CHANGED


def test_replace_lines_refuses_a_range_already_that_content(tmp_path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.replace_lines(ctx, "m.py", 2, 2, "    return a + b")
    assert caught.value.code == ERROR_NOTHING_CHANGED


def test_replace_symbol_body_still_refuses(tmp_path):
    """It always did. Kept so the three cannot drift apart again."""
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.replace_symbol_body(ctx, "m.py", "suma", "    return a + b")
    assert caught.value.code == ERROR_NOTHING_CHANGED


def test_a_refused_no_op_leaves_the_file_alone(tmp_path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError):
        tools.edit(ctx, "m.py", "return a + b", "return a + b")
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == BODY


def test_a_refused_no_op_does_not_claim_the_file_changed(tmp_path):
    """The half that corrupted the record, not just the half that wasted a turn."""
    ctx = build(tmp_path)
    for call in (
        lambda: tools.edit(ctx, "m.py", "return a + b", "return a + b"),
        lambda: tools.replace_lines(ctx, "m.py", 2, 2, "    return a + b"),
    ):
        with pytest.raises(ToolError):
            call()
    assert ctx.changed_files == set(), (
        "changed_files fed the run record; a path in there that the tree does "
        "not have makes the evidence disagree with the repository")


def test_a_real_edit_still_applies(tmp_path):
    ctx = build(tmp_path)
    tools.edit(ctx, "m.py", "a + b", "a - b")
    assert "a - b" in (tmp_path / "m.py").read_text(encoding="utf-8")
    assert ctx.changed_files == {"m.py"}


def test_a_real_line_replacement_still_applies(tmp_path):
    ctx = build(tmp_path)
    tools.replace_lines(ctx, "m.py", 2, 2, "    return a * b")
    assert "a * b" in (tmp_path / "m.py").read_text(encoding="utf-8")
    assert ctx.changed_files == {"m.py"}


def test_the_refusal_says_what_to_do(tmp_path):
    ctx = build(tmp_path)
    with pytest.raises(ToolError) as caught:
        tools.edit(ctx, "m.py", "return a + b", "return a + b")
    message = str(caught.value)
    assert "identicos" in message
    assert "new" in message and "old" in message, (
        "an error that does not say which argument to change costs another turn")
