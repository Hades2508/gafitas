"""F-102: a tool result must arrive in a channel the engine can read.

Every turn, the harness hands back a result as ``{"role": "tool", ...}`` and
that message is the LAST one before the model speaks. Measured at the wire,
three samples per arm, same conversation:

    gemma3:4b   role="tool"   0 of 3 usable   eval = 1, 1, 1   content empty
    gemma3:4b   role="user"   3 of 3 usable   eval = 20, 20, 20

One token and stop, deterministically. In the clean cohort that engine showed
196 dead turns, a 33% real-call rate, 100% STALLED and a 0% finish rate. It was
never able to see a single result the harness gave it.

Seven other engines read the tool role correctly, which is the whole reason this
is a measured capability and not a rule about a family: the property lives in
the deployment's chat template, not in the model. UNKNOWN keeps the tool role,
because moving a channel on a guess is a defect this campaign has already paid
for once.
"""

from __future__ import annotations

import pytest

from localprog.transcript import Transcript, Turn


def transcript(support):
    box = Transcript(system="S", user="U", tool_role_supported=support)
    box.turns.append(Turn(
        number=1,
        assistant={"role": "assistant", "content": "llamando"},
        tool_name="read_file",
        tool_payload="def f():\n    return 1",
    ))
    return box


def payload_message(support):
    rendered = transcript(support).messages()
    return rendered[-1]


def test_by_default_the_tool_role_is_used():
    """Unmeasured engines keep exactly the behaviour they had."""
    assert payload_message(None)["role"] == "tool"


def test_a_measured_supporting_engine_keeps_the_tool_role():
    got = payload_message(True)
    assert got["role"] == "tool" and got["name"] == "read_file"


def test_a_measured_broken_engine_gets_a_user_message():
    got = payload_message(False)
    assert got["role"] == "user"


def test_the_bytes_are_identical_whichever_channel_is_used():
    """The channel changes, the result does not. Anything else would make the
    two engines answer different questions."""
    body = "def f():\n    return 1"
    assert body in payload_message(True)["content"]
    assert body in payload_message(False)["content"]


def test_the_tool_name_survives_the_channel_change():
    """role="tool" carries the name in a field. A user message has no field to
    carry it, so it goes in the text -- losing it would change WHAT the model is
    told, not just how."""
    assert "read_file" in payload_message(False)["content"]


def test_the_last_message_is_the_result_either_way():
    """The shape that broke the engine is a conversation ENDING on the result.
    A probe that appended anything after it measured nothing, and this asserts
    the harness really does produce that shape."""
    for support in (True, False, None):
        assert transcript(support).messages()[-1] is not None
        assert "def f():" in transcript(support).messages()[-1]["content"]


@pytest.mark.parametrize("support", [True, False, None])
def test_the_system_and_objective_are_untouched(support):
    rendered = transcript(support).messages()
    assert rendered[0] == {"role": "system", "content": "S"}
    assert rendered[1] == {"role": "user", "content": "U"}


def test_an_elided_payload_still_goes_through_the_chosen_channel():
    """The elision path builds its own payload, and it must not fall back to a
    channel the engine cannot read."""
    box = Transcript(system="S", user="U", tool_role_supported=False,
                     keep_turns=0, elide_over_chars=10)
    for n in range(3):
        box.turns.append(Turn(number=n + 1,
                              assistant={"role": "assistant", "content": "x"},
                              tool_name="read_file",
                              tool_payload="y" * 500))
    for message in box.messages():
        assert message["role"] != "tool"
