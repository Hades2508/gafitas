"""Two defects that only met each other under budget pressure.

F-172, from an independent correctness review of the transcript window.

THE CRASH
---------
The budget pass read the tool's name off the MESSAGE:

    self._placeholder_text(payload_message["name"])

and a message only carries one under role="tool". On the text protocol -- where
a result arrives as role="user" with the name inside the text, because the
engine was measured unable to read the tool role -- that raised

    KeyError: 'name'

`loop.py` catches only HarnessInvalid, so the run died. It bit exactly the
engines that need the window most: the ones driven by text have the smaller
context and therefore meet budget pressure soonest.

WHY IT STAYED HIDDEN
--------------------
The first pass already shortens old payloads, so almost nothing long survives
to reach the budget pass. Almost: an ERROR is deliberately skipped there. So
the crash needed an old error, over 400 characters, on the text protocol, with
the budget exceeded -- and every one of those conditions is ordinary.

THE SECOND DEFECT, WHICH WAS THE ONLY ROUTE TO THE FIRST
--------------------------------------------------------
The module's header promises errors are never elided, and the first pass
honours it. The budget pass did not check `is_error` at all, so an old error
could be replaced by a placeholder while newer successful output survived --
losing precisely the turns worth keeping, since an error is what tells the
agent why the last attempt did not work.
"""

from __future__ import annotations

import pytest

from localprog.transcript import Transcript, Turn

LONG = 3000
ARGUMENT_ELIDE_OVER_CHARS = 400


def build(tool_role_supported, *, is_error: bool, turns: int = 6):
    transcript = Transcript(system="sys", user="obj", keep_turns=2,
                            elide_over_chars=ARGUMENT_ELIDE_OVER_CHARS,
                            budget_chars=1200,
                            tool_role_supported=tool_role_supported)
    for n in range(1, turns + 1):
        transcript.add(Turn(
            number=n, assistant={"role": "assistant", "content": "x" * 200},
            tool_name="read_file",
            tool_payload=("E" if is_error else "R") * LONG,
            is_error=is_error))
    return transcript


@pytest.mark.parametrize("tool_role_supported", [True, False, None],
                         ids=["tool-role", "text-protocol", "unmeasured"])
def test_the_budget_pass_does_not_crash_on_any_protocol(tool_role_supported):
    """role="user" carries no "name", and the budget pass used to demand one."""
    transcript = build(tool_role_supported, is_error=True)
    messages = transcript.messages()
    assert messages, "it must render, not raise"


@pytest.mark.parametrize("tool_role_supported", [True, False, None],
                         ids=["tool-role", "text-protocol", "unmeasured"])
def test_errors_survive_the_budget_pass(tool_role_supported):
    """The module header promises errors are never elided. Now both passes agree.

    An error is what tells the agent why its last attempt failed. Dropping the
    old ones under budget pressure loses exactly the turns worth keeping.
    """
    transcript = build(tool_role_supported, is_error=True)
    intact = sum(1 for m in transcript.messages()
                 if "E" * 100 in str(m.get("content", "")))
    assert intact == 6, "every error turn must arrive whole"


@pytest.mark.parametrize("tool_role_supported", [True, False, None],
                         ids=["tool-role", "text-protocol", "unmeasured"])
def test_ordinary_results_are_still_elided(tool_role_supported):
    """The budget has to keep working; protecting errors is not disabling it."""
    transcript = build(tool_role_supported, is_error=False)
    intact = sum(1 for m in transcript.messages()
                 if "R" * 100 in str(m.get("content", "")))
    assert intact < 6, "a budget that elides nothing is not a budget"


def test_the_tool_name_reaches_the_placeholder_on_the_text_protocol():
    """Reading it off the turn rather than the message is the actual fix."""
    transcript = build(False, is_error=False)
    joined = " ".join(str(m.get("content", "")) for m in transcript.messages())
    assert "read_file" in joined, (
        "the placeholder has to say which tool was elided; the name lives on "
        "the turn, and only the tool-role message ever carried it")
