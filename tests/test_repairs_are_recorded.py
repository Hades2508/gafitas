"""A repaired call and a clean one are not the same event, and the seal said they were.

F-175. Three places in this codebase already knew this:

    repair.py's header      "**Announce.** A recovered call carries ``repaired``
                             and the tier that fired."
    ParsedCall's comment    "...and a repaired one as the same event, and they
                             are not the same event."
    protocol.py:364         sets it.

And `loop.py` read the flag nowhere. So a call the harness had to reconstruct
from truncated or malformed output was sealed identically to one the model
emitted correctly, and nothing downstream could tell them apart.

That matters most for the thing this project spends its GPU on. An engine
whose calls need repairing half the time and one whose calls never do are not
the same engine, and every cohort comparison made here read them as identical
because nothing counted it. The flag existed, the announcement was documented,
and the announcement had no audience.

Sealing it end to end took three edits, and the middle one is the lesson F-165
already taught: the field on the result is not enough, `to_dict` names its keys
and a new one has to be added there too or the evidence never sees it.
"""

from __future__ import annotations

from localprog import loop, protocol, work


def test_the_parser_reports_which_tier_repaired_a_call():
    """The flag exists and is set. Pinned so the rest of this has a subject."""
    clean = protocol.parse_json_call('{"tool":"read_file","arguments":{"path":"a.py"}}')
    assert clean.repaired is None

    truncated = protocol.parse_json_call(
        '{"tool":"write_file","arguments":{"path":"a.py","content":"return 1')
    assert truncated.name == "write_file"
    assert truncated.repaired, (
        "a call reconstructed from a cut-off generation has to say so")


def test_the_loop_result_carries_a_repair_count():
    assert loop.LoopResult(outcome="x").repairs == {}


def test_the_work_result_carries_it_too():
    assert work.WorkResult(ticket_id="t", model="m").repairs == {}


def test_the_count_reaches_the_sealed_record():
    """F-165's lesson: a field on the result is not a field in the evidence.

    to_dict names its keys. The seed was added to WorkResult, the assignment
    was made, and the sealed file still said nothing -- because to_dict had not
    been told. This is the same shape and would have failed the same way.
    """
    result = work.WorkResult(ticket_id="t", model="m")
    result.repairs = {"json_tail": 3}
    sealed = result.to_dict()
    assert "repairs" in sealed
    assert sealed["repairs"] == {"json_tail": 3}


def test_a_clean_run_seals_an_empty_count_rather_than_nothing():
    """Zero repairs is a measurement. Absence of the key is not."""
    sealed = work.WorkResult(ticket_id="t", model="m").to_dict()
    assert sealed["repairs"] == {}
