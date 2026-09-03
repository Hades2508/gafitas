"""A tool that cannot succeed under this mission is not offered.

Measured, not designed. With the payload channel repaired, granite4.1:3b's
matched runs came out clean of protocol errors and still scored 2% against 50%
in one prompt. The trace said why: 18 of the first 23 runs called ``edit`` as
their FIRST tool, on a mission whose write scope is empty and whose only
writable thing is a file that does not exist yet. ``edit`` could not succeed for
any argument. We offered it anyway, the engine took the tool whose name matches
the verb in the objective, and quit at turn three with ERROR_NOT_IN_WRITE_SCOPE.

This is progressive tool disclosure arrived at from the only direction that
cannot overfit: a tool is withheld because it is IMPOSSIBLE here, never because
a particular engine seemed to prefer fewer.
"""

from __future__ import annotations

import json

from localprog import tools, work


def test_an_ordinary_ticket_keeps_every_tool():
    """The property that protects everything already measured: a mission with a
    real write scope sees exactly the surface it always saw."""
    assert tools.legal_tools(("src/",), ()) == tuple(tools.SPECS)


def test_a_create_only_mission_is_not_offered_edit():
    """The matched shape: nothing may be modified, one file may be created."""
    legal = tools.legal_tools((), ("answer.txt",))
    assert "edit" not in legal and "replace_lines" not in legal
    assert "write_file" in legal


def test_a_read_only_mission_is_offered_no_writing_at_all():
    legal = tools.legal_tools((), ())
    assert not ({"edit", "replace_lines", "write_file"} & set(legal))


def test_looking_is_never_withheld():
    """An agent that cannot look cannot decide, whatever the scope."""
    for scope, new in ((), ()), ((), ("a.txt",)), (("src/",), ()):
        legal = set(tools.legal_tools(scope, new))
        assert {"read_file", "grep", "search_code", "list_dir"} <= legal


def test_finish_is_never_withheld():
    """Removing the way out turns a wrong turn into a hung run."""
    for scope, new in ((), ()), ((), ("a.txt",)), (("src/",), ()):
        assert "finish" in tools.legal_tools(scope, new)


def test_the_order_is_the_canonical_one():
    legal = tools.legal_tools((), ("a.txt",))
    assert list(legal) == [n for n in tools.SPECS if n in set(legal)]


def test_every_legal_tool_is_a_real_tool():
    """native_schema raises on an unknown name, which is the guard that caught
    an earlier runner advertising a list_dir it did not implement."""
    for scope, new in ((), ()), ((), ("a.txt",)), (("src/",), ()):
        assert tools.native_schema(tools.legal_tools(scope, new))


# ------------------------------------------------- the two tiers must agree

def _ticket(scope, new):
    return work.Ticket(ticket_id="t", repo=work.Path("."), objective="o",
                       write_scope=scope, allowed_new_files=new,
                       acceptance_tests=(), full_suite=False, max_turns=5)


def test_the_text_manual_advertises_the_same_surface_as_the_schema():
    """Protocol J is where the small engines run. A manual that offered a tool
    the schema withholds would mean the two tiers disagree about what this
    mission permits."""
    ticket = _ticket((), ("answer.txt",))
    manual = work.system_prompt(ticket, "J")
    assert "edit(" not in manual and "replace_lines(" not in manual
    assert "write_file(" in manual

    schema_names = {t["function"]["name"] for t in
                    tools.native_schema(tools.legal_tools((), ("answer.txt",)))}
    for name in schema_names:
        assert f"{name}(" in manual, f"{name} in schema but not in the manual"


def test_an_ordinary_ticket_still_documents_editing():
    manual = work.system_prompt(_ticket(("src/",), ()), "J")
    assert "edit(" in manual and "replace_lines(" in manual
