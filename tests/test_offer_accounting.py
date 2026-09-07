"""The ledger recorded offers the model never saw, and I widened it.

F-177. `_offer_a_primitive` wrote two concrete offers into the adoption ledger
BEFORE deciding whether any offer would be emitted. Four separate paths below
that call return "" -- already using the primitive, no resolvable target, a
path the mission does not authorise, or neither primitive applicable -- and
every one left the ledger claiming the model had been shown two calls it never
received.

F-148 is one of those four paths, and it was added the same night as this fix.
Making the harness correctly silent about forbidden paths widened an accounting
error in the instrument that supports this project's attribution claims. The
silence was right; the bookkeeping behind it was not.

Why it matters is in the ledger's own header:

    OFFERED    it was named at a point the model was reading
    ...
    Collapsing any two of these hides the actual failure.

An offer recorded but never made collapses OFFERED into APPLICABLE, which is
exactly the distinction the chain exists to keep apart. Every adoption rate
computed from those counts had a denominator inflated by offers that were never
made.
"""

from __future__ import annotations

import pytest

from localprog import adoption, tools

SOURCE = "def f():\n    return 1\n"


def build(tmp_path, scope):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "prohibido").mkdir()
    (tmp_path / "prohibido" / "x.py").write_text(SOURCE, encoding="utf-8")
    ledger = adoption.AdoptionLedger()
    ctx = tools.ToolContext(root=tmp_path, write_scope=scope,
                            adoption_ledger=ledger, concrete_offers=True)
    return ctx, ledger


def offer(ctx, path):
    ctx.opened.add(path)
    return tools._offer_a_primitive(
        ctx, "edit", {"path": path, "old": "x", "new": "y"}, "ERROR_X")


def offered_count(ledger) -> int:
    return sum(1 for o in ledger.opportunities if o.offered)


def test_an_offer_that_is_made_is_recorded(tmp_path):
    ctx, ledger = build(tmp_path, ("src/",))
    assert offer(ctx, "src/a.py"), "the offer text must reach the model"
    assert offered_count(ledger) == 2


def test_a_forbidden_path_records_no_offer(tmp_path):
    """F-148 makes the harness silent here. The ledger has to agree."""
    ctx, ledger = build(tmp_path, ("src/",))
    assert offer(ctx, "prohibido/x.py") == "", "F-148: no offer for a forbidden path"
    assert offered_count(ledger) == 0, (
        "two offers were recorded for a path the model was correctly told "
        "nothing about")


def test_a_missing_file_records_no_offer(tmp_path):
    ctx, ledger = build(tmp_path, ("src/",))
    assert offer(ctx, "src/noexiste.py") == ""
    assert offered_count(ledger) == 0


def test_the_opportunity_itself_is_still_recorded(tmp_path):
    """Recording the opportunity is unconditional; only the OFFER is the treatment.

    The control arm's whole value is that it records everything it declined to
    say, so this must not become "silence means nothing happened".
    """
    ctx, ledger = build(tmp_path, ("src/",))
    offer(ctx, "prohibido/x.py")
    assert ledger.opportunities, "the moment still happened and is still evidence"


# ------------------------------------------------------- the ordering guard

def test_an_impossible_row_is_flagged():
    """accepted without called cannot happen, and nothing used to check."""
    ledger = adoption.AdoptionLedger()
    ledger.record(adoption.Opportunity(
        turn=1, primitive="replace_file", available=False, applicable=True,
        applicable_because="x", accepted=True))
    sealed = ledger.to_dict()
    assert sealed["violations"], "an impossible row must announce itself"
    assert "accepted without called" in sealed["violations"][0]["broken"]


def test_a_well_formed_row_is_not_flagged():
    ledger = adoption.AdoptionLedger()
    ledger.record(adoption.Opportunity(
        turn=1, primitive="replace_file", available=True, applicable=True,
        applicable_because="x", offered=True, selected=True, called=True,
        accepted=True))
    assert ledger.to_dict()["violations"] == []


def test_applicable_without_offered_is_legitimate():
    """The control arm is exactly this, and must never be called a violation."""
    ledger = adoption.AdoptionLedger()
    ledger.record(adoption.Opportunity(
        turn=1, primitive="replace_file", available=True, applicable=True,
        applicable_because="x", offered=False))
    assert ledger.to_dict()["violations"] == []


def test_a_flagged_row_is_kept_not_dropped():
    """A measurement instrument must not destroy the evidence of its own defect."""
    ledger = adoption.AdoptionLedger()
    ledger.record(adoption.Opportunity(
        turn=1, primitive="replace_file", available=False, applicable=True,
        applicable_because="x", accepted=True))
    assert len(ledger.to_dict()["events"]) == 1


def test_violations_are_empty_on_a_clean_ledger():
    assert adoption.AdoptionLedger().to_dict()["violations"] == []
