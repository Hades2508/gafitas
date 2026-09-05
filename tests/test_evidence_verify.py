"""F-114: evidence that verifies itself, because nothing was verifying it.

A tool argument too long to sit in an event is replaced by
``{truncated, chars, sha256}`` and the full text goes to a ``payloads/``
sidecar. The reading side of that -- ``read_payload``, ``sha256_text``,
``sha256_file`` -- existed since the sidecar did and had NO CALLER: not in the
package, not in the tests, not in any benchmark script.

So the seals were verifiable and had never been verified. The forensic audit
wrote the first caller by hand and found 251 of 251 references recoverable with
zero hash mismatches, which is a good result and was luck rather than a
guarantee. `_seal` now verifies what it has just written, every time.

These tests break the seal deliberately, because a verifier that has only ever
seen intact evidence has not been tested.
"""

from __future__ import annotations

import json

from localprog import evidence


def seal_one(directory, *, payloads, events):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "t.json").write_text(
        json.dumps({"record": {}, "events": events}), encoding="utf-8")
    if payloads:
        store = directory / "payloads"
        store.mkdir(exist_ok=True)
        for digest, text in payloads.items():
            (store / f"{digest}.txt").write_text(text, encoding="utf-8")
    return directory / "t.json"


def reference(text):
    digest = evidence.sha256_text(text).split(":", 1)[1]
    return digest, {"turn": 1, "tool": "write_file",
                    "args": {"content": {"truncated": text[:20],
                                         "chars": len(text), "sha256": digest}}}


def test_an_intact_seal_verifies(tmp_path):
    digest, event = reference("def f():\n    return 1\n")
    ticket = seal_one(tmp_path, payloads={digest: "def f():\n    return 1\n"},
                      events=[event])
    got = evidence.verify_ticket(ticket, tmp_path)
    assert got["intact"] is True
    assert got["references"] == 1 and got["recoverable"] == 1


def test_a_missing_payload_is_caught_and_named(tmp_path):
    _digest, event = reference("def f():\n    return 1\n")
    ticket = seal_one(tmp_path, payloads={}, events=[event])
    got = evidence.verify_ticket(ticket, tmp_path)
    assert got["intact"] is False
    assert got["missing"] and "t.json:1:content" in got["missing"][0]


def test_a_payload_whose_content_changed_is_caught(tmp_path):
    """The case a filename check cannot see: the sidecar is present and holds
    something other than what the event says it holds."""
    digest, event = reference("original\n")
    ticket = seal_one(tmp_path, payloads={digest: "TAMPERED\n"}, events=[event])
    got = evidence.verify_ticket(ticket, tmp_path)
    assert got["intact"] is False
    assert got["mismatched"] and not got["missing"]


def test_a_ticket_with_no_long_payloads_is_intact_not_suspicious(tmp_path):
    ticket = seal_one(tmp_path, payloads={},
                      events=[{"turn": 1, "tool": "read_file",
                               "args": {"path": "a.py"}}])
    got = evidence.verify_ticket(ticket, tmp_path)
    assert got["intact"] is True and got["references"] == 0


def test_an_unreadable_ticket_does_not_crash_the_verifier(tmp_path):
    """A verifier that dies on the first damaged file cannot tell you how much
    of a cohort is damaged, which is the only interesting question."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    broken = tmp_path / "t.json"
    broken.write_text("{not json", encoding="utf-8")
    assert evidence.verify_ticket(broken, tmp_path)["intact"] is True


def test_a_cohort_is_verified_as_a_whole(tmp_path):
    good, event_good = reference("good\n")
    _bad, event_bad = reference("gone\n")
    (tmp_path / "payloads").mkdir(parents=True)
    (tmp_path / "payloads" / f"{good}.txt").write_text("good\n", encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps({"events": [event_good]}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"events": [event_bad]}), encoding="utf-8")

    got = evidence.verify(tmp_path)
    assert got["tickets"] == 2
    assert got["references"] == 2 and got["recoverable"] == 1
    assert len(got["missing"]) == 1
    assert got["intact"] is False


def test_the_real_cohorts_on_disk_are_intact():
    """Not a unit test: the reason this module exists. Skipped when the
    cohorts are not present, so the suite stays runnable anywhere."""
    from pathlib import Path
    import pytest

    root = Path(r"D:\GATE-A-WS\GAFITAS_EXTERNAL_BENCHMARKS\REPOQA\RAW_RESULTS")
    directories = sorted(root.glob("matched_*_c3/evidence/tickets"))
    if not directories:
        pytest.skip("cohort evidence not present on this machine")
    for directory in directories:
        got = evidence.verify(directory)
        assert got["intact"], f"{directory}: {got['missing']} {got['mismatched']}"
