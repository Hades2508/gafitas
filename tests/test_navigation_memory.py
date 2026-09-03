"""What the loop remembers about where the agent has already looked (F-60).

Every fact here was already in the loop's own state and was never said out
loud. The transcript window is finite, so an agent twelve turns into a search
genuinely cannot see the query it ran on turn two -- but the harness can, and
staying quiet about it is how a run spends its budget re-proving that a term
does not appear in the repository.
"""

from __future__ import annotations

import pytest

from localprog import loop, tools
from localprog.provider import FakeProvider


def tc(name, **arguments):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def alpha():\n"
        '    """Return the first letter."""\n'
        "    return 'a'\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def ctx(repo):
    return tools.ToolContext(root=repo, write_scope=("pkg/**/*.py",))


def drive(ctx, script, max_turns=20):
    provider = FakeProvider(script)
    result = loop.run_loop(provider=provider, ctx=ctx, system="S", objective="O",
                           protocol_name="A", max_turns=max_turns)
    return result, provider


def payloads(provider):
    """Every tool result the model was ever shown, in order."""
    out = []
    for call in provider.calls:
        for message in call["messages"]:
            if message.get("role") == "tool":
                out.append(message.get("content", ""))
    return out


# ------------------------------------------------------- the same call twice

def test_an_exact_repeat_is_named_with_the_turn_it_first_happened(ctx):
    """Not consecutive: the point is the one the agent cannot see any more."""
    script = [
        tc("grep", pattern="alpha"),
        tc("list_dir", path="pkg"),
        tc("read_file", path="pkg/a.py"),
        tc("grep", pattern="alpha"),
        tc("finish", summary="fin", status="BLOCKED"),
    ]
    _result, provider = drive(ctx, script)
    shown = payloads(provider)
    assert any("turno 1" in text and "EXACTAMENTE" in text for text in shown), shown


def test_a_first_call_is_not_accused_of_repeating(ctx):
    _result, provider = drive(ctx, [tc("grep", pattern="alpha"),
                                    tc("finish", summary="x", status="BLOCKED")])
    assert not any("EXACTAMENTE esta llamada" in text for text in payloads(provider))


# ------------------------------------------------------------- dead ends

def test_fruitless_searches_are_remembered_and_listed(ctx):
    script = [
        tc("grep", pattern="quaternion"),
        tc("grep", pattern="holography"),
        tc("grep", pattern="isospin"),
        tc("finish", summary="no encontrado", status="BLOCKED"),
    ]
    result, provider = drive(ctx, script)
    assert result.dead_ends == ["grep('quaternion')", "grep('holography')",
                                "grep('isospin')"]
    shown = payloads(provider)
    assert any("3 busquedas que no han encontrado nada" in text for text in shown), shown


def test_the_list_is_not_shown_before_it_is_a_pattern(ctx):
    """Two misses is a coincidence. Nagging at the first one is noise."""
    _result, provider = drive(ctx, [tc("grep", pattern="quaternion"),
                                    tc("finish", summary="x", status="BLOCKED")])
    assert not any("busquedas que no han encontrado" in t for t in payloads(provider))


def test_a_search_that_found_something_is_not_a_dead_end(ctx):
    result, _provider = drive(ctx, [tc("grep", pattern="alpha"),
                                    tc("finish", summary="x", status="BLOCKED")])
    assert result.dead_ends == []


def test_a_failed_search_call_counts_as_a_dead_end(ctx):
    """ERROR_NO_MATCH from search_code is a dead end exactly like an empty grep."""
    script = [
        tc("search_code", query="quaternion holography"),
        tc("grep", pattern="isospin"),
        tc("grep", pattern="chromodynamics"),
        tc("finish", summary="x", status="BLOCKED"),
    ]
    result, provider = drive(ctx, script)
    assert len(result.dead_ends) == 3
    assert any("busquedas que no han encontrado nada" in t for t in payloads(provider))


def test_dead_ends_reach_the_sealed_record(ctx):
    result, _ = drive(ctx, [tc("grep", pattern="quaternion"),
                            tc("finish", summary="x", status="BLOCKED")])
    assert result.to_dict()["dead_ends"] == ["grep('quaternion')"]


# ------------------------------------------------------------ the arguments

def test_the_evidence_records_what_was_actually_searched_for(ctx):
    """F-58: the sealed events claimed to carry arguments and did not, which is
    why the first retrieval audit could not see a single query the agent ran."""
    result, _ = drive(ctx, [tc("grep", pattern="alpha", context=2),
                            tc("finish", summary="x", status="BLOCKED")])
    first = result.events[0]
    assert first["tool"] == "grep"
    assert first["args"]["pattern"] == "alpha"
    assert first["args"]["context"] == 2


def test_a_huge_argument_is_hashed_and_kept_whole_in_a_sidecar(ctx):
    """A2. Truncation alone made write_file bodies unrecoverable, and that
    blocked two analyses in the campaign that found it: the only way to tell a
    correct copy from an over-copy is to read what was written, and the
    evidence had thrown it away to save space. A hash plus a sidecar costs the
    same space per DISTINCT payload and nothing per repeat."""
    body = "x = 1" + chr(10) + "z = 2" + chr(10)
    body = body * 300
    result, _ = drive(ctx, [tc("write_file", path="pkg/big.py", content=body),
                            tc("finish", summary="x", status="BLOCKED")])
    sealed = result.events[0]["args"]["content"]
    assert isinstance(sealed, dict)
    assert sealed["chars"] == len(body)
    assert len(sealed["truncated"]) < len(body), "the event itself stays small"
    assert result.payloads[sealed["sha256"]] == body, "the whole thing survives"


def test_a_short_argument_is_still_just_the_string(ctx):
    result, _ = drive(ctx, [tc("grep", pattern="alpha"),
                            tc("finish", summary="x", status="BLOCKED")])
    assert result.events[0]["args"]["pattern"] == "alpha"
    assert result.payloads == {}, "no sidecar for something that fits"


def test_the_same_payload_twice_is_stored_once(ctx):
    body = ("y = 2" + chr(10)) * 300
    result, _ = drive(ctx, [tc("write_file", path="pkg/one.py", content=body),
                            tc("write_file", path="pkg/two.py", content=body),
                            tc("finish", summary="x", status="BLOCKED")])
    assert len(result.payloads) == 1
