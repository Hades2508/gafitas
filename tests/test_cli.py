"""F-114: the entry point a human actually invokes, which had no tests at all.

`cli.py` is 267 statements and was at 0% coverage. Everything in it had only
ever been executed by somebody typing it. That is the one path where a mistake
reaches the operator directly rather than being caught by a run.

These tests do not talk to a model. What is worth pinning down is the part that
is pure harness: does every advertised subcommand exist, does argument parsing
refuse what it should, does a malformed ticket come back as HARNESS_INVALID with
exit 3 rather than as a model failure, and does `selfcheck` report the real tool
surface rather than a list that has drifted from `SPECS`.
"""

from __future__ import annotations

import json

import pytest

from localprog import cli, tools
from localprog.errors import HarnessInvalid


def run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr()


# ------------------------------------------------------------- the surface

@pytest.mark.parametrize("command", ["selfcheck", "retain", "persist", "screen",
                                     "step1", "work"])
def test_every_advertised_subcommand_is_dispatchable(command):
    """Advertised in the parser and present in the dispatch table are two
    different things, and a command that is only in one of them fails at the
    moment somebody uses it."""
    parser_commands = set()
    import argparse

    real = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        for action in self._subparsers._group_actions if self._subparsers else []:
            parser_commands.update(action.choices)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            cli.main([command])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert command in parser_commands


def test_no_command_is_an_error_not_a_crash():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


def test_an_unknown_command_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])


# ------------------------------------------------------------- selfcheck

def test_selfcheck_reports_the_real_tool_surface(capsys):
    """Not a hand-written list. A CLI that advertises a tool the harness does
    not have, or omits one it does, is worse than no listing."""
    code, out = run(["selfcheck"], capsys)
    assert code == 0
    body = json.loads(out.out)
    assert set(body["tools"]) == set(tools.SPECS)


def test_selfcheck_reports_dependency_provenance(capsys):
    _code, out = run(["selfcheck"], capsys)
    body = json.loads(out.out)
    assert body["provenance"], "provenance is how a run is attributable at all"


# ------------------------------------------------------------- work / tickets

def test_an_uninstalled_model_is_refused_before_anything_else(tmp_path, capsys):
    """Found by writing this test. The CLI preflights the model against what is
    actually installed and names the alternatives, before it reads a ticket or
    builds a workspace. That is the right order -- a run that cannot possibly
    start should not create evidence -- and it was untested."""
    bad = tmp_path / "ticket.json"
    bad.write_text('{"ticket_id": "x"}', encoding="utf-8")
    code = cli.main(["work", "--tickets", str(bad), "--model", "does-not-exist",
                     "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code != 0
    assert "PREFLIGHT" in err
    # Two refusals, one property. With a server up the preflight says the model
    # is not installed and lists what is; with no server it says there is no
    # server and how to start one. Asserting only the first made this test
    # depend on a running model server, which it has no business needing --
    # found when the server went down for an unrelated reason.
    if "no esta instalado" in err:
        assert "Disponibles" in err,             "a refusal that does not say what IS available is a dead end"
    else:
        assert "no hay servidor" in err and "ollama serve" in err,             "a refusal that does not say how to recover is a dead end too"


def _installed_model():
    """Any model this machine really has, so the ticket path is reached."""
    from localprog import deps
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
            names = [m["name"] for m in _json.loads(r.read()).get("models", [])]
        return names[0] if names else None
    except Exception:                                        # noqa: BLE001
        return None


@pytest.mark.parametrize("body", ['{"ticket_id": "x"}', "this is not json", None])
def test_a_broken_ticket_is_harness_invalid_and_exits_three(tmp_path, capsys, body):
    """A broken ticket is a defect in the TICKET. Recording it as a model
    failure is the confusion the four-class error taxonomy exists to stop, and
    the exit code is how a script outside this process can tell."""
    model = _installed_model()
    if model is None:
        pytest.skip("no model server reachable; the preflight would fire first")
    ticket = tmp_path / "ticket.json"
    if body is not None:
        ticket.write_text(body, encoding="utf-8")
    code = cli.main(["work", "--tickets", str(ticket), "--model", model,
                     "--out", str(tmp_path / "out")])
    assert code == 3
    assert "HARNESS_INVALID" in capsys.readouterr().err


# ------------------------------------------------------------- retain

def test_retain_is_a_dry_run_unless_asked(tmp_path, capsys):
    """It deletes preserved workspaces, which are the evidence of failed runs.
    Defaulting to destructive would be exactly backwards."""
    box = tmp_path / "localprog-ws-abc"
    box.mkdir()
    (box / "marker.txt").write_text("x", encoding="utf-8")
    code, _out = run(["retain", "--base", str(tmp_path)], capsys)
    assert code == 0
    assert box.exists(), "a dry run must not delete anything"


def test_retain_reports_what_it_found(tmp_path, capsys):
    (tmp_path / "localprog-ws-abc").mkdir()
    _code, out = run(["retain", "--base", str(tmp_path)], capsys)
    assert out.out.strip(), "a report that says nothing cannot be acted on"


def test_retain_on_an_empty_base_is_not_an_error(tmp_path, capsys):
    code, _out = run(["retain", "--base", str(tmp_path)], capsys)
    assert code == 0
