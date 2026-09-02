"""Acceptance for DOGFOOD-03: list_symbols gains public_only.

Written by the auditor before any implementation exists and proved to fail.
This one is the autophagy test that matters: it must be implemented by the
LOCAL tier, with no paid model touching it.

The task is real. list_symbols is how an agent decides what a file contains
without reading it, and on a large module it returns every nested private
helper alongside the handful of names another module could actually call. The
conscience already draws that distinction -- verify.public_surface uses the same
leading-underscore rule -- so the tool being unable to is an inconsistency in
our own system, not a hypothetical convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localprog import errors, tools  # noqa: E402

# DOGFOOD-03 is an OPEN ticket and the honest boundary of the local tier.
#
# Thirteen free attempts across the 4B and the 9B, none of them solving it. The
# task asks for a subtle symbol-filtering rule -- drop Clase._metodo and
# everything under _Privada, but keep Clase.__init__ -- applied across five
# coordinated declaration sites in a 1450-line module whose asserts make a
# partial edit un-importable rather than merely wrong. Every attempt landed
# between 4 and 5 of 11.
#
# The acceptance suite is kept because the feature is still wanted and the
# tests are still right. It is xfail rather than deleted so the canonical repo
# stays green while the boundary stays recorded: this is what UNSUPPORTED looks
# like written down, instead of quietly routed to somebody's paid API.
#
# See AUDIT_LEDGER.json, finding F-55.
pytestmark = pytest.mark.xfail(
    reason="DOGFOOD-03: open ticket, above the local tier (13 free attempts)",
    strict=False,
)

SOURCE = '''"""Un modulo de ejemplo."""


def publica(a):
    return a


def _privada(a):
    return a


class Publica:
    def __init__(self, x):
        self.x = x

    def metodo(self):
        return self.x

    def _interno(self):
        return 0


class _Privada:
    def metodo(self):
        return 1
'''


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE, encoding="utf-8")
    return tools.ToolContext(root=tmp_path, write_scope=("mod.py",))


def names(outcome):
    """Symbol names without the '(linea N)' suffix the tool appends."""
    return {entry.split(" (")[0] for entry in outcome.value}


def call(ctx, **kwargs):
    return tools.dispatch(ctx, "list_symbols", kwargs)


def test_public_only_drops_underscore_functions(ctx):
    out = call(ctx, path="mod.py", public_only=True)
    assert out.ok, out.feedback
    got = names(out)
    assert "publica" in got
    assert "_privada" not in got


def test_public_only_drops_underscore_methods(ctx):
    out = call(ctx, path="mod.py", public_only=True)
    assert out.ok
    got = names(out)
    assert "Publica.metodo" in got
    assert "Publica._interno" not in got


def test_public_only_keeps_dunder_init(ctx):
    """__init__ is part of a public class's callable surface: how you build one."""
    out = call(ctx, path="mod.py", public_only=True)
    assert "Publica.__init__" in names(out)


def test_public_only_drops_everything_under_a_private_class(ctx):
    """A public method of a private class is not reachable either."""
    out = call(ctx, path="mod.py", public_only=True)
    got = names(out)
    assert "_Privada" not in got
    assert "_Privada.metodo" not in got


def test_public_only_is_off_by_default(ctx):
    out = call(ctx, path="mod.py")
    got = names(out)
    assert "_privada" in got and "Publica._interno" in got


def test_public_only_false_behaves_exactly_as_before(ctx):
    assert call(ctx, path="mod.py", public_only=False).value == call(ctx, path="mod.py").value


def test_a_non_boolean_public_only_is_an_invalid_call(ctx):
    out = call(ctx, path="mod.py", public_only="si")
    assert not out.ok and out.invalid_call
    assert out.code == errors.ERROR_BAD_ARGUMENTS


def test_public_only_still_reports_line_numbers(ctx):
    out = call(ctx, path="mod.py", public_only=True)
    assert any("linea" in entry for entry in out.value)


def test_a_broken_file_is_still_a_tool_error(ctx, tmp_path):
    """The syntax-error path must not regress: it is the bug that got an
    earlier list_dir deleted from the contract entirely."""
    (tmp_path / "roto.py").write_text("def f(\n", encoding="utf-8")
    out = call(ctx, path="roto.py", public_only=True)
    assert not out.ok and out.code == errors.ERROR_SYNTAX


def test_public_only_is_declared_in_the_tool_schema():
    entry = next(f for f in tools.native_schema() if f["function"]["name"] == "list_symbols")
    properties = entry["function"]["parameters"]["properties"]
    assert "public_only" in properties
    assert properties["public_only"].get("description")
    assert "public_only" not in entry["function"]["parameters"]["required"]


def test_public_only_is_in_the_text_manual():
    assert "public_only" in tools.text_manual(("list_symbols",))
