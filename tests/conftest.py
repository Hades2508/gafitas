from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localprog import screen, tools  # noqa: E402

CALC = screen.CALC_PY
TEST_CALC = screen.TEST_CALC_PY


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    (tmp_path / "test_calc.py").write_text(TEST_CALC, encoding="utf-8", newline="\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    return tmp_path


@pytest.fixture
def ctx(repo: Path) -> tools.ToolContext:
    return tools.ToolContext(
        root=repo,
        write_scope=("calc.py",),
        allowed_new_files=("nuevo.py",),
        acceptance_tests=("test_calc.py",),
    )


def call(ctx, name, **kwargs):
    """Dispatch helper: returns the ToolOutcome."""
    return tools.dispatch(ctx, name, kwargs)
