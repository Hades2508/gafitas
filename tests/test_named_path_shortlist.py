"""F-118: the objective named the file, and the shortlist ranked past it.

`opening_candidates` scores every region in the repository against the
objective. It never noticed that the objective contained a path which resolves
to a real file in that repository, so regions the objective had already ruled
out competed for all five slots with the ones it pointed at.

Measured offline over the frozen fifty-case corpus, no engine invoked:

    ranking                          recall@1   recall@5
    whole repository (before)          18/50      32/50
    three slots reserved (now)         23/50      41/50
    every slot reserved                23/50      43/50

Five cases gained rank 1 and none lost it. Reserving all five would find two
more, and would give up the only route to the answer when the named file is a
red herring, so three is what shipped.

The property that matters most is the LAST one here: when the objective names
nothing that exists, the ranking must come out byte-identical to before. A
retrieval change that quietly alters every mission is not this change.
"""

from __future__ import annotations

import pytest

from localprog import orient

OTHER = '''
def rate_limit_bucket(tokens, window):
    """Throttle requests inside a sliding window of seconds."""
    return tokens / window


def throttle_window(tokens, window):
    """Throttle requests inside a sliding window of seconds."""
    return window
'''

WANTED = '''
def unrelated_helper(a):
    return a


def sliding_window_rate(tokens, window):
    """Throttle requests inside a sliding window of seconds."""
    return tokens // window
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    # Two files whose text answers the objective equally well. Without the
    # constraint the ranking has no reason to prefer either.
    (tmp_path / "pkg" / "other.py").write_text(OTHER, encoding="utf-8")
    (tmp_path / "pkg" / "wanted.py").write_text(WANTED, encoding="utf-8")
    return tmp_path


def objective(path: str) -> str:
    return (f"Dentro del fichero {path} de este repositorio hay UNA funcion que "
            f"throttles requests inside a sliding window of seconds. Escribe su "
            f"codigo completo y literal en ANSWER.txt sin cambiar nada.")


# ------------------------------------------------------------ path detection

def test_a_path_that_exists_is_detected(repo):
    assert orient.named_paths(objective("pkg/wanted.py"), repo) == ["pkg/wanted.py"]


def test_a_path_that_does_not_exist_is_not_a_constraint(repo):
    """Existence is the whole test. A string that merely looks like a path is
    the model's or the mission's guess, and constraining on a guess would hide
    the answer for good."""
    assert orient.named_paths(objective("pkg/imaginary.py"), repo) == []


def test_a_windows_separator_resolves_too(repo):
    assert orient.named_paths(r"mira en pkg\wanted.py ahora", repo) == ["pkg/wanted.py"]


def test_prose_that_is_not_a_path_is_ignored(repo):
    assert orient.named_paths("arregla el bug. gracias. el fichero es raro.", repo) == []


def test_detection_is_bounded(repo):
    """A mission that names forty files must not turn the shortlist into a
    directory listing."""
    text = " ".join(["pkg/wanted.py", "pkg/other.py"] * 20)
    assert len(orient.named_paths(text, repo, limit=2)) == 2


# --------------------------------------------------------------- the ranking

def test_the_named_file_reaches_the_shortlist(repo):
    block = orient.opening_candidates(repo, objective("pkg/wanted.py"))
    assert "pkg/wanted.py" in block


def test_the_rest_of_the_repository_is_still_reachable(repo):
    """Three slots, not five. The other file must still be listed, because the
    named file is sometimes the wrong file and the agent needs a way out."""
    block = orient.opening_candidates(repo, objective("pkg/wanted.py"))
    assert "pkg/other.py" in block


def test_naming_nothing_that_exists_changes_nothing(repo):
    """The no-op case, and the one that protects every mission that does not
    name a file.

    Compared against the index's own unconstrained ranking of the SAME text --
    not against a second, differently worded objective, which would score
    differently for reasons that have nothing to do with this change.
    """
    from localprog import retrieval

    text = objective("pkg/imaginary.py")
    plain = [(r.path, r.name) for r, _s in retrieval.Index(repo).search(text, limit=5)]
    block = orient.opening_candidates(repo, text)
    shown = [l for l in block.splitlines() if l.strip()[:1].isdigit()]
    assert len(shown) == len(plain)
    for line, (path, name) in zip(shown, plain):
        assert path in line and (name or "") in line


def test_the_shortlist_still_says_it_is_not_a_recommendation(repo):
    """F-85 is the standing lesson and reordering must not cost the disclaimer:
    a list that looks like an answer gets taken as one."""
    block = orient.opening_candidates(repo, objective("pkg/wanted.py"))
    assert "NO" in block and "certeza" in block


def test_the_shortlist_is_never_longer_than_its_limit(repo):
    block = orient.opening_candidates(repo, objective("pkg/wanted.py"), limit=2)
    numbered = [l for l in block.splitlines() if l.strip()[:2] in ("1.", "2.", "3.")]
    assert len(numbered) <= 2


def test_no_region_is_listed_twice(repo):
    """The reserved slots and the wide ranking overlap by construction. A
    shortlist that showed the same region at rank 1 and rank 3 would be
    spending a slot to say nothing."""
    block = orient.opening_candidates(repo, objective("pkg/wanted.py"))
    rows = [l.strip() for l in block.splitlines() if l.strip()[:1].isdigit()]
    bodies = [r.split(".", 1)[1] for r in rows]
    assert len(set(bodies)) == len(bodies)


def test_an_unreadable_repository_still_returns_a_string(repo):
    """A shortlist is a convenience. A run must never fail because building
    one did."""
    assert isinstance(orient.opening_candidates(repo / "nope", objective("x.py")), str)
