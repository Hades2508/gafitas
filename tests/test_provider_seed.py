"""Every run records the seed it used, so every run can be got back.

F-165. Nothing was ever sent in ``options``, so Ollama drew its own seed and
told nobody. The evidence sealed every call, every argument, every result and
every diff -- and omitted the one number needed to reproduce any of it.

The cost of that is measured, not hypothetical. Seven SWE-bench Verified
instances that had produced an empty patch were re-run with nothing whatsoever
changed, and five of them produced a patch. Any cohort comparison across that
much movement is measuring the dice unless the dice are written down.

DRAWN, NOT PINNED. A constant would make every run of a cohort sample the same
way and hide exactly that variance. Drawing one per provider leaves the
distribution untouched -- Ollama was already drawing one -- and only makes each
individual draw repeatable. Pinning to a fixed value is a treatment and would
need its own experiment.

That the seed actually controls the output was verified against the live
engine before this was written, not assumed:

    seed 12345, run 1   "Un pajaro azul canta solo para que el cielo se ..."
    seed 12345, run 2   "Un pajaro azul canta solo para que el cielo se ..."
    seed 99999          "Un pajaro azul canta en silencio, porque su voz ..."
"""

from __future__ import annotations

import json
from unittest.mock import patch as mock_patch

from localprog.provider import OllamaProvider


class Response:
    """Minimal stand-in for what urlopen yields."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


BODY = json.dumps({"message": {"content": "ok"}, "done": True})


def sent_payload(provider: OllamaProvider) -> dict:
    """Drive one chat and return the JSON the provider put on the wire."""
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response(BODY)

    with mock_patch("urllib.request.urlopen", side_effect=fake_urlopen):
        provider.chat([{"role": "user", "content": "hola"}])
    return captured["payload"]


def test_a_seed_is_drawn_when_none_is_given():
    provider = OllamaProvider("m")
    assert isinstance(provider.seed, int)
    assert provider.seed >= 0


def test_the_seed_is_recorded_in_describe():
    provider = OllamaProvider("m")
    described = provider.describe()
    assert described["seed"] == provider.seed, (
        "describe() is what reaches the sealed evidence; a seed that is drawn "
        "and not recorded reproduces nothing")


def test_the_seed_is_actually_sent():
    provider = OllamaProvider("m", seed=4242)
    payload = sent_payload(provider)
    assert payload["options"]["seed"] == 4242, (
        "recording a seed the engine never received would be worse than not "
        "recording one: the evidence would claim a reproducibility it has not")


def test_an_explicit_seed_is_honoured():
    provider = OllamaProvider("m", seed=7)
    assert provider.seed == 7
    assert provider.describe()["seed"] == 7


def test_the_seed_is_not_a_constant():
    """A fixed default would hide the run-to-run variance it exists to expose."""
    seeds = {OllamaProvider("m").seed for _ in range(20)}
    assert len(seeds) > 1, (
        "every provider drew the same seed; that is not reproducibility, it is "
        "a cohort silently sampling one point of the distribution")


def test_the_other_options_still_go_through():
    provider = OllamaProvider("m", num_ctx=4096, num_predict=256, seed=1)
    options = sent_payload(provider)["options"]
    assert options["num_ctx"] == 4096
    assert options["num_predict"] == 256
    assert options["seed"] == 1
