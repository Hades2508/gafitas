"""Model transport. One job: a request in, a message dict out, or ProviderError.

Every failure mode of the server is converted into ``ProviderError`` with a
closed ``kind``. Nothing here raises a bare ``URLError`` or lets a 500 look
like a model that refused to answer -- the previous runner caught
``Exception`` around the call, counted it as an invalid model call, and broke
out of the loop, so a server hiccup and a model failure produced the same
score.
"""

from __future__ import annotations

import json
import random
import os
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ProviderError

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_NUM_CTX = 8192          # contract §C.2: the only 100%-GPU rung measured

#: Working context (F-06). 8192 is the largest window that keeps a 7B-class
#: model wholly on 12 GB of VRAM, and it was the right call for a five-call
#: screen. For real work the transcript alone -- three file reads and two test
#: runs -- passes it before the agent has decided anything, and the elision that
#: follows deletes exactly the history a debugging loop needs. 16384 still fits
#: the 4B-class models this is aimed at; models that do not fit degrade in
#: speed, which is visible in usage.provider_seconds rather than silent.
#:
#: Raised 16384 -> 32768 after the dogfood run spent its entire life pegged
#: at the smaller ceiling (F-28). 16384 was inherited from the screen, whose
#: constraint was keeping a 7B model wholly on 12 GB. For the 4B doing this
#: work the KV cache at 32k is about 4.7 GB against 2.5 GB of weights, so
#: 7.2 GB total -- comfortably inside the same 12 GB.
WORK_NUM_CTX = 32768
#: Raised 2048 -> 4096 (F-52). Ollama returned
#:   500 invalid tool call arguments for "edit": unexpected end of JSON input
#: which is the model being CUT OFF mid-argument, not malformed output: a
#: write_file or an edit carrying a hundred lines of new code does not fit in
#: 2048 tokens, and the truncated JSON is then rejected by the server. The
#: symptom looks like a broken model and is an output budget one token too
#: small for the tool we asked it to call.
WORK_NUM_PREDICT = 4096
DEFAULT_NUM_PREDICT = 1024
DEFAULT_TIMEOUT = 120.0

#: The slowest generation rate a request is allowed to assume, in tokens per
#: second. A timeout has to cover the budget it granted, and these two numbers
#: were set in different places for different reasons: the clean cohort ran a
#: 2B against an 8192-token budget behind a fixed 300 s ceiling, and 13 of its
#: 50 runs died on TIMEOUT with nothing to show. Those runs are not evidence
#: about the engine; they are evidence that we asked for more than we were
#: willing to wait for.
#:
#: 12 tok/s is well under anything measured on this machine -- the cohort's
#: engines ran at 121 to 126 tok/s -- so the floor only binds when something is
#: genuinely pathological, and a caller's explicit timeout still wins when it is
#: already generous enough.
MIN_TOKENS_PER_SECOND = 12.0
#: Prompt evaluation and transport, before a single token is generated.
TIMEOUT_BASE_SECONDS = 30.0
#: An upper bound, so a hung server is still a bounded failure.
TIMEOUT_CEILING_SECONDS = 1800.0


def timeout_for(num_predict: int, requested: float) -> float:
    """A timeout that can actually accommodate the budget being granted.

    Returns the LARGER of what the caller asked for and what the budget needs,
    bounded above. Never smaller than the request: a caller that wants to wait
    longer is expressing a policy, and this is a floor, not an override.
    """
    needed = TIMEOUT_BASE_SECONDS + max(0, num_predict) / MIN_TOKENS_PER_SECOND
    return min(TIMEOUT_CEILING_SECONDS, max(float(requested or 0.0), needed))

#: How long Ollama keeps the weights resident after a request. Ollama's own
#: default is five minutes, which on a machine that is also somebody's desktop
#: means a 9B model sits on 6.6 GB of VRAM and keeps the fans up long after the
#: run has finished. Keeping it loaded BETWEEN turns is worth it -- reloading
#: every turn would dominate the wall time -- so this is generous enough to
#: span a turn and short enough to let go afterwards.
DEFAULT_KEEP_ALIVE = "90s"


@dataclass
class Call:
    """One provider round trip, for the transcript."""

    ok: bool
    message: dict = field(default_factory=dict)
    error: dict | None = None
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class OllamaProvider:
    """Ollama ``/api/chat``. Native tools when ``tools`` is passed."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        timeout: float = DEFAULT_TIMEOUT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        # F-165. Nothing was ever sent, so Ollama drew its own seed and told
        # nobody, and NO RUN IN THIS PROJECT COULD BE REPRODUCED. The evidence
        # sealed every call, every argument and every result, and omitted the
        # one number needed to get the same run back.
        #
        # Measured cost of that: seven SWE-bench instances that had produced an
        # empty patch were re-run with nothing whatsoever changed, and five of
        # them produced a patch. A cohort comparison across that much
        # run-to-run movement is measuring the dice unless the dice are
        # recorded.
        #
        # DRAWN, NOT PINNED. A constant would make every run of a cohort sample
        # the same way and hide exactly the variance above; drawing one per
        # provider leaves the distribution of outcomes untouched -- Ollama was
        # already drawing one -- and only makes each individual draw
        # repeatable. Pinning to 0, or to anything, is a treatment and would
        # need an experiment of its own.
        self.seed = random.randrange(2 ** 31) if seed is None else int(seed)
        # F-167. The first version sent THE SAME seed on every request of a
        # run, and the comment above claimed that left the distribution
        # untouched. It did not, and the claim was unsupported: before, no seed
        # was sent at all, so Ollama drew a FRESH one per request. Sending one
        # fixed value makes every turn sample under identical conditions, which
        # is a different process, not a recorded version of the old one.
        #
        # A generator seeded once from the recorded value gives back what was
        # there before -- an independent draw per request -- while staying
        # exactly reproducible from the single number the evidence carries.
        # Caught by an independent review; the fix is what makes the sentence
        # in the docstring true rather than merely confident.
        self._draws = random.Random(self.seed)
        # The budget and the patience are now decided together (F-98). They used
        # to be set in different places, and the gap between them scored 13 runs
        # as zero for an engine that was simply still generating.
        self.requested_timeout = timeout
        self.timeout = timeout_for(num_predict, timeout)
        self.keep_alive = keep_alive

    def describe(self) -> dict:
        return {
            "provider": "ollama",
            "model_class": "LOCAL",
            "model": self.model,
            "endpoint": self.endpoint,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
            "keep_alive": self.keep_alive,
            # Sealed with the run. Without it the evidence describes what
            # happened and cannot get it back.
            "seed": self.seed,
        }

    def release(self) -> None:
        """Ask Ollama to drop the weights now.

        Called when a batch finishes. Without it a 9B keeps 6.6 GB of VRAM and
        the fans up for Ollama's default five minutes after the last request,
        on a machine that is also somebody's desktop. Best effort: failing to
        free memory must never fail a run that has already succeeded.
        """
        try:
            request = urllib.request.Request(
                self.endpoint.replace("/api/chat", "/api/generate"),
                data=json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=15):
                pass
        except Exception:  # noqa: BLE001 - freeing memory is never worth an error
            pass

    def preflight(self) -> str | None:
        """Return why this provider is unusable, or None if it looks fine.

        F-41. An eight-ticket sweep once produced eight PROVIDER_ERRORs and
        reported "0/8 PASS" because the Ollama server had stopped. The routing
        behaved correctly -- a provider error is terminal, so nothing escalated
        to a paid tier -- but the operator got a report that reads like eight
        failures and an afternoon of GPU time that never happened. One request
        before the batch turns that into one clear line.
        """
        try:
            request = urllib.request.Request(self.endpoint.replace("/api/chat", "/api/tags"))
            with urllib.request.urlopen(request, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return (f"no hay servidor en {self.endpoint}: {exc}. "
                    f"Arranca ollama (`ollama serve`) y vuelve a intentarlo.")
        names = {m.get("name") for m in body.get("models", []) if isinstance(m, dict)}
        if names and self.model not in names:
            return (f"el modelo {self.model!r} no esta instalado. "
                    f"Disponibles: {', '.join(sorted(n for n in names if n))[:300]}")
        return None

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                        # Per request, drawn from the run's recorded seed.
                        "seed": self._draws.randrange(2 ** 31)},
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001 - the status is the useful part
                pass
            # A context overflow is OUR defect, not the server's and not the
            # model's, so it gets a name of its own instead of hiding among
            # ordinary HTTP failures. It stays a ProviderError -- and therefore
            # stays unscoreable -- because scoring the model for a prompt this
            # harness built too large would be exactly backwards.
            kind = "CONTEXT_OVERFLOW" if "exceed_context_size" in detail else "HTTP_STATUS"
            raise ProviderError(kind, f"{exc.code} {exc.reason}: {detail}", status=exc.code) from None
        except socket.timeout:
            raise ProviderError("TIMEOUT", f"sin respuesta en {self.timeout:g}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise ProviderError("TIMEOUT", f"sin respuesta en {self.timeout:g}s") from None
            raise ProviderError("TRANSPORT", f"{exc.reason}") from None
        except OSError as exc:
            raise ProviderError("TRANSPORT", f"{type(exc).__name__}: {exc}") from None

        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise ProviderError("BAD_BODY", f"la respuesta no es JSON: {exc}; primeros bytes: {body[:200]!r}") from None
        if not isinstance(parsed, dict):
            raise ProviderError("BAD_BODY", f"la respuesta no es un objeto: {type(parsed).__name__}")
        if "error" in parsed and "message" not in parsed:
            raise ProviderError("BAD_BODY", f"el servidor devolvió error: {parsed['error']!r}")
        message = parsed.get("message")
        if not isinstance(message, dict):
            raise ProviderError("BAD_BODY", f"sin campo 'message' utilizable: claves={sorted(parsed)}")
        return parsed


class CodexProvider:
    """Luna (gpt-5.6) through the Codex CLI. Protocol J only.

    The escalation tier (F-13). Codex is an agent, not a chat endpoint, so it
    cannot be handed a tool schema and cannot hold a conversation across calls.
    Each turn therefore renders the whole transcript into one prompt and asks
    for one JSON tool call back -- which is what protocol J exists for.

    Isolation, and why it matters here more than for a local model: Codex is
    invoked with its cwd pointed at an empty scratch directory, ``--sandbox
    read-only`` and ``--skip-git-repo-check``. It never sees the task worktree.
    Everything it knows arrives in the prompt this harness built, and the only
    way it can affect anything is by naming a tool that this harness then runs
    under the same containment as every other tier. An escalation tier that
    could reach around the sandbox would make every safety property above it
    conditional on which model happened to be answering.
    """

    BIN = os.environ.get("LOCALPROG_CODEX_BIN", r"D:\S5\cli\npm\codex.cmd")

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        *,
        effort: str = "low",
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self._scratch = Path(tempfile.mkdtemp(prefix="localprog_codex_"))

    def describe(self) -> dict:
        return {
            "provider": "codex",
            "model_class": "LUNA",
            "model": f"{self.model}:{self.effort}",
            "effort": self.effort,
            "timeout": self.timeout,
        }

    #: Codex is an AGENT, not a completion endpoint: it has its own sandbox and
    #: its own file tools, and it reaches for them first. On the first real run
    #: it looked at its own empty scratch directory, concluded it could not see
    #: the repository, and called finish(BLOCKED) on turn 2 -- while the
    #: harness had already handed it a correct listing of the real tree.
    #:
    #: It was not wrong about what it could see. It was wrong about which
    #: hands were its own. So the provider states its own execution model,
    #: because that is a fact about how this tier is invoked rather than
    #: anything to do with the task.
    PREAMBLE = """IMPORTANTE - COMO FUNCIONA ESTE ENTORNO

No estas trabajando directamente sobre el repositorio. Tu directorio actual
esta VACIO a proposito, y tus propias herramientas de fichero no sirven aqui:
no busques, no leas y no ejecutes nada por tu cuenta, porque no veras el
repositorio real y concluiras en falso que esta vacio.

La UNICA forma de actuar es responder con un bloque ```json describiendo una
llamada. Otro proceso la ejecuta sobre el repositorio real, con permisos, y te
devuelve el resultado en el siguiente turno, marcado como RESULTADO DE <tool>.

Los RESULTADO DE ... que ves mas abajo son reales: vienen del repositorio de
verdad. Fiate de ellos y no de lo que veas en tu propio sandbox.
"""

    @staticmethod
    def _render(messages: list[dict]) -> str:
        """Flatten the transcript into one prompt.

        Roles are labelled rather than dropped: without them a tool result and
        the agent's own reasoning become the same kind of text, and the model
        starts answering questions it already answered.
        """
        parts: list[str] = [CodexProvider.PREAMBLE]
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                parts.append(str(content))
            elif role == "user":
                parts.append(f"TAREA:\n{content}")
            elif role == "assistant":
                calls = message.get("tool_calls")
                if calls:
                    parts.append(f"TU TURNO ANTERIOR (llamada): {json.dumps(calls, ensure_ascii=False)[:2000]}")
                elif content:
                    parts.append(f"TU TURNO ANTERIOR: {content}")
            elif role == "tool":
                parts.append(f"RESULTADO DE {message.get('name')}:\n{content}")
        parts.append("Tu turno. Responde SOLO con el bloque ```json de la siguiente llamada.")
        return "\n\n".join(parts)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        prompt = self._render(messages)
        argv = [
            self.BIN, "exec",
            "-m", self.model,
            "-c", f'model_reasoning_effort="{self.effort}"',
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "-C", str(self._scratch),
            "-",
        ]
        try:
            proc = subprocess.run(
                argv, input=prompt, capture_output=True, text=True,
                # F-35: text=True encodes stdin with the LOCALE codec, which on
                # this machine is cp1252. A repository containing a single
                # accented character then reaches Codex as invalid UTF-8 and it
                # refuses the whole prompt -- which is what killed the second
                # Luna run at turn 6, after five turns of correct exploration,
                # on a byte inside a Spanish docstring it had just read.
                encoding="utf-8", errors="replace",
                timeout=self.timeout, shell=False,
            )
        except subprocess.TimeoutExpired:
            raise ProviderError("TIMEOUT", f"codex sin respuesta en {self.timeout:g}s") from None
        except OSError as exc:
            raise ProviderError("TRANSPORT", f"no se pudo ejecutar codex: {exc}") from None
        if proc.returncode != 0:
            raise ProviderError(
                "HTTP_STATUS",
                f"codex salio con {proc.returncode}: {(proc.stderr or '')[-400:]}",
                status=proc.returncode,
            )
        text = proc.stdout or ""
        if not text.strip():
            raise ProviderError("BAD_BODY", "codex no devolvio nada")
        # No usage numbers: the CLI does not report them. Zeros here would read
        # as "this tier was free", which is the opposite of true, so the fields
        # are left absent and the cost is carried by wall time and call count.
        return {"message": {"role": "assistant", "content": text}}


class FakeProvider:
    """Scripted provider for the deterministic suite.

    Each script entry is either a dict (returned as the ``message``) or an
    exception instance (raised). This is what lets the whole tool and error
    surface be covered without a GPU, which is the point of the exercise: real
    models stop being the fuzzer for this harness.
    """

    def __init__(self, script: list[Any], *, model: str = "fake-model") -> None:
        self.script = list(script)
        self.model = model
        self.calls: list[dict] = []
        self.index = 0

    def describe(self) -> dict:
        return {"provider": "fake", "model_class": "FAKE", "model": self.model,
                "scripted_steps": len(self.script)}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": bool(tools)})
        if self.index >= len(self.script):
            # Running off the end means the loop asked for more turns than the
            # test scripted. Surfacing it as a provider error keeps the test
            # honest instead of hanging or repeating the last answer.
            raise ProviderError("BAD_BODY", "FakeProvider script exhausted")
        step = self.script[self.index]
        self.index += 1
        if isinstance(step, BaseException):
            raise step
        return {"message": step}


def tool_message(name: str, payload: str) -> dict:
    return {"role": "tool", "name": name, "content": payload}
