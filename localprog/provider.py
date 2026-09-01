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
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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
WORK_NUM_PREDICT = 2048
DEFAULT_NUM_PREDICT = 1024
DEFAULT_TIMEOUT = 120.0


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
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout

    def describe(self) -> dict:
        return {
            "provider": "ollama",
            "model_class": "LOCAL",
            "model": self.model,
            "endpoint": self.endpoint,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
        }

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": self.num_predict},
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
