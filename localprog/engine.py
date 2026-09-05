"""The boundary between GAFITAS and whatever brain it is driving.

WHY THIS EXISTS
---------------
Everything measured in this project so far was measured with one engine:
``qwen3:4b-instruct-2507-q4_K_M``. An audit for model-specific logic in the core
found none -- no ``if model ==`` anywhere -- which is the good news. The bad news
is subtler and it is what this module fixes: the core did not know a model's
properties at all, so it assumed them.

    WORK_NUM_CTX = 32768
    WORK_NUM_PREDICT = 4096

Those are module constants. They are the reference engine's numbers wearing the
costume of universal truth. Point the harness at a 2B model with an 8k window
and it does not adapt, it overflows -- and a context overflow is recorded as a
provider error, which is to say as the engine's fault. A harness that misreports
its own misconfiguration as somebody else's failure cannot be used to compare
engines, which is exactly what we want to do next.

Protocol is the same story. Whether a model can drive native tool calls or needs
the JSON text protocol is a property OF THE MODEL, and until now it was a
parameter the caller guessed.

WHAT A CAPABILITY IS, AND IS NOT
--------------------------------
Every field here is either OBSERVED by asking the server, or DECLARED by an
operator who knows. Nothing is inferred from a model's name, and nothing is
guessed from its size. ``UNKNOWN`` is a legal value and it is honest: a
capability nobody established is not a capability, and the correct response to
not knowing is to say so, not to pick a plausible default and hope.

The probe is deliberately small. It answers the questions that change what the
core does, and refuses the temptation to build a model zoo.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "GAFITAS_ENGINE_CAPABILITIES_V1"

#: What a capability says when nobody has established it. Never silently a
#: default: the core must be able to tell "no" apart from "we never asked".
UNKNOWN = None

DEFAULT_TAGS_URL = "http://127.0.0.1:11434/api/tags"
DEFAULT_SHOW_URL = "http://127.0.0.1:11434/api/show"
DEFAULT_CHAT_URL = "http://127.0.0.1:11434/api/chat"

#: Where a probe's findings are cached, so certifying an engine is a one-off.
REGISTRY = Path(__file__).resolve().parent.parent / "ENGINES.json"

#: What the core falls back to when a context window could not be established.
#: Conservative on purpose: too small wastes budget, too large loses the run.
CONSERVATIVE_CONTEXT = 8192
#: Kept for the probes, which deliberately ask small questions and should not
#: pay for a large ceiling to do it. It is NO LONGER the working output budget:
#: see working_output() and F-101 for why a fixed guess there was cutting real
#: answers off, including the reference engine's.
CONSERVATIVE_OUTPUT = 1024

#: Keys under which a server returns an engine's reasoning separately from its
#: answer. Checked as a set of observed field names rather than by engine name,
#: because that is what the property IS -- and a new key here is a deployment
#: fact, not a model rule.
REASONING_KEYS = ("thinking", "reasoning", "reasoning_content")

#: The payload a real mission asks a tool to carry, in characters. The median
#: RepoQA needle body is around 1200 and a source edit is routinely larger, so
#: an engine that cannot push a kilobyte through its native channel is better
#: driven by the text protocol. A threshold, not a per-engine rule: it is
#: compared against a number every engine is measured for.
PAYLOAD_FLOOR = 1024


#: I15. Every engine here is used exactly as its authors published it. Nothing
#: in this module describes a model we trained, tuned, adapted or re-quantised,
#: and nothing ever should: a capability is something an engine BROUGHT, and a
#: registry that mixed those with capabilities we installed would stop being a
#: description of what is available and become a description of what we happen
#: to have built. That is also what makes a number measured here transfer to the
#: next engine instead of belonging to one checkpoint.


@dataclass(frozen=True)
class EngineCapabilities:
    """What one engine can actually do, observed or declared -- never assumed."""

    name: str
    #: What the ARCHITECTURE supports, observed from the server's metadata.
    #: For the reference engine this reads 262144, which is true and useless on
    #: its own: it is a ceiling, not an allocation.
    context_window: int | None = UNKNOWN
    #: What THIS DEPLOYMENT actually serves -- bounded by OLLAMA_CONTEXT_LENGTH,
    #: by VRAM, and by what the campaign asks for. Declared, because only the
    #: operator running the server knows it, and the difference between the two
    #: numbers is the difference between a transcript budget that fits and one
    #: that silently overflows.
    served_context: int | None = UNKNOWN
    #: Tokens we ask it to produce per turn. Declared, bounded by the window.
    max_output_tokens: int | None = UNKNOWN
    #: Will it emit a structured tool call when handed a tool schema?
    supports_native_tools: bool | None = UNKNOWN
    #: How many characters of payload survive a NATIVE tool call, measured.
    #: ``supports_native_tools`` answers "can it", which for granite4.1:3b is
    #: yes and was worth nothing: it emits a flawless call below ~400
    #: characters, and above that Ollama's parser for its template returns
    #: content empty AND tool_calls empty -- three hundred generated tokens
    #: gone, nothing left to recover. This answers "how much", which is the
    #: question the work actually asks.
    native_payload_limit: int | None = UNKNOWN
    #: Can it be driven by the JSON-in-text protocol instead?
    supports_text_tool_protocol: bool | None = UNKNOWN
    #: The same number for the text protocol, usually far larger because no
    #: provider-side template parser stands between the tokens and us.
    text_payload_limit: int | None = UNKNOWN
    #: Whether this engine can READ a tool result delivered under role="tool"
    #: as the last message of the conversation -- which is the exact shape the
    #: harness produces every single turn. Observed at the wire, never inferred
    #: from a family: one engine answers with a single token and empty content
    #: 3 times out of 3 on the tool role, and correctly 3 times out of 3 on the
    #: same conversation delivered as a user message. UNKNOWN keeps the current
    #: behaviour, because moving an engine's channel on a guess is how the
    #: protocol rule went wrong the first time (F-102).
    tool_role_supported: bool | None = UNKNOWN
    #: Whether the server returns this engine's reasoning in a channel of its
    #: own, separate from the answer. Observed, never inferred from the name.
    #: It changes the output budget and nothing else: an engine that thinks
    #: out loud must be able to afford the thought AND the call, and
    #: granite4.2:3b lost 70% of its turns to a budget that only covered one.
    emits_reasoning_channel: bool | None = UNKNOWN
    #: F-120. How many tokens this engine has been MEASURED to spend before it
    #: answers, at the high end rather than the median -- a budget has to cover
    #: a bad turn, not a typical one. Declared only from sealed telemetry, never
    #: from a family, a name or a parameter count.
    #:
    #: It does not change the budget. `working_output()` is bounded by a quarter
    #: of the served context and that bound is principled; raising it on the
    #: strength of a measurement of the symptom would be tuning. What this does
    #: is make the shortfall VISIBLE, so a run whose budget cannot cover the
    #: engine's own deliberation plus an answer says so instead of being
    #: recorded as the engine failing.
    measured_reasoning_tokens: int | None = UNKNOWN
    structured_output: bool | None = UNKNOWN
    streaming: bool | None = UNKNOWN
    #: Does the server report token counts? Without this, cost is unmeasurable
    #: and every economic claim about the engine is decoration.
    usage_reporting: bool | None = UNKNOWN
    #: Free-text constraints an operator or a probe established the hard way.
    known_protocol_constraints: tuple[str, ...] = ()
    #: How this was established, so a declared capability is never mistaken for
    #: an observed one.
    source: str = "declared"
    probed_at: float | None = None
    parameter_size: str | None = None
    quantization: str | None = None

    # ------------------------------------------------------------------ core
    @property
    def protocol(self) -> str:
        """Which tool protocol the core should use with this engine.

        'A' native, 'J' JSON-in-text. Not a caller's guess any more: it is a
        property of the engine, and an engine that supports neither is not
        usable and says so rather than failing on turn one.

        "Supports native tools" is necessary and not sufficient. An engine whose
        native channel loses everything above PAYLOAD_FLOOR cannot write a file
        through it, so a measured limit below that floor sends it to the text
        protocol. An UNMEASURED limit changes nothing -- every engine certified
        before this existed keeps the protocol it was certified with.
        """
        native = bool(self.supports_native_tools)
        text_limit = (self.text_payload_limit
                      if isinstance(self.text_payload_limit, int) else -1)
        native_limit = (self.native_payload_limit
                        if isinstance(self.native_payload_limit, int) else -1)
        if (native and native_limit >= 0
                and native_limit < PAYLOAD_FLOOR
                and self.supports_text_tool_protocol
                # ...and only if the text channel is actually BETTER. Sending an
                # engine to a narrower channel because its wide one is not wide
                # enough is not an adaptation, it is a downgrade with a reason
                # attached. qwen3.5:2b measures 800 native against 400 text:
                # both under the floor, and native is twice the size.
                and text_limit > native_limit):
            # Measured, not guessed: this engine's native channel cannot carry
            # what a real edit weighs. Choosing it anyway on the strength of
            # supports_native_tools alone is exactly the reading that cost
            # fifty granite runs, every one of them scored as a model failure.
            return "J"
        if native:
            return "A"
        if self.supports_text_tool_protocol:
            return "J"
        raise ValueError(
            f"engine {self.name!r} declares neither native tool calls nor the "
            f"text protocol. Probe it (localprog engines --probe) or declare "
            f"one explicitly; the core will not guess."
        )

    def working_context(self) -> int:
        """What the core may actually plan against.

        The smaller of what the architecture can hold and what this deployment
        hands out. Taking the architecture's number alone is how a 262144-token
        ceiling becomes a transcript budget on a server configured for 32768.
        """
        limits = [n for n in (self.context_window, self.served_context) if n]
        return min(limits) if limits else CONSERVATIVE_CONTEXT

    def working_output(self) -> int:
        """Output budget, never more than a quarter of the window.

        A model asked to produce more than it can hold alongside its own prompt
        spends the run being truncated. The quarter is a bound, not a tuning
        knob: it exists so an engine with a small window degrades instead of
        breaking.

        Where the engine DECLARES a limit, that limit is obeyed. Where it does
        not, the harness used to guess 1024, and that guess was cutting answers
        off: an engine with a reasoning channel spent a median 974 of its 1024
        tokens thinking and returned empty content in 170 of 242 turns (F-95),
        and an engine with NO reasoning channel hit the same cap on 8 bare
        cases out of 8 (F-101). One symptom, two mechanisms, one cause -- a
        number too small for the answer being asked for.
        """
        ceiling = max(self.working_context() // 4, 256)
        if self.max_output_tokens:
            return min(self.max_output_tokens, ceiling)
        # F-101. There used to be a CONSERVATIVE_OUTPUT of 1024 here, raised to
        # the ceiling only for engines with an observed reasoning channel. The
        # ladder killed that rule: measured on the real prompt over eight cases,
        # cap-hits by budget were 0/8 at 1024 for one engine, 3/8 at 1024 for
        # the reference engine, and 8/8 for an engine with no reasoning channel
        # at all. The condition was diagnosing how an engine SPENDS the budget
        # when the question is whether the budget covers the answer.
        #
        # The ceiling is the principled bound and always was: a quarter of the
        # served context, so nothing is asked to produce more than it can hold
        # beside its own prompt. The guess underneath it goes.
        return ceiling

    #: What an answer needs after the thinking is paid for. Not a guess about
    #: any engine: it is the size of the largest single artefact the tool
    #: surface can be asked to emit in one call, which is a property of the
    #: harness, not of a model.
    ANSWER_ALLOWANCE = 1024

    def output_shortfall(self) -> int:
        """Tokens by which this deployment's budget falls short, or 0.

        F-120. Every engine in the campaign is served 32768 and so gets 8192
        output, and one of them still ends 16 of 50 bare answers in a state a
        finished answer does not reach -- at 8192 tokens, which is far more
        than any answer asked for. The budget is not too small for the ANSWER.
        It is too small for the answer plus what that engine spends before it.

        Returns 0 when nothing has been measured. An unmeasured engine is not
        declared healthy here; it is declared unmeasured, and the caller can
        tell the two apart because the measurement is a field it can read.
        """
        if not self.measured_reasoning_tokens:
            return 0
        needed = self.measured_reasoning_tokens + self.ANSWER_ALLOWANCE
        return max(needed - self.working_output(), 0)

    def output_is_sufficient(self) -> bool | None:
        """True, False, or None when it has never been measured.

        Three-valued on purpose. Collapsing "measured and fine" with "never
        looked" is how F-101 survived two cohorts.
        """
        if not self.measured_reasoning_tokens:
            return None
        return self.output_shortfall() == 0

    def served_context_for(self, shortfall: int | None = None) -> int:
        """The served context that would close the shortfall.

        Stated, not applied. Raising a served context costs VRAM that has not
        been measured, and that is the operator's call and a GPU question.
        This exists so the recommendation is arithmetic rather than a feeling.
        """
        gap = self.output_shortfall() if shortfall is None else shortfall
        if gap <= 0:
            return self.working_context()
        return (self.working_output() + gap) * 4

    def to_dict(self) -> dict:
        out = asdict(self)
        out["known_protocol_constraints"] = list(self.known_protocol_constraints)
        out["schema"] = SCHEMA
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "EngineCapabilities":
        allowed = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in allowed}
        clean["known_protocol_constraints"] = tuple(
            clean.get("known_protocol_constraints") or ())
        return cls(**clean)


# ------------------------------------------------------------------- probing


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _context_from_show(model: str, *, show_url: str, timeout: float) -> tuple:
    """The window, the parameter count and the quantisation, from the server.

    Ollama reports the architecture's own ``context_length`` under a key named
    after the architecture, which differs per model family, so the key is
    searched for rather than assumed.
    """
    try:
        body = _post(show_url, {"model": model}, timeout)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return UNKNOWN, UNKNOWN, UNKNOWN
    info = body.get("model_info") or {}
    window = UNKNOWN
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            window = value
            break
    details = body.get("details") or {}
    return window, details.get("parameter_size"), details.get("quantization_level")


#: The smallest possible tool schema. If an engine will not call THIS, it will
#: not call anything, and the failure is the engine's rather than the task's.
_PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "report_ok",
        "description": "Report that you can call a tool. Call it once.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string",
                                     "description": "the word ok"}},
            "required": ["value"],
        },
    },
}]


def probe(model: str, *, chat_url: str = DEFAULT_CHAT_URL,
          show_url: str = DEFAULT_SHOW_URL, timeout: float = 180.0,
          num_ctx: int = 8192) -> EngineCapabilities:
    """Establish what an engine can do by asking it, not by recognising its name.

    Two questions decide everything the core needs: how much context, and which
    tool protocol. The rest is recorded because it is free once we are talking
    to the server anyway.
    """
    window, params, quant = _context_from_show(model, show_url=show_url,
                                               timeout=timeout)
    constraints: list[str] = []

    native = False
    usage = False
    reasoning = False
    try:
        body = _post(chat_url, {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Call report_ok with value 'ok'. Nothing else."}],
            "tools": _PROBE_TOOL,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": 128},
        }, timeout)
        message = (body or {}).get("message") or {}
        native = bool(message.get("tool_calls"))
        usage = isinstance(body.get("eval_count"), int)
        reasoning = any(str(message.get(key) or "").strip() for key in REASONING_KEYS)
        if not native and message.get("content"):
            constraints.append(
                "handed a tool schema it answered in prose: native tool calls "
                "not observed")
    except urllib.error.HTTPError as exc:
        constraints.append(f"native tool probe rejected with HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        constraints.append(f"native tool probe failed: {type(exc).__name__}")

    text_protocol = UNKNOWN
    if not native:
        # Only worth asking when the native answer was no: an engine that can do
        # native tools is driven that way and the fallback is never exercised.
        try:
            body = _post(chat_url, {
                "model": model,
                "messages": [{"role": "user", "content":
                              'Answer with one JSON object and nothing else: '
                              '{"tool": "report_ok", "arguments": {"value": "ok"}}'}],
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": 128},
            }, timeout)
            content = ((body or {}).get("message") or {}).get("content") or ""
            text_protocol = '"tool"' in content and "report_ok" in content
            if not text_protocol:
                constraints.append("did not reproduce the JSON tool protocol on demand")
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            text_protocol = UNKNOWN

    return EngineCapabilities(
        name=model,
        context_window=window,
        served_context=num_ctx,   # what we just proved it will accept
        emits_reasoning_channel=reasoning,

        max_output_tokens=UNKNOWN,          # declared per campaign, never guessed
        supports_native_tools=native,
        supports_text_tool_protocol=True if native else text_protocol,
        structured_output=UNKNOWN,
        streaming=UNKNOWN,
        usage_reporting=usage,
        known_protocol_constraints=tuple(constraints),
        source="probed",
        probed_at=time.time(),
        parameter_size=params,
        quantization=quant,
    )


# ------------------------------------------------------------------ registry


def load_registry(path: Path | None = None) -> dict[str, EngineCapabilities]:
    target = Path(path or REGISTRY)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: EngineCapabilities.from_dict(body)
            for name, body in (raw.get("engines") or {}).items()}


def save_registry(engines: dict[str, EngineCapabilities],
                  path: Path | None = None) -> Path:
    target = Path(path or REGISTRY)
    target.write_text(json.dumps({
        "schema": SCHEMA,
        "engines": {name: caps.to_dict() for name, caps in sorted(engines.items())},
    }, indent=1) + "\n", encoding="utf-8")
    return target


def capabilities_for(model: str, *, path: Path | None = None,
                     probe_if_missing: bool = False) -> EngineCapabilities:
    """What we know about *model*, from the registry, or freshly probed.

    An engine nobody has certified comes back with everything UNKNOWN and a
    conservative working context. It runs; it simply does not get the benefit
    of assumptions nobody checked.
    """
    known = load_registry(path)
    if model in known:
        return known[model]
    if probe_if_missing:
        caps = probe(model)
        known[model] = caps
        save_registry(known, path)
        return caps
    return EngineCapabilities(name=model, source="unregistered")
