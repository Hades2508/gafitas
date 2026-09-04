"""MULTI_TURN_TOOL_PERSISTENCE: can this engine still drive the loop at turn 20?

WHY THIS EXISTS
---------------
The engine probe in ``engine.py`` established ``supports_native_tools`` with one
call in one turn, and that number was wrong about the thing it was used for.
granite4.1:3b and qwen2.5-coder:3b both passed it, and both then answered ~50%
of a task in a single prompt and 4% / 0% through the loop. Fifty granite runs
produced 98 turns of prose where a tool call was required.

Emitting one tool call is not the same capability as emitting the ninetieth,
with a transcript in front of it, after a tool error, while holding an objective
stated nineteen turns ago. This module measures THAT, before a benchmark spends
GPU discovering it.

WHAT IT IS NOT
--------------
It is not a task. Every step has one unambiguous correct call, stated in the
turn immediately before it, and the tool results are synthetic and fixed. An
engine that fails here did not fail to program -- it failed to stay in the
protocol, which is the only thing being asked.

Keeping it free of task difficulty is what makes it cheap (24 calls, a couple of
minutes) and what makes a failure attributable. A probe that could also be
failed by not understanding the repository would tell us nothing about why.

THE EIGHT MEASUREMENTS
----------------------
valid_tool_call_rate  a parseable call to the tool that was asked for
no_tool_call          prose where a call was required
wrong_tool            a call, but to something else
payload_loss          a token planted at turn 1 that must survive to turn 20
objective_loss        the mission, restated on demand at the end
stall                 consecutive dead turns, the loop's own STALLED rule
premature_finish      finish() before the script says so
recovery              a corrected call after a deliberate tool error

CHECKPOINTS
-----------
Reported at turns 1, 3, 5, 10 and 20, because the interesting number is not the
mean -- it is the shape. An engine that holds until turn 5 and collapses is a
context/attention problem; one that never starts is a protocol problem; one that
degrades smoothly is a size problem. Those three want different fixes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import protocol, tools
from .errors import ERROR_NO_TOOL_CALL, InvalidCall, ProviderError

#: Reported checkpoints. The last one is the loop's own default budget.
CHECKPOINTS = (1, 3, 5, 10, 20)

#: The planted token. Deliberately not a word: it cannot be reconstructed from
#: context or guessed from the task, so reproducing it is evidence of carrying
#: it rather than of having re-derived it.
TOKEN = "QX7K4M2R"

STEPS = 20

SYSTEM = """Eres un agente que trabaja llamando a herramientas.

En cada turno se te dice EXACTAMENTE que llamada tienes que hacer. Hazla.
No expliques, no razones en voz alta, no escribas texto: emite la llamada.

El identificador de esta sesion es {token}. Tendras que repetirlo mas adelante.
El objetivo de la mision es: {objective}
"""

OBJECTIVE = "revisar el modulo de facturacion del repositorio de ejemplo"

#: The script. Each entry is (instructed phrasing, goal phrasing, expected
#: tool, acceptable tools, checker).
#: ``checker`` receives the parsed arguments and returns None if correct, or a
#: short reason. Every instruction names the tool and its arguments literally --
#: there is nothing to infer.
_SCRIPT: list[tuple[str, str, Any]] = []


#: What the model is told each turn.
#:
#: INSTRUCTED names the call and its arguments. It measures whether the engine
#: can STAY IN the protocol -- nothing is left to decide.
#:
#: GOAL states what is needed and leaves the choice of tool to the engine. The
#: difference between the two is the cost of DECIDING, which is the thing the
#: real loop charges every turn and the probe's first version did not measure.
INSTRUCTED = "instructed"
GOAL = "goal"
MODES = (INSTRUCTED, GOAL)


def _arg(args: Any, key: str) -> str:
    if not isinstance(args, dict):
        return ""
    value = args.get(key)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _build_script() -> list[tuple[str, str, str, tuple[str, ...], Any]]:
    """Twenty steps: eight reads, four searches, an error, a payload recall.

    Each step carries BOTH phrasings. ``acceptable`` is the set of tools a
    careful engineer could defend for that goal -- it is not scored as success,
    but it separates "picked a different reasonable tool" from "picked something
    that cannot do this", and those two are different defects.
    """
    script: list[tuple[str, str, str, tuple[str, ...], Any]] = []

    def path_step(n: int) -> None:
        rel = f"src/modulo_{n:02d}.py"
        script.append((
            f"Llama a read_file con path='{rel}'.",
            f"Necesitas ver el contenido de {rel}.",
            "read_file", ("read_file",),
            lambda a, rel=rel: None if _arg(a, "path") == rel
            else f"path={_arg(a, 'path')!r} en vez de {rel!r}",
        ))

    def search_step(term: str, goal: str) -> None:
        script.append((
            f"Llama a search_code con query='{term}'.",
            goal,
            "search_code", ("search_code", "grep"),
            lambda a, term=term: None if _arg(a, "query") == term
            else f"query={_arg(a, 'query')!r} en vez de {term!r}",
        ))

    path_step(1)                                             # 1
    search_step("calcular impuesto",                         # 2
                "Necesitas encontrar en el repositorio donde se calcula el impuesto.")
    path_step(2)                                             # 3
    script.append((                                          # 4
        "Llama a grep con pattern='def facturar'.",
        "Necesitas las lineas que contienen literalmente el texto 'def facturar'.",
        "grep", ("grep", "search_code"),
        lambda a: None if _arg(a, "pattern") == "def facturar"
        else f"pattern={_arg(a, 'pattern')!r}",
    ))
    path_step(3)                                             # 5
    search_step("aplicar descuento",                         # 6
                "Necesitas encontrar donde se aplica el descuento.")
    script.append((                                          # 7
        "Llama a list_dir con path='src'.",
        "Necesitas saber que ficheros hay dentro del directorio src.",
        "list_dir", ("list_dir",),
        lambda a: None if _arg(a, "path") == "src" else f"path={_arg(a, 'path')!r}",
    ))
    path_step(4)                                             # 8
    search_step("total pedido",                              # 9
                "Necesitas encontrar donde se calcula el total del pedido.")
    path_step(5)                                             # 10
    script.append((                                          # 11
        "Llama a read_file con path='src/modulo_06.py' y start=10 y end=40.",
        "Necesitas ver SOLO las lineas 10 a 40 de src/modulo_06.py.",
        "read_file", ("read_file",),
        lambda a: None if (_arg(a, "path") == "src/modulo_06.py"
                           and str(a.get("start")) == "10"
                           and str(a.get("end")) == "40")
        else f"path/start/end = {_arg(a, 'path')!r}/{a.get('start')}/{a.get('end')}",
    ))
    script.append((                                          # 12 -- the error
        "Llama a read_file con path='src/no_existe.py'.",
        "Necesitas ver el contenido de src/no_existe.py.",
        "read_file", ("read_file",),
        lambda a: None if _arg(a, "path") == "src/no_existe.py"
        else f"path={_arg(a, 'path')!r}",
    ))
    script.append((                                          # 13 -- RECOVERY
        "Ese fichero no existe. Llama a read_file con path='src/modulo_07.py'.",
        "Ese fichero no existe. El que necesitas ver es src/modulo_07.py.",
        "read_file", ("read_file",),
        lambda a: None if _arg(a, "path") == "src/modulo_07.py"
        else f"path={_arg(a, 'path')!r}",
    ))
    search_step("validar cliente",                           # 14
                "Necesitas encontrar donde se valida el cliente.")
    path_step(8)                                             # 15
    script.append((                                          # 16
        "Llama a grep con pattern='import decimal' y glob='**/*.py'.",
        "Necesitas las lineas que contienen literalmente 'import decimal'.",
        "grep", ("grep", "search_code"),
        lambda a: None if _arg(a, "pattern") == "import decimal"
        else f"pattern={_arg(a, 'pattern')!r}",
    ))
    path_step(9)                                             # 17
    script.append((                                          # 18 -- PAYLOAD
        "Llama a search_code con query igual al identificador de sesion que se "
        "te dio al principio.",
        "Busca en el repositorio el identificador de sesion que se te dio al "
        "principio de la conversacion.",
        "search_code", ("search_code", "grep"),
        lambda a: None if TOKEN in _arg(a, "query").upper()
        else f"query={_arg(a, 'query')!r} no contiene el identificador",
    ))
    script.append((                                          # 19 -- OBJECTIVE
        "Llama a search_code con query igual al objetivo de la mision que se te "
        "dio al principio.",
        "Busca en el repositorio el objetivo de la mision que se te dio al "
        "principio de la conversacion.",
        "search_code", ("search_code", "grep"),
        lambda a: None if "factura" in _arg(a, "query").lower()
        else f"query={_arg(a, 'query')!r} no recoge el objetivo",
    ))
    script.append((                                          # 20 -- FINISH
        "Ya esta. Llama a finish con status='NO_CHANGE' y un summary corto.",
        "El trabajo esta terminado y no has cambiado ningun fichero. Cierra la "
        "mision declarando que no hizo falta ningun cambio.",
        "finish", ("finish",),
        lambda a: None if str(a.get("status", "")).upper() == "NO_CHANGE"
        else f"status={a.get('status')!r} en vez de NO_CHANGE",
    ))
    return script


_SCRIPT = _build_script()
assert len(_SCRIPT) == STEPS, "the script must be exactly STEPS long"

#: Synthetic tool results. Sized like real ones so the transcript grows the way
#: a real transcript grows -- a probe on a transcript of one-line replies would
#: measure a context pressure nobody ever experiences.
_FILLER = ("def facturar(pedido, cliente):\n"
           "    total = sum(l.importe for l in pedido.lineas)\n"
           "    total = aplicar_descuento(total, cliente.descuento)\n"
           "    return calcular_impuesto(total, cliente.region)\n")


def _synthetic_result(step: int, name: str) -> str:
    if step == 12:                      # the deliberate error
        return ("ERROR_PATH_NOT_FOUND: src/no_existe.py no existe en el "
                "repositorio.")
    if name == "read_file":
        return f"1  # modulo de facturacion\n{_FILLER}" * 3
    if name == "search_code":
        return ("src/modulo_03.py:14  def aplicar_descuento(total, pct)\n"
                "src/modulo_07.py:22  def calcular_impuesto(base, region)\n"
                "src/modulo_02.py:8   def facturar(pedido, cliente)\n")
    if name == "grep":
        return "src/modulo_02.py:8: def facturar(pedido, cliente):\n"
    if name == "list_dir":
        return "src/: " + ", ".join(f"modulo_{i:02d}.py" for i in range(1, 13))
    return "ok"


@dataclass
class TurnRecord:
    step: int
    expected: str
    got: str | None
    valid: bool
    no_tool_call: bool
    wrong_tool: bool
    invalid_call: bool
    reason: str | None
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PersistenceReport:
    """What an engine can sustain, and where it stops sustaining it."""

    engine: str
    protocol: str
    mode: str
    declared_tools: int
    acceptable_tool_rate: float
    steps: int
    valid_tool_call_rate: float
    no_tool_call: int
    wrong_tool: int
    invalid_call: int
    payload_loss: bool
    objective_loss: bool
    stall: bool
    longest_dead_run: int
    premature_finish: int | None
    recovery: bool | None
    checkpoints: dict[str, float]
    seconds: float
    input_tokens: int
    output_tokens: int
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def can_drive_the_loop(self) -> bool:
        """The bar for spending GPU on a benchmark with this engine.

        Deliberately not "perfect". The reference engine is not perfect either.
        It is: mostly stays in the protocol, does not stall, does not quit
        early, and still holds the objective at the end -- because each of those
        four, on its own, was observed to destroy a run.
        """
        return (self.valid_tool_call_rate >= 0.70
                and not self.stall
                and not self.premature_finish
                and not self.objective_loss)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["can_drive_the_loop"] = self.can_drive_the_loop
        return d


def measure(provider, *, engine_name: str, protocol_name: str = "A",
            steps: int = STEPS, mode: str = INSTRUCTED,
            declare: tuple[str, ...] | None = None) -> PersistenceReport:
    """Drive *provider* through the script and report what it sustained.

    The transcript is built the way the loop builds one, with no elision: this
    is the worst case the loop can present, and an engine that survives it will
    survive the windowed version.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    script = _SCRIPT[:steps]
    schema = tools.native_schema(declare) if protocol_name == "A" else None
    system = SYSTEM.format(token=TOKEN, objective=OBJECTIVE)
    if protocol_name != "A":
        system += "\n" + tools.text_manual(declare) + "\n" + protocol.JSON_INSTRUCTIONS

    messages: list[dict] = [{"role": "system", "content": system}]
    records: list[TurnRecord] = []
    dead_run = 0
    longest_dead = 0
    premature: int | None = None
    recovery: bool | None = None
    usage_in = usage_out = 0
    started = time.time()
    error: str | None = None

    acceptable_count = 0
    for step, (instructed, goal, expected, acceptable, check) in enumerate(script, 1):
        messages.append({"role": "user",
                         "content": instructed if mode == INSTRUCTED else goal})
        t0 = time.time()
        try:
            raw = provider.chat(messages, schema)
        except ProviderError as exc:
            error = f"{exc.kind}: {exc}"
            break
        seconds = time.time() - t0
        message = raw.get("message", {}) if isinstance(raw, dict) else {}
        usage = raw if isinstance(raw, dict) else {}
        turn_in = int(usage.get("prompt_eval_count") or 0)
        turn_out = int(usage.get("eval_count") or 0)
        usage_in += turn_in
        usage_out += turn_out

        got: str | None = None
        reason: str | None = None
        no_call = wrong = invalid = False
        try:
            call = protocol.parse(protocol_name, message)
            got = call.name
            if got != expected:
                wrong = True
                reason = f"llamo a {got} en vez de {expected}"
                if got == "finish" and step < len(script):
                    premature = premature or step
            else:
                reason = check(call.arguments)
        except InvalidCall as exc:
            if getattr(exc, "code", "") == ERROR_NO_TOOL_CALL:
                no_call = True
            else:
                invalid = True
            reason = str(exc)[:200]

        valid = got == expected and reason is None
        if got in acceptable:
            acceptable_count += 1
        if step == 13:
            recovery = valid
        if not valid:
            dead_run += 1
            longest_dead = max(longest_dead, dead_run)
        else:
            dead_run = 0

        records.append(TurnRecord(
            step=step, expected=expected, got=got, valid=valid,
            no_tool_call=no_call, wrong_tool=wrong, invalid_call=invalid,
            reason=reason, seconds=round(seconds, 2),
            input_tokens=turn_in, output_tokens=turn_out,
        ))

        # Whatever it did, the script moves on: this measures persistence, not
        # whether the harness can rescue it. The synthetic result is for the
        # tool that was ASKED for, so a wrong call still gets plausible context
        # back and the next step stays answerable.
        messages.append(_assistant_turn(message, protocol_name))
        messages.append({"role": "user",
                         "content": f"RESULTADO:\n{_synthetic_result(step, expected)}"})

    valid_count = sum(1 for r in records if r.valid)
    rate = valid_count / len(records) if records else 0.0
    checkpoints = {}
    for cp in CHECKPOINTS:
        upto = [r for r in records if r.step <= cp]
        if upto:
            checkpoints[f"turn_{cp}"] = round(
                sum(1 for r in upto if r.valid) / len(upto), 3)

    payload_step = next((r for r in records if r.step == 18), None)
    objective_step = next((r for r in records if r.step == 19), None)

    return PersistenceReport(
        engine=engine_name, protocol=protocol_name, mode=mode,
        declared_tools=len(schema) if schema is not None
        else len(declare or tools.SPECS),
        acceptable_tool_rate=round(acceptable_count / len(records), 3) if records else 0.0,
        steps=len(records),
        valid_tool_call_rate=round(rate, 3),
        no_tool_call=sum(1 for r in records if r.no_tool_call),
        wrong_tool=sum(1 for r in records if r.wrong_tool),
        invalid_call=sum(1 for r in records if r.invalid_call),
        payload_loss=(payload_step is None or not payload_step.valid),
        objective_loss=(objective_step is None or not objective_step.valid),
        stall=longest_dead >= 3,
        longest_dead_run=longest_dead,
        premature_finish=premature,
        recovery=recovery,
        checkpoints=checkpoints,
        seconds=round(time.time() - started, 1),
        input_tokens=usage_in, output_tokens=usage_out,
        turns=records, error=error,
    )


def _assistant_turn(message: dict, protocol_name: str) -> dict:
    """Echo the assistant's own turn back, as the loop does.

    Kept verbatim including tool_calls, because an engine's next turn is
    conditioned on the shape of its previous one -- normalising it here would
    make the probe measure a conversation the engine never had.
    """
    out: dict[str, Any] = {"role": "assistant",
                           "content": message.get("content") or ""}
    calls = message.get("tool_calls")
    if calls:
        out["tool_calls"] = calls
    return out


def report_path(base: Path, engine_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", engine_name)
    return base / f"PERSISTENCE_{safe}.json"


# --------------------------------------------------------------- payload arm

#: Sizes in characters. Chosen to bracket what the matched task actually asks
#: for: the median RepoQA needle body is around 1200 characters.
PAYLOAD_SIZES = (400, 1200, 3000)

_PAYLOAD_UNIT = """def calcular_impuesto_{n:02d}(base, region, exento=False):
    \"\"\"Devuelve el impuesto aplicable a *base* en *region*.\"\"\"
    if exento or base <= 0:
        return 0
    tipo = TIPOS.get(region, TIPO_GENERAL)
    bruto = base * tipo
    return round(bruto, 2)

"""


def _payload_text(size: int) -> str:
    """A body of about *size* characters, built from repeated real-looking code.

    Real-looking matters: quotes, docstrings, indentation and a decimal point
    are exactly the characters that a JSON string argument has to escape, and a
    payload of lorem ipsum would measure a channel nobody uses.
    """
    out = []
    n = 0
    while sum(len(x) for x in out) < size:
        n += 1
        out.append(_PAYLOAD_UNIT.format(n=n))
    text = "".join(out)
    return text[:size].rstrip() + "\n"


def _normalise(text: str) -> str:
    """Compare the way the matched evaluator compares: whitespace-insensitive.

    Byte-exactness is reported separately. Holding an engine to bytes when the
    thing that scores it does not would invent a failure.
    """
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@dataclass
class PayloadRecord:
    size: int
    channel: str                 # "prose" | "tool"
    called_tool: bool | None
    got_chars: int
    exact: bool
    normalised_exact: bool
    line_recall: float
    seconds: float
    reason: str | None = None


def measure_payload(provider, *, engine_name: str, protocol_name: str = "A",
                    sizes: tuple[int, ...] = PAYLOAD_SIZES) -> dict:
    """Reproduce the same text twice per size: as prose, then as a tool argument.

    Order is fixed (prose first) and the two are independent conversations, so
    neither can prime the other.
    """
    schema = tools.native_schema(("write_file",)) if protocol_name == "A" else None
    manual = ""
    if protocol_name != "A":
        manual = ("\n" + tools.text_manual(("write_file",))
                  + "\n" + protocol.JSON_INSTRUCTIONS)
    records: list[PayloadRecord] = []

    for size in sizes:
        text = _payload_text(size)
        wanted = _normalise(text)
        wanted_lines = [ln for ln in wanted.splitlines() if ln.strip()]

        # ---- A: prose ----------------------------------------------------
        t0 = time.time()
        prompt = ("Copia EXACTAMENTE el texto que hay entre las marcas, sin "
                  "cambiar nada y sin anadir explicaciones.\n\n"
                  f"--- INICIO ---\n{text}--- FIN ---\n")
        try:
            raw = provider.chat([{"role": "user", "content": prompt}], None)
            got = (raw.get("message", {}) or {}).get("content", "") or ""
            reason = None
        except ProviderError as exc:
            got, reason = "", f"{exc.kind}: {exc}"
        records.append(_score_payload(size, "prose", None, got, text, wanted,
                                      wanted_lines, time.time() - t0, reason))

        # ---- B: the same text, through write_file -------------------------
        t0 = time.time()
        system = ("Eres un agente que trabaja llamando a herramientas." + manual)
        prompt = ("Copia EXACTAMENTE el texto que hay entre las marcas en el "
                  "fichero respuesta.txt, llamando a "
                  "write_file(path='respuesta.txt', content=<el texto entero>). "
                  "El texto va en el argumento content, no en tu respuesta.\n\n"
                  f"--- INICIO ---\n{text}--- FIN ---\n")
        called = False
        got = ""
        reason = None
        try:
            raw = provider.chat([{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}], schema)
            message = raw.get("message", {}) if isinstance(raw, dict) else {}
            try:
                call = protocol.parse(protocol_name, message)
                called = call.name == "write_file"
                if not called:
                    reason = f"llamo a {call.name}"
                else:
                    value = (call.arguments or {}).get("content")
                    got = value if isinstance(value, str) else ""
                    if not isinstance(value, str):
                        reason = f"content es {type(value).__name__}"
            except InvalidCall as exc:
                reason = f"{getattr(exc, 'code', '')}: {exc}"[:200]
        except ProviderError as exc:
            reason = f"{exc.kind}: {exc}"
        records.append(_score_payload(size, "tool", called, got, text, wanted,
                                      wanted_lines, time.time() - t0, reason))

    return {
        "engine": engine_name,
        "protocol": protocol_name,
        "sizes": list(sizes),
        "records": [asdict(r) for r in records],
        # The headline: does the tool channel cost anything the prose channel
        # does not? Everything else in this dict exists to explain this number.
        "prose_normalised": round(sum(
            1 for r in records if r.channel == "prose" and r.normalised_exact
        ) / len(sizes), 3),
        "tool_normalised": round(sum(
            1 for r in records if r.channel == "tool" and r.normalised_exact
        ) / len(sizes), 3),
    }


def _score_payload(size, channel, called, got, text, wanted, wanted_lines,
                   seconds, reason) -> PayloadRecord:
    body = got
    if channel == "prose":
        # A prose answer legitimately arrives inside a fence. Stripping one is
        # not leniency: the matched evaluator does the same, and refusing it
        # here would score formatting rather than reproduction.
        stripped = body.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[-1].strip().startswith("```"):
                body = "\n".join(lines[1:-1])
    norm = _normalise(body)
    present = sum(1 for ln in wanted_lines if ln in norm)
    return PayloadRecord(
        size=size, channel=channel, called_tool=called, got_chars=len(got),
        exact=(body.strip() == text.strip()),
        normalised_exact=(norm == wanted),
        line_recall=round(present / len(wanted_lines), 3) if wanted_lines else 0.0,
        seconds=round(seconds, 1), reason=reason,
    )


#: Ladder for the payload-limit probe. Coarse on purpose: the useful output is
#: an order of magnitude, and each rung costs a full generation.
LIMIT_LADDER = (200, 400, 800, 1600, 3200)


def _one_payload_attempt(provider, protocol_name, schema, manual, size):
    """One generation at one rung. Returns (survived, reason, repaired, tokens).

    Split out so a rung can be SAMPLED. A single attempt is a coin flip: the
    same engine on the same ladder measured 400, then 0, then 3200 on three
    consecutive passes.
    """
    text = _payload_text(size)
    try:
        raw = provider.chat([
            {"role": "system",
             "content": "Eres un agente que trabaja llamando a herramientas." + manual},
            {"role": "user",
             "content": "Copia EXACTAMENTE este texto en respuesta.txt "
                        "llamando a write_file:\n\n" + text},
        ], schema)
    except ProviderError as exc:
        return False, exc.kind, None, 0
    message = raw.get("message", {}) if isinstance(raw, dict) else {}
    tokens = int((raw or {}).get("eval_count") or 0)
    try:
        call = protocol.parse(protocol_name, message)
    except InvalidCall as exc:
        return False, (getattr(exc, "code", "") or str(exc)[:80]), None, tokens
    value = (call.arguments or {}).get("content") if isinstance(call.arguments, dict) else None
    # A call that arrives with a truncated payload is not a surviving call: the
    # tool would write the wrong file. Half is the bar, because below it the
    # answer is unusable and above it the loss is recoverable by re-reading.
    survived = (call.name == "write_file" and isinstance(value, str)
                and len(value) >= size * 0.5)
    reason = None if survived else f"content={len(value) if isinstance(value, str) else None}"
    return survived, reason, call.repaired, tokens


def measure_payload_limit(provider, *, protocol_name: str = "A",
                          ladder: tuple[int, ...] = LIMIT_LADDER,
                          samples: int = 3) -> dict:
    """Largest payload that survives a tool call, in characters.

    SAMPLED and exhaustive, both for the same reason: this number chooses the
    protocol, and the first version could not choose anything.

    It stopped at the first lost rung, on the theory that the failure is a
    parser cliff rather than a gradient. That is true for granite4.1:3b --
    perfect at 400, empty envelope at 800, nothing above -- and false in
    general. And each rung was one generation, so a single stochastic miss set
    the answer: qwen3.5:2b measured 400, then 0, then 3200 on three consecutive
    passes of the same ladder; phi4-mini:3.8b measured 1600, 200, 400.

    Now every rung is climbed and sampled, ``limit`` is the highest rung a
    majority of samples survived, and ``clean_cliff`` says whether the failures
    are all above the survivors -- a real cliff -- or interleaved with them,
    which is noise wearing a cliff's clothes.
    """
    schema = tools.native_schema(("write_file",)) if protocol_name == "A" else None
    manual = ""
    if protocol_name != "A":
        manual = ("\n" + tools.text_manual(("write_file",))
                  + "\n" + protocol.JSON_INSTRUCTIONS)

    rungs = []
    limit = 0
    for size in ladder:
        survived = 0
        reason = repaired = None
        tokens = 0
        for _attempt in range(max(1, samples)):
            ok, why, rep, tok = _one_payload_attempt(
                provider, protocol_name, schema, manual, size)
            survived += 1 if ok else 0
            reason = why or reason
            repaired = rep or repaired
            tokens = tok or tokens
        rate = survived / max(1, samples)
        rungs.append({"size": size, "survived": rate >= 0.5, "rate": round(rate, 2),
                      "samples": samples, "reason": reason, "repaired": repaired,
                      "generated_tokens": tokens})
        if rate >= 0.5:
            limit = size

    ok_sizes = [r["size"] for r in rungs if r["survived"]]
    lost = [r["size"] for r in rungs if not r["survived"]]
    return {
        "protocol": protocol_name,
        "limit": limit,
        "ceiling_reached": limit == ladder[-1],
        "clean_cliff": bool(lost) and bool(ok_sizes) and min(lost) > max(ok_sizes),
        "rungs": rungs,
    }
