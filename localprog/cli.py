"""``python -m localprog {selfcheck,screen,step1}``.

Running a real measurement is deliberately an explicit act: no subcommand does
anything by default, every one needs an ``--out`` directory, and ``screen``
refuses to start unless the model list is stated. The harness being buildable
and the gate being run are different decisions belonging to different people.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import deps, evidence, screen, step1, telemetry_bridge, work
from .errors import HarnessInvalid
from .provider import CodexProvider, OllamaProvider


def _write_screen_report(out: Path, records, verdicts, suffix_used: bool) -> None:
    lines = ["# SCREEN_V0", "", "## Runs", ""]
    lines += [
        f"- `{r.model}` / {r.protocol} / run {r.run}: **{sum(r.steps)}/5** "
        f"({r.outcome}{'' if r.scoreable else ', NO PUNTUABLE'}), "
        f"{r.turnos_usados} turnos, {r.llamadas_invalidas} inválidas, {r.bucles} bucles"
        for r in records
    ]
    lines += ["", "## Veredicto por modelo (§D.5, congelado)", ""]
    lines += [f"- `{model}` / {protocol}: **{verdict}**" for (model, protocol), verdict in verdicts.items()]
    advancing = sorted({m for (m, _), v in verdicts.items() if v in screen.ADVANCING})
    lines += ["", "## Avanzan al Paso 1", ""]
    lines += [f"- {m}" for m in advancing] or ["- (ninguno)"]
    if not advancing:
        lines += ["", "**STOP.** §D.5: si cero modelos alcanzan PASS_*, no construir más.",
                  "No ajustar el prompt y reintentar."]
    if suffix_used:
        lines += ["", "> Nota: PROTOCOL_B_SUFFIX aplicado en las runs de protocolo B "
                  "(formato §D.4). El prompt §D.3 está intacto."]
    (out / "SCREEN_V0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_selfcheck(args) -> int:
    print(json.dumps({"provenance": deps.provenance(),
                      "tools": sorted(__import__("localprog.tools", fromlist=["SPECS"]).SPECS)},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_screen(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models or list(screen.DEFAULT_MODELS)
    factory = lambda name: OllamaProvider(name, num_ctx=args.num_ctx)  # noqa: E731

    records = []
    for model in models:
        for number in range(1, args.runs + 1):
            records.append(screen.run_once(model, "A", number, provider_factory=factory, out_dir=out))

    if not args.protocol_a_only:
        by_model = {}
        for r in records:
            by_model.setdefault(r.model, []).append(r)
        for model in models:
            if screen.classify(by_model[model]) not in screen.ADVANCING:
                for number in range(1, args.runs + 1):
                    records.append(screen.run_once(model, "B", number, provider_factory=factory, out_dir=out))

    grouped: dict = {}
    for r in records:
        grouped.setdefault((r.model, r.protocol), []).append(r)
    verdicts = {key: screen.classify(runs) for key, runs in grouped.items()}

    (out / "SCREEN_V0_RESULTS.json").write_text(
        json.dumps({"provenance": deps.provenance(),
                    "verdicts": {f"{m}|{p}": v for (m, p), v in verdicts.items()},
                    "runs": [r.to_dict() for r in records]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    voided = [r.to_dict() | {"label": r.label} for r in records if r.harness_invalid]
    if voided:
        evidence.harness_invalid_notice(out, runs=voided)
    _write_screen_report(out, records, verdicts, any(r.protocol == "B" for r in records))
    print(f"{len(records)} runs -> {out}")
    return 2 if voided else 0


def cmd_step1(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    telemetry = telemetry_bridge.open_telemetry(
        args.run_id, Path(args.telemetry_db) if args.telemetry_db else None
    )
    factory = lambda name: OllamaProvider(name, num_ctx=args.num_ctx)  # noqa: E731
    records = []
    try:
        for path in args.missions:
            mission = step1.load_mission(Path(path))
            records.append(step1.run_mission(
                mission, args.model, protocol=args.protocol,
                provider_factory=factory, out_dir=out, telemetry=telemetry,
            ))
    finally:
        telemetry.close()

    passed = [r for r in records if r.tester_pass]
    (out / "STEP1_RESULTS.json").write_text(
        json.dumps({"provenance": deps.provenance(), "model": args.model,
                    "tester_pass": len(passed), "total": len(records),
                    "runs": [r.to_dict() for r in records]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    voided = [r.to_dict() | {"label": r.label} for r in records if r.harness_invalid]
    if voided:
        evidence.harness_invalid_notice(out, runs=voided)
    (out / "STEP1_REPORT.md").write_text(
        "# STEP1\n\n"
        + "\n".join(f"- `{r.mission_id}`: tester_pass={r.tester_pass} ({r.outcome}, "
                    f"{r.turns_used} turnos, {r.invalid_calls} inválidas)" for r in records)
        + f"\n\n**{len(passed)}/{len(records)}**. La puerta §E la evalúa el operador.\n",
        encoding="utf-8",
    )
    print(f"{len(passed)}/{len(records)} tester_pass -> {out}")
    return 2 if voided else 0


def cmd_work(args) -> int:
    """Do real tickets. The productive entrypoint (F-11).

    Unlike ``screen`` and ``step1`` this is not a gate: it does not classify a
    model, it does a job. The exit code is about whether the WORK is usable, so
    a caller can chain it -- 0 every ticket passed, 1 something failed or was
    refused by the conscience, 2 a harness defect voided a run.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    telemetry = telemetry_bridge.open_telemetry(
        args.run_id, Path(args.telemetry_db) if args.telemetry_db else None
    )
    # LOCAL and LUNA are reached completely differently -- one is a tool-calling
    # HTTP endpoint, the other a CLI agent driven through a JSON text protocol --
    # and the loop, the tools, the containment and the conscience are identical
    # for both. That is the whole point of the tier being a provider rather than
    # a second runner (F-13).
    if args.tier == "LUNA":
        factory = lambda name: CodexProvider(name, effort=args.effort, timeout=args.timeout)  # noqa: E731
        protocol = "J"
    else:
        factory = lambda name: OllamaProvider(  # noqa: E731
            name, num_ctx=args.num_ctx, num_predict=args.num_predict, timeout=args.timeout
        )
        protocol = "A"
    records = []
    try:
        for path in args.tickets:
            ticket = work.load_ticket(Path(path))
            if args.max_turns:
                ticket = replace(ticket, max_turns=args.max_turns)
            record = work.run_ticket(
                ticket, args.model, provider_factory=factory,
                out_dir=out, telemetry=telemetry, num_ctx=args.num_ctx,
                protocol=protocol,
            )
            records.append(record)
            print(f"  {record.ticket_id:<16} {record.outcome:<24} "
                  f"{record.turns_used:>3} turnos  "
                  f"{record.usage.get('output_tokens', 0):>6} tok_out  "
                  f"{record.wall_seconds:>6.1f}s")
    finally:
        telemetry.close()

    passed = [r for r in records if r.outcome in (work.PASS, work.PASS_UNCONFIRMED)]
    voided = [r for r in records if r.outcome in (work.HARNESS_INVALID, work.NON_DISCRIMINATING)]
    (out / "WORK_RESULTS.json").write_text(
        json.dumps({
            "provenance": deps.provenance(), "model": args.model,
            "passed": len(passed), "total": len(records),
            "usage_total": _sum_usage(records),
            "records": [r.to_dict() for r in records],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_work_report(out, records)
    print(f"{len(passed)}/{len(records)} PASS -> {out}")
    if any(r.outcome == work.HARNESS_INVALID for r in records):
        evidence.harness_invalid_notice(
            out, runs=[r.to_dict() | {"label": r.label} for r in records
                       if r.outcome == work.HARNESS_INVALID]
        )
        return 2
    return 0 if len(passed) == len(records) and not voided else 1


def _sum_usage(records) -> dict:
    """Cost, split by model class. Local, Luna and Claude are different budgets
    with different prices, and adding them up hides which part is expensive."""
    total: dict = {}
    for r in records:
        bucket = total.setdefault(r.model_class, {
            "tickets": 0, "calls": 0, "input_tokens": 0,
            "output_tokens": 0, "cached_tokens": 0, "wall_seconds": 0.0,
        })
        bucket["tickets"] += 1
        bucket["wall_seconds"] = round(bucket["wall_seconds"] + r.wall_seconds, 2)
        for key in ("calls", "input_tokens", "output_tokens", "cached_tokens"):
            bucket[key] += int(r.usage.get(key, 0) or 0)
    return total


NL = chr(10)


def _write_work_report(out: Path, records) -> None:
    lines = ["# GAFITAS WORK", ""]
    for r in records:
        lines.append(
            f"- `{r.ticket_id}` **{r.outcome}** ({r.model}, {r.loop_outcome}, "
            f"{r.turns_used} turnos, {r.invalid_calls} inv, {r.tool_errors} err)"
        )
        if r.changed_files:
            lines.append(f"  - cambiados: {', '.join(r.changed_files[:8])}")
        for note in r.notes[:4]:
            lines.append(f"  - {note}")
    lines += ["", "## Coste por clase de modelo", ""]
    for cls, bucket in _sum_usage(records).items():
        lines.append(
            f"- **{cls}**: {bucket['tickets']} tickets, {bucket['calls']} llamadas, "
            f"{bucket['input_tokens']} tok_in, {bucket['output_tokens']} tok_out, "
            f"{bucket['wall_seconds']:.1f}s"
        )
    (out / "WORK_REPORT.md").write_text(NL.join(lines) + NL, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localprog")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selfcheck", help="Print dependency provenance and the tool list")

    p = sub.add_parser("screen", help="Paso 0 (contract §D)")
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--runs", type=int, default=screen.RUNS_PER_MODEL)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--protocol-a-only", action="store_true")

    p = sub.add_parser("step1", help="Paso 1 (contract §E)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--missions", nargs="+", required=True)
    p.add_argument("--protocol", choices=("A", "B"), default="A")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--run-id", default="LOCAL_PROGRAMMER_STEP1")
    p.add_argument("--telemetry-db", default=None)

    p = sub.add_parser("work", help="Do real tickets (the productive path)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tickets", nargs="+", required=True)
    p.add_argument("--tier", choices=("LOCAL", "LUNA"), default="LOCAL",
                   help="LOCAL: Ollama + native tool calls. LUNA: Codex CLI + protocol J.")
    p.add_argument("--effort", default="low", help="LUNA only: Codex reasoning effort.")
    p.add_argument("--num-ctx", type=int, default=work.WORK_NUM_CTX)
    p.add_argument("--num-predict", type=int, default=work.WORK_NUM_PREDICT)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--run-id", default="GAFITAS_WORK")
    p.add_argument("--telemetry-db", default=None)

    args = parser.parse_args(argv)
    try:
        return {"selfcheck": cmd_selfcheck, "screen": cmd_screen,
                "step1": cmd_step1, "work": cmd_work}[args.command](args)
    except HarnessInvalid as exc:
        print(f"HARNESS_INVALID: {exc.detail}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
