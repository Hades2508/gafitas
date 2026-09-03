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
import tempfile
from dataclasses import replace
from pathlib import Path

from . import deps, evidence, route, screen, step1, telemetry_bridge, work
from .errors import HarnessInvalid
from . import runstate
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
    def local_tier(tier: str = route.LOCAL, model: str | None = None,
                   attempts: int = 1) -> dict:
        # Protocol is a property of the MODEL, not of the tier. qwen2.5-coder:7b
        # emits no native tool call at all -- 3 of 3, stalled immediately -- and
        # that looked like an unusable model. Protocol J is model-agnostic and
        # already exists for Luna, so a local model that cannot drive Ollama's
        # native tool channel can still drive the identical loop through JSON.
        # Making this reachable costs one flag and no new infrastructure.
        return {
            "tier": tier, "model": model or args.model,
            "attempts": attempts, "protocol": args.local_protocol,
            "provider_factory": lambda name: OllamaProvider(
                name, num_ctx=args.num_ctx, num_predict=args.num_predict,
                timeout=args.timeout,
            ),
        }

    def luna_tier(model: str) -> dict:
        return {
            "tier": route.LUNA, "model": model, "protocol": "J",
            "provider_factory": lambda name: CodexProvider(
                name, effort=args.effort, timeout=max(args.timeout, 600.0)
            ),
        }

    if args.tier == "LUNA":
        tiers = [luna_tier(args.model)]
    elif args.tier == "AUTO":
        # The normal path is free. LOCAL_FAST retries before LOCAL_STRONG is
        # loaded, and Luna appears only when explicitly asked for -- it is a
        # teacher and a control during development, not a runtime dependency.
        tiers = [local_tier(route.LOCAL, args.model, args.local_attempts)]
        if args.strong_model:
            tiers.append(local_tier(route.LOCAL_STRONG, args.strong_model,
                                    args.strong_attempts))
        if args.allow_luna:
            tiers.append(luna_tier(args.luna_model))
    else:
        tiers = [local_tier(route.LOCAL, args.model, args.local_attempts)]
    # F-41: fail fast and legibly when the provider is not there, instead of
    # producing a batch of PROVIDER_ERRORs that reads like a batch of failures.
    probe = tiers[0]["provider_factory"](tiers[0]["model"])
    complaint = probe.preflight() if hasattr(probe, "preflight") else None
    if complaint:
        print(f"PREFLIGHT: {complaint}", file=sys.stderr)
        telemetry.close()
        return 3

    records = []
    routed_all = []
    try:
        for path in args.tickets:
            base = work.load_ticket(Path(path))
            if args.max_turns:
                base = replace(base, max_turns=args.max_turns)
            for attempt in range(1, args.repeat + 1):
                # Repetitions get distinct ids so their evidence does not
                # overwrite itself. A stochastic system measured once is
                # measured badly: the same model on the same ticket has
                # produced both 5-of-6 tests green and zero edits in forty
                # turns, and a single sample cannot tell a real regression
                # from that spread.
                ticket = base if args.repeat == 1 else replace(
                    base, ticket_id=f"{base.ticket_id}#{attempt}"
                )
                routed = route.run_with_ladder(
                    ticket, tiers=tiers,
                    out_dir=out, telemetry=telemetry, num_ctx=args.num_ctx,
                )
                routed_all.append(routed)
                record = routed.result
                records.append(record)
                trail = " -> ".join(f"{a.tier}:{a.outcome}" for a in routed.attempts)
                print(f"  {record.ticket_id:<16} {record.outcome:<24} "
                      f"{record.turns_used:>3} turnos  "
                      f"{record.usage.get('output_tokens', 0):>6} tok_out  "
                      f"{record.wall_seconds:>6.1f}s  [{trail}]")
    finally:
        telemetry.close()
        # Let go of the VRAM. Ollama holds the weights for five more minutes
        # by default, which on a shared desktop means a 9B sitting on 6.6 GB
        # with the fans up long after the work is done.
        for step in tiers:
            released = step["provider_factory"](step["model"])
            if hasattr(released, "release"):
                released.release()

    passed = [r for r in records if r.outcome in (work.PASS, work.PASS_UNCONFIRMED)]
    voided = [r for r in records if r.outcome in (work.HARNESS_INVALID, work.NON_DISCRIMINATING)]
    (out / "WORK_RESULTS.json").write_text(
        json.dumps({
            "provenance": deps.provenance(), "model": args.model,
            "tier": args.tier,
            "passed": len(passed), "total": len(records),
            "usage_total": _sum_usage(records),
            "routing": route.summarise(routed_all),
            "records": [r.to_dict() for r in records],
            "ladder": [r.to_dict() for r in routed_all],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_work_report(out, records, routed_all)
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


def _write_work_report(out: Path, records, routed_all=()) -> None:
    lines = ["# GAFITAS WORK", ""]
    by_ticket: dict = {}
    for r in records:
        by_ticket.setdefault(r.ticket_id.split("#")[0], []).append(r.outcome)
    if any(len(v) > 1 for v in by_ticket.values()):
        lines += ["## Tasa de exito por ticket", ""]
        for name, outcomes in by_ticket.items():
            ok = sum(1 for o in outcomes if o in ("PASS", "PASS_UNCONFIRMED"))
            lines.append(f"- `{name}`: **{ok}/{len(outcomes)}**  ({', '.join(outcomes)})")
        lines.append("")
    for r in records:
        lines.append(
            f"- `{r.ticket_id}` **{r.outcome}** ({r.model}, {r.loop_outcome}, "
            f"{r.turns_used} turnos, {r.invalid_calls} inv, {r.tool_errors} err)"
        )
        if r.changed_files:
            lines.append(f"  - cambiados: {', '.join(r.changed_files[:8])}")
        for note in r.notes[:4]:
            lines.append(f"  - {note}")
    if routed_all:
        summary = route.summarise(list(routed_all))
        lines += ["", "## Escalado", "",
                  f"- resueltos: **{summary['solved']}/{summary['tickets']}**",
                  f"- resueltos SIN PAGAR NADA: **{summary['solved_free']}**",
                  f"- tickets que tocaron un tier de pago: "
                  f"**{summary['paid_tickets']}** ({summary['paid_share']:.0%})",
                  f"- cambiaron de tier: **{summary['escalated']}**", ""]
        for tier, bucket in summary["by_tier"].items():
            cost = (
                f"{bucket['input_tokens']} tok_in, {bucket['output_tokens']} tok_out"
                if bucket.get("tokens_reported", True)
                else "tokens NO reportados por el proveedor (0 no significa gratis)"
            )
            lines.append(
                f"  - **{tier}**: {bucket['attempts']} intentos, "
                f"{bucket['solved']} resueltos, {bucket['calls']} llamadas, "
                f"{cost}, {bucket['wall_seconds']:.0f}s"
            )
    lines += ["", "## Coste por clase de modelo", ""]
    for cls, bucket in _sum_usage(records).items():
        lines.append(
            f"- **{cls}**: {bucket['tickets']} tickets, {bucket['calls']} llamadas, "
            f"{bucket['input_tokens']} tok_in, {bucket['output_tokens']} tok_out, "
            f"{bucket['wall_seconds']:.1f}s"
        )
    (out / "WORK_REPORT.md").write_text(NL.join(lines) + NL, encoding="utf-8")


def _cmd_retain(args) -> int:
    """Report on preserved workspaces, and prune them only when asked twice.

    Dry run by default: --apply is required before anything is written or
    deleted, because the thing being deleted is evidence.
    """
    base = Path(args.base)
    if args.adopt:
        report = runstate.adopt(base, dry_run=not args.apply)
        verb = "would be marked" if report["dry_run"] else "marked"
        print(f"adopt: {report['adopted']} workspaces {verb}")
        print(f"  already marked: {report['already_marked']}")
        for path in report["sample"]:
            print(f"  e.g. {path}")
        if report["dry_run"]:
            print("  (dry run: pass --apply to write the markers)")
        return 0

    report = runstate.retain(
        base,
        max_age_days=args.max_age_days,
        max_total_bytes=int(args.max_total_mb * 1e6) if args.max_total_mb else None,
        repo=Path(args.repo) if args.repo else None,
        dry_run=not args.apply,
    )
    print(f"scanned {report['scanned']} preserved workspaces, "
          f"{report['total_bytes'] / 1e6:.1f} MB")
    print(f"  protected (pinned or unmarked): {report['protected']}")
    print(f"  selected by policy: {len(report['selected'])} "
          f"({report['selected_bytes'] / 1e6:.1f} MB)")
    for entry in report["selected"][:10]:
        print(f"    {entry['path']}  {entry['bytes'] / 1e6:.1f} MB  {entry['reason']}")
    if report["dry_run"]:
        print("  (dry run: nothing was deleted. Pass --apply to act.)")
    else:
        print(f"  removed: {len(report['removed'])}, failed: {len(report['failed'])}")
        print(f"  log: {report['log']}")
        if report["worktrees_pruned"] is not None:
            print(f"  git worktree prune: {report['worktrees_pruned']}")
    return 0


def _cmd_persist(args) -> int:
    """MULTI_TURN_TOOL_PERSISTENCE: the gate before an engine costs GPU.

    Deliberately its own command and not part of ``certify``. Certification
    asks whether an engine can do the SEVEN THINGS a mission needs; this asks
    the cheaper prior question of whether it can still be asked anything at
    turn twenty. Running the expensive one first was how fifty granite runs got
    spent learning it could not.
    """
    from . import engine as engine_mod
    from . import persistence
    from .provider import OllamaProvider

    out = Path(args.out) if args.out else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        caps = engine_mod.capabilities_for(name, probe_if_missing=True)
        proto = args.protocol or caps.protocol
        provider = OllamaProvider(name, num_ctx=caps.working_context(),
                                  num_predict=caps.working_output(),
                                  timeout=args.timeout)
        declare = tuple(args.declare) if args.declare else None
        report = persistence.measure(provider, engine_name=name,
                                     protocol_name=proto, steps=args.steps,
                                     mode=args.mode, declare=declare)
        path = persistence.report_path(
            out, f"{name}_{proto}_{args.mode}_{report.declared_tools}")
        path.write_text(json.dumps(report.to_dict(), indent=1, ensure_ascii=False),
                        encoding="utf-8")
        rows.append(report)
        print(f"{name} [{proto}/{args.mode}/{report.declared_tools}t]  "
              f"valid={report.valid_tool_call_rate:.0%} "
              f"acceptable={report.acceptable_tool_rate:.0%}  "
              f"no_call={report.no_tool_call} wrong={report.wrong_tool} "
              f"invalid={report.invalid_call}  stall={report.stall}  "
              f"payload_loss={report.payload_loss} objective_loss={report.objective_loss}  "
              f"recovery={report.recovery}  "
              f"CAN_DRIVE={report.can_drive_the_loop}  -> {path.name}")
        print(f"    checkpoints: {report.checkpoints}")
    return 0 if all(r.can_drive_the_loop for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localprog")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selfcheck", help="Print dependency provenance and the tool list")

    p = sub.add_parser("retain", help="Report on, and optionally prune, preserved "
                                      "workspaces. Dry run unless --apply.")
    p.add_argument("--base", default=tempfile.gettempdir(),
                   help="where preserved workspaces live (default: the temp dir)")
    p.add_argument("--max-age-days", type=float, default=None)
    p.add_argument("--max-total-mb", type=float, default=None)
    p.add_argument("--repo", default=None,
                   help="also prune git worktree registrations whose directories are gone")
    p.add_argument("--adopt", action="store_true",
                   help="mark workspaces that predate the marker so the policy can "
                        "see them. Separate on purpose: adopting is what makes "
                        "deletion possible.")
    p.add_argument("--apply", action="store_true",
                   help="actually do it. Without this nothing is written or deleted.")

    p = sub.add_parser("persist", help="MULTI_TURN_TOOL_PERSISTENCE probe: can an "
                                      "engine still drive the loop at turn 20?")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--protocol", default=None, choices=["A", "B", "J", "S"])
    p.add_argument("--mode", default="instructed", choices=["instructed", "goal"],
                   help="instructed names the call; goal names only what is "
                        "needed and charges the engine for deciding")
    p.add_argument("--declare", nargs="*", default=None,
                   help="restrict the DECLARED tool surface (dispatch is unchanged)")
    p.add_argument("--timeout", type=float, default=300.0)

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
    p.add_argument("--tier", choices=("LOCAL", "LUNA", "AUTO"), default="LOCAL",
                   help="LOCAL: Ollama + native tool calls. LUNA: Codex CLI + "
                        "protocol J. AUTO: try LOCAL, escalate to LUNA only when "
                        "LOCAL demonstrably failed.")
    p.add_argument("--local-attempts", type=int, default=1,
                   help="Times to retry the local model before moving up. "
                        "Local inference is free; most local misses are "
                        "sampling rather than a ceiling.")
    p.add_argument("--strong-model", default=None,
                   help="A larger LOCAL model to try when the fast one "
                        "runs out of attempts. Still free.")
    p.add_argument("--strong-attempts", type=int, default=1)
    p.add_argument("--allow-luna", action="store_true",
                   help="Permit the paid LUNA rung in --tier AUTO. Off by "
                        "default: the normal path must not depend on it.")
    p.add_argument("--luna-model", default="gpt-5.6-luna",
                   help="Model for the LUNA rung, when --allow-luna is given.")
    p.add_argument("--effort", default="low", help="LUNA only: Codex reasoning effort.")
    p.add_argument("--local-protocol", choices=("A", "J"), default="A",
                   help="How to drive the LOCAL model. A: native tool calls. "
                        "J: one JSON object per turn, for models whose native "
                        "tool channel does not work.")
    p.add_argument("--num-ctx", type=int, default=work.WORK_NUM_CTX)
    p.add_argument("--num-predict", type=int, default=work.WORK_NUM_PREDICT)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--repeat", type=int, default=1,
                   help="Run each ticket N times. A stochastic system "
                        "measured once is measured badly.")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--run-id", default="GAFITAS_WORK")
    p.add_argument("--telemetry-db", default=None)

    args = parser.parse_args(argv)
    try:
        return {"selfcheck": cmd_selfcheck, "screen": cmd_screen,
                "step1": cmd_step1, "work": cmd_work,
            "retain": _cmd_retain,
            "persist": _cmd_persist}[args.command](args)
    except HarnessInvalid as exc:
        print(f"HARNESS_INVALID: {exc.detail}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
