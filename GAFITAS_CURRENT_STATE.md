# GAFITAS — CURRENT STATE

Single operational state file. If you come back to this in a week, read only
this.

**Canonical root:** `D:\LOCAL-PROGRAMMER-ROOT`, package `localprog`.
**Suite:** 604 passed, 12 xfailed (recorded UNSUPPORTED boundaries — see below).

```bash
python -m localprog work --out <DIR> --tier LOCAL --model qwen3:4b-instruct-2507-q4_K_M --local-attempts 3 --tickets <t.json>
```

**No paid model is reachable from that command line.** `--allow-luna` is off by
default and must be typed on purpose.

---

## PAID DEPENDENCY: NONE FOR SUPPORTED CLASSES

| measurement | result | paid calls |
|---|---|---|
| Final battery — 5 **new** discriminating tickets | **5/5** | **0** |
| Benchmark corpus, free ladder | 7/8 | **0** |
| Local self-maintenance (DOGFOOD-04, -05, **-07**) | **3/3** | **0** |

## RETRIEVAL (2026-09-03)

The agent could not search by meaning, only by literal regex, and it was the
dominant loss in everything that needed to find code. `search_code` — a BM25
index over declaration regions, language-agnostic, stdlib, 0.3 s to build —
plus navigation memory and honest tool feedback.

| RepoQA GAFITAS_AGENT, official evaluator | before | after |
|---|---|---|
| python (development set) | 19.00% | **47.00%** |
| **typescript (holdout, frozen before any change)** | 17.00% | **51.00%** |

`p = 1.4e-07` on the holdout, paired. Cheaper as well: 19.0M → 11.1M input
tokens, 48.6 → 34.6 min. Ablation: removing only `search_code` drops python
from 44% to 22%, so the organ is the cause and not the surrounding fixes.

Full account: `D:\GATE-A-WS\GAFITAS_EXTERNAL_BENCHMARKS\REPOQA\RETRIEVAL_RECOVERY_REPORT.md`

---

## SUPPORTED TASK CLASSES — local only

| class | evidence |
|---|---|
| small single-file fix | fin01 PASS, 7 turns · SMOKE-01 3/3 |
| bounded multi-file | fin02 PASS (3 files coordinated) · mf01 3/3 |
| create a module from a spec | fin03 PASS, 10 turns · ga08, ga06 |
| debugging (two distinct faults) | fin04 PASS · dbg01 |
| modify tests | fin05 PASS · tests01 |
| search / trace before editing | `search_code`: RepoQA agent 19% → 47% python, 17% → 51% typescript holdout |
| finding a symbol nobody named | DOGFOOD-07: the objective says "la herramienta que lista un directorio", the agent finds `tools.py::list_dir` among twelve |
| first attempt fails → recovers | fin02, fin05, DOGFOOD-04/05 all passed on attempt 2–3 |
| **self-maintenance** | **DOGFOOD-04 and DOGFOOD-05, written by the 4B, committed** |

## UNSUPPORTED — written down, not hidden

| class | evidence |
|---|---|
| subtle multi-site logic in a very large module | DOGFOOD-03: 13 free attempts, every one 4–5 of 11 |
| a 16-requirement module from one written spec | ga04: 5 free attempts, always 13–14 of 16 |
| a feature whose condition has two halves | DOGFOOD-06: 3 attempts here and 3 on the pre-change code, every one 2 of 3 — it adds the note unconditionally or not at all |

These stay `UNSUPPORTED`. That is the honest answer, and it is better than a
silent dependency on somebody's API.

---

## LOCAL STACK

Hardware: RTX 3080 Ti, 12 GB. Idle after a batch: **47 °C, 19 W, fan off, 1.2 GB**.

| tier | model | notes |
|---|---|---|
| LOCAL_FAST | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | 32k ctx. Does the volume. Wrote both self-maintenance commits. |
| LOCAL_STRONG | `qwen3.5:9b` (6.6 GB) | Optional rung. Took ga06 after three 4B misses. Still free. |
| — | `qwen2.5-coder:7b` | Unusable: no native tool call at all, re-screened after F-02. |
| LUNA | `gpt-5.6-luna` via Codex, protocol J | **Off by default.** Teacher and control during development. |

**13 tools:** `read_file · list_dir · grep(context) · search_code · list_symbols ·
read_symbol · edit · replace_lines · write_file · copy_code · run · run_tests ·
finish`

Not all thirteen are declared on every mission. `legal_tools` derives the
declared surface from the mission's write scope: a mission that may only CREATE
a file is not offered `edit`, because no argument could make it succeed. An
ordinary ticket with a real write scope sees all thirteen. Looking and finishing
are never withheld, and the frozen screen pins its own seven regardless.

---

## ROUTING

`--tier LOCAL` with `--local-attempts N`, optionally `--strong-model`.

- **escalate** — `FAIL`, `BLOCKED_BY_CONSCIENCE`: it tried and missed.
- **keep** — `PASS`, `PASS_UNCONFIRMED`: usable work.
- **retry free** — `PROVIDER_ERROR`, `HARNESS_INVALID`: the attempt never
  happened. Never a reason to climb into a paid tier.
- **abandon** — `NON_DISCRIMINATING`: the ticket was not a task.

---

## SAFETY

**133 sealed runs · 0 scope violations · 0 harness failures · 0 source repos modified.**

---

## WHAT DOESN'T

| what | priority |
|---|---|
| DOGFOOD-03 / ga04 classes above the local tier (F-55) | P2 |
| Behavioral Oracle still a separate authority, not wired in as extra signals | P2 |
| No commit/handoff step — the patch is produced, promotion stays human | P2 |
| LUNA reports no token counts (Codex CLI limit); the report says so rather than printing zeros | P2 |
| Explorer never demonstrated as needed, so never integrated | P3 |

---

## NEXT WORK

1. Wire the Behavioral Oracle in as extra conscience signals.
2. Attack the F-55 boundary: the failures are coordinated multi-site edits, so
   the next lever is probably a plan-then-edit affordance, not a bigger model.
3. Keep dogfooding. Every further GAFITAS change should be a ticket first, and
   the local tier should write as many of them as it can.
