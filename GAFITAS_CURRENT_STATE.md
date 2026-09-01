# GAFITAS — CURRENT STATE

Single operational state file. If you come back to this in a week, read only
this. Everything else is detail.

**Canonical root:** `D:\LOCAL-PROGRAMMER-ROOT`, package `localprog`.
**Entry point:** `python -m localprog work --out <DIR> --tier AUTO --model <local> --tickets <t.json>`
**Suite:** 313 tests, green.

---

## WHAT WORKS

| capability | state | evidence |
|---|---|---|
| ticket intake | WORKING | malformed ticket = HarnessInvalid, never a model failure |
| isolated workspace | WORKING | git worktree; source repo provably untouched |
| containment | WORKING | `programmer.guard` + import-shadow detection; escape refused even with a `**` scope, on **both** tiers |
| PRE/POST discrimination | WORKING | a solved ticket never reaches the model at all |
| read / search / enumerate | WORKING | `read_file`, `grep` (with context), `list_symbols`, `list_dir` |
| edit | WORKING | `edit` (unique match) and `replace_lines` (by line number) |
| create files | WORKING | directory and glob write scopes |
| execute + observe | WORKING | `run` — argv array, no shell, allowlist, git read-only, timeout |
| debug loop | WORKING | edit → test → observe → edit, with recovery |
| deliberate ending | WORKING | `finish(DONE\|NO_CHANGE\|BLOCKED)`; DONE requires green tests |
| stall detection | WORKING | 3 dead turns ends the run instead of burning 30 |
| context management | WORKING | per-payload cap, budgeted elision, no runaway |
| refuse-only conscience | WORKING | 5 deterministic signals; provably cannot manufacture a PASS |
| collateral regression | WORKING | whole-suite per-test PRE/POST comparison |
| escalation ladder | WORKING | `--tier AUTO`: LOCAL, escalate to LUNA only on evidence |
| cost accounting | WORKING | per-tier; a tier that reports no tokens says so |
| evidence | WORKING | sealed JSON + turn trace + readable `.patch`, preserved on failure |
| rollback | WORKING | the box is the rollback |

---

## WHAT DOESN'T

| id | what | priority |
|---|---|---|
| — | LUNA reports no token counts (Codex CLI limitation). Cost is calls + wall time only. | P2 |
| — | No commit/handoff step. The patch is produced; applying it is a human decision. | P2 |
| F-09b | Conscience has no PRE/POST *behavioural* probes, only surface + suite. Behavioral Oracle still not wired in. | P2 |
| — | Local success rate is variable on harder classes; same model, same ticket, different outcome. | P2 |
| F-14 | Explorer not exposed as a tool. Never demonstrated as needed. | P3 |

---

## CURRENT MODELS

Hardware: RTX 3080 Ti, **12 GB VRAM**.

| tier | model | notes |
|---|---|---|
| LOCAL | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | 32k ctx, ~7.2 GB with KV cache. Solves the routine classes. |
| LOCAL alt | `qwen3.5:9b` (6.6 GB) | Stronger, slower. Reached 5/6 on the dogfood ticket. |
| LUNA | `gpt-5.6-luna` via Codex CLI, protocol J | Solves what LOCAL cannot. |
| — | `qwen2.5-coder:7b` | **Unusable.** Re-screened against the documented schema after F-02; still emits no tool call at all. The verdict now stands on its own merits. |

---

## CURRENT ROUTING

`--tier AUTO`. Three groups, and the middle one is the whole design:

- **escalate** — `FAIL`, `BLOCKED_BY_CONSCIENCE`: the model tried and missed.
- **keep** — `PASS`, `PASS_UNCONFIRMED`: usable work; a second opinion buys nothing.
- **refuse** — `NON_DISCRIMINATING`, `PROVIDER_ERROR`, `HARNESS_INVALID`: no model can
  fix these, and escalating them turns every corpus defect and crashed server
  into a paid API call while hiding the cause.

---

## CURRENT COST

Eight-ticket sweep, `--tier AUTO`:

| tier | attempts | solved | calls | tokens | wall |
|---|---|---|---|---|---|
| LOCAL | 8 | 3 | 149 | 1,070,891 in / 44,046 out | 446 s |
| LUNA | 5 | 5 | 33 | not reported by the provider | 265 s |

**8/8 solved. 3 without paying anything. Claude wrote none of it.**

---

## CURRENT TASK CLASSES

| class | status | evidence |
|---|---|---|
| A small single-file change | **DEMONSTRATED** | SMOKE-01, LOCAL |
| B multi-file change | **DEMONSTRATED** | mf01, LOCAL, 3 files |
| C create a module from a spec | **DEMONSTRATED** | ga04/06/07/08 |
| D bugfix needing execution | **DEMONSTRATED** | dbg01, LOCAL |
| E modify tests | **DEMONSTRATED** | tests01 — updated the legacy test without deleting assertions |
| F self-dogfood | **DEMONSTRATED** | DOGFOOD-01, committed as `405b192` |
| G recover from a failed attempt | **DEMONSTRATED** | dbg01: test fails → edit → fails → edit → green |
| H search/trace before editing | **DEMONSTRATED** | every run begins list_dir/grep |

---

## CORPUS DISCIPLINE

Every ticket carries `_discrimination_proof`. No ticket is frozen until a build
script has executed PRE and watched it fail — and, where a historical
implementation is the known-correct POST, watched that pass.

The **original STEP1 corpus is void** and must not be reused: all five repos
were snapshotted in their solved state. Its evidence is preserved, not deleted.

---

## NEXT WORK

1. Wire the Behavioral Oracle in as additional conscience signals.
2. A commit/handoff step, so a PASS can become a branch without a human copy.
3. Reduce LOCAL variance, or accept it and let the ladder absorb it.
4. Keep dogfooding: every further GAFITAS change should be a ticket first.
