# GAFITAS — CURRENT STATE

Single operational state file. Updated every round. If you are coming back to
this after a week, read only this file.

**Canonical root:** `D:\LOCAL-PROGRAMMER-ROOT` — package `localprog`.
Everything else is either a borrowed authority or evidence.

**Last updated:** round 1 complete, round 2 in progress.

---

## WHAT WORKS

| capability | state | evidence |
|---|---|---|
| ticket intake | WORKING | `work.load_ticket`, malformed ticket = HarnessInvalid, never a model failure |
| isolated workspace | WORKING | git worktree, copy fallback, source repo provably untouched (`test_the_source_repository_is_never_modified`) |
| containment | WORKING | `programmer.guard` on every path + import-shadow detection; escape attempts refused even with a `**` write scope |
| **PRE/POST discrimination** | **WORKING** | model is not invoked at all on a solved ticket (`test_a_solved_ticket_is_refused_before_the_model_runs`) |
| read / search / enumerate | WORKING | `read_file`, `grep`, `list_symbols`, `list_dir` |
| edit | WORKING | unique-match, syntax-checked, atomic, multi-edit safe |
| create files | WORKING | directory and glob write scopes, not an exact-path list |
| execute + observe | WORKING | `run` — argv array, no shell, interpreter allowlist, git read-only, timeout, bounded output |
| deliberate ending | WORKING | `finish(status=DONE\|NO_CHANGE\|BLOCKED)` |
| refuse-only conscience | WORKING | 6 signals; proven unable to manufacture a PASS |
| collateral regression | WORKING | whole-suite PRE/POST per-test comparison |
| cost accounting | WORKING | per-call tokens, split by model class |
| evidence | WORKING | sealed JSON + turn trace + readable `.patch` |
| rollback | WORKING | the box is the rollback; preserved on failure |

**Test suite: 169 green** (`python -m pytest -q` in the canonical root).

---

## WHAT DOESN'T (open blockers)

| id | what | priority |
|---|---|---|
| F-13 | No model routing. LOCAL only; no LUNA, no CLAUDE escalation. | P1 |
| F-09b | Conscience has no PRE/POST *behavioural* probes, only surface + suite. | P2 |
| F-12 | Duplicate runners still committed in `PROGRAMMER-BUILD-ROOT`. | P2 |
| — | No commit/handoff step; the patch is produced but not applied anywhere. | P2 |
| F-14 | Explorer not exposed as an optional tool. Not yet demonstrated as needed. | P3 |

---

## CURRENT MODELS

Hardware: RTX 3080 Ti, **12 GB VRAM**.

| role | model | why |
|---|---|---|
| local worker | `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) | passed the frozen screen 5/5, and is the first model to complete a real ticket end to end |
| local alternate | `qwen3.5:4b` (3.4 GB) | also PASS_NOISY on the screen |
| local larger | `qwen2.5-coder:7b` (4.7 GB) | screen said INCOMPATIBLE — **that verdict is void**, it was measured against an undocumented tool schema (F-02). Needs re-screening. |
| escalation | `gpt-5.6-luna` via Codex CLI | adapter exists but is orphaned in `GATE-A-WS`; not wired in |

---

## CURRENT ROUTING

None. Every ticket goes to one named local model. Routing is deliberately not
built until there is a functioning programmer to route *for*.

---

## CURRENT COST

Measured per ticket, split by model class (`WORK_RESULTS.json` → `usage_total`).

First real ticket (SMOKE-01, single-file fix, `qwen3:4b`):
**6 turns · 15,688 input tokens · 668 output tokens · 17.3 s · 0 external spend.**

---

## CURRENT TASK CLASSES

| class | status | evidence |
|---|---|---|
| A small single-file change | **DEMONSTRATED** | SMOKE-01 PASS, conscience clean, trace `list_dir→read_file→grep→read_file→edit→finish` |
| H search/trace before editing | **DEMONSTRATED** | same run — it grepped before editing |
| C create a module from a spec | in progress | ga04/ga06/ga07/ga08, PRE-FAIL proven |
| B multi-file change | probe ready | `mf01`, PRE 3 failed / 1 passed |
| D bugfix needing execution | probe ready | `dbg01`, PRE 4 failed / 1 passed |
| E modify tests | not started | |
| F self-dogfood | not started | |
| G recover from a failed attempt | not started | measured opportunistically |

---

## CORPUS DISCIPLINE

Every ticket carries `_discrimination_proof`. No ticket is frozen until a build
script has personally executed PRE and watched it fail — and, where a historical
implementation is used as the known-correct POST, watched that pass.

The old STEP1 corpus is **void** and must not be reused: all five repos were
snapshotted in their solved state. Its evidence is preserved, not deleted.

---

## NEXT WORK

1. Finish class C (ga0x) and classes B/D.
2. Re-screen `qwen2.5-coder:7b` against the documented schema — its
   INCOMPATIBLE verdict was an artefact of F-02.
3. Self-dogfood: give GAFITAS a real ticket against its own repository.
4. Routing, once there is evidence about which classes need escalation.
