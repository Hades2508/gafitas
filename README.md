# GAFITAS

A local coding-agent harness, built to answer one question honestly: **can a
small model running on one desktop produce real patches?**

The interesting part is not the agent. It is what had to be built around the
agent before its results could be believed.

```
Explorer → agent loop → guard → sealed evidence → two independent scorers → audit
```

- **A deterministic tool loop.** Fifteen tools, an explicit turn budget, a
  transcript window with a stated elision policy, and deterministic repair of
  malformed calls. Zero LLM calls anywhere in the judging path.
- **A write-scope guard.** Every write is checked against the mission's scope
  before it lands, in its own disposable git worktree. Across the 168 recorded
  runs of this campaign, 464 writes: **zero unauthorised writes.**
- **Per-turn sealed evidence.** Every call, its arguments, its verdict, its
  token counts and its latency, written down as it happens. Every number in
  this repository can be recomputed from that record months later — and several
  have been, which is how some of them turned out to be wrong.
- **Two independent scorers.** A run is scored in its workspace *and* by
  replaying its sealed writes into a freshly rebuilt frozen tree. When the two
  disagree there is no verdict, only a divergence to be explained.
- **1023 tests.**

## What it measured

| | result |
|---|---|
| A capable model **inside** the harness vs **bare** | **4/4 vs 0/4**, then **5/5 vs 0/5** on fresh tasks |
| A local 4B model on the factory's own task bank | 3 correct patches, causally attributed |
| A local 3B model, six cohorts | **0 of 24** |
| A local 4B on **SWE-bench Verified**, official evaluator | **1 of 3**, with 0 empty and 0 errored |

That last row is a smoke test, not a rate. The three instances are the three
easiest in a single repository, chosen to prove the pipeline end to end against
a gold-patch positive control; published SWE-bench figures are over all 500.
What it does show is that all three predictions were patches the official
harness could apply and judge, produced against code the agent had never seen
and with no acceptance test to check itself against.

The harness is not the obstacle: plug a stronger brain into the same body and
the same tools, the same scorer and the same budget produce correct patches
where a weaker one produces none. What remains open is whether a small local
model can be brought to that behaviour, and the honest current answer is
*sometimes, unreliably*.

## The rules that make those numbers mean something

- **Attribution.** A feature gets credit only when it was available, applicable,
  offered, called, accepted, **and causally in the path** — established by
  replaying the run with that one call removed and confirming the suite stops
  passing. The best funnel figures this project ever produced were refused three
  times under this rule, because the mechanism they were credited to had not
  fired before the result appeared.
- **Freeze before you measure.** Code, corpus and acceptance are frozen and
  hashed before the first result is seen. Freezing is a moment, not a state:
  the manifest is written immediately before launch, because one written
  earlier described a cohort that never ran.
- **Failure is data.** A failed run is preserved, not retried into a better
  number. Contaminated runs are marked and kept, never deleted.
- **Infrastructure failure is not model failure.** A dead provider, a missing
  dependency, an unusable test suite — each is its own verdict. Conflating them
  with "the model was wrong" is the single most common way to publish a
  confident falsehood, and this project has done it and had to correct it.

## What went wrong, and why it is in the README

The defects worth reading about are the ones that produced *plausible* numbers:

- A verification route silently blind to two of the six tools that can change a
  file. It rebuilt trees without those changes, reported a disagreement, and
  invalidated a cohort — by manufacturing the disagreement out of its own
  blindness. It also hid a real success for two cohorts.
- A metric that counted a harness bookkeeping file as the model's patch. One
  prediction was *entirely* that file: an agent that changed nothing produced a
  result that looked real.
- Four proxy measurements that read the wrong field and produced a confident
  number before anyone checked what the field meant. `ok` on a test-run event
  means the tool ran, not that the tests passed.
- A treatment that, at the moment it was supposed to help, would have suggested
  editing the acceptance test that judges the run. Caught by sizing it against
  sealed evidence before spending a single GPU-hour on it.

Each is recorded with its cause and its blast radius. A measurement apparatus
that cannot describe its own failures is not an apparatus, it is a claim.

## Layout

```
localprog/     the harness: tool surface, loop, guard, evidence, scoring
tests/         1023 tests
INVARIANTS.md  the rules above, in their binding form
```

## Related

Three RL/eval environments built from this work, published on the Prime
Intellect Environments Hub, with their validation harness:
[gafitas-environments](https://github.com/Hades2508/gafitas-environments).

## Licence

MIT.
