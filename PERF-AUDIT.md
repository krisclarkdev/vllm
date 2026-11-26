# PERF-AUDIT — FA2 int4 KV (find-and-measure)

Date started: 2026-11-25  
Branch: `fa2-int4-perf` (report-only; tip of audited code tracks `fa2-int4-upstream`)  
Audited tip: `f5e5b04386` (`Merge remote-tracking branch 'upstream/main' into fa2-int4-upstream`)  
Serving target: live `muse-glimmer` = `fa2-int4-m1g` on **hal B70 only**  
Hard rules: no product-code edits; **`fa2-int4-upstream` frozen** (PR #70037); fork-only (`krisclarkdev/vllm`)

## Scope

**In:** FlashAttention-2 + direct XPU dispatch — int4 KV decode inside FA2, in-kernel dequant, invocation (shapes / pages / splits) where loss is in or at the FA2 boundary.

**Out:** Triton prefill internals; reshape_and_cache except bad layout into FA2; sampler; scheduler; non-attention. If loss is outside FA2 → one-line OOS evidence, move on.

## Method

Each round: SURVEY → HYPOTHESIZE (falsifiable) → MEASURE (B70) → VERDICT (`confirmed` / `refuted` / `inconclusive`).  
Converge when **two consecutive** rounds: no new confirmed major area **and** no surviving untested hypothesis.  
Major-area bar: ≥~10% on a realistic workload, or blocks a larger win (FA2 fallback, sync, layout forcing extra K/V passes).

---

## Baseline (Round 1 — pending)

| Workload | ctx | batch | metric | value | notes |
|----------|-----|-------|--------|-------|-------|
| *(awaiting llama-benchy on B70)* | | | | | |

### Step breakdown (one decode + one prefill)

| Step | FA kernel | surrounding | evidence | notes |
|------|-----------|-------------|----------|-------|
| decode | *pending* | *pending* | | |
| prefill | *pending* | *pending* | | |

---

## Hypothesis backlog

| ID | Hypothesis | Status | Round |
|----|------------|--------|-------|
| H0 | Round 1 seed: int4 KV on; decode-heavy + mixed; ctx short / 8k / 32k / longest card holds; batch 1+; kernel breakdown FA vs surrounding | untested | 1 |

---

## Confirmed findings (ranked)

*(none yet)*

| Rank | Mechanism | File / fn | Evidence | Gain range | Effort | Risk |
|------|-----------|-----------|----------|------------|--------|------|

## Refuted

*(none yet)*

## First-to-implement pick

*(deferred until ≥1 confirmed major area)*

---

## Rounds

### Round 1 — llama-benchy baseline + FA vs surrounding breakdown

- **Status:** in progress (perf measuring on B70 against live `muse-glimmer`; docs awaiting numbers to land)
- **Survey:** —
- **Hypotheses tested:** H0 (seed)
- **Measure:** —
- **Verdict:** —

---

## Changelog

| When | What |
|------|------|
| 2026-11-25 | Scaffold PERF-AUDIT.md on `fa2-int4-perf`; Round 1 pending perf/scout numbers |
