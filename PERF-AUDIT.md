# PERF-AUDIT — FA2 int4 KV (find-and-measure)

## Mission meta

| Field | Value |
|-------|-------|
| Date started | 2026-11-25 |
| Branch | `fa2-int4-perf` (report-only commits) |
| Audited tip | `f5e5b04386` (matches `origin/fa2-int4-upstream`) |
| Serving | live `muse-glimmer` = `fa2-int4-m1g` |
| Hardware | **hal B70 only** — numbers from elsewhere are void |
| Frozen | `fa2-int4-upstream` (PR #70037) — never commit/push/rebase |
| Fork | `krisclarkdev/vllm` only — no upstream PRs/pushes |

**Scope (FA2 ONLY):** Flash attention kernels + direct XPU dispatch — int4 KV decode inside FA2, in-kernel dequant, invocation (shapes/pages/splits) where loss is in/at FA2 boundary.

**Out of scope:** Triton prefill internals; reshape_and_cache except bad layout into FA2; sampler; scheduler; non-attention. If loss is outside FA2 → one-line OOS evidence, move on.

**Loop:** SURVEY → HYPOTHESIZE (falsifiable) → MEASURE (B70) → VERDICT (`confirmed` / `refuted` / `inconclusive`). Converge when two consecutive rounds: no new confirmed major area AND no surviving untested hypothesis. Major-area bar: ≥~10% on realistic workload OR blocks a larger win.

**Deliverable sections (fill as rounds land):** baseline table + step breakdown; ranked confirmed findings; refuted list; first-to-implement pick.

---

## Hypothesis backlog

| ID | Hypothesis | Status | Round |
|----|------------|--------|-------|

*(empty — filled from survey/hypothesize each round)*

---

## Baseline

| Workload | ctx | batch | metric | value | notes |
|----------|-----|-------|--------|-------|-------|

### Step breakdown (one decode + one prefill)

| Step | FA kernel | surrounding | evidence | notes |
|------|-----------|-------------|----------|-------|
| decode | | | | |
| prefill | | | | |

---

## Confirmed findings (ranked)

| Rank | Mechanism | File / fn | Evidence | Gain range | Effort | Risk |
|------|-----------|-----------|----------|------------|--------|------|

## Refuted

*(none yet)*

## First-to-implement pick

*(deferred until ≥1 confirmed major area)*

---

## Round log

### Round 1 — *(placeholder)*

- **Seed (mission):** llama-benchy int4 KV on; decode-heavy + mixed; ctx short / 8k / 32k / longest card holds; batch 1+; kernel-level breakdown one decode + one prefill (FA kernel vs surrounding)
- **Status:** awaiting numbers from perf
- **Survey:** —
- **Hypotheses tested:** —
- **Measure:** —
- **Verdict:** —

---

## Changelog

| When | What |
|------|------|
| 2026-11-25 | Initial skeleton on `fa2-int4-perf` (docs worktree) |
