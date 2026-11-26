# REPORT-IMPL — FA2 int4 perf impl board

## Mission meta

| Field | Value |
|-------|-------|
| Date started | 2026-11-26 |
| Branch | `fa2-int4-perf` @ fork `krisclarkdev/vllm` |
| Audit tip | `ade8aebae6` (PERF-AUDIT CONVERGED R1–R6) |
| Kernels tree | `/home/herdr/work/coder/vllm-xpu-kernels` branch `fa2-int4-xpu` (fork only) |
| Frozen | `fa2-int4-upstream` — no upstream PRs; no prod deploy of this work |
| Hardware | **hal B70 only** |
| Spec decode | **OFF** for all GPU profile / correct / benchy |
| Audit evidence | [`PERF-AUDIT.md`](./PERF-AUDIT.md) |
| Scout diagnose | [`DIAGNOSE-fa2-impl.md`](./DIAGNOSE-fa2-impl.md) (WS2/WS3 call graphs + WS1 pointer); WS1 detail [`DIAGNOSE-WS1.md`](./DIAGNOSE-WS1.md) |

**First-to-implement (from audit):** Rank 1 = FA int4 `paged_decode M1` (long-ctx); Rank 2 = XPU RHT / Hadamard (short-ctx). H13 reshape write is OOS for FA kernel audit but is WS3 product scope.

---

## Package board

| WS | Tree / writer | Scope | Diagnose | Design | Impl | Test | Bench |
|----|---------------|-------|----------|--------|------|------|-------|
| **WS1** | `vllm-xpu-kernels` / coder-ws1 | int4 `paged_decode M1` → adapt fp16 path + int4 dequant front-end | done | pending | pending | pending | pending |
| **WS2** | `vllm` (`fa2-int4-perf`) / coder-ws2 | native XPU Hadamard for Q/out; kill host butterfly + fp32 cast | done | pending | pending | pending | pending |
| **WS3** | `vllm` (separate worktree) / coder-ws3 | fused `reshape_and_cache_int4` native op | done | pending | pending | pending | pending |
| Hadamard-spec | shared design only / arch | one rotation spec both WS2+WS3 implement | — | pending | — | — | — |

Per-WS detail: [`REPORT-WS1.md`](./REPORT-WS1.md) · [`REPORT-WS2.md`](./REPORT-WS2.md) · [`REPORT-WS3.md`](./REPORT-WS3.md)

**Loop per WS:** Confirm baseline → design → implement → correctness (logit + WS3 byte-compat) → benchy → keep or revert (2 attempts max). Signed commits (`-s`).

---

## B70 queue

- **Owner:** tester. Serialize all GPU: profile, correct, benchy.
- Spec decode **OFF** for WS profile / correct / benchy (mission rule).
- **JOB1 interim:** live `muse-glimmer` still has **ngram ON** → baseline labeled **APPROX**; ngram-OFF redeploy deferred until first WS bench enqueue (captain).
- Scale `muse-glimmer` only after agent-move log; restore from `PRODUCTION-BASELINE.yaml`.
- One heavy job at a time on hal. Queue log: `/home/herdr/work/tester/B70-QUEUE.md` (when present).

---

## Before / after llama-benchy

Configs match PERF-AUDIT Round 1 seed: warm once; `--pp 512 2048 8192 --tg 128 --depth 0 8192 32768 --concurrency 1 4 --runs 3`; latency-mode generation; exact-tg; no-cache; skip-coherence; **spec OFF** for keep/revert benches. B70 only.

### JOB1 — baseline confirm (APPROX)

| Field | Value |
|-------|-------|
| Status | **DONE APPROX** (pod IP + svc re-confirm) |
| Target | `http://10.42.0.27:8100/v1` (`muse-glimmer` / `fa2-int4-m1g`) |
| Spec | **ngram ON** → numbers not comparable 1:1 to audit (audit H6: spec confounds tg) |
| Artifacts | `/home/herdr/work/tester/results/baseline-confirm-job1.json` (+ summary); svc re-confirm also DONE APPROX per B70-QUEUE job 1 |
| Captain | ngram-OFF redeploy deferred until first WS bench enqueue; **interim baseline = APPROX** |

| Workload | ctx | batch | pp tok/s | tg tok/s | vs audit | notes |
|----------|-----|-------|----------|----------|----------|-------|
| pp512/tg128 | 0 | 1 | **2442±5** | **31.13±0.25** | audit pp~3235 (Δ−793); tg~16.9 (Δ+14.2) | APPROX (ngram ON) |
| pp512/tg128 | 8192 | 1 | **614** | **12.85±4.08** | audit tg~19 (Δ−6.2); runs 16.93 / 8.77 noisy | APPROX |

### Before (baseline — interim = JOB1 APPROX; strict OFF re-bench later)

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms | notes |
|----------|-----|-------|----------|----------|---------|-------|
| pp512/tg128 | 0 | 1 | 2442±5 | 31.13±0.25 | — | JOB1 APPROX |
| pp512/tg128 | 8192 | 1 | 614 | 12.85±4.08 | — | JOB1 APPROX; noisy |

### After (post-impl — fill per keep decision)

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms | notes |
|----------|-----|-------|----------|----------|---------|-------|
| | | | | | | |

**Audit reference @ pp512/ctx0/c1 (FA2 int4 m1g, Round 1):** tg=16.9 · pp=3235 (see [`PERF-AUDIT.md`](./PERF-AUDIT.md) baseline). fp8 prior: tg≈49.4 · pp≈1907. Use audit for FA-scope attribution; use JOB1 APPROX as interim product e2e until ngram-OFF confirm.

---

## Changelog

| When | What |
|------|------|
| 2026-11-26 | Board + WS stubs created on `fa2-int4-perf` |
| 2026-11-26 | Scout DIAGNOSE linked; WS1/WS2/WS3 Diagnose → done (REPORT-WS* summaries) |
| 2026-11-26 | JOB1 baseline confirm **APPROX** (ngram ON); interim before-table filled; OFF redeploy deferred |
