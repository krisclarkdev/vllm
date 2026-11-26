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

**First-to-implement (from audit):** Rank 1 = FA int4 `paged_decode M1` (long-ctx); Rank 2 = XPU RHT / Hadamard (short-ctx). H13 reshape write is OOS for FA kernel audit but is WS3 product scope.

---

## Package board

| WS | Tree / writer | Scope | Diagnose | Design | Impl | Test | Bench |
|----|---------------|-------|----------|--------|------|------|-------|
| **WS1** | `vllm-xpu-kernels` / coder-ws1 | int4 `paged_decode M1` → adapt fp16 path + int4 dequant front-end | pending | pending | pending | pending | pending |
| **WS2** | `vllm` (`fa2-int4-perf`) / coder-ws2 | native XPU Hadamard for Q/out; kill host butterfly + fp32 cast | pending | pending | pending | pending | pending |
| **WS3** | `vllm` (separate worktree) / coder-ws3 | fused `reshape_and_cache_int4` native op | pending | pending | pending | pending | pending |
| Hadamard-spec | shared design only / arch | one rotation spec both WS2+WS3 implement | — | pending | — | — | — |

Per-WS detail: [`REPORT-WS1.md`](./REPORT-WS1.md) · [`REPORT-WS2.md`](./REPORT-WS2.md) · [`REPORT-WS3.md`](./REPORT-WS3.md)

**Loop per WS:** Confirm baseline → design → implement → correctness (logit + WS3 byte-compat) → benchy → keep or revert (2 attempts max). Signed commits (`-s`).

---

## B70 queue

- **Owner:** tester. Serialize all GPU: profile, correct, benchy.
- Spec decode **OFF**.
- Scale `muse-glimmer` only after agent-move log; restore from `PRODUCTION-BASELINE.yaml`.
- One heavy job at a time on hal.

---

## Before / after llama-benchy

Configs match PERF-AUDIT Round 1 seed: warm once; `--pp 512 2048 8192 --tg 128 --depth 0 8192 32768 --concurrency 1 4 --runs 3`; latency-mode generation; exact-tg; no-cache; skip-coherence; **spec OFF**. B70 only.

### Before (baseline — fill from audit / re-bench)

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms | notes |
|----------|-----|-------|----------|----------|---------|-------|
| | | | | | | |

### After (post-impl — fill per keep decision)

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms | notes |
|----------|-----|-------|----------|----------|---------|-------|
| | | | | | | |

**Audit reference @ pp512/ctx0/c1 (FA2 int4 m1g):** tg=16.9 · pp=3235 (see [`PERF-AUDIT.md`](./PERF-AUDIT.md) baseline). fp8 prior: tg≈49.4 · pp≈1907.

---

## Changelog

| When | What |
|------|------|
| 2026-11-26 | Board + WS stubs created on `fa2-int4-perf` |
