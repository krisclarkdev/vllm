# REPORT-WS1 — FA int4 paged_decode M1

Tree: `vllm-xpu-kernels` (`fa2-int4-xpu`) · Writer: coder-ws1  
Parent board: [`REPORT-IMPL.md`](./REPORT-IMPL.md) · Audit: [`PERF-AUDIT.md`](./PERF-AUDIT.md)

## Diagnosis

**Status:** complete (diagnose-only; no fix). Full write-up: [`DIAGNOSE-WS1.md`](./DIAGNOSE-WS1.md).

**Gap:** int4 M1 SYCL decode ≈ **1.5 GB/s** vs fp16 Cute FA2 ≈ **445 GB/s** (~300×). int4 path is correctness-first scaffolding, not Cute/DPAS (`paged_decode_int4.hpp`).

**Binding order (summary):**

1. **PRIMARY — no DPAS / tiled MMA:** scalar float Q·K and P·V; Cute `DecodeFwdMainloop::Int4KV` is scaffold-only (`can_implement=false`). fp16 uses `XE_DPAS_TT` + `cute::gemm`.
2. **PRIMARY / co-binding — per-token scalar dequant** in the hot loop (`dequant_byte` × K+V × kv_len); wasted dual-nibble work + `memcpy` scale bitcasts.
3. **HIGH — token-serial gather** vs fp16 Xe 2D block-load + prefetch over 64-token tiles.
4. **HIGH (long-seq) — split-KV forced to 1** (`num_kv_splits=1`); fp16 tunes up to 32.
5. **MEDIUM–HIGH — WG shape:** int4 = 1 SG / 32 WI + per-token WG reduce; fp16 block-64 = 4 SG, fragment-local softmax.
6. **LOW — missing SLM** is a follow-on design gap, not the root BW binder (fp16 hits ~445 GB/s without mainloop SLM).

**Causal one-liner:** ~1.5 GB/s is expected from a single-subgroup, token-serial, scalar-dequant, scalar-dot kernel with no split-KV and no 2D block loads.

**Design-only fix order (do not implement in diagnose):** (1) enter Cute/DPAS after vectorized tile-wise int4→ElementQ convert → (2) page/tile 2D loads + dequant once per tile → (3) re-enable split-KV → (4) match multi-SG policy → (5) optional SLM staging after (1)–(3).

## Design

**Path:** [`DESIGN-fa2-impl.md`](./DESIGN-fa2-impl.md) (shared impl design; Hadamard-spec shared with WS2/WS3).  
WS1 kernel design follow-ons tracked there / in kernels tree — *(pending WS1-specific design fill)*.

## Implement

*(pending)*

## Correctness

*(pending)*

## Bench

*(pending)*

## Verdict

*(pending — keep or revert; 2 attempts max)*
