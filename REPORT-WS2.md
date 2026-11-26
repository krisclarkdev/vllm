# REPORT-WS2 — native XPU Hadamard (Q/out RHT)

Tree: `vllm` on `fa2-int4-perf` · Writer: coder-ws2  
Parent board: [`REPORT-IMPL.md`](./REPORT-IMPL.md) · Audit: [`PERF-AUDIT.md`](./PERF-AUDIT.md)

## Diagnosis

**Status:** complete (scout diagnose). Full write-up: [`DIAGNOSE-fa2-impl.md`](./DIAGNOSE-fa2-impl.md) § “Decode FA RHT / Hadamard (WS2)”.

**Call graph (decode FA path):**

```
FlashAttentionImpl.forward                     flash_attn.py:1304
  → prepare_fa_int4_query                      fa_int4.py:37  (call :1583)
      → single_rht(Q.float())                  fa_int4.py:51
  → flash_attn_varlen_func                     (int4 pages → kernels M1)
  → finalize_fa_int4_output                    fa_int4.py:56  (call :1613)
      → single_rht(out, inverse=True)          fa_int4.py:64
```

**XPU Hadamard:** `single_rht` → `fast_hadamard_transform` (`int4_per_token_head.py:1179` → `:1099`). Hadacore is CUDA-only; Triton path gated on `x.is_cuda` (`:1131–1136`). **XPU falls to tier-3 butterfly** (`:1138–1146`) — fp32 cast + host-launched — on every Q and inverse-out RHT (×52 layers). Matches PERF-AUDIT H4 (~75 ms/step / ~32% of 233 ms ITL).

## Design

**Path:** `DESIGN-fa2-impl.md` (Hadamard-spec shared with WS3 / arch) — leave untracked until captain asks to add. Full design summary *(pending)*.

## Implement

*(pending)*

## Correctness

*(pending)*

## Bench

*(pending)*

## Verdict

*(pending — keep or revert; 2 attempts max)*
