# REPORT-WS3 — fused reshape_and_cache_int4 native op

Tree: `vllm` (separate worktree) · Writer: coder-ws3  
Parent board: [`REPORT-IMPL.md`](./REPORT-IMPL.md) · Audit: [`PERF-AUDIT.md`](./PERF-AUDIT.md)

## Diagnosis

**Status:** complete (scout diagnose). Full write-up: [`DIAGNOSE-fa2-impl.md`](./DIAGNOSE-fa2-impl.md) § “Decode write + RHT (WS3 / reshape)”.

**Call graph (KV write path):**

```
FlashAttentionImpl.do_kv_cache_update          flash_attn.py:1620
  → triton_reshape_and_cache_flash_per_token_head_quant
       triton_reshape_and_cache_flash.py:286  (call flash_attn.py:1651)
    → reshape_and_cache_int4                   :311 → int4_per_token_head.py:880
      → single_rht(key.float())                :891
      → single_rht(value.float())              :892
        → fast_hadamard_transform              (XPU = butterfly)
      → _run_reshape_kernel(_reshape_cache_int4_kernel)  :893–903  pack+write
```

**XPU tax:** same tier-3 butterfly as WS2 on **K and V** every decode write (`fast_hadamard_transform` `:1138–1146`). PERF-AUDIT H13 timed reshape_and_cache_int4 ≈ 1783 µs/call (×52 ≈ 39.8% of 233 ms ITL) including RHT(K)+RHT(V)+pack — OOS for FA-kernel audit, **in scope for WS3**.

## Design

**Path:** `DESIGN-fa2-impl.md` (Hadamard-spec shared with WS2 / arch) — leave untracked until captain asks to add. Full design summary *(pending)*.

## Implement

*(pending)*

## Correctness

*(pending — logit + **byte-compat** vs current Triton `reshape_and_cache_int4` write)*

## Bench

*(pending)*

## Verdict

*(pending — keep or revert; 2 attempts max)*
