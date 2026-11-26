# REPORT-WS3: Fused `reshape_and_cache_int4` (C=136)

Date: 2026-11-26  
Tree: `/home/herdr/work/coder/vllm-ws3`  
Branch: `fa2-int4-perf-ws3`  
Status: **ready for B70 queue** (byte-compat test needs XPU/Triton; not run on this host)

## What landed

Single-pass Triton writer on XPU: **RHT(K/V) → asymmetric int4 quant → nibble pack → store**, no separate `single_rht` launch and no host fp32 KV temps.

| Piece | Location |
|-------|----------|
| Unnormalized FWHT + D₁ (DESIGN §1) | `_fwht_unnormalized` / `_rht_forward_vec` in `int4_per_token_head.py` |
| Fused kernel | `_reshape_cache_int4_fused_rht_kernel` |
| XPU dispatch | `reshape_and_cache_int4` → fused when power-of-two equal heads |
| Reference / CUDA | `reshape_and_cache_int4_unfused` (prior `single_rht` + pack) |
| Byte-compat test | `test_reshape_and_cache_int4_fused_byte_compat` |

## Contract

- Signs: seed `0x9E3779B9`, round 0 (same `_get_rht_signs`)
- Forward: `y = H (D₁ ⊙ x)`, fp32 butterfly, **no** per-stage `1/√2`
- After RHT: cast via input dtype (fp16/bf16) then quant — matches unfused round-trip
- Layout: unchanged C=136 / separate K,V uint8 + scale caches (FA views untouched)
- Did **not** edit `fa_int4.py` (WS2) or `vllm-xpu-kernels`

## Verify on B70

```bash
VLLM_TEST_FORCE_LOAD_FORMAT=auto pytest -v -s \
  tests/quantization/test_per_token_kv_cache.py::test_reshape_and_cache_int4_fused_byte_compat
```

Optional microbench: time `reshape_and_cache_int4` vs unfused on decode-shaped updates (audit baseline ~1783 µs/call).

## Non-goals / follow-ups

- WS2 standalone Hadamard for Q/out (prepare/finalize unchanged)
- SYCL rewrite (Triton fused is the preferred path per impl note)
- Push: wait for captain
