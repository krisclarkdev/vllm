# DIAGNOSE — FA2 IMPL (WS2/WS3 focus)

Date: 2026-11-26 · scout read-only

## Trees

| Role | Path | Tip |
|------|------|-----|
| kernels (WS1) | `/home/herdr/work/scout/vllm-xpu-kernels` @ `fa2-int4-xpu` | `8eb946b` |
| vllm (WS2/WS3) | `/home/herdr/work/scout/vllm` @ `origin/fa2-int4-perf` | `6f8ebe598a` |
| fallback kernels | `/home/herdr/work/scout/inbox-ws1/` | same sources |

Kernel files present: `csrc/xpu/attn/xe_2/paged_decode_int4.hpp`, `csrc/xpu/attn/int4_kv_dequant.hpp`, `csrc/xpu/attn/xe_2/paged_decode.hpp`, `csrc/flash_attn/flash_api.cpp`, `DIAGNOSE-WS1.md`.

## WS1 (pointer only)

Reuse `vllm-xpu-kernels/DIAGNOSE-WS1.md`. Binding: **no DPAS / scalar Q·K·P·V** + **scalar dequant** (`paged_decode_int4.hpp`, `int4_kv_dequant.hpp`) vs fp16 DPAS (`paged_decode.hpp`) / forced splits=1 (`flash_api.cpp`).

---

## WS2/WS3 — symbols (`6f8ebe598a`)

| Symbol | Location |
|--------|----------|
| `reshape_and_cache_int4` | `vllm/v1/attention/ops/int4_per_token_head.py:880` |
| `single_rht` | `:1179` → `fast_hadamard_transform` |
| `fast_hadamard_transform` | `:1099` — Hadacore CUDA; Triton if `x.is_cuda` (`:1131–1136`); **XPU → tier-3 butterfly** (`:1138–1146`) |

## Decode write + RHT (WS3 / reshape)

```
FlashAttentionImpl.do_kv_cache_update          flash_attn.py:1620
  → triton_reshape_and_cache_flash_per_token_head_quant
       triton_reshape_and_cache_flash.py:286  (call :1651)
    → reshape_and_cache_int4                   :311 → int4_per_token_head.py:880
      → single_rht(key.float())                :891
      → single_rht(value.float())              :892
        → fast_hadamard_transform              (XPU = butterfly)
      → _run_reshape_kernel(_reshape_cache_int4_kernel)  :893–903  pack+write
```

## Decode FA RHT / Hadamard (WS2)

```
FlashAttentionImpl.forward                     flash_attn.py:1304
  → prepare_fa_int4_query                      fa_int4.py:37  (call :1583)
      → single_rht(Q.float())                  :51
  → flash_attn_varlen_func                     (int4 pages → kernels M1)
  → finalize_fa_int4_output                    fa_int4.py:56  (call :1613)
      → single_rht(out, inverse=True)          :64
```

XPU: no Hadacore / no `x.is_cuda` → every RHT is fp32 cast + host-launched butterfly (Q/out on FA path; K+V on write path).
