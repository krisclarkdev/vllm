# REPORT-WS2 — native XPU Hadamard (Q/out RHT)

Tree: `vllm` on `fa2-int4-perf-ws2` · Writer: coder-ws2  
Parent board: docs `REPORT-IMPL.md` · Design: `DESIGN-fa2-impl.md` · Audit: `PERF-AUDIT.md`

## Diagnosis

PERF-AUDIT H4: `prepare_fa_int4_query` + `finalize_fa_int4_output` ≈1.45 ms/layer ×52 ≈ **75 ms** (~32% of 233 ms short-ctx ITL). XPU fell through to **tier-3 PyTorch butterfly** (Hadacore CUDA-only; Triton Hadamard gated on `x.is_cuda`). Cast alone ~42 µs (H8 refuted as sole cause).

## Design

Follow DESIGN §1 / §3 attempt #1: **standalone dense fp32 Hadamard** (cached Sylvester `H`, one device GEMM) dispatched from `fast_hadamard_transform` / `single_rht` when `device.type == "xpu"` and `d ∈ [16, 256]`. Same algebra as butterfly (`H @ H = d I`); signs seed `0x9E3779B9` round 0; `rht_norm = b`. No FA fuse (stretch). No `reshape_and_cache_int4` edits (WS3). No kernels-tree edits.

## Implement

| File | Change |
|------|--------|
| `vllm/v1/attention/ops/int4_per_token_head.py` | `_dense_fp32_hadamard_transform`; XPU tier in `fast_hadamard_transform`; `single_rht` fp32 accumulate + dtype preserve; `init_rht_signs` warms H on XPU |
| `vllm/v1/attention/backends/fa_int4.py` | prepare/finalize call `single_rht` without host `.float()` butterfly wrapper |
| `tests/v1/attention/test_int4_rht_xpu.py` | dense vs butterfly parity; dtype round-trip; sign seed; prepare/finalize invert |

**Not touched:** `reshape_and_cache_int4`, `vllm-xpu-kernels`.

## Correctness

CPU unit tests (no B70 in this seat):

```text
PYTHONPATH=. /home/herdr/work/coder/.venv-xpu-build/bin/python -m pytest \
  tests/v1/attention/test_int4_rht_xpu.py -v --noconftest
# 9 passed, 2 warnings in 2.71s
```

Gate vs DESIGN: dense vs butterfly `allclose(atol=1e-4, rtol=1e-4)` on fp32; fp16/bf16 `single_rht` matches legacy float-cast path bit-exact after cast-back.

## Bench

Pending tester on B70 (spec OFF): prepare+finalize ≪ ~1.45 ms/layer; short-ctx ITL; no long-ctx regression.

## Verdict

**Keep** attempt #1 (standalone dense XPU Hadamard) pending B70 microbench confirmation. Idle for captain push order.
