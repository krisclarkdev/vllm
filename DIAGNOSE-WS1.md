# DIAGNOSE-WS1 — int4 paged_decode ~1.5 GB/s vs fp16 ~445 GB/s

**Scope:** diagnose only (no fix). Branch `fa2-int4-xpu`.  
**Artifacts:** `FA_INT4.md`, `csrc/xpu/attn/xe_2/paged_decode_int4.hpp`,
`paged_decode_int4_xe2.cpp`, `paged_decode.hpp` (fp16 Cute path),
`int4_kv_dequant.hpp`, plus the fp16 `DecodeFwdMainloop` / flash_api wiring.

**Observed gap:** int4 M1 SYCL decode ≈ **1.5 GB/s** vs fp16 Cute FA2 decode ≈
**445 GB/s** (~300×). The int4 path is correctness-first scaffolding, not a
Cute/DPAS kernel (`FA_INT4.md` L10–12; `paged_decode_int4.hpp` L9–11).

---

## Ranking (binding → secondary)

### 1. No DPAS / tiled MMA — scalar float Q·K and P·V  **[PRIMARY BINDING]**

| Path | Evidence |
|------|----------|
| **int4** | Explicitly “not Cute/DPAS FA2” (`paged_decode_int4.hpp:9–11`). Per-token partial dot is scalar `q_reg[i] * kd` then WG reduce (`:162–175`); V update is scalar `acc[i] * alpha + p_ij * vd` (`:181–189`). |
| **fp16** | `PagedDecodeConfig` defaults MMA to `XE_DPAS_TT<…>` (`paged_decode.hpp:504–507`). Mainloop builds `TiledMMAQK` / `TiledMMAPV` and runs `cute::gemm(mma_qk, …)` / `cute::gemm(mma_pv, …)` (`chunk_prefill_mainloop.hpp:1121–1123`, `:1282`, and PV gemm shortly after). |

**Why it binds bandwidth:** Xe matrix engines + block loads deliver hundreds of
GB/s on fp16 (same stack cites ~440 GB/s at head 256 in `fmha_utils.hpp:219`).
Scalar FMA over 128 dims cannot issue enough concurrent memory work; the kernel
is latency-bound on ALU + reductions, so measured BW collapses to ~1.5 GB/s.

Cute `DecodeFwdMainloop::Int4KV` is scaffold-only (`can_implement=false`)
(`chunk_prefill_mainloop.hpp:880–884`, `:942–946`) — int4 never enters this
DPAS path.

---

### 2. Per-token scalar dequant in the hot loop  **[PRIMARY / co-binding with #1]**

| Path | Evidence |
|------|----------|
| **int4** | Each KV position unpacks scale/zp (`paged_decode_int4.hpp:141–160`) then calls `dequant_byte` for every WI dim on K (`:165–171`) and again on V (`:182–188`). Helpers are pure scalar nibble math (`int4_kv_dequant.hpp:49–61`: mask, cast, `(q-zp)*scale`). |
| **fp16** | No per-element dequant. K/V are already ElementQ; loads go register→`reorder`→DPAS. Fp8 (closest analog) folds **per-tensor** scale out of the inner loop (`chunk_prefill_mainloop.hpp:1246–1258`) and uses subgroup-vectorized cast in `reorder` (`:1277–1280`). |

**Extra waste on int4:** for each logical dim `d`, the kernel dequants **both**
nibbles of `k_row[byte_i]` / `v_row[byte_i]` then discards one
(`paged_decode_int4.hpp:167–170`, `:185–187`). Adjacent dims that share a byte
also re-load/re-dequant the same packed byte. Scale unpack uses `memcpy`
bitcasts every token (`int4_kv_dequant.hpp:38–46`).

This is not “gather→fp16 materialize” (header L10 claims in-loop dequant), but
it is still **scalar dequant × (K+V) × kv_len × dims_per_WI**, sitting on the
critical path before any useful FMA.

---

### 3. Token-serial gather vs Xe 2D block-load + prefetch  **[HIGH]**

| Path | Evidence |
|------|----------|
| **int4** | Loop is `for (pos = 0; pos < kv_len; ++pos)` (`paged_decode_int4.hpp:128`). Every position: `block_table` lookup (`:135–139`), then scalar pointer math to one K row / V row / scale (`:141–157`), then `k_row[byte_i]` scalar loads (`:169`). No prefetch API, no 2D block copy. |
| **fp16** | Outer loop is **KV-tile** blocked (`for (K = blk_k0; K < blk_k1; K++)`, `chunk_prefill_mainloop.hpp:1262`). Page translate once per tile (`:1165–1174`), then `make_block_2d_copy_*` + `prefetch` + `copy(copy_k/v, …)` (`:1116–1119`, `:1150–1154`, `:1235–1237`, `:1274–1275`). Policy for block=64 uses `ShapeQK = (q_packed, 64, 64)` — **64 tokens per tile** (`fmha_utils.hpp:315–320`). |

**Effect:** int4 pays a gather + pointer chase **per token**; fp16 amortizes
page-table indirection over a 64-token DPAS tile and keeps the LSC busy with
2D block loads. Irregular scalar gathers destroy coalescing and hide latency
poorly → BW stays tiny even though packed int4 touches fewer unique bytes.

---

### 4. No split-KV (forced `num_kv_splits = 1`)  **[HIGH for long-seq occupancy]**

| Path | Evidence |
|------|----------|
| **int4** | `flash_api.cpp:488–491` forces `num_kv_splits = is_int4_kv ? 1 : get_num_splits(...)`. Host rejects splits (`paged_decode_xe2.cpp:53–57`). Launch grid is only `batch * num_heads_q` WGs (`paged_decode_int4.hpp:207–216`). FA notes: “splits=1” (`FA_INT4.md:12`). |
| **fp16** | Tuner `num_splits(...)` can raise splits up to 32 to fill Xe cores when `batch * Hkv` under-occupies (`paged_decode_tuning.h:87–136`). Kernel is `XeFMHAFwdSplitKVKernel` + `ReduceSplitK` (`paged_decode.hpp:577–586`). |

**Effect:** for long `kv_len` / small batch, fp16 parallelizes across the
sequence with split-KV; int4 serializes the entire sequence inside one WG.
This alone can leave most EUs idle and multiplies wall time; it does **not**
explain the full ~300× by itself but caps any micro-optimization of the
scalar kernel.

---

### 5. Subgroup / WG shape: 1×SG scalar vs multi-SG DPAS tile  **[MEDIUM–HIGH]**

| Path | Evidence |
|------|----------|
| **int4** | `kWgSize = 32` — “one Xe subgroup; each WI owns 4 dims” (`paged_decode_int4.hpp:28–31`). Score needs `sycl::reduce_over_group` **every token** (`:173–175`) — full WG sync/reduction on the critical path. |
| **fp16** | For `kv_tile=_64`, `SubgroupLayoutQK = Layout<Shape<_1,_4,_1>>` → **4 subgroups / WG** (`fmha_utils.hpp:320`). Block shape `SGPerWG * sg_size` (`paged_decode_kernel.hpp:221–223`). Softmax is fragment-local with `exp2` (`chunk_prefill_mainloop.hpp` softmax helper ~`:714–737`), not a per-token WG reduce. |

**Effect:** int4 uses 1/4 the threads per WG vs the block-64 fp16 policy, no
cross-SG Q packing over GQA in-WG, and inserts a WG reduction per KV position.
Occupancy and latency both suffer.

---

### 6. SLM — not the primary bandwidth binder  **[LOW / design gap]**

| Path | Evidence |
|------|----------|
| **int4** | No `local_accessor` / SharedStorage; everything in private registers + global gathers. |
| **fp16 decode mainloop** | `DecodeFwdMainloop::SharedStorage` is **empty** (`chunk_prefill_mainloop.hpp:911–912`). SLM appears in **split reduce** (`paged_decode_kernel.hpp:614–616`, LSE SLM ~`:818+`) and is discussed for large-head epilogue budgets (`fmha_utils.hpp:214–222`). |

**Verdict:** missing SLM staging of dequantized K/V tiles is a real follow-on
design gap (could amortize dequant and feed a future DPAS tile), but fp16
decode is already near ~445 GB/s **without** mainloop SLM — by using DPAS +
2D loads + split-KV. Calling “no SLM” the root cause would mis-rank; fix order
is DPAS/tile path first, then optional SLM dequant buffers.

---

## Side-by-side (hd=128, block=64 decode)

| Mechanism | int4 M1 (`paged_decode_int4.hpp`) | fp16 Cute (`DecodeFwdMainloop`) |
|-----------|-----------------------------------|----------------------------------|
| Compute | Scalar float FMA | `XE_DPAS_TT` via `cute::gemm` |
| KV traversal | `pos = 0 .. kv_len-1` | Tiles of 64 KV tokens |
| Loads | Scalar `uint8` + scale gathers | Xe 2D block copy + prefetch |
| Dequant | In-loop scalar nibble (`dequant_byte`) | N/A (native fp16); fp8 = vector cast + folded scale |
| Split-KV | Forced 1 | Tuned up to 32 |
| WG | 32 threads (1 SG) | 4 SG (layout `_1,_4,_1`) for tile 64 |
| Grid | `B × Hq` | `B × H × splits × …` via tile scheduler |
| Status | Correctness scaffold | Production FA2 path |

---

## Causal summary

The ~1.5 GB/s number is what you expect from a **single-subgroup, token-serial,
scalar-dequant, scalar-dot** kernel with **no split-KV** and **no 2D block
loads**. The fp16 ~445 GB/s number is what you expect from **DPAS + tiled 2D
loads + occupancy via split-KV** on the same Xe2 stack.

**Binding order to close the gap (design only — do not implement here):**

1. Leave the scalar SYCL path; enter Cute `DecodeFwdMainloop` (or equivalent)
   with DPAS after a **vectorized / tile-wise** int4→ElementQ convert (today
   `Int4KV` `can_implement=false`).
2. Replace per-token scalar gathers with **page/tile 2D loads** of packed
   uint8 (+ scale rows), dequant into MMA fragments once per tile.
3. Re-enable **split-KV** (+ reduce) so long sequences fill the device.
4. Match fp16 **multi-SG** decode policy for block=64.
5. Consider **SLM** only as a dequant/staging optimization after (1)–(3).

---

## Files cited

- `FA_INT4.md`
- `csrc/xpu/attn/int4_kv_dequant.hpp`
- `csrc/xpu/attn/xe_2/paged_decode_int4.hpp`
- `csrc/xpu/attn/xe_2/paged_decode_int4_xe2.cpp`
- `csrc/xpu/attn/xe_2/paged_decode_xe2.cpp`
- `csrc/xpu/attn/xe_2/paged_decode.hpp`
- `csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp`
- `csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp`
- `csrc/xpu/attn/fmha_utils.hpp`
- `csrc/xpu/attn/paged_decode_tuning.h`
- `csrc/flash_attn/flash_api.cpp`
