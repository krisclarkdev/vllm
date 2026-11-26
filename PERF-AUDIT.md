# PERF-AUDIT — FA2 int4 KV (find-and-measure)

## Mission meta

| Field | Value |
|-------|-------|
| Date started | 2026-11-25 |
| Branch | `fa2-int4-perf` (report-only commits) |
| Audited tip | `f5e5b04386` (matches `origin/fa2-int4-upstream`) |
| Serving | live `muse-glimmer` = `fa2-int4-m1g` @ `http://10.42.0.25:8100/v1` |
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
| H1 | Decode tg gap vs fp8 FA is mostly **inside FA int4 kernel / in-kernel dequant** | **confirmed (refined)** — int4 FA ~70× slower than fp16 FA at ctx=8k on same B70; ~1.5 GB/s vs ~445 GB/s implied; dominates long-ctx. Short-ctx dominated by H4 not H1. | 3 |
| H2 | Prefill q_len>1 is Triton int4, not FA2 | **confirmed** | 1 |
| H3 | Long-context decode slowdown is KV BW / dequant traffic scaling with ctx | **confirmed (refined)** — scales with ctx via FA int4 kernel time, but **not** near hardware BW (refuted as HBM-limit); implementation-bound | 1+3 |
| H4 | RHT(Q)+inv-RHT(out) around FA decode is a **≥10%** decode tax | **confirmed** — exclusive XPU: ~1.45 ms/layer ×52 ≈ 75 ms/step; ~32% of probe ITL 233 ms; at ctx=256 RHT ≫ FA (~71 vs ~10 ms×52) | 2 |
| H5 | Concurrency>1 does not recover fp8-like per-stream tg at long ctx | **confirmed** | 1 |
| H6 | Spec/ngram confounds e2e tg vs pure FA step time | **confirmed (metrics confounder)** — not a FA kernel loss; explains benchy tg=16.9 (≈59 ms) vs probe ITL 233 ms / RHT 75 ms paradox | 2 |
| H7 | FA int4 requires block_size%64==0 (`FA_INT4_BLOCK_SIZE=64`) — layout constraint | noted (fact); not separately timed as ≥10% loss | 3 |
| H8 | fp32 cast in prepare/finalize is itself ≥10% | **refuted** — cast round-trip ~42 µs vs Hadamard ~650–720 µs (decomp) | 2 |
| H9 | num_splits tuning recovers ≥10% on FA int4 decode | **refuted** — ctx=8k: splits 0/1 both ~5.31 ms; splits>1 NotImplemented on FA2 | 4 |
| H10 | FA int4 kernel is near hardware BW/XMX limit (no headroom) | **refuted (headroom)** — same-card fp16 FA reaches ~445 GB/s at 8k while int4 stuck ~1.58 GB/s | 3 |
| H11 | Lazy `from vllm…` imports inside `FlashAttentionImpl.forward` / `do_kv_cache_update` cost ≥10% decode (scout: imports-in-forward) | **untested** | seed |
| H12 | `ensure_int4_scale_caches` every forward (views/as_strided) costs ≥10% decode (scout) | **untested** — R2 showed RHT dwarfs cast; cache carve not timed | seed |
| H13 | Triton `reshape_and_cache` int4 write every decode step is ≥10% (scout) | **untested** / likely **OOS** unless it forces extra K/V passes into FA2 | seed |

---

## Baseline

Source: `/home/herdr/work/exl3-sc/benchy-FA2-PERF-R1.json` (llama-benchy 0.3.8, 2026-11-26 08:06:18Z).  
Config: warm once; `--pp 512 2048 8192 --tg 128 --depth 0 8192 32768 --concurrency 1 4 --runs 3 --latency-mode generation --exact-tg --no-cache --skip-coherence`.  
Generation-latency calibration: **347.46 ms**. Tokenizer: `/tank/nas/models/turboderp-muse-glimmer-30b-sc-3bpw`. **32k depth: no OOM.**

### FA2 int4 (this round) — means

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms | notes |
|----------|-----|-------|----------|----------|---------|-------|
| pp512/tg128 | 0 | 1 | 3235 | 16.9 | 506 | tg std 9.0 (noisy) |
| pp2048/tg128 | 0 | 1 | 1048 | 7.8 | 2301 | |
| pp8192/tg128 | 0 | 1 | 633 | 5.2 | 13287 | |
| pp512/tg128 | 8192 | 1 | 615 | 19.0 | 14496 | |
| pp2048/tg128 | 8192 | 1 | 576 | 4.7 | 18130 | |
| pp8192/tg128 | 8192 | 1 | 463 | 3.4 | 35773 | |
| pp512/tg128 | 32768 | 1 | 308 | 7.2 | 108455 | 32k OK |
| pp2048/tg128 | 32768 | 1 | 299 | 2.3 | 116863 | |
| pp8192/tg128 | 32768 | 1 | 267 | 7.9 | 153602 | tg non-monotonic vs pp2048 |
| pp512/tg128 | 0 | 4 | 1112 | 46.6 | 1510 | aggregate tg scales |
| pp2048/tg128 | 0 | 4 | 928 | 13.9 | 7196 | |
| pp8192/tg128 | 0 | 4 | 617 | 7.4 | 39859 | |
| pp512/tg128 | 8192 | 4 | 602 | 7.3 | 42809 | |
| pp2048/tg128 | 8192 | 4 | 567 | 6.0 | 51950 | |
| pp8192/tg128 | 8192 | 4 | 462 | 4.7 | 115227 | |
| pp512/tg128 | 32768 | 4 | 308 | 1.5 | 293819 | conc fails to help |
| pp2048/tg128 | 32768 | 4 | 299 | 1.3 | 308697 | |
| pp8192/tg128 | 32768 | 4 | 268 | 1.0 | 398043 | |

### Reference fp8 FA (prior B70, `benchy-BEFORE.json`, same tool family)

| Workload | ctx | batch | pp tok/s | tg tok/s | ttfr ms |
|----------|-----|-------|----------|----------|---------|
| pp512/tg128 | 0 | 1 | 1907 | 49.4 | 382 |
| pp2048/tg128 | 0 | 1 | 2002 | 30.1 | 1137 |
| pp8192/tg128 | 0 | 1 | 1893 | 37.8 | 4442 |
| pp512/tg128 | 16384 | 1 | 1496 | 32.6 | 12756 |

**vs fp8 @ pp512/ctx0/c1:** FA2 int4 tg **16.9 vs 49.4 (~0.34×)**; pp **3235 vs 1907 (~1.7× faster short prefill)**.

### Step breakdown (one decode + one prefill)

Non-invasive only (prod stayed up; **no exclusive XPU profile**). Probes after matrix: `/home/herdr/work/perf/results/r1-step-breakdown.json`. Prometheus: `request_prefill_time_seconds`, `request_decode_time_seconds`, `inter_token_latency_seconds`, TTFT.

| Step | FA kernel | surrounding / e2e | evidence | notes |
|------|-----------|-------------------|----------|-------|
| prefill | **n/a (not FA2)** | `request_prefill_time` 4.669 s for ~3606 prompt tok (~772 tok/s); TTFT=e2e | metrics Δ + code `run_int4_triton_attention` when `max_seqlen_q>1` | **H2 confirmed** — prefill loss is Triton/OOS for FA2 kernel audit |
| decode | **unknown** (no kernel timer without scale-down profile) | short-prompt probe: prefill 0.41 s + decode 7.22 s for 32 tok → mean ITL **233 ms/tok** (~4.3 tok/s); benchy c1 short-ctx tg **16.9** (high variance) | metrics Δ + benchy tg/ttfr | Full forward+sampler+ngram in decode bucket; **cannot attribute ≥10% to FA kernel yet** → ask captain before exclusive XPU profile |

---

## Confirmed findings (ranked)

| Rank | Mechanism | File / fn | Evidence | Gain range | Effort | Risk |
|------|-----------|-----------|----------|------------|--------|------|
| 1 | **FA int4 paged decode kernel (M1) far below fp16 FA / HBM** — in-kernel int4 path ~70× slower at 8k; ~1.5 GB/s implied | `vllm_xpu_kernels` `flash_attn_varlen_func` / `int4 paged_decode M1`; dispatch `flash_attn.py` | R3 exclusive microbench int4 vs fp16 — `/home/herdr/work/perf/results/r3-fa-kernel.json`, `/home/herdr/work/perf/results/r3-fa-bw.json` | **Long-ctx: 2–10×+ decode** if int4 FA reaches even ~10–20% of fp16 FA efficiency; short-ctx smaller | high (kernel) | high (numerics/layout) |
| 2 | **XPU RHT uses tier-3 butterfly + fp32 cast around every FA decode layer** | `fa_int4.py` `prepare_fa_int4_query`/`finalize_fa_int4_output`; `int4_per_token_head.fast_hadamard_transform` | R2 exclusive ~75 ms/52 layers (~32% of 233 ms ITL); ≫ FA at ctx=256 — `/home/herdr/work/perf/results/r2-rht-microbench-xpu-exclusive.json`, `/home/herdr/work/perf/results/r2-rht-decomp.json` | **Short-ctx: ~15–30% step** if native XPU Hadamard / fuse into FA; long-ctx dilutes to <10% of attn time | med | med (must match RHT basis) |
| 3 | Prefill is Triton not FA2 (routing) | `fa_int4.run_int4_triton_attention` | R1 | n/a for FA kernel | — | — |
| 4 | Long-ctx + conc collapse (product of #1) | serving / KV traffic | R1 benchy | follows #1 | — | — |

## Refuted

| ID | Killed by |
|----|-----------|
| H8 | cast 42 µs vs Hadamard ~700 µs (`/home/herdr/work/perf/results/r2-rht-decomp.json`) |
| H9 | splits 0/1 identical; >1 unsupported (`/home/herdr/work/perf/results/r4-num-splits-out.txt`) |
| H10 | fp16 FA ~445 GB/s vs int4 ~1.58 GB/s same card/ctx (`/home/herdr/work/perf/results/r3-fa-bw.json`) |

## First-to-implement pick

**Rank 1 — optimize / replace FA int4 paged_decode M1** (or fuse dequant into a BW-capable FA). Long-ctx and concurrency pain are this kernel. RHT (rank 2) is the right first fix only if the product priority is **short-ctx decode TPS**; for 8k–32k Muse capacity wins, fix the FA int4 kernel first.

---

## Round log

### Round 1 — llama-benchy baseline + non-invasive step split

- **Seed (mission):** llama-benchy int4 KV on; decode-heavy + mixed; ctx short / 8k / 32k; batch 1+; FA vs surrounding breakdown
- **Status:** **complete** (measure landed; verdict = baseline)
- **Survey:** Live `muse-glimmer`/`fa2-int4-m1g` on B70; FA int4 decode-only (q_len==1), Triton int4 for prefill; `/metrics` has prefill/decode/ITL histograms; spec/ngram counters active; prior fp8 BEFORE + m1g AFTER tables available
- **Hypotheses tested:** H2 confirmed (routing); H3 lean-confirmed; H5 confirmed at 32k; H1/H4 untested (need kernel profile); H6 open
- **Measure:** Full matrix JSON `exl3-sc/benchy-FA2-PERF-R1.json` (18 cells, 32k no OOM). Step probes `perf/results/r1-step-breakdown.json`. No product-code edits; no exclusive profile
- **Verdict:** **baseline** — e2e table established. Short-ctx decode ~⅓ of fp8 tg; short prefill faster than fp8. FA-kernel share of decode still unknown. No new implementable FA2 major area confirmed at kernel level this round

### Round 2 — H4 RHT tax (exclusive B70)

- **Survey:** `prepare_fa_int4_query` / `finalize_fa_int4_output` wrap every FA decode layer (`flash_attn.py` ~1254–1290). Both call `single_rht`; on XPU `fast_hadamard_transform` falls through to **tier-3 PyTorch butterfly** (Hadacore CUDA-only; Triton Hadamard gated on `x.is_cuda`).
- **Hypothesis (H4):** RHT(Q)+inv-RHT(out) costs ≥10% of decode step because XPU uses host-launched butterfly + fp32 cast, ×52 layers.
- **Measure (exclusive):** Scaled `muse-glimmer`→0; pod `fa2-perf-probe` same image `fa2-int4-m1g`. Scripts: `r2_rht_microbench_v2.py`, `r2_rht_decomp.py`. Artifacts: `/home/herdr/work/perf/results/r2-rht-microbench-xpu-exclusive.json`, `/home/herdr/work/perf/results/r2-rht-decomp.json`.
  - tok=1 prepare+finalize: **~1446 µs/layer** (event); stack52 **~75.5 ms**
  - vs probe ITL 233 ms → **~32%**; vs noisy benchy 59 ms → >100% (reconcile via H6)
  - decomp: prepare-like 767 µs; had_fp32 719 µs; cast 42 µs; torch.compile can cut prepare-like to 276 µs (still ~14 ms×52)
- **Agent moves:** none (no Glimmer-tier fleet seats). Log: `AGENT-MOVE-LOG.md`.
- **Verdict:** **H4 confirmed** major area. **H8 refuted**. **H6 confirmed** as e2e confounder (not a kernel fix target).

### Round 3 — H1 FA int4 kernel vs fp16 / BW

- **Survey:** Decode uses `flash_attn_varlen_func` with int4 pages + `k_descale`/`v_descale` (ESIMD page-attn requires head_size==256 — Muse D=128 skips it). Kernel enforces `block_size % 64 == 0` (`FA_INT4_BLOCK_SIZE=64`); error text names **`int4 paged_decode M1`**.
- **Hypothesis (H1):** Gap vs fp8/fp16 is mostly in-kernel int4 dequant / FA int4 decode implementation.
- **Measure (exclusive):** `r3_fa_kernel_microbench.py`, `r3_fa_bw_check.py`. Artifacts: `/home/herdr/work/perf/results/r3-fa-kernel.json`, `/home/herdr/work/perf/results/r3-fa-bw.json`. nkv=8, block=64, C=136.

| ctx | ms int4 FA/layer | ms fp16 FA/layer | int4/fp16 | int4 GB/s | fp16 GB/s | RHT ms×52 | FA int4 ms×52 |
|-----|------------------|------------------|-----------|-----------|-----------|-----------|---------------|
| 256 | 0.17–0.19 | 0.059 | ~2.9× | ~1.5 | ~18 | ~71 | ~10 |
| 2048 | 1.33–1.47 | 0.037 | ~36× | ~1.57 | ~228 | ~79 | ~76 |
| 8192 | 5.30–5.87 | 0.075 | **~70×** | **~1.58** | **~445** | ~79 | ~305 |
| 32768 | 37.2 | — | — | ~0.90 | — | ~67 | ~1933 |

- **Verdict:** **H1 confirmed** — FA int4 decode kernel is the long-ctx rock; **not** HBM-limited (**H10 refuted headroom**). **H3 refined**. Short-ctx: H4 > H1.

### Round 4 — H9 num_splits

- **Hypothesis:** num_splits choice leaves ≥10% on the table for FA int4 decode.
- **Measure:** ctx=8192 int4 FA; num_splits 0→5.308 ms, 1→5.309 ms; >1 raises `NotImplementedError('FA2 does not support num_splits > 1')`. Artifact: `/home/herdr/work/perf/results/r4-num-splits-out.txt`.
- **Verdict:** **H9 refuted**. No new major area.

---

## Changelog

| When | What |
|------|------|
| 2026-11-25 | Initial skeleton on `fa2-int4-perf` (docs worktree) |
| 2026-11-26 | Round 1 baseline: benchy R1 JSON + non-invasive prefill/decode split; H2/H5 confirmed; H1 deferred pending captain profile window |
| 2026-11-26 | Round 2: exclusive RHT microbench — H4 confirmed (~32% of ITL); H8 refuted; H6 confirmed confounder |
| 2026-11-26 | Round 3: exclusive FA int4 vs fp16 + BW — H1 confirmed (~70× at 8k); H10 refuted headroom; H3 refined; first-to-implement = FA int4 kernel |
| 2026-11-26 | Round 4: num_splits sweep — H9 refuted (0/1 identical; >1 unsupported on FA2) |
| 2026-11-26 | Seed H11–H13 from scout (imports-in-forward; ensure_int4_scale_caches; reshape write) as untested |
