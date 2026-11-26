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
| H1 | Decode tg gap vs fp8 FA (≥~2× at short ctx) is mostly **inside FA int4 kernel / in-kernel dequant**, not scheduler/sampler | untested (need exclusive XPU FA-kernel timer) | 1 |
| H2 | Prefill q_len>1 is **Triton int4**, not FA2 — FA2 not on prefill critical path | **confirmed (routing)**; prefill loss OOS for FA2 | 1 |
| H3 | Long-context decode slowdown is **KV BW / dequant traffic** scaling with ctx, not fixed per-step overhead | **lean confirmed** (tg falls with ctx; conc=4 collapses at 32k) | 1 |
| H4 | RHT(Q)+inv-RHT(out) around FA decode is a **≥10%** decode tax | untested | 1+ |
| H5 | Concurrency>1 does not recover fp8-like per-stream tg at long ctx (still BW-bound) | **confirmed** at 32k (c=4 tg≈1.0–1.5) | 1 |
| H6 | Spec/ngram decode confounds e2e tg vs pure FA step time | open (metrics show spec counters active) | 1 |

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
| 1 | Prefill is Triton int4, not FA2 — FA2 audit must not chase prefill tok/s as FA kernel win | `fa_int4.py` `run_int4_triton_attention`; `flash_attn.py` q_len>1 branch | code + metrics prefill≈e2e on 1-tok completion | n/a (routing fact); unblocks correct focus on **decode** | — | — |
| 2 | Long-ctx + concurrency: e2e decode throughput collapses (BW-bound product behavior) | serving path / KV traffic (FA int4 decode reads packed pages) | c=4 @32k tg 1.0–1.5 vs c=1 tg 2.3–7.9; ctx0 c=4 scales to 46.6 | if FA/dequant BW improved, headroom largest at long ctx | high | med |

## Refuted

*(none yet as major-area kernel claims — H1 still untested at FA-kernel granularity)*

## First-to-implement pick

*(deferred — Round 1 is baseline. Next measure should falsify H1/H4 with FA-kernel vs surrounding timers; ask captain for exclusive GPU profile window.)*

---

## Round log

### Round 1 — llama-benchy baseline + non-invasive step split

- **Seed (mission):** llama-benchy int4 KV on; decode-heavy + mixed; ctx short / 8k / 32k; batch 1+; FA vs surrounding breakdown
- **Status:** **complete** (measure landed; verdict = baseline)
- **Survey:** Live `muse-glimmer`/`fa2-int4-m1g` on B70; FA int4 decode-only (q_len==1), Triton int4 for prefill; `/metrics` has prefill/decode/ITL histograms; spec/ngram counters active; prior fp8 BEFORE + m1g AFTER tables available
- **Hypotheses tested:** H2 confirmed (routing); H3 lean-confirmed; H5 confirmed at 32k; H1/H4 untested (need kernel profile); H6 open
- **Measure:** Full matrix JSON `exl3-sc/benchy-FA2-PERF-R1.json` (18 cells, 32k no OOM). Step probes `perf/results/r1-step-breakdown.json`. No product-code edits; no exclusive profile
- **Verdict:** **baseline** — e2e table established. Short-ctx decode ~⅓ of fp8 tg; short prefill faster than fp8. FA-kernel share of decode still unknown. No new implementable FA2 major area confirmed at kernel level this round

---

## Changelog

| When | What |
|------|------|
| 2026-11-25 | Initial skeleton on `fa2-int4-perf` (docs worktree) |
| 2026-11-26 | Round 1 baseline: benchy R1 JSON + non-invasive prefill/decode split; H2/H5 confirmed; H1 deferred pending captain profile window |
