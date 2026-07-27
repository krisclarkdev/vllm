# XPU graphs feature — 3-arm deep-metrics benchmark

Feature branch: `xpu-graphs-feature-validation`
Plan: `../../docs/xpu/` (validation write-up lands as
`XPU_GRAPHS_FEATURE_VALIDATION.md` with the results).

Measures the consolidated XPU-graphs feature (01 PIECEWISE + 02 oneAPI 2026
image + 03 FA-in-graph + 04 Ornith canary, all on fork `main`) in three arms
with a full metrics sweep. The harness is **endpoint-pointable**: arms G and E
are whatever servers the 05 (production cutover) session stands up — this
harness only benches them, it does not duplicate their serving or their
correctness gates.

| Arm | What | Who serves it |
|-----|------|---------------|
| G | fork `main` image, graphs ON (FA-in-graph, `FULL_AND_PIECEWISE`) | session 05 |
| E | same image, graphs OFF (`VLLM_XPU_ENABLE_XPU_GRAPH=0 --enforce-eager`) | session 05 |
| O | original pre-graphs image (`577e1a932`, oneAPI 2025.3 / torch 2.12) | `serve_arm_o.sh` here |

**Profile-consistency rule:** all three arms must be benched at the same serve
profile (model, context, kv dtype, `--max-num-seqs`, util). Bench arms G/E at
whatever profile 05 serves; start arm O with the same values via env.

**Coordination rules (single Arc GPU on hal):** only one session benches or
docker-builds on hal at a time; arm O runs in a window when 05's server is
down; never prune images without checking tags; never leave the node serving
anything but 05's intended state.

## Files

| File | Purpose |
|------|---------|
| `bench_sweep.py` | stdlib-only sweep: concurrency x length grid, streaming TTFT/TPOT/ITL/E2E percentiles, prefill+decode token throughput |
| `flops.py` | derives FLOPs/token from model `config.json` (MoE/hybrid aware), reports prefill/decode TFLOPS + MFU |
| `telemetry.sh` | `xpu-smi dump` sidecar (start/stop/summarize) — utilization, power, frequency, VRAM, temperature, Joules/token inputs |
| `run_arm.sh` | one arm end-to-end: health check, /metrics snapshots, telemetry, sweep, TFLOPS merge -> `bench_<ARM>_<STAMP>.json` |
| `serve_arm_o.sh` | arm O only: original image + production-equivalent `is_padding` runtime patch, serve + sanity smoke, trap-kill on failure |
| `patch_topk_is_padding.py` | replica of the production ConfigMap runtime patch (removes the stale XPU `is_padding`-omitting branches) |
| `gen_compare.py` | merges `bench_*_<STAMP>.json` into `GRAPHS_FEATURE_COMPARE_<STAMP>.{json,md}` with G-vs-E / G-vs-O / E-vs-O deltas |

## Run (on hal)

```bash
# Bench a server 05 already has up (example: graphs arm on port 8004):
ARM=G BASE_URL=http://127.0.0.1:8004 MODEL=ornith \
  MODEL_CONFIG=/models/Ornith-1.0-35B-MXFP4/config.json \
  bash deploy/xpu-graphs-bench/run_arm.sh

# Arm O (GPU must be free; uses the rebuilt original image):
ARM_O_IMAGE=hal/vllm-xpu:kris-fork-577e1a932-rebuild \
  MODEL_HOST=/models/Ornith-1.0-35B-MXFP4 \
  bash deploy/xpu-graphs-bench/serve_arm_o.sh
ARM=O BASE_URL=http://127.0.0.1:8021 MODEL=ornith-arm-o \
  MODEL_CONFIG=/models/Ornith-1.0-35B-MXFP4/config.json \
  bash deploy/xpu-graphs-bench/run_arm.sh
docker rm -f ornith-arm-o   # tear down as soon as the sweep ends

# Report over whatever arms completed (same STAMP):
python3 deploy/xpu-graphs-bench/gen_compare.py results <STAMP>
```

Sweep shape (override via env): `CONCURRENCIES="1 2 4 8"`,
`LENGTHS="128:128 2048:256 8192:256"` (input:output tokens; decode-heavy,
mixed, prefill-heavy). Greedy, `ignore_eos` for stable token counts; unique
per-request prompt prefixes defeat prefix caching so arms with and without it
are comparable.

TFLOPS: `flops.py` computes active-params FLOPs/token from the model config
(MoE top-k experts + shared expert + full-attention quadratic term for
prefill; linear-attention/GDN layers counted from their state-space params —
approximation documented in the script, override with `ACTIVE_PARAMS_B`).
MFU needs `PEAK_BF16_TFLOPS` for the GPU (Arc Pro B70) in the env.

Results land under `results/` (`bench_*`, `telemetry_*`, `metrics_*`,
`GRAPHS_FEATURE_COMPARE_*`; serve logs gitignored).
