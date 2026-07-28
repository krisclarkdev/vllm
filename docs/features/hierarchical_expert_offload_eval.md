# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eval / bakeoff notes for hierarchical expert staging (PR-A metrics spine).

Per AGENTS.md, model-affecting changes must report eval commands and results.
PR-A is **metrics-only**: no claimed tok/s regression or improvement yet.

## Unit tests

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_hierarchical_offload.py -v
```

## Bakeoff harness (hal)

Default model on **hal**: Mixtral-8x7B Instruct AWQ
(`/tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ`). Ornith uses the same
NAS path when present. Pause the k8s DaemonSet before Arc-only runs.

Force in-process engine so `get_tier_manager().stats` is non-empty in the
driver (default in the harness via `VLLM_ENABLE_V1_MULTIPROCESSING=0`):

```bash
# Hierarchical (warm W, then measure)
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --warm 4 --max-tokens 32 \
  --num-prompts 4 --prompt Hi --output /tmp/hier_pr_a_bakeoff.json

# Baseline in a separate invocation
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --baseline --warm 4 --max-tokens 32 \
  --output /tmp/hier_pr_a_baseline.json

# Optional comparison JSON
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py --compare \
  --hierarchical-json /tmp/hier_pr_a_bakeoff.json \
  --baseline-json /tmp/hier_pr_a_baseline.json \
  --output /tmp/hier_pr_a_compare.json
```

Default `--prompt Hi` keeps unique experts within a 4-slot budget; longer prompts may raise `cannot allocate a slot`. Use `--tier-num-slots 4` (of 8 experts) so staging is exercised; raise to 8
for full residency (less interesting for hit-rate metrics). Optional:
`--colibri-tok-s <float>`, `--measure-prefill`.

## Bakeoff JSON schema (hierarchical / baseline)

```json
{
  "mode": "hierarchical",
  "model": "/tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ",
  "elapsed_s": 0.0,
  "total_tokens": 0,
  "tok_s_warm": 0.0,
  "ttft_proxy_s": 0.0,
  "prefill_proxy_s": null,
  "warm_steps": 4,
  "num_prompts": 4,
  "max_tokens": 32,
  "tier_stats": {
    "device_hits": 0,
    "device_misses": 0,
    "ram_hits": 0,
    "ram_misses": 0,
    "disk_hits": 0,
    "disk_misses": 0,
    "h2d_bytes": 0,
    "h2d_stall_ns": 0,
    "h2d_stall_ms": 0.0,
    "disk_bytes": 0,
    "disk_wait_ns": 0,
    "disk_wait_ms": 0.0,
    "unique_experts_sum": 0,
    "unique_experts_hist": {},
    "ensure_calls": 0,
    "device_hit_rate": 0.0,
    "ram_hit_rate": 0.0,
    "disk_hit_rate": 0.0
  },
  "tier_stats_note": null,
  "colibri_tok_s": null,
  "speedup_vs_colibri": null,
  "config": {
    "offload_backend": "hierarchical",
    "tier_num_slots": 4,
    "tier_ram_gb": 8.0,
    "tier_disk_path": null,
    "tier_pilot": false,
    "max_model_len": 1024,
    "gpu_memory_utilization": 0.85,
    "v1_multiprocessing": "0"
  },
  "note": "PR-A metrics spine; no claimed tok/s win yet"
}
```

`disk_*` stay 0 when no `--tier-disk-path` is configured. Compare mode adds
`tok_s_warm_hierarchical`, `tok_s_warm_baseline`, `speedup_vs_baseline`.

## PR-B residency bakeoff (hal)

```bash
# Full residency (slots=E=8) — should approach baseline tok/s
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 8 --tier-ram-gb 8 --warm 8 --max-tokens 64 \
  --max-num-batched-tokens 64 --max-num-seqs 1 \
  --output /tmp/hier_pr_b_residency.json

# Forced staging (slots=4)
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --warm 8 --max-tokens 64 \
  --prompt Hi --max-num-batched-tokens 1 --max-num-seqs 1 \
  --output /tmp/hier_pr_b_slots4.json
```

Acceptance (aspirational): with `slots=E`, warm hierarchical decode within
~10% of baseline tok/s on the same hardware.

### Hal / Arc Pro B70 results (2026-07-28)

Model: Mixtral-8x7B-Instruct-v0.1-AWQ @ ~32 GiB XPU. Image tip
`3deb3160c` (PR-B). Unit tests: 13/13.

| Setup | tok_s_warm | device hit rate | Notes |
|-------|------------|-----------------|-------|
| baseline (no offload) | ~40.66 | n/a | `/tmp/hier_pr_b_baseline.json` |
| hierarchical slots=8 | — | — | **XPU OOM** during profile / `XpuFusedMoe` init |
| hierarchical slots=7..5 | — | — | OOM (slot sweep) |
| hierarchical slots=4 | ~0.72 | ~0.994 | `/tmp/hier_pr_b_slots4.json`; staging thrash |

Full-residency tok/s parity is **not** claimed on this GPU/model. Treat
**slots=4** as the accepted bakeoff / PR-C comparison baseline (same config
with vs without async+pilot). Revisit slots=`E` on a smaller model, lower
profile util, or after memory wins.

## PR-C async / PILOT / O_DIRECT (hal)

```bash
# Control (PR-B path): slots=4, no pilot
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --warm 8 --max-tokens 64 \
  --prompt Hi --max-num-batched-tokens 1 --max-num-seqs 1 \
  --output /tmp/hier_pr_c_nopilot.json

# Treatment: schedule/wait + PILOT
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --tier-pilot --warm 8 --max-tokens 64 \
  --prompt Hi --max-num-batched-tokens 1 --max-num-seqs 1 \
  --output /tmp/hier_pr_c_pilot.json
```

Acceptance: warm decode shows lower `h2d_stall_ns` and/or higher
`device_hit_rate` vs the no-pilot control (same slots).

### Hal results (2026-07-28)

Affinity `--tier-pilot` (no `--tier-pilot-real`) on slots=4 **regressed**
vs control: extra ensure/DMA thrash (ensure 4536 vs 2304; h2d_bytes ~18×).
Pilot predict hit rate ~0.64 was not enough to offset critical-path prefetch
cost. Prefer gate-real pilot and/or RAM-only / free-slot prefetch before
claiming a win.

| Setup | tok_s_warm | h2d_stall_ms | device hit rate | Notes |
|-------|------------|--------------|-----------------|-------|
| slots=4 no pilot | ~1.09 | ~104 | ~0.990 | `/tmp/hier_pr_c_nopilot.json` |
| slots=4 + pilot | ~0.51 | ~114 | ~0.828 | `/tmp/hier_pr_c_pilot.json`; predict hit ~0.64 |

## PR-D dual NVMe + NUMA

```bash
vllm serve <moe> --offload-backend hierarchical \
  --tier-disk-path /nvme0/expert_store \
  --tier-disk-mirror /nvme1/expert_store \
  --tier-disk-weights 1,1 \
  --tier-numa
```

Acceptance: dual-path reads show non-zero `disk_bytes_primary` and
`disk_bytes_mirror` (or `MIRROR:` log); single-disk path unchanged when mirror
is unset. Unit tests cover hash stability, partial mirror, and mirror fallback.

## PR-E device pipeline + graphs

Audit (Mixtral hierarchical + XPU MoE):

- Activations stay on-device through ensure→GEMM.
- `wait_ensure` joins weight H2D events only.
- Unavoidable host sync: expert-id `unique→tolist` (`host_expert_id_syncs`).
- EP `expert_map` lookups cached on CPU at `post_init` (no per-forward `.item()`).
- Slot `data_ptr` stability asserted after repeated `ensure_from_host_rows`.

Graph matrix: default eager; `--tier-allow-cuda-graphs` allows piecewise /
attention experiments only — full MoE+remap capture is not claimed.

## PR-F speculative decoding coexistence

Unit coverage: `test_slot_pool_extra_protect_across_calls`,
`test_spec_step_protects_verify_experts`, `test_spec_pin_skips_balanced_repin`
in `tests/model_executor/offloader/test_hierarchical_offload.py`.

### How to measure (hal / Ornith)

Compare warm tok/s and acceptance with hierarchical fixed
(`--tier-num-slots 4 --tier-policy balanced`) and speculation on vs off:

```bash
# Spec off (control)
.venv/bin/python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 --tier-ram-gb 8 --tier-policy balanced \
  --warm 4 --max-tokens 64 --prompt Hi \
  --output /tmp/hier_pr_f_nospec.json

# Spec on (same hierarchical flags + speculative config as in serve)
# Prefer an existing draft method the model supports; record acceptance from
# engine metrics / logs. If warm tok_s_warm(spec) < tok_s_warm(no-spec),
# disable speculation for that cold-cache / slots budget — acceptance alone
# is not enough when H2D stalls dominate.
```

Acceptance for merge: unit tests green; no draft/verify expert desync under
SPEC_PIN; PR body includes support matrix from
`hierarchical_expert_offload.md` and any hardware smoke numbers available.

AI assistance was used for this feature implementation.
