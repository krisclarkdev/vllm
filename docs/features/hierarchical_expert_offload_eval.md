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
  --num-prompts 4 --output /tmp/hier_pr_a_bakeoff.json

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

Use `--tier-num-slots 4` (of 8 experts) so staging is exercised; raise to 8
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

## Results (fill on hardware)

| Setup | tok_s_warm | TTFT proxy | device hit rate | Notes |
|-------|------------|------------|-----------------|-------|
| baseline (no offload) | | | n/a | Mixtral-8x7B AWQ |
| hierarchical slots=4 | | | | Mixtral-8x7B AWQ |
| Colibri reference | | | | optional `--colibri-tok-s` |

AI assistance was used for this feature implementation.
"""
