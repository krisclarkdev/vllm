# Hierarchical (Colibri-style) MoE Expert Offloading

vLLM can stage Mixture-of-Experts **routed expert weights** across a
three-tier hierarchy inspired by [Colibrì](https://github.com/JustVugg/colibri):

```text
NVMe ExpertStore  →  pinned system RAM (LFRU + learned pins)  →  device slots (XPU/CUDA)
```

**Placement only affects speed** — never precision or router semantics.

Dense components (attention, embeddings, norms, gates, shared experts, LM head)
stay resident on device. Only routed experts (`w13_weight` / `w2_weight` and
scales) are streamed.

## When to use it

- MoE models that do not fit entirely in device memory
- Intel **XPU** (Xe2/Xe3) primary target; CUDA works for the RAM↔device path
- Models larger than RAM when `--tier-disk-path` points at an ExpertStore

## Quick start

```bash
# Hardware bakeoff default: Mixtral-8x22B Instruct AWQ (Q4)
vllm serve MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ \
  --offload-backend hierarchical \
  --tier-num-slots 4 \
  --tier-ram-gb 32 \
  --tier-policy quality \
  --enforce-eager
```

Force a disk tier (builds ExpertStore on first run if missing):

```bash
vllm serve <moe-model> \
  --offload-backend hierarchical \
  --tier-device-expert-gb 4 \
  --tier-ram-gb 16 \
  --tier-disk-path /nvme/expert_store \
  --tier-pilot \
  --tier-policy balanced \
  --tier-repin-tokens 64
```

## CLI reference

| Flag | Meaning |
|------|---------|
| `--offload-backend hierarchical` | Enable hierarchical staging |
| `--tier-device-expert-gb` | Max GiB for device expert slots |
| `--tier-ram-gb` | Pinned RAM cache GiB (`-1` = auto) |
| `--tier-disk-path` | ExpertStore directory |
| `--tier-disk-mirror` | Optional second NVMe ExpertStore root (read-only) |
| `--tier-disk-weights a,b` | Primary/mirror bandwidth weights (or auto-probe) |
| `--tier-numa` | Interleave pinned RAM via libnuma (also `VLLM_TIER_NUMA`) |
| `--tier-spec-pin` / `--no-tier-spec-pin` | Freeze LFRU repin + protect draft/verify expert union during a speculative step (default on) |
| `--tier-policy quality\|balanced` | Live LFRU repin off/on |
| `--tier-repin-tokens N` | Repin interval (balanced) |
| `--tier-pilot` / `--tier-pilot-real` | Router-lookahead prefetch |
| `--tier-io-workers N` | Disk→RAM workers (default 8) |
| `--tier-direct` | Prefer `O_DIRECT` reads |
| `--tier-usage-path` | Learned usage heat-map path |
| `--tier-atlas-path` | Optional expert atlas JSON (affinity pins) |
| `--tier-affinity-topic` | Explicit atlas topic id (v1; no auto detect) |
| `--tier-dense-prefetch` | Also stage dense attention leftovers |
| `--tier-num-slots` | Override slots per MoE layer |
| `--tier-allow-cuda-graphs` | Experimental graphs (default off) |

## Memory planning

At startup vLLM logs a **tier plan** similar to Colibri’s `coli plan`:

```text
Hierarchical expert tier plan:
  policy=quality
  moe_layers=... local_experts=... slots/layer=...
  full_residency=yes|no
  batch_union_floor=...
  device_slots=... GiB
  ram_cache=... GiB
  disk_backing=... GiB
  predicted_bottleneck=pcie_or_ram_hits|nvme|none_full_residency
```

### Auto slot sizing

- `--tier-num-slots N` (N>0): honor and clamp to local expert count `E`.
- `--tier-num-slots 0` (default): derive from `--tier-device-expert-gb`, else
  from **free device memory** after a dense+KV reserve (~6 GiB).
- Slots are raised to a **batch-union floor**
  `min(E, max(top_k, estimated_unique))` so Mixtral-class workloads do not
  thrash at an accidentally tiny slot count.

### Full residency (`PIN_GB=all` analogue)

When `slots >= E` (`full_residency=yes`), hierarchical staging keeps every
local expert in the device slot pool after init and uses a **fast path**:
identity-style remap with no H2D/disk churn per forward (PR-A metrics still
recorded as device hits). This is the Colibri-like `PIN_GB=all` analogue.

```bash
# Mixtral-8x7B: 8 experts → full residency (needs free VRAM after profile)
vllm serve ... --offload-backend hierarchical --tier-num-slots 8
```

On Arc Pro B70 (~32 GiB) with Mixtral-8x7B AWQ, `slots=8` currently OOMs in
engine init; use `--tier-num-slots 4` for staging bakeoffs (see
`hierarchical_expert_offload_eval.md`).

Reservation order: dense resident → KV cache → activation scratch → expert
slots. Hierarchical weight offload **cannot** be combined with UVA
`--cpu-offload-gb`. It may coexist with KV CPU offload; leave headroom in
`--tier-ram-gb` for the KV tier.

### Pinned hot vs pageable cold

`--tier-ram-gb` caps the **OS-pinned** hot arena (`resolve_ram_budget_bytes`).
Hot experts (usage heat + initial fill) live in pinned frames for fast H2D;
overflow uses a **pageable** arena. Load-time VRAM parks respect the same
pinned budget instead of blindly using `pin_memory=False`.

## How it works

1. After weight load, full expert packs move to host (pinned while under the
   RAM budget; pageable overflow) and optionally ExpertStore on NVMe.
2. Each MoE layer gets a fixed **device slot pool** of `E_slots` experts.
   `XpuFusedMoe` / modular kernels see a dense pack of size `E_slots`.
3. On every forward, after `select_experts`, the tier manager **batch-unions**
   unique expert ids, ensures they are in slots (RAM hit → DMA, disk miss →
   O_DIRECT/io thread → DMA), and **remaps** `topk_ids` to slot indices.
   Full residency skips ensure churn after the initial fill.
4. Optional **PILOT** (`--tier-pilot`) schedules next-layer experts after the
   current ensure wait so DMA can overlap this layer’s expert GEMM. With
   `--tier-pilot-real`, the next layer’s registered gate runs on the current
   hidden state (extra gate cost) instead of reusing the current topk hint.
5. **Learned pins** (`.vllm_expert_usage`) seed device slots + pinned RAM at
   `post_init`; with `--tier-policy balanced`, `notify_tokens` from the worker
   triggers live LFRU `repin_hottest` every `--tier-repin-tokens`. Usage is
   flushed periodically and on shutdown.

Ensure is split into **schedule** (kick H2D on the hierarchical copy stream)
and **wait** (block immediately before the MoE GEMM); wait time goes to
`h2d_stall_ns`. ExpertStore reads prefer aligned `O_DIRECT` windows; failures
increment `disk_direct_fallback` and use buffered I/O. Demand I/O outranks
PILOT prefetch in the disk worker queue.

### Dual NVMe mirror

`--tier-disk-mirror` points at a second ExpertStore root (same layer files).
Routing is deterministic ``hash(layer, expert)`` skewed by
`--tier-disk-weights` (or a startup probe). Partial mirrors are fine: missing
or mismatched files stay on the primary. Mirror is read-only; usage heat maps
and manifests live on the primary. Shutdown logs
``MIRROR: served primary=… GiB mirror=… GiB``.

### NUMA pinned arenas

With `--tier-numa` / `VLLM_TIER_NUMA=1` on a multi-node Linux host that has
``libnuma``, pinned expert frames are passed through
``numa_interleave_memory`` after allocation. Unsupported hosts log once and
keep default OS placement.

### Activation pipeline (CUDA_PIPE spirit)

Hidden states / activations stay **on-device** through router → ensure → MoE
GEMM. `wait_ensure` joins only **weight** H2D events on the hierarchical copy
stream (not activations). Expert *ids* are still materialized on the host
(`torch.unique→tolist`) for slot/RAM/disk lookup; that D2H is counted as
`host_expert_id_syncs` and logged once.

Device slot packs are allocated once; in-place row copies keep
`param.data_ptr()` stable across ensures (required for any future MoE graph
experiments).

### Experimental graphs

| Mode | Hierarchical default | Notes |
|------|----------------------|-------|
| Eager MoE + overlapped H2D | **Default** (`enforce_eager`) | Supported |
| Piecewise / attention graphs | `--tier-allow-cuda-graphs` | MoE region stays eager; remaps outside capture |
| Full-graph MoE + dynamic remap | **Not supported** | Slot ptrs are stable, but unique expert sets change per step |

Requires platform graph enablement (e.g. `VLLM_XPU_ENABLE_XPU_GRAPH=1` on
XPU). Graph boundaries still drain the copy stream via
`sync_prev_onload` / `join_after_forward`.

## Speculative decoding coexistence (SPEC_PIN)

Draft and target forwards share **one** `ExpertTierManager` singleton on the
worker (`get_tier_manager()`). There is no separate draft tier manager; MoE
layers that share a `layer_id` also share the same slot pool.

During a speculative step (`execute_model` verify → `sample_tokens` draft):

1. `begin_spec_step` / `end_spec_step` bracket the step.
2. Slot eviction protects the **union** of local expert ids ensured so far in
   the step (verify residents cannot be draft-evicted).
3. With `--tier-policy balanced` and `--tier-spec-pin` (default),
   `notify_tokens` still records usage but **does not** live LFRU
   `repin_hottest` mid-step — pins must not diverge between verify and draft.

### Support matrix

| Spec method | Hierarchical | Notes |
|-------------|--------------|-------|
| ngram / suffix / medusa heads | OK | No draft MoE; SPEC_PIN still freezes repin harmlessly |
| EAGLE / EAGLE3 | Supported | Shared manager; draft MoE uses same pools when layer ids match |
| Draft-model MoE | Supported | Same singleton; protect union across verify+draft ensures |
| MTP (Gemma4 / Step3.5 / etc.) | Supported with caveats | Prefer matching draft/target expert quant; see warnings below |
| DFlash | Supported | Same SPEC_PIN window |

**Known-bad combinations (document / measure; no hard abort unless vLLM already
checks):**

- Draft MoE int8 / mismatched quant vs target when experts are remapped through
  the same slot pack (weights must compute the same function).
- Cold ExpertStore + speculation: acceptance rate may look fine while tok/s
  drops — disable speculation when warm tok/s with spec < without (see eval doc).

Disable SPEC_PIN only for debugging: `--no-tier-spec-pin`.

## Expert atlas (optional affinity pins)

Offline probes measure which experts fire for which **topics**. At serve time
you can boost cold-start pins for a session topic. **Atlas never changes
router outputs or weights** — only which experts are preferred in RAM/device
slots (placement).

### Build an atlas

```bash
# Live MoE probes (records hierarchical usage per topic)
VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python \
  benchmarks/hierarchical_expert_atlas.py \
  --model /path/to/moe \
  --probes benchmarks/hierarchical_atlas_probes.example.json.txt \
  --tier-num-slots 4 --tier-ram-gb 8 \
  --output /nvme/expert_store/.vllm_expert_atlas.json

# Or merge pre-recorded counts (CI / synthetic)
.venv/bin/python benchmarks/hierarchical_expert_atlas.py \
  --merge-json /tmp/atlas_counts.json \
  --output /tmp/.vllm_expert_atlas.json
```

Sample schema:

```json
{
  "version": 1,
  "model": "/path/to/moe",
  "topics": {
    "code": {
      "counts": {"0:3": 12, "0:5": 8, "1:2": 4},
      "probe_prompts": 2
    }
  }
}
```

### Serve with affinity

```bash
vllm serve <moe> --offload-backend hierarchical \
  --tier-atlas-path /nvme/expert_store/.vllm_expert_atlas.json \
  --tier-affinity-topic code \
  --tier-num-slots 4 --tier-ram-gb 8 --enforce-eager
```

Unset `--tier-atlas-path` (default) → pure LFRU / usage pins. Missing atlas
file or unknown topic falls back the same way.

Eval: held-out prompts for a topic with affinity on vs cold usage — compare
`device_hit_rate` / `ram_hit_rate` and `tok_s_warm` via
`benchmarks/hierarchical_tier_bakeoff.py` (`--tier-atlas-path` /
`--tier-affinity-topic`).

## Metrics

Prometheus (when enabled):

- `vllm_tier_expert_hits_total{tier=device|ram|disk}`
- `vllm_tier_expert_h2d_bytes_total` / `vllm_tier_expert_h2d_stall_seconds_total`
- `vllm_tier_expert_disk_direct_fallback_total`
- `vllm_tier_expert_pilot_predict_hits_total` /
  `vllm_tier_expert_pilot_predict_misses_total`
- `vllm_tier_expert_device_hit_rate`

## Limitations (v1)

- Mutual exclusion with **EPLB** (expert-row ownership conflicts)
- Default **eager** mode; `--tier-allow-cuda-graphs` is experimental on XPU
- Expert Parallelism is supported (tiers operate on local experts)
- Lossy router top-p cutting is intentionally not implemented (`quality` policy)

## Related

- Layer-wise PrefetchOffloader: `--offload-group-size` (whole layers, not experts)
- KV offload: [kv_offloading_usage.md](kv_offloading_usage.md)
