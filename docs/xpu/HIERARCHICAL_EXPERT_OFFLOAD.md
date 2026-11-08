# Hierarchical MoE expert staging on XPU

See the full user guide:
[`docs/features/hierarchical_expert_offload.md`](../features/hierarchical_expert_offload.md).

## XPU-specific notes

- Prefer **explicit DMA** (`tensor.copy_(..., non_blocking=True)` on an XPU
  copy stream) into device expert slots. Do **not** rely on UVA host-pointer
  kernel loads for expert GEMMs.
- Disk→RAM uses `O_DIRECT` into already-pinned frames (no `cudaHostRegister`
  analogue required for v1). Optional `--tier-disk-mirror` for dual-NVMe.
- Multi-stream compute∥copy overlap can be weaker than CUDA; deepen
  `--tier-pilot` carefully (affinity pilot at tiny slot counts can thrash).
- Mutual exclusion with EPLB in v1.

## Activation pipeline

- Activations remain on XPU through select_experts → schedule/wait ensure →
  `XpuFusedMoe` GEMM.
- `wait_ensure` waits only on weight H2D events recorded on the hierarchical
  copy stream.
- Expert-id `unique→tolist` is an intentional host sync for staging metadata
  (`host_expert_id_syncs`); it is not an activation D2H.

## Graphs (experimental)

Default hierarchical serving forces `enforce_eager` unless
`--tier-allow-cuda-graphs`.

| Mode | Status on XPU |
|------|----------------|
| Eager MoE + overlapped weight H2D | Supported (default) |
| Piecewise / attention-only graphs | Opt-in; keep MoE eager |
| Full cudagraph/xpugraph of MoE + dynamic remap | **Not supported** |

Slot buffers are address-stable (in-place row copies). Remapping still changes
which experts occupy those slots each step, so capturing MoE inside a graph is
unsafe unless the captured unique set is fixed or ensure runs outside the
graph. Also set `VLLM_XPU_ENABLE_XPU_GRAPH=1` when experimenting.

## Bakeoff

Default hardware model: **Mixtral-8x22B Instruct AWQ (Q4)**
(`MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ`, local path
`/tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ`). Mixtral has 8 experts;
use `--tier-num-slots 4` to force RAM↔device staging.

```bash
python benchmarks/hierarchical_tier_bakeoff.py \
  --model /tank/nas/models/Mixtral-8x22B-Instruct-v0.1-AWQ \
  --tier-num-slots 4 \
  --tier-ram-gb 32 \
  --colibri-tok-s <published_colibri_number> \
  --output /tmp/tier_bakeoff.json
```
