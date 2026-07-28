# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hierarchical (Colibri-style) expert staging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vllm.config.offload import HierarchicalOffloadConfig, OffloadConfig
from vllm.model_executor.offloader.base import create_offloader
from vllm.model_executor.offloader.hierarchical.device_slots import ExpertSlotPool
from vllm.model_executor.offloader.hierarchical.format import (
    convert_layer_from_device_params,
    load_manifest,
    pack_expert_row_torch,
    unpack_expert_row,
)
from vllm.model_executor.offloader.hierarchical.planner import (
    batch_union_floor,
    build_tier_plan,
    compute_slots_per_layer,
    resolve_ram_budget_bytes,
)
from vllm.model_executor.offloader.hierarchical.ram_cache import PinnedExpertRamCache
from vllm.model_executor.offloader.hierarchical.usage import ExpertUsageStore
from vllm.model_executor.offloader.hierarchical_offloader import HierarchicalOffloader
from vllm.platforms import current_platform


class _FakeExperts(nn.Module):
    def __init__(self, num_experts: int = 8, hidden: int = 16, inter: int = 32):
        super().__init__()
        self.top_k = 2
        self.w13_weight = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.w2_weight = nn.Parameter(torch.randn(num_experts, hidden, inter))


def test_create_offloader_hierarchical():
    cfg = OffloadConfig(
        offload_backend="hierarchical",
        hierarchical=HierarchicalOffloadConfig(tier_num_slots=4),
    )
    off = create_offloader(cfg)
    assert isinstance(off, HierarchicalOffloader)


def test_create_offloader_auto_hierarchical():
    cfg = OffloadConfig(
        offload_backend="auto",
        hierarchical=HierarchicalOffloadConfig(tier_device_expert_gb=1.0),
    )
    off = create_offloader(cfg)
    assert isinstance(off, HierarchicalOffloader)


def test_planner_slots_and_ram():
    cfg = HierarchicalOffloadConfig(tier_num_slots=16, tier_ram_gb=2.0)
    assert compute_slots_per_layer(
        cfg, num_moe_layers=4, num_local_experts=64, expert_row_bytes=1024
    ) == 16
    assert resolve_ram_budget_bytes(cfg) == int(2 * 1024**3)
    plan = build_tier_plan(
        cfg,
        num_moe_layers=4,
        num_local_experts=64,
        expert_row_bytes=1024 * 1024,
        top_k=8,
    )
    assert plan.slots_per_layer == 16
    assert plan.full_residency is False
    assert "Hierarchical expert tier plan" in plan.summary()
    assert "full_residency=no" in plan.summary()


def test_planner_auto_slots_clamp_and_floor():
    # Explicit slots clamp to E.
    cfg = HierarchicalOffloadConfig(tier_num_slots=128)
    assert (
        compute_slots_per_layer(
            cfg, num_moe_layers=2, num_local_experts=8, expert_row_bytes=1024
        )
        == 8
    )
    # Auto from device GB budget.
    cfg = HierarchicalOffloadConfig(tier_num_slots=0, tier_device_expert_gb=1.0)
    # 1 GiB / 2 layers / 64 MiB row → 8 slots before floor; floor may raise.
    row = 64 * 1024 * 1024
    slots = compute_slots_per_layer(
        cfg,
        num_moe_layers=2,
        num_local_experts=64,
        expert_row_bytes=row,
        top_k=8,
        estimated_unique=16,
    )
    assert slots >= batch_union_floor(
        num_local_experts=64, top_k=8, estimated_unique=16
    )
    assert slots <= 64
    # Free-device auto path.
    cfg = HierarchicalOffloadConfig(tier_num_slots=0, tier_device_expert_gb=0)
    slots_free = compute_slots_per_layer(
        cfg,
        num_moe_layers=4,
        num_local_experts=8,
        expert_row_bytes=1024 * 1024,
        top_k=2,
        free_device_bytes=20 * 1024**3,
        device_reserve_bytes=6 * 1024**3,
        estimated_unique=4,
    )
    assert slots_free == 8  # full residency fits
    plan = build_tier_plan(
        cfg,
        num_moe_layers=4,
        num_local_experts=8,
        expert_row_bytes=1024 * 1024,
        top_k=2,
        free_device_bytes=20 * 1024**3,
        estimated_unique=4,
    )
    assert plan.full_residency is True
    assert "full_residency=yes" in plan.summary()


def test_ram_cache_pinned_budget_overflow():
    """Soft-pinned hot rows stay in the pinned arena; overflow is pageable."""
    row = torch.arange(64, dtype=torch.uint8)
    cache = PinnedExpertRamCache(
        capacity_bytes=64 * 2,  # 2 pinned frames
        row_nbytes=64,
        pageable_capacity_bytes=64 * 4,
    )
    cache.put(0, 0, row, pinned=True)
    cache.put(0, 1, row + 1, pinned=True)
    # Additional hot puts spill to pageable when pinned soft-pins fill arena.
    cache.put(0, 2, row + 2, pinned=True)
    assert cache.get(0, 0) is not None
    assert cache.get(0, 1) is not None
    assert cache.get(0, 2) is not None
    assert cache.pinned_bytes_used <= cache.pinned_capacity_bytes
    # Cold (unpinned) rows land in pageable without evicting soft pins.
    cache.put(0, 3, row + 3, pinned=False)
    assert cache.get(0, 0) is not None
    assert cache.get(0, 3) is not None


def test_usage_store_roundtrip(tmp_path: Path):
    path = tmp_path / ".vllm_expert_usage"
    store = ExpertUsageStore(str(path))
    store.record(0, [1, 2, 2, 3])
    store.record(0, [2])
    store.flush()
    store2 = ExpertUsageStore(str(path))
    hot = store2.hottest(0, 2, 8)
    assert hot[0] == 2


def test_ram_cache_put_get_evict():
    row = torch.arange(64, dtype=torch.uint8)
    cache = PinnedExpertRamCache(capacity_bytes=64 * 2, row_nbytes=64)
    assert cache.enabled
    cache.put(0, 0, row, pinned=True)
    cache.put(0, 1, row + 1)
    got = cache.get(0, 0)
    assert got is not None
    assert torch.equal(got, row)
    # Force eviction of non-pinned (pageable) occupants
    cache.put(0, 2, row + 2)
    assert cache.get(0, 0) is not None  # soft-pinned survives


def test_expert_store_format(tmp_path: Path):
    w13 = torch.randn(4, 8, 4)
    w2 = torch.randn(4, 4, 8)
    meta = convert_layer_from_device_params(
        tmp_path, layer_id=0, weight_tensors=[w13, w2], model_id="test"
    )
    assert meta.num_experts == 4
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert len(manifest.layers) == 1
    packed = pack_expert_row_torch([w13[1], w2[1]])
    specs = meta.tensor_specs
    unpacked = unpack_expert_row(packed, specs)
    assert torch.allclose(unpacked[0], w13[1])
    assert torch.allclose(unpacked[1], w2[1])


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_ensure_remap():
    device = torch.device(f"{current_platform.device_type}:0")
    E, H, I = 8, 16, 32
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    stream = current_platform.Stream()
    pool = ExpertSlotPool(
        layer_id=0,
        weight_templates=[host_w13, host_w2],
        num_slots=4,
        copy_stream=stream,
        device=device,
    )
    ids = [0, 3, 5]
    host_rows = {e: [host_w13[e], host_w2[e]] for e in ids}
    remap, events = pool.ensure_from_host_rows(ids, host_rows)
    compute = current_platform.current_stream()
    for ev in events:
        compute.wait_event(ev)
    assert set(remap.keys()) == set(ids)
    for e, s in remap.items():
        assert torch.allclose(pool.slot_weights[0][s].cpu(), host_w13[e], atol=1e-5)


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_protects_same_batch_residents():
    """Same-batch ensure must not evict experts already selected this call."""
    device = torch.device(f"{current_platform.device_type}:0")
    E, H, I = 8, 8, 8
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    stream = current_platform.Stream()
    pool = ExpertSlotPool(
        layer_id=0,
        weight_templates=[host_w13, host_w2],
        num_slots=2,
        copy_stream=stream,
        device=device,
    )
    # Fill both slots.
    first = [0, 1]
    host_rows = {e: [host_w13[e], host_w2[e]] for e in range(E)}
    remap, events = pool.ensure_from_host_rows(first, host_rows)
    compute = current_platform.current_stream()
    for ev in events:
        compute.wait_event(ev)
    assert pool.contains(0) and pool.contains(1)

    # Request resident 0 plus a new expert: must keep 0, replace 1.
    remap2, events2 = pool.ensure_from_host_rows([0, 2], host_rows)
    for ev in events2:
        compute.wait_event(ev)
    assert set(remap2.keys()) == {0, 2}
    assert pool.contains(0) and pool.contains(2)
    assert not pool.contains(1)
    assert torch.allclose(
        pool.slot_weights[0][remap2[0]].cpu(), host_w13[0], atol=1e-5
    )
    assert torch.allclose(
        pool.slot_weights[0][remap2[2]].cpu(), host_w13[2], atol=1e-5
    )

    # Oversubscribe beyond slots → clear error (not silent corruption).
    with pytest.raises(RuntimeError, match="cannot allocate a slot"):
        pool.ensure_from_host_rows([0, 2, 3], host_rows)


def test_hierarchical_offloader_registers_modules():
    cfg = HierarchicalOffloadConfig(tier_num_slots=2, tier_ram_gb=0.01)
    off = HierarchicalOffloader(cfg)

    def gen():
        for _ in range(2):
            block = nn.Sequential(_FakeExperts())
            # Nest experts so finder sees w13_weight
            yield block

    # Build modules that contain FakeExperts as children
    modules = []
    for i in range(2):
        m = nn.Module()
        m.add_module("mlp", nn.Module())
        m.mlp.add_module("experts", _FakeExperts())  # type: ignore[attr-defined]
        modules.append(m)

    def modules_gen():
        yield from modules

    wrapped = off.wrap_modules(modules_gen())
    assert len(wrapped) == 2
    assert len(off.manager._pending_modules) >= 2
    off.shutdown()


def test_tier_stats_snapshot_and_reset():
    from vllm.model_executor.offloader.hierarchical.metrics import TierStats

    stats = TierStats()
    stats.device_hits = 3
    stats.device_misses = 1
    stats.ensure_calls = 2
    stats.h2d_bytes = 100
    stats.unique_experts_sum = 5
    stats.unique_experts_hist[2] = 1
    snap = stats.snapshot()
    assert snap["device_hits"] == 3
    assert snap["device_misses"] == 1
    assert snap["ensure_calls"] == 2
    assert snap["h2d_bytes"] == 100
    assert snap["unique_experts_sum"] == 5
    assert snap["device_hit_rate"] == 0.75
    assert snap["unique_experts_hist"]["2"] == 1
    stats.reset()
    assert stats.device_hits == 0
    assert stats.ensure_calls == 0
    assert stats.unique_experts_hist == {}


def test_tier_stats_move_on_fake_ensure(monkeypatch):
    """Counters must move on a CPU/mock ensure path (no real DMA)."""
    from unittest.mock import MagicMock

    from vllm.model_executor.offloader.hierarchical.manager import (
        ExpertTierManager,
        LayerTierState,
    )

    class _FakeStream:
        def wait_event(self, _ev):
            return None

    monkeypatch.setattr(
        "vllm.model_executor.offloader.hierarchical.manager.current_platform.current_stream",
        lambda: _FakeStream(),
    )

    cfg = HierarchicalOffloadConfig(tier_num_slots=2, tier_ram_gb=0.01)
    mgr = ExpertTierManager(cfg)

    E, H, I = 4, 8, 16
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    host_weights = [host_w13, host_w2]
    row_nbytes = int(host_w13[0].nbytes + host_w2[0].nbytes)

    resident: set[int] = set()
    pool = MagicMock()
    pool.num_slots = 2

    def contains(eid: int) -> bool:
        return eid in resident

    def ensure_from_host_rows(ids, host_rows):
        remap: dict[int, int] = {}
        for eid in ids:
            if eid < 0:
                continue
            assert eid in host_rows
            if eid not in resident:
                resident.add(eid)
                remap[eid] = len(resident) - 1
        return remap, []

    def slot_of(eid: int):
        if eid not in resident:
            return None
        return sorted(resident).index(eid)

    pool.contains.side_effect = contains
    pool.ensure_from_host_rows.side_effect = ensure_from_host_rows
    pool.slot_of.side_effect = slot_of
    pool.mark_ready.side_effect = lambda _ids: None

    mgr.layers[0] = LayerTierState(
        layer_id=0,
        module=_FakeExperts(num_experts=E, hidden=H, inter=I),
        host_weights=host_weights,
        param_names=["w13_weight", "w2_weight"],
        slot_pool=pool,
        row_nbytes=row_nbytes,
    )
    mgr._ram = PinnedExpertRamCache(
        capacity_bytes=row_nbytes * 8, row_nbytes=row_nbytes
    )

    mgr._ensure_layer(0, [0, 1], record_usage=False)
    snap1 = mgr.stats.snapshot()
    assert snap1["ensure_calls"] == 1
    assert snap1["device_misses"] == 2
    assert snap1["device_hits"] == 0
    assert snap1["ram_misses"] == 2  # host pack fallback
    assert snap1["disk_hits"] == 0
    assert snap1["disk_misses"] == 0
    assert snap1["h2d_bytes"] == 2 * row_nbytes
    assert snap1["unique_experts_sum"] == 2
    assert snap1["ram_hits"] == 0

    # Second ensure: device-resident + RAM-cached host rows.
    mgr._ensure_layer(0, [0, 1], record_usage=False)
    snap2 = mgr.stats.snapshot()
    assert snap2["ensure_calls"] == 2
    assert snap2["device_hits"] == 2
    assert snap2["device_misses"] == 2
    assert snap2["ram_hits"] == 2
    assert snap2["unique_experts_sum"] == 4
    assert snap2["h2d_bytes"] == 2 * row_nbytes  # no new DMA

    # Disk miss path when a store is attached without this layer.
    disk = MagicMock()
    disk.has_layer.return_value = False
    mgr._disk = disk
    mgr.stats.reset()
    resident.clear()
    mgr._ensure_layer(0, [2], record_usage=False)
    snap3 = mgr.stats.snapshot()
    assert snap3["ensure_calls"] == 1
    assert snap3["disk_misses"] == 1
    assert snap3["disk_hits"] == 0
    assert snap3["ram_misses"] == 1
