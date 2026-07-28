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


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_extra_protect_across_calls():
    """extra_protect keeps prior-step experts non-evictable (SPEC_PIN union)."""
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
    host_rows = {e: [host_w13[e], host_w2[e]] for e in range(E)}
    compute = current_platform.current_stream()

    remap, events = pool.ensure_from_host_rows([0, 1], host_rows)
    for ev in events:
        compute.wait_event(ev)
    assert pool.contains(0) and pool.contains(1)

    # Draft needs expert 2 but must not evict verify's expert 0.
    remap2, events2 = pool.ensure_from_host_rows(
        [2], host_rows, extra_protect={0}
    )
    for ev in events2:
        compute.wait_event(ev)
    assert pool.contains(0) and pool.contains(2)
    assert not pool.contains(1)
    assert set(remap2.keys()) == {2}


def test_spec_step_protects_verify_experts(monkeypatch):
    """begin_spec_step unions protect across schedule_ensure calls."""
    from vllm.model_executor.offloader.hierarchical import manager as mgr_mod
    from vllm.model_executor.offloader.hierarchical.manager import ExpertTierManager

    monkeypatch.setattr(mgr_mod.current_platform, "Stream", lambda: object())

    cfg = HierarchicalOffloadConfig(tier_num_slots=2, tier_ram_gb=0.01)
    mgr = ExpertTierManager(cfg)

    class _FakePool:
        def __init__(self):
            self.calls: list[tuple[list[int], set[int] | None]] = []
            self._resident: set[int] = set()
            self.num_slots = 2

        def contains(self, eid: int) -> bool:
            return eid in self._resident

        def ensure_from_host_rows(self, expert_ids, host_rows, *, extra_protect=None):
            self.calls.append((list(expert_ids), set(extra_protect or ())))
            for eid in expert_ids:
                if eid >= 0:
                    self._resident.add(int(eid))
            return {int(e): i for i, e in enumerate(expert_ids) if e >= 0}, []

        def mark_ready(self, _ids):
            return None

        def slot_of(self, eid: int):
            return 0 if eid in self._resident else None

    class _FakeState:
        def __init__(self, pool):
            self.slot_pool = pool
            self.row_nbytes = 8
            self.num_experts = 8
            self.full_residency = False
            self.host_weights = [
                torch.zeros(8, 4),
                torch.zeros(8, 4),
            ]

        def to_local(self, gid: int) -> int:
            return gid

    pool = _FakePool()
    mgr.layers[0] = _FakeState(pool)  # type: ignore[assignment]
    mgr._ram = None
    mgr._disk = None

    mgr.begin_spec_step()
    # Verify ensures 0,1
    mgr._schedule_ensure_layer(
        0, [0, 1], record_usage=False
    )
    assert pool.calls[0][1] == set()  # first call: no prior protect
    # Draft ensures 2 — prior union {0,1} must be passed as extra_protect
    mgr._schedule_ensure_layer(0, [2], record_usage=False)
    assert {0, 1}.issubset(pool.calls[1][1])
    mgr.end_spec_step()
    assert not mgr.in_spec_step


def test_spec_pin_skips_balanced_repin(monkeypatch):
    """During a SPEC_PIN step, balanced notify_tokens must not repin."""
    from vllm.model_executor.offloader.hierarchical import manager as mgr_mod
    from vllm.model_executor.offloader.hierarchical.manager import ExpertTierManager

    monkeypatch.setattr(mgr_mod.current_platform, "Stream", lambda: object())

    cfg = HierarchicalOffloadConfig(
        tier_num_slots=2,
        tier_ram_gb=0.01,
        tier_policy="balanced",
        tier_repin_tokens=1,
        tier_spec_pin=True,
    )
    mgr = ExpertTierManager(cfg)
    called = {"n": 0}

    class _FakeRam:
        enabled = True

        def repin_hottest(self, *args, **kwargs):
            called["n"] += 1

    mgr._ram = _FakeRam()  # type: ignore[assignment]
    mgr._usage = type(
        "U",
        (),
        {"hottest": staticmethod(lambda *a, **k: [0]), "flush": lambda self: None},
    )()

    class _FakeState:
        def __init__(self):
            self.slot_pool = type("P", (), {"num_slots": 2})()
            self.num_experts = 8

    mgr.layers[0] = _FakeState()  # type: ignore[assignment]

    mgr.begin_spec_step()
    mgr.notify_tokens(10)
    assert called["n"] == 0
    mgr.end_spec_step()
    mgr.notify_tokens(10)
    assert called["n"] == 1


def test_tier_spec_pin_default_on():
    cfg = HierarchicalOffloadConfig()
    assert cfg.tier_spec_pin is True


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


def test_coalesce_expert_ranges():
    from vllm.model_executor.offloader.hierarchical.format import (
        coalesce_expert_ranges,
    )

    assert coalesce_expert_ranges([]) == []
    assert coalesce_expert_ranges([3, 1, 2, 5]) == [(1, 3), (5, 5)]
    assert coalesce_expert_ranges([-1, 0, 0, 2]) == [(0, 0), (2, 2)]


def test_direct_read_window_alignment():
    from vllm.model_executor.offloader.hierarchical.disk_store import (
        DIRECT_ALIGN,
        direct_read_window,
    )

    off, length, pad = direct_read_window(100, 50, align=DIRECT_ALIGN)
    assert off % DIRECT_ALIGN == 0
    assert length % DIRECT_ALIGN == 0
    assert pad == 100 - off
    assert off + pad + 50 <= off + length


def test_expert_store_direct_fallback_counter(tmp_path: Path, monkeypatch):
    """When O_DIRECT open fails, reader increments disk_direct_fallback."""
    import os

    from vllm.model_executor.offloader.hierarchical import disk_store as ds
    from vllm.model_executor.offloader.hierarchical.disk_store import (
        ExpertStoreReader,
    )

    w13 = torch.randn(2, 4, 4)
    w2 = torch.randn(2, 4, 4)
    convert_layer_from_device_params(
        tmp_path, layer_id=0, weight_tensors=[w13, w2], model_id="t"
    )

    monkeypatch.setattr(ds, "O_DIRECT", 0x4000)
    monkeypatch.setattr(ds, "probe_o_direct", lambda _p: True)

    real_open = os.open

    def flaky_open(path, flags, *args, **kwargs):
        if flags & ds.O_DIRECT:
            raise OSError(22, "simulated EINVAL")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(ds.os, "open", flaky_open)
    reader = ExpertStoreReader(str(tmp_path), prefer_direct=True)
    assert reader._use_direct
    blob = reader.read_row_sync(0, 0)
    assert blob.numel() > 0
    assert reader.disk_direct_fallback >= 1
    reader.close()


def test_priority_pool_demand_before_pilot():
    import threading
    import time

    from vllm.model_executor.offloader.hierarchical.disk_store import (
        IoPriority,
        _PriorityIoPool,
    )

    order: list[str] = []
    pool = _PriorityIoPool(num_workers=1)
    gate = threading.Event()

    def pilot_job():
        gate.wait(timeout=2.0)
        order.append("pilot")
        return "pilot"

    def demand_job():
        order.append("demand")
        return "demand"

    fut_p = pool.submit(pilot_job, priority=IoPriority.PILOT)
    time.sleep(0.05)  # let worker pick up pilot and block on gate
    fut_d = pool.submit(demand_job, priority=IoPriority.DEMAND)
    fut_p2 = pool.submit(
        lambda: order.append("pilot2") or "pilot2", priority=IoPriority.PILOT
    )
    gate.set()
    assert fut_p.result(timeout=2.0) == "pilot"
    assert fut_d.result(timeout=2.0) == "demand"
    assert fut_p2.result(timeout=2.0) == "pilot2"
    # After the in-flight pilot, DEMAND must run before the queued PILOT.
    assert order.index("demand") < order.index("pilot2")
    pool.shutdown(wait=True)


def test_schedule_wait_ordering(monkeypatch):
    """schedule must not wait; wait_ensure records stall and finishes DMA."""
    from unittest.mock import MagicMock

    from vllm.model_executor.offloader.hierarchical.manager import (
        ExpertTierManager,
        LayerTierState,
    )

    waits: list[object] = []

    class _FakeStream:
        def wait_event(self, ev):
            waits.append(ev)

    monkeypatch.setattr(
        "vllm.model_executor.offloader.hierarchical.manager.current_platform.current_stream",
        lambda: _FakeStream(),
    )

    cfg = HierarchicalOffloadConfig(tier_num_slots=2, tier_ram_gb=0.01)
    mgr = ExpertTierManager(cfg)
    E, H, I = 4, 8, 16
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    row_nbytes = int(host_w13[0].nbytes + host_w2[0].nbytes)
    resident: set[int] = set()
    pool = MagicMock()
    pool.num_slots = 2

    def contains(eid: int) -> bool:
        return eid in resident

    def ensure_from_host_rows(ids, host_rows):
        remap: dict[int, int] = {}
        events = []
        for eid in ids:
            if eid < 0:
                continue
            if eid not in resident:
                resident.add(eid)
                remap[eid] = len(resident) - 1
                events.append(f"ev-{eid}")
            else:
                remap[eid] = sorted(resident).index(eid)
        return remap, events

    pool.contains.side_effect = contains
    pool.ensure_from_host_rows.side_effect = ensure_from_host_rows
    pool.slot_of.side_effect = lambda eid: (
        sorted(resident).index(eid) if eid in resident else None
    )
    pool.mark_ready.side_effect = lambda _ids: None

    mgr.layers[0] = LayerTierState(
        layer_id=0,
        module=_FakeExperts(num_experts=E, hidden=H, inter=I),
        host_weights=[host_w13, host_w2],
        param_names=["w13_weight", "w2_weight"],
        slot_pool=pool,
        row_nbytes=row_nbytes,
    )
    mgr._ram = PinnedExpertRamCache(
        capacity_bytes=row_nbytes * 8, row_nbytes=row_nbytes
    )

    topk = torch.tensor([[0, 1]], dtype=torch.long)
    pending = mgr.schedule_ensure_and_remap(0, topk)
    assert waits == []
    assert pending.events
    assert mgr.stats.h2d_stall_ns == 0
    out = mgr.wait_ensure(pending)
    assert waits == pending.events
    assert mgr.stats.h2d_bytes == 2 * row_nbytes
    assert out.shape == topk.shape


def test_pilot_gate_registration_and_prefetch(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.model_executor.offloader.hierarchical.manager import (
        ExpertTierManager,
        LayerTierState,
    )
    from vllm.model_executor.offloader.hierarchical.pilot import PilotPrefetcher

    cfg = HierarchicalOffloadConfig(
        tier_num_slots=2, tier_ram_gb=0.01, tier_pilot=True, tier_pilot_real=True
    )
    mgr = ExpertTierManager(cfg)
    E, H, I = 4, 4, 4
    host_w13 = torch.randn(E, 2 * I, H)
    host_w2 = torch.randn(E, H, I)
    row_nbytes = int(host_w13[0].nbytes + host_w2[0].nbytes)

    for lid in (0, 1):
        pool = MagicMock()
        pool.num_slots = 2
        pool.contains.return_value = True
        pool.slot_of.side_effect = lambda eid, _lid=lid: eid % 2
        pool.ensure_from_host_rows.return_value = ({}, [])
        pool.mark_ready.side_effect = lambda _ids: None
        mgr.layers[lid] = LayerTierState(
            layer_id=lid,
            module=_FakeExperts(num_experts=E, hidden=H, inter=I),
            host_weights=[host_w13, host_w2],
            param_names=["w13_weight", "w2_weight"],
            slot_pool=pool,
            row_nbytes=row_nbytes,
        )
        # Force non-full-residency path off for simplicity.
        mgr.layers[lid].full_residency = False

    called: list[tuple] = []

    def fake_prefetch(layer_id, expert_ids, *, block=False):
        called.append((layer_id, list(expert_ids), block))

    mgr.prefetch_experts = fake_prefetch  # type: ignore[method-assign]
    pilot = PilotPrefetcher(mgr, real=True)
    mgr._pilot = pilot

    class _Gate(nn.Module):
        def forward(self, x):
            # Prefer experts 1 and 2.
            logits = torch.zeros(x.shape[0], E)
            logits[..., 1] = 10
            logits[..., 2] = 9
            return logits

    pilot.register_gate(1, _Gate())
    h = torch.randn(2, H)
    pilot.prefetch_next(0, h, current_expert_ids=[0])
    assert called
    assert called[0][0] == 1
    assert called[0][2] is False
    assert 1 in called[0][1] and 2 in called[0][1]

    pilot.score_prediction(1, [1, 3])
    snap = mgr.stats.snapshot()
    assert snap["pilot_predict_hits"] >= 1
    assert snap["pilot_predict_misses"] >= 1


def test_volume_for_expert_stable_and_skewed():
    from vllm.model_executor.offloader.hierarchical.mirror import (
        parse_disk_weights,
        volume_for_expert,
    )

    assert parse_disk_weights(None) == (1.0, 1.0)
    assert parse_disk_weights("2,1") == (2.0, 1.0)
    a = volume_for_expert(3, 7, 1.0, 1.0)
    b = volume_for_expert(3, 7, 1.0, 1.0)
    assert a == b
    # Extreme skew → almost all primary.
    primary_count = sum(
        1 for e in range(200) if volume_for_expert(0, e, 100.0, 1.0) == 0
    )
    assert primary_count > 150
    # Mirror-only weights.
    assert all(volume_for_expert(0, e, 0.0, 1.0) == 1 for e in range(20))


def test_validate_partial_mirror(tmp_path: Path):
    from vllm.model_executor.offloader.hierarchical.mirror import (
        validate_mirror_files,
    )

    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    primary.mkdir()
    mirror.mkdir()
    w13 = torch.randn(4, 4, 4)
    w2 = torch.randn(4, 4, 4)
    convert_layer_from_device_params(
        primary, layer_id=0, weight_tensors=[w13, w2], model_id="t"
    )
    convert_layer_from_device_params(
        primary, layer_id=1, weight_tensors=[w13, w2], model_id="t"
    )
    # Mirror only layer 0.
    import shutil

    shutil.copy2(primary / "L000.experts", mirror / "L000.experts")
    ok = validate_mirror_files(primary, mirror)
    assert "L000.experts" in ok
    assert "L001.experts" not in ok


def test_mirrored_reader_fallback(tmp_path: Path, monkeypatch):
    from vllm.model_executor.offloader.hierarchical.disk_store import (
        ExpertStoreReader,
        MirroredExpertStoreReader,
    )
    from vllm.model_executor.offloader.hierarchical.mirror import (
        volume_for_expert,
    )

    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    primary.mkdir()
    mirror.mkdir()
    w13 = torch.randn(8, 4, 4)
    w2 = torch.randn(8, 4, 4)
    convert_layer_from_device_params(
        primary, layer_id=0, weight_tensors=[w13, w2], model_id="t"
    )
    import shutil

    shutil.copy2(primary / "L000.experts", mirror / "L000.experts")
    shutil.copy2(primary / "manifest.json", mirror / "manifest.json")

    primary_r = ExpertStoreReader(str(primary), prefer_direct=False)
    mirrored = MirroredExpertStoreReader(
        primary_r,
        mirror_path=str(mirror),
        prefer_direct=False,
        disk_weights="0,1",  # force mirror route when file mirrored
    )
    assert mirrored._mirror is not None

    # Force mirror reads to fail → primary fallback.
    def boom(*_a, **_k):
        raise OSError("simulated mirror failure")

    monkeypatch.setattr(mirrored._mirror, "read_rows_sync", boom)
    rows = mirrored.read_rows_sync(0, [0, 1, 2])
    assert set(rows) == {0, 1, 2}
    assert mirrored._mirror_fallback_warned
    # Primary still served bytes.
    assert primary_r.bytes_served > 0
    # Routing stable for demand vs "pilot".
    assert volume_for_expert(0, 3, 0.0, 1.0) == mirrored.route_volume(0, 3)
    mirrored.close()


def test_mirrored_reader_bytes_on_both_volumes(tmp_path: Path):
    from vllm.model_executor.offloader.hierarchical.disk_store import (
        ExpertStoreReader,
        MirroredExpertStoreReader,
        ensure_store_or_none,
    )

    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    primary.mkdir()
    mirror.mkdir()
    w13 = torch.randn(16, 2, 2)
    w2 = torch.randn(16, 2, 2)
    convert_layer_from_device_params(
        primary, layer_id=0, weight_tensors=[w13, w2], model_id="t"
    )
    import shutil

    shutil.copy2(primary / "L000.experts", mirror / "L000.experts")

    reader = ensure_store_or_none(
        str(primary),
        num_workers=2,
        prefer_direct=False,
        disk_mirror=str(mirror),
        disk_weights="1,1",
    )
    assert isinstance(reader, MirroredExpertStoreReader)
    ids = list(range(16))
    rows = reader.read_rows_sync(0, ids)
    assert len(rows) == 16
    stats = reader.mirror_stats()
    assert stats["primary_bytes"] > 0
    assert stats["mirror_bytes"] > 0
    reader.close()

    # Single-disk unchanged.
    solo = ensure_store_or_none(
        str(primary), num_workers=1, prefer_direct=False
    )
    assert isinstance(solo, ExpertStoreReader)
    assert solo.read_row_sync(0, 0).numel() > 0
    solo.close()


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU for device slot DMA",
)
def test_slot_pool_pointer_stability():
    """Slot-backed param buffers must keep a stable data_ptr across ensures."""
    device = torch.device(f"{current_platform.device_type}:0")
    E, H, I = 8, 8, 8
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
    ptrs0 = pool.slot_data_ptrs()
    host_rows = {e: [host_w13[e], host_w2[e]] for e in range(E)}
    compute = current_platform.current_stream()
    for batch in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]):
        remap, events = pool.ensure_from_host_rows(batch, host_rows)
        for ev in events:
            compute.wait_event(ev)
        pool.mark_ready(list(remap.keys()))
        pool.assert_pointer_stable()
        assert pool.slot_data_ptrs() == ptrs0


def test_tier_allow_cuda_graphs_default_forces_eager_doc():
    """Hierarchical defaults to eager unless --tier-allow-cuda-graphs."""
    cfg = HierarchicalOffloadConfig()
    assert cfg.tier_allow_cuda_graphs is False


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU",
)
def test_hierarchical_graph_opt_in_smoke():
    """Guarded smoke: with tier_allow_cuda_graphs, config stays non-eager opt-in.

    Full XPU/CUDA graph capture of MoE+remap is not claimed; this only checks
    the flag plumbing. Skip deeper graph capture unless the platform reports
    graph support.
    """
    cfg = HierarchicalOffloadConfig(
        tier_num_slots=2, tier_ram_gb=0.01, tier_allow_cuda_graphs=True
    )
    assert cfg.tier_allow_cuda_graphs is True
    # Platform-specific graph capability probe (best-effort).
    has_graphs = False
    try:
        if current_platform.is_cuda():
            has_graphs = hasattr(torch.cuda, "CUDAGraph")
        elif current_platform.is_xpu():
            import os

            has_graphs = os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH", "0") == "1"
    except Exception:
        has_graphs = False
    if not has_graphs:
        pytest.skip("No CUDA/XPU graph runtime enabled for deeper smoke")
    # Deeper capture is out of scope for PR-E honesty matrix.
    pytest.skip("Full MoE+remap graph capture not supported; see docs matrix")
