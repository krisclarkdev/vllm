# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup memory planner / tier plan logger for hierarchical offload."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.config.offload import HierarchicalOffloadConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Dense weights + KV/activation headroom kept free when auto-sizing slots from
# free device memory (GiB → bytes). Documented Colibri-style reserve.
DEFAULT_DEVICE_RESERVE_BYTES = 6 * 1024**3


@dataclass
class TierPlan:
    """Predicted placement for hierarchical expert staging."""

    device_expert_gb: float
    ram_expert_gb: float
    disk_expert_gb: float
    num_moe_layers: int
    num_local_experts: int
    slots_per_layer: int
    expert_row_bytes: int
    bottleneck: str
    policy: str
    disk_path: str | None
    full_residency: bool = False
    batch_union_floor: int = 0

    def summary(self) -> str:
        lines = [
            "Hierarchical expert tier plan:",
            f"  policy={self.policy}",
            f"  moe_layers={self.num_moe_layers} "
            f"local_experts={self.num_local_experts} "
            f"slots/layer={self.slots_per_layer}",
            f"  full_residency={'yes' if self.full_residency else 'no'}",
            f"  batch_union_floor={self.batch_union_floor}",
            f"  expert_row={self.expert_row_bytes / 1e6:.2f} MB",
            f"  device_slots={self.device_expert_gb:.3f} GiB",
            f"  ram_cache={self.ram_expert_gb:.3f} GiB",
            f"  disk_backing={self.disk_expert_gb:.3f} GiB "
            f"path={self.disk_path!r}",
            f"  predicted_bottleneck={self.bottleneck}",
        ]
        return "\n".join(lines)


def _mem_available_bytes() -> int:
    """Best-effort MemAvailable (Linux) else a conservative fallback."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size * 0.25)
    except (ValueError, OSError, AttributeError):
        return 8 * 1024**3


def resolve_ram_budget_bytes(cfg: HierarchicalOffloadConfig) -> int:
    """Resolve pinned RAM budget in bytes from config."""
    if cfg.tier_ram_gb == 0:
        return 0
    if cfg.tier_ram_gb > 0:
        return int(cfg.tier_ram_gb * 1024**3)
    # auto: take up to 50% of MemAvailable with a 2 GiB OS reserve.
    available = max(0, _mem_available_bytes() - 2 * 1024**3)
    return int(available * 0.5)


def batch_union_floor(
    *,
    num_local_experts: int,
    top_k: int,
    estimated_unique: int | None = None,
) -> int:
    """Minimum slots so a typical batch-union of experts can stay resident."""
    if num_local_experts <= 0:
        return 0
    est = estimated_unique if estimated_unique is not None else max(top_k * 2, top_k)
    return min(num_local_experts, max(top_k, est, 1))


def compute_slots_per_layer(
    cfg: HierarchicalOffloadConfig,
    *,
    num_moe_layers: int,
    num_local_experts: int,
    expert_row_bytes: int,
    top_k: int = 8,
    free_device_bytes: int | None = None,
    device_reserve_bytes: int = DEFAULT_DEVICE_RESERVE_BYTES,
    estimated_unique: int | None = None,
) -> int:
    """Derive device slot count per MoE layer.

    - ``tier_num_slots > 0``: honor and clamp to ``num_local_experts``.
    - ``tier_num_slots == 0``: derive from ``tier_device_expert_gb``, else from
      free device memory after ``device_reserve_bytes`` (dense + KV headroom).
    - Always apply a batch-union lower bound
      ``min(E, max(top_k, estimated_unique))``.
    """
    e = max(num_local_experts, 0)
    if e == 0:
        return 0

    floor = batch_union_floor(
        num_local_experts=e, top_k=top_k, estimated_unique=estimated_unique
    )

    if cfg.tier_num_slots > 0:
        slots = min(cfg.tier_num_slots, e)
    elif num_moe_layers <= 0 or expert_row_bytes <= 0:
        slots = min(max(top_k * 4, 16), e)
    elif cfg.tier_device_expert_gb > 0:
        budget = int(cfg.tier_device_expert_gb * 1024**3)
        per_layer = max(1, budget // max(num_moe_layers, 1))
        slots = max(1, per_layer // expert_row_bytes)
        slots = min(slots, e)
    elif free_device_bytes is not None and free_device_bytes > 0:
        usable = max(0, free_device_bytes - device_reserve_bytes)
        per_layer = usable // max(num_moe_layers, 1)
        slots = max(1, per_layer // max(expert_row_bytes, 1))
        slots = min(slots, e)
        logger.info(
            "Auto slots from free device mem: free=%.2f GiB reserve=%.2f GiB "
            "→ slots/layer=%d (floor=%d, E=%d)",
            free_device_bytes / 1024**3,
            device_reserve_bytes / 1024**3,
            slots,
            floor,
            e,
        )
    else:
        # Heuristic when backend forced on without explicit budget / free mem:
        # enough slots for a few batches of unique top-k experts.
        slots = min(max(top_k * 4, 32), e)

    if slots < floor:
        logger.info(
            "Raising slots/layer %d → %d to satisfy batch-union floor "
            "(top_k=%d estimated_unique=%s E=%d)",
            slots,
            floor,
            top_k,
            estimated_unique,
            e,
        )
        slots = floor

    return max(1, min(slots, e))


def build_tier_plan(
    cfg: HierarchicalOffloadConfig,
    *,
    num_moe_layers: int,
    num_local_experts: int,
    expert_row_bytes: int,
    top_k: int = 8,
    free_device_bytes: int | None = None,
    device_reserve_bytes: int = DEFAULT_DEVICE_RESERVE_BYTES,
    estimated_unique: int | None = None,
) -> TierPlan:
    """Build and return a tier placement plan."""
    floor = batch_union_floor(
        num_local_experts=num_local_experts,
        top_k=top_k,
        estimated_unique=estimated_unique,
    )
    slots = compute_slots_per_layer(
        cfg,
        num_moe_layers=num_moe_layers,
        num_local_experts=num_local_experts,
        expert_row_bytes=expert_row_bytes,
        top_k=top_k,
        free_device_bytes=free_device_bytes,
        device_reserve_bytes=device_reserve_bytes,
        estimated_unique=estimated_unique,
    )
    device_bytes = slots * expert_row_bytes * max(num_moe_layers, 1)
    total_expert_bytes = (
        num_local_experts * expert_row_bytes * max(num_moe_layers, 1)
    )
    ram_budget = resolve_ram_budget_bytes(cfg)
    ram_bytes = min(ram_budget, total_expert_bytes)
    disk_bytes = max(0, total_expert_bytes - ram_bytes)
    full = slots >= num_local_experts > 0

    if disk_bytes > 0 and cfg.tier_disk_path is None:
        bottleneck = "disk_required_but_unset"
    elif disk_bytes > 0:
        bottleneck = "nvme"
    elif slots < num_local_experts:
        bottleneck = "pcie_or_ram_hits"
    else:
        bottleneck = "none_full_residency"

    return TierPlan(
        device_expert_gb=device_bytes / 1024**3,
        ram_expert_gb=ram_bytes / 1024**3,
        disk_expert_gb=disk_bytes / 1024**3,
        num_moe_layers=num_moe_layers,
        num_local_experts=num_local_experts,
        slots_per_layer=slots,
        expert_row_bytes=expert_row_bytes,
        bottleneck=bottleneck,
        policy=cfg.tier_policy,
        disk_path=cfg.tier_disk_path,
        full_residency=full,
        batch_union_floor=floor,
    )


def format_tier_plan(plan: TierPlan) -> str:
    """Format a TierPlan for logging."""
    return plan.summary()


def log_tier_plan(plan: TierPlan) -> None:
    """Log the tier plan once at startup."""
    logger.info_once("%s", plan.summary())
    if plan.full_residency:
        logger.info_once(
            "Hierarchical full_residency=yes (slots=%d >= E=%d); "
            "Colibri-like PIN_GB=all analogue — prefer identity remap",
            plan.slots_per_layer,
            plan.num_local_experts,
        )
