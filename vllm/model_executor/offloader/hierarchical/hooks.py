# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoERunner integration hooks for hierarchical expert staging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager

if TYPE_CHECKING:
    from vllm.model_executor.offloader.hierarchical.manager import PendingEnsure


def maybe_schedule_ensure(
    layer_id: int,
    topk_ids: torch.Tensor,
) -> PendingEnsure | None:
    """Kick H2D / disk staging without waiting on the copy stream."""
    mgr = get_tier_manager()
    if mgr is None or not mgr._initialized:
        return None
    return mgr.schedule_ensure_and_remap(layer_id, topk_ids)


def maybe_wait_ensure(
    pending: PendingEnsure | None,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Block until scheduled staging is ready; attribute ``h2d_stall_ns``.

    Waits only on weight H2D events — activations are never synchronized here.
    """
    if pending is None:
        return topk_ids
    mgr = get_tier_manager()
    if mgr is None:
        return topk_ids
    return mgr.wait_ensure(pending)


def maybe_pilot_after_ensure(
    layer_id: int,
    hidden_states: torch.Tensor | None,
    topk_ids: torch.Tensor | None = None,
    *,
    pending: PendingEnsure | None = None,
) -> None:
    """Fire PILOT so next-layer DMA can overlap this layer's GEMM.

    Prefer ``pending.local_ids`` (already on host from schedule) over a fresh
    ``unique→tolist`` of ``topk_ids``.
    """
    if hidden_states is None:
        return
    mgr = get_tier_manager()
    if mgr is None or not mgr._initialized:
        return
    local_ids = pending.local_ids if pending is not None else None
    mgr.maybe_pilot_prefetch(
        layer_id,
        hidden_states,
        topk_ids,
        local_expert_ids=local_ids,
    )


def maybe_ensure_and_remap(
    layer_id: int,
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Schedule + wait ensure; optionally fire PILOT after weights are ready."""
    pending = maybe_schedule_ensure(layer_id, topk_ids)
    remapped = maybe_wait_ensure(pending, topk_ids)
    maybe_pilot_after_ensure(
        layer_id, hidden_states, topk_ids, pending=pending
    )
    return remapped


def register_routed_experts(layer_id: int, module) -> None:
    """Register a RoutedExperts module with the active tier manager."""
    mgr = get_tier_manager()
    if mgr is None:
        return
    mgr.register_moe_module(layer_id, module)


def register_moe_gate(layer_id: int, gate) -> None:
    """Register a MoE gate / router module for PILOT lookahead."""
    if gate is None:
        return
    mgr = get_tier_manager()
    if mgr is None:
        return
    mgr.register_gate(layer_id, gate)
