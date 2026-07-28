# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-lookahead (PILOT) expert prefetch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.hierarchical.metrics import increment_prom

if TYPE_CHECKING:
    from vllm.model_executor.offloader.hierarchical.manager import ExpertTierManager

logger = init_logger(__name__)


class PilotPrefetcher:
    """Prefetch next-layer experts using a cheap routing heuristic.

    Colibri's PILOT applies layer L+1's gate to layer L's post-attention
    state. With ``tier_pilot`` alone we reuse the current layer's selected
    experts as a same-token affinity hint. With ``tier_pilot_real`` we also
    run the registered next-layer gate (extra gate forward cost) for a
    stronger lookahead.
    """

    def __init__(self, manager: ExpertTierManager, *, real: bool = False):
        self.manager = manager
        self.real = real
        self._gates: dict[int, torch.nn.Module] = {}
        # layer_id -> last predicted local expert ids (for hit/miss scoring).
        self._last_pred: dict[int, set[int]] = {}

    def register_gate(self, layer_id: int, gate: torch.nn.Module) -> None:
        self._gates[layer_id] = gate

    @torch.inference_mode()
    def prefetch_next(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        current_expert_ids: list[int],
    ) -> None:
        next_id = layer_id + 1
        if next_id not in self.manager.layers:
            return

        state = self.manager.layers[next_id]
        predicted = list(current_expert_ids)
        used_gate = False
        gate = self._gates.get(next_id)
        if self.real and gate is not None and hidden_states is not None:
            try:
                # Use last token only for a cheap lookahead.
                h = (
                    hidden_states[-1:]
                    if hidden_states.dim() >= 2
                    else hidden_states
                )
                logits = gate(h)
                if isinstance(logits, tuple):
                    logits = logits[0]
                topk = min(
                    int(getattr(state.module, "top_k", 8) or 8),
                    int(logits.shape[-1]),
                )
                _, idx = torch.topk(
                    logits.reshape(-1, logits.shape[-1]), topk, dim=-1
                )
                predicted = [int(x) for x in idx.reshape(-1).tolist()]
                used_gate = True
            except Exception as e:
                logger.debug("PILOT gate failed for layer %d: %s", next_id, e)

        local_pred: list[int] = []
        for eid in predicted:
            local = state.to_local(int(eid))
            if local >= 0:
                local_pred.append(local)
        self._last_pred[next_id] = set(local_pred)

        if used_gate:
            logger.debug(
                "PILOT real gate lookahead layer=%d -> %d experts",
                next_id,
                len(local_pred),
            )

        # Fire-and-forget: schedule H2D / disk without blocking the GEMM path.
        self.manager.prefetch_experts(next_id, local_pred, block=False)

    def score_prediction(self, layer_id: int, actual_expert_ids: list[int]) -> None:
        """Compare prior PILOT prediction for ``layer_id`` to real topk."""
        pred = self._last_pred.pop(layer_id, None)
        if pred is None:
            return
        actual = {int(e) for e in actual_expert_ids if int(e) >= 0}
        if not actual:
            return
        hits = len(pred & actual)
        misses = len(actual - pred)
        stats = self.manager.stats
        stats.pilot_predict_hits += hits
        stats.pilot_predict_misses += misses
        for _ in range(hits):
            increment_prom(pilot_hit=True)
        for _ in range(misses):
            increment_prom(pilot_hit=False)
