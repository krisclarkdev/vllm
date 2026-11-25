# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up v1 block-table Triton kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_SLOT_MAPPING_WARMUP_TOKENS = 8
_SLOT_MAPPING_WARMUP_BLOCK_SIZES = (3, 16)
_SLOT_MAPPING_WARMUP_CP_KV_CACHE_INTERLEAVE_SIZE = 1


def _sync_device(device: torch.device) -> None:
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _warm_slot_mapping(
    block_table: BlockTable,
    num_reqs: int,
    num_tokens: int,
) -> None:
    if block_table.slot_mapping_mode == SlotMappingMode.NONE:
        return
    if num_reqs <= 0 or num_tokens <= 0:
        return
    if num_reqs > block_table.max_num_reqs:
        return
    if num_tokens > block_table.max_num_batched_tokens:
        return

    # Enough KV-manager blocks to cover positions (pre-hybrid ids for add_row).
    kernel_blocks_needed = max(
        1, (num_tokens + block_table.block_size - 1) // block_table.block_size
    )
    kv_cap = max(1, block_table.max_num_blocks_per_req // block_table.blocks_per_kv_block)
    kv_blocks = min(
        kv_cap,
        max(
            1,
            (kernel_blocks_needed + block_table.blocks_per_kv_block - 1)
            // block_table.blocks_per_kv_block,
        ),
    )
    for row in range(num_reqs):
        block_table.add_row(list(range(kv_blocks)), row)
    block_table.commit_block_table(num_reqs)

    # Evenly split tokens across requests (decode=1 / small prefill).
    tokens_per_req = [num_tokens // num_reqs] * num_reqs
    for i in range(num_tokens % num_reqs):
        tokens_per_req[i] += 1
    starts = [0]
    for n in tokens_per_req:
        starts.append(starts[-1] + n)
    query_start_loc = torch.tensor(
        starts, dtype=torch.int32, device=block_table.device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=block_table.device)
    block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    for row in range(num_reqs):
        block_table.clear_row(row)
    block_table.clear()


def _warm_from_live_block_tables(
    runner: "GPUModelRunner",
    device: torch.device,
) -> None:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    multi = getattr(input_batch, "block_table", None)
    if multi is None:
        return
    live_tables = getattr(multi, "block_tables", None)
    if not live_tables:
        return

    max_num_seqs = int(runner.scheduler_config.max_num_seqs)
    max_batched = int(runner.scheduler_config.max_num_batched_tokens)
    if max_num_seqs <= 0 or max_batched <= 0:
        return

    num_reqs_grid = sorted(
        {
            1,
            min(4, max_num_seqs),
            max_num_seqs,
        }
    )
    num_tokens_grid = sorted(
        {
            1,
            min(8, max_batched),
        }
    )

    # Use live tables so block_table_stride / DCP constexprs match production.
    # Warmup runs before serving; clear rows afterward to restore empty state.
    for src in live_tables:
        if src.slot_mapping_mode == SlotMappingMode.NONE:
            continue
        try:
            for num_reqs in num_reqs_grid:
                if num_reqs > src.max_num_reqs:
                    continue
                for num_tokens in num_tokens_grid:
                    if num_tokens > src.max_num_batched_tokens:
                        continue
                    _warm_slot_mapping(src, num_reqs, num_tokens)
        except Exception:
            logger.warning(
                "Skipping live block-table slot-mapping warmup for one group.",
                exc_info=True,
            )

    _sync_device(device)


def warm_v1_block_table_kernels(
    device: torch.device,
    max_tokens: int,
    runner: "GPUModelRunner | None" = None,
) -> None:
    # Baseline synthetic warm (hardcoded block sizes) — keep for coverage of
    # common compile keys even when live tables are not yet constructed.
    num_tokens = max(0, min(_SLOT_MAPPING_WARMUP_TOKENS, max_tokens))
    if num_tokens > 0:
        query_start_loc = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=device
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        for block_size in _SLOT_MAPPING_WARMUP_BLOCK_SIZES:
            max_num_blocks_per_req = max(
                1, (max(num_tokens, max_tokens) + block_size - 1) // block_size
            )
            max_num_blocks_per_req = ((max_num_blocks_per_req + 15) // 16) * 16
            block_table = BlockTable(
                block_size=block_size,
                max_num_reqs=1,
                max_num_blocks_per_req=max_num_blocks_per_req,
                max_num_batched_tokens=max(num_tokens, max_tokens),
                pin_memory=False,
                device=device,
                kernel_block_size=block_size,
                cp_kv_cache_interleave_size=(
                    _SLOT_MAPPING_WARMUP_CP_KV_CACHE_INTERLEAVE_SIZE
                ),
            )
            block_table.add_row(list(range(max_num_blocks_per_req)), 0)
            block_table.commit_block_table(1)
            block_table.compute_slot_mapping(1, query_start_loc, positions)

    # Additionally warm from live runner BlockTables (real block_size / strides /
    # DCP constexprs) across a small num_reqs × num_tokens grid.
    if runner is not None:
        _warm_from_live_block_tables(runner, device)

    _sync_device(device)
