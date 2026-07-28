# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned + pageable RAM expert cache with LFRU eviction and learned pins."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import should_pin_memory

logger = init_logger(__name__)


@dataclass
class RamFrame:
    layer_id: int
    expert_id: int
    offset: int
    nbytes: int
    heat: float = 0.0
    last_access: float = 0.0
    soft_pinned: bool = False  # LFRU protect within arena
    valid: bool = False
    arena: str = "pinned"  # "pinned" | "pageable"


class _Arena:
    """One contiguous uint8 buffer (OS-pinned or pageable) + frame table."""

    def __init__(
        self,
        name: str,
        capacity_bytes: int,
        row_nbytes: int,
        *,
        pin_memory: bool,
        numa: bool = False,
    ):
        self.name = name
        self.row_nbytes = row_nbytes
        self.capacity_bytes = max(0, capacity_bytes)
        num_frames = (
            max(1, self.capacity_bytes // max(row_nbytes, 1))
            if row_nbytes and self.capacity_bytes > 0
            else 0
        )
        self.num_frames = num_frames
        self._arena: torch.Tensor | None = None
        if num_frames > 0 and row_nbytes > 0:
            from vllm.model_executor.offloader.hierarchical.numa_pin import (
                allocate_uint8_arena,
            )

            self._arena = allocate_uint8_arena(
                num_frames * row_nbytes,
                pin_memory=pin_memory,
                numa=numa,
            )
        self._frames: list[RamFrame] = [
            RamFrame(
                layer_id=-1,
                expert_id=-1,
                offset=i * row_nbytes,
                nbytes=row_nbytes,
                arena=name,
            )
            for i in range(num_frames)
        ]
        self._free: list[int] = list(range(num_frames))

    @property
    def enabled(self) -> bool:
        return self._arena is not None and self.num_frames > 0

    def view(self, frame: RamFrame) -> torch.Tensor:
        assert self._arena is not None
        return self._arena[frame.offset : frame.offset + frame.nbytes]

    def copy_in(self, frame: RamFrame, row: torch.Tensor) -> None:
        assert self._arena is not None
        self._arena[frame.offset : frame.offset + frame.nbytes].copy_(
            row.view(torch.uint8).reshape(-1)
        )


class PinnedExpertRamCache:
    """Hot experts in an OS-pinned arena; overflow in a pageable arena.

    Cap of OS-pinned bytes is ``pinned_capacity_bytes`` (from
    ``resolve_ram_budget_bytes``). Soft pins (LFRU) only apply inside the
    pinned arena so hot rows stay DMA-friendly.
    """

    def __init__(
        self,
        capacity_bytes: int,
        row_nbytes: int,
        *,
        pageable_capacity_bytes: int | None = None,
        device: torch.device | None = None,
        numa: bool = False,
    ):
        self.row_nbytes = row_nbytes
        pin = should_pin_memory()
        # Pageable overflow defaults to same size as pinned budget (or 1 row).
        pageable_bytes = (
            pageable_capacity_bytes
            if pageable_capacity_bytes is not None
            else max(capacity_bytes, row_nbytes)
        )
        self._pinned = _Arena(
            "pinned",
            capacity_bytes,
            row_nbytes,
            pin_memory=pin,
            numa=numa and pin,
        )
        self._pageable = _Arena(
            "pageable", pageable_bytes, row_nbytes, pin_memory=False, numa=False
        )
        self._index: dict[tuple[int, int], tuple[str, int]] = {}
        self._clock = 0.0
        if self._pinned.enabled or self._pageable.enabled:
            logger.info(
                "PinnedExpertRamCache: pinned=%d frames (%.3f GiB, pin=%s, "
                "numa=%s); pageable=%d frames (%.3f GiB)",
                self._pinned.num_frames,
                self._pinned.capacity_bytes / 1024**3,
                pin,
                numa and pin,
                self._pageable.num_frames,
                self._pageable.capacity_bytes / 1024**3,
            )

    @property
    def enabled(self) -> bool:
        return self._pinned.enabled or self._pageable.enabled

    @property
    def num_frames(self) -> int:
        return self._pinned.num_frames + self._pageable.num_frames

    @property
    def pinned_bytes_used(self) -> int:
        n = sum(1 for f in self._pinned._frames if f.valid)
        return n * self.row_nbytes

    @property
    def pinned_capacity_bytes(self) -> int:
        return self._pinned.capacity_bytes

    def _arena(self, name: str) -> _Arena:
        return self._pinned if name == "pinned" else self._pageable

    def _touch(self, frame: RamFrame) -> None:
        self._clock = time.monotonic()
        frame.last_access = self._clock
        frame.heat = frame.heat * 0.99 + 1.0

    def get(self, layer_id: int, expert_id: int) -> torch.Tensor | None:
        """Return a view of the cached row or None on miss."""
        key = (layer_id, expert_id)
        loc = self._index.get(key)
        if loc is None:
            return None
        name, idx = loc
        arena = self._arena(name)
        frame = arena._frames[idx]
        if not frame.valid or arena._arena is None:
            return None
        self._touch(frame)
        return arena.view(frame)

    def put(
        self,
        layer_id: int,
        expert_id: int,
        row: torch.Tensor,
        *,
        pinned: bool = False,
    ) -> torch.Tensor:
        """Insert/replace an expert row; prefer pinned arena when requested."""
        assert row.numel() == self.row_nbytes
        key = (layer_id, expert_id)
        if key in self._index:
            name, idx = self._index[key]
            arena = self._arena(name)
            frame = arena._frames[idx]
            arena.copy_in(frame, row)
            if pinned and name == "pinned":
                frame.soft_pinned = True
            elif pinned and name == "pageable" and self._pinned.enabled:
                # Promote hot row into pinned if possible.
                self._index.pop(key, None)
                frame.valid = False
                arena._free.append(idx)
                return self.put(layer_id, expert_id, row, pinned=True)
            frame.valid = True
            self._touch(frame)
            return arena.view(frame)

        prefer_pinned = pinned and self._pinned.enabled
        if prefer_pinned:
            try:
                return self._insert_into(
                    self._pinned, layer_id, expert_id, row, soft_pinned=True
                )
            except RuntimeError:
                pass
        if self._pageable.enabled:
            return self._insert_into(
                self._pageable, layer_id, expert_id, row, soft_pinned=False
            )
        if self._pinned.enabled:
            return self._insert_into(
                self._pinned, layer_id, expert_id, row, soft_pinned=pinned
            )
        raise RuntimeError("PinnedExpertRamCache has no arenas")

    def _insert_into(
        self,
        arena: _Arena,
        layer_id: int,
        expert_id: int,
        row: torch.Tensor,
        *,
        soft_pinned: bool,
    ) -> torch.Tensor:
        idx = self._alloc_frame(arena, soft_pinned=soft_pinned)
        frame = arena._frames[idx]
        old_key = (frame.layer_id, frame.expert_id)
        if frame.valid and old_key in self._index:
            loc = self._index.get(old_key)
            if loc == (arena.name, idx):
                del self._index[old_key]
        arena.copy_in(frame, row)
        frame.layer_id = layer_id
        frame.expert_id = expert_id
        frame.soft_pinned = soft_pinned and arena.name == "pinned"
        frame.valid = True
        frame.arena = arena.name
        self._index[(layer_id, expert_id)] = (arena.name, idx)
        self._touch(frame)
        return arena.view(frame)

    def pin(self, layer_id: int, expert_id: int) -> None:
        loc = self._index.get((layer_id, expert_id))
        if loc is None:
            return
        name, idx = loc
        if name != "pinned":
            return
        self._pinned._frames[idx].soft_pinned = True

    def _alloc_frame(self, arena: _Arena, *, soft_pinned: bool) -> int:
        if arena._free:
            return arena._free.pop()
        best_idx = -1
        best_score = float("inf")
        now = time.monotonic()
        for i, frame in enumerate(arena._frames):
            if frame.soft_pinned and arena.name == "pinned":
                continue
            age = max(1e-3, now - frame.last_access)
            score = frame.heat / age
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            if arena.name == "pageable":
                best_idx = min(
                    range(len(arena._frames)),
                    key=lambda i: arena._frames[i].heat,
                )
            else:
                raise RuntimeError("pinned arena exhausted (all soft-pinned)")
            arena._frames[best_idx].soft_pinned = False
        return best_idx

    def repin_hottest(
        self,
        layer_id: int,
        hot_experts: list[int],
        *,
        max_swaps: int = 4,
    ) -> int:
        """Live LFRU repin: soft-pin hot experts already in the pinned arena."""
        swaps = 0
        for e in hot_experts:
            if swaps >= max_swaps:
                break
            loc = self._index.get((layer_id, e))
            if loc is None:
                continue
            name, idx = loc
            if name != "pinned":
                continue
            frame = self._pinned._frames[idx]
            if not frame.soft_pinned:
                frame.soft_pinned = True
                swaps += 1
        return swaps
