# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional NUMA placement for pinned hierarchical RAM arenas.

When ``--tier-numa`` / ``VLLM_TIER_NUMA=1`` is set on a multi-node Linux host
with ``libnuma``, freshly allocated pinned expert frames are passed through
``numa_interleave_memory`` so pages are interleaved across NUMA nodes
(Colibri ``COLI_NUMA`` analogue). Unsupported platforms log once and no-op.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import should_pin_memory
from vllm.utils.numa_utils import get_libnuma

logger = init_logger(__name__)


@cache
def _numa_usable() -> bool:
    if os.name != "posix":
        return False
    if not os.path.isdir("/sys/devices/system/node/node1"):
        return False
    lib = get_libnuma()
    if lib is None:
        return False
    try:
        return int(lib.numa_available()) >= 0
    except Exception:
        return False


def allocate_uint8_arena(
    nbytes: int,
    *,
    pin_memory: bool | None = None,
    numa: bool = False,
) -> torch.Tensor:
    """Allocate a flat uint8 CPU buffer, optionally NUMA-interleaved."""
    if nbytes <= 0:
        return torch.empty(0, dtype=torch.uint8)
    pin = should_pin_memory() if pin_memory is None else pin_memory
    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
    if numa:
        apply_numa_interleave(buf)
    return buf


def apply_numa_interleave(buf: torch.Tensor) -> bool:
    """Best-effort ``numa_interleave_memory`` on ``buf``'s storage.

    Returns True when the call succeeded. Logs once and returns False when
    NUMA is unavailable.
    """
    if buf.numel() == 0 or buf.device.type != "cpu":
        return False
    if not _numa_usable():
        logger.info_once(
            "tier_numa requested but libnuma / multi-node NUMA is unavailable; "
            "pinned arena uses default OS placement"
        )
        return False
    lib = get_libnuma()
    assert lib is not None
    try:
        # numa_allocate_nodemask + numa_bitmask_setall + numa_interleave_memory
        nodemask = lib.numa_allocate_nodemask()
        if not nodemask:
            return False
        try:
            lib.numa_bitmask_setall(nodemask)
            ptr = ctypes.c_void_p(buf.data_ptr())
            size = ctypes.c_size_t(buf.numel() * buf.element_size())
            lib.numa_interleave_memory(ptr, size, nodemask)
        finally:
            lib.numa_bitmask_free(nodemask)
        logger.info_once(
            "tier_numa: applied numa_interleave_memory to pinned expert arena "
            "(%d bytes)",
            buf.numel() * buf.element_size(),
        )
        return True
    except Exception as e:
        logger.info_once(
            "tier_numa: numa_interleave_memory failed (%s); using default placement",
            e,
        )
        return False
