# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-2 H4: Triton-XPU Hadamard vs butterfly (DESIGN-rht-xpu-r2 pick A)."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.int4_per_token_head import (
    _butterfly_hadamard_transform,
    _triton_hadamard_transform,
    fast_hadamard_transform,
    init_rht_signs,
    single_rht,
)

pytest.importorskip("torch.xpu")

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
requires_xpu = pytest.mark.skipif(not _HAS_XPU, reason="XPU device required")


@requires_xpu
@pytest.mark.parametrize("d", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_hadamard_allclose_butterfly_xpu(
    d: int, dtype: torch.dtype
) -> None:
    """Tier-2 Triton MMA must match forced butterfly within 1e-3."""
    torch.manual_seed(0)
    device = torch.device("xpu")
    x = torch.randn(5, 8, d, dtype=dtype, device=device)
    y_tri = _triton_hadamard_transform(x)
    y_ref = _butterfly_hadamard_transform(x.float()).to(dtype)
    torch.testing.assert_close(y_tri.float(), y_ref.float(), atol=1e-3, rtol=1e-3)
    assert y_tri.dtype == dtype


@requires_xpu
@pytest.mark.parametrize("inverse", [False, True])
def test_single_rht_triton_dispatch_xpu(inverse: bool) -> None:
    """Dispatch selects Triton on XPU (or dense fallback); keeps dtype."""
    import vllm.v1.attention.ops.int4_per_token_head as m

    torch.manual_seed(1)
    device = torch.device("xpu")
    init_rht_signs(128, device)
    x = torch.randn(2, 16, 128, dtype=torch.float16, device=device)
    y = single_rht(x, inverse=inverse)
    assert y.dtype == torch.float16
    assert y.shape == x.shape
    # Probe flag set by init or first transform.
    assert m._TRITON_HADAMARD_XPU_OK in (True, False)
    y_disp = fast_hadamard_transform(x)
    assert y_disp.dtype == torch.float16
    if m._TRITON_HADAMARD_XPU_OK:
        y_forced = _triton_hadamard_transform(x)
        torch.testing.assert_close(y_disp, y_forced, atol=0, rtol=0)
