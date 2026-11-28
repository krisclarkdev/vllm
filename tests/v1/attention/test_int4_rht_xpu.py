# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for WS2 Hadamard / single_rht (DESIGN §1 + rht-xpu-r2)."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.fa_int4 import (
    finalize_fa_int4_output,
    prepare_fa_int4_query,
)
from vllm.v1.attention.ops.int4_per_token_head import (
    _butterfly_hadamard_transform,
    _dense_fp32_hadamard_transform,
    _get_rht_signs,
    _triton_hadamard_transform,
    fast_hadamard_transform,
    init_rht_signs,
    single_rht,
)

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
requires_xpu = pytest.mark.skipif(not _HAS_XPU, reason="XPU device required")


@pytest.mark.parametrize("d", [64, 128, 256])
def test_dense_fp32_hadamard_matches_butterfly(d: int) -> None:
    """Dense GEMM path must match tier-3 butterfly (fp32 accumulate)."""
    torch.manual_seed(0)
    x = torch.randn(7, 32, d, dtype=torch.float32)
    y_ref = _butterfly_hadamard_transform(x.clone())
    y = _dense_fp32_hadamard_transform(x.clone())
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_single_rht_keeps_dtype_vs_fp32_ref(
    dtype: torch.dtype, inverse: bool
) -> None:
    """Native dtype path stays close to legacy float cast (CPU butterfly)."""
    torch.manual_seed(1)
    d = 128
    x = torch.randn(4, 32, d, dtype=dtype)
    y_legacy = single_rht(x.float(), inverse=inverse).to(dtype)
    y_native = single_rht(x, inverse=inverse)
    assert y_native.dtype == dtype
    torch.testing.assert_close(
        y_native.float(), y_legacy.float(), atol=1e-3, rtol=1e-3
    )


def test_rht_sign_seed_round0() -> None:
    """DESIGN §1.1: seed 0x9E3779B9 + round*0x517CC1B7 with round=0."""
    d = 128
    signs = _get_rht_signs(d, 0, torch.device("cpu"))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x9E3779B9)
    expected = (
        2.0
        * torch.bernoulli(
            torch.full((d,), 0.5, dtype=torch.float32),
            generator=gen,
        )
        - 1.0
    )
    assert torch.equal(signs, expected)


def test_prepare_finalize_rht_norm_and_roundtrip() -> None:
    """prepare/finalize use rht_norm=b and invert the forward rotation."""
    torch.manual_seed(2)
    d = 128
    q = torch.randn(1, 32, d, dtype=torch.float16)
    scale = 0.1
    q_rot, scale_eff, rht_norm, q_dtype = prepare_fa_int4_query(q, scale)
    assert q_dtype == torch.float16
    assert rht_norm == 128.0
    assert scale_eff == pytest.approx(scale / 128.0)
    assert q_rot.dtype == torch.float16

    out = q_rot.clone()
    finalize_fa_int4_output(out, rht_norm, q_dtype)
    # H(D q) then D(H ·)/b recovers q under H² = b I and D² = I.
    torch.testing.assert_close(out.float(), q.float(), atol=2e-3, rtol=2e-3)


@requires_xpu
@pytest.mark.parametrize("d", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_hadamard_allclose_butterfly_xpu(
    d: int, dtype: torch.dtype
) -> None:
    """Round-2: Triton-XPU MMA allclose vs forced butterfly (atol/rtol 1e-3)."""
    torch.manual_seed(0)
    device = torch.device("xpu")
    x = torch.randn(5, 8, d, dtype=dtype, device=device)
    y_tri = _triton_hadamard_transform(x)
    y_ref = _butterfly_hadamard_transform(x.float()).to(dtype)
    torch.testing.assert_close(y_tri.float(), y_ref.float(), atol=1e-3, rtol=1e-3)
    assert y_tri.dtype == dtype


@requires_xpu
def test_fast_hadamard_selects_triton_on_xpu() -> None:
    """Dispatch prefers Triton when probe succeeded; else dense fallback."""
    import vllm.v1.attention.ops.int4_per_token_head as m

    device = torch.device("xpu")
    init_rht_signs(128, device)
    x = torch.randn(2, 16, 128, dtype=torch.float16, device=device)
    y = fast_hadamard_transform(x)
    assert y.dtype == torch.float16
    assert m._TRITON_HADAMARD_XPU_OK in (True, False)
    if m._TRITON_HADAMARD_XPU_OK:
        y_forced = _triton_hadamard_transform(x)
        torch.testing.assert_close(y, y_forced, atol=0, rtol=0)
