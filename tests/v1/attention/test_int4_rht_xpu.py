# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for WS2 XPU dense Hadamard / single_rht (DESIGN §1)."""

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
    single_rht,
)


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
def test_single_rht_fp16_roundtrip_vs_fp32_ref(
    dtype: torch.dtype, inverse: bool
) -> None:
    """fp16/bf16 → fp32 WHT → dtype must match legacy float cast path."""
    torch.manual_seed(1)
    d = 128
    x = torch.randn(4, 32, d, dtype=dtype)
    # Legacy callers: single_rht(x.float()).to(dtype) before dtype preserve.
    y_legacy = single_rht(x.float(), inverse=inverse).to(dtype)
    y_native = single_rht(x, inverse=inverse)
    torch.testing.assert_close(y_native.float(), y_legacy.float(), atol=0, rtol=0)


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
