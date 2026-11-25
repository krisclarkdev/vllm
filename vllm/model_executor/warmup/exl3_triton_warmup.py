# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Soft-import warmup for exl3xpu Triton fallthrough kernels.

No-op when the EXL3 XPU plugin is absent. Covers `_had_in_kernel`,
`_had_out_kernel`, and `_gemm_kernel` compile keys (M→BM, K, CB, S) that
live traffic can hit when ESIMD rejects a shape or large-M falls to Triton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# BM thresholds: had_in/out use 16/32; gemm_small uses 16/32/64.
_DEFAULT_M = (1, 2, 4, 8, 16, 32, 64, 128)
# SC + plain Muse bit-widths observed in prod overlays.
_DEFAULT_K = (3, 4, 5, 6)
# mcg=1 / mul1=2 as used by exl3xpu.
_DEFAULT_CB = (1, 2)
# Single linear + fused qkv shard groups.
_DEFAULT_S = (1, 3)

# Tiny legal dims: k/n multiples of 128 (Hadamard + BN=128).
_WARM_K_DIM = 128
_WARM_N_DIM = 128


def _sync_device(device: torch.device) -> None:
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _collect_exl3_k_cb(worker: "Worker") -> set[tuple[int, int]]:
    """Prefer K/cb from loaded EXL3 modules; empty if none found."""
    found: set[tuple[int, int]] = set()

    def _add(k: object, cb: object) -> None:
        try:
            ki, cbi = int(k), int(cb)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if ki > 0 and cbi > 0:
            found.add((ki, cbi))

    contexts: list[object] = []
    runner = getattr(worker, "model_runner", None)
    if runner is not None:
        vllm_config = getattr(runner, "vllm_config", None)
        compilation = getattr(vllm_config, "compilation_config", None)
        sfc = getattr(compilation, "static_forward_context", None)
        if isinstance(sfc, dict):
            contexts.append(sfc)
        try:
            model = runner.get_model()
        except Exception:
            model = None
        if model is not None:
            contexts.append(model)

    for ctx in contexts:
        modules = ctx.values() if isinstance(ctx, dict) else ctx.modules()
        for module in modules:
            _add(getattr(module, "exl3_K", None), getattr(module, "exl3_cb", None))
            qm = getattr(module, "quant_method", None)
            if qm is not None:
                _add(getattr(qm, "K", None), getattr(qm, "cb", None))
    return found


def _warm_one(
    tk: object,
    *,
    device: torch.device,
    M: int,
    K: int,
    cb: int,
    S: int,
) -> None:
    k = _WARM_K_DIM
    n = _WARM_N_DIM
    x = torch.randn((M, k), dtype=torch.float16, device=device)
    suh = torch.randn((S, k), dtype=torch.float16, device=device)
    svh = torch.randn((n,), dtype=torch.float16, device=device)
    trellis = torch.randint(
        0,
        2**15,
        (k // 16, n // 16, 16 * K),
        dtype=torch.int16,
        device=device,
    )
    shard_of_nb = torch.zeros((n // 128,), dtype=torch.int32, device=device)

    xh = tk.had_in(x, suh)  # type: ignore[attr-defined]
    y32 = tk.gemm_small(xh, trellis, K, cb, shard_of_nb)  # type: ignore[attr-defined]
    out = torch.empty((M, n), dtype=torch.float16, device=device)
    tk.had_out(y32, svh, out)  # type: ignore[attr-defined]


@torch.inference_mode()
def exl3_triton_warmup(worker: "Worker") -> None:
    """Warm EXL3 Triton kernels; no-op if plugin import fails."""
    try:
        import exl3xpu.triton_kernels as tk
    except ImportError:
        return

    runner = getattr(worker, "model_runner", None)
    device = getattr(runner, "device", None)
    if device is None:
        return
    if getattr(device, "type", None) not in ("xpu", "cuda"):
        # Synthetic tensors need a real accelerator; skip CPU-only hosts.
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return
        device = torch.device("xpu")

    observed = _collect_exl3_k_cb(worker)
    k_values = sorted({k for k, _ in observed} | set(_DEFAULT_K))
    cb_values = sorted({cb for _, cb in observed} | set(_DEFAULT_CB))
    # If we observed concrete (K, cb) pairs, prefer those unioned with defaults
    # as a Cartesian product still — keeps the matrix small and complete.
    pairs = {(K, cb) for K in k_values for cb in cb_values}

    logger.info(
        "Warming EXL3 Triton kernels for M=%s K/cb=%s S=%s",
        list(_DEFAULT_M),
        sorted(pairs),
        list(_DEFAULT_S),
    )
    try:
        for M in _DEFAULT_M:
            for K, cb in sorted(pairs):
                for S in _DEFAULT_S:
                    _warm_one(tk, device=device, M=M, K=K, cb=cb, S=S)
        _sync_device(device)
    except Exception:
        logger.warning("Skipping EXL3 Triton warmup after failure.", exc_info=True)
