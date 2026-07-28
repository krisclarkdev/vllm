# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark harness: hierarchical expert staging vs baseline (Colibri bakeoff aid).

Loads once with hierarchical offload, warms W steps (discarded), then measures
decode tok/s (and optional prefill timing) over N prompts. TierStats are
collected in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0 by default so
``get_tier_manager().stats`` is visible to this process).

Hal defaults (document Mixtral-8x7B AWQ; Ornith uses the same path when present):

  # Hierarchical measure
  VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python \\
    benchmarks/hierarchical_tier_bakeoff.py \\
    --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \\
    --tier-num-slots 4 --tier-ram-gb 8 --warm 4 --max-tokens 32 \\
    --output /tmp/hier_pr_a_bakeoff.json

  # Optional baseline in a separate invocation, then compare:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python \\
    benchmarks/hierarchical_tier_bakeoff.py \\
    --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \\
    --baseline --warm 4 --max-tokens 32 \\
    --output /tmp/hier_pr_a_baseline.json

  .venv/bin/python benchmarks/hierarchical_tier_bakeoff.py --compare \\
    --hierarchical-json /tmp/hier_pr_a_bakeoff.json \\
    --baseline-json /tmp/hier_pr_a_baseline.json \\
    --output /tmp/hier_pr_a_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any


DEFAULT_HAL_MODEL = "/tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=DEFAULT_HAL_MODEL,
        help=f"Model path (default: {DEFAULT_HAL_MODEL})",
    )
    p.add_argument("--tier-num-slots", type=int, default=4)
    p.add_argument("--tier-ram-gb", type=float, default=8.0)
    p.add_argument("--tier-disk-path", default=None)
    p.add_argument("--tier-pilot", action="store_true")
    p.add_argument("--tier-atlas-path", default=None)
    p.add_argument("--tier-affinity-topic", default=None)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument(
        "--prompt",
        default="Hi",
        help="Prompt text (default short 'Hi' so slots=4 smoke stays within "
        "unique-expert budget; longer prompts may need more slots)",
    )
    p.add_argument("--warm", type=int, default=4, help="Warmup generate steps")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Keep small so unique experts per step fit in --tier-num-slots",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=1,
        help="1 = one token/step (Mixtral top-2 fits in 4 slots); raise for throughput",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Run without hierarchical offload (separate invocation)",
    )
    p.add_argument(
        "--measure-prefill",
        action="store_true",
        help="Also report a coarse prefill-only proxy (1 token)",
    )
    p.add_argument(
        "--colibri-tok-s",
        type=float,
        default=None,
        help="Optional Colibri baseline tok/s for the same model",
    )
    p.add_argument(
        "--keep-multiprocessing",
        action="store_true",
        help="Do not force VLLM_ENABLE_V1_MULTIPROCESSING=0 "
        "(tier_stats may be empty in the driver)",
    )
    p.add_argument("--output", default=None)
    # Compare mode
    p.add_argument(
        "--compare",
        action="store_true",
        help="Merge hierarchical + baseline JSON into a comparison file",
    )
    p.add_argument("--hierarchical-json", default=None)
    p.add_argument("--baseline-json", default=None)
    return p.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _compare(args: argparse.Namespace) -> None:
    if not args.hierarchical_json or not args.baseline_json:
        raise SystemExit("--compare requires --hierarchical-json and --baseline-json")
    hier = _load_json(args.hierarchical_json)
    base = _load_json(args.baseline_json)
    h_tok = float(hier.get("tok_s_warm") or 0.0)
    b_tok = float(base.get("tok_s_warm") or 0.0)
    result = {
        "mode": "compare",
        "hierarchical": hier,
        "baseline": base,
        "tok_s_warm_hierarchical": h_tok,
        "tok_s_warm_baseline": b_tok,
        "speedup_vs_baseline": (h_tok / b_tok) if b_tok > 0 else None,
        "colibri_tok_s": hier.get("colibri_tok_s") or base.get("colibri_tok_s"),
        "speedup_vs_colibri": (
            h_tok / float(hier["colibri_tok_s"])
            if hier.get("colibri_tok_s")
            else None
        ),
        "note": "metrics/bakeoff only; no claimed win until later PRs",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)


def _run_bakeoff(args: argparse.Namespace) -> None:
    # In-process engine so get_tier_manager() sees TierStats after generate.
    if not args.keep_multiprocessing:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams
    from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager

    base = args.prompt.strip() or "Hi"
    prompts = [
        base if args.num_prompts == 1 else f"{base} #{i}."
        for i in range(max(args.num_prompts, 1))
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    warm_sampling = SamplingParams(temperature=0.0, max_tokens=min(8, args.max_tokens))

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": True,
        "trust_remote_code": True,
    }
    if not args.baseline:
        llm_kwargs.update(
            {
                "offload_backend": "hierarchical",
                "tier_num_slots": args.tier_num_slots,
                "tier_ram_gb": args.tier_ram_gb,
                "tier_disk_path": args.tier_disk_path,
                "tier_pilot": args.tier_pilot,
                "tier_atlas_path": args.tier_atlas_path,
                "tier_affinity_topic": args.tier_affinity_topic,
            }
        )

    llm = LLM(**llm_kwargs)

    # Warm W steps (discard)
    for i in range(max(args.warm, 0)):
        llm.generate(prompts[:1], warm_sampling)

    mgr = get_tier_manager()
    if mgr is not None:
        mgr.stats.reset()

    prefill_s: float | None = None
    if args.measure_prefill:
        prefill_sampling = SamplingParams(temperature=0.0, max_tokens=1)
        t_pf = time.perf_counter()
        llm.generate(prompts[:1], prefill_sampling)
        prefill_s = time.perf_counter() - t_pf
        if mgr is not None:
            # Prefill is optional; keep measure window clean for decode stats.
            mgr.stats.reset()

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_s = total_tokens / max(elapsed, 1e-6)
    ttft_proxy = elapsed / max(len(prompts), 1)

    tier_stats: dict[str, Any] = {}
    tier_stats_note: str | None = None
    if args.baseline:
        tier_stats_note = "baseline_run_no_hierarchical"
    elif mgr is not None:
        tier_stats = mgr.stats.snapshot()
        if not tier_stats.get("ensure_calls"):
            tier_stats_note = (
                "manager_present_but_ensure_calls_zero_"
                "check_moe_hook_or_slots_full_residency"
            )
    else:
        tier_stats_note = (
            "get_tier_manager_none_"
            "set_VLLM_ENABLE_V1_MULTIPROCESSING=0_or_omit_--keep-multiprocessing"
        )

    result: dict[str, Any] = {
        "mode": "baseline" if args.baseline else "hierarchical",
        "model": args.model,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
        "tok_s_warm": tok_s,
        "ttft_proxy_s": ttft_proxy,
        "prefill_proxy_s": prefill_s,
        "warm_steps": args.warm,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "tier_stats": tier_stats,
        "tier_stats_note": tier_stats_note,
        "colibri_tok_s": args.colibri_tok_s,
        "speedup_vs_colibri": (
            tok_s / args.colibri_tok_s if args.colibri_tok_s else None
        ),
        "config": {
            "offload_backend": None if args.baseline else "hierarchical",
            "tier_num_slots": None if args.baseline else args.tier_num_slots,
            "tier_ram_gb": None if args.baseline else args.tier_ram_gb,
            "tier_disk_path": None if args.baseline else args.tier_disk_path,
            "tier_pilot": None if args.baseline else args.tier_pilot,
            "tier_atlas_path": None if args.baseline else args.tier_atlas_path,
            "tier_affinity_topic": (
                None if args.baseline else args.tier_affinity_topic
            ),
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prompt": args.prompt,
            "v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING", "1"
            ),
        },
        "note": "PR-A metrics spine; no claimed tok/s win yet",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)


def main() -> None:
    args = _parse_args()
    if args.compare:
        _compare(args)
    else:
        _run_bakeoff(args)


if __name__ == "__main__":
    main()
