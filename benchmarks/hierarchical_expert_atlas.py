# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline expert-atlas probe tool (Colibri-style measured topic affinity).

Builds ``.vllm_expert_atlas.json`` by tagging hierarchical usage heat with
probe topics. Atlas affects **placement only** — never router outputs.

Examples:

  # Merge synthetic / pre-recorded counts (unit / CI)
  .venv/bin/python benchmarks/hierarchical_expert_atlas.py \\
    --merge-json /tmp/atlas_counts.json --output /tmp/.vllm_expert_atlas.json

  # Probe a live MoE (in-process so get_tier_manager() sees usage)
  VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python \\
    benchmarks/hierarchical_expert_atlas.py \\
    --model /tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ \\
    --probes benchmarks/hierarchical_atlas_probes.example.json.txt \\
    --tier-num-slots 4 --tier-ram-gb 8 \\
    --output /tmp/.vllm_expert_atlas.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probes",
        default=None,
        help="JSON probe config: {\"topics\": {\"code\": [\"prompt\", ...]}}",
    )
    p.add_argument(
        "--merge-json",
        default=None,
        help="Merge a counts JSON into an atlas without loading a model. "
        "Shape: {\"model\":..., \"topics\": {\"t\": {\"counts\": {\"0:1\": 3}}}} "
        "or flat {\"topic\": \"code\", \"counts\": {...}}",
    )
    p.add_argument("--model", default=None, help="MoE model path for live probes")
    p.add_argument("--output", required=True, help="Atlas JSON output path")
    p.add_argument("--tier-num-slots", type=int, default=4)
    p.add_argument("--tier-ram-gb", type=float, default=8.0)
    p.add_argument("--tier-disk-path", default=None)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=1)
    p.add_argument(
        "--keep-multiprocessing",
        action="store_true",
        help="Do not force VLLM_ENABLE_V1_MULTIPROCESSING=0",
    )
    return p.parse_args()


def _merge_offline(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.model_executor.offloader.hierarchical.atlas import ExpertAtlas

    raw = json.loads(Path(args.merge_json).read_text())
    atlas = ExpertAtlas()
    atlas.model = raw.get("model") or args.model
    if "topics" in raw:
        for topic, body in raw["topics"].items():
            atlas.merge_counts(
                topic,
                body.get("counts") or {},
                probe_prompts=int(body.get("probe_prompts") or 0),
            )
    elif "topic" in raw and "counts" in raw:
        atlas.merge_counts(
            str(raw["topic"]),
            raw["counts"],
            probe_prompts=int(raw.get("probe_prompts") or 0),
        )
    else:
        raise SystemExit(
            "--merge-json needs topics{} or {topic, counts} shape"
        )
    out = atlas.save(args.output)
    print(f"ATLAS_WRITTEN {out} topics={atlas.topics}")
    return atlas.to_dict()


def _snapshot_usage_counts(mgr) -> dict[str, int]:
    usage = getattr(mgr, "_usage", None)
    if usage is None:
        return {}
    return {
        f"{layer}:{expert}": count
        for (layer, expert), count in usage._counts.items()
    }


def _clear_usage(mgr) -> None:
    usage = getattr(mgr, "_usage", None)
    if usage is None:
        return
    usage._counts.clear()
    usage._dirty = False


def _run_probes(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model:
        raise SystemExit("--model is required with --probes")
    if not args.keep_multiprocessing:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    probes = json.loads(Path(args.probes).read_text())
    topics = probes.get("topics") or {}
    if not topics:
        raise SystemExit("probe config has empty topics")

    from vllm import LLM, SamplingParams
    from vllm.model_executor.offloader.hierarchical.atlas import ExpertAtlas
    from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager

    # Isolate usage so each topic snapshot is clean.
    usage_dir = tempfile.mkdtemp(prefix="vllm_atlas_usage_")
    usage_path = str(Path(usage_dir) / ".vllm_expert_usage")

    llm = LLM(
        model=args.model,
        offload_backend="hierarchical",
        tier_num_slots=args.tier_num_slots,
        tier_ram_gb=args.tier_ram_gb,
        tier_disk_path=args.tier_disk_path,
        tier_usage_path=usage_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    mgr = get_tier_manager()
    if mgr is None:
        raise SystemExit(
            "get_tier_manager() is None — use VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )

    atlas = ExpertAtlas()
    atlas.model = args.model
    for topic, prompts in topics.items():
        if isinstance(prompts, dict):
            prompts = prompts.get("prompts") or []
        prompts = list(prompts)
        _clear_usage(mgr)
        for prompt in prompts:
            llm.generate([str(prompt)], sampling)
        counts = _snapshot_usage_counts(mgr)
        atlas.merge_counts(str(topic), counts, probe_prompts=len(prompts))
        print(
            f"TOPIC {topic}: prompts={len(prompts)} "
            f"unique_experts={len(counts)}"
        )

    out = atlas.save(args.output)
    print(f"ATLAS_WRITTEN {out} topics={atlas.topics}")
    return atlas.to_dict()


def main() -> None:
    args = _parse_args()
    if args.merge_json:
        _merge_offline(args)
    elif args.probes:
        _run_probes(args)
    else:
        raise SystemExit("Provide --probes or --merge-json")


if __name__ == "__main__":
    main()
