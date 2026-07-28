# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert atlas: measured per-topic routing heat for affinity pins.

Placement only — never changes router outputs or model weights. Affinity is
measured routing (Colibri-style), not a learned embedding.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ATLAS_VERSION = 1


class ExpertAtlas:
    """Topic-tagged per-(layer, expert) hit counts for cold-start pin boosts.

    Schema (``.vllm_expert_atlas.json``)::

        {
          "version": 1,
          "model": "<optional>",
          "topics": {
            "code": {
              "counts": {"0:3": 12, "1:5": 4},
              "probe_prompts": 4
            }
          }
        }
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.model: str | None = None
        # topic -> (layer, expert) -> count
        self._topics: dict[str, dict[tuple[int, int], int]] = {}
        self._topic_meta: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            self.load(self.path)

    @property
    def topics(self) -> list[str]:
        return sorted(self._topics)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        data = json.loads(path.read_text())
        self.path = path
        self.model = data.get("model")
        self._topics.clear()
        self._topic_meta.clear()
        for topic, body in (data.get("topics") or {}).items():
            counts: dict[tuple[int, int], int] = {}
            for key, count in (body.get("counts") or {}).items():
                layer_s, expert_s = str(key).split(":")
                counts[(int(layer_s), int(expert_s))] = int(count)
            self._topics[str(topic)] = counts
            meta = {k: v for k, v in body.items() if k != "counts"}
            if meta:
                self._topic_meta[str(topic)] = meta
        logger.info(
            "Loaded expert atlas from %s (%d topics, %d entries)",
            path,
            len(self._topics),
            sum(len(c) for c in self._topics.values()),
        )

    def to_dict(self) -> dict[str, Any]:
        topics_out: dict[str, Any] = {}
        for topic, counts in self._topics.items():
            body: dict[str, Any] = {
                "counts": {
                    f"{layer}:{expert}": count
                    for (layer, expert), count in sorted(counts.items())
                }
            }
            body.update(self._topic_meta.get(topic, {}))
            topics_out[topic] = body
        return {
            "version": ATLAS_VERSION,
            "model": self.model,
            "topics": topics_out,
        }

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.path
        if out is None:
            raise ValueError("atlas save requires a path")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, out)
        self.path = out
        return out

    def record(
        self,
        topic: str,
        layer_id: int,
        expert_ids: list[int],
        *,
        weight: int = 1,
    ) -> None:
        bucket = self._topics.setdefault(topic, {})
        for e in expert_ids:
            if e < 0:
                continue
            bucket[(layer_id, e)] = bucket.get((layer_id, e), 0) + weight

    def merge_counts(
        self,
        topic: str,
        counts: dict[str, int] | dict[tuple[int, int], int],
        *,
        probe_prompts: int = 0,
    ) -> None:
        """Merge string-keyed ``\"layer:expert\"`` or tuple counts into topic."""
        bucket = self._topics.setdefault(topic, {})
        for key, count in counts.items():
            if isinstance(key, tuple):
                layer_id, expert_id = int(key[0]), int(key[1])
            else:
                layer_s, expert_s = str(key).split(":")
                layer_id, expert_id = int(layer_s), int(expert_s)
            bucket[(layer_id, expert_id)] = bucket.get(
                (layer_id, expert_id), 0
            ) + int(count)
        meta = self._topic_meta.setdefault(topic, {})
        meta["probe_prompts"] = int(meta.get("probe_prompts", 0)) + int(
            probe_prompts
        )

    def hottest(
        self,
        topic: str,
        layer_id: int,
        limit: int,
        num_experts: int,
    ) -> list[int]:
        """Return up to ``limit`` hottest experts for ``topic`` at ``layer_id``."""
        counts = self._topics.get(topic)
        if not counts:
            return []
        scored = [
            (counts.get((layer_id, e), 0), e) for e in range(num_experts)
        ]
        scored.sort(reverse=True)
        return [e for count, e in scored[:limit] if count > 0]

    def boost_scores(
        self,
        topic: str,
        layer_id: int,
        num_experts: int,
        *,
        base: dict[int, float] | None = None,
        weight: float = 1.0,
    ) -> dict[int, float]:
        """Return expert→score map: ``base + weight * atlas_count``."""
        out: dict[int, float] = defaultdict(float)
        if base:
            for e, s in base.items():
                out[int(e)] = float(s)
        counts = self._topics.get(topic) or {}
        for e in range(num_experts):
            c = counts.get((layer_id, e), 0)
            if c:
                out[e] = out.get(e, 0.0) + weight * float(c)
        return dict(out)

    def affinity_hottest(
        self,
        topic: str,
        layer_id: int,
        limit: int,
        num_experts: int,
        *,
        usage_counts: dict[tuple[int, int], int] | None = None,
        atlas_weight: float = 10.0,
    ) -> list[int]:
        """Blend usage heat with atlas topic counts; atlas-weighted for cold start."""
        base: dict[int, float] = {}
        if usage_counts:
            for e in range(num_experts):
                c = usage_counts.get((layer_id, e), 0)
                if c:
                    base[e] = float(c)
        scores = self.boost_scores(
            topic, layer_id, num_experts, base=base, weight=atlas_weight
        )
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [e for e, s in ranked[:limit] if s > 0]


def default_atlas_path(disk_path: str | None, model_path: str | None) -> str | None:
    """Resolve default atlas sidecar path (next to usage when possible)."""
    if disk_path:
        return str(Path(disk_path) / ".vllm_expert_atlas.json")
    if model_path:
        return str(Path(model_path) / ".vllm_expert_atlas.json")
    return None


def load_atlas_or_none(path: str | None) -> ExpertAtlas | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(
            "tier_atlas_path set but missing: %s (falling back to usage)", path
        )
        return None
    try:
        return ExpertAtlas(p)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to load expert atlas from %s: %s", path, e)
        return None
