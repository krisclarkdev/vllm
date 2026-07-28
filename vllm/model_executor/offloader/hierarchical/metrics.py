# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for hierarchical expert staging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TierStats:
    """In-process counters for hierarchical staging."""

    device_hits: int = 0
    device_misses: int = 0
    ram_hits: int = 0
    ram_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    h2d_bytes: int = 0
    h2d_stall_ns: int = 0
    disk_bytes: int = 0
    disk_wait_ns: int = 0
    unique_experts_sum: int = 0
    ensure_calls: int = 0
    disk_direct_fallback: int = 0
    pilot_predict_hits: int = 0
    pilot_predict_misses: int = 0
    # Optional coarse histogram: unique-expert count → occurrences.
    unique_experts_hist: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, int | float | dict[str, int]]:
        device_lookups = self.device_hits + self.device_misses
        ram_lookups = self.ram_hits + self.ram_misses
        disk_lookups = self.disk_hits + self.disk_misses
        pilot_lookups = self.pilot_predict_hits + self.pilot_predict_misses
        return {
            "device_hits": self.device_hits,
            "device_misses": self.device_misses,
            "ram_hits": self.ram_hits,
            "ram_misses": self.ram_misses,
            "disk_hits": self.disk_hits,
            "disk_misses": self.disk_misses,
            "h2d_bytes": self.h2d_bytes,
            "h2d_stall_ns": self.h2d_stall_ns,
            "h2d_stall_ms": self.h2d_stall_ns / 1e6,
            "disk_bytes": self.disk_bytes,
            "disk_wait_ns": self.disk_wait_ns,
            "disk_wait_ms": self.disk_wait_ns / 1e6,
            "unique_experts_sum": self.unique_experts_sum,
            "unique_experts_hist": {
                str(k): v for k, v in sorted(self.unique_experts_hist.items())
            },
            "ensure_calls": self.ensure_calls,
            "disk_direct_fallback": self.disk_direct_fallback,
            "pilot_predict_hits": self.pilot_predict_hits,
            "pilot_predict_misses": self.pilot_predict_misses,
            "pilot_predict_hit_rate": self.pilot_predict_hits
            / max(1, pilot_lookups),
            "device_hit_rate": self.device_hits / max(1, device_lookups),
            "ram_hit_rate": self.ram_hits / max(1, ram_lookups),
            "disk_hit_rate": self.disk_hits / max(1, disk_lookups),
        }

    def reset(self) -> None:
        """Zero all counters (e.g. between bakeoff warm and measure)."""
        for f in asdict(self):
            if f == "unique_experts_hist":
                self.unique_experts_hist.clear()
            else:
                setattr(self, f, 0)


_PROM_REGISTERED = False
_prom_counters: dict[str, object] = {}


def _ensure_prometheus() -> None:
    global _PROM_REGISTERED
    if _PROM_REGISTERED:
        return
    try:
        from prometheus_client import Counter, Gauge

        _prom_counters["hits"] = Counter(
            "vllm_tier_expert_hits_total",
            "Hierarchical expert staging hits by tier",
            ["tier"],
        )
        _prom_counters["misses"] = Counter(
            "vllm_tier_expert_misses_total",
            "Hierarchical expert staging misses by tier",
            ["tier"],
        )
        _prom_counters["h2d_bytes"] = Counter(
            "vllm_tier_expert_h2d_bytes_total",
            "Bytes DMA'd into device expert slots",
        )
        _prom_counters["h2d_stall_seconds"] = Counter(
            "vllm_tier_expert_h2d_stall_seconds_total",
            "Seconds stalled waiting for expert H2D DMA",
        )
        _prom_counters["disk_bytes"] = Counter(
            "vllm_tier_expert_disk_bytes_total",
            "Bytes read from ExpertStore disk tier",
        )
        _prom_counters["disk_wait_seconds"] = Counter(
            "vllm_tier_expert_disk_wait_seconds_total",
            "Seconds waiting on ExpertStore disk reads",
        )
        _prom_counters["ensure_calls"] = Counter(
            "vllm_tier_expert_ensure_calls_total",
            "Calls to hierarchical ensure_layer",
        )
        _prom_counters["disk_direct_fallback"] = Counter(
            "vllm_tier_expert_disk_direct_fallback_total",
            "ExpertStore O_DIRECT reads that fell back to buffered I/O",
        )
        _prom_counters["pilot_predict_hits"] = Counter(
            "vllm_tier_expert_pilot_predict_hits_total",
            "PILOT predictions that matched the next layer's real topk",
        )
        _prom_counters["pilot_predict_misses"] = Counter(
            "vllm_tier_expert_pilot_predict_misses_total",
            "PILOT predictions that missed the next layer's real topk",
        )
        _prom_counters["hit_rate"] = Gauge(
            "vllm_tier_expert_device_hit_rate",
            "Device-tier hit rate for hierarchical expert staging",
        )
    except Exception:
        pass
    _PROM_REGISTERED = True


def record_stats(stats: TierStats) -> None:
    """Push current stats into Prometheus if available."""
    _ensure_prometheus()
    snap = stats.snapshot()
    gauge = _prom_counters.get("hit_rate")
    if gauge is not None:
        try:
            gauge.set(snap["device_hit_rate"])  # type: ignore[attr-defined]
        except Exception:
            pass


def increment_prom(
    tier: str | None = None,
    *,
    hit: bool | None = None,
    h2d_bytes: int = 0,
    h2d_stall_ns: int = 0,
    disk_bytes: int = 0,
    disk_wait_ns: int = 0,
    ensure_call: bool = False,
    disk_direct_fallback: int = 0,
    pilot_hit: bool | None = None,
) -> None:
    """Increment prometheus counters for a staging event."""
    _ensure_prometheus()
    if tier is not None and hit is not None:
        key = "hits" if hit else "misses"
        counter = _prom_counters.get(key)
        if counter is not None:
            try:
                counter.labels(tier=tier).inc()  # type: ignore[attr-defined]
            except Exception:
                pass
    if h2d_bytes and (c := _prom_counters.get("h2d_bytes")) is not None:
        try:
            c.inc(h2d_bytes)  # type: ignore[attr-defined]
        except Exception:
            pass
    if h2d_stall_ns and (c := _prom_counters.get("h2d_stall_seconds")) is not None:
        try:
            c.inc(h2d_stall_ns / 1e9)  # type: ignore[attr-defined]
        except Exception:
            pass
    if disk_bytes and (c := _prom_counters.get("disk_bytes")) is not None:
        try:
            c.inc(disk_bytes)  # type: ignore[attr-defined]
        except Exception:
            pass
    if disk_wait_ns and (c := _prom_counters.get("disk_wait_seconds")) is not None:
        try:
            c.inc(disk_wait_ns / 1e9)  # type: ignore[attr-defined]
        except Exception:
            pass
    if ensure_call and (c := _prom_counters.get("ensure_calls")) is not None:
        try:
            c.inc()  # type: ignore[attr-defined]
        except Exception:
            pass
    if disk_direct_fallback and (
        c := _prom_counters.get("disk_direct_fallback")
    ) is not None:
        try:
            c.inc(disk_direct_fallback)  # type: ignore[attr-defined]
        except Exception:
            pass
    if pilot_hit is not None:
        key = "pilot_predict_hits" if pilot_hit else "pilot_predict_misses"
        if (c := _prom_counters.get(key)) is not None:
            try:
                c.inc()  # type: ignore[attr-defined]
            except Exception:
                pass
