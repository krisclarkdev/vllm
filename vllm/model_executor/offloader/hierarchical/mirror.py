# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-NVMe ExpertStore mirror routing (Colibri-style)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from vllm.logger import init_logger
from vllm.model_executor.offloader.hierarchical.format import (
    MANIFEST_NAME,
    load_manifest,
)

logger = init_logger(__name__)


def parse_disk_weights(raw: str | None) -> tuple[float, float]:
    """Parse ``a,b`` bandwidth weights; default equal ``1,1``."""
    if not raw or not str(raw).strip():
        return (1.0, 1.0)
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"tier_disk_weights must be 'a,b' (got {raw!r})"
        )
    a, b = float(parts[0]), float(parts[1])
    if a < 0 or b < 0 or (a + b) <= 0:
        raise ValueError(f"tier_disk_weights must be non-negative with a+b>0: {raw!r}")
    return (a, b)


def volume_for_expert(
    layer_id: int,
    expert_id: int,
    weight_primary: float,
    weight_mirror: float,
) -> int:
    """Deterministic 0=primary / 1=mirror routing from weights.

    Uses blake2b so PILOT and demand always agree for the same
    ``(layer_id, expert_id)``.
    """
    total = weight_primary + weight_mirror
    if total <= 0 or weight_mirror <= 0:
        return 0
    if weight_primary <= 0:
        return 1
    digest = hashlib.blake2b(
        f"{layer_id}:{expert_id}".encode(),
        digest_size=8,
    ).digest()
    x = int.from_bytes(digest, "little") / float(1 << 64)
    thresh = weight_primary / total
    return 0 if x < thresh else 1


def probe_disk_weights(primary: Path, mirror: Path) -> tuple[float, float]:
    """Rough sequential-read probe; falls back to equal weights on failure."""
    try:
        wp = _probe_path_mibs(primary)
        wm = _probe_path_mibs(mirror)
        if wp > 0 and wm > 0:
            logger.info(
                "Probed ExpertStore disk weights primary=%.1f MiB/s mirror=%.1f MiB/s",
                wp,
                wm,
            )
            return (wp, wm)
    except Exception as e:
        logger.info("Disk weight probe failed (%s); using equal weights", e)
    return (1.0, 1.0)


def _probe_path_mibs(root: Path) -> float:
    """Read up to 4 MiB from the first expert file under ``root``."""
    manifest = load_manifest(root)
    if manifest is None or not manifest.layers:
        return 0.0
    layer = manifest.layers[0]
    path = root / layer.file_name
    if not path.is_file():
        return 0.0
    nbytes = min(4 * 1024 * 1024, path.stat().st_size)
    if nbytes <= 0:
        return 0.0
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        got = 0
        while got < nbytes:
            chunk = f.read(min(1024 * 1024, nbytes - got))
            if not chunk:
                break
            got += len(chunk)
    dt = max(time.perf_counter() - t0, 1e-6)
    return (got / dt) / (1024 * 1024)


def validate_mirror_files(primary: Path, mirror: Path) -> set[str]:
    """Return expert file names present on mirror with matching size.

    Missing or mismatched files stay primary-only (partial mirror OK).
    Manifest / usage sidecars are never required on the mirror.
    """
    ok: set[str] = set()
    if not mirror.is_dir():
        logger.warning(
            "tier_disk_mirror=%s is not a directory; mirror disabled", mirror
        )
        return ok
    primary_manifest = load_manifest(primary)
    if primary_manifest is None:
        return ok
    for layer in primary_manifest.layers:
        name = layer.file_name
        p_file = primary / name
        m_file = mirror / name
        if not p_file.is_file():
            continue
        if not m_file.is_file():
            logger.info(
                "Mirror missing %s; routing that layer to primary only", name
            )
            continue
        p_sz = p_file.stat().st_size
        m_sz = m_file.stat().st_size
        if p_sz != m_sz:
            logger.warning(
                "Mirror size mismatch for %s primary=%d mirror=%d; "
                "using primary only",
                name,
                p_sz,
                m_sz,
            )
            continue
        # Optional header identity: compare first 64 bytes when present.
        try:
            with open(p_file, "rb") as fp, open(m_file, "rb") as fm:
                if fp.read(64) != fm.read(64):
                    logger.warning(
                        "Mirror header mismatch for %s; using primary only", name
                    )
                    continue
        except OSError as e:
            logger.warning("Mirror header check failed for %s: %s", name, e)
            continue
        ok.add(name)
    # Never treat manifest as a mirrored expert payload.
    ok.discard(MANIFEST_NAME)
    logger.info(
        "ExpertStore mirror validated: %d/%d layer files mirrored under %s",
        len(ok),
        len(primary_manifest.layers),
        mirror,
    )
    return ok


def resolve_tier_numa(flag: bool | None = None) -> bool:
    """Resolve ``--tier-numa`` with ``VLLM_TIER_NUMA`` env override."""
    env = os.environ.get("VLLM_TIER_NUMA", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool(flag)
