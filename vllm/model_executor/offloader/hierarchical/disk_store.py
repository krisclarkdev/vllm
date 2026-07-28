# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk→RAM ExpertStore reader with O_DIRECT / priority-queue I/O."""

from __future__ import annotations

import heapq
import mmap
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.hierarchical.format import (
    ExpertStoreManifest,
    LayerExpertMeta,
    coalesce_expert_ranges,
    load_manifest,
    unpack_expert_row,
)
from vllm.v1.kv_offload.tiering.fs.io import O_DIRECT, probe_o_direct

logger = init_logger(__name__)

# Many filesystems require 4 KiB alignment for O_DIRECT; 512 is the floor.
DIRECT_ALIGN = 4096


class IoPriority(IntEnum):
    """Lower value = higher priority (demand beats PILOT prefetch)."""

    DEMAND = 0
    PILOT = 1


def align_down(value: int, align: int) -> int:
    return value - (value % align)


def align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def direct_read_window(
    offset: int, nbytes: int, *, align: int = DIRECT_ALIGN
) -> tuple[int, int, int]:
    """Return ``(aligned_offset, aligned_nbytes, pad_before)`` for O_DIRECT.

    The caller reads ``aligned_nbytes`` at ``aligned_offset`` into an aligned
    buffer, then keeps ``buf[pad_before : pad_before + nbytes]``.
    """
    if nbytes < 0 or offset < 0:
        raise ValueError(f"invalid pread window offset={offset} nbytes={nbytes}")
    aligned_off = align_down(offset, align)
    pad_before = offset - aligned_off
    aligned_len = align_up(pad_before + nbytes, align)
    return aligned_off, aligned_len, pad_before


@dataclass(order=True)
class _IoJob:
    priority: int
    seq: int
    fn: Callable[[], object] = field(compare=False)
    future: Future = field(compare=False)


class _PriorityIoPool:
    """Thread pool that always drains demand jobs before PILOT prefetch."""

    def __init__(self, num_workers: int, *, thread_name_prefix: str = "expert-store"):
        self._cv = threading.Condition()
        self._heap: list[_IoJob] = []
        self._seq = 0
        self._closed = False
        self._workers = [
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}-{i}",
                daemon=True,
            )
            for i in range(max(1, num_workers))
        ]
        for t in self._workers:
            t.start()

    def submit(self, fn: Callable[[], object], *, priority: IoPriority) -> Future:
        fut: Future = Future()
        with self._cv:
            if self._closed:
                fut.set_exception(RuntimeError("ExpertStore I/O pool is closed"))
                return fut
            self._seq += 1
            heapq.heappush(
                self._heap, _IoJob(int(priority), self._seq, fn, fut)
            )
            self._cv.notify()
        return fut

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._cv:
            self._closed = True
            if cancel_futures:
                while self._heap:
                    job = heapq.heappop(self._heap)
                    job.future.cancel()
            self._cv.notify_all()
        if wait:
            for t in self._workers:
                t.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._heap and not self._closed:
                    self._cv.wait()
                if not self._heap:
                    return
                job = heapq.heappop(self._heap)
            if job.future.set_running_or_notify_cancel():
                try:
                    job.future.set_result(job.fn())
                except Exception as e:
                    job.future.set_exception(e)


class ExpertStoreReader:
    """Reads expert rows from an on-disk ExpertStore into pinned uint8 buffers."""

    def __init__(
        self,
        disk_path: str,
        *,
        num_workers: int = 8,
        prefer_direct: bool = True,
        layers: dict[int, LayerExpertMeta] | None = None,
        share_pool: _PriorityIoPool | None = None,
    ):
        self.disk_path = Path(disk_path)
        self.prefer_direct = prefer_direct and bool(O_DIRECT)
        self._use_direct = False
        self.disk_direct_fallback = 0
        if self.prefer_direct:
            self._use_direct = probe_o_direct(str(self.disk_path))
            if not self._use_direct:
                self.disk_direct_fallback += 1
                logger.info(
                    "O_DIRECT unavailable under %s; using buffered I/O",
                    self.disk_path,
                )
        if layers is not None:
            self.manifest = None
            self._layers = dict(layers)
        else:
            self.manifest = load_manifest(self.disk_path)
            if self.manifest is None:
                raise FileNotFoundError(
                    f"ExpertStore manifest not found under {self.disk_path}"
                )
            self._layers = {
                layer.layer_id: layer for layer in self.manifest.layers
            }
        self._owns_pool = share_pool is None
        self._pool = share_pool or _PriorityIoPool(num_workers)
        self._lock = threading.Lock()
        self.bytes_served = 0
        self.volume_name = "primary"

    def close(self) -> None:
        if self._owns_pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def layer_meta(self, layer_id: int) -> LayerExpertMeta:
        return self._layers[layer_id]

    def read_row_sync(self, layer_id: int, expert_id: int) -> torch.Tensor:
        """Read one expert row into a CPU uint8 tensor."""
        rows = self.read_rows_sync(layer_id, [expert_id])
        return rows[expert_id]

    def read_rows_sync(
        self, layer_id: int, expert_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Read expert rows, coalescing contiguous ids into one pread each."""
        meta = self._layers[layer_id]
        path = self.disk_path / meta.file_name
        out: dict[int, torch.Tensor] = {}
        for start, end in coalesce_expert_ranges(expert_ids):
            count = end - start + 1
            nbytes = count * meta.row_nbytes
            blob = self._pread(path, start * meta.row_nbytes, nbytes)
            with self._lock:
                self.bytes_served += nbytes
            for i, eid in enumerate(range(start, end + 1)):
                off = i * meta.row_nbytes
                chunk = blob[off : off + meta.row_nbytes]
                out[eid] = torch.frombuffer(
                    bytearray(chunk), dtype=torch.uint8
                ).clone()
        return out

    def read_row_async(
        self,
        layer_id: int,
        expert_id: int,
        *,
        priority: IoPriority = IoPriority.DEMAND,
    ) -> Future:
        return self._pool.submit(
            lambda: self.read_row_sync(layer_id, expert_id),
            priority=priority,
        )

    def read_rows_async(
        self,
        layer_id: int,
        expert_ids: list[int],
        *,
        priority: IoPriority = IoPriority.DEMAND,
    ) -> Future:
        ids = list(expert_ids)
        return self._pool.submit(
            lambda: self.read_rows_sync(layer_id, ids),
            priority=priority,
        )

    def unpack_row(
        self, layer_id: int, blob: torch.Tensor
    ) -> list[torch.Tensor]:
        meta = self._layers[layer_id]
        return unpack_expert_row(blob, meta.tensor_specs)

    def _pread(self, path: Path, offset: int, nbytes: int) -> bytes:
        if self._use_direct:
            try:
                return self._pread_direct(path, offset, nbytes)
            except OSError as e:
                with self._lock:
                    self.disk_direct_fallback += 1
                    fallback = self.disk_direct_fallback
                logger.warning(
                    "O_DIRECT read failed (%s); falling back to buffered I/O "
                    "(fallback #%d)",
                    e,
                    fallback,
                )
        return self._pread_buffered(path, offset, nbytes)

    def _pread_buffered(self, path: Path, offset: int, nbytes: int) -> bytes:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            remaining = nbytes
            chunks: list[bytes] = []
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) != nbytes:
                raise OSError(
                    f"Short read {path}@{offset}: {len(data)}/{nbytes}"
                )
            return data
        finally:
            os.close(fd)

    def _pread_direct(self, path: Path, offset: int, nbytes: int) -> bytes:
        """O_DIRECT pread via an aligned mmap window, then copy the payload."""
        aligned_off, aligned_len, pad_before = direct_read_window(offset, nbytes)
        mm = mmap.mmap(-1, aligned_len)
        view: memoryview | None = None
        try:
            view = memoryview(mm)
            fd = os.open(str(path), os.O_RDONLY | O_DIRECT)
            try:
                os.lseek(fd, aligned_off, os.SEEK_SET)
                got = 0
                while got < aligned_len:
                    n = os.readv(fd, [view[got:]])
                    if n <= 0:
                        break
                    got += n
                if got < pad_before + nbytes:
                    raise OSError(
                        f"Short O_DIRECT read {path}@{aligned_off}: "
                        f"{got}/{aligned_len}"
                    )
            finally:
                os.close(fd)
            return bytes(view[pad_before : pad_before + nbytes])
        finally:
            if view is not None:
                view.release()
            mm.close()


def ensure_store_or_none(
    disk_path: str | None,
    *,
    num_workers: int,
    prefer_direct: bool,
    disk_mirror: str | None = None,
    disk_weights: str | None = None,
) -> ExpertStoreReader | None:
    if not disk_path:
        return None
    path = Path(disk_path)
    if not (path / "manifest.json").exists():
        logger.warning(
            "tier_disk_path=%s has no manifest yet; disk tier inactive "
            "until ExpertStore is built",
            disk_path,
        )
        return None
    primary = ExpertStoreReader(
        disk_path, num_workers=num_workers, prefer_direct=prefer_direct
    )
    if not disk_mirror:
        return primary
    return MirroredExpertStoreReader(
        primary,
        mirror_path=disk_mirror,
        num_workers=num_workers,
        prefer_direct=prefer_direct,
        disk_weights=disk_weights,
    )


class MirroredExpertStoreReader:
    """Primary + optional read-only mirror with weighted expert routing.

    Usage / KV / sidecars stay on the primary root. Mirror is expert payloads
    only. PILOT and demand share ``volume_for_expert`` so the same expert
    always hits the same volume.
    """

    def __init__(
        self,
        primary: ExpertStoreReader,
        *,
        mirror_path: str,
        num_workers: int = 8,
        prefer_direct: bool = True,
        disk_weights: str | None = None,
    ):
        from vllm.model_executor.offloader.hierarchical.mirror import (
            parse_disk_weights,
            probe_disk_weights,
            validate_mirror_files,
            volume_for_expert,
        )

        self._volume_for_expert = volume_for_expert
        self.primary = primary
        self.disk_path = primary.disk_path
        self.manifest = primary.manifest
        self._layers = primary._layers
        self.disk_direct_fallback = primary.disk_direct_fallback
        self._mirror_path = Path(mirror_path)
        self._mirrored_files = validate_mirror_files(
            primary.disk_path, self._mirror_path
        )
        self._mirror: ExpertStoreReader | None = None
        self._mirror_fallback_warned = False
        self._lock = threading.Lock()
        if self._mirrored_files:
            try:
                self._mirror = ExpertStoreReader(
                    str(self._mirror_path),
                    num_workers=1,
                    prefer_direct=prefer_direct,
                    layers=primary._layers,
                    share_pool=primary._pool,
                )
                self._mirror.volume_name = "mirror"
                self._mirror.manifest = primary.manifest
            except Exception as e:
                logger.warning(
                    "Failed to open ExpertStore mirror at %s: %s",
                    self._mirror_path,
                    e,
                )
                self._mirror = None
                self._mirrored_files = set()

        if disk_weights:
            self._w_primary, self._w_mirror = parse_disk_weights(disk_weights)
        elif self._mirror is not None:
            self._w_primary, self._w_mirror = probe_disk_weights(
                primary.disk_path, self._mirror_path
            )
        else:
            self._w_primary, self._w_mirror = (1.0, 0.0)

        logger.info(
            "MIRROR: ExpertStore dual-root enabled primary=%s mirror=%s "
            "weights=%.3f,%.3f mirrored_files=%d",
            primary.disk_path,
            self._mirror_path if self._mirror else None,
            self._w_primary,
            self._w_mirror,
            len(self._mirrored_files),
        )

    @property
    def bytes_served(self) -> int:
        return int(self.primary.bytes_served) + int(
            self._mirror.bytes_served if self._mirror else 0
        )

    def mirror_stats(self) -> dict[str, float | int]:
        p = int(self.primary.bytes_served)
        m = int(self._mirror.bytes_served) if self._mirror else 0
        return {
            "primary_bytes": p,
            "mirror_bytes": m,
            "primary_gib": p / 1024**3,
            "mirror_gib": m / 1024**3,
            "mirrored_files": len(self._mirrored_files),
            "weight_primary": self._w_primary,
            "weight_mirror": self._w_mirror,
        }

    def log_mirror_stats(self) -> None:
        s = self.mirror_stats()
        logger.info(
            "MIRROR: served primary=%.3f GiB mirror=%.3f GiB "
            "(weights=%.2f,%.2f files=%d)",
            s["primary_gib"],
            s["mirror_gib"],
            s["weight_primary"],
            s["weight_mirror"],
            s["mirrored_files"],
        )

    def close(self) -> None:
        self.log_mirror_stats()
        self.primary.close()
        if self._mirror is not None:
            self._mirror.close()

    def has_layer(self, layer_id: int) -> bool:
        return self.primary.has_layer(layer_id)

    def layer_meta(self, layer_id: int):
        return self.primary.layer_meta(layer_id)

    def unpack_row(self, layer_id: int, blob: torch.Tensor):
        return self.primary.unpack_row(layer_id, blob)

    def route_volume(self, layer_id: int, expert_id: int) -> int:
        meta = self._layers.get(layer_id)
        if (
            meta is None
            or self._mirror is None
            or meta.file_name not in self._mirrored_files
        ):
            return 0
        return self._volume_for_expert(
            layer_id, expert_id, self._w_primary, self._w_mirror
        )

    def read_row_sync(self, layer_id: int, expert_id: int) -> torch.Tensor:
        return self.read_rows_sync(layer_id, [expert_id])[expert_id]

    def read_rows_sync(
        self, layer_id: int, expert_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        primary_ids: list[int] = []
        mirror_ids: list[int] = []
        for eid in expert_ids:
            if int(eid) < 0:
                continue
            if self.route_volume(layer_id, int(eid)) == 1:
                mirror_ids.append(int(eid))
            else:
                primary_ids.append(int(eid))

        out: dict[int, torch.Tensor] = {}
        if primary_ids:
            out.update(self.primary.read_rows_sync(layer_id, primary_ids))
        if mirror_ids and self._mirror is not None:
            try:
                out.update(self._mirror.read_rows_sync(layer_id, mirror_ids))
            except Exception as e:
                if not self._mirror_fallback_warned:
                    logger.warning(
                        "Mirror read failed (%s); falling back to primary "
                        "(further failures use primary silently)",
                        e,
                    )
                    self._mirror_fallback_warned = True
                out.update(self.primary.read_rows_sync(layer_id, mirror_ids))
        return out

    def read_row_async(
        self,
        layer_id: int,
        expert_id: int,
        *,
        priority: IoPriority = IoPriority.DEMAND,
    ) -> Future:
        reader = (
            self._mirror
            if self.route_volume(layer_id, expert_id) == 1 and self._mirror
            else self.primary
        )
        assert reader is not None

        def _read():
            try:
                return reader.read_row_sync(layer_id, expert_id)
            except Exception:
                if reader is self._mirror:
                    if not self._mirror_fallback_warned:
                        logger.warning(
                            "Mirror async read failed; falling back to primary"
                        )
                        self._mirror_fallback_warned = True
                    return self.primary.read_row_sync(layer_id, expert_id)
                raise

        # Use primary pool so demand/PILOT priority is global.
        return self.primary._pool.submit(_read, priority=priority)

    def read_rows_async(
        self,
        layer_id: int,
        expert_ids: list[int],
        *,
        priority: IoPriority = IoPriority.DEMAND,
    ) -> Future:
        ids = list(expert_ids)
        return self.primary._pool.submit(
            lambda: self.read_rows_sync(layer_id, ids),
            priority=priority,
        )
