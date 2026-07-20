#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Per-process RWKV7 dispatch hit/fallback counters.

Instruments recurrent_scan, mix6, kk_pre, and epilogue dispatches in
patch_rwkv7.py to count:
- hits: kernel dispatched successfully
- fallback_guard_false: guard returned False, fell back to reference
- fallback_exception: guard was True but kernel raised, fell back to reference

Controlled by VLLM_ASCEND_RWKV7_PROFILE. When disabled (default), increment()
is a no-op with no overhead. When enabled, a structured summary is emitted
at process exit via atexit.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Literal

_logger = logging.getLogger("vllm_ascend")


class DispatchKind(Enum):
    RECURRENT_SCAN = "recurrent_scan"
    RECURRENT_SCAN_VARLEN = "recurrent_scan_varlen"
    MIX6 = "mix6"
    KK_PRE = "kk_pre"
    EPILOGUE = "epilogue"


class FallbackReason(Enum):
    GUARD_FALSE = "guard_false"
    KERNEL_EXCEPTION = "kernel_exception"


@dataclass
class _RWKV7CountersSnapshot:
    recurrent_scan_hits: int
    recurrent_scan_fallback_guard_false: int
    recurrent_scan_fallback_kernel_exception: int
    recurrent_scan_varlen_hits: int
    recurrent_scan_varlen_fallback_guard_false: int
    recurrent_scan_varlen_fallback_kernel_exception: int
    mix6_hits: int
    mix6_fallback_guard_false: int
    mix6_fallback_kernel_exception: int
    kk_pre_hits: int
    kk_pre_fallback_guard_false: int
    kk_pre_fallback_kernel_exception: int
    epilogue_hits: int
    epilogue_fallback_guard_false: int
    epilogue_fallback_kernel_exception: int


class _DisabledCounters:
    __slots__ = ()

    def hit(self, kind: DispatchKind) -> None:
        pass

    def fallback(self, kind: DispatchKind, reason: Literal["guard_false", "kernel_exception"]) -> None:
        pass

    def snapshot(self) -> _RWKV7CountersSnapshot:
        return _RWKV7CountersSnapshot(
            recurrent_scan_hits=0,
            recurrent_scan_fallback_guard_false=0,
            recurrent_scan_fallback_kernel_exception=0,
            recurrent_scan_varlen_hits=0,
            recurrent_scan_varlen_fallback_guard_false=0,
            recurrent_scan_varlen_fallback_kernel_exception=0,
            mix6_hits=0,
            mix6_fallback_guard_false=0,
            mix6_fallback_kernel_exception=0,
            kk_pre_hits=0,
            kk_pre_fallback_guard_false=0,
            kk_pre_fallback_kernel_exception=0,
            epilogue_hits=0,
            epilogue_fallback_guard_false=0,
            epilogue_fallback_kernel_exception=0,
        )

    def reset(self) -> None:
        pass


class _EnabledCounters:
    __slots__ = ("_hits", "_fallback_guard_false", "_fallback_kernel_exception", "_lock", "_atexit_registered")

    def __init__(self) -> None:
        self._hits: dict[str, int] = {k.value: 0 for k in DispatchKind}
        self._fallback_guard_false: dict[str, int] = {k.value: 0 for k in DispatchKind}
        self._fallback_kernel_exception: dict[str, int] = {k.value: 0 for k in DispatchKind}
        self._lock = threading.Lock()
        self._atexit_registered = False
        self._maybe_register_atexit()

    def _maybe_register_atexit(self) -> None:
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self._emit_summary)

    def hit(self, kind: DispatchKind) -> None:
        with self._lock:
            self._hits[kind.value] += 1

    def fallback(
        self, kind: DispatchKind, reason: Literal["guard_false", "kernel_exception"]
    ) -> None:
        with self._lock:
            if reason == FallbackReason.GUARD_FALSE.value:
                self._fallback_guard_false[kind.value] += 1
            else:
                self._fallback_kernel_exception[kind.value] += 1

    def snapshot(self) -> _RWKV7CountersSnapshot:
        with self._lock:
            h = self._hits
            gf = self._fallback_guard_false
            gk = self._fallback_kernel_exception
            return _RWKV7CountersSnapshot(
                recurrent_scan_hits=h["recurrent_scan"],
                recurrent_scan_fallback_guard_false=gf["recurrent_scan"],
                recurrent_scan_fallback_kernel_exception=gk["recurrent_scan"],
                recurrent_scan_varlen_hits=h["recurrent_scan_varlen"],
                recurrent_scan_varlen_fallback_guard_false=gf["recurrent_scan_varlen"],
                recurrent_scan_varlen_fallback_kernel_exception=gk["recurrent_scan_varlen"],
                mix6_hits=h["mix6"],
                mix6_fallback_guard_false=gf["mix6"],
                mix6_fallback_kernel_exception=gk["mix6"],
                kk_pre_hits=h["kk_pre"],
                kk_pre_fallback_guard_false=gf["kk_pre"],
                kk_pre_fallback_kernel_exception=gk["kk_pre"],
                epilogue_hits=h["epilogue"],
                epilogue_fallback_guard_false=gf["epilogue"],
                epilogue_fallback_kernel_exception=gk["epilogue"],
            )

    def reset(self) -> None:
        with self._lock:
            for k in DispatchKind:
                self._hits[k.value] = 0
                self._fallback_guard_false[k.value] = 0
                self._fallback_kernel_exception[k.value] = 0

    def _emit_summary(self) -> None:
        snap = self.snapshot()
        summary = (
            "[vllm-ascend] [rwkv7_profile] RWKV7 dispatch summary:\n"
            "  recurrent_scan:        hits={}  fallback_guard_false={}  fallback_kernel_exception={}\n"
            "  recurrent_scan_varlen: hits={}  fallback_guard_false={}  fallback_kernel_exception={}\n"
            "  mix6:                   hits={}  fallback_guard_false={}  fallback_kernel_exception={}\n"
            "  kk_pre:                 hits={}  fallback_guard_false={}  fallback_kernel_exception={}\n"
            "  epilogue:               hits={}  fallback_guard_false={}  fallback_kernel_exception={}"
        ).format(
            snap.recurrent_scan_hits,
            snap.recurrent_scan_fallback_guard_false,
            snap.recurrent_scan_fallback_kernel_exception,
            snap.recurrent_scan_varlen_hits,
            snap.recurrent_scan_varlen_fallback_guard_false,
            snap.recurrent_scan_varlen_fallback_kernel_exception,
            snap.mix6_hits,
            snap.mix6_fallback_guard_false,
            snap.mix6_fallback_kernel_exception,
            snap.kk_pre_hits,
            snap.kk_pre_fallback_guard_false,
            snap.kk_pre_fallback_kernel_exception,
            snap.epilogue_hits,
            snap.epilogue_fallback_guard_false,
            snap.epilogue_fallback_kernel_exception,
        )
        _logger.info("%s", summary)


_ENABLED = bool(int(os.getenv("VLLM_ASCEND_RWKV7_PROFILE", "0")))
_counters: _DisabledCounters | _EnabledCounters = (
    _EnabledCounters() if _ENABLED else _DisabledCounters()
)


def dispatch_hit(kind: DispatchKind) -> None:
    """Record a successful kernel dispatch."""
    _counters.hit(kind)


def dispatch_fallback(
    kind: DispatchKind, reason: Literal["guard_false", "kernel_exception"]
) -> None:
    """Record a fallback to the reference implementation."""
    _counters.fallback(kind, reason)


def snapshot_counters() -> _RWKV7CountersSnapshot:
    """Return a thread-safe snapshot of current counter values."""
    return _counters.snapshot()


def reset_counters() -> None:
    """Reset all counters to zero. For testing use only."""
    _counters.reset()


def is_enabled() -> bool:
    """Return whether RWKV7 profiling is currently enabled."""
    return _ENABLED