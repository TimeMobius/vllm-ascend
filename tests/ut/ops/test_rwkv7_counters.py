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
Unit tests for RWKV7 dispatch counters (rwkv7_counters.py).

Verifies counter increment, snapshot, and reset behavior when enabled,
and no-op behavior when disabled.
"""

import unittest


class TestRWKV7CountersDisabled(unittest.TestCase):
    """Verify counters are a no-op when VLLM_ASCEND_RWKV7_PROFILE is unset."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ.pop("VLLM_ASCEND_RWKV7_PROFILE", None)

    def test_disabled_is_enabled_returns_false(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        self.assertFalse(mod.is_enabled())

    def test_disabled_snapshot_is_all_zeros(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_hits, 0)
        self.assertEqual(snap.recurrent_scan_fallback_guard_false, 0)
        self.assertEqual(snap.mix6_hits, 0)
        self.assertEqual(snap.kk_pre_hits, 0)
        self.assertEqual(snap.epilogue_hits, 0)


class TestRWKV7CountersEnabled(unittest.TestCase):
    """Verify counter increments when VLLM_ASCEND_RWKV7_PROFILE=1."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ["VLLM_ASCEND_RWKV7_PROFILE"] = "1"

    def setUp(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.reset_counters()

    def test_enabled_is_enabled_returns_true(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        self.assertTrue(mod.is_enabled())

    def test_hit_increments_recurrent_scan(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_hits, 1)

    def test_hit_increments_varlen_scan(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN_VARLEN)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_varlen_hits, 1)

    def test_hit_increments_mix6(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.MIX6)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.mix6_hits, 1)

    def test_hit_increments_kk_pre(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.KK_PRE)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.kk_pre_hits, 1)

    def test_hit_increments_epilogue(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.EPILOGUE)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.epilogue_hits, 1)

    def test_fallback_guard_false_increments_correct_counter(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_fallback(mod.DispatchKind.RECURRENT_SCAN, "guard_false")
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_fallback_guard_false, 1)

    def test_fallback_kernel_exception_increments_correct_counter(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_fallback(mod.DispatchKind.MIX6, "kernel_exception")
        snap = mod.snapshot_counters()
        self.assertEqual(snap.mix6_fallback_kernel_exception, 1)

    def test_reset_zeros_all_counters(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
        mod.dispatch_hit(mod.DispatchKind.MIX6)
        mod.dispatch_fallback(mod.DispatchKind.KK_PRE, "guard_false")
        mod.reset_counters()
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_hits, 0)
        self.assertEqual(snap.mix6_hits, 0)
        self.assertEqual(snap.kk_pre_fallback_guard_false, 0)


if __name__ == "__main__":
    unittest.main()