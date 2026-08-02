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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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

    def test_disabled_hits_are_noops(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        mod.dispatch_hit(mod.DispatchKind.MIX6)
        mod.dispatch_fallback(mod.DispatchKind.KK_PRE, "guard_false")
        snap = mod.snapshot_counters()
        self.assertEqual(snap.mix6_hits, 0)
        self.assertEqual(snap.kk_pre_fallback_guard_false, 0)


class TestRWKV7CountersEnabled(unittest.TestCase):
    """Verify counters track hits/fallbacks/reset when VLLM_ASCEND_RWKV7_PROFILE=1."""

    @classmethod
    def setUpClass(cls):
        import os

        os.environ["VLLM_ASCEND_RWKV7_PROFILE"] = "1"

    @classmethod
    def tearDownClass(cls):
        import os

        os.environ.pop("VLLM_ASCEND_RWKV7_PROFILE", None)

    def _reload(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        return mod

    def test_hit_increments_correct_kind(self):
        mod = self._reload()
        mod.reset_counters()
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_T1)
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
        mod.dispatch_hit(mod.DispatchKind.MIX6)
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_t1_hits, 1)
        self.assertEqual(snap.recurrent_scan_hits, 2)
        self.assertEqual(snap.mix6_hits, 1)

    def test_fallback_guard_false_and_kernel_exception(self):
        mod = self._reload()
        mod.reset_counters()
        mod.dispatch_fallback(mod.DispatchKind.MIX6, "guard_false")
        mod.dispatch_fallback(mod.DispatchKind.KK_PRE, "kernel_exception")
        snap = mod.snapshot_counters()
        self.assertEqual(snap.mix6_fallback_guard_false, 1)
        self.assertEqual(snap.kk_pre_fallback_kernel_exception, 1)

    def test_reset_zeros_all_counters(self):
        mod = self._reload()
        mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
        mod.dispatch_hit(mod.DispatchKind.MIX6)
        mod.dispatch_fallback(mod.DispatchKind.KK_PRE, "guard_false")
        mod.reset_counters()
        snap = mod.snapshot_counters()
        self.assertEqual(snap.recurrent_scan_hits, 0)
        self.assertEqual(snap.mix6_hits, 0)
        self.assertEqual(snap.kk_pre_fallback_guard_false, 0)


class TestRWKV7CountersGraphSafe(unittest.TestCase):
    """Counter mutation/observation must be graph-safe so that torch._dynamo's
    fullgraph_capture (triggered by vLLM's profile_run) does not trip on
    `with self._lock`.
    """

    @classmethod
    def setUpClass(cls):
        import os

        os.environ["VLLM_ASCEND_RWKV7_PROFILE"] = "1"

    @classmethod
    def tearDownClass(cls):
        import os

        os.environ.pop("VLLM_ASCEND_RWKV7_PROFILE", None)

    def _reload(self):
        import importlib
        import sys

        sys.path.insert(0, "/mnt/data/Codes/vllm-ascend")
        import vllm_ascend.profiler.rwkv7_counters as mod

        importlib.reload(mod)
        return mod

    def test_counter_methods_exist(self):
        mod = self._reload()
        for method_name in ("hit", "fallback", "snapshot", "reset"):
            method = getattr(mod._counters, method_name)
            self.assertTrue(callable(method))

    def test_counter_methods_work_under_torch_compile_fullgraph(self):
        """Compile a trivial graph that calls hit/fallback under fullgraph.
        Must not raise `Unsupported context manager` from `with self._lock`.
        """
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this test environment")

        mod = self._reload()

        @torch.compile(fullgraph=True, dynamic=False)
        def _exercise():
            mod.dispatch_hit(mod.DispatchKind.RECURRENT_SCAN)
            mod.dispatch_hit(mod.DispatchKind.MIX6)
            mod.dispatch_fallback(mod.DispatchKind.MIX6, "guard_false")
            mod.dispatch_fallback(mod.DispatchKind.KK_PRE, "kernel_exception")
            mod.dispatch_hit(mod.DispatchKind.EPILOGUE)
            return mod.snapshot_counters()

        try:
            snap = _exercise()
        except Exception as e:
            self.fail(f"graph-mode counter call failed: {type(e).__name__}: {e}")

        self.assertGreaterEqual(snap.recurrent_scan_hits, 1)
        self.assertGreaterEqual(snap.mix6_hits, 1)
        self.assertGreaterEqual(snap.mix6_fallback_guard_false, 1)
        self.assertGreaterEqual(snap.kk_pre_fallback_kernel_exception, 1)
        self.assertGreaterEqual(snap.epilogue_hits, 1)


if __name__ == "__main__":
    unittest.main()
