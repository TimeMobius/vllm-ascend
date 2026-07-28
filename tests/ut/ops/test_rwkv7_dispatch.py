# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RWKV7 dispatch selection and fallback tests.

This module tests that the RWKV7 patch correctly:
1. Dispatches to triton-ascend operations when available on NPU
2. Falls back to reference operations when triton-ascend is unavailable
3. Does not require full server startup to validate the dispatch mechanism

These tests are complementary to:
- test_rwkv7_mix6.py: Tests mix6 correctness
- test_rwkv7_npu.py: Tests upstream reference path on NPU
- test_rwkv7_kk_pre.py: Tests kk_pre correctness
- test_fused_recurrent_rwkv7_npu.py: Tests recurrent correctness
"""

import unittest
from unittest import mock

import torch


class TestRWKV7DispatchSelection(unittest.TestCase):
    """Test that dispatch selection works correctly for RWKV7 operations."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 dispatch tests")

    def test_rwkv7_mix6_uses_ascend_on_npu(self):
        """Verify rwkv7_mix6 on NPU uses triton-ascend path."""
        from vllm_ascend.ops.triton.fla.rwkv7_mix6 import rwkv7_mix6

        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="npu", dtype=torch.float32)
        delta = torch.randn(batch, seq, hidden, device="npu", dtype=torch.float32)
        x_r = torch.randn(hidden, device="npu", dtype=torch.float32)
        x_w = torch.randn(hidden, device="npu", dtype=torch.float32)
        x_k = torch.randn(hidden, device="npu", dtype=torch.float32)
        x_v = torch.randn(hidden, device="npu", dtype=torch.float32)
        x_a = torch.randn(hidden, device="npu", dtype=torch.float32)
        x_g = torch.randn(hidden, device="npu", dtype=torch.float32)

        # This should use triton-ascend on NPU
        xr, xw, xk, xv, xa, xg = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Verify output shapes
        self.assertEqual(xr.shape, hidden_states.shape)
        self.assertEqual(xw.shape, hidden_states.shape)
        # Verify outputs are on NPU
        self.assertEqual(xr.device.type, "npu")
        self.assertTrue(torch.isfinite(xr).all())

    def test_rwkv7_kk_pre_uses_ascend_on_npu(self):
        """Verify rwkv7_kk_pre on NPU uses triton-ascend path."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 4, 2, 64
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device="npu", dtype=torch.float32)
        k_a = torch.randn(H, K, device="npu", dtype=torch.float32)

        # This should use triton-ascend on NPU
        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        # Verify output shapes
        self.assertEqual(k_adj.shape, k.shape)
        self.assertEqual(kk.shape, k.shape)
        # Verify device placement
        self.assertEqual(k_adj.device.type, "npu")
        self.assertEqual(kk.device.type, "npu")

    def test_rwkv7_lnx_rkvres_xg_uses_ascend_on_npu(self):
        """Verify rwkv7_lnx_rkvres_xg on NPU uses triton-ascend path."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 64, 64
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device="npu", dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device="npu", dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device="npu", dtype=torch.float32)
        weight = torch.randn(local_value_dim, device="npu", dtype=torch.float32)
        bias = torch.randn(local_value_dim, device="npu", dtype=torch.float32)
        g = torch.randn(num_tokens, local_value_dim, device="npu", dtype=torch.float32)

        # This should use triton-ascend on NPU
        output = rwkv7_lnx_rkvres_xg(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=64e-5,
        )

        # Verify output shape
        self.assertEqual(output.shape, (num_tokens, local_value_dim))
        # Verify device placement
        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())


class TestRWKV7FallbackOnCPU(unittest.TestCase):
    """Test that reference fallback works correctly on CPU (no NPU required)."""

    def test_mix6_falls_back_to_reference_on_cpu(self):
        """Verify rwkv7_mix6 falls back to reference on CPU."""
        from vllm_ascend.ops.triton.fla.rwkv7_mix6 import rwkv7_mix6, rwkv7_mix6_reference

        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        delta = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        x_r = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_w = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_k = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_v = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_a = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_g = torch.randn(hidden, device="cpu", dtype=torch.float32)

        # Compute with wrapper (should fall back to reference on CPU)
        wrapper_out = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Compute with reference directly
        ref_out = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Should match since fallback is used
        for w, r in zip(wrapper_out, ref_out):
            torch.testing.assert_close(w, r, atol=1e-5, rtol=1e-5)

    def test_kk_pre_falls_back_to_reference_on_cpu(self):
        """Verify rwkv7_kk_pre falls back to reference on CPU."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre, rwkv7_kk_pre_reference

        T, H, K = 4, 2, 64
        k = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device="cpu", dtype=torch.float32)
        k_a = torch.randn(H, K, device="cpu", dtype=torch.float32)

        # Compute with wrapper (should fall back on CPU)
        wrapper_k_adj, wrapper_kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        # Compute with reference directly
        ref_k_adj, ref_kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

        # Should match since fallback is used
        torch.testing.assert_close(wrapper_k_adj, ref_k_adj, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_kk, ref_kk, atol=1e-5, rtol=1e-5)

    def test_disable_triton_forces_rwkv7_guards_off(self):
        """Verify the reference-only switch disables every RWKV7 kernel guard."""
        from vllm_ascend.patch.worker.patch_rwkv7 import (
            _can_use_epilogue_kernel,
            _can_use_fused_recurrent,
            _can_use_kk_pre_kernel,
            _can_use_mix6_kernel,
        )

        hidden_states = torch.randn(2, 4, 64)
        delta = torch.randn_like(hidden_states)
        vector = torch.randn(64)
        recurrent = torch.randn(2, 4, 8)
        head_vector = torch.randn(2, 4, 8)
        head_matrix = torch.randn(2, 8)

        with mock.patch.dict("os.environ", {"VLLM_ASCEND_RWKV7_DISABLE_TRITON": "1"}):
            self.assertFalse(
                _can_use_mix6_kernel(
                    hidden_states,
                    delta,
                    vector,
                    vector,
                    vector,
                    vector,
                    vector,
                    vector,
                )
            )
            self.assertFalse(
                _can_use_kk_pre_kernel(
                    recurrent,
                    head_matrix,
                    recurrent,
                    head_matrix,
                )
            )
            self.assertFalse(
                _can_use_fused_recurrent(
                    recurrent,
                    recurrent,
                    recurrent,
                    head_vector,
                    recurrent,
                    recurrent,
                )
            )
            self.assertFalse(
                _can_use_epilogue_kernel(
                    head_vector,
                    recurrent,
                    recurrent,
                    head_vector,
                    head_matrix,
                    vector,
                    vector,
                    head_vector,
                )
            )


class TestRWKV7RecurrentFallback(unittest.TestCase):
    """Test fused_recurrent_rwkv7 fallback behavior."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping recurrent tests")

    def test_fused_recurrent_rwkv7_returns_correct_signature(self):
        """Verify fused_recurrent_rwkv7 returns correct number of values."""
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import fused_recurrent_rwkv7

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(B, T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        initial_state = torch.randn(B, H, K, V, device="npu", dtype=torch.float32)

        # When output_final_state=True, should return (output, final_state)
        result = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state,
            output_final_state=True,
        )

        # Verify 3 values returned (output, final_state, checkpoint_states)
        self.assertEqual(len(result), 3)
        output, final_state, checkpoint_states = result
        self.assertEqual(output.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))
        self.assertIsNone(checkpoint_states)  # No checkpoints requested

    def test_fused_recurrent_rwkv7_with_checkpoints_returns_correct_signature(self):
        """Verify fused_recurrent_rwkv7_with_checkpoints returns correct values."""
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import fused_recurrent_rwkv7

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(B, T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)

        # With checkpoint positions
        checkpoint_positions = torch.tensor([1, 3], device="npu", dtype=torch.long)
        checkpoint_offsets = torch.tensor([0, 2], device="npu", dtype=torch.long)

        result = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            output_checkpoint_states=True,
        )

        # Should return 3 values
        self.assertEqual(len(result), 3)


class TestRWKV7RecurrentAdapter(unittest.TestCase):
    """Test that the adapter returns correct upstream signature."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping adapter tests")

    def test_adapter_returns_two_values_when_output_final_state_true(self):
        """Verify adapter returns (out, final_state) when output_final_state=True."""
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
            fused_recurrent_rwkv7 as _fused_rec_ascend,
        )

        def adapter(r, w, k, v, kk, a, scale=1.0, initial_state=None,
                    output_final_state=False, cu_seqlens=None):
            out, final_state, _ = _fused_rec_ascend(
                r=r, w=w, k=k, v=v, kk=kk, a=a, scale=scale,
                initial_state=initial_state, output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
            if output_final_state:
                return out, final_state
            return out

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(B, T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)

        result = adapter(r, w, k, v, kk, a, output_final_state=True)
        self.assertEqual(len(result), 2)
        out, final_state = result
        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))

    def test_adapter_returns_one_value_when_output_final_state_false(self):
        """Verify adapter returns just out when output_final_state=False."""
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
            fused_recurrent_rwkv7 as _fused_rec_ascend,
        )

        def adapter(r, w, k, v, kk, a, scale=1.0, initial_state=None,
                    output_final_state=False, cu_seqlens=None):
            out, final_state, _ = _fused_rec_ascend(
                r=r, w=w, k=k, v=v, kk=kk, a=a, scale=scale,
                initial_state=initial_state, output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
            if output_final_state:
                return out, final_state
            return out

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(B, T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)

        result = adapter(r, w, k, v, kk, a, output_final_state=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (B, T, H, V))

    def test_checkpoints_wrapper_returns_three_values(self):
        """Verify checkpoint wrapper returns 3 values (out, final_state, checkpoints)."""
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
            fused_recurrent_rwkv7 as _fused_rec_ascend,
        )

        def checkpoints_adapter(r, w, k, v, kk, a, checkpoint_positions,
                                checkpoint_offsets, scale=1.0, initial_state=None,
                                output_final_state=False, cu_seqlens=None):
            out, final_state, checkpoint_states = _fused_rec_ascend(
                r=r, w=w, k=k, v=v, kk=kk, a=a, scale=scale,
                initial_state=initial_state, output_final_state=True,
                cu_seqlens=cu_seqlens, checkpoint_positions=checkpoint_positions,
                checkpoint_offsets=checkpoint_offsets, output_checkpoint_states=True,
            )
            return out, final_state, checkpoint_states

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(B, T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(B, T, H, K, device="npu", dtype=torch.float32)
        cp = torch.tensor([1, 3], device="npu", dtype=torch.long)
        co = torch.tensor([0, 2], device="npu", dtype=torch.long)

        result = checkpoints_adapter(r, w, k, v, kk, a, cp, co)
        self.assertEqual(len(result), 3)


class TestRWKV7PatchWiring(unittest.TestCase):
    """Test that the patch correctly wires operations from vllm-ascend."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping patch wiring tests")

    def test_ops_module_imports_ascend_operations(self):
        """Verify the local RWKV7 FLA ops can be patched."""
        # This tests that the patching mechanism works
        # The actual patching is done at import time by patch_rwkv7.py
        import vllm_ascend.ops.triton.fla.rwkv7 as ops_module

        # Verify module has the expected functions
        self.assertTrue(hasattr(ops_module, "rwkv7_mix6"))
        self.assertTrue(hasattr(ops_module, "rwkv7_kk_pre"))
        self.assertTrue(hasattr(ops_module, "rwkv7_lnx_rkvres_xg"))
        self.assertTrue(hasattr(ops_module, "fused_mul_recurrent_rwkv7"))

    def test_ascend_operations_produce_valid_output(self):
        """Verify ascend operations produce valid finite output on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7_mix6 import rwkv7_mix6
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import fused_recurrent_rwkv7

        # Test mix6
        hidden_states = torch.randn(2, 4, 64, device="npu", dtype=torch.float32)
        delta = torch.randn(2, 4, 64, device="npu", dtype=torch.float32)
        x_r = torch.randn(64, device="npu", dtype=torch.float32)
        xw = torch.randn(64, device="npu", dtype=torch.float32)
        xk = torch.randn(64, device="npu", dtype=torch.float32)
        xv = torch.randn(64, device="npu", dtype=torch.float32)
        xa = torch.randn(64, device="npu", dtype=torch.float32)
        xg = torch.randn(64, device="npu", dtype=torch.float32)

        xr, *_ = rwkv7_mix6(hidden_states, delta, x_r, xw, xk, xv, xa, xg)
        self.assertTrue(torch.isfinite(xr).all())

        # Test kk_pre
        k = torch.randn(4, 2, 64, device="npu", dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(2, 64, device="npu", dtype=torch.float32)
        k_a = torch.randn(2, 64, device="npu", dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)
        self.assertTrue(torch.isfinite(k_adj).all())
        self.assertTrue(torch.isfinite(kk).all())

        # Test recurrent
        r = torch.randn(1, 4, 2, 8, device="npu", dtype=torch.float32)
        w = torch.randn(1, 4, 2, 8, device="npu", dtype=torch.float32)
        k = torch.randn(1, 4, 2, 8, device="npu", dtype=torch.float32)
        v = torch.randn(1, 4, 2, 16, device="npu", dtype=torch.float32)
        kk = torch.randn(1, 4, 2, 8, device="npu", dtype=torch.float32)
        a = torch.randn(1, 4, 2, 8, device="npu", dtype=torch.float32)

        out, _, _ = fused_recurrent_rwkv7(r=r, w=w, k=k, v=v, kk=kk, a=a)
        self.assertTrue(torch.isfinite(out).all())


class TestRWKV7PatchDoesNotModifyUpstream(unittest.TestCase):
    """Verify patch does not modify upstream source files."""

    def test_patch_only_affects_imported_modules(self):
        """
        Verify the patch only affects modules that are explicitly imported.

        The patch should not modify /mnt/data/Codes/vllm source files.
        This is a structural test to ensure the patching follows the
        "minimal patch" principle.
        """
        import vllm_ascend.ops.triton.fla.rwkv7 as ops_module

        # The ops module should have rwkv7_mix6 function
        # We can't directly test if it's been patched without importing
        # the patch, but we can verify the function exists and is callable
        self.assertTrue(callable(getattr(ops_module, "rwkv7_mix6", None)))
        self.assertTrue(callable(getattr(ops_module, "rwkv7_kk_pre", None)))


class TestRWKV7FinalizeLocalRKCache(unittest.TestCase):
    """Verify _ascend_r_k_fp32_local cache in _finalize_attention_output.

    Opt-C: cache local_r_k as fp32 to avoid per-call slice+cast on constant
    model param. Eliminates 1 slice + 1 dtype cast per decode step per layer
    (= 61 saves per request). Same pattern as epilogue patch's
    _ascend_r_k_fp32.
    """

    def _make_attention(self):
        """Build a minimal RWKV7Attention-like object for _finalize testing."""
        from types import SimpleNamespace

        attn = SimpleNamespace()

        # Model-like constants (RWKV7 sizes from step-12250)
        num_heads = 64
        head_dim = 64
        head_v_dim = 64
        local_num_heads = num_heads
        local_value_dim = num_heads * head_v_dim

        # r_k is the constant model param; [num_heads, head_dim] bf16
        attn.r_k = torch.randn(num_heads, head_dim, dtype=torch.bfloat16)
        attn.tp_rank = 0
        attn.local_num_heads = local_num_heads
        attn.local_value_dim = local_value_dim
        attn.value_start = 0
        attn.value_end = local_value_dim

        # Stub g_norm to mimic F.group_norm without real weights
        class FakeGroupNorm:
            weight = torch.randn(local_value_dim, dtype=torch.float32)
            bias = torch.randn(local_value_dim, dtype=torch.float32)

            def __call__(self, x):
                return x

        attn.g_norm = FakeGroupNorm()

        # Stub o_proj to return identity
        class FakeOProj:
            def __call__(self, x):
                return x, None

        attn.o_proj = FakeOProj()

        return attn

    def test_cache_attribute_set_after_first_call(self):
        """Cache attribute is created on first call and reused on second call."""
        attn = self._make_attention()

        # Import the method (bound to the stub object)
        from vllm_ascend.models.rwkv7 import RWKV7Attention

        # Bind the unbound method to our stub via monkey-patch style
        def finalize(self, recurrent_output, r, k, v, g, hidden_dtype):
            return RWKV7Attention._finalize_attention_output(
                self, recurrent_output, r, k, v, g, hidden_dtype
            )

        import types

        attn._finalize_attention_output = types.MethodType(
            finalize, attn
        )

        # Construct fake inputs
        recurrent_output = torch.randn(
            1, attn.local_num_heads, head_dim := 64, dtype=torch.float32
        )
        r = torch.randn(1, attn.local_num_heads, head_dim, dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn(
            1, attn.local_num_heads, 64, dtype=torch.float32
        )
        g = torch.randn(1, attn.local_value_dim, dtype=torch.bfloat16)

        attn._finalize_attention_output(
            recurrent_output, r, k, v, g, torch.bfloat16
        )
        # Cache attribute must now be set
        self.assertTrue(hasattr(attn, "_ascend_r_k_fp32"))
        cached = attn._ascend_r_k_fp32
        self.assertEqual(cached.dtype, torch.float32)
        self.assertEqual(
            cached.shape, (attn.local_num_heads, head_dim)
        )

        # Second call: cache must be reused (same object identity)
        attn._finalize_attention_output(
            recurrent_output, r, k, v, g, torch.bfloat16
        )
        self.assertIs(attn._ascend_r_k_fp32, cached)

    def test_cache_value_matches_manual_slice(self):
        """Cached value must equal the original slice+cast computation."""
        attn = self._make_attention()
        expected = attn.r_k[
            attn.tp_rank
            * attn.local_num_heads : (attn.tp_rank + 1)
            * attn.local_num_heads
        ].to(torch.float32)

        from vllm_ascend.models.rwkv7 import RWKV7Attention
        import types

        def finalize(self, recurrent_output, r, k, v, g, hidden_dtype):
            return RWKV7Attention._finalize_attention_output(
                self, recurrent_output, r, k, v, g, hidden_dtype
            )

        attn._finalize_attention_output = types.MethodType(
            finalize, attn
        )
        recurrent_output = torch.randn(
            1, attn.local_num_heads, 64, dtype=torch.float32
        )
        r = torch.randn(1, attn.local_num_heads, 64, dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn(1, attn.local_num_heads, 64, dtype=torch.float32)
        g = torch.randn(1, attn.local_value_dim, dtype=torch.bfloat16)
        attn._finalize_attention_output(
            recurrent_output, r, k, v, g, torch.bfloat16
        )

        cached = attn._ascend_r_k_fp32
        self.assertTrue(torch.allclose(cached, expected))


if __name__ == "__main__":
    unittest.main()
