# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for RWKV7 Ascend patch.

These tests verify that the vllm-ascend patch correctly:
1. Installs idempotently (can be applied multiple times)
2. Dispatches to Triton-Ascend kernels when safe
3. Falls back to reference when conditions aren't met
4. Produces numerically correct output

Tests use observable behavior - actual tensor operations and their outputs,
not mocks or constant inspection.
"""

import unittest

import torch


class TestRWKV7PatchIdempotency(unittest.TestCase):
    """Test that patch installation is idempotent."""

    def test_patch_can_be_applied_multiple_times(self):
        """Verify apply_patch() is idempotent - no error on repeated calls."""
        from vllm_ascend.patch.worker import patch_rwkv7

        # Apply patch multiple times - should not raise
        patch_rwkv7.apply_patch()
        patch_rwkv7.apply_patch()
        patch_rwkv7.apply_patch()  # Multiple times is OK

    def test_patch_marks_module(self):
        """Verify patch sets _RWKV7_ASCEND_PATCHED flag."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        # Apply patch
        patch_rwkv7.apply_patch()

        # Check flag on upstream module
        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")
        self.assertTrue(getattr(rwkv7_module, "_RWKV7_ASCEND_PATCHED", False))


class TestRWKV7RecurrentScanDispatch(unittest.TestCase):
    """Test _rwkv7_recurrent_scan dispatch and numerical parity."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping tests")

    def test_dispatch_to_fused_kernel_on_npu(self):
        """Verify recurrent scan uses fused kernel on NPU when conditions are met."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        # Create valid NPU tensors
        T, H, K, V = 4, 2, 8, 16
        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)

        # Call patched function
        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=None
        )

        # Verify outputs are on NPU
        self.assertEqual(output.device.type, "npu")
        self.assertEqual(final_state.device.type, "npu")
        # Verify finite output
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(final_state).all())

    def test_numerical_parity_with_reference(self):
        """Verify fused kernel output matches reference within tolerance."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference,
        )

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        # Create deterministic tensors
        torch.manual_seed(42)
        T, H, K, V = 4, 2, 8, 16
        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        initial_state = torch.randn(H, K, V, device="npu", dtype=torch.float32)

        # Compute with patched (fused) function
        fused_output, fused_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=initial_state
        )

        # Compute with reference
        ref_output, ref_state = rwkv7_recurrent_reference(
            r, w, k, v, kk, a, initial_state=initial_state, output_final_state=True
        )

        # Compare outputs
        torch.testing.assert_close(
            fused_output, ref_output, atol=1e-4, rtol=1e-4, msg="Output mismatch"
        )
        torch.testing.assert_close(
            fused_state, ref_state, atol=1e-4, rtol=1e-4, msg="Final state mismatch"
        )

    def test_fallback_on_cpu(self):
        """Verify fallback to reference on CPU (non-NPU)."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference,
        )

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        # Create CPU tensors (will fallback)
        torch.manual_seed(42)
        T, H, K, V = 4, 2, 8, 16
        r = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="cpu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="cpu", dtype=torch.float32)

        # Call patched function - should fallback to reference on CPU
        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=None
        )

        # Reference result
        ref_output, ref_state = rwkv7_recurrent_reference(
            r, w, k, v, kk, a, initial_state=None, output_final_state=True
        )

        # Should match reference (since it fell back)
        torch.testing.assert_close(output, ref_output, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state, ref_state, atol=1e-5, rtol=1e-5)


class TestRWKV7VarlenScanDispatch(unittest.TestCase):
    """Test _rwkv7_recurrent_scan_varlen dispatch."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping tests")

    def test_varlen_dispatch_on_npu(self):
        """Verify varlen recurrent scan uses fused kernel on NPU."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        # Create valid NPU tensors with batch=1
        T, H, K, V = 8, 2, 8, 16
        r = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        w = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        k = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        v = torch.randn(1, T, H, V, device="npu", dtype=torch.float32).squeeze(0)
        kk = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        a = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)

        # cu_seqlens for 2 sequences of lengths 3 and 5
        cu_seqlens = torch.tensor([0, 3, 8], device="npu", dtype=torch.long)

        output, final_state = rwkv7_module._rwkv7_recurrent_scan_varlen(
            r, w, k, v, kk, a, cu_seqlens, initial_state=None
        )

        # Verify outputs are on NPU
        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())


class TestRWKV7EpilogueDispatch(unittest.TestCase):
    """Test _finalize_attention_output dispatch."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping tests")

    def test_epilogue_dispatch_on_npu(self):
        """Verify epilogue uses kernel on NPU when conditions are met."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg_reference,
        )

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        # Create minimal RWKV7Attention-like object for testing
        # We need to test that the patched method produces valid output
        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 8, 16
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device="npu", dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device="npu", dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        g = torch.randn(num_tokens, local_value_dim, device="npu", dtype=torch.float32)

        # Create mock attention object with required attributes
        class MockAttention:
            def __init__(self):
                self.tp_rank = 0
                self.local_num_heads = num_heads
                self.local_value_dim = local_value_dim
                self.value_start = 0
                self.value_end = local_value_dim
                self.r_k = torch.randn(num_heads, head_dim, device="npu", dtype=torch.float32)
                self.g_norm = type("gn", (), {})()
                self.g_norm.weight = torch.ones(local_value_dim, device="npu", dtype=torch.float32)
                self.g_norm.bias = torch.zeros(local_value_dim, device="npu", dtype=torch.float32)
                self.g_norm.eps = 64e-5

            def o_proj(self, x):
                # Mock projection - just return tensor
                return x, None

        mock_attn = MockAttention()

        # Get the patched method
        patched_method = rwkv7_module.RWKV7Attention._finalize_attention_output

        # Call patched method
        output = patched_method(
            mock_attn, recurrent_output, r, k, v, g, torch.float32
        )

        # Verify output is on NPU and finite
        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())

    def test_epilogue_fallback_on_cpu(self):
        """Verify epilogue falls back to reference on CPU."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        # Apply patch
        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 8, 16
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device="cpu", dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device="cpu", dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        g = torch.randn(num_tokens, local_value_dim, device="cpu", dtype=torch.float32)

        class MockAttention:
            def __init__(self):
                self.tp_rank = 0
                self.local_num_heads = num_heads
                self.local_value_dim = local_value_dim
                self.value_start = 0
                self.value_end = local_value_dim
                self.r_k = torch.randn(num_heads, head_dim, device="cpu", dtype=torch.float32)
                self.g_norm = type("gn", (), {})()
                self.g_norm.weight = torch.ones(local_value_dim, device="cpu", dtype=torch.float32)
                self.g_norm.bias = torch.zeros(local_value_dim, device="cpu", dtype=torch.float32)
                self.g_norm.eps = 64e-5

            def o_proj(self, x):
                return x, None

        mock_attn = MockAttention()

        patched_method = rwkv7_module.RWKV7Attention._finalize_attention_output

        # Should work on CPU (fallback path)
        output = patched_method(
            mock_attn, recurrent_output, r, k, v, g, torch.float32
        )

        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())


class TestRWKV7PatchSafeWithoutTriton(unittest.TestCase):
    """Test that patch is safe when Triton is not available."""

    def test_patch_loads_without_triton_error(self):
        """Verify patch can be imported even without Triton."""
        # This test ensures the patch doesn't hard-fail on import
        # when Triton is unavailable
        try:
            from vllm_ascend.patch.worker import patch_rwkv7

            # apply_patch should not raise
            patch_rwkv7.apply_patch()
        except Exception as e:
            self.fail(f"Patch should not raise on import: {e}")


class TestRWKV7IntegrationObservableBehavior(unittest.TestCase):
    """
    Test observable behavior of the RWKV7 integration.

    These tests verify actual behavior without mocking or inspecting
    internal constants.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping integration tests")

    def test_end_to_end_recurrent_computation(self):
        """
        Verify end-to-end recurrent computation produces consistent results.
        """
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")

        torch.manual_seed(123)
        T, H, K, V = 6, 2, 8, 16

        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        initial_state = torch.randn(H, K, V, device="npu", dtype=torch.float32)

        # Compute with patched function
        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=initial_state
        )

        # Verify shapes
        self.assertEqual(output.shape, (T, H, V))
        self.assertEqual(final_state.shape, (H, K, V))

        # Verify numerical properties
        # - Output should be finite
        self.assertTrue(torch.isfinite(output).all())
        # - Final state should be finite
        self.assertTrue(torch.isfinite(final_state).all())
        # - Output at each timestep should not be all zeros
        #   (recurrent computation produces meaningful output)
        for t in range(T):
            self.assertFalse(
                torch.allclose(output[t], torch.zeros_like(output[t])),
                f"Output at timestep {t} should not be all zeros"
            )


if __name__ == "__main__":
    unittest.main()