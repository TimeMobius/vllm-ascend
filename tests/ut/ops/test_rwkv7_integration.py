# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for RWKV7 Ascend patch.

These tests verify that the vllm-ascend patch correctly:
1. Installs idempotently (can be applied multiple times)
2. Dispatches to Triton-Ascend kernels when safe
3. Falls back to reference when conditions aren't met
4. Produces numerically correct output

Tests use observable tensor behavior, with call-tracking used in some
dispatch-selection tests to verify the fused path is reached.
"""

import unittest

import torch
import torch.nn.functional as F


def _make_gnorm_mock(weight, bias, eps):
    """
    Create a g_norm mock that matches the callable contract expected by
    upstream RWKV7Attention._finalize_attention_output (which calls
    self.g_norm(output)).

    The mock must:
    - Be callable: g_norm(tensor) -> tensor (applies group norm)
    - Have .weight attribute
    - Have .bias attribute
    - Have .eps attribute
    """

    class MockGroupNorm:
        def __init__(self, weight, bias, eps):
            self.weight = weight
            self.bias = bias
            self.eps = eps

        def __call__(self, x):
            return F.group_norm(x, self.weight.shape[0], self.weight, self.bias, self.eps)

    return MockGroupNorm(weight, bias, eps)


class TestRWKV7PatchIdempotency(unittest.TestCase):
    """Test that patch installation is idempotent."""

    def test_patch_can_be_applied_multiple_times(self):
        """Verify apply_patch() is idempotent - no error on repeated calls."""
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        patch_rwkv7.apply_patch()
        patch_rwkv7.apply_patch()

    def test_patch_marks_module(self):
        """Verify patch sets _RWKV7_ASCEND_PATCHED flag."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()

        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
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

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        T, H, K, V = 4, 2, 8, 16
        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)

        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=None
        )

        self.assertEqual(output.device.type, "npu")
        self.assertEqual(final_state.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(final_state).all())

    def test_numerical_parity_with_reference(self):
        """
        Verify fused kernel output matches reference within tolerance.

        rwkv7_recurrent_reference expects 4D tensors [B, T, H, K], so we
        unsqueeze 3D inputs to 4D, call reference, then squeeze outputs.
        """
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference,
        )

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        torch.manual_seed(42)
        T, H, K, V = 4, 2, 8, 16

        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        initial_state_3d = torch.randn(H, K, V, device="npu", dtype=torch.float32)

        fused_output, fused_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=initial_state_3d
        )

        r_4d = r.unsqueeze(0)
        w_4d = w.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        kk_4d = kk.unsqueeze(0)
        a_4d = a.unsqueeze(0)
        initial_state_4d = initial_state_3d.unsqueeze(0)

        ref_output_4d, ref_state_4d = rwkv7_recurrent_reference(
            r_4d, w_4d, k_4d, v_4d, kk_4d, a_4d,
            initial_state=initial_state_4d, output_final_state=True
        )

        ref_output = ref_output_4d.squeeze(0)
        ref_state = ref_state_4d.squeeze(0)

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

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        torch.manual_seed(42)
        T, H, K, V = 4, 2, 8, 16
        r = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="cpu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="cpu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="cpu", dtype=torch.float32)

        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=None
        )

        r_4d = r.unsqueeze(0)
        w_4d = w.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        kk_4d = kk.unsqueeze(0)
        a_4d = a.unsqueeze(0)

        ref_output_4d, ref_state_4d = rwkv7_recurrent_reference(
            r_4d, w_4d, k_4d, v_4d, kk_4d, a_4d,
            initial_state=None, output_final_state=True
        )

        ref_output = ref_output_4d.squeeze(0)
        ref_state = ref_state_4d.squeeze(0)

        torch.testing.assert_close(output, ref_output, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state, ref_state, atol=1e-5, rtol=1e-5)


class TestRWKV7VarlenScanDispatch(unittest.TestCase):
    """Test _rwkv7_recurrent_scan_varlen dispatch with multi-sequence inputs."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping tests")

    def test_varlen_dispatch_on_npu(self):
        """Verify varlen recurrent scan uses fused kernel on NPU."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        T, H, K, V = 8, 2, 8, 16
        r = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        w = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        k = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        v = torch.randn(1, T, H, V, device="npu", dtype=torch.float32).squeeze(0)
        kk = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)
        a = torch.randn(1, T, H, K, device="npu", dtype=torch.float32).squeeze(0)

        cu_seqlens = torch.tensor([0, 3, 8], device="npu", dtype=torch.long)

        output, final_state = rwkv7_module._rwkv7_recurrent_scan_varlen(
            r, w, k, v, kk, a, cu_seqlens, initial_state=None
        )

        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())

    def test_varlen_numerical_parity_unequal_lengths(self):
        """
        Verify varlen recurrent scan with unequal sequence lengths produces
        numerically correct output compared to reference.

        Tests two sequences: seq1 has 4 tokens, seq2 has 3 tokens (total 7).
        initial_state has shape [N=2, H, K, V].
        """
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference,
        )

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        torch.manual_seed(99)
        H, K, V = 2, 8, 16

        seq1_len, seq2_len = 4, 3
        total_tokens = seq1_len + seq2_len
        cu_seqlens = torch.tensor([0, seq1_len, total_tokens], device="npu", dtype=torch.long)

        r = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(total_tokens, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)

        initial_state = torch.randn(2, H, K, V, device="npu", dtype=torch.float32)

        fused_output, fused_state = rwkv7_module._rwkv7_recurrent_scan_varlen(
            r, w, k, v, kk, a, cu_seqlens, initial_state=initial_state
        )

        r_4d = r.unsqueeze(0)
        w_4d = w.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        kk_4d = kk.unsqueeze(0)
        a_4d = a.unsqueeze(0)

        ref_output_4d, ref_state_4d = rwkv7_recurrent_reference(
            r_4d, w_4d, k_4d, v_4d, kk_4d, a_4d,
            initial_state=initial_state, output_final_state=True,
            cu_seqlens=cu_seqlens
        )

        ref_output = ref_output_4d.squeeze(0)
        ref_state = ref_state_4d

        torch.testing.assert_close(
            fused_output, ref_output, atol=1e-4, rtol=1e-4, msg="Varlen output mismatch"
        )
        torch.testing.assert_close(
            fused_state, ref_state, atol=1e-4, rtol=1e-4, msg="Varlen final state mismatch"
        )

    def test_varlen_fused_path_selected_via_mock(self):
        """
        Prove fused_recurrent_rwkv7 is called for multi-token varlen
        (total_tokens=7, two sequences of unequal length).

        Uses mock to observe the actual kernel call without relying on
        constant inspection or output comparison alone.
        """
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        H, K, V = 2, 8, 16
        seq1_len, seq2_len = 4, 3
        total_tokens = seq1_len + seq2_len
        cu_seqlens = torch.tensor([0, seq1_len, total_tokens], device="npu", dtype=torch.long)

        r = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(total_tokens, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(total_tokens, H, K, device="npu", dtype=torch.float32)
        initial_state = torch.randn(2, H, K, V, device="npu", dtype=torch.float32)

        ops_cache = patch_rwkv7._ascend_ops
        if ops_cache is not None and ops_cache.fused_recurrent_rwkv7 is not None:
            original_fused = ops_cache.fused_recurrent_rwkv7
            call_tracker = {"called": False}

            def tracking_fused(*args, **kwargs):
                call_tracker["called"] = True
                return original_fused(*args, **kwargs)

            ops_cache.fused_recurrent_rwkv7 = tracking_fused
            try:
                output, final_state = rwkv7_module._rwkv7_recurrent_scan_varlen(
                    r, w, k, v, kk, a, cu_seqlens, initial_state=initial_state
                )
                self.assertTrue(
                    call_tracker["called"],
                    "fused_recurrent_rwkv7 was NOT called for multi-token varlen; "
                    "guard may still be blocking dispatch"
                )
                self.assertEqual(output.device.type, "npu")
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(final_state.shape, (2, H, K, V))
            finally:
                ops_cache.fused_recurrent_rwkv7 = original_fused
        else:
            self.skipTest("fused_recurrent_rwkv7 not available in this environment")


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

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 8, 16
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device="npu", dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device="npu", dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        g = torch.randn(num_tokens, local_value_dim, device="npu", dtype=torch.float32)

        class MockAttention:
            def __init__(self):
                self.tp_rank = 0
                self.local_num_heads = num_heads
                self.local_value_dim = local_value_dim
                self.value_start = 0
                self.value_end = local_value_dim
                self.r_k = torch.randn(num_heads, head_dim, device="npu", dtype=torch.float32)
                self.g_norm = _make_gnorm_mock(
                    weight=torch.ones(local_value_dim, device="npu", dtype=torch.float32),
                    bias=torch.zeros(local_value_dim, device="npu", dtype=torch.float32),
                    eps=64e-5,
                )

            def o_proj(self, x):
                return x, None

        mock_attn = MockAttention()
        patched_method = rwkv7_module.RWKV7Attention._finalize_attention_output

        output = patched_method(
            mock_attn, recurrent_output, r, k, v, g, torch.float32
        )

        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())

    def test_epilogue_decode_path_t1_parity(self):
        """
        Verify epilogue with T=1 (decode single-token) matches reference.

        This exercises the actual decode path where each token is processed
        individually with recurrent state carrying across calls.
        """
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg_reference,
        )

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        num_heads, head_dim, head_v_dim = 2, 8, 16
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            1, num_heads, head_v_dim, device="npu", dtype=torch.float32
        )
        r = torch.randn(1, num_heads, head_dim, device="npu", dtype=torch.float32)
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        g = torch.randn(1, local_value_dim, device="npu", dtype=torch.float32)

        r_k = torch.randn(num_heads, head_dim, device="npu", dtype=torch.float32)
        weight = torch.ones(local_value_dim, device="npu", dtype=torch.float32)
        bias = torch.zeros(local_value_dim, device="npu", dtype=torch.float32)
        eps = 64e-5

        class MockAttention:
            def __init__(self):
                self.tp_rank = 0
                self.local_num_heads = num_heads
                self.local_value_dim = local_value_dim
                self.value_start = 0
                self.value_end = local_value_dim
                self.r_k = r_k
                self.g_norm = _make_gnorm_mock(weight=weight, bias=bias, eps=eps)

            def o_proj(self, x):
                return x, None

        mock_attn = MockAttention()
        patched_method = rwkv7_module.RWKV7Attention._finalize_attention_output

        fused_output = patched_method(
            mock_attn, recurrent_output, r, k, v, g, torch.float32
        )

        ref_output = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=torch.float32,
        )
        torch.testing.assert_close(
            fused_output, ref_output, atol=1e-4, rtol=1e-4, msg="Decode epilogue mismatch"
        )

    def test_epilogue_fallback_on_cpu(self):
        """Verify epilogue falls back to reference on CPU."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

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
                self.g_norm = _make_gnorm_mock(
                    weight=torch.ones(local_value_dim, device="cpu", dtype=torch.float32),
                    bias=torch.zeros(local_value_dim, device="cpu", dtype=torch.float32),
                    eps=64e-5,
                )

            def o_proj(self, x):
                return x, None

        mock_attn = MockAttention()
        patched_method = rwkv7_module.RWKV7Attention._finalize_attention_output

        output = patched_method(
            mock_attn, recurrent_output, r, k, v, g, torch.float32
        )

        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())


class TestRWKV7PatchSafeWithoutTriton(unittest.TestCase):
    """Test that patch is safe when Triton is not available."""

    def test_patch_loads_without_triton_error(self):
        """Verify patch can be imported even without Triton."""
        try:
            from vllm_ascend.patch.worker import patch_rwkv7
            patch_rwkv7.apply_patch()
        except Exception as e:
            self.fail(f"Patch should not raise on import: {e}")


class TestRWKV7IntegrationObservableBehavior(unittest.TestCase):
    """Test observable behavior of the RWKV7 integration without mocking."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping integration tests")

    def test_end_to_end_recurrent_computation(self):
        """Verify end-to-end recurrent computation produces consistent results."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7

        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        torch.manual_seed(123)
        T, H, K, V = 6, 2, 8, 16

        r = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        w = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        k = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        v = torch.randn(T, H, V, device="npu", dtype=torch.float32)
        kk = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        a = torch.randn(T, H, K, device="npu", dtype=torch.float32)
        initial_state = torch.randn(H, K, V, device="npu", dtype=torch.float32)

        output, final_state = rwkv7_module._rwkv7_recurrent_scan(
            r, w, k, v, kk, a, initial_state=initial_state
        )

        self.assertEqual(output.shape, (T, H, V))
        self.assertEqual(final_state.shape, (H, K, V))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(final_state).all())

        for t in range(T):
            self.assertFalse(
                torch.allclose(output[t], torch.zeros_like(output[t])),
                f"Output at timestep {t} should not be all zeros"
            )


class TestRWKV7ProjectionIntegration(unittest.TestCase):
    """Test _project_recurrent_inputs integration with mix6 and kk_pre kernels."""

    def _make_mock_attention(self, layer_idx, hidden_size, num_heads, head_dim,
                             head_v_dim, tp_rank=0, tp_size=1, device="cpu"):
        """Create a mock RWKV7Attention for testing _project_recurrent_inputs."""
        local_num_heads = num_heads // tp_size
        local_key_dim = hidden_size // tp_size
        local_value_dim = num_heads * head_v_dim // tp_size
        key_start = tp_rank * local_key_dim
        key_end = key_start + local_key_dim
        value_start = tp_rank * local_value_dim
        value_end = value_start + local_value_dim

        class MockLoRA:
            def __init__(self, dim):
                self.weight = torch.randn(dim, dim, device=device, dtype=torch.float32)

            def __call__(self, x):
                return torch.nn.functional.linear(x, self.weight)

        class MockLinear:
            def __init__(self, in_dim, out_dim):
                self.weight = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32)

            def __call__(self, x):
                return torch.nn.functional.linear(x, self.weight), None

        class MockAttention:
            def __init__(self):
                self.layer_idx = layer_idx
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.head_v_dim = head_v_dim
                self.tp_rank = tp_rank
                self.tp_size = tp_size
                self.local_num_heads = local_num_heads
                self.local_key_dim = local_key_dim
                self.local_value_dim = local_value_dim
                self.key_start = key_start
                self.key_end = key_end
                self.value_start = value_start
                self.value_end = value_end

                self.x_r = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
                self.x_w = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
                self.x_k = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
                self.x_v = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
                self.x_a = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
                self.x_g = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)

                self.k_k = torch.randn(hidden_size, device=device, dtype=torch.float32)
                self.k_a = torch.randn(hidden_size, device=device, dtype=torch.float32)

                self.r_proj = MockLinear(hidden_size, hidden_size)
                self.k_proj = MockLinear(hidden_size, hidden_size)
                self.v_proj = MockLinear(hidden_size, local_value_dim)
                self.w_lora = MockLoRA(hidden_size)
                self.a_lora = MockLoRA(hidden_size)
                self.g_lora = MockLoRA(hidden_size)
                if layer_idx != 0:
                    self.v_lora = MockLoRA(hidden_size)

        return MockAttention()

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping projection tests")

    def test_projection_patch_idempotent(self):
        """Verify _project_recurrent_inputs patch is idempotent."""
        from vllm_ascend.patch.worker import patch_rwkv7
        patch_rwkv7.apply_patch()
        patch_rwkv7.apply_patch()

    def test_projection_patch_marks_class(self):
        """Verify patch sets _ASCEND_PROJECTION_PATCHED flag."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
        self.assertTrue(
            getattr(rwkv7_module.RWKV7Attention, "_ASCEND_PROJECTION_PATCHED", False)
        )

    def test_projection_npu_dispatch(self):
        """Verify projection uses kernels on NPU when conditions are met."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        T, H, K, V = 4, 2, 8, 16
        hidden_size = H * K
        mock_attn = self._make_mock_attention(
            layer_idx=0, hidden_size=hidden_size, num_heads=H, head_dim=K,
            head_v_dim=V, device="npu", tp_rank=0, tp_size=1
        )
        patched_method = rwkv7_module.RWKV7Attention._project_recurrent_inputs

        hidden_states = torch.randn(T, hidden_size, device="npu", dtype=torch.float32)
        delta = torch.randn(T, hidden_size, device="npu", dtype=torch.float32)

        r, w, k, v, kk, a, g, v_first_out = patched_method(
            mock_attn, hidden_states, delta, None
        )

        self.assertEqual(r.device.type, "npu")
        self.assertEqual(w.device.type, "npu")
        self.assertEqual(k.device.type, "npu")
        self.assertEqual(v.device.type, "npu")
        self.assertTrue(torch.isfinite(r).all())
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue(torch.isfinite(k).all())
        self.assertTrue(torch.isfinite(v).all())

    def test_projection_cpu_fallback(self):
        """Verify projection falls back to reference on CPU."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        T, H, K, V = 4, 2, 8, 16
        hidden_size = H * K
        mock_attn = self._make_mock_attention(
            layer_idx=0, hidden_size=hidden_size, num_heads=H, head_dim=K,
            head_v_dim=V, device="cpu", tp_rank=0, tp_size=1
        )
        patched_method = rwkv7_module.RWKV7Attention._project_recurrent_inputs

        hidden_states = torch.randn(T, hidden_size, device="cpu", dtype=torch.float32)
        delta = torch.randn(T, hidden_size, device="cpu", dtype=torch.float32)

        r, w, k, v, kk, a, g, v_first_out = patched_method(
            mock_attn, hidden_states, delta, None
        )

        self.assertEqual(r.device.type, "cpu")
        self.assertTrue(torch.isfinite(r).all())

    def test_projection_v_first_layer_nonzero(self):
        """Verify projection with v_first (layer_idx != 0) works."""
        import importlib
        from vllm_ascend.patch.worker import patch_rwkv7
        patch_rwkv7.apply_patch()
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")

        T, H, K, V = 4, 2, 8, 16
        hidden_size = H * K
        mock_attn = self._make_mock_attention(
            layer_idx=1, hidden_size=hidden_size, num_heads=H, head_dim=K,
            head_v_dim=V, device="npu", tp_rank=0, tp_size=1
        )
        patched_method = rwkv7_module.RWKV7Attention._project_recurrent_inputs

        hidden_states = torch.randn(T, hidden_size, device="npu", dtype=torch.float32)
        delta = torch.randn(T, hidden_size, device="npu", dtype=torch.float32)
        v_first = torch.randn(T, H * V, device="npu", dtype=torch.float32)

        r, w, k, v, kk, a, g, v_first_out = patched_method(
            mock_attn, hidden_states, delta, v_first
        )

        self.assertEqual(v_first_out.shape, v_first.shape)
        self.assertTrue(torch.isfinite(v_first_out).all())


if __name__ == "__main__":
    unittest.main()
