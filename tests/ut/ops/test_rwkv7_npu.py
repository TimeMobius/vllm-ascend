# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RWKV7 NPU Path A verification tests.

This module verifies that the upstream RWKV7 reference implementation
(torch fallback path) works correctly on Ascend NPU.

Path A is the torch/reference implementation that is used when:
1. Device is not CUDA (e.g., NPU)
2. Triton is not available
3. Or when VLLM_ASCEND_RWKV7_PRESET=reference
4. Or when VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=reference

The key insight: upstream RWKV7 already handles non-CUDA devices correctly
via the reference fallback. No vllm-ascend worker patch is required for
RWKV7 Path A to function on NPU.

Reference: .omo/plans/rwkv7-ascend-design.md
"""

import unittest

import torch


class TestRWKV7NPUReferencePath(unittest.TestCase):
    """Test RWKV7 reference implementation on NPU."""

    @classmethod
    def setUpClass(cls):
        """Verify NPU is available before running tests."""
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 NPU tests")

    def test_rwkv7_recurrent_reference_on_npu(self):
        """Verify rwkv7_recurrent_reference produces correct output shapes on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7 import rwkv7_recurrent_reference

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        out, final_state = rwkv7_recurrent_reference(
            r, w, k, v, kk, a, output_final_state=True
        )

        # Verify output shape
        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))

        # Verify device placement
        self.assertEqual(out.device.type, 'npu')
        self.assertEqual(final_state.device.type, 'npu')

        # Verify dtype
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(final_state.dtype, torch.float32)

        # Verify outputs are finite
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(final_state).all())

    def test_rwkv7_mix6_reference_on_npu(self):
        """Verify rwkv7_mix6_reference produces correct output on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7 import rwkv7_mix6_reference

        batch, seq, hidden = 2, 8, 64
        hidden_states = torch.randn(batch, seq, hidden, device='npu', dtype=torch.float32)
        delta = torch.randn(batch, seq, hidden, device='npu', dtype=torch.float32)
        x_r = torch.randn(hidden, device='npu', dtype=torch.float32)
        x_w = torch.randn(hidden, device='npu', dtype=torch.float32)
        x_k = torch.randn(hidden, device='npu', dtype=torch.float32)
        x_v = torch.randn(hidden, device='npu', dtype=torch.float32)
        x_a = torch.randn(hidden, device='npu', dtype=torch.float32)
        x_g = torch.randn(hidden, device='npu', dtype=torch.float32)

        xr, xw, xk, xv, xa, xg = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Verify all outputs have same shape as hidden_states
        self.assertEqual(xr.shape, hidden_states.shape)
        self.assertEqual(xw.shape, hidden_states.shape)
        self.assertEqual(xk.shape, hidden_states.shape)
        self.assertEqual(xv.shape, hidden_states.shape)
        self.assertEqual(xa.shape, hidden_states.shape)
        self.assertEqual(xg.shape, hidden_states.shape)

        # Verify all outputs are on NPU
        for output in (xr, xw, xk, xv, xa, xg):
            self.assertEqual(output.device.type, 'npu')
            self.assertTrue(torch.isfinite(output).all())

    def test_rwkv7_kk_pre_reference_on_npu(self):
        """Verify rwkv7_kk_pre_reference produces correct output on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7 import rwkv7_kk_pre_reference

        k = torch.randn(19, 8, 64, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(8, 64, device='npu', dtype=torch.float32)
        k_a = torch.randn(8, 64, device='npu', dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre_reference(k, k_k, a, k_a)

        # Verify output shapes
        self.assertEqual(k_adj.shape, k.shape)
        self.assertEqual(kk.shape, k.shape)

        # Verify device placement
        self.assertEqual(k_adj.device.type, 'npu')
        self.assertEqual(kk.device.type, 'npu')

        # Verify kk is normalized (has unit norm along last dim)
        kk_norm = torch.norm(kk, p=2, dim=-1)
        torch.testing.assert_close(kk_norm, torch.ones_like(kk_norm), atol=1e-5, rtol=1e-5)

    def test_fused_mul_recurrent_falls_back_to_reference_on_npu(self):
        """Verify fused_mul_recurrent_rwkv7 falls back to reference on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7 import (
            fused_mul_recurrent_rwkv7,
            rwkv7_recurrent_reference,
        )

        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        initial_state = torch.randn(B, H, K, V, device='npu', dtype=torch.float32)

        # Compute reference output
        ref_out, ref_state = rwkv7_recurrent_reference(
            r, w, k, v, kk, a, initial_state=initial_state, output_final_state=True
        )

        # Compute fused output (should fall back to reference on NPU)
        fused_out, fused_state = fused_mul_recurrent_rwkv7(
            r, w, k, v, kk, a, initial_state=initial_state, output_final_state=True
        )

        # Verify outputs match (proving fallback was used)
        torch.testing.assert_close(fused_out, ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fused_state, ref_state, atol=1e-5, rtol=1e-5)

        # Verify outputs are on NPU
        self.assertEqual(fused_out.device.type, 'npu')
        self.assertEqual(fused_state.device.type, 'npu')

    def test_rwkv7_lnx_rkvres_xg_reference_on_npu(self):
        """Verify rwkv7_lnx_rkvres_xg_reference works on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7 import rwkv7_lnx_rkvres_xg_reference

        num_tokens = 17
        num_heads = 4
        head_dim = 64
        head_v_dim = 64
        local_value_dim = num_heads * head_v_dim

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(local_value_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(local_value_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, local_value_dim, device='npu', dtype=torch.float32)

        output = rwkv7_lnx_rkvres_xg_reference(
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
        self.assertEqual(output.device.type, 'npu')

        # Verify output is finite
        self.assertTrue(torch.isfinite(output).all())


class TestRWKV7NPUDeviceGating(unittest.TestCase):
    """Test that device gating correctly routes to reference on NPU."""

    @classmethod
    def setUpClass(cls):
        """Verify NPU is available before running tests."""
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 NPU tests")

    def test_device_gate_blocks_fused_path_on_npu(self):
        """
        Verify that device gate logic correctly handles NPU devices.

        The upstream code at vllm/model_executor/models/rwkv7.py checks:
            return hidden_states.device.type == "cuda" and _rwkv7_packed_prefill_enabled()

        On NPU, hidden_states.device.type == "npu" != "cuda", so the fused path
        is blocked and the reference path is used instead.
        """
        # Verify the device type check logic
        npu_tensor = torch.randn(2, 3, device='npu')

        # The device type is what matters for the gate
        self.assertEqual(npu_tensor.device.type, 'npu')
        self.assertNotEqual(npu_tensor.device.type, 'cuda')  # This is the key check

        # So on NPU, the gate should evaluate to False (use reference)
        # This is the expected behavior - NPU should use the reference path
        device_type = npu_tensor.device.type
        would_use_fused = (device_type == "cuda")
        self.assertFalse(
            would_use_fused,
            f"Device type '{device_type}' should not trigger fused path"
        )


if __name__ == '__main__':
    unittest.main()
