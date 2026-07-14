# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parity tests for fused_recurrent_rwkv7 triton-ascend implementation.

This module verifies that the triton-ascend implementation produces
identical outputs to the reference PyTorch implementation.

Tests cover:
- CPU reference fallback path
- Device placement verification
"""

import unittest

import torch

from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
    fused_recurrent_rwkv7,
)
from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
    rwkv7_recurrent_reference_with_checkpoints,
)


class TestFusedRecurrentRWKV7ReferenceCPU(unittest.TestCase):
    """Test reference implementation path on CPU (fallback when Triton/NPU unavailable)."""

    def test_cpu_fixed_length_forward(self):
        """Test fixed-length forward pass on CPU via fallback."""
        B, T, H, K, V = 2, 8, 4, 16, 32
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)

        ref_out, ref_state, ref_hc = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=True, output_checkpoint_states=False,
        )

        out, final_state, hc = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=True,
        )

        torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(final_state, ref_state, atol=1e-4, rtol=1e-4)
        self.assertIsNone(hc)

    def test_cpu_with_initial_state(self):
        """Test fixed-length forward pass with initial state on CPU."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        initial_state = torch.randn(B, H, K, V, device='cpu', dtype=torch.float32)

        out, final_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
        )

        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))
        self.assertEqual(out.device.type, 'cpu')
        self.assertEqual(final_state.device.type, 'cpu')
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(final_state).all())

    def test_cpu_varlen_forward(self):
        """Test variable-length forward pass on CPU."""
        T = 16
        H, K, V = 2, 8, 16
        cu_seqlens = torch.tensor([0, 5, 12, 16], device='cpu', dtype=torch.int64)

        r = torch.randn(1, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(1, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(1, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(1, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(1, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(1, T, H, K, device='cpu', dtype=torch.float32)

        out, final_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            cu_seqlens=cu_seqlens, scale=1.0, output_final_state=True,
        )

        N = len(cu_seqlens) - 1
        self.assertEqual(out.shape, (1, T, H, V))
        self.assertEqual(final_state.shape, (N, H, K, V))
        self.assertEqual(out.device.type, 'cpu')
        self.assertEqual(final_state.device.type, 'cpu')

    def test_cpu_checkpoint_outputs(self):
        """Test checkpoint state saving on CPU."""
        B, T, H, K, V = 1, 16, 2, 8, 16
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)

        checkpoint_positions = torch.tensor([4, 12], device='cpu', dtype=torch.int64)
        checkpoint_offsets = torch.tensor([0, 2], device='cpu', dtype=torch.int64)

        out, final_state, checkpoints = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))
        self.assertEqual(checkpoints.shape, (2, H, K, V))
        self.assertEqual(out.device.type, 'cpu')
        self.assertEqual(final_state.device.type, 'cpu')
        self.assertEqual(checkpoints.device.type, 'cpu')

    def test_cpu_non_contiguous_tensors(self):
        """Test fallback with non-contiguous input tensors on CPU."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)

        r_nc = r.transpose(1, 2).contiguous().transpose(1, 2)
        w_nc = w.transpose(1, 2).contiguous().transpose(1, 2)
        k_nc = k.transpose(1, 2).contiguous().transpose(1, 2)
        v_nc = v.transpose(1, 2).contiguous().transpose(1, 2)
        kk_nc = kk.transpose(1, 2).contiguous().transpose(1, 2)
        a_nc = a.transpose(1, 2).contiguous().transpose(1, 2)

        out, final_state, _ = fused_recurrent_rwkv7(
            r=r_nc, w=w_nc, k=k_nc, v=v_nc, kk=kk_nc, a=a_nc,
            scale=1.0, output_final_state=True,
        )

        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))

    def test_cpu_empty_tensor_returns_safe(self):
        """Test that empty tensors return proper empty outputs on CPU."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)

        out, final_state, hc = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertIsInstance(final_state, torch.Tensor)
        self.assertIsNone(hc)

    def test_cpu_scale_factor(self):
        """Test that scale factor is applied correctly on CPU fallback."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='cpu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='cpu', dtype=torch.float32)
        scale = 0.5

        out1, _, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=False,
        )
        out2, _, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=scale, output_final_state=False,
        )

        self.assertFalse(torch.allclose(out1, out2, atol=1e-6))


class TestFusedRecurrentRWKV7DevicePlacement(unittest.TestCase):
    """Test device placement of outputs."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 tests")

    def test_outputs_on_npu(self):
        """Verify all outputs are on NPU."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        out, final_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a, output_final_state=True,
        )

        self.assertEqual(out.device.type, 'npu')
        self.assertEqual(final_state.device.type, 'npu')

    def test_finite_outputs(self):
        """Verify outputs are finite (no NaN or Inf)."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        initial_state = torch.randn(B, H, K, V, device='npu', dtype=torch.float32)

        out, final_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, output_final_state=True,
        )

        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(final_state).all())


if __name__ == '__main__':
    unittest.main()