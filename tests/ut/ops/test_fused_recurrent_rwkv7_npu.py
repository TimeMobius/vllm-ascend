# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NPU-specific parity tests for fused_recurrent_rwkv7 triton-ascend implementation.

This module verifies that the triton-ascend implementation produces
identical outputs to the reference PyTorch implementation on NPU hardware.
"""

import unittest

import torch

from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
    fused_recurrent_rwkv7,
)
from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
    rwkv7_recurrent_reference_with_checkpoints,
)


class TestFusedRecurrentRWKV7Parity(unittest.TestCase):
    """Test parity between triton-ascend and reference implementations."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 tests")

    def test_fixed_length_forward(self):
        """Test fixed-length forward pass without state."""
        B, T, H, K, V = 2, 8, 4, 16, 32
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        ref_out, _, _ = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=False, output_checkpoint_states=False,
        )

        triton_out, _, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=False,
        )

        torch.testing.assert_close(
            triton_out, ref_out, atol=1e-4, rtol=1e-4,
            msg="Fixed-length forward pass output mismatch"
        )

    def test_fixed_length_with_initial_state(self):
        """Test fixed-length forward pass with initial state."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        initial_state = torch.randn(B, H, K, V, device='npu', dtype=torch.float32)

        ref_out, ref_state = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
        )

        triton_out, triton_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)

    def test_fixed_length_with_final_state(self):
        """Test fixed-length forward pass with final state output."""
        B, T, H, K, V = 2, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        ref_out, ref_state = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=True,
        )

        triton_out, triton_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=1.0, output_final_state=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)

    def test_varlen_forward(self):
        """Test variable-length forward pass."""
        T = 16
        H, K, V = 2, 8, 16
        cu_seqlens = torch.tensor([0, 5, 12, 16], device='npu', dtype=torch.int64)

        r = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(1, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)

        ref_out, ref_state = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            cu_seqlens=cu_seqlens, scale=1.0, output_final_state=True,
        )

        triton_out, triton_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            cu_seqlens=cu_seqlens, scale=1.0, output_final_state=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)

    def test_varlen_with_initial_state(self):
        """Test variable-length forward pass with initial state."""
        T = 16
        H, K, V = 2, 8, 16
        cu_seqlens = torch.tensor([0, 4, 10, 16], device='npu', dtype=torch.int64)
        N = len(cu_seqlens) - 1

        r = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(1, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(1, T, H, K, device='npu', dtype=torch.float32)
        initial_state = torch.randn(N, H, K, V, device='npu', dtype=torch.float32)

        ref_out, ref_state = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, cu_seqlens=cu_seqlens,
            scale=1.0, output_final_state=True,
        )

        triton_out, triton_state, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, cu_seqlens=cu_seqlens,
            scale=1.0, output_final_state=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)

    def test_checkpoint_outputs(self):
        """Test checkpoint state saving."""
        B, T, H, K, V = 1, 16, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        checkpoint_positions = torch.tensor([4, 12], device='npu', dtype=torch.int64)
        checkpoint_offsets = torch.tensor([0, 2], device='npu', dtype=torch.int64)

        ref_out, ref_state, ref_checkpoints = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        triton_out, triton_state, triton_checkpoints = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_checkpoints, ref_checkpoints, atol=1e-4, rtol=1e-4)

    def test_multi_sequence_checkpoints(self):
        """Test checkpoint saving with multiple sequences."""
        B, T, H, K, V = 2, 8, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)

        checkpoint_positions = torch.tensor([2, 4, 1, 3, 5, 7], device='npu', dtype=torch.int64)
        checkpoint_offsets = torch.tensor([0, 2, 6], device='npu', dtype=torch.int64)

        ref_out, ref_state, ref_checkpoints = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        triton_out, triton_state, triton_checkpoints = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            scale=1.0, output_final_state=True, output_checkpoint_states=True,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_state, ref_state, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_checkpoints, ref_checkpoints, atol=1e-4, rtol=1e-4)

    def test_scale_factor(self):
        """Test that scale factor is applied correctly."""
        B, T, H, K, V = 1, 4, 2, 8, 16
        r = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, V, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, K, device='npu', dtype=torch.float32)
        scale = 0.5

        ref_out, _ = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=scale, output_final_state=False,
        )

        triton_out, _, _ = fused_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=scale, output_final_state=False,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        B, T, H, K, V = 2, 8, 4, 16, 32
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

        self.assertEqual(out.shape, (B, T, H, V))
        self.assertEqual(final_state.shape, (B, H, K, V))
        self.assertEqual(out.device.type, 'npu')
        self.assertEqual(final_state.device.type, 'npu')


if __name__ == '__main__':
    unittest.main()