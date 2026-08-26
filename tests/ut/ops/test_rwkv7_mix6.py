# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RWKV7 mix6 operation parity tests.

This module tests that the vllm_ascend triton-ascend implementation of
rwkv7_mix6 produces results consistent with the upstream reference
implementation.

Tests cover:
1. Correctness: triton-ascend output matches torch reference output
2. Shape preservation: output shapes match input shapes
3. Dtype handling: output dtype respects input dtypes
4. Device placement: outputs are on correct device (NPU)
5. Edge cases: empty tensors, boundary conditions
"""

import unittest

import torch

from vllm_ascend.ops.triton.fla.rwkv7_mix6 import rwkv7_mix6, rwkv7_mix6_reference


class TestRWKV7Mix6Parity(unittest.TestCase):
    """Test parity between triton-ascend rwkv7_mix6 and reference implementation."""

    @classmethod
    def setUpClass(cls):
        """Verify NPU is available before running tests."""
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping rwkv7_mix6 tests")

    def _run_parity_check(
        self,
        batch: int,
        seq: int,
        hidden: int,
        dtype: torch.dtype = torch.float32,
    ):
        """Helper to run parity check with given parameters."""
        hidden_states = torch.randn(
            batch, seq, hidden, device="npu", dtype=dtype
        )
        delta = torch.randn(batch, seq, hidden, device="npu", dtype=dtype)
        x_r = torch.randn(hidden, device="npu", dtype=dtype)
        x_w = torch.randn(hidden, device="npu", dtype=dtype)
        x_k = torch.randn(hidden, device="npu", dtype=dtype)
        x_v = torch.randn(hidden, device="npu", dtype=dtype)
        x_a = torch.randn(hidden, device="npu", dtype=dtype)
        x_g = torch.randn(hidden, device="npu", dtype=dtype)

        # Compute reference output
        ref_xr, ref_xw, ref_xk, ref_xv, ref_xa, ref_xg = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Compute triton output
        triton_xr, triton_xw, triton_xk, triton_xv, triton_xa, triton_xg = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Compare outputs
        torch.testing.assert_close(triton_xr, ref_xr, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_xw, ref_xw, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_xk, ref_xk, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_xv, ref_xv, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_xa, ref_xa, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_xg, ref_xg, atol=1e-4, rtol=1e-4)

    def test_parity_small(self):
        """Test parity with small hidden dimension."""
        self._run_parity_check(batch=2, seq=4, hidden=32)

    def test_parity_medium(self):
        """Test parity with medium hidden dimension."""
        self._run_parity_check(batch=2, seq=8, hidden=128)

    def test_parity_large(self):
        """Test parity with large hidden dimension."""
        self._run_parity_check(batch=1, seq=4, hidden=512)

    def test_parity_bfloat16(self):
        """Test parity with bfloat16 dtype."""
        self._run_parity_check(batch=2, seq=4, hidden=64, dtype=torch.bfloat16)

    def test_parity_float16(self):
        """Test parity with float16 dtype."""
        self._run_parity_check(batch=2, seq=4, hidden=64, dtype=torch.float16)

    def test_output_shapes(self):
        """Verify output shapes match input shapes."""
        batch, seq, hidden = 3, 7, 96
        hidden_states = torch.randn(batch, seq, hidden, device="npu")
        delta = torch.randn(batch, seq, hidden, device="npu")
        x_r = torch.randn(hidden, device="npu")
        x_w = torch.randn(hidden, device="npu")
        x_k = torch.randn(hidden, device="npu")
        x_v = torch.randn(hidden, device="npu")
        x_a = torch.randn(hidden, device="npu")
        x_g = torch.randn(hidden, device="npu")

        xr, xw, xk, xv, xa, xg = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        expected_shape = (batch, seq, hidden)
        self.assertEqual(xr.shape, expected_shape)
        self.assertEqual(xw.shape, expected_shape)
        self.assertEqual(xk.shape, expected_shape)
        self.assertEqual(xv.shape, expected_shape)
        self.assertEqual(xa.shape, expected_shape)
        self.assertEqual(xg.shape, expected_shape)

    def test_output_device(self):
        """Verify outputs are on NPU device."""
        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="npu")
        delta = torch.randn(batch, seq, hidden, device="npu")
        x_r = torch.randn(hidden, device="npu")
        x_w = torch.randn(hidden, device="npu")
        x_k = torch.randn(hidden, device="npu")
        x_v = torch.randn(hidden, device="npu")
        x_a = torch.randn(hidden, device="npu")
        x_g = torch.randn(hidden, device="npu")

        outputs = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        for output in outputs:
            self.assertEqual(output.device.type, "npu")

    def test_output_dtype(self):
        """Verify output dtype respects input dtypes."""
        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="npu", dtype=torch.bfloat16)
        delta = torch.randn(batch, seq, hidden, device="npu", dtype=torch.bfloat16)
        x_r = torch.randn(hidden, device="npu", dtype=torch.float32)

        xr, *_ = rwkv7_mix6(
            hidden_states, delta, x_r,
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
        )

        # Output dtype should be result_type(hidden_states, x_r) = bfloat16
        self.assertEqual(xr.dtype, torch.bfloat16)

    def test_hidden_states_delta_shape_mismatch(self):
        """Verify error is raised when hidden_states and delta shapes differ."""
        hidden_states = torch.randn(2, 4, 64, device="npu")
        delta = torch.randn(2, 5, 64, device="npu")  # Different seq length
        x_r = torch.randn(64, device="npu")
        x_w = torch.randn(64, device="npu")
        x_k = torch.randn(64, device="npu")
        x_v = torch.randn(64, device="npu")
        x_a = torch.randn(64, device="npu")
        x_g = torch.randn(64, device="npu")

        with self.assertRaises(ValueError) as context:
            rwkv7_mix6(hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g)

        self.assertIn("must have the same shape", str(context.exception))

    def test_3d_hidden_states_accepted(self):
        """Verify 3D hidden_states (batch, seq, hidden) is accepted and output preserves shape."""
        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="npu")  # 3D
        delta = torch.randn(batch, seq, hidden, device="npu")
        x_r = torch.randn(hidden, device="npu")
        x_w = torch.randn(hidden, device="npu")
        x_k = torch.randn(hidden, device="npu")
        x_v = torch.randn(hidden, device="npu")
        x_a = torch.randn(hidden, device="npu")
        x_g = torch.randn(hidden, device="npu")

        # Should not raise - wrapper flattens 3D to 2D internally
        xr, xw, xk, xv, xa, xg = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Output should preserve 3D shape
        expected_shape = (batch, seq, hidden)
        self.assertEqual(xr.shape, expected_shape)
        self.assertEqual(xw.shape, expected_shape)
        self.assertEqual(xk.shape, expected_shape)
        self.assertEqual(xv.shape, expected_shape)
        self.assertEqual(xa.shape, expected_shape)
        self.assertEqual(xg.shape, expected_shape)

    def test_single_token_sequence(self):
        """Test with single token sequence."""
        self._run_parity_check(batch=1, seq=1, hidden=128)

    def test_large_batch(self):
        """Test with large batch size."""
        self._run_parity_check(batch=32, seq=1, hidden=64)

    def test_parity_large_grid(self):
        """Regression: logical grid >= 65536 must not exceed the Ascend coreDim limit.

        With block_size = next_power_of_2(64) = 64, batch * seq = 128 * 512
        gives numel = 4,194,304 and num_blocks = 65,536, which exceeds the
        Ascend coreDim limit of 65,535. The NPU launch must bound the physical
        grid by the vector-core count and cover all logical blocks via the
        kernel-internal grid-stride loop.
        """
        self._run_parity_check(batch=128, seq=512, hidden=64)


class TestRWKV7Mix6ReferenceOnly(unittest.TestCase):
    """Test the reference implementation directly (for debugging)."""

    @classmethod
    def setUpClass(cls):
        """Verify NPU is available before running tests."""
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping rwkv7_mix6 tests")

    def test_reference_computes_correctly(self):
        """Verify reference implementation produces expected values."""
        # Simple case with known values
        batch, seq, hidden = 1, 1, 4
        hidden_states = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]], device="npu"
        )
        delta = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0]], device="npu"
        )
        x_r = torch.tensor([1.0, 0.0, 0.0, 0.0], device="npu")  # Selects first element

        xr, *_ = rwkv7_mix6_reference(
            hidden_states, delta, x_r,
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
            torch.randn(hidden, device="npu"),
        )

        # xr = hidden_states + delta * x_r = [1,2,3,4] + [1,0,0,0] = [2,2,3,4]
        expected = torch.tensor([[2.0, 2.0, 3.0, 4.0]], device="npu")
        torch.testing.assert_close(xr, expected, atol=1e-5, rtol=1e-5)


class TestRWKV7Mix6ReferenceCPU(unittest.TestCase):
    """Test reference implementation on CPU (no NPU required).

    These tests exercise the CPU/reference path which does NOT require NPU.
    They verify the pure torch fallback path works correctly.
    """

    def test_reference_3d_cpu(self):
        """Verify reference works with 3D input on CPU."""
        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        delta = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        x_r = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_w = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_k = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_v = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_a = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_g = torch.randn(hidden, device="cpu", dtype=torch.float32)

        xr, xw, xk, xv, xa, xg = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Shape should be preserved
        self.assertEqual(xr.shape, hidden_states.shape)
        self.assertEqual(xw.shape, hidden_states.shape)
        self.assertEqual(xk.shape, hidden_states.shape)
        self.assertEqual(xv.shape, hidden_states.shape)
        self.assertEqual(xa.shape, hidden_states.shape)
        self.assertEqual(xg.shape, hidden_states.shape)

    def test_reference_2d_cpu(self):
        """Verify reference works with 2D input on CPU."""
        batch_seq, hidden = 8, 64
        hidden_states = torch.randn(batch_seq, hidden, device="cpu", dtype=torch.float32)
        delta = torch.randn(batch_seq, hidden, device="cpu", dtype=torch.float32)
        x_r = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_w = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_k = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_v = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_a = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_g = torch.randn(hidden, device="cpu", dtype=torch.float32)

        xr, xw, xk, xv, xa, xg = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Shape should be preserved
        self.assertEqual(xr.shape, hidden_states.shape)

    def test_reference_parity_cpu(self):
        """Verify reference output matches expected addcmul semantics."""
        # Simple known values
        hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
        delta = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device="cpu")
        x_r = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cpu")

        xr, *_ = rwkv7_mix6_reference(
            hidden_states, delta, x_r,
            torch.randn(4, device="cpu"),
            torch.randn(4, device="cpu"),
            torch.randn(4, device="cpu"),
            torch.randn(4, device="cpu"),
            torch.randn(4, device="cpu"),
        )

        # xr = hidden_states + delta * x_r = [1,2,3,4] + [1,0,0,0] = [2,2,3,4]
        expected = torch.tensor([[2.0, 2.0, 3.0, 4.0]], device="cpu")
        torch.testing.assert_close(xr, expected, atol=1e-5, rtol=1e-5)

    def test_wrapper_reference_fallback_cpu(self):
        """Verify wrapper falls back to reference on CPU and produces correct output."""
        batch, seq, hidden = 2, 4, 64
        hidden_states = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        delta = torch.randn(batch, seq, hidden, device="cpu", dtype=torch.float32)
        x_r = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_w = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_k = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_v = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_a = torch.randn(hidden, device="cpu", dtype=torch.float32)
        x_g = torch.randn(hidden, device="cpu", dtype=torch.float32)

        # Call wrapper (should fall back to reference on CPU)
        wrapper_xr, wrapper_xw, wrapper_xk, wrapper_xv, wrapper_xa, wrapper_xg = rwkv7_mix6(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Call reference directly
        ref_xr, ref_xw, ref_xk, ref_xv, ref_xa, ref_xg = rwkv7_mix6_reference(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        )

        # Should match
        torch.testing.assert_close(wrapper_xr, ref_xr, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_xw, ref_xw, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_xk, ref_xk, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_xv, ref_xv, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_xa, ref_xa, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wrapper_xg, ref_xg, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()