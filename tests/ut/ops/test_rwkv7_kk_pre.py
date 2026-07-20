# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RWKV7 kk_pre parity tests for triton-ascend implementation.

This module tests the rwkv7_kk_pre operation ported from upstream vLLM,
verifying parity between the triton-ascend implementation and the reference.

The kk_pre operation performs key preprocessing for RWKV7:
1. kk = normalize(k * k_k, dim=-1) - normalized key-key product
2. k_adj = k * (1 + (a - 1) * k_a) - adjusted key with activation

Reference: vllm/model_executor/layers/fla/ops/rwkv7.py::rwkv7_kk_pre
"""

import unittest

import torch


class TestRWKV7KKPreCPUReference(unittest.TestCase):
    """CPU-only tests for the reference implementation.

    These tests verify the reference implementation correctness
    without requiring NPU hardware.
    """

    def test_reference_output_shapes(self):
        """Verify reference produces correct output shapes."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference

        T, H, K = 19, 8, 64
        k = torch.randn(T, H, K, dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, dtype=torch.float32)
        k_a = torch.randn(H, K, dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

        self.assertEqual(k_adj.shape, k.shape)
        self.assertEqual(kk.shape, k.shape)

    def test_reference_kk_normalized(self):
        """Verify kk output is L2 normalized along last dim."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference

        T, H, K = 8, 4, 64
        k = torch.randn(T, H, K, dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, dtype=torch.float32)
        k_a = torch.randn(H, K, dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

        kk_norm = torch.norm(kk, p=2, dim=-1)
        torch.testing.assert_close(kk_norm, torch.ones_like(kk_norm), atol=1e-5, rtol=1e-5)

    def test_reference_different_sizes(self):
        """Verify reference across different tensor sizes."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference

        test_cases = [
            (1, 1, 32),
            (8, 4, 64),
            (32, 16, 128),
            (19, 8, 64),
            (5, 3, 17),
        ]

        for T, H, K in test_cases:
            with self.subTest(T=T, H=H, K=K):
                k = torch.randn(T, H, K, dtype=torch.float32)
                a = torch.randn_like(k)
                k_k = torch.randn(H, K, dtype=torch.float32)
                k_a = torch.randn(H, K, dtype=torch.float32)

                k_adj, kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

                self.assertEqual(k_adj.shape, (T, H, K))
                self.assertEqual(kk.shape, (T, H, K))

    def test_reference_dtypes(self):
        """Verify reference handles different dtypes correctly."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference

        T, H, K = 8, 4, 64

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            with self.subTest(dtype=dtype):
                k = torch.randn(T, H, K, dtype=dtype)
                a = torch.randn_like(k)
                k_k = torch.randn(H, K, dtype=dtype)
                k_a = torch.randn(H, K, dtype=dtype)

                k_adj, kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

                self.assertTrue(torch.isfinite(k_adj).all())
                self.assertTrue(torch.isfinite(kk).all())

                kk_norm = torch.norm(kk.float(), p=2, dim=-1)
                torch.testing.assert_close(
                    kk_norm, torch.ones_like(kk_norm), atol=1e-3, rtol=1e-3
                )

    def test_reference_empty_tensor(self):
        """Verify reference handles empty tensors."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference

        k = torch.randn(0, 8, 64, dtype=torch.float32)
        a = torch.randn(0, 8, 64, dtype=torch.float32)
        k_k = torch.randn(8, 64, dtype=torch.float32)
        k_a = torch.randn(8, 64, dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
        self.assertEqual(k_adj.shape, k.shape)
        self.assertEqual(kk.shape, k.shape)

    def test_reference_matches_upstream(self):
        """Verify our reference matches upstream implementation."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre_reference
        from vllm_ascend.ops.triton.fla.rwkv7 import (
            rwkv7_kk_pre_reference as upstream
        )

        T, H, K = 19, 8, 64
        torch.manual_seed(42)
        k = torch.randn(T, H, K, dtype=torch.float32)
        a = torch.randn_like(k)
        torch.manual_seed(43)
        k_k = torch.randn(H, K, dtype=torch.float32)
        k_a = torch.randn(H, K, dtype=torch.float32)

        our_k_adj, our_kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
        upstream_k_adj, upstream_kk = upstream(k=k, k_k=k_k, a=a, k_a=k_a)

        torch.testing.assert_close(our_k_adj, upstream_k_adj, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(our_kk, upstream_kk, atol=1e-6, rtol=1e-6)


class TestRWKV7KKPreNPU(unittest.TestCase):
    """NPU-specific tests for the Triton kernel path.

    These tests require NPU hardware and verify the triton-ascend
    implementation matches the reference.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping NPU tests")

    def test_triton_output_shapes(self):
        """Verify triton kernel produces correct output shapes."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 19, 8, 64
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        self.assertEqual(k_adj.shape, k.shape)
        self.assertEqual(kk.shape, k.shape)

    def test_triton_device_placement(self):
        """Verify triton outputs are on NPU."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 19, 8, 64
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        self.assertEqual(k_adj.device.type, 'npu')
        self.assertEqual(kk.device.type, 'npu')

    def test_triton_kk_normalized(self):
        """Verify kk output is L2 normalized along last dim."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 19, 8, 64
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        kk_norm = torch.norm(kk, p=2, dim=-1)
        torch.testing.assert_close(kk_norm, torch.ones_like(kk_norm), atol=1e-5, rtol=1e-5)

    def test_triton_parity_with_reference(self):
        """Verify triton matches reference implementation."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre, rwkv7_kk_pre_reference

        T, H, K = 19, 8, 64
        torch.manual_seed(42)
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        torch.manual_seed(43)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        ref_k_adj, ref_kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
        triton_k_adj, triton_kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        torch.testing.assert_close(triton_k_adj, ref_k_adj, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(triton_kk, ref_kk, atol=1e-4, rtol=1e-4)

    def test_triton_parity_different_sizes(self):
        """Verify triton parity across different tensor sizes."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre, rwkv7_kk_pre_reference

        test_cases = [
            (1, 1, 32),
            (8, 4, 64),
            (32, 16, 128),
            (19, 8, 64),
        ]

        for T, H, K in test_cases:
            with self.subTest(T=T, H=H, K=K):
                torch.manual_seed(42)
                k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
                a = torch.randn_like(k)
                torch.manual_seed(43)
                k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
                k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

                ref_k_adj, ref_kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
                triton_k_adj, triton_kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

                torch.testing.assert_close(
                    triton_k_adj, ref_k_adj, atol=1e-4, rtol=1e-4,
                    msg=f"k_adj mismatch for size ({T}, {H}, {K})"
                )
                torch.testing.assert_close(
                    triton_kk, ref_kk, atol=1e-4, rtol=1e-4,
                    msg=f"kk mismatch for size ({T}, {H}, {K})"
                )

    def test_triton_dtypes(self):
        """Verify triton handles different dtypes correctly."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 8, 4, 64

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                k = torch.randn(T, H, K, device='npu', dtype=dtype)
                a = torch.randn_like(k)
                torch.manual_seed(43)
                k_k = torch.randn(H, K, device='npu', dtype=dtype)
                k_a = torch.randn(H, K, device='npu', dtype=dtype)

                k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

                self.assertTrue(torch.isfinite(k_adj).all())
                self.assertTrue(torch.isfinite(kk).all())

                kk_norm = torch.norm(kk.float(), p=2, dim=-1)
                torch.testing.assert_close(
                    kk_norm, torch.ones_like(kk_norm), atol=1e-3, rtol=1e-3
                )

    def test_triton_eps_parameter(self):
        """Verify eps parameter is respected."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 8, 4, 64
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn_like(k)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        for eps in [1e-12, 1e-8, 1e-4, 1e-2]:
            k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a, eps=eps)
            self.assertEqual(k_adj.shape, k.shape)
            self.assertEqual(kk.shape, k.shape)

    def test_triton_non_contiguous_fallback(self):
        """Verify triton falls back to reference for non-contiguous input."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre, rwkv7_kk_pre_reference

        T, H, K = 8, 4, 64
        torch.manual_seed(42)
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(T, H, K, device='npu', dtype=torch.float32).transpose(0, 1).transpose(1, 2)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        k = k.contiguous()
        self.assertFalse(a.is_contiguous())

        ref_k_adj, ref_kk = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
        triton_k_adj, triton_kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)

        torch.testing.assert_close(triton_k_adj, ref_k_adj, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(triton_kk, ref_kk, atol=1e-5, rtol=1e-5)

    def test_triton_validation_shapes(self):
        """Verify shape validation works correctly."""
        from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre

        T, H, K = 8, 4, 64
        k = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        a = torch.randn(T, H, K, device='npu', dtype=torch.float32)
        k_k = torch.randn(H, K, device='npu', dtype=torch.float32)
        k_a = torch.randn(H, K, device='npu', dtype=torch.float32)

        k_adj, kk = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)
        self.assertIsNotNone(k_adj)
        self.assertIsNotNone(kk)

        with self.assertRaises(ValueError):
            a_wrong = torch.randn(T + 1, H, K, device='npu', dtype=torch.float32)
            rwkv7_kk_pre(k=k, k_k=k_k, a=a_wrong, k_a=k_a)

        with self.assertRaises(ValueError):
            k_2d = torch.randn(T * H, K, device='npu', dtype=torch.float32)
            rwkv7_kk_pre(k=k_2d, k_k=k_k, a=a, k_a=k_a)

        with self.assertRaises(ValueError):
            k_a_wrong = torch.randn(H, K + 1, device='npu', dtype=torch.float32)
            rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a_wrong)

        with self.assertRaises(ValueError):
            k_k_1d = torch.randn(H * K, device='npu', dtype=torch.float32)
            rwkv7_kk_pre(k=k, k_k=k_k_1d, a=a, k_a=k_a)

        with self.assertRaises(ValueError):
            k_k_wrong = torch.randn(H + 1, K, device='npu', dtype=torch.float32)
            rwkv7_kk_pre(k=k, k_k=k_k_wrong, a=a, k_a=k_a)


if __name__ == '__main__':
    unittest.main()
