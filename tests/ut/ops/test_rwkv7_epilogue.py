# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parity tests for RWKV7 rwkv7_lnx_rkvres_xg epilogue operation.

This module verifies that the Ascend Triton implementation of rwkv7_lnx_rkvres_xg
produces correct results that match the upstream reference implementation.

Reference: vllm/model_executor/layers/fla/ops/rwkv7.py::rwkv7_lnx_rkvres_xg
"""

import unittest

import torch


class TestRWKV7EpilogueCPUFallback(unittest.TestCase):
    """Test reference implementation on CPU (fallback path)."""

    def test_reference_cpu_small_shape(self):
        """Test reference on CPU with small hidden dimensions."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='cpu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='cpu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='cpu')
        weight = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='cpu', dtype=torch.float32)

        out = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        expected_shape = (num_tokens, num_heads * head_v_dim)
        self.assertEqual(out.shape, expected_shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_reference_cpu_fallback_via_triton_interface(self):
        """Test that rwkv7_lnx_rkvres_xg falls back to reference on CPU."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='cpu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='cpu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='cpu')
        weight = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='cpu', dtype=torch.float32)

        out = rwkv7_lnx_rkvres_xg(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        self.assertEqual(out.shape, (num_tokens, num_heads * head_v_dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_reference_cpu_empty_input(self):
        """Test reference handles empty input correctly."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_heads, head_dim, head_v_dim = 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(0, num_heads, head_v_dim, device='cpu', dtype=torch.float32)
        r = torch.randn(0, num_heads, head_dim, device='cpu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='cpu')
        weight = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        g = torch.randn(0, num_heads * head_v_dim, device='cpu', dtype=torch.float32)

        out = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        self.assertEqual(out.shape, (0, num_heads * head_v_dim))

    def test_reference_cpu_with_upstream_parity(self):
        """Test reference matches upstream implementation."""
        try:
            from vllm_ascend.ops.triton.fla.rwkv7 import (
                rwkv7_lnx_rkvres_xg_reference as upstream_ref,
            )
        except ImportError:
            self.skipTest("Upstream reference not available")

        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 8, 4, 32, 32
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='cpu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='cpu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='cpu')
        weight = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='cpu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='cpu', dtype=torch.float32)

        upstream_out = upstream_ref(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        ascend_ref_out = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        torch.testing.assert_close(upstream_out, ascend_ref_out, atol=1e-4, rtol=1e-4)


class TestRWKV7EpilogueGroupNormSemantics(unittest.TestCase):
    """Test that Group Norm semantics are preserved exactly."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping tests")

    def test_group_norm_per_token_head(self):
        """Verify GroupNorm is applied per-token per-head, not globally."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_v_dim = 3, 4, 32
        head_dim = 32
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.ones(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.zeros(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.ones(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

        ref_out = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        triton_out = rwkv7_lnx_rkvres_xg(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)
        self.assertFalse(torch.allclose(ref_out[0], ref_out[1]))

    def test_large_grid_parity(self):
        """Verify token-head rows above the Ascend grid limit remain correct."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 2048, 64, 64, 64
        eps = 64e-5
        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device="npu", dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device="npu")
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device="npu")
        weight = torch.randn(num_heads * head_v_dim, device="npu")
        bias = torch.randn(num_heads * head_v_dim, device="npu")
        g = torch.randn(
            num_tokens, num_heads * head_v_dim, device="npu", dtype=torch.float32
        )

        ref_out = rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )
        triton_out = rwkv7_lnx_rkvres_xg(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
        )

        torch.testing.assert_close(triton_out, ref_out, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
