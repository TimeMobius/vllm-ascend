# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NPU-specific parity tests for RWKV7 rwkv7_lnx_rkvres_xg epilogue operation.

This module verifies that the Ascend Triton implementation of rwkv7_lnx_rkvres_xg
produces correct results on NPU hardware.

Reference: vllm/model_executor/layers/fla/ops/rwkv7.py::rwkv7_lnx_rkvres_xg
"""

import unittest

import torch


class TestRWKV7EpilogueNPUParity(unittest.TestCase):
    """Test parity between Ascend Triton and reference implementations on NPU."""

    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping NPU-specific tests")

    def test_parity_small_shape(self):
        """Test parity with small hidden dimensions."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertEqual(ref_out.shape, triton_out.shape)
        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)

    def test_parity_typical_shape(self):
        """Test parity with typical hidden dimensions (similar to RWKV7-1.6B)."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 17, 8, 64, 64
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertEqual(ref_out.shape, triton_out.shape)
        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)

    def test_parity_large_head_dim(self):
        """Test parity with larger head dimension (power of 2)."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 8, 4, 128, 128
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertEqual(ref_out.shape, triton_out.shape)
        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)

    def test_parity_asymmetric_head_dims(self):
        """Test parity with asymmetric head_dim and head_v_dim."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 5, 3, 32, 64
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertEqual(ref_out.shape, triton_out.shape)
        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)

    def test_parity_single_token(self):
        """Test parity with single token (decoding scenario)."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
            rwkv7_lnx_rkvres_xg,
            rwkv7_lnx_rkvres_xg_reference,
        )

        num_tokens, num_heads, head_dim, head_v_dim = 1, 8, 64, 64
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertEqual(ref_out.shape, triton_out.shape)
        torch.testing.assert_close(ref_out, triton_out, atol=1e-4, rtol=1e-4)

    def test_parity_with_upstream_reference(self):
        """Test parity against the actual upstream reference implementation."""
        from vllm.model_executor.layers.fla.ops.rwkv7 import rwkv7_lnx_rkvres_xg_reference as upstream_ref

        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 12, 6, 48, 48
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        ascend_out = rwkv7_lnx_rkvres_xg(
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

        self.assertEqual(upstream_out.shape, ascend_out.shape)
        torch.testing.assert_close(upstream_out, ascend_out, atol=1e-4, rtol=1e-4)

    def test_output_dtype_preservation(self):
        """Test that output dtype is correctly preserved."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

        out_bf16 = rwkv7_lnx_rkvres_xg(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(out_bf16.dtype, torch.bfloat16)

        out_default = rwkv7_lnx_rkvres_xg(
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
        self.assertEqual(out_default.dtype, g.dtype)

    def test_input_validation_errors(self):
        """Test that input validation raises appropriate errors."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 4, 2, 16, 16
        eps = 64e-5

        recurrent_output = torch.randn(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.randn(num_tokens, num_heads, head_dim, device='npu')
        k = torch.randn_like(r)
        v = torch.randn_like(recurrent_output)
        r_k = torch.randn(num_heads, head_dim, device='npu')
        weight = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.randn(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.randn(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

        k_wrong = torch.randn(num_tokens, num_heads + 1, head_dim, device='npu')
        with self.assertRaises(ValueError):
            rwkv7_lnx_rkvres_xg(
                recurrent_output=recurrent_output,
                r=r,
                k=k_wrong,
                v=v,
                r_k=r_k,
                weight=weight,
                bias=bias,
                g=g,
                eps=eps,
            )

        v_wrong = torch.randn(num_tokens, num_heads, head_v_dim + 1, device='npu')
        with self.assertRaises(ValueError):
            rwkv7_lnx_rkvres_xg(
                recurrent_output=recurrent_output,
                r=r,
                k=k,
                v=v_wrong,
                r_k=r_k,
                weight=weight,
                bias=bias,
                g=g,
                eps=eps,
            )

        r_k_wrong = torch.randn(num_heads + 1, head_dim, device='npu')
        with self.assertRaises(ValueError):
            rwkv7_lnx_rkvres_xg(
                recurrent_output=recurrent_output,
                r=r,
                k=k,
                v=v,
                r_k=r_k_wrong,
                weight=weight,
                bias=bias,
                g=g,
                eps=eps,
            )

    def test_finite_output(self):
        """Test that outputs are finite (no NaN/Inf) for normal inputs."""
        from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg

        num_tokens, num_heads, head_dim, head_v_dim = 8, 4, 64, 64
        eps = 64e-5

        recurrent_output = torch.rand(
            num_tokens, num_heads, head_v_dim, device='npu', dtype=torch.float32
        )
        r = torch.rand(num_tokens, num_heads, head_dim, device='npu')
        k = torch.rand_like(r)
        v = torch.rand_like(recurrent_output)
        r_k = torch.rand(num_heads, head_dim, device='npu')
        weight = torch.rand(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        bias = torch.rand(num_heads * head_v_dim, device='npu', dtype=torch.float32)
        g = torch.rand(num_tokens, num_heads * head_v_dim, device='npu', dtype=torch.float32)

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

        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()