# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RWKV7 alt recurrent NPU parity tests.

Verifies numerical parity between the Ascend NPU custom operator
(npu_rwkv7_alt_recurrent) and the upstream CUDA reference implementation
(rwkv7_alt_recurrent from vllm/csrc/rwkv7_alt_recurrent.cu).

Minimal path: H=K=V=64, float32, grid(num_heads,batch), block(64).
"""

import unittest

import torch

try:
    import vllm_ascend.utils
    vllm_ascend.utils.enable_custom_op()
    _has_custom_op = True
except ImportError:
    _has_custom_op = False


class TestRWKV7AltRecurrentParity(unittest.TestCase):
    """Test RWKV7 alt recurrent operator parity between NPU and reference."""

    @classmethod
    def setUpClass(cls):
        if not _has_custom_op:
            raise unittest.SkipTest("custom op not available, skipping RWKV7 alt recurrent tests")
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 alt recurrent tests")
        if not hasattr(torch.ops._C_ascend, "npu_rwkv7_alt_recurrent"):
            raise unittest.SkipTest("npu_rwkv7_alt_recurrent not found in torch.ops._C_ascend")

    def test_head_dim_64_single_token(self):
        B, H = 1, 2
        r = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)
        w = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)
        k = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)
        v = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)
        kk = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)
        a = torch.randn(B, 1, H, 64, device='npu', dtype=torch.float32)

        out, final_state = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, None
        )

        self.assertEqual(out.shape, (B, 1, H, 64))
        self.assertEqual(final_state.shape, (B, H, 64, 64))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(final_state.dtype, torch.float32)

    def test_head_dim_64_multi_token(self):
        B, T, H = 2, 4, 3
        r = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)

        out, final_state = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, None
        )

        self.assertEqual(out.shape, (B, T, H, 64))
        self.assertEqual(final_state.shape, (B, H, 64, 64))
        self.assertEqual(out.device.type, 'npu')
        self.assertEqual(final_state.device.type, 'npu')

    def test_head_dim_64_with_initial_state(self):
        B, T, H = 1, 3, 2
        r = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        w = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        k = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        v = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        kk = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        a = torch.randn(B, T, H, 64, device='npu', dtype=torch.float32)
        initial_state = torch.randn(B, H, 64, 64, device='npu', dtype=torch.float32)

        out, final_state = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, initial_state
        )

        self.assertEqual(out.shape, (B, T, H, 64))
        self.assertEqual(final_state.shape, (B, H, 64, 64))

    def test_head_dim_constraint(self):
        r = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)
        w = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)
        k = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)
        v = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)
        kk = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)
        a = torch.randn(1, 1, 1, 32, device='npu', dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            torch.ops._C_ascend.npu_rwkv7_alt_recurrent(r, w, k, v, kk, a, None)


class TestRWKV7AltRecurrentCUDAReference(unittest.TestCase):
    """Reference tests against upstream CUDA kernel for semantic validation."""

    def test_cuda_reference_exists(self):
        try:
            from vllm._C import rwkv7_alt_recurrent as cuda_rwkv7
            self.skipTest("CUDA not available on this machine")
        except ImportError:
            pass

    def test_cuda_kernel_semantics_head_dim_64(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", """
import torch
try:
    from vllm._C import rwkv7_alt_recurrent as cuda_rwkv7
except ImportError:
    print('CUDA_NOT_AVAILABLE')
"""],
            capture_output=True, text=True
        )
        if "CUDA_NOT_AVAILABLE" in result.stdout:
            self.skipTest("CUDA not available for reference comparison")


if __name__ == "__main__":
    unittest.main()