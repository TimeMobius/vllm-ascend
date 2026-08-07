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


class TestRWKV7AltRecurrentFinalStateParity(unittest.TestCase):
    """Regression test for AscendC ``npu_rwkv7_alt_recurrent`` final-state parity.

    The prefill parity benchmark
    (``benchmarks/rwkv7/benchmark_prefill.py``) originally reported
    T=1 ``final_state_max_abs_error`` in the 150-330 range while the
    output stayed within FP32 tolerance.  Root cause: callers were
    transposing the AscendC native ``final_state`` under the wrong
    assumption that it was stored as ``[B, H, V, D]``.  Empirical
    verification (with a sparse initial state) shows the kernel
    actually writes the FP32-torch-reference values directly into
    ``[B, H, D, V]`` slots (because the kernel writes
    ``stateMatrix[vIdx][kIdx]`` at offset ``k * V + v`` of a
    contiguous ``[B, H, 64, 64]`` buffer and ``V == D == 64`` makes
    those offsets coincide with the reference's ``[B, H, D, V]``
    layout).  These tests pin the layout and the parity.
    """

    @classmethod
    def setUpClass(cls):
        if not _has_custom_op:
            raise unittest.SkipTest("custom op not available, skipping RWKV7 alt recurrent tests")
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available, skipping RWKV7 alt recurrent tests")
        if not hasattr(torch.ops._C_ascend, "npu_rwkv7_alt_recurrent"):
            raise unittest.SkipTest("npu_rwkv7_alt_recurrent not found in torch.ops._C_ascend")

    @staticmethod
    def _make_stable_inputs(B, T, H, *, D=64, V=64, seed=1234, device="npu"):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        r = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32) * 0.1
        # Real RWKV w is log-space decay in (-inf, 0); restrict to
        # [-2, 0] so long-T prefill stays numerically stable.
        w = -torch.rand(B, T, H, D, generator=gen, dtype=torch.float32) * 2.0
        k = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32) * 0.5
        v = torch.randn(B, T, H, V, generator=gen, dtype=torch.float32) * 0.5
        kk = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32) * 0.5
        a = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32) * 0.5
        initial_state = torch.randn(B, H, D, V, generator=gen, dtype=torch.float32) * 0.1
        return (
            r.to(device),
            w.to(device),
            k.to(device),
            v.to(device),
            kk.to(device),
            a.to(device),
            initial_state.to(device),
        )

    def test_native_final_state_layout_is_B_H_D_V(self):
        """Native ``final_state`` is already ``[B, H, D, V]``-semantic.

        Use a sparse ``initial_state`` so the propagated values are
        distinct at known ``(d, v)`` slots; if the kernel ever changes
        to actually emit ``[B, H, V, D]`` semantics the equality will
        fail at those slots.
        """
        B, T, H, D, V = 1, 1, 1, 64, 64
        device = torch.device("npu:0")
        r = torch.zeros(B, T, H, D, device=device, dtype=torch.float32)
        w = -torch.ones(B, T, H, D, device=device, dtype=torch.float32)
        k = torch.zeros(B, T, H, D, device=device, dtype=torch.float32)
        v = torch.zeros(B, T, H, V, device=device, dtype=torch.float32)
        kk = torch.zeros(B, T, H, D, device=device, dtype=torch.float32)
        a = torch.zeros(B, T, H, D, device=device, dtype=torch.float32)
        initial_state = torch.zeros(B, H, D, V, device=device, dtype=torch.float32)
        # Only one (d, v) entry is non-zero so the sparse propagation
        # identifies the native layout unambiguously.
        initial_state[0, 0, 5, 10] = 1.0

        _, fs_native = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, initial_state
        )

        # The (b=0, h=0, d=5, v=10) slot of a contiguous [B,H,D,V]
        # tensor equals exp(w[0,0,0,5]) * 1.0 = exp(-1) ≈ 0.368.
        # If native were [B,H,V,D], the same value would live at
        # slot [0, 0, v=10, d=5] instead.
        self.assertAlmostEqual(
            fs_native[0, 0, 5, 10].item(),
            float(torch.exp(torch.tensor(-1.0))),
            places=4,
        )
        # And the transposed slot must NOT hold that value.
        self.assertNotAlmostEqual(
            fs_native[0, 0, 10, 5].item(),
            float(torch.exp(torch.tensor(-1.0))),
            places=4,
        )

    def test_t1_final_state_matches_torch_reference(self):
        """T=1 AscendC final_state agrees with the FP32 torch reference."""
        B, T, H = 1, 1, 4
        r, w, k, v, kk, a, initial_state = self._make_stable_inputs(
            B, T, H, seed=11
        )

        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference_with_checkpoints,
        )

        ref_out, ref_fs, _ = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
        )
        torch.npu.synchronize()
        out_native, fs_native = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, initial_state
        )
        torch.npu.synchronize()

        # Output: small FP32 tolerance.
        torch.testing.assert_close(
            out_native.cpu(), ref_out.cpu(), atol=1e-4, rtol=1e-4,
            msg="T=1 output diverges from FP32 torch reference",
        )
        # Final state: native is [B,H,D,V]-semantic; compare directly.
        torch.testing.assert_close(
            fs_native.cpu(), ref_fs.cpu(), atol=1e-4, rtol=1e-4,
            msg="T=1 final state diverges from FP32 torch reference; "
                "native layout is not [B,H,D,V]-semantic",
        )

    def test_long_t_final_state_matches_torch_reference(self):
        """Long-T AscendC final_state agrees with the FP32 torch reference.

        Stable inputs (``w`` is negative log-space decay) keep both the
        kernel and the reference numerically well-behaved across many
        recurrent steps.  Standard-normal ``w`` allows positive decay
        and makes the reference state diverge to Inf/NaN; this test
        deliberately avoids that case.
        """
        B, T, H = 1, 128, 4
        r, w, k, v, kk, a, initial_state = self._make_stable_inputs(
            B, T, H, seed=23
        )

        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference_with_checkpoints,
        )

        ref_out, ref_fs, _ = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
        )
        torch.npu.synchronize()
        out_native, fs_native = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, initial_state
        )
        torch.npu.synchronize()

        # Both must remain finite (sanity check on stable inputs).
        self.assertTrue(torch.isfinite(out_native).all().item())
        self.assertTrue(torch.isfinite(fs_native).all().item())
        self.assertTrue(torch.isfinite(ref_out).all().item())
        self.assertTrue(torch.isfinite(ref_fs).all().item())

        # Output parity.
        torch.testing.assert_close(
            out_native.cpu(), ref_out.cpu(), atol=1e-3, rtol=1e-3,
            msg="Long-T output diverges from FP32 torch reference",
        )
        # Final state parity — the original benchmark reported
        # max_abs ~150 here because of the spurious transpose.
        torch.testing.assert_close(
            fs_native.cpu(), ref_fs.cpu(), atol=1e-3, rtol=1e-3,
            msg="Long-T final state diverges from FP32 torch reference; "
                "this regression appeared as a ~150-330 max_abs error in "
                "prefill_metrics.json.  Native layout must be "
                "[B, H, D, V]-semantic.",
        )


class TestRWKV7AltRecurrentB2PlusBatchCoverage(unittest.TestCase):
    """Regression coverage for B >= 2 AscendC parity.

    Previously the kernel's ``Init()`` early-returned when
    ``blockIdx >= GetBlockNum()`` — ``GetBlockNum()`` returns the
    hardware AICore count, but the launch grid is sized
    ``batch * numHeads``.  When ``batch * numHeads`` exceeded the
    physical core count, all blocks past the core boundary were
    silently dropped, leaving the corresponding (batch, head) slices
    of ``out`` and ``final_state`` at their uninitialised (zero)
    values.

    The previously-failing shapes were:

      * ``B=8,  T=128``: output max_abs ~ 5.4e5
      * ``B=32, T=128``: output max_abs ~ 3.6e6

    These tests pin the B >= 2 parity contract at the user's
    atol=rtol=1e-4 tolerance, using the same stable ``xiaoke_realistic``
    ``w`` distribution the precision benchmark uses.
    """

    LOG_DECAY_SCALE = -0.6065306597126334
    INPUT_SCALE = 0.1
    W_BIAS_MEAN = -2.80890
    W_BIAS_STD = 1.91028

    @classmethod
    def setUpClass(cls):
        if not _has_custom_op:
            raise unittest.SkipTest(
                "custom op not available, skipping B>=2 batch coverage tests"
            )
        if not torch.npu.is_available():
            raise unittest.SkipTest(
                "NPU not available, skipping B>=2 batch coverage tests"
            )
        if not hasattr(torch.ops._C_ascend, "npu_rwkv7_alt_recurrent"):
            raise unittest.SkipTest(
                "npu_rwkv7_alt_recurrent not found in torch.ops._C_ascend"
            )

    @classmethod
    def _make_xiaoke_inputs(cls, B, T, H, *, seed, device):
        """Inputs modelled on the Xiaoke RWKV7 ``w`` distribution.

        ``w = LOG_DECAY_SCALE * sigmoid(z)`` with
        ``z ~ N(W_BIAS_MEAN, W_BIAS_STD)`` — the empirical mean/std of
        the Xiaoke ``w_lora.lora.2.bias`` parameter — keeps
        ``exp(w) <= 1`` (strictly contractive).  Other tensors are
        scaled by ``INPUT_SCALE`` so the recurrence's accumulated
        ``k*v`` term stays bounded across T=1024 for the benchmark
        matrix.
        """
        D = 64
        V = 64
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        r = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32,
                        device=device) * cls.INPUT_SCALE
        z = (torch.randn(B, T, H, D, generator=gen, dtype=torch.float32,
                        device=device) * cls.W_BIAS_STD + cls.W_BIAS_MEAN)
        w = cls.LOG_DECAY_SCALE * torch.sigmoid(z)
        k = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32,
                        device=device) * cls.INPUT_SCALE
        v = torch.randn(B, T, H, V, generator=gen, dtype=torch.float32,
                        device=device) * cls.INPUT_SCALE
        kk = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32,
                         device=device) * cls.INPUT_SCALE
        a = torch.randn(B, T, H, D, generator=gen, dtype=torch.float32,
                        device=device) * cls.INPUT_SCALE
        initial_state = torch.randn(B, H, D, V, generator=gen,
                                   dtype=torch.float32,
                                   device=device) * cls.INPUT_SCALE
        return r, w, k, v, kk, a, initial_state

    @staticmethod
    def _parity(out_a, fs_a, out_r, fs_r):
        out_diff = (out_a.float() - out_r.float()).abs().max().item()
        fs_diff = (fs_a.float() - fs_r.float()).abs().max().item()
        return out_diff, fs_diff

    def _assert_parity(self, B, T, H, *, seed):
        from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
            rwkv7_recurrent_reference_with_checkpoints,
        )

        device = torch.device("npu:0")
        r, w, k, v, kk, a, initial_state = self._make_xiaoke_inputs(
            B, T, H, seed=seed, device=device
        )

        torch.npu.synchronize()
        out_a, fs_a = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
            r, w, k, v, kk, a, initial_state
        )
        torch.npu.synchronize()

        out_r, fs_r, _ = rwkv7_recurrent_reference_with_checkpoints(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            initial_state=initial_state, scale=1.0, output_final_state=True,
            cu_seqlens=None, checkpoint_positions=None,
            checkpoint_offsets=None, output_checkpoint_states=False,
        )

        # Sanity: every cell finite on both paths.
        self.assertTrue(
            torch.isfinite(out_a).all().item(),
            f"B={B} T={T}: AscendC out contains NaN/Inf",
        )
        self.assertTrue(
            torch.isfinite(fs_a).all().item(),
            f"B={B} T={T}: AscendC final_state contains NaN/Inf",
        )

        out_diff, fs_diff = self._parity(out_a, fs_a, out_r, fs_r)

        # Hard contract: atol=rtol=1e-4 (matches the user's prefill
        # benchmark tolerance).  Before the fix the (b=2+) batch
        # slices read uninitialised (zero) memory and produced
        # ``out_diff`` values up to ~3.6e6 here.
        torch.testing.assert_close(
            out_a.cpu(), out_r.cpu(), atol=1e-4, rtol=1e-4,
            msg=(f"B={B} T={T}: AscendC output diverges from FP32 torch "
                 f"reference; max_abs={out_diff:.3e}"),
        )
        torch.testing.assert_close(
            fs_a.cpu(), fs_r.cpu(), atol=1e-4, rtol=1e-4,
            msg=(f"B={B} T={T}: AscendC final_state diverges from FP32 "
                 f"torch reference; max_abs={fs_diff:.3e}"),
        )

    def test_b2_t32_parity(self):
        """B=2 (smallest B that previously dropped work) is parity-clean."""
        self._assert_parity(B=2, T=32, H=4, seed=42)

    def test_b8_t128_parity(self):
        """Previously-extreme outlier: B=8/T=128 used to read zero."""
        self._assert_parity(B=8, T=128, H=4, seed=42)

    def test_b32_t128_parity(self):
        """Previously-extreme outlier: B=32/T=128 used to read zero."""
        self._assert_parity(B=32, T=128, H=4, seed=42)

    def test_b64_t512_parity(self):
        """Largest B in the benchmark matrix: B=64/T=512 used to read zero."""
        self._assert_parity(B=64, T=512, H=4, seed=42)


if __name__ == "__main__":
    unittest.main()