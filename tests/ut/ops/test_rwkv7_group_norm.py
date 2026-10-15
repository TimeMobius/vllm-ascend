# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for RWKV7GroupNorm.

The 310P fused ``torch.nn.functional.group_norm`` kernel leaves most of the
output buffer unwritten inside the worker process (the rest keeps stale pool
memory), which garbles model output. ``RWKV7GroupNorm`` therefore falls back
to an explicit mean/var/affine decomposition on 310P. These tests pin the
numerical equivalence of both paths and the mode-selection logic.
"""

import os
import unittest
from unittest import mock

import torch


class TestRWKV7GroupNormModeSelection(unittest.TestCase):
    def _resolve(self, value: str) -> bool:
        from vllm_ascend.models.rwkv7 import RWKV7GroupNorm

        with mock.patch.dict(os.environ, {"VLLM_ASCEND_RWKV7_GNORM_MANUAL": value}):
            return RWKV7GroupNorm._resolve_reference_gnorm()

    def test_mode_one_forces_reference(self):
        self.assertTrue(self._resolve("1"))

    def test_mode_two_forces_fused(self):
        self.assertFalse(self._resolve("2"))

    def test_mode_zero_follows_soc_detection(self):
        from vllm_ascend.models import rwkv7

        for soc_is_310p in (True, False):
            with (
                mock.patch.dict(os.environ, {"VLLM_ASCEND_RWKV7_GNORM_MANUAL": "0"}),
                mock.patch.object(rwkv7, "is_310p", return_value=soc_is_310p),
            ):
                self.assertEqual(rwkv7.RWKV7GroupNorm._resolve_reference_gnorm(), soc_is_310p)


class TestRWKV7GroupNormReferenceParity(unittest.TestCase):
    def test_reference_and_fused_match_functional_group_norm(self):
        from vllm_ascend.models.rwkv7 import RWKV7GroupNorm

        num_heads, head_dim = 4, 8
        value_dim = num_heads * head_dim
        gn = RWKV7GroupNorm(
            num_heads=num_heads,
            head_dim=head_dim,
            value_dim=value_dim,
            eps=1e-5,
        )
        torch.manual_seed(0)
        with torch.no_grad():
            gn.weight.copy_(torch.randn(value_dim))
            gn.bias.copy_(torch.randn(value_dim))

        x = torch.randn(6, value_dim)
        expected = torch.nn.functional.group_norm(x.unsqueeze(-1), num_heads, gn.weight, gn.bias, gn.eps).squeeze(-1)

        gn._use_reference_gnorm = True
        reference_out = gn(x)

        gn._use_reference_gnorm = False
        fused_out = gn(x)

        torch.testing.assert_close(reference_out, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fused_out, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(reference_out, fused_out, rtol=1e-4, atol=1e-5)
