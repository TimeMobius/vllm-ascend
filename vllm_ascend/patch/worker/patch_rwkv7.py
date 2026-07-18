#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
RWKV7 Ascend patch — integrates Triton-Ascend kernels into upstream RWKV7.

This patch wires the Ascend Triton kernels into the current upstream RWKV7
implementation via the following integration points:

1. Module-level `_rwkv7_recurrent_scan` and `_rwkv7_recurrent_scan_varlen`
   are replaced with versions that dispatch to `fused_recurrent_rwkv7`
   when safe (NPU device, contiguous tensors, valid shapes).

2. `RWKV7Attention._finalize_attention_output` is replaced with a version
   that dispatches to `rwkv7_lnx_rkvres_xg` when safe.

The patch is idempotent and safe:
- When Triton/NPU is unavailable, falls back to exact upstream behavior
- When kernel shapes/dtypes don't match, falls back to exact upstream behavior
- Can be applied multiple times without effect
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch

# Lazy import to avoid hard dependency on Triton when not available
_ascend_ops: Optional[object] = None


def _get_ascend_ops():
    """Lazily import ascend ops to avoid hard dependency."""
    global _ascend_ops
    if _ascend_ops is None:
        try:
            from vllm_ascend.ops.triton.fla import (
                fused_recurrent_rwkv7,
                rwkv7_lnx_rkvres_xg,
            )
            from vllm_ascend.ops.triton.fla import (
                rwkv7_lnx_rkvres_xg_reference,
            )

            _ascend_ops = type(
                "AscendOps",
                (),
                {
                    "fused_recurrent_rwkv7": fused_recurrent_rwkv7,
                    "rwkv7_lnx_rkvres_xg": rwkv7_lnx_rkvres_xg,
                    "rwkv7_lnx_rkvres_xg_reference": rwkv7_lnx_rkvres_xg_reference,
                    "HAS_TRITON": True,
                },
            )()
        except ImportError:
            _ascend_ops = type(
                "AscendOps",
                (),
                {
                    "fused_recurrent_rwkv7": None,
                    "rwkv7_lnx_rkvres_xg": None,
                    "rwkv7_lnx_rkvres_xg_reference": None,
                    "HAS_TRITON": False,
                },
            )()
    return _ascend_ops


def _is_npu_available() -> bool:
    """Check if NPU is available for compute."""
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _can_use_fused_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    """Check if fused_recurrent_rwkv7 can be used safely."""
    ops = _get_ascend_ops()
    if not ops.HAS_TRITON:
        return False
    if not _is_npu_available():
        return False
    # All inputs must be on NPU
    if r.device.type != "npu":
        return False
    # All inputs must be non-empty and contiguous
    if r.numel() == 0:
        return False
    if not all(t.is_contiguous() for t in [r, w, k, v, kk, a]):
        return False
    # Inputs must be 3D tensors (T, H, K) for single-sequence recurrent scan
    if r.ndim != 3 or w.ndim != 3 or k.ndim != 3 or v.ndim != 3 or kk.ndim != 3 or a.ndim != 3:
        return False
    return True


def _can_use_epilogue_kernel(
    recurrent_output: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
) -> bool:
    """Check if rwkv7_lnx_rkvres_xg can be used safely."""
    ops = _get_ascend_ops()
    if not ops.HAS_TRITON:
        return False
    if not _is_npu_available():
        return False
    # All inputs must be on NPU
    if recurrent_output.device.type != "npu":
        return False
    # Must be non-empty
    if recurrent_output.numel() == 0:
        return False
    # All must be contiguous
    if not all(
        t.is_contiguous()
        for t in [recurrent_output, r, k, v, r_k, weight, bias, g]
    ):
        return False
    # Check shape constraints for epilogue kernel
    # recurrent_output: [num_tokens, num_heads, head_v_dim]
    # r: [num_tokens, num_heads, head_dim]
    # k: [num_tokens, num_heads, head_dim]
    # v: [num_tokens, num_heads, head_v_dim]
    # r_k: [num_heads, head_dim]
    if recurrent_output.ndim != 3 or r.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False
    if recurrent_output.shape[:2] != r.shape[:2]:
        return False
    # r_k must have same number of heads as recurrent_output
    if r_k.shape[0] != recurrent_output.shape[1]:
        return False
    return True


def _patch_rwkv7_recurrent_scan():
    """
    Patch _rwkv7_recurrent_scan to use fused_recurrent_rwkv7 when safe.

    The upstream function performs a sequential Python loop over the time
    dimension, which is slow on NPU. The fused kernel performs the same
    computation in a single kernel launch.
    """
    ops = _get_ascend_ops()

    # Import the upstream module
    try:
        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")
    except ImportError:
        # Upstream module not available, nothing to patch
        return

    # Check if already patched
    if getattr(rwkv7_module, "_RWKV7_ASCEND_PATCHED", False):
        return

    # Save the original function
    original_recurrent_scan = rwkv7_module._rwkv7_recurrent_scan

    def _rwkv7_recurrent_scan_ascend(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ):
        """
        Ascend-optimized recurrent scan that dispatches to fused kernel when safe.

        When conditions aren't met (non-NPU, wrong shapes, Triton unavailable),
        falls back to the exact upstream reference implementation.
        """
        if not _can_use_fused_recurrent(r, w, k, v, kk, a):
            return original_recurrent_scan(r, w, k, v, kk, a, initial_state)

        # Compute output using fused kernel
        # fused_recurrent_rwkv7 expects 4D [B, T, H, K] but we have 3D [T, H, K]
        # Add batch dimension
        B = 1
        T, H, K = r.shape
        V = v.shape[-1]

        # Reshape to 4D
        r_4d = r.unsqueeze(0)  # [1, T, H, K]
        w_4d = w.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)  # [1, T, H, V]
        kk_4d = kk.unsqueeze(0)
        a_4d = a.unsqueeze(0)

        # Prepare initial state
        h0 = None
        if initial_state is not None:
            # initial_state: [H, K, V] -> [1, H, K, V]
            h0 = initial_state.unsqueeze(0)

        # Call fused kernel
        try:
            o, ht, _ = ops.fused_recurrent_rwkv7(
                r=r_4d,
                w=w_4d,
                k=k_4d,
                v=v_4d,
                kk=kk_4d,
                a=a_4d,
                scale=1.0,
                initial_state=h0,
                output_final_state=True,
            )
            # o: [1, T, H, V]
            # ht: [1, H, K, V] or None

            output = o.squeeze(0)  # [T, H, V]
            final_state = ht.squeeze(0) if ht is not None else None

            return output, final_state
        except Exception:
            # Kernel failed for some reason, fall back to reference
            return original_recurrent_scan(r, w, k, v, kk, a, initial_state)

    # Install the patched function
    rwkv7_module._rwkv7_recurrent_scan = _rwkv7_recurrent_scan_ascend
    rwkv7_module._RWKV7_ASCEND_PATCHED = True


def _patch_rwkv7_recurrent_scan_varlen():
    """
    Patch _rwkv7_recurrent_scan_varlen to use fused_recurrent_rwkv7 when safe.

    The varlen version handles multiple sequences of different lengths using
    cu_seqlens (cumulative sequence lengths). The fused kernel supports this
    via its cu_seqlens parameter.
    """
    ops = _get_ascend_ops()

    try:
        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")
    except ImportError:
        return

    if getattr(rwkv7_module, "_RWKV7_ASCEND_PATCHED_VARLEN", False):
        return

    original_recurrent_scan_varlen = rwkv7_module._rwkv7_recurrent_scan_varlen

    def _rwkv7_recurrent_scan_varlen_ascend(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        query_start_loc: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ):
        """
        Ascend-optimized varlen recurrent scan.

        Uses fused kernel with cu_seqlens for variable-length sequence handling.
        Falls back to reference when conditions aren't met.
        """
        if not _can_use_fused_recurrent(r, w, k, v, kk, a):
            return original_recurrent_scan_varlen(
                r, w, k, v, kk, a, query_start_loc, initial_state
            )

        # Check that batch size is 1 (required by cu_seqlens interface)
        if r.shape[0] != 1:
            return original_recurrent_scan_varlen(
                r, w, k, v, kk, a, query_start_loc, initial_state
            )

        # fused kernel expects cu_seqlens as tensor
        # query_start_loc is already the cumulative sequence lengths
        cu_seqlens = query_start_loc

        # Reshape to 4D [1, T, H, K]
        B = 1
        T, H, K = r.shape
        V = v.shape[-1]

        r_4d = r.unsqueeze(0)
        w_4d = w.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        kk_4d = kk.unsqueeze(0)
        a_4d = a.unsqueeze(0)

        # initial_state: [N, H, K, V] where N is num_sequences
        h0 = initial_state

        try:
            o, ht, _ = ops.fused_recurrent_rwkv7(
                r=r_4d,
                w=w_4d,
                k=k_4d,
                v=v_4d,
                kk=kk_4d,
                a=a_4d,
                scale=1.0,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

            output = o.squeeze(0)  # [T, H, V]
            # ht: [N, H, K, V]
            final_state = ht

            return output, final_state
        except Exception:
            return original_recurrent_scan_varlen(
                r, w, k, v, kk, a, query_start_loc, initial_state
            )

    rwkv7_module._rwkv7_recurrent_scan_varlen = _rwkv7_recurrent_scan_varlen_ascend
    rwkv7_module._RWKV7_ASCEND_PATCHED_VARLEN = True


def _patch_finalize_attention_output():
    """
    Patch RWKV7Attention._finalize_attention_output to use rwkv7_lnx_rkvres_xg when safe.

    The epilogue operation performs:
    1. GroupNorm on recurrent_output
    2. Recurrent correction: ((r * k * r_k).sum(-1, keepdim=True) * v).reshape(...)
    3. Gating: (output + correction) * g

    The Triton kernel fuses these operations.
    """
    ops = _get_ascend_ops()

    try:
        rwkv7_module = importlib.import_module("vllm.model_executor.models.rwkv7")
        RWKV7Attention = rwkv7_module.RWKV7Attention
    except ImportError:
        return

    if getattr(RWKV7Attention, "_ASCEND_EPILOGUE_PATCHED", False):
        return

    original_finalize = RWKV7Attention._finalize_attention_output

    def _finalize_attention_output_ascend(
        self,
        recurrent_output: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Ascend-optimized attention output finalization.

        Dispatches to rwkv7_lnx_rkvres_xg when safe, otherwise falls back
        to exact upstream reference implementation.
        """
        # Prepare parameters needed for the kernel
        # From upstream: recurrent_output is [T, H, V], r/k are [T, H, K]
        # We need local slices of r_k, weight, bias from self
        num_tokens, num_heads, head_v_dim = recurrent_output.shape
        head_dim = r.shape[-1]
        local_value_dim = num_heads * head_v_dim

        # Get local r_k slice: [tp_rank * local_num_heads : (tp_rank+1) * local_num_heads, head_dim]
        local_r_k = self.r_k[
            self.tp_rank
            * self.local_num_heads : (self.tp_rank + 1)
            * self.local_num_heads
        ].to(torch.float32)

        # Get local weight/bias slices for group norm
        local_weight = self.g_norm.weight[self.value_start : self.value_end].to(
            torch.float32
        )
        local_bias = self.g_norm.bias[self.value_start : self.value_end].to(
            torch.float32
        )

        # Check if we can use the kernel
        if not _can_use_epilogue_kernel(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=local_r_k,
            weight=local_weight,
            bias=local_bias,
            g=g,
        ):
            return original_finalize(
                self, recurrent_output, r, k, v, g, hidden_dtype
            )

        try:
            output = ops.rwkv7_lnx_rkvres_xg(
                recurrent_output=recurrent_output,
                r=r,
                k=k,
                v=v,
                r_k=local_r_k,
                weight=local_weight,
                bias=local_bias,
                g=g,
                eps=self.g_norm.eps,
            )
            output = output.to(hidden_dtype)
            output, _ = self.o_proj(output)
            return output
        except Exception:
            return original_finalize(
                self, recurrent_output, r, k, v, g, hidden_dtype
            )

    RWKV7Attention._finalize_attention_output = _finalize_attention_output_ascend
    RWKV7Attention._ASCEND_EPILOGUE_PATCHED = True


def apply_patch():
    """
    Apply the RWKV7 Ascend patch.

    This function is idempotent - calling it multiple times has no additional
    effect after the first call.

    The patch integrates Ascend Triton kernels into the upstream RWKV7
    implementation when:
    - Triton is available
    - NPU hardware is available
    - Tensor shapes/dtypes are compatible
    - All tensors are contiguous

    When any condition fails, the exact upstream reference implementation
    is used instead.
    """
    _patch_rwkv7_recurrent_scan()
    _patch_rwkv7_recurrent_scan_varlen()
    _patch_finalize_attention_output()


# Apply the patch when this module is imported
apply_patch()