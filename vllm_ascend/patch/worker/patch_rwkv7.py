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
import torch.nn.functional as F

from vllm_ascend import envs as envs_ascend
from vllm_ascend.profiler.rwkv7_counters import (
    dispatch_fallback,
    dispatch_hit,
    DispatchKind,
)

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
            from vllm_ascend.ops.triton.fla import rwkv7_mix6
            from vllm_ascend.ops.triton.fla import rwkv7_kk_pre

            # Use staticmethod to prevent functions from becoming bound methods
            # when accessed via instance. Without this, Python's descriptor
            # protocol would inject an implicit self, causing TypeError like:
            # "got multiple values for argument 'recurrent_output'"
            _ascend_ops = type(
                "AscendOps",
                (),
                {
                    "fused_recurrent_rwkv7": staticmethod(fused_recurrent_rwkv7),
                    "rwkv7_lnx_rkvres_xg": staticmethod(rwkv7_lnx_rkvres_xg),
                    "rwkv7_lnx_rkvres_xg_reference": staticmethod(
                        rwkv7_lnx_rkvres_xg_reference
                    ),
                    "rwkv7_mix6": staticmethod(rwkv7_mix6),
                    "rwkv7_kk_pre": staticmethod(rwkv7_kk_pre),
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
                    "rwkv7_mix6": None,
                    "rwkv7_kk_pre": None,
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
    if (
        envs_ascend.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or envs_ascend.RWKV7_DISABLE_FUSED_RECURRENT
        or envs_ascend.RWKV7_DISABLE_FUSED_PREFILL
    ):
        return False
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
    if (
        envs_ascend.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or not envs_ascend.RWKV7_USE_FUSED_LNX_RKVRES_XG
    ):
        return False
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


def _can_use_mix6_kernel(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> bool:
    """Check if rwkv7_mix6 can be used safely."""
    if (
        envs_ascend.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or not envs_ascend.RWKV7_USE_FUSED_MIX6
    ):
        return False
    ops = _get_ascend_ops()
    if not ops.HAS_TRITON:
        return False
    if not _is_npu_available():
        return False
    # All inputs must be on NPU
    if hidden_states.device.type != "npu":
        return False
    # Must be non-empty
    if hidden_states.numel() == 0:
        return False
    # All must be contiguous
    if not all(
        t.is_contiguous()
        for t in [hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g]
    ):
        return False
    # hidden_states and delta must have same shape
    if hidden_states.shape != delta.shape:
        return False
    return True


def _can_use_kk_pre_kernel(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
) -> bool:
    """Check if rwkv7_kk_pre can be used safely."""
    if (
        envs_ascend.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or not envs_ascend.RWKV7_USE_FUSED_KK_PRE
    ):
        return False
    ops = _get_ascend_ops()
    if not ops.HAS_TRITON:
        return False
    if not _is_npu_available():
        return False
    # All inputs must be on NPU
    if k.device.type != "npu":
        return False
    # Must be non-empty and contiguous
    if k.numel() == 0:
        return False
    if not all(t.is_contiguous() for t in [k, k_k, a, k_a]):
        return False
    # k and a must be 3D [T, H, K], k_k and k_a must be 2D [H, K]
    if k.ndim != 3 or a.ndim != 3:
        return False
    if k_k.ndim != 2 or k_a.ndim != 2:
        return False
    # Shape constraints: k_k and k_a must match head layout of k
    if k.shape[1:] != k_k.shape:
        return False
    if k.shape != a.shape:
        return False
    if k_k.shape != k_a.shape:
        return False
    return True


def _can_use_alt_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> bool:
    if envs_ascend.RWKV7_DISABLE_FUSED_RECURRENT:
        return False
    if not envs_ascend.RWKV7_USE_ALT_RECURRENT_KERNEL:
        return False
    if not _is_npu_available() or r.device.type != "npu" or r.numel() == 0:
        return False
    if r.shape[-1] != 64 or v.shape[-1] != 64:
        return False
    tensors = (r, w, k, v, kk, a)
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        return False
    if initial_state is not None:
        if initial_state.shape != (r.shape[1], 64, 64):
            return False
        if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
            return False
    return hasattr(torch.ops.ascend, "npu_rwkv7_alt_recurrent")


def _can_use_direct_linear(linear, hidden_states: torch.Tensor) -> bool:
    quant_method = getattr(linear, "quant_method", None)
    return (
        envs_ascend.RWKV7_USE_DIRECT_LINEAR
        and hidden_states.device.type == "npu"
        and getattr(linear, "tp_size", 1) == 1
        and getattr(quant_method, "__class__", type(None)).__name__
        == "UnquantizedLinearMethod"
    )


def _direct_linear(linear, hidden_states: torch.Tensor) -> torch.Tensor:
    bias = None
    if getattr(linear, "bias", None) is not None and not linear.skip_bias_add:
        bias = linear.bias
    return F.linear(hidden_states, linear.weight, bias)


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
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
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
        if _can_use_alt_recurrent(r, w, k, v, kk, a, initial_state):
            try:
                initial_state_npu = (
                    None
                    if initial_state is None
                    else initial_state.transpose(-1, -2).unsqueeze(0).contiguous()
                )
                output, final_state = torch.ops.ascend.npu_rwkv7_alt_recurrent(
                    r.unsqueeze(0),
                    w.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    kk.unsqueeze(0),
                    a.unsqueeze(0),
                    initial_state_npu,
                )
                dispatch_hit(DispatchKind.RECURRENT_SCAN)
                return output.squeeze(0), final_state.squeeze(0).transpose(-1, -2)
            except Exception:
                dispatch_fallback(DispatchKind.RECURRENT_SCAN, "alt_kernel_exception")

        if not _can_use_fused_recurrent(r, w, k, v, kk, a):
            dispatch_fallback(DispatchKind.RECURRENT_SCAN, "guard_false")
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
            dispatch_hit(DispatchKind.RECURRENT_SCAN)
            output = o.squeeze(0)  # [T, H, V]
            final_state = ht.squeeze(0) if ht is not None else None

            return output, final_state
        except Exception:
            dispatch_fallback(DispatchKind.RECURRENT_SCAN, "kernel_exception")
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
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
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
            dispatch_fallback(DispatchKind.RECURRENT_SCAN_VARLEN, "guard_false")
            return original_recurrent_scan_varlen(
                r, w, k, v, kk, a, query_start_loc, initial_state
            )

        # fused kernel expects cu_seqlens and B=1 in 4D; unsqueeze(0) gives [1, T, H, K]
        # where B=1 satisfies the cu_seqlens constraint. The 3D r.shape[0] is total_tokens,
        # not batch size, so it can be > 1 for multi-token varlen.

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
            dispatch_hit(DispatchKind.RECURRENT_SCAN_VARLEN)
            output = o.squeeze(0)  # [T, H, V]
            # ht: [N, H, K, V]
            final_state = ht

            return output, final_state
        except Exception:
            dispatch_fallback(DispatchKind.RECURRENT_SCAN_VARLEN, "kernel_exception")
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
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
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
            dispatch_fallback(DispatchKind.EPILOGUE, "guard_false")
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
            dispatch_hit(DispatchKind.EPILOGUE)
            output = output.to(hidden_dtype)
            if _can_use_direct_linear(self.o_proj, output):
                output = _direct_linear(self.o_proj, output)
            else:
                output, _ = self.o_proj(output)
            return output
        except Exception:
            dispatch_fallback(DispatchKind.EPILOGUE, "kernel_exception")
            return original_finalize(
                self, recurrent_output, r, k, v, g, hidden_dtype
            )

    RWKV7Attention._finalize_attention_output = _finalize_attention_output_ascend
    RWKV7Attention._ASCEND_EPILOGUE_PATCHED = True


def _patch_recurrent_inputs():
    """
    Patch RWKV7Attention._project_recurrent_inputs to use mix6 and kk_pre kernels when safe.

    The projection path performs:
    1. Mixing: xr = hidden_states + delta * x_r (6 mixed states)
    2. Linear projections: r, w, k, v via r_proj, k_proj, v_proj
    3. LoRA projections: w via w_lora, a via a_lora, g via g_lora
    4. v_first handling for layer_idx != 0
    5. Reshapes to [T, H, K] and [T, H, V]
    6. kk = normalize(k * k_k) and k = k * (1 + (a-1) * k_a)

    mix6 fuses the 6 mixing operations.
    kk_pre fuses the kk normalization and k adjustment.
    """
    ops = _get_ascend_ops()

    try:
        rwkv7_module = importlib.import_module("vllm_ascend.models.rwkv7")
        RWKV7Attention = rwkv7_module.RWKV7Attention
    except ImportError:
        return

    if getattr(RWKV7Attention, "_ASCEND_PROJECTION_PATCHED", False):
        return

    original_project = RWKV7Attention._project_recurrent_inputs

    # LOG_DECAY_SCALE constant from upstream
    LOG_DECAY_SCALE = rwkv7_module.LOG_DECAY_SCALE

    def _project_recurrent_inputs_ascend(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        v_first: torch.Tensor | None,
    ):
        x_r = self.x_r.squeeze(0).squeeze(0)
        x_w = self.x_w.squeeze(0).squeeze(0)
        x_k = self.x_k.squeeze(0).squeeze(0)
        x_v = self.x_v.squeeze(0).squeeze(0)
        x_a = self.x_a.squeeze(0).squeeze(0)
        x_g = self.x_g.squeeze(0).squeeze(0)

        # Try mix6 kernel first for the 6 mixing operations
        mix6_guard_passed = False
        mix6_used = False
        if _can_use_mix6_kernel(
            hidden_states, delta, x_r, x_w, x_k, x_v, x_a, x_g
        ):
            mix6_guard_passed = True
            try:
                xr, xw, xk, xv, xa, xg = ops.rwkv7_mix6(
                    hidden_states=hidden_states,
                    delta=delta,
                    x_r=x_r,
                    x_w=x_w,
                    x_k=x_k,
                    x_v=x_v,
                    x_a=x_a,
                    x_g=x_g,
                )
                mix6_used = True
                dispatch_hit(DispatchKind.MIX6)
            except Exception:
                dispatch_fallback(DispatchKind.MIX6, "kernel_exception")
                mix6_used = False

        if not mix6_used:
            xr = hidden_states.addcmul(delta, x_r)
            xw = hidden_states.addcmul(delta, x_w)
            xk = hidden_states.addcmul(delta, x_k)
            xv = hidden_states.addcmul(delta, x_v)
            xa = hidden_states.addcmul(delta, x_a)
            xg = hidden_states.addcmul(delta, x_g)
            if not mix6_guard_passed:
                dispatch_fallback(DispatchKind.MIX6, "guard_false")

        if _can_use_direct_linear(self.r_proj, xr):
            r = _direct_linear(self.r_proj, xr)
        else:
            r, _ = self.r_proj(xr)
        w = LOG_DECAY_SCALE * self.w_lora(xw).sigmoid()
        if _can_use_direct_linear(self.k_proj, xk):
            k = _direct_linear(self.k_proj, xk)
        else:
            k, _ = self.k_proj(xk)
        if _can_use_direct_linear(self.v_proj, xv):
            v = _direct_linear(self.v_proj, xv)
        else:
            v, _ = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError("RWKV7 layers after layer 0 require `v_first`.")
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
            v_first_out = v_first

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        r = r.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        w = w.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        k = k.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        a = a.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        v = v.view(-1, self.local_num_heads, self.head_v_dim).to(torch.float32)

        local_k_k = self.k_k[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )
        local_k_a = self.k_a[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )

        # Try kk_pre kernel for fused kk normalization and k adjustment
        # kk_pre expects k_k and k_a as [H, K], but local_k_k/a are [1, H, K]
        local_k_k_2d = local_k_k.squeeze(0)  # [H, K]
        local_k_a_2d = local_k_a.squeeze(0)  # [H, K]

        if _can_use_kk_pre_kernel(k, local_k_k_2d, a, local_k_a_2d):
            try:
                k_adj, kk = ops.rwkv7_kk_pre(
                    k=k,
                    k_k=local_k_k_2d,
                    a=a,
                    k_a=local_k_a_2d,
                )
                dispatch_hit(DispatchKind.KK_PRE)
                k = k_adj
            except Exception:
                dispatch_fallback(DispatchKind.KK_PRE, "kernel_exception")
                kk = torch.nn.functional.normalize(
                    k * local_k_k.to(torch.float32), dim=-1, p=2.0
                )
                k = k * (1 + (a - 1) * local_k_a.to(torch.float32))
        else:
            dispatch_fallback(DispatchKind.KK_PRE, "guard_false")
            kk = torch.nn.functional.normalize(
                k * local_k_k.to(torch.float32), dim=-1, p=2.0
            )
            k = k * (1 + (a - 1) * local_k_a.to(torch.float32))

        return r, w, k, v, kk, a, g, v_first_out

    RWKV7Attention._project_recurrent_inputs = _project_recurrent_inputs_ascend
    RWKV7Attention._ASCEND_PROJECTION_PATCHED = True


def _patch_linear_attention_metadata():
    """Expose cache-all block indices on vLLM's linear attention metadata."""
    try:
        linear_attn = importlib.import_module("vllm.v1.attention.backends.linear_attn")
    except ImportError:
        return

    builder = linear_attn.LinearAttentionMetadataBuilder
    if getattr(builder, "_RWKV7_CACHE_ALL_PATCHED", False):
        return

    original_build = builder.build

    def build_with_cache_all(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            return metadata

        block_size = self.kv_cache_spec.block_size
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        metadata.state_indices_tensor = common_attn_metadata.block_table_tensor
        metadata.num_computed_tokens = num_computed_tokens
        metadata.block_idx_last_computed_token = torch.clamp(
            torch.div(
                num_computed_tokens + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            - 1,
            min=0,
        )
        metadata.block_idx_first_scheduled_token = (
            torch.div(
                num_computed_tokens + block_size,
                block_size,
                rounding_mode="floor",
            )
            - 1
        )
        metadata.block_idx_last_scheduled_token = torch.clamp(
            torch.div(
                common_attn_metadata.seq_lens + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            - 1,
            min=0,
        )
        return metadata

    builder.build = build_with_cache_all
    builder._RWKV7_CACHE_ALL_PATCHED = True


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
    _patch_recurrent_inputs()
    _patch_linear_attention_metadata()


# Apply the patch when this module is imported
apply_patch()
