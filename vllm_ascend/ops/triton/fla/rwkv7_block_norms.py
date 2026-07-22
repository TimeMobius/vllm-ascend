# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attn_norm + ffn_norm Triton-Ascend kernel for RWKV7 decode.

Replaces two ``F.layer_norm`` calls in ``RWKV7Block._run_decode_batch``
with a single Triton launch. The two norms have different inputs
(``residual`` vs ``residual + attn_out``) and different parameters
(``attn_norm`` vs ``ffn_norm``), but operate on identical shape
``[hidden_size]`` during the T=1 decode path. 61 calls per request
eliminated from the launch-overhead-limited path.

Math (per LayerNorm):

  mean = sum(x) / H
  var  = sum((x - mean)^2) / H
  out  = (x - mean) * rsqrt(var + eps) * weight + bias
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton


def _rwkv7_block_norms_reference(
    residual: torch.Tensor,
    hidden_after_attn: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    attn_norm_bias: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    ffn_norm_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the fused block norms."""
    import torch.nn.functional as F

    attn_input = F.layer_norm(
        residual,
        normalized_shape=residual.shape[-1:],
        weight=attn_norm_weight,
        bias=attn_norm_bias,
        eps=eps,
    )
    ffn_input = F.layer_norm(
        hidden_after_attn,
        normalized_shape=hidden_after_attn.shape[-1:],
        weight=ffn_norm_weight,
        bias=ffn_norm_bias,
        eps=eps,
    )
    return attn_input, ffn_input


if HAS_TRITON:

    @triton.jit
    def rwkv7_block_norms_fwd_kernel(
        residual_ptr,
        hidden_ptr,
        attn_w_ptr,
        attn_b_ptr,
        ffn_w_ptr,
        ffn_b_ptr,
        attn_out_ptr,
        ffn_out_ptr,
        H: tl.constexpr,
        eps,
        BLOCK: tl.constexpr,
    ):
        """One program per element; loop to compute mean/var over H.

        For T=1 the inputs are 1D [H], so each program processes one
        element and reduces over H to get mean/var.
        """
        idx = tl.program_id(0)
        if idx >= H:
            return

        offsets = tl.arange(0, BLOCK)
        mask = offsets < H

        # attn_norm on residual
        x = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / H
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(attn_w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(attn_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        attn_out = (centered * rstd) * w + b
        tl.store(attn_out_ptr + offsets, attn_out.to(hidden_ptr.dtype.element_ty), mask=mask)

        # ffn_norm on hidden_after_attn
        x = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / H
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(ffn_w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ffn_b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        ffn_out = (centered * rstd) * w + b
        tl.store(ffn_out_ptr + offsets, ffn_out.to(hidden_ptr.dtype.element_ty), mask=mask)


def rwkv7_block_norms(
    residual: torch.Tensor,
    hidden_after_attn: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    attn_norm_bias: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    ffn_norm_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused attn_norm(residual) + ffn_norm(hidden_after_attn).

    Returns ``(attn_input, ffn_input)``. Falls back to PyTorch when guards
    fail. Guards: Triton available, NPU/CUDA, contiguous + matching
    dtype + matching shape, H <= 8192.
    """
    if (
        not HAS_TRITON
        or residual.device.type not in ("npu", "cuda")
        or residual.shape != hidden_after_attn.shape
        or residual.dtype != hidden_after_attn.dtype
        or attn_norm_weight.shape != attn_norm_bias.shape
        or ffn_norm_weight.shape != ffn_norm_bias.shape
        or attn_norm_weight.dtype != residual.dtype
        or ffn_norm_weight.dtype != residual.dtype
        or not residual.is_contiguous()
        or not hidden_after_attn.is_contiguous()
    ):
        return _rwkv7_block_norms_reference(
            residual,
            hidden_after_attn,
            attn_norm_weight,
            attn_norm_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            eps,
        )

    H = residual.shape[-1]
    if (
        H > 8192
        or attn_norm_weight.shape[0] != H
        or ffn_norm_weight.shape[0] != H
    ):
        return _rwkv7_block_norms_reference(
            residual,
            hidden_after_attn,
            attn_norm_weight,
            attn_norm_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            eps,
        )

    attn_out = torch.empty_like(residual)
    ffn_out = torch.empty_like(residual)

    BLOCK = triton.next_power_of_2(H)
    if BLOCK > 8192:
        return _rwkv7_block_norms_reference(
            residual,
            hidden_after_attn,
            attn_norm_weight,
            attn_norm_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            eps,
        )

    num_warps = 4 if BLOCK <= 512 else 8
    rwkv7_block_norms_fwd_kernel[(H,)](
        residual,
        hidden_after_attn,
        attn_norm_weight,
        attn_norm_bias,
        ffn_norm_weight,
        ffn_norm_bias,
        attn_out,
        ffn_out,
        H=H,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return attn_out, ffn_out