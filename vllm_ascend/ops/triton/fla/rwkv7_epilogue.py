# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# This module provides the RWKV7 epilogue operation (rwkv7_lnx_rkvres_xg)
# adapted for Ascend NPU using Triton.
#
# The operation computes:
#   output = (GroupNorm(recurrent_output) + recurrent_correction) * gating
#
# Where:
#   - GroupNorm is applied per-token per-head over head_v_dim
#   - recurrent_correction = sum(r * k * r_k, axis=-1, keepdim=True) * v
#   - gating is element-wise multiplication with g
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit(do_not_specialize=["eps"])
def rwkv7_lnx_rkvres_xg_fwd_kernel(
    recurrent_output,
    r,
    k,
    v,
    r_k,
    weight,
    bias,
    g,
    out,
    num_heads,
    head_dim,
    head_v_dim,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    RWKV7 epilogue kernel: GroupNorm + RecurrentCorrection + Gating.

    Args:
        recurrent_output: [num_tokens, num_heads, head_v_dim] - recurrent hidden states
        r: [num_tokens, num_heads, head_dim] - reset gate
        k: [num_tokens, num_heads, head_dim] - key
        v: [num_tokens, num_heads, head_v_dim] - value
        r_k: [num_heads, head_dim] - recurrent correction weights
        weight: [num_heads * head_v_dim] - group norm weight
        bias: [num_heads * head_v_dim] - group norm bias
        g: [num_tokens, num_heads * head_v_dim] - gating
        out: [num_tokens, num_heads * head_v_dim] - output
    """
    row_head = tl.program_id(0).to(tl.int64)
    head_idx = row_head % num_heads
    token_idx = row_head // num_heads

    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = tl.arange(0, BLOCK_V)
    mask_k = k_offsets < head_dim
    mask_v = v_offsets < head_v_dim

    key_base = row_head * head_dim
    value_base = row_head * head_v_dim
    gate_base = token_idx * num_heads * head_v_dim + head_idx * head_v_dim
    r_k_base = head_idx * head_dim
    affine_base = head_idx * head_v_dim

    # GroupNorm computation on recurrent_output
    # Load recurrent_output values
    x_vals = tl.load(
        recurrent_output + value_base + v_offsets,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)

    # Compute mean over head_v_dim
    mean = tl.sum(x_vals, axis=0) / head_v_dim
    centered = tl.where(mask_v, x_vals - mean, 0.0)

    # Compute variance and rstd
    var = tl.sum(centered * centered, axis=0) / head_v_dim
    rstd = tl.rsqrt(var + eps)

    # Load r, k, r_k for recurrent correction
    r_vals = tl.load(r + key_base + k_offsets, mask=mask_k, other=0).to(tl.float32)
    k_vals = tl.load(k + key_base + k_offsets, mask=mask_k, other=0).to(tl.float32)
    r_k_vals = tl.load(r_k + r_k_base + k_offsets, mask=mask_k, other=0).to(tl.float32)

    # Compute correction scale: sum(r * k * r_k, axis=-1)
    correction_scale = tl.sum(r_vals * k_vals * r_k_vals, axis=0)

    # Load v, weight, bias, g for final computation
    v_vals = tl.load(v + value_base + v_offsets, mask=mask_v, other=0).to(tl.float32)
    weight_vals = tl.load(weight + affine_base + v_offsets, mask=mask_v, other=0).to(
        tl.float32
    )
    bias_vals = tl.load(bias + affine_base + v_offsets, mask=mask_v, other=0).to(
        tl.float32
    )
    g_vals = tl.load(g + gate_base + v_offsets, mask=mask_v, other=0).to(tl.float32)

    # Compute final output:
    # y = centered * rstd * weight + bias + correction_scale * v
    # out = y * g
    y = centered * rstd * weight_vals + bias_vals + correction_scale * v_vals
    tl.store(out + gate_base + v_offsets, (y * g_vals).to(out.dtype.element_ty), mask=mask_v)


def rwkv7_lnx_rkvres_xg(
    recurrent_output: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    RWKV7 epilogue: GroupNorm + RecurrentCorrection + Gating.

    This is the standalone Ascend Triton implementation that mirrors the upstream
    vLLM `rwkv7_lnx_rkvres_xg` operation semantics.

    Args:
        recurrent_output: [num_tokens, num_heads, head_v_dim] - recurrent hidden states
        r: [num_tokens, num_heads, head_dim] - reset gate
        k: [num_tokens, num_heads, head_dim] - key
        v: [num_tokens, num_heads, head_v_dim] - value
        r_k: [num_heads, head_dim] - recurrent correction weights
        weight: [num_heads * head_v_dim] - group norm weight
        bias: [num_heads * head_v_dim] - group norm bias
        g: [num_tokens, num_heads * head_v_dim] - gating
        eps: GroupNorm epsilon
        output_dtype: Output dtype (defaults to g.dtype)

    Returns:
        out: [num_tokens, num_heads * head_v_dim]
    """
    # Input validation (mirrors upstream)
    if recurrent_output.ndim != 3:
        raise ValueError(
            f"`recurrent_output` must be 3D, got {recurrent_output.ndim}."
        )
    if r.shape != k.shape:
        raise ValueError(f"`r` and `k` must match, got {r.shape} and {k.shape}.")
    if recurrent_output.shape != v.shape:
        raise ValueError(
            "`recurrent_output` and `v` must match, got "
            f"{recurrent_output.shape} and {v.shape}."
        )
    if recurrent_output.shape[:2] != r.shape[:2]:
        raise ValueError(
            "`recurrent_output` and `r` must share token/head dimensions, got "
            f"{recurrent_output.shape[:2]} and {r.shape[:2]}."
        )
    if r_k.shape != r.shape[1:]:
        raise ValueError(f"`r_k` must have shape {r.shape[1:]}, got {r_k.shape}.")

    num_tokens, num_heads, head_v_dim = recurrent_output.shape
    head_dim = r.shape[-1]
    local_value_dim = num_heads * head_v_dim

    if weight.shape != (local_value_dim,) or bias.shape != (local_value_dim,):
        raise ValueError(
            "`weight` and `bias` must match the flattened local value dimension "
            f"{local_value_dim}, got {weight.shape} and {bias.shape}."
        )
    if g.shape != (num_tokens, local_value_dim):
        raise ValueError(
            f"`g` must have shape {(num_tokens, local_value_dim)}, got {g.shape}."
        )

    if output_dtype is None:
        output_dtype = g.dtype

    # Fallback guards: check Triton availability, device type, empty inputs, and contiguity
    # These guards are consistent with existing vllm-ascend patterns (see rwkv7_mix6.py)
    if (
        not HAS_TRITON
        or recurrent_output.device.type not in ("npu", "cuda")
        or recurrent_output.numel() == 0
        or not recurrent_output.is_contiguous()
        or not r.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not r_k.is_contiguous()
        or not weight.is_contiguous()
        or not bias.is_contiguous()
        or not g.is_contiguous()
    ):
        return rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=output_dtype,
        )

    # Compute block sizes
    block_k = triton.next_power_of_2(head_dim)
    block_v = triton.next_power_of_2(head_v_dim)

    # Fallback for large block sizes (same threshold as upstream: 1024)
    if block_k > 1024 or block_v > 1024:
        return rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=output_dtype,
        )

    # Allocate output
    out = torch.empty(
        (num_tokens, local_value_dim),
        device=g.device,
        dtype=output_dtype,
    )

    # Launch kernel
    num_warps = 4 if max(block_k, block_v) <= 64 else 8
    grid = (num_tokens * num_heads,)
    rwkv7_lnx_rkvres_xg_fwd_kernel[grid](
        recurrent_output=recurrent_output,
        r=r,
        k=k,
        v=v,
        r_k=r_k,
        weight=weight,
        bias=bias,
        g=g,
        out=out,
        num_heads=num_heads,
        head_dim=head_dim,
        head_v_dim=head_v_dim,
        eps=eps,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    return out


def rwkv7_lnx_rkvres_xg_reference(
    recurrent_output: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Reference implementation of rwkv7_lnx_rkvres_xg using PyTorch ops.

    This is the pure PyTorch fallback that preserves exact semantics.
    """
    if output_dtype is None:
        output_dtype = g.dtype

    num_heads = recurrent_output.shape[1]
    local_value_dim = recurrent_output.shape[1] * recurrent_output.shape[2]

    # GroupNorm via torch.nn.functional.group_norm
    output = torch.nn.functional.group_norm(
        recurrent_output.reshape(-1, local_value_dim).to(torch.float32).unsqueeze(-1),
        num_groups=num_heads,
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        eps=eps,
    ).squeeze(-1)

    # Recurrent correction: ((r * k * r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v).reshape(...)
    correction = ((r * k * r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v).reshape(
        -1, local_value_dim
    )

    return ((output + correction) * g.to(torch.float32)).to(output_dtype)