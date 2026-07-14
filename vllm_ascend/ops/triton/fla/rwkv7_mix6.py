# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501
# mypy: ignore-errors

"""
RWKV7 mix6 operation for triton-ascend.

This module provides the rwkv7_mix6 operation that fuses:
    xr = hidden_states + delta * x_r
    xw = hidden_states + delta * x_w
    xk = hidden_states + delta * x_k
    xv = hidden_states + delta * x_v
    xa = hidden_states + delta * x_a
    xg = hidden_states + delta * x_g

The triton kernel provides acceleration on NPU via the triton-ascend backend.
When triton-ascend is unavailable, falls back to the pure torch reference.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def rwkv7_mix6_fwd_kernel(
    x,
    delta,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    xr,
    xw,
    xk,
    xv,
    xa,
    xg,
    numel,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RWKV7 mix6 kernel.

    Computes 6 output tensors where each is: x + delta * x_i
    for i in {r, w, k, v, a, g}.

    Args:
        x: hidden_states tensor of shape [numel]
        delta: delta tensor of shape [numel]
        x_r/x_w/x_k/x_v/x_a/x_g: mixing tensors of shape [hidden_size]
        xr/xw/xk/xv/xa/xg: output tensors of shape [numel]
        numel: total number of elements (batch * seq * hidden_size)
        hidden_size: last dimension size
        BLOCK_SIZE: triton block size
    """
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    cols = offsets % hidden_size

    x_vals = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    delta_vals = tl.load(delta + offsets, mask=mask, other=0).to(tl.float32)

    x_r_vals = tl.load(x_r + cols, mask=mask, other=0).to(tl.float32)
    x_w_vals = tl.load(x_w + cols, mask=mask, other=0).to(tl.float32)
    x_k_vals = tl.load(x_k + cols, mask=mask, other=0).to(tl.float32)
    x_v_vals = tl.load(x_v + cols, mask=mask, other=0).to(tl.float32)
    x_a_vals = tl.load(x_a + cols, mask=mask, other=0).to(tl.float32)
    x_g_vals = tl.load(x_g + cols, mask=mask, other=0).to(tl.float32)

    tl.store(xr + offsets, x_vals + delta_vals * x_r_vals, mask=mask)
    tl.store(xw + offsets, x_vals + delta_vals * x_w_vals, mask=mask)
    tl.store(xk + offsets, x_vals + delta_vals * x_k_vals, mask=mask)
    tl.store(xv + offsets, x_vals + delta_vals * x_v_vals, mask=mask)
    tl.store(xa + offsets, x_vals + delta_vals * x_a_vals, mask=mask)
    tl.store(xg + offsets, x_vals + delta_vals * x_g_vals, mask=mask)


def rwkv7_mix6_reference(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Pure torch reference implementation of RWKV7 mix6.

    Computes: xr = hidden_states + delta * x_r (same for other tensors)
    """
    xr = hidden_states.addcmul(delta, x_r)
    xw = hidden_states.addcmul(delta, x_w)
    xk = hidden_states.addcmul(delta, x_k)
    xv = hidden_states.addcmul(delta, x_v)
    xa = hidden_states.addcmul(delta, x_a)
    xg = hidden_states.addcmul(delta, x_g)
    return xr, xw, xk, xv, xa, xg


def rwkv7_mix6(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    RWKV7 mix6 operation.

    Fuses 6 mixing operations: xr = hidden_states + delta * x_r (and similar).

    This function provides acceleration via triton-ascend kernel on NPU.
    When triton-ascend is unavailable or conditions are not met, falls back
    to the pure torch reference implementation.

    Args:
        hidden_states: Tensor of shape [batch, seq, hidden_size] (3D) or
                       [batch*seq, hidden_size] (2D, flattened)
        delta: Tensor of same shape as hidden_states
        x_r, x_w, x_k, x_v, x_a, x_g: Mixing vectors of shape [hidden_size]

    Returns:
        Tuple of 6 tensors, each of same shape as hidden_states:
        (xr, xw, xk, xv, xa, xg)
    """
    if hidden_states.shape != delta.shape:
        raise ValueError(
            "`hidden_states` and `delta` must have the same shape, got "
            f"{hidden_states.shape} and {delta.shape}."
        )

    # Store original shape for 3D restoration after triton kernel
    original_shape = hidden_states.shape
    is_3d = hidden_states.ndim == 3

    # Flatten 3D to 2D for triton kernel (which expects [batch*seq, hidden])
    if is_3d:
        hidden_states = hidden_states.flatten(0, 1)
        delta = delta.flatten(0, 1)

    # Fall back to reference when triton-ascend is unavailable or device is not suitable
    if (
        not HAS_TRITON
        or hidden_states.device.type not in ("npu", "cuda")
        or hidden_states.numel() == 0
        or not hidden_states.is_contiguous()
        or not delta.is_contiguous()
    ):
        xr, xw, xk, xv, xa, xg = rwkv7_mix6_reference(
            hidden_states=hidden_states,
            delta=delta,
            x_r=x_r,
            x_w=x_w,
            x_k=x_k,
            x_v=x_v,
            x_a=x_a,
            x_g=x_g,
        )
        # Restore 3D shape if input was 3D (consistent with triton path)
        if is_3d:
            xr = xr.view(original_shape)
            xw = xw.view(original_shape)
            xk = xk.view(original_shape)
            xv = xv.view(original_shape)
            xa = xa.view(original_shape)
            xg = xg.view(original_shape)
        return xr, xw, xk, xv, xa, xg

    numel = hidden_states.numel()
    hidden_size = hidden_states.shape[-1]
    output_dtype = torch.result_type(hidden_states, x_r)
    xr = torch.empty_like(hidden_states, dtype=output_dtype)
    xw = torch.empty_like(hidden_states, dtype=output_dtype)
    xk = torch.empty_like(hidden_states, dtype=output_dtype)
    xv = torch.empty_like(hidden_states, dtype=output_dtype)
    xa = torch.empty_like(hidden_states, dtype=output_dtype)
    xg = torch.empty_like(hidden_states, dtype=output_dtype)

    block_size = min(2048, triton.next_power_of_2(hidden_size))
    num_warps = 4 if block_size <= 1024 else 8
    grid = (triton.cdiv(numel, block_size),)

    rwkv7_mix6_fwd_kernel[grid](
        x=hidden_states,
        delta=delta,
        x_r=x_r,
        x_w=x_w,
        x_k=x_k,
        x_v=x_v,
        x_a=x_a,
        x_g=x_g,
        xr=xr,
        xw=xw,
        xk=xk,
        xv=xv,
        xa=xa,
        xg=xg,
        numel=numel,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    if is_3d:
        xr = xr.view(original_shape)
        xw = xw.view(original_shape)
        xk = xk.view(original_shape)
        xv = xv.view(original_shape)
        xa = xa.view(original_shape)
        xg = xg.view(original_shape)
    return xr, xw, xk, xv, xa, xg