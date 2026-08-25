# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# This implementation ports the upstream RWKV7 kk_pre Triton+FLA operation
# to a standalone triton-ascend implementation with graceful fallback.
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


def rwkv7_kk_pre_available() -> bool:
    """Check if triton-ascend is available and can run rwkv7_kk_pre."""
    if not HAS_TRITON:
        return False
    if not torch.npu.is_available():
        return False
    return True


@triton.jit
def rwkv7_kk_pre_fwd_kernel(
    k,
    a,
    k_k,
    k_a,
    k_out,
    kk_out,
    num_rows,
    num_heads,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton-Ascend kernel for RWKV7 kk_pre operation.

    This kernel performs:
    1. kk_raw = k * k_k (element-wise multiplication)
    2. kk = normalize(kk_raw, dim=-1) using rsqrt for L2 normalization
    3. k_adj = k * (1 + (a - 1) * k_a)

    Args:
        k: [num_rows, head_dim] - key tensor (flattened from [T, H, K])
        a: [num_rows, head_dim] - activation tensor (same shape as k)
        k_k: [num_heads, head_dim] - key modulation weights
        k_a: [num_heads, head_dim] - activation modulation weights
        k_out: [num_rows, head_dim] - adjusted key output
        kk_out: [num_rows, head_dim] - normalized kk output
        num_rows: total number of rows (T * H)
        num_heads: number of attention heads
        head_dim: dimension of each head
        eps: small constant for numerical stability in rsqrt
    """
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0)
    for row in tl.range(pid, num_rows, stride):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < head_dim
        head_idx = row % num_heads

        row_offset = row * head_dim
        head_offset = head_idx * head_dim

        # Load k and a values for this row
        k_vals = tl.load(k + row_offset + offsets, mask=mask, other=0).to(tl.float32)
        a_vals = tl.load(a + row_offset + offsets, mask=mask, other=0).to(tl.float32)

        # Load k_k and k_a for this head
        k_k_vals = tl.load(k_k + head_offset + offsets, mask=mask, other=0).to(tl.float32)
        k_a_vals = tl.load(k_a + head_offset + offsets, mask=mask, other=0).to(tl.float32)

        # Compute kk_raw = k * k_k (element-wise)
        kk_raw = k_vals * k_k_vals

        # Compute L2 normalization using rsqrt
        # kk = kk_raw / ||kk_raw||_2
        # Using rsqrt for efficiency: rsqrt(x) = 1/sqrt(x)
        kk_squared_sum = tl.sum(kk_raw * kk_raw, axis=0)
        rstd = tl.rsqrt(kk_squared_sum + eps)
        kk_vals = kk_raw * rstd

        # Compute k_adj = k * (1 + (a - 1) * k_a)
        # This applies the activation modulation to the key
        k_adj = k_vals * (1.0 + (a_vals - 1.0) * k_a_vals)

        # Store results
        tl.store(k_out + row_offset + offsets, k_adj, mask=mask)
        tl.store(kk_out + row_offset + offsets, kk_vals, mask=mask)


def rwkv7_kk_pre_reference(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of rwkv7_kk_pre using PyTorch operations.

    This is the fallback implementation when triton-ascend is unavailable.

    Args:
        k: [T, H, K] - key tensor
        k_k: [H, K] - key modulation weights
        a: [T, H, K] - activation tensor
        k_a: [H, K] - activation modulation weights

    Returns:
        k_adj: [T, H, K] - adjusted key tensor
        kk: [T, H, K] - normalized kk tensor
    """
    kk = torch.nn.functional.normalize(k * k_k, dim=-1, p=2.0)
    k_adj = k * (1 + (a - 1) * k_a)
    return k_adj, kk


def rwkv7_kk_pre(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RWKV7 kk_pre operation with triton-ascend support.

    This operation performs key preprocessing for the RWKV7 recurrent mechanism:
    1. Computes kk = normalize(k * k_k, dim=-1) - normalized key-key product
    2. Computes k_adj = k * (1 + (a - 1) * k_a) - adjusted key with activation

    Args:
        k: [T, H, K] - key tensor
        k_k: [H, K] - key modulation weights
        a: [T, H, K] - activation tensor
        k_a: [H, K] - activation modulation weights
        eps: small constant for numerical stability (default: 1e-12)

    Returns:
        k_adj: [T, H, K] - adjusted key tensor
        kk: [T, H, K] - normalized kk tensor
    """
    # Validate input shapes
    if k.shape != a.shape:
        raise ValueError(f"`k` and `a` must match, got {k.shape} and {a.shape}.")
    if k.ndim != 3:
        raise ValueError(f"`k` must be 3D, got {k.ndim}.")
    if k_k.shape != k_a.shape:
        raise ValueError(f"`k_k` and `k_a` must match, got {k_k.shape} and {k_a.shape}.")
    if k_k.ndim != 2:
        raise ValueError(f"`k_k` must be 2D, got {k_k.ndim}.")
    if k.shape[1:] != k_k.shape:
        raise ValueError(
            "`k_k`/`k_a` must match the head layout of `k`, got "
            f"{k.shape[1:]} and {k_k.shape}."
        )

    # Check if we can use triton-ascend
    # Conditions for using Triton kernel:
    # 1. Triton must be available
    # 2. Device must be NPU
    # 3. Tensors must be non-empty and contiguous
    can_use_triton = (
        rwkv7_kk_pre_available()
        and k.device.type == "npu"
        and k.numel() > 0
        and k.is_contiguous()
        and a.is_contiguous()
        and k_k.is_contiguous()
        and k_a.is_contiguous()
    )

    if not can_use_triton:
        return rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

    # Allocate output tensors
    output_dtype = torch.result_type(k, k_k)
    k_out = torch.empty_like(k, dtype=output_dtype)
    kk_out = torch.empty_like(k, dtype=output_dtype)

    # Compute grid dimensions
    num_rows = k.shape[0] * k.shape[1]  # T * H
    num_heads = k.shape[1]  # H
    head_dim = k.shape[2]  # K

    # Triton kernel configuration
    block_size = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_size <= 64 else 8

    # Bound the physical launch and process the logical rows with a grid-stride loop.
    init_device_properties_triton()
    grid_size = min(num_rows, get_vectorcore_num())
    rwkv7_kk_pre_fwd_kernel[(grid_size,)](
        k=k,
        a=a,
        k_k=k_k,
        k_a=k_a,
        k_out=k_out,
        kk_out=kk_out,
        num_rows=num_rows,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return k_out, kk_out
