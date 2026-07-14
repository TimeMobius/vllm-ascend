# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project
# and the vLLM RWKV7 implementation.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# This is a standalone triton-ascend implementation of the RWKV7 fused
# recurrent operation with checkpoint support.
# ruff: noqa: E501
# mypy: ignore-errors

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
    rwkv7_recurrent_reference_with_checkpoints,
)


def _fused_recurrent_rwkv7_available() -> bool:
    """Check if triton-ascend is available and can run fused_recurrent_rwkv7."""
    if not HAS_TRITON:
        return False
    if not torch.npu.is_available():
        return False
    return True


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_CHECKPOINT_STATE": lambda args: args["hc"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    o,
    h0,
    ht,
    checkpoint_positions,
    checkpoint_offsets,
    hc,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_CHECKPOINT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused RWKV7 recurrent forward kernel.

    Key equations:
        b_act_a = -b_kk
        b_b = b_kk * b_a
        b_h = exp(b_w) * b_h + b_b * sum(b_act_a * b_h) + k * v

    Args:
        r: [B, T, H, K] - reception gate
        w: [B, T, H, K] - forget gate (in log space)
        k: [B, T, H, K] - key
        v: [B, T, H, V] - value
        kk: [B, T, H, K] - key normalization
        a: [B, T, H, K] - a gate
        o: [B, T, H, V] - output
        h0: [N, H, K, V] - initial state (N = number of sequences)
        ht: [N, H, K, V] - final state
        checkpoint_positions: [num_checkpoints] - positions to save checkpoints
        checkpoint_offsets: [N+1] - offsets for each sequence's checkpoints
        hc: [num_checkpoints, H, K, V] - checkpoint states
        cu_seqlens: [N+1] - cumulative sequence lengths for varlen
        scale: float - scaling factor for r
    """
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + bos * H * K + i_h * K + o_k
    p_w = w + bos * H * K + i_h * K + o_k
    p_k = k + bos * H * K + i_h * K + o_k
    p_v = v + bos * H * V + i_h * V + o_v
    p_a = a + bos * H * K + i_h * K + o_k
    p_kk = kk + bos * H * K + i_h * K + o_k
    p_o = o + bos * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if STORE_CHECKPOINT_STATE:
        checkpoint_idx = tl.load(checkpoint_offsets + i_n).to(tl.int64)
        checkpoint_end = tl.load(checkpoint_offsets + i_n + 1).to(tl.int64)
        has_checkpoint = checkpoint_idx < checkpoint_end
        checkpoint_pos = tl.load(
            checkpoint_positions + checkpoint_idx,
            mask=has_checkpoint,
            other=0,
        ).to(tl.int64)

    for t in range(0, T):
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_act_a = -b_kk
        b_b = b_kk * b_a

        b_h = (
            tl.exp(b_w)[:, None] * b_h
            + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
        )
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if STORE_CHECKPOINT_STATE:
            should_store = has_checkpoint & (t == checkpoint_pos)
            safe_checkpoint_idx = tl.where(has_checkpoint, checkpoint_idx, 0)
            p_hc = hc + (safe_checkpoint_idx * H + i_h) * K * V + o_k[:, None] * V + o_v
            tl.store(
                p_hc,
                b_h.to(p_hc.dtype.element_ty),
                mask=mask_h & should_store,
            )
            checkpoint_idx += should_store.to(tl.int64)
            has_checkpoint = checkpoint_idx < checkpoint_end
            checkpoint_pos = tl.load(
                checkpoint_positions + checkpoint_idx,
                mask=has_checkpoint,
                other=checkpoint_pos,
            ).to(tl.int64)

        p_r += H * K
        p_w += H * K
        p_k += H * K
        p_v += H * V
        p_a += H * K
        p_kk += H * K
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def _validate_inputs(
    r: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    checkpoint_positions: torch.Tensor | None,
    checkpoint_offsets: torch.Tensor | None,
    output_checkpoint_states: bool,
) -> None:
    """Validate input tensor dimensions and checkpoint parameters."""
    if r.ndim != 4:
        raise ValueError(f"`r` must be 4D, got {r.ndim}.")
    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError("When `cu_seqlens` is provided, the batch size must be 1.")
    if output_checkpoint_states and (
        checkpoint_positions is None or checkpoint_offsets is None
    ):
        raise ValueError(
            "`checkpoint_positions` and `checkpoint_offsets` are required "
            "when `output_checkpoint_states=True`."
        )


def _prepare_state_tensors(
    r: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    output_checkpoint_states: bool,
    checkpoint_positions: torch.Tensor | None,
    checkpoint_offsets: torch.Tensor | None,
    B: int,
    H: int,
    K: int,
    V: int,
    N: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Prepare output state tensors (h0, ht, hc)."""
    h0 = initial_state
    ht = None
    if output_final_state:
        if initial_state is None:
            h0 = r.new_zeros((N, H, K, V), dtype=torch.float32)
        ht = r.new_empty((N, H, K, V), dtype=torch.float32)

    hc = None
    if output_checkpoint_states:
        num_checkpoints = int(checkpoint_offsets[-1].item())
        hc = r.new_empty((num_checkpoints, H, K, V), dtype=torch.float32)

    return h0, ht, hc


def fused_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Fused RWKV7 recurrent operation with optional checkpoint support.

    This is a standalone triton-ascend implementation that provides the same
    functionality as the upstream vLLM fused_mul_recurrent_rwkv7 but optimized
    for Ascend NPUs.

    Args:
        r (torch.Tensor): Reception gate of shape [B, T, H, K]
        w (torch.Tensor): Forget gate of shape [B, T, H, K] (in log space)
        k (torch.Tensor): Key of shape [B, T, H, K]
        v (torch.Tensor): Value of shape [B, T, H, V]
        kk (torch.Tensor): Key normalization of shape [B, T, H, K]
        a (torch.Tensor): A gate of shape [B, T, H, K]
        scale (float): Scaling factor for r. Default: 1.0
        initial_state (torch.Tensor, optional): Initial state of shape [N, H, K, V]
        output_final_state (bool): Whether to output final state. Default: False
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for varlen
        checkpoint_positions (torch.Tensor, optional): Positions to save checkpoints
        checkpoint_offsets (torch.Tensor, optional): Offsets for each sequence's checkpoints
        output_checkpoint_states (bool): Whether to output checkpoint states. Default: False

    Returns:
        tuple: (output, final_state, checkpoint_states)
            - output: [B, T, H, V] output tensor
            - final_state: [N, H, K, V] or None
            - checkpoint_states: [num_checkpoints, H, K, V] or None
    """
    _validate_inputs(r, cu_seqlens, checkpoint_positions, checkpoint_offsets, output_checkpoint_states)

    # Check if we can use triton-ascend
    # Conditions for using Triton kernel:
    # 1. Triton-ascend must be available
    # 2. Device must be NPU
    # 3. All tensors must be non-empty and contiguous
    can_use_triton = (
        _fused_recurrent_rwkv7_available()
        and r.device.type == "npu"
        and r.numel() > 0
        and r.is_contiguous()
        and w.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and kk.is_contiguous()
        and a.is_contiguous()
    )

    if not can_use_triton:
        # Fall back to reference implementation
        return rwkv7_recurrent_reference_with_checkpoints(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            output_checkpoint_states=output_checkpoint_states,
            scale=scale,
        )

    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)

    # Determine block sizes
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 64)

    # Prepare output tensor
    o = torch.empty_like(v)

    # Prepare state tensors
    h0 = initial_state
    ht = None
    if output_final_state:
        if initial_state is None:
            h0 = r.new_zeros((N, H, K, V), dtype=torch.float32)
        ht = r.new_empty((N, H, K, V), dtype=torch.float32)

    hc = None
    if output_checkpoint_states:
        if checkpoint_positions is None or checkpoint_offsets is None:
            raise ValueError(
                "`checkpoint_positions` and `checkpoint_offsets` are required "
                "when `output_checkpoint_states=True`."
            )
        num_checkpoints = int(checkpoint_offsets[-1].item())
        hc = r.new_empty((num_checkpoints, H, K, V), dtype=torch.float32)

    # Grid: (triton.cdiv(V, BV), N * H)
    grid = (triton.cdiv(V, BV), N * H)
    fused_recurrent_rwkv7_fwd_kernel[grid](
        r=r.contiguous(),
        w=w.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        kk=kk.contiguous(),
        a=a.contiguous(),
        o=o,
        h0=h0,
        ht=ht,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        hc=hc,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    return o, ht, hc