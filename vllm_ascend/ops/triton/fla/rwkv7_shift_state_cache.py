# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shift-state cache update for RWKV7 decode.

Replaces two consecutive ``kv_cache[0].index_copy_(...)`` (attention shift)
and ``kv_cache[2].index_copy_(...)`` (FFN shift) calls at the decode output
with a single Triton launch. The recurrent ``kv_cache[1]`` path is left to
its existing launcher.

The kernel uses a tiled 2D grid ``(ceil(B / BLOCK_B), ceil(D / BLOCK_D))``
so production row widths (e.g. D=4096) are supported without falling back
to the reference path. Each program writes a ``BLOCK_D``-wide tile of
``BLOCK_B`` batch rows into both caches in one launch.

The helper is opt-in by environment (``VLLM_ASCEND_RWKV7_DISABLE_TRITON``)
and falls back to a pure-PyTorch reference when any precondition is not
met (cache/source dtype mismatch, non-contiguous layout, missing Triton,
CPU device, slot=-1 padding, etc.). ``slot_id < 0`` leaves both caches
untouched for that row, matching the existing state-cache contract.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend import envs

__all__ = ["rwkv7_shift_state_cache"]

# Rows per program for the grouped cache kernel; mirrors the existing
# ``BLOCK_B_CACHE`` policy used by ``rwkv7_recurrent_t1_matrix``.
_BLOCK_B_SHIFT_STATE: int = 4

# Compile-time feature-tile width. Power-of-two required by ``tl.arange``;
# 512 lets D=4096 dispatch with 8 feature tiles per batch group.
_BLOCK_D_SHIFT_STATE: int = 512


def _can_use_triton(
    attn_cache: torch.Tensor,
    ffn_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    attn_shift: torch.Tensor,
    ffn_shift: torch.Tensor,
) -> bool:
    """Return True only when the fused Triton path is safe to dispatch."""
    if not HAS_TRITON or envs.VLLM_ASCEND_RWKV7_DISABLE_TRITON or attn_cache.device.type not in ("npu", "cuda"):
        return False
    if (
        attn_cache.dtype != ffn_cache.dtype
        or attn_cache.device != ffn_cache.device
        or not attn_cache.is_contiguous()
        or not ffn_cache.is_contiguous()
    ):
        return False
    if (
        attn_shift.dtype != attn_cache.dtype
        or ffn_shift.dtype != ffn_cache.dtype
        or attn_shift.device != attn_cache.device
        or ffn_shift.device != ffn_cache.device
        or not attn_shift.is_contiguous()
        or not ffn_shift.is_contiguous()
    ):
        return False
    if slot_ids.device != attn_cache.device or slot_ids.dtype != torch.long or not slot_ids.is_contiguous():
        return False
    if attn_cache.ndim != 2 or ffn_cache.ndim != 2 or attn_cache.shape[1] != ffn_cache.shape[1]:
        return False
    if attn_cache.shape[1] <= 0:
        return False
    B = slot_ids.shape[0]
    return attn_shift.shape == (B, attn_cache.shape[1]) and ffn_shift.shape == (B, attn_cache.shape[1])


if HAS_TRITON:

    @triton.jit
    def _rwkv7_shift_state_cache_fwd_grouped_kernel(
        attn_cache_ptr,
        ffn_cache_ptr,
        slot_ids_ptr,
        attn_shift_ptr,
        ffn_shift_ptr,
        B,
        D,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Write ``attn_shift``/``ffn_shift`` rows into both caches per slot.

        Grid: ``(ceil(B / BLOCK_B), ceil(D / BLOCK_D))``. Each program
        handles ``BLOCK_B`` consecutive batch rows and a ``BLOCK_D``-wide
        feature tile. Padding lanes (``batch_idx >= B``), out-of-range
        feature lanes (``d_offset >= D``), and lanes with ``slot_id < 0``
        leave both caches unchanged, matching the existing
        ``index_copy_``/state-cache contract.
        """
        b_group = tl.program_id(0)
        d_tile = tl.program_id(1)
        d_offsets = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
        d_in_range = d_offsets < D

        for i in tl.static_range(0, BLOCK_B):
            batch_idx = b_group * BLOCK_B + i
            valid_lane = batch_idx < B

            slot_id = tl.load(
                slot_ids_ptr + batch_idx,
                mask=valid_lane,
                other=-1,
            ).to(tl.int64)

            shift_row = attn_shift_ptr + batch_idx * D + d_offsets
            store_mask = valid_lane & d_in_range & (slot_id >= 0)
            attn_src = tl.load(shift_row, mask=store_mask)
            tl.store(
                attn_cache_ptr + slot_id * D + d_offsets,
                attn_src,
                mask=store_mask,
            )

            ffn_src = tl.load(ffn_shift_ptr + batch_idx * D + d_offsets, mask=store_mask)
            tl.store(
                ffn_cache_ptr + slot_id * D + d_offsets,
                ffn_src,
                mask=store_mask,
            )


def _shift_state_cache_reference(
    attn_cache: torch.Tensor,
    ffn_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    attn_shift: torch.Tensor,
    ffn_shift: torch.Tensor,
) -> None:
    """CPU/reference equivalent of the fused kernel."""
    valid_mask = slot_ids >= 0
    if not torch.any(valid_mask):
        return
    valid_slot_ids = slot_ids[valid_mask].to(torch.long)
    attn_cache.index_copy_(
        0,
        valid_slot_ids,
        attn_shift[valid_mask].to(attn_cache.dtype),
    )
    ffn_cache.index_copy_(
        0,
        valid_slot_ids,
        ffn_shift[valid_mask].to(ffn_cache.dtype),
    )


def rwkv7_shift_state_cache(
    attn_cache: torch.Tensor,
    ffn_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    attn_shift: torch.Tensor,
    ffn_shift: torch.Tensor,
) -> None:
    """Fused update of ``kv_cache[0]`` (attn shift) and ``kv_cache[2]`` (ffn shift).

    Mirrors the existing ``index_copy_(0, slot_ids, src.to(cache.dtype))``
    semantics for valid slots and leaves the caches unchanged for any row
    where ``slot_id < 0``. Falls back to a reference path when Triton is
    unavailable, the disable flag is set, or the cache/source layout is
    incompatible with the fused kernel.
    """
    if not _can_use_triton(
        attn_cache,
        ffn_cache,
        slot_ids,
        attn_shift,
        ffn_shift,
    ):
        _shift_state_cache_reference(
            attn_cache,
            ffn_cache,
            slot_ids,
            attn_shift,
            ffn_shift,
        )
        return

    B = slot_ids.shape[0]
    D = attn_cache.shape[1]
    grid = (triton.cdiv(B, _BLOCK_B_SHIFT_STATE), triton.cdiv(D, _BLOCK_D_SHIFT_STATE))
    _rwkv7_shift_state_cache_fwd_grouped_kernel[grid](
        attn_cache,
        ffn_cache,
        slot_ids,
        attn_shift,
        ffn_shift,
        B,
        D,
        BLOCK_B=_BLOCK_B_SHIFT_STATE,
        BLOCK_D=_BLOCK_D_SHIFT_STATE,
        num_warps=4,
    )
