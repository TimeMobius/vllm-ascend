# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend import envs

__all__ = ["rwkv7_recurrent_t1_cache"]


def rwkv7_recurrent_t1_cache(
    recurrent_cache: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Update 64x64 FP32 states in their cache rows and return recurrent output."""

    def reference() -> torch.Tensor:
        # Route through the non-cache T1 entrypoint, which owns its own
        # reference fallback, so a persistent-cache guard failure degrades to
        # non-cache T1 -> PyTorch reference per the documented backend chain.
        from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 import (
            rwkv7_recurrent_t1,
        )

        valid_slots = slot_ids >= 0
        reduce_out = torch.zeros(
            (slot_ids.shape[0], recurrent_cache.shape[1], recurrent_cache.shape[-1]),
            device=recurrent_cache.device,
            dtype=torch.float32,
        )
        if not torch.any(valid_slots):
            return reduce_out
        valid_slot_ids = slot_ids[valid_slots]
        state = recurrent_cache.index_select(0, valid_slot_ids)
        new_state, valid_output = rwkv7_recurrent_t1(
            state,
            w[valid_slots],
            kk[valid_slots],
            a[valid_slots],
            k[valid_slots],
            v[valid_slots],
            r[valid_slots],
        )
        recurrent_cache.index_copy_(0, valid_slot_ids, new_state)
        reduce_out[valid_slots] = valid_output
        return reduce_out

    if (
        not HAS_TRITON
        or envs.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or recurrent_cache.device.type not in ("npu", "cuda")
        or recurrent_cache.dtype != torch.float32
        or slot_ids.device != recurrent_cache.device
        or slot_ids.dtype != torch.long
        or not recurrent_cache.is_contiguous()
        or not slot_ids.is_contiguous()
    ):
        return reference()

    if recurrent_cache.ndim != 4:
        raise ValueError("rwkv7_recurrent_t1_cache requires cache shape [S, H, D, V].")

    _, H, D, V = recurrent_cache.shape
    B = slot_ids.shape[0]
    if (
        D != 64
        or V != 64
        or w.shape != (B, H, D)
        or kk.shape != (B, H, D)
        or a.shape != (B, H, D)
        or k.shape != (B, H, D)
        or v.shape != (B, H, V)
        or r.shape != (B, H, D)
        or any(tensor.dtype != torch.float32 or not tensor.is_contiguous() for tensor in (w, kk, a, k, v, r))
    ):
        return reference()

    from vllm_ascend.ops.triton.fla import rwkv7_recurrent_t1_matrix as _matrix_mod

    BLOCK_B_CACHE = _matrix_mod.BLOCK_B_CACHE
    reduce_out = torch.empty((B, H, V), device=recurrent_cache.device, dtype=torch.float32)
    grid = (triton.cdiv(B, BLOCK_B_CACHE), H)
    _matrix_mod.rwkv7_recurrent_t1_cache_fwd_grouped_kernel[grid](
        recurrent_cache,
        slot_ids,
        w,
        kk,
        a,
        k,
        v,
        r,
        reduce_out,
        B,
        H=H,
        D=D,
        V=V,
        BLOCK_B=BLOCK_B_CACHE,
        num_warps=4,
    )
    return reduce_out
