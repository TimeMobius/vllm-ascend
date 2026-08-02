# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused T=1 RWKV7 recurrent step + reduce Triton-Ascend kernel.

Replaces the PyTorch reference path used in ``forward_decode_batch`` for the
single-token decode regime. The upstream reference calls
``_rwkv7_recurrent_step`` followed by ``(state * r.unsqueeze(-1)).sum(-2)``,
both of which materialize intermediate tensors. This kernel performs:

  sa[h, v] = sum_d state[h, d, v] * (-kk[h, d])
  new_state[h, d, v] = exp(w[h, d]) * state[h, d, v]
                      + (kk[h, d] * a[h, d]) * sa[h, v]
                      + k[h, d] * v[h, v]
  reduce_out[h, v] = sum_d new_state[h, d, v] * r[h, d]

in a single launch with parallel work over ``[B, H_local, BLOCK_V]``.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend import envs
from vllm_ascend.profiler.rwkv7_counters import (
    DispatchKind,
    dispatch_hit,
)


def _rwkv7_recurrent_t1_reference(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference used as fallback and as the utest oracle."""
    sa = (recurrent_state * (-kk).unsqueeze(-1)).sum(dim=-2)
    new_state = (
        torch.exp(w).unsqueeze(-1) * recurrent_state
        + (kk * a).unsqueeze(-1) * sa.unsqueeze(-2)
        + k.unsqueeze(-1) * v.unsqueeze(-2)
    )
    reduce_out = (new_state * r.unsqueeze(-1)).sum(dim=-2)
    return new_state, reduce_out


if HAS_TRITON:

    @triton.jit
    def _rwkv7_recurrent_t1_fwd_kernel(
        state_ptr,
        w_ptr,
        kk_ptr,
        a_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        out_state_ptr,
        out_reduce_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One program per (batch, head, v_block) triple.

        All tensors are contiguous along the relevant axes:
          state:      [B, H, D, V] (row-major)
          w/kk/a/k/r: [B, H, D]    (row-major)
          v:          [B, H, V]    (row-major)
          out_state:  [B, H, D, V] (row-major)
          out_reduce: [B, H, V]    (row-major)
        """
        batch_idx = tl.program_id(0)
        h = tl.program_id(1)
        v_block_idx = tl.program_id(2)

        v_offsets = v_block_idx * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = v_offsets < V

        state_base = state_ptr + (batch_idx * H + h) * D * V
        out_state_base = out_state_ptr + (batch_idx * H + h) * D * V

        # Load [V] (length V) constant within this program
        v_vals = tl.load(
            v_ptr + (batch_idx * H + h) * V + v_offsets,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)
        sa_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        # sa[v] = sum_d state[h, d, v] * (-kk[h, d])
        for d in tl.static_range(0, D):
            kk_d = tl.load(kk_ptr + (batch_idx * H + h) * D + d).to(tl.float32)
            state_d = tl.load(
                state_base + d * V + v_offsets,
                mask=v_mask,
                other=0.0,
            ).to(tl.float32)
            sa_acc += state_d * (-kk_d)
        # sa_acc now holds sa[h, v_block] for this v_block

        # Loop over d again to write new_state and accumulate reduce
        reduce_acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for d in tl.static_range(0, D):
            w_d = tl.exp(tl.load(w_ptr + (batch_idx * H + h) * D + d).to(tl.float32))
            kk_d = tl.load(kk_ptr + (batch_idx * H + h) * D + d).to(tl.float32)
            a_d = tl.load(a_ptr + (batch_idx * H + h) * D + d).to(tl.float32)
            k_d = tl.load(k_ptr + (batch_idx * H + h) * D + d).to(tl.float32)
            r_d = tl.load(r_ptr + (batch_idx * H + h) * D + d).to(tl.float32)

            state_d = tl.load(
                state_base + d * V + v_offsets,
                mask=v_mask,
                other=0.0,
            ).to(tl.float32)
            ka = kk_d * a_d
            new_d = w_d * state_d + ka * sa_acc + k_d * v_vals
            tl.store(out_state_base + d * V + v_offsets, new_d, mask=v_mask)
            reduce_acc += new_d * r_d

        tl.store(
            out_reduce_ptr + (batch_idx * H + h) * V + v_offsets,
            reduce_acc,
            mask=v_mask,
        )


def rwkv7_recurrent_t1(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused T=1 recurrent step + reduce; returns (new_state, reduce_out).

    Falls back to the pure PyTorch reference when guards fail. Guards:
      - Triton available
      - device is NPU/CUDA
      - all inputs float32, contiguous, on the same device
      - shapes match [B, H, D, V], [B, H, D], [B, H, V], or their rank-3
        unbatched equivalents
      - head_dim and BLOCK_V fit within Triton power-of-two
    """
    if (
        not HAS_TRITON
        or envs.VLLM_ASCEND_RWKV7_DISABLE_TRITON
        or recurrent_state.device.type not in ("npu", "cuda")
        or recurrent_state.dtype != torch.float32
        or w.dtype != torch.float32
        or kk.dtype != torch.float32
        or a.dtype != torch.float32
        or k.dtype != torch.float32
        or v.dtype != torch.float32
        or r.dtype != torch.float32
        or not recurrent_state.is_contiguous()
        or not w.is_contiguous()
        or not kk.is_contiguous()
        or not a.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not r.is_contiguous()
    ):
        return _rwkv7_recurrent_t1_reference(
            recurrent_state, w, kk, a, k, v, r
        )

    if recurrent_state.ndim == 3:
        is_batched = False
        state_4d = recurrent_state.unsqueeze(0)
        w_3d, kk_3d, a_3d, k_3d, v_3d, r_3d = (
            tensor.unsqueeze(0) for tensor in (w, kk, a, k, v, r)
        )
    elif recurrent_state.ndim == 4:
        is_batched = True
        state_4d = recurrent_state
        w_3d, kk_3d, a_3d, k_3d, v_3d, r_3d = w, kk, a, k, v, r
    else:
        return _rwkv7_recurrent_t1_reference(
            recurrent_state, w, kk, a, k, v, r
        )

    B, H, D, V = state_4d.shape
    if (
        w_3d.shape != (B, H, D)
        or kk_3d.shape != (B, H, D)
        or a_3d.shape != (B, H, D)
        or k_3d.shape != (B, H, D)
        or v_3d.shape != (B, H, V)
        or r_3d.shape != (B, H, D)
    ):
        return _rwkv7_recurrent_t1_reference(
            recurrent_state, w, kk, a, k, v, r
        )

    BLOCK_V = triton.next_power_of_2(V)
    if D > 256 or BLOCK_V > 256:
        return _rwkv7_recurrent_t1_reference(
            recurrent_state, w, kk, a, k, v, r
        )

    new_state = torch.empty_like(state_4d)
    reduce_out = torch.empty(
        (B, H, V), device=recurrent_state.device, dtype=torch.float32
    )

    grid = (B, H, triton.cdiv(V, BLOCK_V))
    _rwkv7_recurrent_t1_fwd_kernel[grid](
        state_4d,
        w_3d,
        kk_3d,
        a_3d,
        k_3d,
        v_3d,
        r_3d,
        new_state,
        reduce_out,
        H=H,
        D=D,
        V=V,
        BLOCK_V=BLOCK_V,
        num_warps=4 if BLOCK_V <= 64 else 8,
    )
    dispatch_hit(DispatchKind.RECURRENT_T1)
    if is_batched:
        return new_state, reduce_out
    return new_state.squeeze(0), reduce_out.squeeze(0)
