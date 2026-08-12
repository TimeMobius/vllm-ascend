# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matrix-vectorized FP32 RWKV7 T=1 kernel for the common 64x64 state."""

from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def rwkv7_recurrent_t1_matrix_fwd_kernel(
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
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        state_idx = batch_idx * H + head_idx
        d_offsets = tl.arange(0, D)[:, None]
        v_offsets = tl.arange(0, V)
        state_base = state_ptr + state_idx * D * V
        scalar_base = state_idx * D
        value_base = state_idx * V

        state = tl.load(state_base + d_offsets * V + v_offsets[None, :]).to(
            tl.float32
        )
        kk = tl.load(kk_ptr + scalar_base + d_offsets).to(tl.float32)
        sa = tl.sum(state * (-kk), axis=0)
        w = tl.exp(tl.load(w_ptr + scalar_base + d_offsets).to(tl.float32))
        a = tl.load(a_ptr + scalar_base + d_offsets).to(tl.float32)
        k = tl.load(k_ptr + scalar_base + d_offsets).to(tl.float32)
        r = tl.load(r_ptr + scalar_base + d_offsets).to(tl.float32)
        value = tl.load(v_ptr + value_base + v_offsets)[None, :].to(tl.float32)
        new_state = w * state + (kk * a) * sa + k * value

        tl.store(
            out_state_ptr + state_idx * D * V + d_offsets * V + v_offsets[None, :],
            new_state,
        )
        tl.store(
            out_reduce_ptr + value_base + v_offsets,
            tl.sum(new_state * r, axis=0),
        )


    @triton.jit
    def rwkv7_recurrent_t1_cache_fwd_kernel(
        cache_ptr,
        slot_ids_ptr,
        w_ptr,
        kk_ptr,
        a_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        out_reduce_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        slot_id = tl.load(slot_ids_ptr + batch_idx).to(tl.int64)
        d_offsets = tl.arange(0, D)[:, None]
        v_offsets = tl.arange(0, V)
        scalar_base = (batch_idx * H + head_idx) * D
        value_base = (batch_idx * H + head_idx) * V
        out_ptr = out_reduce_ptr + value_base + v_offsets

        if slot_id < 0:
            tl.store(out_ptr, 0.0)
            return

        cache_base = cache_ptr + (slot_id * H + head_idx) * D * V
        state = tl.load(cache_base + d_offsets * V + v_offsets[None, :]).to(
            tl.float32
        )
        kk = tl.load(kk_ptr + scalar_base + d_offsets).to(tl.float32)
        sa = tl.sum(state * (-kk), axis=0)
        w = tl.exp(tl.load(w_ptr + scalar_base + d_offsets).to(tl.float32))
        a = tl.load(a_ptr + scalar_base + d_offsets).to(tl.float32)
        k = tl.load(k_ptr + scalar_base + d_offsets).to(tl.float32)
        r = tl.load(r_ptr + scalar_base + d_offsets).to(tl.float32)
        value = tl.load(v_ptr + value_base + v_offsets)[None, :].to(tl.float32)
        new_state = w * state + (kk * a) * sa + k * value

        tl.store(cache_base + d_offsets * V + v_offsets[None, :], new_state)
        tl.store(out_ptr, tl.sum(new_state * r, axis=0))
