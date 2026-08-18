# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matrix-vectorized FP32 RWKV7 T=1 kernel for the common 64x64 state."""

from vllm.triton_utils import HAS_TRITON, tl, triton

# Rows per program for the grouped cache kernel; shrinks the launch grid from
# (B, H) to (ceil(B / BLOCK_B_CACHE), H) at BLOCK_B_CACHE rows per program.
BLOCK_B_CACHE: int = 4

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

        state = tl.load(state_base + d_offsets * V + v_offsets[None, :]).to(tl.float32)
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
        state = tl.load(cache_base + d_offsets * V + v_offsets[None, :]).to(tl.float32)
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

    @triton.jit
    def rwkv7_recurrent_t1_cache_fwd_grouped_kernel(
        cache_ptr,
        slot_ids_ptr,
        w_ptr,
        kk_ptr,
        a_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        out_reduce_ptr,
        B,
        H: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        """Cache kernel that handles BLOCK_B batch rows per program.

        Mirrors ``rwkv7_recurrent_t1_cache_fwd_kernel`` per row, with the
        same FP32 casts and arithmetic. The grid is
        ``(ceil(B / BLOCK_B), H)``. Lanes with ``batch_idx >= B`` (tail when
        B is not a multiple of BLOCK_B) and padding lanes (``slot_id < 0``)
        write zero to the output and leave the cache row untouched.
        """
        b_group = tl.program_id(0)
        head_idx = tl.program_id(1)
        d_offsets = tl.arange(0, D)[:, None]
        v_offsets = tl.arange(0, V)

        for i in tl.static_range(0, BLOCK_B):
            batch_idx = b_group * BLOCK_B + i
            valid_lane = batch_idx < B

            slot_id = tl.load(slot_ids_ptr + batch_idx, mask=valid_lane, other=-1).to(tl.int64)

            scalar_base = (batch_idx * H + head_idx) * D
            value_base = (batch_idx * H + head_idx) * V
            out_ptr = out_reduce_ptr + value_base + v_offsets

            if slot_id < 0:
                tl.store(
                    out_ptr,
                    tl.zeros((V,), dtype=tl.float32),
                    mask=valid_lane,
                )
            else:
                cache_base = cache_ptr + (slot_id * H + head_idx) * D * V
                state = tl.load(cache_base + d_offsets * V + v_offsets[None, :]).to(tl.float32)
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
