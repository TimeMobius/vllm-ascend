# SPDX-License-Identifier: Apache-2.0
"""Token-head tiled implementation used by the RWKV7 ``kk_pre`` operator."""

from __future__ import annotations

from typing import Final

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

TOKEN_HEAD_LOCAL_HEADS: Final = (64, 32, 16, 8)
PROFILE_HEAD_DIM: Final = 64
MEASURED_FLOOR: Final = 4
TOKEN_HEAD_TB64: Final = 64
TOKEN_HEAD_TB32: Final = 32
TB32_LOCAL_HEADS: Final = 8
TB32_TOKEN_FLOOR: Final = 131072
MAX_GRID: Final = 65535


def select_token_head_tile(num_tokens: int, num_heads: int, head_dim: int) -> int | None:
    """Return a measured token tile, or ``None`` for the legacy path."""
    if head_dim != PROFILE_HEAD_DIM:
        return None
    if num_heads not in TOKEN_HEAD_LOCAL_HEADS:
        return None
    if num_tokens < MEASURED_FLOOR:
        return None
    if num_heads == TB32_LOCAL_HEADS and num_tokens >= TB32_TOKEN_FLOOR:
        return TOKEN_HEAD_TB32
    return TOKEN_HEAD_TB64


def token_head_grid(num_tokens: int, num_heads: int, token_block: int) -> int:
    """Bound the token-head launch grid by available vector cores."""
    num_token_blocks = (num_tokens + token_block - 1) // token_block
    return min(num_heads * num_token_blocks, get_vectorcore_num(), MAX_GRID)


@triton.jit
def rwkv7_kk_pre_token_head_2d_kernel(
    k,
    a,
    k_k,
    k_a,
    k_out,
    kk_out,
    num_tokens,
    num_heads,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    """Process one fixed head and a vectorized token block."""
    pid = tl.program_id(0).to(tl.int64)
    grid = tl.num_programs(0)
    num_token_blocks = tl.cdiv(num_tokens, TOKEN_BLOCK)
    num_work_items = num_heads * num_token_blocks
    token_lanes = tl.arange(0, TOKEN_BLOCK)
    offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = offsets < head_dim
    for work_id in tl.range(pid, num_work_items, grid):
        head_idx = work_id % num_heads
        token_start = (work_id // num_heads) * TOKEN_BLOCK
        tokens = token_start + token_lanes
        token_mask = tokens < num_tokens
        mask = token_mask[:, None] & dim_mask[None, :]
        head_offset = head_idx * head_dim
        rows = tokens * num_heads + head_idx
        row_offsets = rows[:, None] * head_dim + offsets[None, :]

        k_k_vals = tl.load(k_k + head_offset + offsets, mask=dim_mask, other=0).to(tl.float32)
        k_a_vals = tl.load(k_a + head_offset + offsets, mask=dim_mask, other=0).to(tl.float32)
        k_vals = tl.load(k + row_offsets, mask=mask, other=0).to(tl.float32)
        a_vals = tl.load(a + row_offsets, mask=mask, other=0).to(tl.float32)

        kk_raw = k_vals * k_k_vals[None, :]
        rstd = tl.rsqrt(tl.sum(kk_raw * kk_raw, axis=1) + eps)
        kk_vals = kk_raw * rstd[:, None]
        k_adj = k_vals * (1.0 + (a_vals - 1.0) * k_a_vals[None, :])

        tl.store(k_out + row_offsets, k_adj, mask=mask)
        tl.store(kk_out + row_offsets, kk_vals, mask=mask)
