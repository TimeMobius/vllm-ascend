# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch
import triton

from vllm_ascend.ops.triton.fla import rwkv7_shift_state_cache as shift_state_cache_mod
from vllm_ascend.ops.triton.fla.rwkv7_shift_state_cache import (
    _BLOCK_B_SHIFT_STATE,
    _BLOCK_D_SHIFT_STATE,
    HAS_TRITON,
    rwkv7_shift_state_cache,
)


def test_shift_state_cache_updates_two_states_and_preserves_padding() -> None:
    attn_cache = torch.zeros(5, 4)
    ffn_cache = torch.ones(5, 4)
    attn_shift = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ffn_shift = -attn_shift
    slot_ids = torch.tensor([3, -1, 1], dtype=torch.long)

    rwkv7_shift_state_cache(
        attn_cache,
        ffn_cache,
        slot_ids,
        attn_shift,
        ffn_shift,
    )

    expected_attn = torch.zeros_like(attn_cache)
    expected_attn[3] = attn_shift[0]
    expected_attn[1] = attn_shift[2]
    expected_ffn = torch.ones_like(ffn_cache)
    expected_ffn[3] = ffn_shift[0]
    expected_ffn[1] = ffn_shift[2]
    torch.testing.assert_close(attn_cache, expected_attn)
    torch.testing.assert_close(ffn_cache, expected_ffn)


def test_npu_shift_state_cache_d4096_matches_reference_and_dispatches_single_kernel() -> None:
    """Production D=4096 must dispatch the fused kernel in one launch.

    Verifies three contracts on the NPU path:
      * the Triton path is taken (no reference fallback),
      * exactly one launch is issued with the 2D tiled grid
        ``(cdiv(B, BLOCK_B), cdiv(D, BLOCK_D))``,
      * both cache tensors match the reference scatter for valid slots and
        remain unchanged for padding (``slot_id < 0``).
    """
    if not torch.npu.is_available() or not HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")

    cache_slots = 16
    batch_size = 8
    feature_dim = 4096  # production RWKV7 shift-state width
    generator = torch.Generator(device="npu").manual_seed(91)

    attn_cache = torch.randn(cache_slots, feature_dim, generator=generator, device="npu", dtype=torch.bfloat16)
    ffn_cache = torch.randn(cache_slots, feature_dim, generator=generator, device="npu", dtype=torch.bfloat16)
    attn_shift = torch.randn(batch_size, feature_dim, generator=generator, device="npu", dtype=torch.bfloat16)
    ffn_shift = torch.randn(batch_size, feature_dim, generator=generator, device="npu", dtype=torch.bfloat16)
    slot_ids = torch.tensor([3, -1, 7, 0, -1, 12, 5, 1], dtype=torch.long, device="npu")

    expected_attn = attn_cache.clone()
    expected_ffn = ffn_cache.clone()
    valid_mask = slot_ids >= 0
    expected_attn.index_copy_(0, slot_ids[valid_mask], attn_shift[valid_mask].to(expected_attn.dtype))
    expected_ffn.index_copy_(0, slot_ids[valid_mask], ffn_shift[valid_mask].to(expected_ffn.dtype))

    expected_grid = (
        triton.cdiv(batch_size, _BLOCK_B_SHIFT_STATE),
        triton.cdiv(feature_dim, _BLOCK_D_SHIFT_STATE),
    )
    calls = []
    real_kernel = shift_state_cache_mod._rwkv7_shift_state_cache_fwd_grouped_kernel

    class ShiftStateCacheSpy:
        def __getitem__(self, grid):
            calls.append(tuple(grid))
            return real_kernel[grid]

    with mock.patch.object(
        shift_state_cache_mod,
        "_rwkv7_shift_state_cache_fwd_grouped_kernel",
        ShiftStateCacheSpy(),
    ):
        rwkv7_shift_state_cache(attn_cache, ffn_cache, slot_ids, attn_shift, ffn_shift)
        torch.npu.synchronize()

    assert calls == [expected_grid], f"D={feature_dim} expected single tiled launch {expected_grid}, got {calls}"
    torch.testing.assert_close(attn_cache, expected_attn, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ffn_cache, expected_ffn, atol=2e-2, rtol=2e-2)
