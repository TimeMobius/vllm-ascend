# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch
import triton

import vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 as recurrent_t1
import vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1_cache as recurrent_t1_cache
from vllm_ascend.ops.triton.fla import rwkv7_recurrent_t1_matrix as matrix_mod


def _make_t1_inputs(batch_size, *, heads=2, head_dim=8, value_dim=8, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(17)
    state = torch.randn(batch_size, heads, head_dim, value_dim, generator=generator, device="cpu", dtype=dtype)
    w = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    kk = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    a = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    k = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    value = torch.randn(batch_size, heads, value_dim, generator=generator, device="cpu", dtype=dtype)
    r = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    return tuple(tensor.to(device) for tensor in (state, w, kk, a, k, value, r))


def _make_cache_inputs(
    batch_size, *, cache_slots=310, heads=64, head_dim=64, value_dim=64, device="cpu", dtype=torch.float32
):
    generator = torch.Generator(device="cpu").manual_seed(31)
    cache = torch.randn(cache_slots, heads, head_dim, value_dim, generator=generator, device="cpu", dtype=dtype)
    w = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    kk = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    a = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    k = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    value = torch.randn(batch_size, heads, value_dim, generator=generator, device="cpu", dtype=dtype)
    r = torch.randn(batch_size, heads, head_dim, generator=generator, device="cpu", dtype=dtype)
    return tuple(tensor.to(device) for tensor in (cache, w, kk, a, k, value, r))


def test_cache_wrapper_matches_gather_recurrence_scatter_reference():
    cache, w, kk, a, k, value, r = _make_t1_inputs(7)
    slot_ids = torch.tensor([6, 1, 4, 0], dtype=torch.long)
    expected_cache = cache.clone()
    gathered_state = expected_cache.index_select(0, slot_ids)
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        gathered_state, w[:4], kk[:4], a[:4], k[:4], value[:4], r[:4]
    )
    expected_cache.index_copy_(0, slot_ids, expected_state)
    actual_cache = cache.clone()
    actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(
        actual_cache, slot_ids, w[:4], kk[:4], a[:4], k[:4], value[:4], r[:4]
    )
    torch.testing.assert_close(actual_cache, expected_cache)
    torch.testing.assert_close(actual_output, expected_output)


def test_cache_wrapper_leaves_padding_slot_and_cache_unchanged():
    cache, w, kk, a, k, value, r = _make_t1_inputs(4)
    slot_ids = torch.tensor([3, -1, 1], dtype=torch.long)
    expected_cache = cache.clone()
    gathered_state = expected_cache.index_select(0, torch.tensor([3, 1]))
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        gathered_state, w[[0, 2]], kk[[0, 2]], a[[0, 2]], k[[0, 2]], value[[0, 2]], r[[0, 2]]
    )
    expected_cache.index_copy_(0, torch.tensor([3, 1]), expected_state)
    actual_cache = cache.clone()
    actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(
        actual_cache, slot_ids, w[:3], kk[:3], a[:3], k[:3], value[:3], r[:3]
    )
    torch.testing.assert_close(actual_cache, expected_cache)
    torch.testing.assert_close(actual_output[[0, 2]], expected_output)
    torch.testing.assert_close(actual_output[1], torch.zeros_like(actual_output[1]))


def test_npu_cache_wrapper_matches_gather_recurrence_scatter_reference():
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")
    H, D, V = 64, 64, 64
    generator = torch.Generator(device="npu").manual_seed(17)
    cache = torch.randn(12, H, D, V, generator=generator, device="npu", dtype=torch.float32)
    w = torch.randn(12, H, D, generator=generator, device="npu", dtype=torch.float32)
    kk = torch.randn(12, H, D, generator=generator, device="npu", dtype=torch.float32)
    a = torch.randn(12, H, D, generator=generator, device="npu", dtype=torch.float32)
    k = torch.randn(12, H, D, generator=generator, device="npu", dtype=torch.float32)
    value = torch.randn(12, H, V, generator=generator, device="npu", dtype=torch.float32)
    r = torch.randn(12, H, D, generator=generator, device="npu", dtype=torch.float32)
    slot_ids = torch.tensor([11, 1, 9, 4, 7, 0, 5, 2], device="npu")
    expected_cache = cache.clone()
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        expected_cache.index_select(0, slot_ids), w[:8], kk[:8], a[:8], k[:8], value[:8], r[:8]
    )
    expected_cache.index_copy_(0, slot_ids, expected_state)
    actual_cache = cache.clone()
    actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(
        actual_cache, slot_ids, w[:8], kk[:8], a[:8], k[:8], value[:8], r[:8]
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_cache, expected_cache, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_output, expected_output, atol=2e-4, rtol=2e-4)


def test_npu_cache_wrapper_supports_c128_production_shape():
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")
    H, D, V = 64, 64, 64
    generator = torch.Generator(device="npu").manual_seed(17)
    w = torch.randn(128, H, D, generator=generator, device="npu", dtype=torch.float32)
    kk = torch.randn(128, H, D, generator=generator, device="npu", dtype=torch.float32)
    a = torch.randn(128, H, D, generator=generator, device="npu", dtype=torch.float32)
    k = torch.randn(128, H, D, generator=generator, device="npu", dtype=torch.float32)
    value = torch.randn(128, H, V, generator=generator, device="npu", dtype=torch.float32)
    r = torch.randn(128, H, D, generator=generator, device="npu", dtype=torch.float32)
    recurrent_cache = torch.randn(310, 64, 64, 64, device="npu", dtype=torch.float32)
    slot_ids = torch.randperm(310, device="npu")[:128].to(torch.long)
    expected_cache = recurrent_cache.clone()
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        expected_cache.index_select(0, slot_ids), w, kk, a, k, value, r
    )
    expected_cache.index_copy_(0, slot_ids, expected_state)
    actual_cache = recurrent_cache.clone()
    actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(actual_cache, slot_ids, w, kk, a, k, value, r)
    torch.npu.synchronize()
    torch.testing.assert_close(actual_cache, expected_cache, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_output, expected_output, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 64, 128])
def test_cache_wrapper_dispatches_grouped_kernel_with_cdiv_grid(batch_size):
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")
    cache, w, kk, a, k, value, r = _make_cache_inputs(batch_size, device="npu")
    slot_ids = torch.arange(batch_size, dtype=torch.long, device="npu")
    expected_grid = (triton.cdiv(batch_size, matrix_mod.BLOCK_B_CACHE), 64)
    calls = []
    real_grouped = matrix_mod.rwkv7_recurrent_t1_cache_fwd_grouped_kernel

    class GroupedKernelSpy:
        def __getitem__(self, grid):
            calls.append(tuple(grid))
            return real_grouped[grid]

    with (
        mock.patch.object(matrix_mod, "rwkv7_recurrent_t1_cache_fwd_grouped_kernel", GroupedKernelSpy()),
        mock.patch.object(
            recurrent_t1,
            "_rwkv7_recurrent_t1_reference",
            side_effect=AssertionError("valid NPU cache input used Torch fallback"),
        ),
    ):
        recurrent_t1.rwkv7_recurrent_t1_cache(cache, slot_ids, w, kk, a, k, value, r)
        torch.npu.synchronize()
    assert calls == [expected_grid], f"expected grouped kernel grid {expected_grid} for B={batch_size}, got {calls}"


def test_cache_wrapper_handles_non_divisible_b_and_padding():
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")
    batch_size = 7
    cache, w, kk, a, k, value, r = _make_cache_inputs(batch_size, cache_slots=64, device="npu")
    slot_ids = torch.tensor([3, -1, 1, -1, 0, 5, 2], dtype=torch.long, device="npu")
    expected_grid = (triton.cdiv(batch_size, matrix_mod.BLOCK_B_CACHE), 64)
    calls = []
    real_grouped = matrix_mod.rwkv7_recurrent_t1_cache_fwd_grouped_kernel

    class GroupedKernelSpy:
        def __getitem__(self, grid):
            calls.append(tuple(grid))
            return real_grouped[grid]

    expected_cache = cache.clone()
    valid_slot_ids = slot_ids[slot_ids >= 0]
    expected_state, valid_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        expected_cache.index_select(0, valid_slot_ids),
        w[slot_ids >= 0],
        kk[slot_ids >= 0],
        a[slot_ids >= 0],
        k[slot_ids >= 0],
        value[slot_ids >= 0],
        r[slot_ids >= 0],
    )
    expected_cache.index_copy_(0, valid_slot_ids, expected_state)
    expected_output_full = torch.zeros(
        (batch_size, value.shape[1], value.shape[2]), device=value.device, dtype=torch.float32
    )
    expected_output_full[slot_ids >= 0] = valid_output
    with mock.patch.object(matrix_mod, "rwkv7_recurrent_t1_cache_fwd_grouped_kernel", GroupedKernelSpy()):
        actual_cache = cache.clone()
        actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(actual_cache, slot_ids, w, kk, a, k, value, r)
        torch.npu.synchronize()
    assert calls == [expected_grid], f"non-divisible B={batch_size} expected grouped grid {expected_grid}, got {calls}"
    torch.testing.assert_close(actual_cache, expected_cache, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_output, expected_output_full, atol=2e-4, rtol=2e-4)
    padding_positions = (slot_ids < 0).nonzero(as_tuple=True)[0]
    torch.testing.assert_close(actual_output[padding_positions], torch.zeros_like(actual_output[padding_positions]))


def test_cache_wrapper_b128_grouped_grid_matches_cdiv():
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")
    batch_size = 128
    cache, w, kk, a, k, value, r = _make_cache_inputs(batch_size, cache_slots=310, device="npu")
    slot_ids = torch.randperm(310, device="npu")[:batch_size].to(torch.long)
    expected_grid = (triton.cdiv(batch_size, matrix_mod.BLOCK_B_CACHE), 64)
    calls = []
    real_grouped = matrix_mod.rwkv7_recurrent_t1_cache_fwd_grouped_kernel

    class GroupedKernelSpy:
        def __getitem__(self, grid):
            calls.append(tuple(grid))
            return real_grouped[grid]

    with mock.patch.object(matrix_mod, "rwkv7_recurrent_t1_cache_fwd_grouped_kernel", GroupedKernelSpy()):
        recurrent_t1.rwkv7_recurrent_t1_cache(cache, slot_ids, w, kk, a, k, value, r)
        torch.npu.synchronize()
    assert calls == [expected_grid], f"B=128 production cache shape expected grouped grid {expected_grid}, got {calls}"


def test_cpu_fallback_matches_gather_reference_scatter():
    cache, w, kk, a, k, value, r = _make_t1_inputs(7)
    slot_ids = torch.tensor([6, 1, 4, 0], dtype=torch.long)
    expected_cache = cache.clone()
    gathered_state = expected_cache.index_select(0, slot_ids)
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(
        gathered_state, w[:4], kk[:4], a[:4], k[:4], value[:4], r[:4]
    )
    expected_cache.index_copy_(0, slot_ids, expected_state)
    actual_cache = cache.clone()
    with (
        mock.patch.object(recurrent_t1, "HAS_TRITON", False),
        mock.patch.object(recurrent_t1_cache, "HAS_TRITON", False),
    ):
        actual_output = recurrent_t1.rwkv7_recurrent_t1_cache(
            actual_cache, slot_ids, w[:4], kk[:4], a[:4], k[:4], value[:4], r[:4]
        )
    torch.testing.assert_close(actual_cache, expected_cache)
    torch.testing.assert_close(actual_output, expected_output)
