# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch

import vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 as recurrent_t1


def _make_t1_inputs(
    batch_size: int,
    *,
    heads: int = 2,
    head_dim: int = 8,
    value_dim: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(17)
    state = torch.randn(
        batch_size,
        heads,
        head_dim,
        value_dim,
        generator=generator,
        device="cpu",
        dtype=dtype,
    )
    recurrent = tuple(
        torch.randn(
            batch_size,
            heads,
            head_dim,
            generator=generator,
            device="cpu",
            dtype=dtype,
        )
        for _ in range(5)
    )
    value = torch.randn(
        batch_size,
        heads,
        value_dim,
        generator=generator,
        device="cpu",
        dtype=dtype,
    )
    return tuple(tensor.to(device) for tensor in (state, *recurrent[:3], recurrent[3], value, recurrent[4]))


def test_rank4_cpu_wrapper_matches_reference() -> None:
    state, w, kk, a, k, value, r = _make_t1_inputs(4)

    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(state, w, kk, a, k, value, r)
    actual_state, actual_output = recurrent_t1.rwkv7_recurrent_t1(state, w, kk, a, k, value, r)

    torch.testing.assert_close(actual_state, expected_state)
    torch.testing.assert_close(actual_output, expected_output)
    assert actual_state.shape == state.shape
    assert actual_output.shape == (4, 2, 8)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 48, 64, 96, 128])
def test_npu_rank4_batch_launch_does_not_fallback(batch_size: int) -> None:
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")

    calls: list[tuple[int, ...]] = []
    real_kernel = recurrent_t1._rwkv7_recurrent_t1_fwd_kernel

    class KernelSpy:
        def __getitem__(self, grid: tuple[int, ...]):
            calls.append(tuple(grid))
            return real_kernel[grid]

    inputs = _make_t1_inputs(batch_size, device="npu")
    with (
        mock.patch.object(
            recurrent_t1,
            "_rwkv7_recurrent_t1_fwd_kernel",
            KernelSpy(),
        ),
        mock.patch.object(
            recurrent_t1,
            "_rwkv7_recurrent_t1_reference",
            side_effect=AssertionError("valid NPU T1 input used Torch fallback"),
        ),
    ):
        state, *_ = inputs
        actual_state, actual_output = recurrent_t1.rwkv7_recurrent_t1(*inputs)
        torch.npu.synchronize()

    assert calls == [(batch_size, 2, 1)]
    assert actual_state.shape == state.shape
    assert actual_output.shape == (batch_size, 2, 8)


def test_npu_rank4_parity_with_reference() -> None:
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")

    inputs = _make_t1_inputs(4, device="npu")
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(*inputs)
    actual_state, actual_output = recurrent_t1.rwkv7_recurrent_t1(*inputs)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_state, expected_state, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_output, expected_output, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_npu_t1_fp32_matches_reference_through_batch_64(batch_size: int) -> None:
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")

    inputs = _make_t1_inputs(
        batch_size,
        heads=32,
        head_dim=64,
        value_dim=64,
        device="npu",
    )
    expected_state, expected_output = recurrent_t1._rwkv7_recurrent_t1_reference(*inputs)
    actual_state, actual_output = recurrent_t1.rwkv7_recurrent_t1(*inputs)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_state, expected_state, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_output, expected_output, atol=2e-4, rtol=2e-4)


def _timed_t1_latency_us(step_fn, inputs: tuple[torch.Tensor, ...]) -> float:
    state, *projections = inputs
    warmup_state = state.clone()
    for _ in range(20):
        warmup_state, _ = step_fn(warmup_state, *projections)
    torch.npu.synchronize()

    timed_state = state.clone()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        timed_state, _ = step_fn(timed_state, *projections)
    end_event.record()
    torch.npu.synchronize()
    return float(start_event.elapsed_time(end_event)) * 10


def test_npu_t1_batch_32_outperforms_torch_reference() -> None:
    if not torch.npu.is_available() or not recurrent_t1.HAS_TRITON:
        pytest.skip("requires an NPU with Triton-Ascend")

    inputs = _make_t1_inputs(
        32,
        heads=32,
        head_dim=64,
        value_dim=64,
        device="npu",
    )
    torch_reference_latency_us = _timed_t1_latency_us(recurrent_t1._rwkv7_recurrent_t1_reference, inputs)
    triton_latency_us = _timed_t1_latency_us(recurrent_t1.rwkv7_recurrent_t1, inputs)

    assert triton_latency_us < torch_reference_latency_us, (
        f"Triton T1 {triton_latency_us:.2f}us must beat the Torch reference {torch_reference_latency_us:.2f}us at B=32"
    )
