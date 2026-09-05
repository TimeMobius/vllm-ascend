# SPDX-License-Identifier: Apache-2.0
"""Tests for the production RWKV7 token-head ``kk_pre`` path."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import (
    rwkv7_kk_pre,
    rwkv7_kk_pre_reference,
)
from vllm_ascend.ops.triton.fla.rwkv7_kk_pre_token_head import (
    select_token_head_tile,
)

LOCAL_HEADS = (64, 32, 16, 8)


@pytest.mark.parametrize("heads", LOCAL_HEADS)
@pytest.mark.parametrize("tokens", (1024, 2048, 32768))
def test_selector_uses_tb64_for_measured_long_context(tokens: int, heads: int) -> None:
    assert select_token_head_tile(tokens, heads, 64) == 64


@pytest.mark.parametrize("heads", LOCAL_HEADS)
def test_selector_keeps_legacy_for_short_context(heads: int) -> None:
    assert select_token_head_tile(3, heads, 64) is None


@pytest.mark.parametrize("heads", LOCAL_HEADS)
def test_selector_uses_new_kernel_from_four_tokens(heads: int) -> None:
    assert select_token_head_tile(4, heads, 64) == 64


@pytest.mark.parametrize(
    ("heads", "tokens", "expected"),
    ((8, 32768, 64), (8, 131072, 32), (16, 131072, 64)),
)
def test_selector_handles_boundary(heads: int, tokens: int, expected: int) -> None:
    assert select_token_head_tile(tokens, heads, 64) == expected


@pytest.mark.parametrize("shape", ((1024, 12, 64), (1024, 64, 128)))
def test_selector_rejects_uncalibrated_shapes(shape: tuple[int, int, int]) -> None:
    assert select_token_head_tile(*shape) is None


def _inputs(tokens: int, heads: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="npu")
    generator.manual_seed(tokens + heads + head_dim)
    return (
        torch.randn(tokens, heads, head_dim, device="npu", generator=generator),
        torch.randn(tokens, heads, head_dim, device="npu", generator=generator),
        torch.randn(heads, head_dim, device="npu", generator=generator),
        torch.randn(heads, head_dim, device="npu", generator=generator),
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
@pytest.mark.parametrize("heads", LOCAL_HEADS)
@pytest.mark.parametrize("tokens", (4, 1024, 2048, 32768))
def test_token_head_path_matches_reference(tokens: int, heads: int) -> None:
    k, a, k_k, k_a = _inputs(tokens, heads, 64)
    expected = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
    actual = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)
    torch.npu.synchronize()
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_token_head_partial_block_matches_reference() -> None:
    k, a, k_k, k_a = _inputs(1025, 8, 64)
    expected = rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)
    actual = rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)
    torch.npu.synchronize()
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, atol=2e-4, rtol=2e-4)
