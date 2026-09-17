# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``rwkv7_triton_warmup``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.model_executor.warmup import rwkv7_triton_warmup as rwt


def _make_rwkv7_worker(
    *,
    hidden_size: int = 2048,
    num_heads: int = 32,
    head_dim: int = 64,
    value_dim: int = 2048,
    tp_size: int = 4,
    max_num_batched_tokens: int = 64,
    max_num_seqs: int = 4,
    architectures: tuple[str, ...] = ("RWKV7ForCausalLM",),
    model_type: str = "rwkv7",
) -> tuple[MagicMock, torch.Tensor]:
    """Build a fake RWKV7 worker plus a sentinel for the live state cache."""
    worker = MagicMock()
    worker.device = torch.device("cpu")

    hf_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        value_dim=[value_dim],
        norm_eps=1e-5,
        model_type=model_type,
    )
    model_config = MagicMock()
    model_config.hf_text_config = hf_config
    model_config.architectures = list(architectures)
    model_config.dtype = torch.float16
    worker.vllm_config.model_config = model_config
    worker.vllm_config.parallel_config.tensor_parallel_size = tp_size
    worker.model_config = model_config

    worker.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    worker.scheduler_config.max_num_seqs = max_num_seqs

    live_recurrent_cache = torch.arange(128, dtype=torch.float32)
    worker.model_runner.recurrent_cache = live_recurrent_cache
    return worker, live_recurrent_cache


def _patch_kernels():
    return (
        patch.object(rwt, "rwkv7_mix6"),
        patch.object(rwt, "rwkv7_kk_pre"),
        patch.object(rwt, "rwkv7_lnx_rkvres_xg"),
        patch.object(rwt, "rwkv7_recurrent_t1_cache"),
    )


def test_rwkv7_triton_warmup_noop_without_triton():
    # Given: an RWKV7 worker while Triton is unavailable.
    worker, _ = _make_rwkv7_worker()
    p_mix6, p_kk_pre, p_epilogue, p_t1_cache = _patch_kernels()

    # When: the warmup runs.
    with (
        p_mix6 as mix6,
        p_kk_pre as kk_pre,
        p_epilogue as epilogue,
        p_t1_cache as t1_cache,
        patch.object(rwt, "HAS_TRITON", False),
    ):
        rwt.rwkv7_triton_warmup(worker)

    # Then: every kernel is skipped.
    mix6.assert_not_called()
    kk_pre.assert_not_called()
    epilogue.assert_not_called()
    t1_cache.assert_not_called()


def test_rwkv7_triton_warmup_noop_for_non_rwkv7():
    # Given: a non-RWKV7 worker with Triton available.
    worker, _ = _make_rwkv7_worker(architectures=("LlamaForCausalLM",), model_type="llama")
    p_mix6, p_kk_pre, p_epilogue, p_t1_cache = _patch_kernels()

    # When: the warmup runs.
    with (
        p_mix6 as mix6,
        p_kk_pre as kk_pre,
        p_epilogue as epilogue,
        p_t1_cache as t1_cache,
        patch.object(rwt, "HAS_TRITON", True),
    ):
        rwt.rwkv7_triton_warmup(worker)

    # Then: other architectures are unaffected.
    mix6.assert_not_called()
    kk_pre.assert_not_called()
    epilogue.assert_not_called()
    t1_cache.assert_not_called()


def test_rwkv7_triton_warmup_calls_kernels_for_rwkv7():
    # Given: a TP4 RWKV7 worker; local heads = 32 // 4 = 8.
    worker, live_recurrent_cache = _make_rwkv7_worker()
    p_mix6, p_kk_pre, p_epilogue, p_t1_cache = _patch_kernels()

    # When: the warmup runs.
    with (
        p_mix6 as mix6,
        p_kk_pre as kk_pre,
        p_epilogue as epilogue,
        p_t1_cache as t1_cache,
        patch.object(rwt, "HAS_TRITON", True),
    ):
        rwt.rwkv7_triton_warmup(worker)

    # Then: prefill kernels are JIT-compiled with config-derived shapes.
    tokens = rwt._PREFILL_WARMUP_TOKENS
    assert mix6.call_count == 1
    hidden, delta, *mixing = mix6.call_args.args
    assert hidden.shape == (tokens, 2048)
    assert delta.shape == hidden.shape
    assert all(tensor.shape == (2048,) for tensor in mixing)

    assert kk_pre.call_count == 1
    assert kk_pre.call_args.kwargs["k"].shape == (tokens, 8, 64)
    assert kk_pre.call_args.kwargs["k_k"].shape == (8, 64)
    assert kk_pre.call_args.kwargs["a"].shape == (tokens, 8, 64)
    assert kk_pre.call_args.kwargs["k_a"].shape == (8, 64)

    # Then: every runtime ROWS variant is compiled.
    assert epilogue.call_count == len(rwt._EPILOGUE_ROWS_CANDIDATES)
    first_epilogue = epilogue.call_args_list[0]
    assert first_epilogue.args[0].shape == (tokens, 8, 64)
    assert first_epilogue.args[4].shape == (8, 64)
    assert first_epilogue.args[5].shape == (8 * 64,)
    assert first_epilogue.args[7].shape == (tokens, 8 * 64)
    assert first_epilogue.kwargs["rows"] == rwt._EPILOGUE_ROWS_CANDIDATES[0]

    # Then: decode warms the cache kernel on a fresh scratch cache, not the
    # live recurrent cache owned by the model runner.
    assert t1_cache.call_count == 1
    warm_cache = t1_cache.call_args.args[0]
    assert warm_cache is not live_recurrent_cache
    assert warm_cache.shape == (4, 8, 64, 64)
    assert t1_cache.call_args.args[1].tolist() == [0, 1, 2, 3]


def test_rwkv7_triton_warmup_skips_t1_cache_for_non_64_state():
    # Given: a worker whose head_v_dim is 128, which the cache kernel rejects.
    worker, _ = _make_rwkv7_worker(value_dim=4096)
    p_mix6, p_kk_pre, p_epilogue, p_t1_cache = _patch_kernels()

    # When: the warmup runs.
    with (
        p_mix6 as mix6,
        p_kk_pre,
        p_epilogue as epilogue,
        p_t1_cache as t1_cache,
        patch.object(rwt, "HAS_TRITON", True),
    ):
        rwt.rwkv7_triton_warmup(worker)

    # Then: prefill still warms, but the 64x64 decode kernel is skipped.
    assert mix6.call_count == 1
    assert epilogue.call_count == len(rwt._EPILOGUE_ROWS_CANDIDATES)
    t1_cache.assert_not_called()


def test_rwkv7_triton_warmup_skips_prefill_below_min_tokens():
    # Given: a scheduler that can never reach the T >= 4 prefill guard.
    worker, _ = _make_rwkv7_worker(max_num_batched_tokens=1)
    p_mix6, p_kk_pre, p_epilogue, p_t1_cache = _patch_kernels()

    # When: the warmup runs.
    with (
        p_mix6 as mix6,
        p_kk_pre as kk_pre,
        p_epilogue as epilogue,
        p_t1_cache as t1_cache,
        patch.object(rwt, "HAS_TRITON", True),
    ):
        rwt.rwkv7_triton_warmup(worker)

    # Then: the prefill kernels are skipped while decode still warms.
    mix6.assert_not_called()
    kk_pre.assert_not_called()
    epilogue.assert_not_called()
    t1_cache.assert_called_once()
