# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up RWKV7 Triton kernels used during prefill and decode on Ascend NPU.

The RWKV7 throughput path dispatches ``rwkv7_mix6`` and ``rwkv7_kk_pre`` plus
the ``rwkv7_lnx_rkvres_xg`` epilogue during prefill, and
``rwkv7_recurrent_t1_cache`` (backed by ``rwkv7_recurrent_t1_matrix``) during
decode. Without this warmup the first real prefill/decode pays the Triton JIT
compilation cost.

Every scratch tensor is allocated here, so the live recurrent state and KV
cache owned by the model runner are never read or written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.fla.rwkv7_epilogue import rwkv7_lnx_rkvres_xg
from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import rwkv7_kk_pre
from vllm_ascend.ops.triton.fla.rwkv7_mix6 import rwkv7_mix6
from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1_cache import (
    rwkv7_recurrent_t1_cache,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

_RWKV7_ARCHITECTURES = ("RWKV7ForCausalLM",)
_RWKV7_MODEL_TYPE = "rwkv7"

# ``rwkv7_recurrent_t1_cache`` hard-requires a 64x64 FP32 state; smaller or
# differently shaped states fall back to the PyTorch reference, so there is no
# Triton kernel to warm for them.
_T1_CACHE_STATE_DIM = 64

# ``ROWS`` constexpr variants accepted by ``rwkv7_lnx_rkvres_xg``. The model
# selects one per batch via ``select_epilogue_rows`` (see
# ``vllm_ascend/models/rwkv7.py``); warm them all so every runtime choice is
# already compiled.
_EPILOGUE_ROWS_CANDIDATES = (1, 2, 4, 8, 16)
# ``rwkv7_lnx_rkvres_xg`` falls back to the reference for larger blocks.
_EPILOGUE_MAX_BLOCK = 1024

# The dispatch guards skip Triton below T=4 (decode M=1), so warm with a small
# prefill-sized batch. JIT compilation is keyed by constexprs/dtypes, not by the
# runtime token count, which keeps the scratch allocation tiny.
_MIN_TRITON_TOKENS = 4
_PREFILL_WARMUP_TOKENS = 16
# The grouped cache kernel has a fixed BLOCK_B constexpr, so a small batch is
# enough to compile the same specialization used during decode.
_DECODE_WARMUP_BATCH = 8


class _RWKV7Shapes(NamedTuple):
    hidden_size: int
    local_heads: int
    head_dim: int
    head_v_dim: int
    norm_eps: float


def _hf_config(model_config):
    config = getattr(model_config, "hf_text_config", None)
    if config is None:
        config = getattr(model_config, "hf_config", None)
    return config


def model_uses_rwkv7(model_config) -> bool:
    """Return whether ``model_config`` describes an RWKV7 checkpoint."""
    architectures = getattr(model_config, "architectures", None)
    if isinstance(architectures, list | tuple) and any(arch in _RWKV7_ARCHITECTURES for arch in architectures):
        return True
    hf_config = _hf_config(model_config)
    return getattr(hf_config, "model_type", None) == _RWKV7_MODEL_TYPE


def _resolve_shapes(worker: NPUWorker) -> _RWKV7Shapes | None:
    """Derive kernel shapes from the model config, or ``None`` if unavailable."""
    model_config = worker.vllm_config.model_config
    hf_config = _hf_config(model_config)

    hidden_size = getattr(hf_config, "hidden_size", None)
    head_dim = getattr(hf_config, "head_dim", None)
    num_heads = getattr(hf_config, "num_heads", None)
    if hidden_size is None or head_dim is None or num_heads is None:
        return None

    tp_size = max(getattr(worker.vllm_config.parallel_config, "tensor_parallel_size", 1), 1)
    local_heads = num_heads // tp_size
    if local_heads < 1:
        return None

    value_dim = getattr(hf_config, "value_dim", None)
    if isinstance(value_dim, list | tuple):
        value_dim = value_dim[0] if value_dim else hidden_size
    if value_dim is None:
        value_dim = hidden_size
    head_v_dim = value_dim // num_heads

    return _RWKV7Shapes(
        hidden_size=hidden_size,
        local_heads=local_heads,
        head_dim=head_dim,
        head_v_dim=head_v_dim,
        norm_eps=float(getattr(hf_config, "norm_eps", 1e-5)),
    )


def _warm_mix6(device: torch.device, dtype: torch.dtype, shapes: _RWKV7Shapes) -> None:
    tokens = _PREFILL_WARMUP_TOKENS
    hidden = torch.zeros(tokens, shapes.hidden_size, dtype=dtype, device=device)
    delta = torch.zeros_like(hidden)
    mixing = torch.zeros(shapes.hidden_size, dtype=dtype, device=device)
    rwkv7_mix6(hidden, delta, mixing, mixing, mixing, mixing, mixing, mixing)


def _warm_kk_pre(device: torch.device, shapes: _RWKV7Shapes) -> None:
    tokens = _PREFILL_WARMUP_TOKENS
    k = torch.zeros(tokens, shapes.local_heads, shapes.head_dim, dtype=torch.float32, device=device)
    a = torch.zeros_like(k)
    k_k = torch.zeros(shapes.local_heads, shapes.head_dim, dtype=torch.float32, device=device)
    k_a = torch.zeros_like(k_k)
    rwkv7_kk_pre(k=k, k_k=k_k, a=a, k_a=k_a)


def _warm_epilogue(device: torch.device, shapes: _RWKV7Shapes) -> None:
    if max(shapes.head_dim, shapes.head_v_dim) > _EPILOGUE_MAX_BLOCK:
        return

    tokens = _PREFILL_WARMUP_TOKENS
    num_heads = shapes.local_heads
    head_dim = shapes.head_dim
    head_v_dim = shapes.head_v_dim
    local_value_dim = num_heads * head_v_dim

    recurrent_output = torch.zeros(tokens, num_heads, head_v_dim, dtype=torch.float32, device=device)
    r = torch.zeros(tokens, num_heads, head_dim, dtype=torch.float32, device=device)
    k = torch.zeros_like(r)
    v = torch.zeros_like(recurrent_output)
    r_k = torch.zeros(num_heads, head_dim, dtype=torch.float32, device=device)
    weight = torch.zeros(local_value_dim, dtype=torch.float32, device=device)
    bias = torch.zeros(local_value_dim, dtype=torch.float32, device=device)
    g = torch.zeros(tokens, local_value_dim, dtype=torch.float32, device=device)

    total_rows = tokens * num_heads
    for rows in _EPILOGUE_ROWS_CANDIDATES:
        if rows > total_rows:
            continue
        rwkv7_lnx_rkvres_xg(
            recurrent_output,
            r,
            k,
            v,
            r_k,
            weight,
            bias,
            g,
            eps=shapes.norm_eps,
            rows=rows,
        )


def _warm_recurrent_t1_cache(worker: NPUWorker, device: torch.device, shapes: _RWKV7Shapes) -> None:
    if shapes.head_dim != _T1_CACHE_STATE_DIM or shapes.head_v_dim != _T1_CACHE_STATE_DIM:
        return

    max_num_seqs = getattr(worker.scheduler_config, "max_num_seqs", 0) or 0
    batch = max(min(max_num_seqs, _DECODE_WARMUP_BATCH), 1)
    num_heads = shapes.local_heads
    dim = shapes.head_dim
    value_dim = shapes.head_v_dim

    # Fresh scratch cache: the live recurrent cache is never touched.
    recurrent_cache = torch.zeros(batch, num_heads, dim, value_dim, dtype=torch.float32, device=device)
    slot_ids = torch.arange(batch, dtype=torch.long, device=device)
    w = torch.zeros(batch, num_heads, dim, dtype=torch.float32, device=device)
    kk = torch.zeros_like(w)
    a = torch.zeros_like(w)
    k = torch.zeros_like(w)
    r = torch.zeros_like(w)
    v = torch.zeros(batch, num_heads, value_dim, dtype=torch.float32, device=device)
    rwkv7_recurrent_t1_cache(recurrent_cache, slot_ids, w, kk, a, k, v, r)


@torch.inference_mode()
def rwkv7_triton_warmup(worker: NPUWorker) -> None:
    """JIT the RWKV7 prefill/decode Triton kernels before the first real call."""
    if not HAS_TRITON:
        return
    if not model_uses_rwkv7(worker.vllm_config.model_config):
        return

    shapes = _resolve_shapes(worker)
    if shapes is None:
        return

    device = worker.device
    dtype = worker.model_config.dtype

    max_num_batched_tokens = getattr(worker.scheduler_config, "max_num_batched_tokens", 0) or 0
    if max_num_batched_tokens >= _MIN_TRITON_TOKENS:
        _warm_mix6(device, dtype, shapes)
        _warm_kk_pre(device, shapes)
        _warm_epilogue(device, shapes)

    _warm_recurrent_t1_cache(worker, device, shapes)
