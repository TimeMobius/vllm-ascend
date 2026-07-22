# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only RWKV7 model."""

from collections.abc import Iterable
from itertools import islice
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN as HF_ACT2FN

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_ascend.ops.triton.fla import fused_recurrent_rwkv7
from vllm_ascend import envs

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)

LOG_DECAY_SCALE = -0.6065306597126334
RWKV7_RUNTIME_DTYPE = torch.float32
_NATIVE_RWKV7_BLOCK_RE = re.compile(r"blocks\.(\d+)\.(.+)")

_NATIVE_RWKV7_TOP_LEVEL_NAME_MAP = {
    "emb.weight": "model.embed_tokens.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}

_NATIVE_RWKV7_BLOCK_NAME_MAP = {
    "ln1.weight": "attn_norm.weight",
    "ln1.bias": "attn_norm.bias",
    "ln2.weight": "ffn_norm.weight",
    "ln2.bias": "ffn_norm.bias",
    "att.x_r": "attn.x_r",
    "att.x_w": "attn.x_w",
    "att.x_k": "attn.x_k",
    "att.x_v": "attn.x_v",
    "att.x_a": "attn.x_a",
    "att.x_g": "attn.x_g",
    "att.k_k": "attn.k_k",
    "att.k_a": "attn.k_a",
    "att.r_k": "attn.r_k",
    "att.receptance.weight": "attn.r_proj.weight",
    "att.key.weight": "attn.k_proj.weight",
    "att.value.weight": "attn.v_proj.weight",
    "att.output.weight": "attn.o_proj.weight",
    "att.ln_x.weight": "attn.g_norm.weight",
    "att.ln_x.bias": "attn.g_norm.bias",
    "ffn.x_k": "ffn.x_k",
    "ffn.key.weight": "ffn.key.weight",
    "ffn.value.weight": "ffn.value.weight",
}

_NATIVE_RWKV7_LORA_SPECS = {
    "att.w1": ("attn.w_lora.lora.0.weight", "transpose"),
    "att.w2": ("attn.w_lora.lora.2.weight", "transpose"),
    "att.w0": ("attn.w_lora.lora.2.bias", "squeeze"),
    "att.a1": ("attn.a_lora.lora.0.weight", "transpose"),
    "att.a2": ("attn.a_lora.lora.2.weight", "transpose"),
    "att.a0": ("attn.a_lora.lora.2.bias", "squeeze"),
    "att.g1": ("attn.g_lora.lora.0.weight", "transpose"),
    "att.g2": ("attn.g_lora.lora.2.weight", "transpose"),
    "att.v1": ("attn.v_lora.lora.0.weight", "transpose"),
    "att.v2": ("attn.v_lora.lora.2.weight", "transpose"),
    "att.v0": ("attn.v_lora.lora.2.bias", "squeeze"),
}


def get_tp_world_size() -> int:
    return (
        get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1
    )


def get_tp_rank() -> int:
    return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def get_activation_fn(name: str):
    if name == "sqrelu":
        return sqrelu
    if name not in HF_ACT2FN:
        raise ValueError(f"Unsupported RWKV7 activation: {name}")
    return HF_ACT2FN[name]


def _transform_native_rwkv7_tensor(
    tensor: torch.Tensor, transform: str | None
) -> torch.Tensor:
    if transform == "transpose":
        return tensor.transpose(0, 1)
    if transform == "squeeze":
        return tensor.reshape(-1)
    return tensor


def _iter_rwkv7_weight_aliases(
    name: str, tensor: torch.Tensor
) -> Iterable[tuple[str, torch.Tensor]]:
    mapped_top_level_name = _NATIVE_RWKV7_TOP_LEVEL_NAME_MAP.get(name)
    if mapped_top_level_name is not None:
        yield mapped_top_level_name, tensor
        return
    if name == "model.embeddings.weight":
        yield "model.embed_tokens.weight", tensor
        return

    match = _NATIVE_RWKV7_BLOCK_RE.fullmatch(name)
    if match is None:
        yield name, tensor
        return

    layer_idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"model.layers.{layer_idx}"
    if suffix == "ln0.weight" and layer_idx == 0:
        yield f"{prefix}.pre_norm.weight", tensor
        return
    if suffix == "ln0.bias" and layer_idx == 0:
        yield f"{prefix}.pre_norm.bias", tensor
        return

    mapped_name = _NATIVE_RWKV7_BLOCK_NAME_MAP.get(suffix)
    if mapped_name is not None:
        transform = "squeeze" if suffix in {"att.k_k", "att.k_a", "ffn.x_k"} else None
        yield f"{prefix}.{mapped_name}", _transform_native_rwkv7_tensor(tensor, transform)
        return

    lora_spec = _NATIVE_RWKV7_LORA_SPECS.get(suffix)
    if lora_spec is None or (suffix.startswith("att.v") and layer_idx == 0):
        return
    mapped_name, transform = lora_spec
    yield f"{prefix}.{mapped_name}", _transform_native_rwkv7_tensor(tensor, transform)


def token_shift_with_cache(
    hidden_states: torch.Tensor, cached_state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.empty_like(hidden_states)
    if hidden_states.shape[0] == 0:
        final_state = (
            cached_state if cached_state is not None else hidden_states.new_empty(0)
        )
        return delta, final_state

    if cached_state is None:
        delta[0] = -hidden_states[0]
    else:
        delta[0] = cached_state.to(hidden_states.dtype) - hidden_states[0]
    if hidden_states.shape[0] > 1:
        delta[1:] = hidden_states[:-1] - hidden_states[1:]
    final_state = hidden_states[-1].to(
        cached_state.dtype if cached_state is not None else hidden_states.dtype
    )
    return delta, final_state


def token_shift_with_cache_varlen(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cached_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.empty_like(hidden_states)
    if hidden_states.shape[0] == 0:
        if cached_state is not None:
            return delta, cached_state
        return delta, hidden_states.new_empty((0, hidden_states.shape[-1]))

    delta[1:] = hidden_states[:-1] - hidden_states[1:]
    first_token_indices = query_start_loc[:-1].to(dtype=torch.long)
    if cached_state is None:
        delta[first_token_indices] = -hidden_states.index_select(0, first_token_indices)
    else:
        delta[first_token_indices] = cached_state.to(
            hidden_states.dtype
        ) - hidden_states.index_select(0, first_token_indices)

    last_token_indices = (query_start_loc[1:] - 1).to(dtype=torch.long)
    final_state = hidden_states.index_select(0, last_token_indices).to(
        cached_state.dtype if cached_state is not None else hidden_states.dtype
    )
    return delta, final_state


def _custom_op_optional_tensor(
    tensor: torch.Tensor | None,
    *,
    like: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if tensor is not None:
        return tensor
    return like.new_empty((0,), dtype=dtype or like.dtype)


def _custom_op_tensor_or_none(tensor: torch.Tensor) -> torch.Tensor | None:
    return None if tensor.numel() == 0 else tensor


def _rwkv7_recurrent_step(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    sa = (recurrent_state * (-kk).unsqueeze(-1)).sum(dim=-2)
    return (
        torch.exp(w).unsqueeze(-1) * recurrent_state
        + (kk * a).unsqueeze(-1) * sa.unsqueeze(-2)
        + k.unsqueeze(-1) * v.unsqueeze(-2)
    )


def _rwkv7_recurrent_scan(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_state is None:
        recurrent_state = torch.zeros(
            r.shape[1],
            r.shape[2],
            v.shape[2],
            device=r.device,
            dtype=torch.float32,
        )
    else:
        recurrent_state = initial_state.to(torch.float32)

    outputs: list[torch.Tensor] = []
    for idx in range(r.shape[0]):
        recurrent_state = _rwkv7_recurrent_step(
            recurrent_state,
            w[idx],
            kk[idx],
            a[idx],
            k[idx],
            v[idx],
        )
        outputs.append((recurrent_state * r[idx].unsqueeze(-1)).sum(dim=-2))

    return torch.stack(outputs, dim=0), recurrent_state


def _rwkv7_recurrent_scan_varlen(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    query_start_loc: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    num_sequences = query_start_loc.numel() - 1

    for seq_idx in range(num_sequences):
        start = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        seq_initial_state = None if initial_state is None else initial_state[seq_idx]
        seq_output, seq_final_state = _rwkv7_recurrent_scan(
            r[start:end],
            w[start:end],
            k[start:end],
            v[start:end],
            kk[start:end],
            a[start:end],
            seq_initial_state,
        )
        outputs.append(seq_output)
        final_states.append(seq_final_state)

    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


def _rwkv7_cache_all_boundary_positions(
    *,
    num_computed_tokens: int,
    block_idx_first_scheduled_token: int,
    block_idx_last_scheduled_token: int,
    block_size: int,
    query_len: int,
    device: torch.device,
) -> torch.Tensor:
    if block_idx_last_scheduled_token <= block_idx_first_scheduled_token:
        return torch.empty((0,), device=device, dtype=torch.long)
    block_indices = torch.arange(
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        device=device,
        dtype=torch.long,
    )
    positions = (block_indices + 1) * block_size - num_computed_tokens - 1
    return positions[(positions >= 0) & (positions < query_len)]


def _rwkv7_checkpoint_offsets_from_counts(
    checkpoint_counts: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.zeros(
        checkpoint_counts.numel() + 1,
        device=checkpoint_counts.device,
        dtype=torch.long,
    )
    offsets[1:] = torch.cumsum(checkpoint_counts, dim=0)
    return offsets


def _rwkv7_cache_all_packed_checkpoint_metadata(
    *,
    prefill_query_start_loc: torch.Tensor,
    cache_all_state_indices: torch.Tensor,
    block_idx_first_scheduled: torch.Tensor,
    block_idx_last_scheduled: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = prefill_query_start_loc.device
    num_prefills = block_idx_first_scheduled.numel()
    checkpoint_counts = torch.zeros(num_prefills, device=device, dtype=torch.long)
    empty = torch.empty(0, device=device, dtype=torch.long)
    if num_prefills == 0:
        return empty, empty, checkpoint_counts, empty

    seq_starts = prefill_query_start_loc[:-1].to(torch.long)
    query_lens = prefill_query_start_loc[1:].to(torch.long) - seq_starts
    first = block_idx_first_scheduled.to(torch.long)
    last = block_idx_last_scheduled.to(torch.long)
    computed = num_computed_tokens.to(torch.long)
    block_counts = (last - first).clamp_min(0)
    max_blocks = int(block_counts.max().item())
    if max_blocks == 0:
        return empty, empty, checkpoint_counts, empty

    relative = torch.arange(max_blocks, device=device, dtype=torch.long)
    scheduled = relative.unsqueeze(0) < block_counts.unsqueeze(1)
    block_indices = first.unsqueeze(1) + relative.unsqueeze(0)
    positions = (block_indices + 1) * block_size - computed.unsqueeze(1) - 1
    valid = scheduled & (positions >= 0) & (positions < query_lens.unsqueeze(1))
    checkpoint_counts = valid.sum(dim=1)
    if int(checkpoint_counts.sum().item()) == 0:
        return empty, empty, checkpoint_counts, empty

    rows = torch.arange(num_prefills, device=device, dtype=torch.long).unsqueeze(1)
    rows = rows.expand_as(block_indices)
    checkpoint_positions = positions[valid]
    absolute_positions = (positions + seq_starts.unsqueeze(1))[valid]
    block_slot_ids = cache_all_state_indices[rows[valid], block_indices[valid]]
    return (
        checkpoint_positions,
        absolute_positions,
        checkpoint_counts,
        block_slot_ids.to(torch.long),
    )


def rwkv7_attention(
    hidden_states: torch.Tensor,
    cached_shift_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    final_shift_state: torch.Tensor,
    final_recurrent_state: torch.Tensor,
    v_first_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    out, shift_state, recurrent, first_value = self._forward(
        hidden_states,
        _custom_op_tensor_or_none(cached_shift_state),
        _custom_op_tensor_or_none(recurrent_state),
        _custom_op_tensor_or_none(v_first),
    )
    output.copy_(out)
    final_shift_state.copy_(shift_state.to(final_shift_state.dtype))
    final_recurrent_state.copy_(recurrent.to(final_recurrent_state.dtype))
    v_first_out.copy_(first_value.to(v_first_out.dtype))


def rwkv7_attention_fake(
    hidden_states: torch.Tensor,
    cached_shift_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    final_shift_state: torch.Tensor,
    final_recurrent_state: torch.Tensor,
    v_first_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="rwkv7_attention",
    op_func=rwkv7_attention,
    mutates_args=[
        "output",
        "final_shift_state",
        "final_recurrent_state",
        "v_first_out",
    ],
    fake_impl=rwkv7_attention_fake,
    # These wrappers are only exercised on CUDA tensors. Register them
    # explicitly on the CUDA dispatch key so they remain available even when
    # current_platform resolves to an unspecified/CPU platform in test envs.
    dispatch_key="CUDA",
)


def rwkv7_block_forward(
    hidden_states: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    v_first_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_runtime(
        hidden_states,
        _custom_op_tensor_or_none(v_first),
        output,
        v_first_out,
    )


def rwkv7_block_forward_fake(
    hidden_states: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    v_first_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="rwkv7_block_forward",
    op_func=rwkv7_block_forward,
    mutates_args=["output", "v_first_out"],
    fake_impl=rwkv7_block_forward_fake,
    dispatch_key="CUDA",
)


class RWKV7LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool,
        activation: str | None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported RWKV7 LoRA activation: {activation}")

        self.lora = nn.Sequential(
            ReplicatedLinear(
                input_dim,
                low_rank_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.lora.0",
            ),
            act,
            ColumnParallelLinear(
                low_rank_dim,
                output_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.lora.2",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lora[0](x)
        x = self.lora[1](x)
        x, bias = self.lora[2](x)
        if bias is not None:
            x = x + bias
        return x


class RWKV7GroupNorm(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_value_dim = self.value_dim // self.tp_size
        self.value_start = self.tp_rank * self.local_value_dim
        self.value_end = self.value_start + self.local_value_dim
        self.eps = self.head_dim * eps

        self.weight = nn.Parameter(torch.ones(self.value_dim))
        self.bias = nn.Parameter(torch.zeros(self.value_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight[self.value_start : self.value_end]
        bias = self.bias[self.value_start : self.value_end]
        x = x.to(torch.float32)
        x = F.group_norm(
            x.unsqueeze(-1),
            num_groups=self.local_num_heads,
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=self.eps,
        )
        return x.squeeze(-1)


class RWKV7FeedForward(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.intermediate_size is None:
            hidden_ratio = 4 if config.hidden_ratio is None else config.hidden_ratio
            intermediate_size = int(config.hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 31) // 32)
        else:
            intermediate_size = config.intermediate_size

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.act_fn = get_activation_fn(config.hidden_act)
        self.x_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.key = ColumnParallelLinear(
            self.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key",
        )
        self.value = RowParallelLinear(
            intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value",
        )

    def _mix_inputs(
        self, hidden_states: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if envs.RWKV7_USE_FUSED_CMIX and hidden_states.device.type == "npu":
            return torch.addcmul(hidden_states, delta, self.x_k)
        return hidden_states.addcmul(delta, self.x_k)

    def forward(
        self, hidden_states: torch.Tensor, cached_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, final_state = token_shift_with_cache(hidden_states, cached_state)
        mixed = self._mix_inputs(hidden_states, delta)
        hidden, _ = self.key(mixed)
        hidden = self.act_fn(hidden)
        hidden, _ = self.value(hidden)
        return hidden, final_state

    def forward_decode_batch(
        self,
        hidden_states: torch.Tensor,
        cached_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = cached_state.to(hidden_states.dtype) - hidden_states
        mixed = self._mix_inputs(hidden_states, delta)
        hidden, _ = self.key(mixed)
        hidden = self.act_fn(hidden)
        hidden, _ = self.value(hidden)
        return hidden, hidden_states

    def forward_prefill_batch(
        self,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cached_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, final_state = token_shift_with_cache_varlen(
            hidden_states,
            query_start_loc,
            cached_state,
        )
        mixed = self._mix_inputs(hidden_states, delta)
        hidden, _ = self.key(mixed)
        hidden = self.act_fn(hidden)
        hidden, _ = self.value(hidden)
        return hidden, final_state


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim[layer_idx]
        self.head_v_dim = self.value_dim // self.num_heads
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_key_dim = self.hidden_size // self.tp_size
        self.local_value_dim = self.value_dim // self.tp_size
        self.key_start = self.tp_rank * self.local_key_dim
        self.key_end = self.key_start + self.local_key_dim
        self.value_start = self.tp_rank * self.local_value_dim
        self.value_end = self.value_start + self.local_value_dim

        self.x_r = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_a = nn.Parameter(torch.zeros(self.hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.r_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.w_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.decay_low_rank_dim,
            bias=True,
            activation="tanh",
            quant_config=quant_config,
            prefix=f"{prefix}.w_lora",
        )
        self.a_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.a_low_rank_dim,
            bias=True,
            activation=None,
            quant_config=quant_config,
            prefix=f"{prefix}.a_lora",
        )
        if self.layer_idx != 0:
            self.v_lora = RWKV7LoRA(
                self.hidden_size,
                self.value_dim,
                config.v_low_rank_dim,
                bias=True,
                activation=None,
                quant_config=quant_config,
                prefix=f"{prefix}.v_lora",
            )
        self.g_lora = RWKV7LoRA(
            self.hidden_size,
            self.value_dim,
            config.gate_low_rank_dim,
            bias=False,
            activation="sigmoid",
            quant_config=quant_config,
            prefix=f"{prefix}.g_lora",
        )
        self.g_norm = RWKV7GroupNorm(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            eps=config.norm_eps,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _project_recurrent_inputs(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x_r = self.x_r.squeeze(0).squeeze(0)
        x_w = self.x_w.squeeze(0).squeeze(0)
        x_k = self.x_k.squeeze(0).squeeze(0)
        x_v = self.x_v.squeeze(0).squeeze(0)
        x_a = self.x_a.squeeze(0).squeeze(0)
        x_g = self.x_g.squeeze(0).squeeze(0)

        xr = hidden_states.addcmul(delta, x_r)
        xw = hidden_states.addcmul(delta, x_w)
        xk = hidden_states.addcmul(delta, x_k)
        xv = hidden_states.addcmul(delta, x_v)
        xa = hidden_states.addcmul(delta, x_a)
        xg = hidden_states.addcmul(delta, x_g)

        r, _ = self.r_proj(xr)
        w = LOG_DECAY_SCALE * self.w_lora(xw).sigmoid()
        k, _ = self.k_proj(xk)
        v, _ = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError("RWKV7 layers after layer 0 require `v_first`.")
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
            v_first_out = v_first

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        r = r.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        w = w.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        k = k.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        a = a.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        v = v.view(-1, self.local_num_heads, self.head_v_dim).to(torch.float32)

        local_k_k = self.k_k[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )
        local_k_a = self.k_a[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )
        kk = F.normalize(k * local_k_k.to(torch.float32), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * local_k_a.to(torch.float32))
        return r, w, k, v, kk, a, g, v_first_out

    def _finalize_attention_output(
        self,
        recurrent_output: torch.Tensor,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = recurrent_output.reshape(-1, self.local_value_dim)
        output = self.g_norm(output)

        # Cache local_r_k as fp32 to avoid per-call slice+cast on constant
        # model param. Lazy init on first call (same pattern as epilogue
        # patch's _ascend_r_k_fp32, but suffixed with _local to avoid
        # collision).
        local_r_k = getattr(self, "_ascend_r_k_fp32_local", None)
        if local_r_k is None:
            local_r_k = self.r_k[
                self.tp_rank * self.local_num_heads : (self.tp_rank + 1)
                * self.local_num_heads
            ].to(torch.float32)
            self._ascend_r_k_fp32_local = local_r_k
        correction = (
            (r * k * local_r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v
        ).reshape(-1, self.local_value_dim)
        output = (output + correction) * g.to(torch.float32)
        output = output.to(hidden_dtype)
        output, _ = self.o_proj(output)
        return output

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift_with_cache(
            hidden_states, cached_shift_state
        )
        r, w, k, v, kk, a, g, v_first_out = self._project_recurrent_inputs(
            hidden_states,
            delta,
            v_first,
        )

        recurrent_output, final_recurrent_state = _rwkv7_recurrent_scan(
            r,
            w,
            k,
            v,
            kk,
            a,
            recurrent_state,
        )

        output = self._finalize_attention_output(
            recurrent_output,
            r,
            k,
            v,
            g,
            hidden_states.dtype,
        )
        return output, final_shift_state, final_recurrent_state, v_first_out

    def forward_prefill_cache_all(
        self,
        hidden_states: torch.Tensor,
        block_boundary_positions: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift_with_cache(
            hidden_states, cached_shift_state
        )
        r, w, k, v, kk, a, g, v_first_out = self._project_recurrent_inputs(
            hidden_states, delta, v_first
        )
        recurrent_output, final_recurrent_state, checkpoint_states = (
            fused_recurrent_rwkv7(
                r.unsqueeze(0),
                w.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                kk.unsqueeze(0),
                a.unsqueeze(0),
                initial_state=(
                    None
                    if recurrent_state is None
                    else recurrent_state.unsqueeze(0).to(torch.float32)
                ),
                output_final_state=True,
                checkpoint_positions=block_boundary_positions,
                checkpoint_offsets=block_boundary_positions.new_tensor(
                    [0, block_boundary_positions.numel()]
                ),
                output_checkpoint_states=True,
            )
        )
        assert final_recurrent_state is not None
        assert checkpoint_states is not None
        output = self._finalize_attention_output(
            recurrent_output.squeeze(0), r, k, v, g, hidden_states.dtype
        )
        return (
            output,
            final_shift_state,
            final_recurrent_state.squeeze(0),
            v_first_out,
            checkpoint_states,
        )

    def forward_prefill_batch_cache_all(
        self,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        checkpoint_positions: torch.Tensor,
        checkpoint_offsets: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift_with_cache_varlen(
            hidden_states, query_start_loc, cached_shift_state
        )
        r, w, k, v, kk, a, g, v_first_out = self._project_recurrent_inputs(
            hidden_states, delta, v_first
        )
        recurrent_output, final_recurrent_state, checkpoint_states = (
            fused_recurrent_rwkv7(
                r.unsqueeze(0),
                w.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                kk.unsqueeze(0),
                a.unsqueeze(0),
                initial_state=(
                    None
                    if recurrent_state is None
                    else recurrent_state.to(torch.float32)
                ),
                output_final_state=True,
                cu_seqlens=query_start_loc,
                checkpoint_positions=checkpoint_positions,
                checkpoint_offsets=checkpoint_offsets,
                output_checkpoint_states=True,
            )
        )
        assert final_recurrent_state is not None
        assert checkpoint_states is not None
        output = self._finalize_attention_output(
            recurrent_output.squeeze(0), r, k, v, g, hidden_states.dtype
        )
        return (
            output,
            final_shift_state,
            final_recurrent_state,
            v_first_out,
            checkpoint_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward(
            hidden_states,
            cached_shift_state,
            recurrent_state,
            v_first,
        )

        output = hidden_states.new_empty(hidden_states.shape[0], self.hidden_size)
        final_shift_state = hidden_states.new_empty(
            self.hidden_size,
            dtype=(
                cached_shift_state.dtype
                if cached_shift_state is not None
                else hidden_states.dtype
            ),
        )
        final_recurrent_state = hidden_states.new_empty(
            self.local_num_heads,
            self.head_dim,
            self.head_v_dim,
            dtype=RWKV7_RUNTIME_DTYPE,
        )
        v_first_out = hidden_states.new_empty(
            hidden_states.shape[0], self.local_value_dim
        )
        torch.ops.vllm.rwkv7_attention(
            hidden_states,
            _custom_op_optional_tensor(cached_shift_state, like=hidden_states),
            _custom_op_optional_tensor(
                recurrent_state,
                like=hidden_states,
                dtype=RWKV7_RUNTIME_DTYPE,
            ),
            _custom_op_optional_tensor(v_first, like=hidden_states),
            output,
            final_shift_state,
            final_recurrent_state,
            v_first_out,
            self.prefix,
        )
        return output, final_shift_state, final_recurrent_state, v_first_out

    def forward_decode_batch(
        self,
        hidden_states: torch.Tensor,
        cached_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = cached_shift_state.to(hidden_states.dtype) - hidden_states
        final_shift_state = hidden_states
        r, w, k, v, kk, a, g, v_first_out = self._project_recurrent_inputs(
            hidden_states,
            delta,
            v_first,
        )
        # recurrent_state is allocated as float32 by get_state_dtype;
        # skip the redundant .to() call to avoid dispatch overhead.
        if envs.RWKV7_USE_FUSED_RECURRENT_T1:
            from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 import (
                rwkv7_recurrent_t1,
            )

            final_recurrent_state, recurrent_output = rwkv7_recurrent_t1(
                recurrent_state,
                w,
                kk,
                a,
                k,
                v,
                r,
            )
        else:
            final_recurrent_state = _rwkv7_recurrent_step(
                recurrent_state,
                w,
                kk,
                a,
                k,
                v,
            )
            recurrent_output = (final_recurrent_state * r.unsqueeze(-1)).sum(dim=-2)

        output = self._finalize_attention_output(
            recurrent_output,
            r,
            k,
            v,
            g,
            hidden_states.dtype,
        )
        return output, final_shift_state, final_recurrent_state, v_first_out

    def forward_prefill_batch(
        self,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift_with_cache_varlen(
            hidden_states,
            query_start_loc,
            cached_shift_state,
        )
        r, w, k, v, kk, a, g, v_first_out = self._project_recurrent_inputs(
            hidden_states,
            delta,
            v_first,
        )

        recurrent_output, final_recurrent_state = _rwkv7_recurrent_scan_varlen(
            r,
            w,
            k,
            v,
            kk,
            a,
            query_start_loc,
            recurrent_state,
        )
        output = self._finalize_attention_output(
            recurrent_output,
            r,
            k,
            v,
            g,
            hidden_states.dtype,
        )
        return output, final_shift_state, final_recurrent_state, v_first_out


class RWKV7Block(nn.Module, MambaBase):
    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim[layer_idx]
        self.local_value_dim = self.value_dim // get_tp_world_size()

        self.pre_norm = None
        if config.norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                elementwise_affine=True,
                bias=config.norm_bias,
            )
        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.attn = RWKV7Attention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.ffn = RWKV7FeedForward(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.hidden_size,),
            (
                self.num_heads // get_tp_world_size(),
                self.head_dim,
                self.value_dim // self.num_heads,
            ),
            (self.hidden_size,),
        )

    def _run_sequence(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attn_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        ffn_shift_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)

        attn_input = self.attn_norm(residual)
        attn_out, attn_shift_state, recurrent_state, v_first_out = self.attn(
            attn_input,
            attn_shift_state,
            recurrent_state,
            v_first,
        )
        hidden_states = residual + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn(ffn_input, ffn_shift_state)
        hidden_states = hidden_states + ffn_out
        return (
            hidden_states,
            v_first_out,
            attn_shift_state,
            recurrent_state,
            ffn_shift_state,
        )

    def _get_kv_state(
        self, slot_id: int, use_initial_state: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not use_initial_state:
            return None, None, None
        return (
            self.kv_cache[0][slot_id],
            self.kv_cache[1][slot_id],
            self.kv_cache[2][slot_id],
        )

    def _get_kv_states(
        self, slot_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.kv_cache[0].index_select(0, slot_ids),
            self.kv_cache[1].index_select(0, slot_ids),
            self.kv_cache[2].index_select(0, slot_ids),
        )

    def _get_prefill_kv_states(
        self,
        slot_ids: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_shift_state, recurrent_state, ffn_shift_state = self._get_kv_states(
            slot_ids
        )
        has_initial_state = has_initial_state.to(attn_shift_state.device)
        attn_shift_state = torch.where(
            has_initial_state[:, None],
            attn_shift_state,
            torch.zeros_like(attn_shift_state),
        )
        recurrent_state = torch.where(
            has_initial_state[:, None, None, None],
            recurrent_state,
            torch.zeros_like(recurrent_state),
        )
        ffn_shift_state = torch.where(
            has_initial_state[:, None],
            ffn_shift_state,
            torch.zeros_like(ffn_shift_state),
        )
        return attn_shift_state, recurrent_state, ffn_shift_state

    @torch.compiler.disable
    def _store_kv_state(
        self,
        slot_id: int,
        attn_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ffn_shift_state: torch.Tensor,
    ) -> None:
        self.kv_cache[0][slot_id].copy_(
            attn_shift_state.to(self.kv_cache[0][slot_id].dtype)
        )
        self.kv_cache[1][slot_id].copy_(
            recurrent_state.to(self.kv_cache[1][slot_id].dtype)
        )
        self.kv_cache[2][slot_id].copy_(
            ffn_shift_state.to(self.kv_cache[2][slot_id].dtype)
        )

    @torch.compiler.disable
    def _store_kv_states(
        self,
        slot_ids: torch.Tensor,
        attn_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ffn_shift_state: torch.Tensor,
    ) -> None:
        self.kv_cache[0].index_copy_(
            0, slot_ids, attn_shift_state.to(self.kv_cache[0].dtype)
        )
        self.kv_cache[1].index_copy_(
            0, slot_ids, recurrent_state.to(self.kv_cache[1].dtype)
        )
        self.kv_cache[2].index_copy_(
            0, slot_ids, ffn_shift_state.to(self.kv_cache[2].dtype)
        )

    def _run_decode_batch(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attn_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ffn_shift_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)

        attn_input = self.attn_norm(residual)
        attn_out, attn_shift_state, recurrent_state, v_first_out = (
            self.attn.forward_decode_batch(
                attn_input,
                attn_shift_state,
                recurrent_state,
                v_first,
            )
        )
        hidden_states = residual + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn.forward_decode_batch(
            ffn_input, ffn_shift_state
        )
        hidden_states = hidden_states + ffn_out
        return (
            hidden_states,
            v_first_out,
            attn_shift_state,
            recurrent_state,
            ffn_shift_state,
        )

    def _run_prefill_batch(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        attn_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        ffn_shift_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)

        attn_input = self.attn_norm(residual)
        attn_out, attn_shift_state, recurrent_state, v_first_out = (
            self.attn.forward_prefill_batch(
                attn_input,
                query_start_loc,
                attn_shift_state,
                recurrent_state,
                v_first,
            )
        )
        hidden_states = residual + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn.forward_prefill_batch(
            ffn_input,
            query_start_loc,
            ffn_shift_state,
        )
        hidden_states = hidden_states + ffn_out
        return (
            hidden_states,
            v_first_out,
            attn_shift_state,
            recurrent_state,
            ffn_shift_state,
        )

    def _run_prefill_sequence_cache_all(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        num_computed_tokens: int,
        block_idx_first_scheduled_token: int,
        block_idx_last_scheduled_token: int,
        state_indices_row: torch.Tensor,
        attn_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        ffn_shift_state: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.cache_config is not None
        assert self.cache_config.mamba_block_size is not None
        boundary_positions = _rwkv7_cache_all_boundary_positions(
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_size=self.cache_config.mamba_block_size,
            query_len=hidden_states.shape[0],
            device=hidden_states.device,
        )
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)
        attn_input = self.attn_norm(residual)
        (
            attn_out,
            attn_shift_state,
            recurrent_state,
            v_first_out,
            checkpoint_recurrent_states,
        ) = self.attn.forward_prefill_cache_all(
            attn_input,
            boundary_positions,
            attn_shift_state,
            recurrent_state,
            v_first,
        )
        hidden_states = residual + attn_out
        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn(ffn_input, ffn_shift_state)
        hidden_states = hidden_states + ffn_out
        block_slot_ids = state_indices_row[
            block_idx_first_scheduled_token:block_idx_last_scheduled_token
        ].to(torch.long)
        return (
            hidden_states,
            v_first_out,
            attn_shift_state,
            recurrent_state,
            ffn_shift_state,
            block_slot_ids,
            attn_input.index_select(0, boundary_positions),
            checkpoint_recurrent_states,
            ffn_input.index_select(0, boundary_positions),
        )

    def _run_prefill_batch_cache_all(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        checkpoint_positions: torch.Tensor,
        checkpoint_offsets: torch.Tensor,
        checkpoint_absolute_positions: torch.Tensor,
        attn_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ffn_shift_state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)
        attn_input = self.attn_norm(residual)
        (
            attn_out,
            attn_shift_state,
            recurrent_state,
            v_first_out,
            checkpoint_recurrent_states,
        ) = self.attn.forward_prefill_batch_cache_all(
            attn_input,
            query_start_loc,
            checkpoint_positions,
            checkpoint_offsets,
            attn_shift_state,
            recurrent_state,
            v_first,
        )
        hidden_states = residual + attn_out
        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn.forward_prefill_batch(
            ffn_input, query_start_loc, ffn_shift_state
        )
        hidden_states = hidden_states + ffn_out
        return (
            hidden_states,
            v_first_out,
            attn_shift_state,
            recurrent_state,
            ffn_shift_state,
            attn_input.index_select(0, checkpoint_absolute_positions),
            checkpoint_recurrent_states,
            ffn_input.index_select(0, checkpoint_absolute_positions),
        )

    def _forward_runtime(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        output: torch.Tensor,
        v_first_out: torch.Tensor,
        attn_metadata: LinearAttentionMetadata | None = None,
    ) -> None:
        if attn_metadata is None and is_forward_context_available():
            forward_context = get_forward_context()
            runtime_attn_metadata = forward_context.attn_metadata
            if runtime_attn_metadata is not None:
                assert isinstance(runtime_attn_metadata, dict)
                maybe_metadata = runtime_attn_metadata.get(self.prefix)
                assert maybe_metadata is None or isinstance(
                    maybe_metadata, LinearAttentionMetadata
                )
                attn_metadata = maybe_metadata

        if attn_metadata is None:
            out, vf_out, _, _, _ = self._run_sequence(
                hidden_states, v_first, None, None, None
            )
            output[: out.shape[0]] = out
            v_first_out[: vf_out.shape[0]] = vf_out
            return

        num_actual_tokens = (
            attn_metadata.num_decode_tokens + attn_metadata.num_prefill_tokens
        )
        hidden_states = hidden_states[:num_actual_tokens]
        if v_first is not None:
            v_first = v_first[:num_actual_tokens]

        output_slice = output[:num_actual_tokens]
        v_first_slice = v_first_out[:num_actual_tokens]
        state_indices = attn_metadata.state_indices_tensor
        cache_all = (
            self.cache_config is not None
            and self.cache_config.mamba_cache_mode == "all"
            and getattr(attn_metadata, "num_computed_tokens", None) is not None
            and getattr(attn_metadata, "block_idx_last_computed_token", None)
            is not None
            and getattr(attn_metadata, "block_idx_first_scheduled_token", None)
            is not None
            and getattr(attn_metadata, "block_idx_last_scheduled_token", None)
            is not None
        )

        if attn_metadata.num_decode_tokens > 0:
            if cache_all:
                decode_slot_ids = state_indices[: attn_metadata.num_decodes].gather(
                    1,
                    attn_metadata.block_idx_last_computed_token[
                        : attn_metadata.num_decodes
                    ].unsqueeze(1),
                ).squeeze(1).to(torch.long)
                decode_output_slot_ids = state_indices[
                    : attn_metadata.num_decodes
                ].gather(
                    1,
                    attn_metadata.block_idx_last_scheduled_token[
                        : attn_metadata.num_decodes
                    ].unsqueeze(1),
                ).squeeze(1).to(torch.long)
            else:
                decode_slot_ids = state_indices[: attn_metadata.num_decodes].to(
                    dtype=torch.long
                )
                decode_output_slot_ids = decode_slot_ids
            states = self._get_kv_states(decode_slot_ids)
            out, vf_out, attn_shift, recurrent, ffn_shift = self._run_decode_batch(
                hidden_states[: attn_metadata.num_decode_tokens],
                None if v_first is None else v_first[: attn_metadata.num_decode_tokens],
                *states,
            )
            output_slice[: attn_metadata.num_decode_tokens] = out
            v_first_slice[: attn_metadata.num_decode_tokens] = vf_out
            self._store_kv_states(
                decode_output_slot_ids,
                attn_shift,
                recurrent,
                ffn_shift,
            )

        prefill_req_offset = attn_metadata.num_decodes
        prefill_token_offset = attn_metadata.num_decode_tokens
        if attn_metadata.num_prefills > 0:
            prefill_req_end = prefill_req_offset + attn_metadata.num_prefills
            if cache_all:
                prefill_query_start_loc = (
                    attn_metadata.query_start_loc[
                        prefill_req_offset : prefill_req_end + 1
                    ]
                    - prefill_token_offset
                )
                cache_all_state_indices = state_indices[
                    prefill_req_offset:prefill_req_end
                ]
                block_idx_last_computed = attn_metadata.block_idx_last_computed_token[
                    prefill_req_offset:prefill_req_end
                ]
                block_idx_first_scheduled = attn_metadata.block_idx_first_scheduled_token[
                    prefill_req_offset:prefill_req_end
                ]
                block_idx_last_scheduled = attn_metadata.block_idx_last_scheduled_token[
                    prefill_req_offset:prefill_req_end
                ]
                num_computed_tokens = attn_metadata.num_computed_tokens[
                    prefill_req_offset:prefill_req_end
                ]
                input_slot_ids = cache_all_state_indices.gather(
                    1, block_idx_last_computed.unsqueeze(1)
                ).squeeze(1).to(torch.long)
                output_slot_ids = cache_all_state_indices.gather(
                    1, block_idx_last_scheduled.unsqueeze(1)
                ).squeeze(1).to(torch.long)
                has_initial_state = num_computed_tokens > 0
                (
                    checkpoint_positions,
                    checkpoint_absolute_positions,
                    checkpoint_counts,
                    block_slot_ids,
                ) = _rwkv7_cache_all_packed_checkpoint_metadata(
                    prefill_query_start_loc=prefill_query_start_loc,
                    cache_all_state_indices=cache_all_state_indices,
                    block_idx_first_scheduled=block_idx_first_scheduled,
                    block_idx_last_scheduled=block_idx_last_scheduled,
                    num_computed_tokens=num_computed_tokens,
                    block_size=self.cache_config.mamba_block_size,
                )
                checkpoint_offsets = _rwkv7_checkpoint_offsets_from_counts(
                    checkpoint_counts
                )
                states = self._get_prefill_kv_states(
                    input_slot_ids, has_initial_state
                )
                (
                    out,
                    vf_out,
                    attn_shift,
                    recurrent,
                    ffn_shift,
                    block_attn_shift_states,
                    checkpoint_recurrent_states,
                    block_ffn_shift_states,
                ) = self._run_prefill_batch_cache_all(
                    hidden_states[prefill_token_offset:num_actual_tokens],
                    None
                    if v_first is None
                    else v_first[prefill_token_offset:num_actual_tokens],
                    prefill_query_start_loc,
                    checkpoint_positions,
                    checkpoint_offsets,
                    checkpoint_absolute_positions,
                    *states,
                )
                output_slice[prefill_token_offset:num_actual_tokens] = out
                v_first_slice[prefill_token_offset:num_actual_tokens] = vf_out
                self._store_kv_states(
                    torch.cat((block_slot_ids, output_slot_ids)),
                    torch.cat((block_attn_shift_states, attn_shift)),
                    torch.cat((checkpoint_recurrent_states, recurrent)),
                    torch.cat((block_ffn_shift_states, ffn_shift)),
                )
                return
            prefill_slot_ids = state_indices[prefill_req_offset:prefill_req_end].to(
                dtype=torch.long
            )
            prefill_query_start_loc = (
                attn_metadata.query_start_loc[prefill_req_offset : prefill_req_end + 1]
                - prefill_token_offset
            )
            query_lens = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
            prefill_seq_lens = attn_metadata.seq_lens[
                prefill_req_offset:prefill_req_end
            ]
            has_initial_state = prefill_seq_lens > query_lens
            states = self._get_prefill_kv_states(
                prefill_slot_ids,
                has_initial_state,
            )
            out, vf_out, attn_shift, recurrent, ffn_shift = self._run_prefill_batch(
                hidden_states[prefill_token_offset:num_actual_tokens],
                (
                    None
                    if v_first is None
                    else v_first[prefill_token_offset:num_actual_tokens]
                ),
                prefill_query_start_loc,
                *states,
            )
            output_slice[prefill_token_offset:num_actual_tokens] = out
            v_first_slice[prefill_token_offset:num_actual_tokens] = vf_out
            self._store_kv_states(
                prefill_slot_ids,
                attn_shift,
                recurrent,
                ffn_shift,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attn_metadata: LinearAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(hidden_states)
        v_first_out = torch.empty(
            (hidden_states.shape[0], self.local_value_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward_runtime(
            hidden_states,
            v_first,
            output,
            v_first_out,
            attn_metadata=attn_metadata,
        )
        return output, v_first_out


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class RWKV7Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.runtime_dtype = model_config.dtype

        if config.attn is not None:
            raise NotImplementedError(
                "Hybrid RWKV7 checkpoints with transformer attention are "
                "not supported yet."
            )

        value_dims = config.value_dim
        if len(set(value_dims)) != 1:
            raise NotImplementedError(
                "RWKV7 with per-layer `value_dim` variation is not supported yet."
            )

        self.local_value_dim = value_dims[0] // get_tp_world_size()
        self.vocab_size = config.vocab_size
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RWKV7Block(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = (
            nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                elementwise_affine=True,
                bias=config.norm_bias,
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _pp_runtime_dtype(self) -> torch.dtype:
        # RWKV7 keeps inter-stage activations in the model dtype so PP dummy runs
        # and stage-to-stage transfers match the numerics used inside blocks.
        return self.runtime_dtype

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        del dtype
        runtime_dtype = self._pp_runtime_dtype()
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=runtime_dtype,
                    device=device,
                ),
                "v_first": torch.zeros(
                    (batch_size, self.local_value_dim),
                    dtype=runtime_dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del positions
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            v_first = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            v_first = intermediate_tensors["v_first"]
            runtime_dtype = self._pp_runtime_dtype()
            if hidden_states.dtype != runtime_dtype:
                hidden_states = hidden_states.to(runtime_dtype)
            if v_first.dtype != runtime_dtype:
                v_first = v_first.to(runtime_dtype)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer_metadata = (
                None if attn_metadata is None else attn_metadata.get(layer.prefix)
            )
            assert layer_metadata is None or isinstance(
                layer_metadata, LinearAttentionMetadata
            )
            hidden_states, v_first = layer(
                hidden_states=hidden_states,
                v_first=v_first,
                attn_metadata=layer_metadata,
            )

        if not get_pp_group().is_last_rank:
            assert v_first is not None
            runtime_dtype = self._pp_runtime_dtype()
            if hidden_states.dtype != runtime_dtype:
                hidden_states = hidden_states.to(runtime_dtype)
            if v_first.dtype != runtime_dtype:
                v_first = v_first.to(runtime_dtype)
            return IntermediateTensors(
                {"hidden_states": hidden_states, "v_first": v_first}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class RWKV7ForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
    SupportsMambaPrefixCaching,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = RWKV7Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        model_dtype = vllm_config.model_config.dtype
        self.model.to(model_dtype)
        if get_pp_group().is_last_rank:
            self.lm_head.to(model_dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits_dtype = getattr(self.lm_head, "weight", hidden_states).dtype
        return self.logits_processor(self.lm_head, hidden_states.to(logits_dtype))

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return (
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        config = vllm_config.model_config.hf_config
        if len(set(config.value_dim)) != 1:
            raise NotImplementedError(
                "RWKV7 with per-layer `value_dim` variation is not supported yet."
            )
        return (
            (config.hidden_size,),
            (
                config.num_heads // vllm_config.parallel_config.tensor_parallel_size,
                config.head_dim,
                config.value_dim[0] // config.num_heads,
            ),
            (config.hidden_size,),
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        conv, temporal = MambaStateCopyFuncCalculator.mamba1_state_copy_func()
        return (conv, temporal, conv)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def iter_weights():
            for name, tensor in weights:
                yield from _iter_rwkv7_weight_aliases(name, tensor)

        loader = AutoWeightsLoader(self)
        return loader.load_weights(iter_weights())
