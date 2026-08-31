# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.patch.worker import patch_rwkv7

BLOCK_SIZE = 4
DEVICE = torch.device("cpu")


def _create_builder(
    cache_mode: str,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_DECODE_ONLY,
) -> LinearAttentionMetadataBuilder:
    patch_rwkv7.apply_patch()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=cache_mode),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=None,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=16),
    )
    return LinearAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=0,
        ),
        layer_names=["model.layers.0.self_attn"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _create_decode_metadata(block_table: list[list[int]]) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([8, 8], dtype=torch.int32, device=DEVICE)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32, device=DEVICE),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=DEVICE),
        _num_computed_tokens_cpu=torch.tensor([7, 7], dtype=torch.int32),
    )


def _create_padded_decode_metadata(
    block_table: list[list[int]],
) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([8, 8, 0, 0], dtype=torch.int32, device=DEVICE)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        num_reqs=4,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32, device=DEVICE),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device=DEVICE),
        _num_computed_tokens_cpu=torch.tensor([7, 7, 0, 0], dtype=torch.int32),
    )


def _create_mixed_metadata(block_table: list[list[int]]) -> CommonAttentionMetadata:
    query_start_loc = torch.tensor([0, 1, 6], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([8, 13], dtype=torch.int32, device=DEVICE)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=5,
        max_seq_len=13,
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32, device=DEVICE),
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=DEVICE),
        _num_computed_tokens_cpu=torch.tensor([7, 8], dtype=torch.int32),
    )


def test_rwkv7_align_metadata_reuses_state_index_storage_and_refreshes_values():
    builder = _create_builder("align")

    first = builder.build(
        0,
        _create_decode_metadata([[10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    second = builder.build(
        0,
        _create_decode_metadata([[30, 31, 32, 33], [40, 41, 42, 43]]),
    )

    assert second.state_indices_tensor.data_ptr() == first.state_indices_tensor.data_ptr()
    assert second.state_indices_tensor.tolist() == [31, 41]


def test_rwkv7_cache_all_metadata_reuses_all_graph_inputs_and_refreshes_values():
    builder = _create_builder("all")

    first = builder.build(
        0,
        _create_decode_metadata([[10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    first_ptrs = {
        name: getattr(first, name).data_ptr()
        for name in (
            "state_indices_tensor",
            "num_computed_tokens",
            "block_idx_last_computed_token",
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
        )
    }

    second = builder.build(
        0,
        _create_decode_metadata([[30, 31, 32, 33], [40, 41, 42, 43]]),
    )

    for name, pointer in first_ptrs.items():
        assert getattr(second, name).data_ptr() == pointer
    assert second.state_indices_tensor.tolist() == [
        [30, 31, 32, 33],
        [40, 41, 42, 43],
    ]
    assert second.num_computed_tokens.tolist() == [7, 7]
    assert second.block_idx_last_computed_token.tolist() == [1, 1]
    assert second.block_idx_first_scheduled_token.tolist() == [1, 1]
    assert second.block_idx_last_scheduled_token.tolist() == [1, 1]


def test_rwkv7_full_piecewise_all_padded_keeps_all_metadata_stable():
    builder = _create_builder("all", CUDAGraphMode.FULL_AND_PIECEWISE)

    first = builder.build(
        0,
        _create_padded_decode_metadata([[10, 11, 12, 13], [20, 21, 22, 23], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )
    first_ptrs = {
        name: getattr(first, name).data_ptr()
        for name in (
            "state_indices_tensor",
            "num_computed_tokens",
            "block_idx_last_computed_token",
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
        )
    }

    second = builder.build(
        0,
        _create_padded_decode_metadata([[30, 31, 32, 33], [40, 41, 42, 43], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )

    for name, pointer in first_ptrs.items():
        assert getattr(second, name).data_ptr() == pointer
    assert second.state_indices_tensor.tolist() == [
        [30, 31, 32, 33],
        [40, 41, 42, 43],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert second.num_computed_tokens.tolist() == [7, 7, 0, 0]
    assert second.block_idx_last_computed_token.tolist() == [1, 1, 0, 0]
    assert second.block_idx_first_scheduled_token.tolist() == [1, 1, 0, 0]
    assert second.block_idx_last_scheduled_token.tolist() == [1, 1, 0, 0]


def test_rwkv7_full_piecewise_none_padded_returns_raw_first_column():
    builder = _create_builder("none", CUDAGraphMode.FULL_AND_PIECEWISE)

    first = builder.build(
        0,
        _create_padded_decode_metadata([[10], [20], [0], [0]]),
    )

    second = builder.build(
        0,
        _create_padded_decode_metadata([[30], [40], [0], [0]]),
    )

    assert second.state_indices_tensor.data_ptr() == first.state_indices_tensor.data_ptr()
    assert second.state_indices_tensor.shape == (4,)
    assert second.state_indices_tensor.tolist() == [30, 40, 0, 0]


def test_rwkv7_full_piecewise_all_mixed_passthrough():
    builder = _create_builder("all", CUDAGraphMode.FULL_AND_PIECEWISE)
    input_block_table = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
        device=DEVICE,
    )
    metadata = _create_mixed_metadata([[10, 11, 12, 13], [20, 21, 22, 23]])
    metadata.block_table_tensor = input_block_table

    built = builder.build(0, metadata)

    assert built.state_indices_tensor.data_ptr() == input_block_table.data_ptr()
    assert built.state_indices_tensor.shape == (2, 4)
    assert built.state_indices_tensor.data_ptr() != builder._rwkv7_state_indices.data_ptr()
    assert built.num_prefills == 1
    assert built.num_decodes == 1
    assert built.num_computed_tokens.tolist() == [7, 8]
    assert built.block_idx_last_computed_token.tolist() == [1, 1]
    assert built.block_idx_first_scheduled_token.tolist() == [1, 2]
    assert built.block_idx_last_scheduled_token.tolist() == [1, 3]
