import torch

from vllm_ascend.models.rwkv7 import (
    RWKV7ForCausalLM,
    _rwkv7_cache_all_boundary_positions,
    _rwkv7_cache_all_packed_checkpoint_metadata,
    _rwkv7_checkpoint_offsets_from_counts,
)


def test_cache_all_boundary_positions_when_blocks_overlap_query():
    positions = _rwkv7_cache_all_boundary_positions(
        num_computed_tokens=3,
        block_idx_first_scheduled_token=0,
        block_idx_last_scheduled_token=3,
        block_size=4,
        query_len=9,
        device=torch.device("cpu"),
    )

    assert torch.equal(positions, torch.tensor([0, 4, 8]))


def test_cache_all_packed_metadata_when_sequences_have_partial_blocks():
    checkpoint_positions, absolute_positions, counts, block_slots = (
        _rwkv7_cache_all_packed_checkpoint_metadata(
            prefill_query_start_loc=torch.tensor([0, 5, 9]),
            cache_all_state_indices=torch.tensor(
                [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
            ),
            block_idx_first_scheduled=torch.tensor([0, 1]),
            block_idx_last_scheduled=torch.tensor([2, 3]),
            num_computed_tokens=torch.tensor([0, 4]),
            block_size=4,
        )
    )

    assert torch.equal(checkpoint_positions, torch.tensor([3, 3]))
    assert torch.equal(absolute_positions, torch.tensor([3, 8]))
    assert torch.equal(counts, torch.tensor([1, 1]))
    assert torch.equal(block_slots, torch.tensor([10, 21]))


def test_checkpoint_offsets_when_counts_are_batched():
    offsets = _rwkv7_checkpoint_offsets_from_counts(torch.tensor([2, 0, 3]))

    assert torch.equal(offsets, torch.tensor([0, 2, 2, 5]))


def test_rwkv7_declares_inner_state_and_attention_free():
    assert RWKV7ForCausalLM.has_inner_state is True
    assert RWKV7ForCausalLM.is_attention_free is True
