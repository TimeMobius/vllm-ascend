# SPDX-License-Identifier: Apache-2.0

"""Safety invariants for the opt-in single-group Mamba partial-hash-hit path.

These lock the state-block copy-on-write contract that the coordinator and
``MambaManager`` rely on. A sub-block prefix-cache hit must never hand a new
request the shared producer block: the request has to be redirected to a
private copy so the producer's cached recurrent/SSM state stays intact for
later reuse. This is the invariant the coordinator routing and the
``MambaManager`` slot bookkeeping must preserve.

Baseline behavior is unchanged: with the env opt-in off the single group stays
on ``UnitaryKVCacheCoordinator`` and no sub-block entry is ever created.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_manager as kcm
from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    get_kv_cache_coordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_resolve_kv_cache_block_sizes,
)

_ENV = "VLLM_ASCEND_ENABLE_MAMBA_FINE_GRAINED_PREFIX_CACHE"
_SUPPORT_CONST = "_MAMBA_FINE_GRAINED_PREFIX_CACHE_WORKER_COPY_SUPPORTED"


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _make_single_mamba_config(block_size: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
        ],
    )


def _make_vllm_config(
    *, block_size: int, prefix_match_unit: int | None
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None,
    )


def _make_request(request_id: str, token_ids: list[int], hash_block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _make_opt_in_manager(
    monkeypatch, *, block_size: int, prefix_match_unit: int
) -> tuple[KVCacheManager, int]:
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(patch_kv_cache_utils, _SUPPORT_CONST, True)
    kv_cache_config = _make_single_mamba_config(block_size)
    vllm_config = _make_vllm_config(
        block_size=block_size, prefix_match_unit=prefix_match_unit
    )
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    monkeypatch.setattr(kcm, "get_kv_cache_coordinator", get_kv_cache_coordinator)
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )
    return manager, hash_block_size


def test_single_group_partial_hit_uses_cow_copy(monkeypatch) -> None:
    # Given an opt-in single Mamba group whose state block spans 8 tokens and a
    # 6-token producer prompt ending inside that block.
    block_size = 8
    hash_block_size = 2
    manager, hash_block_size = _make_opt_in_manager(
        monkeypatch, block_size=block_size, prefix_match_unit=hash_block_size
    )
    assert manager.coordinator.enable_partial_hash_hits

    producer = _make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size)
    producer_blocks, producer_computed, _ = manager.get_computed_blocks(producer)
    assert producer_computed == 0
    assert manager.allocate_slots(producer, 6, producer_computed, producer_blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    partial_hash = producer.block_hashes[6 // hash_block_size - 1]
    partial_block = manager.block_pool.get_cached_block(partial_hash, kv_cache_group_ids=[0])
    assert partial_block is not None
    assert partial_block[0].block_hash_num_tokens == 6

    # When a consumer replays the same 6-token prefix and continues.
    consumer = _make_request("consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 6
    new_blocks = manager.allocate_slots(consumer, 2, num_computed, computed_blocks)
    assert new_blocks is not None

    # Then the consumer is redirected to a private block and the worker is told
    # to copy the producer's state block into it.
    consumer_block_ids = new_blocks.get_block_ids()[0]
    assert len(consumer_block_ids) == 1
    assert consumer_block_ids[0] != partial_block[0].block_id
    copies, retained = manager.take_kv_cache_block_copies()
    assert (
        KVCacheBlockCopy(
            src_block_id=partial_block[0].block_id,
            dst_block_id=consumer_block_ids[0],
        )
        in copies
    )
    # Both copy endpoints stay pinned until the copy has run on the worker.
    assert partial_block[0] in retained


def test_single_group_baseline_stays_unitary_and_has_no_partial_entries(
    monkeypatch,
) -> None:
    # Given the opt-in disabled: the single group keeps the baseline geometry.
    monkeypatch.delenv(_ENV, raising=False)
    block_size = 8
    hash_block_size = 8
    kv_cache_config = _make_single_mamba_config(block_size)
    monkeypatch.setattr(kcm, "get_kv_cache_coordinator", get_kv_cache_coordinator)
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )

    # Then the coordinator is the upstream unitary one and partial hits are off.
    assert isinstance(manager.coordinator, UnitaryKVCacheCoordinator)
    assert manager.coordinator.enable_partial_hash_hits is False

    # And a producer is served the ordinary block-aligned path with no
    # copy-on-write redirection.
    producer = _make_request("producer", list(range(block_size)), hash_block_size)
    blocks, computed, _ = manager.get_computed_blocks(producer)
    assert computed == 0
    assert manager.allocate_slots(producer, block_size, computed, blocks) is not None
    copies, _ = manager.take_kv_cache_block_copies()
    assert copies == []
    manager.free(producer)
    manager.new_step_starts()

    # Exactly the one full block boundary is registered, keyed at the block
    # size; there is no sub-block entry to hit.
    assert len(producer.block_hashes) == 1
    assert manager.block_pool.get_cached_block(producer.block_hashes[0], kv_cache_group_ids=[0]) is not None


def test_single_group_env_on_stays_baseline_while_unsupported(monkeypatch) -> None:
    # Given the env opt-in set but the experimental support still disabled.
    monkeypatch.setenv(_ENV, "1")
    block_size = 8
    kv_cache_config = _make_single_mamba_config(block_size)
    vllm_config = _make_vllm_config(block_size=block_size, prefix_match_unit=2)

    # When block sizes are resolved and the manager is built.
    with patch.object(patch_kv_cache_utils.logger, "warning_once") as mock_warning:
        scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    assert (scheduler_block_size, hash_block_size) == (8, 8)
    monkeypatch.setattr(kcm, "get_kv_cache_coordinator", get_kv_cache_coordinator)
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )

    # Then the baseline unitary coordinator serves the ordinary block-aligned
    # path with no copy-on-write redirection, and the reason was surfaced.
    assert isinstance(manager.coordinator, UnitaryKVCacheCoordinator)
    assert manager.coordinator.enable_partial_hash_hits is False
    producer = _make_request("producer", list(range(block_size)), hash_block_size)
    blocks, computed, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(producer, block_size, computed, blocks) is not None
    copies, _ = manager.take_kv_cache_block_copies()
    assert copies == []
    mock_warning.assert_called_once()
