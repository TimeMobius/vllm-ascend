# SPDX-License-Identifier: Apache-2.0

"""Opt-in fine-grained prefix caching for a single-group Mamba "align" model.

Baseline: one KV cache group maps to ``UnitaryKVCacheCoordinator`` with
``hash_block_size == mamba block size``. The opt-in env var plus
``--prefix-match-unit`` exposes ``hash_block_size < block size`` and routes the
group through ``HybridKVCacheCoordinator`` so partial hash hits are enabled.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_manager as kcm
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator, UnitaryKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
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
    _single_group_mamba_fine_grained_hash_block_size,
)

_ENV = "VLLM_ASCEND_ENABLE_MAMBA_FINE_GRAINED_PREFIX_CACHE"
_SUPPORT_CONST = "_MAMBA_FINE_GRAINED_PREFIX_CACHE_WORKER_COPY_SUPPORTED"


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


@pytest.fixture
def fine_grained_supported(monkeypatch) -> None:
    """Re-enable the retained experimental path for internals-only tests.

    Production ships the fine-grained path gated off; tests that exercise its
    retained internals opt back in explicitly.
    """
    monkeypatch.setattr(patch_kv_cache_utils, _SUPPORT_CONST, True)


def _make_single_mamba_config(block_size: int = 16, mode: str = "align") -> KVCacheConfig:
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
                    mamba_cache_mode=mode,
                ),
            )
        ],
    )


def _make_vllm_config(
    *,
    enable_prefix_caching: bool = True,
    dcp: int = 1,
    block_size: int = 16,
    prefix_match_unit: int | None = None,
    kv_transfer_config: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="align",
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        kv_transfer_config=kv_transfer_config,
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


def test_single_mamba_opt_in_exposes_finer_hash_block_size(monkeypatch, fine_grained_supported) -> None:
    # Given an opt-in single Mamba "align" group with prefix_match_unit < block.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When block sizes are resolved.
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # Then the hash is finer than the scheduler/mamba block size.
    assert (scheduler_block_size, hash_block_size) == (16, 8)


def test_single_mamba_baseline_is_unchanged_when_env_off(monkeypatch) -> None:
    # Given the opt-in disabled (default) but prefix_match_unit set.
    monkeypatch.delenv(_ENV, raising=False)
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When block sizes are resolved.
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # Then the single-group baseline contract hash == block size holds.
    assert (scheduler_block_size, hash_block_size) == (16, 16)


@pytest.mark.parametrize(
    ("config_kwargs", "config_factory", "expected"),
    [
        pytest.param(
            {"prefix_match_unit": None},
            _make_single_mamba_config,
            (16, 16),
            id="no-prefix-match-unit",
        ),
        pytest.param(
            {"dcp": 2, "prefix_match_unit": 8},
            _make_single_mamba_config,
            (32, 32),
            id="context-parallel",
        ),
        pytest.param(
            {"prefix_match_unit": 8, "kv_transfer_config": object()},
            _make_single_mamba_config,
            (16, 16),
            id="kv-connector",
        ),
        pytest.param(
            {"prefix_match_unit": 8},
            lambda: _make_single_mamba_config(mode="none"),
            (16, 16),
            id="mamba-mode-none",
        ),
        pytest.param(
            {"prefix_match_unit": 8, "enable_prefix_caching": False},
            _make_single_mamba_config,
            (16, 16),
            id="prefix-caching-off",
        ),
    ],
)
def test_single_mamba_opt_in_guards_fall_back_to_baseline(
    monkeypatch, fine_grained_supported, config_kwargs, config_factory, expected
) -> None:
    # Given the opt-in enabled but a guard condition unmet.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = config_factory()
    vllm_config = _make_vllm_config(**config_kwargs)

    # When block sizes are resolved.
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # Then the baseline behavior is preserved.
    assert (scheduler_block_size, hash_block_size) == expected


def test_single_mamba_opt_in_requires_mamba_block_size_match(monkeypatch, fine_grained_supported) -> None:
    # Given a decoupled --mamba-block-size that would break coordinator divisibility.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)
    vllm_config.cache_config.block_size = 32

    # When resolving.
    _, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # Then the opt-in is refused (hash stays equal to the scheduler block size).
    assert hash_block_size == 32


def test_single_mamba_opt_in_routes_to_hybrid_coordinator(monkeypatch, fine_grained_supported) -> None:
    # Given a finer hash for a lone Mamba "align" group.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)
    _, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # When the coordinator is selected.
    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=8192,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=hash_block_size,
        scheduler_block_size=16,
    )

    # Then it is the hybrid coordinator with partial hash hits enabled.
    assert isinstance(coordinator, HybridKVCacheCoordinator)
    assert coordinator.enable_partial_hash_hits


def test_single_mamba_baseline_routes_to_unitary_coordinator(monkeypatch) -> None:
    # Given the opt-in disabled.
    monkeypatch.delenv(_ENV, raising=False)
    kv_cache_config = _make_single_mamba_config(block_size=16)

    # When the coordinator is selected with the baseline equal block sizes.
    coordinator = get_kv_cache_coordinator(
        kv_cache_config,
        max_model_len=8192,
        max_num_batched_tokens=1024,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=16,
        scheduler_block_size=16,
    )

    # Then the single-group coordinator is retained.
    assert isinstance(coordinator, UnitaryKVCacheCoordinator)


def test_single_mamba_opt_in_serves_partial_hash_hit(monkeypatch, fine_grained_supported) -> None:
    # Given an opt-in manager whose block is finer than the hash unit.
    monkeypatch.setenv(_ENV, "1")
    block_size = 8
    prefix_match_unit = 2
    kv_cache_config = _make_single_mamba_config(block_size=block_size)
    vllm_config = _make_vllm_config(block_size=block_size, prefix_match_unit=prefix_match_unit)
    scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    monkeypatch.setattr(kcm, "get_kv_cache_coordinator", get_kv_cache_coordinator)
    manager = KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_caching=True,
    )
    assert manager.coordinator.enable_partial_hash_hits

    # When a request of 6 tokens is prefilled and a sibling replays it.
    producer = _make_request("producer", [0, 0, 1, 1, 2, 2], hash_block_size)
    producer_blocks, producer_computed, _ = manager.get_computed_blocks(producer)
    assert producer_computed == 0
    assert manager.allocate_slots(producer, 6, producer_computed, producer_blocks) is not None
    manager.free(producer)
    manager.new_step_starts()

    consumer = _make_request("consumer", [0, 0, 1, 1, 2, 2, 3, 3], hash_block_size)
    _, consumer_computed, _ = manager.get_computed_blocks(consumer)

    # Then the sibling hits inside the physical block, at the hash boundary.
    assert consumer_computed == 6
    assert consumer_computed < block_size


def test_single_group_helper_returns_none_without_opt_in(monkeypatch) -> None:
    # Given the env disabled.
    monkeypatch.delenv(_ENV, raising=False)
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When the helper is queried directly.
    result = _single_group_mamba_fine_grained_hash_block_size(kv_cache_config, vllm_config)

    # Then no opt-in hash is produced.
    assert result is None


def test_single_group_support_constant_defaults_disabled() -> None:
    # Given/Then: production ships the retained path disabled by default.
    assert patch_kv_cache_utils._MAMBA_FINE_GRAINED_PREFIX_CACHE_WORKER_COPY_SUPPORTED is False


def test_single_mamba_env_on_falls_back_when_unsupported(monkeypatch) -> None:
    # Given the env opt-in set while the code path is still unsupported.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When block sizes are resolved.
    with patch.object(patch_kv_cache_utils.logger, "warning_once") as mock_warning:
        scheduler_block_size, hash_block_size = _ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    # Then the baseline block-aligned geometry remains.
    assert (scheduler_block_size, hash_block_size) == (16, 16)
    # And one clear warning names the unsupported reason.
    mock_warning.assert_called_once()
    assert "list-valued recurrent state" in mock_warning.call_args.args[0]


def test_single_group_helper_falls_back_and_warns_when_unsupported(monkeypatch) -> None:
    # Given the env opt-in set while the code path is still unsupported.
    monkeypatch.setenv(_ENV, "1")
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When the helper is queried directly.
    with patch.object(patch_kv_cache_utils.logger, "warning_once") as mock_warning:
        result = _single_group_mamba_fine_grained_hash_block_size(kv_cache_config, vllm_config)

    # Then no finer hash is produced and the reason is surfaced once.
    assert result is None
    mock_warning.assert_called_once()


def test_single_mamba_env_off_emits_no_unsupported_warning(monkeypatch) -> None:
    # Given the env opt-in off.
    monkeypatch.delenv(_ENV, raising=False)
    kv_cache_config = _make_single_mamba_config(block_size=16)
    vllm_config = _make_vllm_config(prefix_match_unit=8)

    # When resolving.
    with patch.object(patch_kv_cache_utils.logger, "warning_once") as mock_warning:
        result = _single_group_mamba_fine_grained_hash_block_size(kv_cache_config, vllm_config)

    # Then the env gate short-circuits before any unsupported-path warning.
    assert result is None
    mock_warning.assert_not_called()
