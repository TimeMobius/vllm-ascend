# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.models import rwkv7


def _make_rwkv_vllm_config(
    *,
    use_v2_model_runner: bool = False,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def _set_breakable(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr(
        rwkv7.envs_vllm,
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        enabled,
        raising=False,
    )


def test_rwkv7_experimental_support_constants_default_disabled() -> None:
    assert rwkv7._RWKV7_BREAKABLE_CUDAGRAPH_SUPPORTED is False
    assert rwkv7._RWKV7_MRV2_SUPPORTED is False


def test_rwkv7_default_config_is_accepted(monkeypatch) -> None:
    # Given default config: breakable off and the v1 model runner.
    _set_breakable(monkeypatch, False)
    config = _make_rwkv_vllm_config()

    # When the compile decision boundary is evaluated.
    should_compile = rwkv7._rwkv7_should_compile(config)

    # Then RWKV7 proceeds untouched on the default path.
    assert should_compile is True


def test_rwkv7_breakable_opt_in_fails_early_with_clear_error(monkeypatch) -> None:
    # Given VLLM_USE_BREAKABLE_CUDAGRAPH explicitly enabled for RWKV7.
    _set_breakable(monkeypatch, True)
    config = _make_rwkv_vllm_config()

    # When the compile decision boundary is evaluated.
    with pytest.raises(NotImplementedError, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
        rwkv7._rwkv7_should_compile(config)


def test_rwkv7_mrv2_opt_in_fails_early_with_distinct_error(monkeypatch) -> None:
    # Given the v2 model runner explicitly enabled for RWKV7.
    _set_breakable(monkeypatch, False)
    config = _make_rwkv_vllm_config(use_v2_model_runner=True)

    # When the compile decision boundary is evaluated.
    with pytest.raises(NotImplementedError, match="v2 model runner"):
        rwkv7._rwkv7_should_compile(config)


def test_rwkv7_breakable_opt_in_rejected_even_with_full_cudagraphs(monkeypatch) -> None:
    # Given breakable enabled while full cudagraphs would skip torch.compile.
    _set_breakable(monkeypatch, True)
    config = _make_rwkv_vllm_config(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)

    # When the compile decision boundary is evaluated.
    with pytest.raises(NotImplementedError, match="VLLM_USE_BREAKABLE_CUDAGRAPH"):
        rwkv7._rwkv7_should_compile(config)


def test_rwkv7_gates_can_be_enabled_by_internal_support_switch(monkeypatch) -> None:
    # Given internal support gates re-enable the retained integrations.
    _set_breakable(monkeypatch, True)
    monkeypatch.setattr(rwkv7, "_RWKV7_BREAKABLE_CUDAGRAPH_SUPPORTED", True)
    monkeypatch.setattr(rwkv7, "_RWKV7_MRV2_SUPPORTED", True)
    config = _make_rwkv_vllm_config(use_v2_model_runner=True)

    # When the compile decision boundary is evaluated.
    should_compile = rwkv7._rwkv7_should_compile(config)

    # Then validation stands down and the normal decision is returned.
    assert should_compile is True
