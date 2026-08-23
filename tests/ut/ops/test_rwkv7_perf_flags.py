import pytest

from vllm_ascend import envs
from vllm_ascend.envs import (
    RWKV7OperatorOverride,
    RWKV7Preset,
    RWKV7RecurrentBackend,
)
from vllm_ascend.rwkv7_config import resolve_rwkv7_config


def test_rwkv7_preset_defaults_to_auto():
    assert envs.VLLM_ASCEND_RWKV7_PRESET is RWKV7Preset.AUTO


def test_rwkv7_observability_defaults_to_off():
    assert envs.VLLM_ASCEND_RWKV7_OBSERVABILITY is envs.RWKV7Observability.OFF


def test_rwkv7_recurrent_backend_defaults_to_auto():
    assert envs.VLLM_ASCEND_RWKV7_RECURRENT_BACKEND is RWKV7RecurrentBackend.AUTO


def test_rwkv7_config_default_allows_all_operators():
    config = resolve_rwkv7_config()
    assert config.operator == {
        "mix6": True,
        "kk_pre": True,
        "epilogue": True,
        "block_norms": True,
    }
    assert config.recurrent_backend is RWKV7RecurrentBackend.AUTO


@pytest.mark.parametrize(
    ("preset", "recurrent", "operator"),
    [
        ("reference", RWKV7RecurrentBackend.REFERENCE, False),
        ("auto", RWKV7RecurrentBackend.AUTO, True),
        (
            "throughput",
            RWKV7RecurrentBackend.TRITON_T1_CACHE,
            True,
        ),
    ],
)
def test_rwkv7_preset_expansion(monkeypatch, preset, recurrent, operator):
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_PRESET", preset)
    monkeypatch.delenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES", raising=False)
    config = resolve_rwkv7_config()
    assert config.preset.value == preset
    assert config.recurrent_backend is recurrent
    assert all(v is operator for v in config.operator.values())


def test_rwkv7_recurrent_backend_overrides_preset(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_PRESET", "throughput")
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", "reference")
    monkeypatch.delenv("VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES", raising=False)
    config = resolve_rwkv7_config()
    assert config.recurrent_backend is RWKV7RecurrentBackend.REFERENCE


def test_rwkv7_operator_override_beats_preset(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_PRESET", "auto")
    monkeypatch.delenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", raising=False)
    monkeypatch.setenv(
        "VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES",
        '{"mix6":"reference","epilogue":"reference"}',
    )
    config = resolve_rwkv7_config()
    assert config.operator["mix6"] is False
    assert config.operator["epilogue"] is False
    assert config.operator["kk_pre"] is True
    assert config.operator["block_norms"] is True


def test_rwkv7_recurrent_key_in_override_highest_precedence(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_PRESET", "throughput")
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", "ascendc")
    monkeypatch.setenv(
        "VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES", '{"recurrent":"triton_t1"}'
    )
    config = resolve_rwkv7_config()
    assert config.recurrent_backend is RWKV7RecurrentBackend.TRITON_T1


def test_rwkv7_operator_override_disable_triton_flag_remains():
    assert envs.VLLM_ASCEND_RWKV7_DISABLE_TRITON == 0


def test_rwkv7_operator_override_enum_values():
    assert {member.value for member in RWKV7OperatorOverride} == {
        "triton",
        "reference",
        "auto",
    }


def test_rwkv7_recurrent_backend_is_typed_enum():
    backend = envs.VLLM_ASCEND_RWKV7_RECURRENT_BACKEND
    assert isinstance(backend, RWKV7RecurrentBackend)
    assert {member.value for member in RWKV7RecurrentBackend} == {
        "auto",
        "reference",
        "ascendc",
        "triton_t1",
        "triton_t1_cache",
    }


def test_rwkv7_recurrent_backend_valid_parsing(monkeypatch):
    expected = {
        "auto": RWKV7RecurrentBackend.AUTO,
        "reference": RWKV7RecurrentBackend.REFERENCE,
        "ascendc": RWKV7RecurrentBackend.ASCENDC,
        "triton_t1": RWKV7RecurrentBackend.TRITON_T1,
        "triton_t1_cache": RWKV7RecurrentBackend.TRITON_T1_CACHE,
    }
    for raw, member in expected.items():
        monkeypatch.setenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", raw)
        assert envs.VLLM_ASCEND_RWKV7_RECURRENT_BACKEND is member


@pytest.mark.parametrize(
    "invalid",
    [
        "triton",
        "ascend",
        "t1",
        "auto_cache",
        "0",
        "",
        "REFERENCE",
    ],
)
def test_rwkv7_recurrent_backend_invalid_value_raises(monkeypatch, invalid):
    monkeypatch.setenv("VLLM_ASCEND_RWKV7_RECURRENT_BACKEND", invalid)
    with pytest.raises(ValueError, match="VLLM_ASCEND_RWKV7_RECURRENT_BACKEND"):
        _ = envs.VLLM_ASCEND_RWKV7_RECURRENT_BACKEND


def test_rwkv7_recurrent_backend_capabilities():
    auto = RWKV7RecurrentBackend.AUTO
    assert auto.uses_ascendc is True
    assert auto.uses_triton_t1 is False
    assert auto.uses_persistent_cache is False

    reference = RWKV7RecurrentBackend.REFERENCE
    assert reference.uses_ascendc is False
    assert reference.uses_triton_t1 is False
    assert reference.uses_persistent_cache is False

    ascendc = RWKV7RecurrentBackend.ASCENDC
    assert ascendc.uses_ascendc is True
    assert ascendc.uses_triton_t1 is False
    assert ascendc.uses_persistent_cache is False

    triton_t1 = RWKV7RecurrentBackend.TRITON_T1
    assert triton_t1.uses_ascendc is False
    assert triton_t1.uses_triton_t1 is True
    assert triton_t1.uses_persistent_cache is False

    triton_t1_cache = RWKV7RecurrentBackend.TRITON_T1_CACHE
    assert triton_t1_cache.uses_ascendc is False
    assert triton_t1_cache.uses_triton_t1 is True
    assert triton_t1_cache.uses_persistent_cache is True


def test_rwkv7_removed_boolean_flags_are_gone():
    for flag in (
        "RWKV7_USE_ALT_RECURRENT_KERNEL",
        "RWKV7_USE_ALT_RECURRENT_DECODE",
        "RWKV7_USE_FUSED_RECURRENT_T1",
        "RWKV7_USE_FUSED_RECURRENT_CACHE_T1",
        "RWKV7_USE_FUSED_MIX6",
        "RWKV7_USE_FUSED_KK_PRE",
        "RWKV7_USE_FUSED_LNX_RKVRES_XG",
        "RWKV7_USE_FUSED_BLOCK_NORMS",
        "RWKV7_DISABLE_FUSED_PREFILL",
        "RWKV7_DISABLE_FUSED_RECURRENT",
        "VLLM_ASCEND_RWKV7_PROFILE",
    ):
        assert flag not in envs.env_variables
        with pytest.raises(AttributeError):
            getattr(envs, flag)
