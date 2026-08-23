import pytest

from vllm_ascend.envs import (
    RWKV7Observability,
    RWKV7Preset,
    RWKV7RecurrentBackend,
)
from vllm_ascend.rwkv7_config import resolve_rwkv7_config

PRESET = "VLLM_ASCEND_RWKV7_PRESET"
RECURRENT = "VLLM_ASCEND_RWKV7_RECURRENT_BACKEND"
OBSERV = "VLLM_ASCEND_RWKV7_OBSERVABILITY"
OVERRIDES = "VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (PRESET, RECURRENT, OBSERV, OVERRIDES):
        monkeypatch.delenv(var, raising=False)


def test_default_config():
    config = resolve_rwkv7_config()
    assert config.preset is RWKV7Preset.AUTO
    assert config.recurrent_backend is RWKV7RecurrentBackend.AUTO
    assert config.operator == {
        "mix6": True,
        "kk_pre": True,
        "epilogue": True,
        "block_norms": True,
    }
    assert config.observability is RWKV7Observability.OFF


@pytest.mark.parametrize(
    ("preset", "recurrent", "expected_operator"),
    [
        ("reference", RWKV7RecurrentBackend.REFERENCE, False),
        ("auto", RWKV7RecurrentBackend.AUTO, True),
        ("throughput", RWKV7RecurrentBackend.TRITON_T1_CACHE, True),
    ],
)
def test_preset_expansion(monkeypatch, preset, recurrent, expected_operator):
    monkeypatch.setenv(PRESET, preset)
    config = resolve_rwkv7_config()
    assert config.preset.value == preset
    assert config.recurrent_backend is recurrent
    assert all(v is expected_operator for v in config.operator.values())


def test_recurrent_backend_overrides_preset(monkeypatch):
    monkeypatch.setenv(PRESET, "throughput")
    monkeypatch.setenv(RECURRENT, "reference")
    config = resolve_rwkv7_config()
    assert config.recurrent_backend is RWKV7RecurrentBackend.REFERENCE
    # Operators still from the throughput preset.
    assert all(config.operator.values())


def test_operator_override_beats_preset(monkeypatch):
    monkeypatch.setenv(PRESET, "auto")
    monkeypatch.setenv(OVERRIDES, '{"mix6":"reference","epilogue":"auto"}')
    config = resolve_rwkv7_config()
    assert config.operator["mix6"] is False
    assert config.operator["epilogue"] is True
    assert config.operator["kk_pre"] is True
    assert config.operator["block_norms"] is True


def test_recurrent_key_in_override_is_highest_precedence(monkeypatch):
    monkeypatch.setenv(PRESET, "auto")
    monkeypatch.setenv(RECURRENT, "ascendc")
    monkeypatch.setenv(OVERRIDES, '{"recurrent":"triton_t1_cache"}')
    config = resolve_rwkv7_config()
    assert config.recurrent_backend is RWKV7RecurrentBackend.TRITON_T1_CACHE


def test_observability_summary(monkeypatch):
    monkeypatch.setenv(OBSERV, "summary")
    config = resolve_rwkv7_config()
    assert config.observability is RWKV7Observability.SUMMARY
    assert config.observability_enabled() is True


def test_observability_off_enabled_false(monkeypatch):
    config = resolve_rwkv7_config()
    assert config.observability_enabled() is False


def test_recurrent_enabled_false_for_reference(monkeypatch):
    monkeypatch.setenv(RECURRENT, "reference")
    config = resolve_rwkv7_config()
    assert config.recurrent_enabled() is False


def test_recurrent_enabled_true_for_non_reference(monkeypatch):
    monkeypatch.setenv(RECURRENT, "triton_t1")
    config = resolve_rwkv7_config()
    assert config.recurrent_enabled() is True


def test_empty_object_override_is_noop(monkeypatch):
    monkeypatch.setenv(PRESET, "throughput")
    monkeypatch.setenv(OVERRIDES, "{}")
    config = resolve_rwkv7_config()
    assert config.recurrent_backend is RWKV7RecurrentBackend.TRITON_T1_CACHE
    assert all(config.operator.values())


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "null",
        '{"unknown":"triton"}',
        '{"mix6":"bogus"}',
        '{"mix6":1}',
        '{"recurrent":"bogus"}',
    ],
)
def test_invalid_operator_overrides_raises(monkeypatch, raw):
    monkeypatch.setenv(OVERRIDES, raw)
    with pytest.raises(ValueError, match=OVERRIDES):
        resolve_rwkv7_config()


@pytest.mark.parametrize("preset", ["bogus", "THROUGHPUT", "1"])
def test_invalid_preset_raises(monkeypatch, preset):
    monkeypatch.setenv(PRESET, preset)
    with pytest.raises(ValueError, match=PRESET):
        resolve_rwkv7_config()


@pytest.mark.parametrize("observ", ["bogus", "1", "SUMMARY"])
def test_invalid_observability_raises(monkeypatch, observ):
    monkeypatch.setenv(OBSERV, observ)
    with pytest.raises(ValueError, match=OBSERV):
        resolve_rwkv7_config()
