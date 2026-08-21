import pytest

from vllm_ascend import envs
from vllm_ascend.envs import RWKV7RecurrentBackend


def test_rwkv7_perf_flags_defaults():
    assert envs.RWKV7_USE_FUSED_MIX6 is False
    assert envs.RWKV7_USE_FUSED_KK_PRE is False
    assert envs.RWKV7_USE_FUSED_CMIX is False
    assert envs.RWKV7_USE_FUSED_LNX_RKVRES_XG is False
    assert envs.RWKV7_USE_DIRECT_LINEAR is False
    assert envs.RWKV7_USE_FUSED_BLOCK_NORMS is False
    assert envs.RWKV7_DISABLE_FUSED_PREFILL is False
    assert envs.RWKV7_DISABLE_FUSED_RECURRENT is False


def test_rwkv7_recurrent_backend_defaults_to_auto():
    assert envs.VLLM_ASCEND_RWKV7_RECURRENT_BACKEND is RWKV7RecurrentBackend.AUTO


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
    # auto: AscendC recurrent via its runtime guard; never Triton T1/cache.
    auto = RWKV7RecurrentBackend.AUTO
    assert auto.uses_ascendc is True
    assert auto.uses_triton_t1 is False
    assert auto.uses_persistent_cache is False

    # reference: no accelerated recurrent path.
    reference = RWKV7RecurrentBackend.REFERENCE
    assert reference.uses_ascendc is False
    assert reference.uses_triton_t1 is False
    assert reference.uses_persistent_cache is False

    # ascendc: AscendC recurrent only.
    ascendc = RWKV7RecurrentBackend.ASCENDC
    assert ascendc.uses_ascendc is True
    assert ascendc.uses_triton_t1 is False
    assert ascendc.uses_persistent_cache is False

    # triton_t1: non-cache Triton T1 decode only.
    triton_t1 = RWKV7RecurrentBackend.TRITON_T1
    assert triton_t1.uses_ascendc is False
    assert triton_t1.uses_triton_t1 is True
    assert triton_t1.uses_persistent_cache is False

    # triton_t1_cache: persistent-cache Triton T1, falling back to non-cache T1.
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
    ):
        assert flag not in envs.env_variables
        with pytest.raises(AttributeError):
            getattr(envs, flag)
