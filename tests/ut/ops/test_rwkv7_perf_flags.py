from vllm_ascend import envs


def test_rwkv7_perf_flags_default_to_reference_paths():
    assert envs.RWKV7_USE_FUSED_MIX6 is False
    assert envs.RWKV7_USE_FUSED_KK_PRE is False
    assert envs.RWKV7_USE_FUSED_LNX_RKVRES_XG is True
    assert envs.RWKV7_USE_FUSED_CMIX is False
    assert envs.RWKV7_USE_DIRECT_LINEAR is False
    assert envs.RWKV7_USE_ALT_RECURRENT_KERNEL is False
    assert envs.RWKV7_DISABLE_FUSED_PREFILL is False
    assert envs.RWKV7_DISABLE_FUSED_RECURRENT is False
