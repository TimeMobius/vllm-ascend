from vllm_ascend import envs


def test_rwkv7_perf_flags_defaults():
    assert envs.RWKV7_USE_FUSED_MIX6 is False
    assert envs.RWKV7_USE_FUSED_KK_PRE is False
    assert envs.RWKV7_USE_FUSED_CMIX is False
    assert envs.RWKV7_USE_FUSED_RECURRENT_T1 is False
    assert envs.RWKV7_USE_FUSED_LNX_RKVRES_XG is True
    assert envs.RWKV7_USE_DIRECT_LINEAR is False
    assert envs.RWKV7_USE_ALT_RECURRENT_KERNEL is True
    assert envs.RWKV7_USE_ALT_RECURRENT_DECODE is True
    assert envs.RWKV7_USE_FUSED_BLOCK_NORMS is False
    assert envs.RWKV7_DISABLE_FUSED_PREFILL is False
    assert envs.RWKV7_DISABLE_FUSED_RECURRENT is False
