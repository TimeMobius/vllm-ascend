from vllm_ascend.renderers.rwkv import RWKVRenderer


def test_rwkv_renderer_declares_rwkv_stop_tokens():
    assert RWKVRenderer._DEFAULT_STOP_TOKENS == ("<|im_end|>", "<|endoftext|>")
