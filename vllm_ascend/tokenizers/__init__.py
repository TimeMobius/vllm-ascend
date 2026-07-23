from vllm.tokenizers.registry import TokenizerRegistry

from vllm_ascend.tokenizers.rwkv import RWKVTokenizer

if "rwkv" not in TokenizerRegistry.tokenizers:
    TokenizerRegistry.register("rwkv", "vllm_ascend.tokenizers.rwkv", "RWKVTokenizer")

__all__ = ["RWKVTokenizer"]
