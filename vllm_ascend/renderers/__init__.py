from vllm.renderers.registry import RENDERER_REGISTRY

from vllm_ascend.renderers.rwkv import RWKVRenderer

# Idempotent: only register if not already present
if "rwkv" not in RENDERER_REGISTRY.renderers:
    RENDERER_REGISTRY.register("rwkv", "vllm_ascend.renderers.rwkv", "RWKVRenderer")

__all__ = ["RWKVRenderer"]
