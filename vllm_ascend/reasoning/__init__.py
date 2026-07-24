from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

from vllm_ascend.reasoning.rwkv_reasoning_parser import RWKVReasoningParser

try:
    ReasoningParserManager.register_module(name="rwkv", module=RWKVReasoningParser, force=False)
except KeyError:
    # Module already registered (e.g., during importlib.reload()); ignore idempotently.
    pass

__all__ = ["RWKVReasoningParser"]
