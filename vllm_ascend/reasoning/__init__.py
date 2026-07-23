from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

from vllm_ascend.reasoning.rwkv_reasoning_parser import RWKVReasoningParser

ReasoningParserManager.register_module(name="rwkv", module=RWKVReasoningParser, force=False)

__all__ = ["RWKVReasoningParser"]
