from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.tool_parsers.rwkv_tool_parser import RWKVToolParser

ToolParserManager.register_module(name="rwkv", module=RWKVToolParser)

__all__ = ["RWKVToolParser"]
