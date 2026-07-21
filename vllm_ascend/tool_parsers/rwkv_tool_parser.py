from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser


class RWKVToolParser(ToolParser):
    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._invoke = re.compile(r"<invoke\s+name\s*=\s*(['\"])(.*?)\1\s*>(.*?)</invoke>", re.DOTALL)
        self._parameter = re.compile(r"<parameter\s+name\s*=\s*(['\"])(.*?)\1\s*>(.*?)</parameter>", re.DOTALL)
        self._tool = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.current_tool_index = 0
        self.emitted_content_len = 0

    @staticmethod
    def _value(raw: str, schema: dict[str, Any] | None) -> Any:
        value = raw.strip()
        types = schema.get("type", []) if isinstance(schema, dict) else []
        types = [types] if isinstance(types, str) else types
        if "integer" in types:
            try:
                return int(value)
            except ValueError:
                pass
        if "number" in types:
            try:
                return float(value)
            except ValueError:
                pass
        if "boolean" in types and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse(self, text: str, request: ChatCompletionRequest) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for block in self._tool.finditer(text):
            for invoke in self._invoke.finditer(block.group(1)):
                name = invoke.group(2).strip()
                arguments: dict[str, Any] = {}
                for parameter in self._parameter.finditer(invoke.group(3)):
                    key = parameter.group(2).strip()
                    schema = None
                    for tool in request.tools or []:
                        function = getattr(tool, "function", None)
                        if getattr(function, "name", None) == name:
                            parameters = getattr(function, "parameters", None)
                            if isinstance(parameters, dict):
                                properties = parameters.get("properties", {})
                                schema = properties.get(key) if isinstance(properties, dict) else None
                    arguments[key] = self._value(parameter.group(3), schema)
                calls.append(ToolCall(type="function", function=FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))))
        return calls

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        calls = self._parse(model_output, request)
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
        start = model_output.find(self.tool_call_start_token)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=calls,
            content=model_output[:start] or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_token_ids, current_token_ids, delta_token_ids
        if not previous_text:
            self.current_tool_index = 0
            self.emitted_content_len = 0
        start = current_text.find(self.tool_call_start_token)
        if start < 0:
            content = delta_text
            return DeltaMessage(content=content) if content else None
        calls = self._parse(current_text, request)
        deltas: list[DeltaToolCall] = []
        for index, call in enumerate(calls[self.current_tool_index :], self.current_tool_index):
            self.current_tool_index = index + 1
            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=make_tool_call_id(id_type="random", func_name=call.function.name, idx=index),
                    type="function",
                    function=DeltaFunctionCall(name=call.function.name, arguments=call.function.arguments),
                )
            )
        content = current_text[:start] if not previous_text else None
        return DeltaMessage(content=content, tool_calls=deltas) if content or deltas else None
