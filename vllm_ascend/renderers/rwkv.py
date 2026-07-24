from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.base import BaseRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.tokenizers import cached_get_tokenizer

from vllm_ascend.tokenizers.rwkv import RWKVTokenizer


class RWKVRenderer(BaseRenderer[RWKVTokenizer]):
    _DEFAULT_STOP_TOKENS = ("\n\n", "")

    @classmethod
    def from_config(
        cls, config: VllmConfig, tokenizer_kwargs: dict[str, Any]
    ) -> RWKVRenderer:
        tokenizer = None
        if not config.model_config.skip_tokenizer_init:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=RWKVTokenizer, **tokenizer_kwargs
            )
        return cls(config, tokenizer)

    def get_generation_config_fields(
        self, generation_config_fields: dict[str, Any]
    ) -> dict[str, Any]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return generation_config_fields
        stop_ids: list[int] = []
        for token in self._DEFAULT_STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id not in stop_ids:
                stop_ids.append(token_id)
        return {
            **generation_config_fields,
            "eos_token_id": stop_ids,
        } if stop_ids else generation_config_fields

    @staticmethod
    def _fill_tool_message_names(conversation: list[ConversationMessage]) -> None:
        names: dict[str, str] = {}
        for message in conversation:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []) or []:
                    if isinstance(call, dict) and isinstance(call.get("function"), dict):
                        function = call["function"]
                        if isinstance(call.get("id"), str) and isinstance(function.get("name"), str):
                            names[call["id"]] = function["name"]
            elif message["role"] == "tool" and not message.get("name"):
                call_id = message.get("tool_call_id")
                if isinstance(call_id, str) and call_id in names:
                    message["name"] = names[call_id]

    def render_messages(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages, self.model_config, content_format="string"
        )
        self._fill_tool_message_names(conversation)
        rendered = self.get_tokenizer().apply_chat_template(
            conversation, **params.get_apply_chat_template_kwargs()
        )
        prompt = parse_dec_only_prompt(rendered)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return conversation, prompt

    async def render_messages_async(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages, self.model_config, content_format="string"
        )
        self._fill_tool_message_names(conversation)
        rendered = self.get_tokenizer().apply_chat_template(
            conversation, **params.get_apply_chat_template_kwargs()
        )
        prompt = parse_dec_only_prompt(rendered)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return conversation, prompt
