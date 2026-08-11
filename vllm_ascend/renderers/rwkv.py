# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Renderer for RWKV chat models on vLLM Ascend.

Aligned with the upstream vLLM ``vllm/renderers/rwkv.py`` so that the
generation EOS set is rebuilt from ChatML control tokens (``<|im_end|>`` /
``<|endoftext|>``) instead of legacy ``"\\n\\n"`` / empty-string stops.
"""

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
    _DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")

    @staticmethod
    def _fill_tool_message_names(conversation: list[ConversationMessage]) -> None:
        tool_call_names: dict[str, str] = {}
        for message in conversation:
            if message["role"] == "assistant":
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue

                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_call_id = tool_call.get("id")
                    function = tool_call.get("function")
                    if (
                        isinstance(tool_call_id, str)
                        and isinstance(function, dict)
                        and isinstance(function.get("name"), str)
                    ):
                        tool_call_names[tool_call_id] = function["name"]
            elif message["role"] == "tool" and not message.get("name"):
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in tool_call_names:
                    message["name"] = tool_call_names[tool_call_id]

    @classmethod
    def from_config(
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "RWKVRenderer":
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=RWKVTokenizer,
                **tokenizer_kwargs,
            )

        return cls(config, tokenizer)

    def get_generation_config_fields(
        self,
        generation_config_fields: dict[str, Any],
    ) -> dict[str, Any]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return generation_config_fields

        stop_token_ids: list[int] = []
        for token in self._DEFAULT_STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id not in stop_token_ids:
                stop_token_ids.append(token_id)

        if not stop_token_ids:
            return generation_config_fields

        updated_generation_config = dict(generation_config_fields)
        updated_generation_config["eos_token_id"] = stop_token_ids
        return updated_generation_config

    def get_eos_token_id(self) -> int | None:
        """Pick the first ``_DEFAULT_STOP_TOKENS`` id as the primary EOS.

        ``update_from_generation_config`` adds this single id to
        ``_all_stop_token_ids`` and then folds the model's
        ``generation_config["eos_token_id"]`` in (after discarding the
        renderer's id). By picking the first ChatML control token here we
        guarantee both ``<|im_end|>`` and ``<|endoftext|>`` end up as
        generation stops regardless of which one the model's
        ``generation_config.json`` declares.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            return None

        for token in self._DEFAULT_STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                return token_id

        return tokenizer.eos_token_id

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        self._fill_tool_message_names(conversation)

        prompt_raw = tokenizer.apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        self._fill_tool_message_names(conversation)

        prompt_raw = tokenizer.apply_chat_template(
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
