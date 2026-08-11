from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vllm.renderers import ChatParams

from vllm_ascend.renderers.rwkv import RWKVRenderer
from vllm_ascend.tokenizers.rwkv import RWKVTokenizer


def _write_vocab(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "1 'S' 1",
                "2 'y' 1",
                "3 's' 1",
                "4 't' 1",
                "5 'e' 1",
                "6 'm' 1",
                "7 ':' 1",
                "8 ' ' 1",
                "9 'h' 1",
                "10 'i' 1",
                "11 '\\n\\n' 2",
                "12 'U' 1",
                "13 'r' 1",
                "14 '' 13",
                "15 '' 12",
                "16 '' 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@dataclass
class MockHFConfig:
    model_type: str = "rwkv7"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = "native-rwkv7"
    tokenizer: str = "native-rwkv7"
    trust_remote_code: bool = False
    max_model_len: int = 128
    tokenizer_revision = None
    tokenizer_mode = "rwkv"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    multimodal_config = None
    allowed_local_media_path = None
    allowed_media_domains = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def test_rwkv_renderer_declares_rwkv_stop_tokens() -> None:
    """Default stop set must be ChatML control tokens, not legacy newlines."""
    assert RWKVRenderer._DEFAULT_STOP_TOKENS == ("<|im_end|>", "<|endoftext|>")


def test_rwkv_renderer_renders_chat_messages(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path))
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(tokenizer=str(vocab_path)),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    conversation, prompt = renderer.render_messages(
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "hi"},
        ],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )

    assert [message["role"] for message in conversation] == ["system", "user"]
    assert "prompt" in prompt
    assert prompt["prompt"].startswith("System: hi\n\nUser: hi\n\nAssistant:")


def test_rwkv_renderer_normalizes_tool_history_before_template() -> None:
    class CaptureTokenizer:
        bos_token_id = None
        eos_token_id = None
        chat_template = "template"

        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            self.messages = messages
            return "prompt"

        def encode(self, text: str, **kwargs):
            del kwargs
            return [ord(char) for char in text]

    tokenizer = CaptureTokenizer()
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    conversation, prompt = renderer.render_messages(
        [
            {"role": "user", "content": "北京天气怎么样"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query": "北京天气"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "今天白天有雷阵雨，夜晚有小雨。",
            },
        ],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )

    assert prompt["prompt"] == "prompt"
    assert tokenizer.messages == conversation
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {"query": "北京天气"}
    assert conversation[2]["name"] == "search_web"


def test_rwkv_renderer_overrides_generation_eos_tokens(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path))
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(tokenizer=str(vocab_path)),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=tokenizer,
    )

    generation_config = renderer.get_generation_config_fields({"eos_token_id": 2})

    # <|im_end|> = id 16, <|endoftext|> = id 14 in this vocab.
    assert generation_config["eos_token_id"] == [16, 14]


def test_rwkv_renderer_construction_with_mock_tokenizer() -> None:
    class _MockTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<|im_end|>": 16, "<|endoftext|>": 14}.get(token, -1)

    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=_MockTok(),
    )

    fields = renderer.get_generation_config_fields({})
    assert fields["eos_token_id"] == [16, 14]


def test_rwkv_renderer_get_generation_config_fields_no_tokenizer() -> None:
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=None,
    )
    fields = renderer.get_generation_config_fields({"existing": "value"})
    assert fields == {"existing": "value"}


class MockChatParams:
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        return self._kwargs


def test_rwkv_renderer_render_messages_forwards_chat_template_kwargs() -> None:
    class _MockTok:
        def apply_chat_template(self, conversation, **kwargs):
            self.kwargs = kwargs
            return "rendered_prompt"

    mock_tokenizer = _MockTok()
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=mock_tokenizer,
    )
    params = MockChatParams(add_generation_prompt=True)

    with patch.object(renderer, "get_tokenizer", return_value=mock_tokenizer):
        renderer.render_messages([], params)

    assert mock_tokenizer.kwargs["add_generation_prompt"] is True


def test_rwkv_renderer_get_tokenizer_raises_when_tokenizer_absent() -> None:
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=None,
    )
    with pytest.raises(ValueError):
        renderer.get_tokenizer()


def test_rwkv_renderer_does_not_use_legacy_newline_stops() -> None:
    """Regression: legacy ``("\\n\\n", "")`` would silently swallow the bug."""
    _ = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=MagicMock(),
    )
    stops = RWKVRenderer._DEFAULT_STOP_TOKENS
    assert "\n\n" not in stops
    assert "" not in stops
    assert "<|im_end|>" in stops
    assert "<|endoftext|>" in stops


def test_rwkv_renderer_get_eos_token_id_prefers_default_stop_set() -> None:
    """``get_eos_token_id`` must yield the first ChatML control token id.

    ``vllm.v1.engine.input_processor`` only forwards one int from the
    renderer; pairing it with the model's ``generation_config.eos_token_id``
    in ``update_from_generation_config`` is what gets both ``<|im_end|>``
    and ``<|endoftext|>`` registered as stops.
    """
    class _MockTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {
                "<|im_end|>": 65530,
                "<|endoftext|>": 65532,
            }[token]

        @property
        def eos_token_id(self) -> int:
            return 65532

    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=_MockTok(),
    )

    assert renderer.get_eos_token_id() == 65530


def test_rwkv_renderer_get_eos_token_id_no_tokenizer() -> None:
    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=None,
    )
    assert renderer.get_eos_token_id() is None


def test_rwkv_renderer_get_eos_token_id_falls_back_to_tokenizer_eos() -> None:
    """If neither ChatML token resolves, fall back to ``tokenizer.eos_token_id``."""

    class _MockTok:
        def convert_tokens_to_ids(self, token: str) -> int:
            # Simulate a vocab where ChatML markers are absent.
            return -1

        @property
        def eos_token_id(self) -> int:
            return 42

    renderer = RWKVRenderer(
        MockVllmConfig(
            MockModelConfig(),
            parallel_config=MockParallelConfig(),
        ),
        tokenizer=_MockTok(),
    )

    assert renderer.get_eos_token_id() == 42
