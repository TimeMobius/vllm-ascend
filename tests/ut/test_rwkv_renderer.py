from unittest.mock import MagicMock, patch

from vllm_ascend.renderers.rwkv import RWKVRenderer


def test_rwkv_renderer_declares_rwkv_stop_tokens():
    assert RWKVRenderer._DEFAULT_STOP_TOKENS == ("<|im_end|>", "<|endoftext|>")


class MockTokenizer:
    def __init__(self) -> None:
        self._token_to_id = {"<|im_end|>": 1, "<|endoftext|>": 2, "<bos>": 0}

    def convert_tokens_to_ids(self, token: str) -> int | list[int]:
        return self._token_to_id.get(token, -1)

    def apply_chat_template(self, conversation: list, **kwargs) -> str | list[int]:
        return "rendered_prompt"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)


class MockModelConfig:
    def __init__(self) -> None:
        self.hf_config = MagicMock()


class MockVllmConfig:
    def __init__(self) -> None:
        self.model_config = MockModelConfig()


def test_rwkv_renderer_construction():
    mock_config = MockVllmConfig()
    mock_tokenizer = MockTokenizer()
    renderer = RWKVRenderer(mock_config, mock_tokenizer)
    assert renderer.get_tokenizer() is mock_tokenizer


def test_rwkv_renderer_get_generation_config_fields_sets_eos_from_stop_tokens():
    mock_tokenizer = MockTokenizer()
    mock_config = MockVllmConfig()
    renderer = RWKVRenderer(mock_config, mock_tokenizer)
    fields = renderer.get_generation_config_fields({})
    assert "eos_token_id" in fields
    assert 1 in fields["eos_token_id"]
    assert 2 in fields["eos_token_id"]


def test_rwkv_renderer_get_generation_config_fields_no_tokenizer():
    mock_config = MockVllmConfig()
    renderer = RWKVRenderer(mock_config, None)
    fields = renderer.get_generation_config_fields({"existing": "value"})
    assert fields == {"existing": "value"}


class MockChatParams:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def get_apply_chat_template_kwargs(self) -> dict:
        return self._kwargs


def test_rwkv_renderer_render_messages_forwards_chat_template_kwargs():
    mock_tokenizer = MockTokenizer()
    mock_config = MockVllmConfig()
    renderer = RWKVRenderer(mock_config, mock_tokenizer)
    params = MockChatParams(
        add_generation_prompt=True,
        media_io_kwargs={"media_type": "image"},
        mm_processor_kwargs={"min_pixels": 256},
    )
    messages = [{"role": "user", "content": "hello"}]
    with patch.object(renderer, "get_tokenizer") as mock_get_tok:
        mock_get_tok.return_value = mock_tokenizer
        conversation, prompt = renderer.render_messages(messages, params)
    assert isinstance(conversation, list)
    assert isinstance(prompt, dict)


def test_rwkv_renderer_stops_without_tokenizer():
    mock_config = MockVllmConfig()
    renderer = RWKVRenderer(mock_config, None)
    assert renderer.get_tokenizer() is None
