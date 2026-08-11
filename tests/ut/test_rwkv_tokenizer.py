from __future__ import annotations

import json
from pathlib import Path

from vllm_ascend.tokenizers.rwkv import RWKVTokenizer

# ---------------------------------------------------------------------------
# Vocab fixtures
# ---------------------------------------------------------------------------


def _write_vocab(path: Path) -> None:
    """Minimal RWKV vocab that exercises the greedy trie."""
    path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 'b' 1",
                "3 'ab' 2",
                "4 '\\n\\n' 2",
                "5 '<|endoftext|>' 13",
                "6 '<|im_start|>' 12",
                "7 '<|im_end|>' 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_im_vocab(path: Path) -> None:
    """im-end / endoftext / think / tool_call vocab used by chat templates."""
    path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 'b' 1",
                "3 'ab' 2",
                "4 '\\n\\n' 2",
                "5 ' ' 1",
                "6 'A' 1",
                "7 's' 1",
                "8 'i' 1",
                "9 't' 1",
                "10 'n' 1",
                "11 ':' 1",
                "12 'U' 1",
                "13 'e' 1",
                "14 'r' 1",
                "15 'S' 1",
                "16 'y' 1",
                "17 'm' 1",
                "65530 '<|im_start|>' 12",
                "65531 '<|im_end|>' 10",
                "65532 '<|endoftext|>' 13",
                "65533 '<|think|>' 9",
                "65534 '<|tool_call|>' 13",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_hf_rwkv_tokenizer_dir(path: Path) -> Path:
    """Mimic an HF-format RWKV tokenizer directory."""
    vocab_path = path / "rwkv_vocab_v20230424.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 'b' 1",
                "3 '\\n\\n' 2",
                "4 'U' 1",
                "5 's' 1",
                "6 'e' 1",
                "7 'r' 1",
                "8 ':' 1",
                "9 ' ' 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "added_tokens.json").write_text(
        json.dumps({"<|rwkv_tokenizer_end_of_text|>": 0}),
        encoding="utf-8",
    )
    special_tokens_map = {
        "bos_token": "<|rwkv_tokenizer_end_of_text|>",
        "eos_token": "\n\n",
        "unk_token": "<|rwkv_tokenizer_end_of_text|>",
        "pad_token": "<|rwkv_tokenizer_end_of_text|>",
    }
    (path / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map),
        encoding="utf-8",
    )
    tokenizer_config = {
        "tokenizer_class": "RwkvTokenizer",
        "added_tokens_decoder": {
            "0": {
                "content": "<|rwkv_tokenizer_end_of_text|>",
                "special": True,
            }
        },
        **special_tokens_map,
        "chat_template": (
            "{{ '<|rwkv_tokenizer_end_of_text|>' }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{'User: ' + message['content'] + '\\n\\n'}}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        ),
    }
    (path / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    return vocab_path


def _write_legacy_control_token_model_dir(path: Path) -> None:
    """Native vocab control tokens that metadata tries to claim as specials."""
    vocab_path = path / "rwkv_vocab_v20260603.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "1 'a' 1",
                "2 '<' 1",
                "3 'think' 5",
                "4 '>' 1",
                "5 'a<' 2",
                "6 '.' 1",
                "7 '.<' 2",
                "65530 '<|im_start|>' 12",
                "65531 '<|im_end|>' 10",
                "65532 '<|endoftext|>' 13",
                "65533 '' 7",
                "65534 '' 11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    controls = {
        "<|im_start|>": 65530,
        "<|im_end|>": 65531,
        "<|endoftext|>": 65532,
        "<think>": 65533,
        "<tool_call>": 65534,
    }
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "RwkvTokenizer",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<think>",
                    "<tool_call>",
                ],
                "added_tokens_decoder": {
                    str(token_id): {"content": token, "special": True} for token, token_id in controls.items()
                },
            }
        ),
        encoding="utf-8",
    )


_IM_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "{{ '<|im_start|>System: ' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' -%}"
    "{{ '<|im_start|>User: ' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ '<|im_start|>Assistant: ' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|im_start|>Assistant: <think>\\n\\n</think>\\n\\n' }}"
    "{%- endif -%}"
)


# ---------------------------------------------------------------------------
# Round-trip / trie behaviour
# ---------------------------------------------------------------------------


def test_rwkv_tokenizer_round_trips_and_prefers_longest_match(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20250609.txt"
    _write_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(vocab_path)

    assert tokenizer.encode("ab") == [3]
    assert tokenizer.decode([3, 2]) == "abb"
    assert tokenizer.convert_ids_to_tokens([3, 2]) == ["ab", "b"]
    assert tokenizer.convert_ids_to_tokens([6, 3, 7], skip_special_tokens=True) == [
        "<|im_start|>",
        "ab",
        "<|im_end|>",
    ]
    assert tokenizer.convert_tokens_to_ids(["ab", "b"]) == [3, 2]
    assert tokenizer.convert_tokens_to_string(["ab", "b"]) == "abb"
    assert tokenizer.get_vocab()["ab"] == 3
    assert tokenizer.get_added_vocab() == {}
    assert tokenizer.eos_token_id == 5
    assert tokenizer.bos_token_id == 5
    assert tokenizer.pad_token_id == 5
    assert tokenizer.all_special_ids == []
    assert tokenizer.vocab_size == 64


def test_rwkv_tokenizer_uses_legacy_trie_match(tmp_path: Path) -> None:
    """Native whole-string trie wins at boundaries; back-compat with old vocab."""
    vocab_path = tmp_path / "rwkv_vocab_v20230424.txt"
    vocab_path.write_text(
        "0 b'a' 1\n1 b'b' 1\n2 b'ab' 2\n3 b' ' 1\n",
        encoding="utf-8",
    )
    tokenizer = RWKVTokenizer.from_pretrained(vocab_path)

    assert tokenizer.encode("ab a") == [2, 3, 0]
    assert tokenizer.decode([2, 3, 0]) == "ab a"


def test_rwkv_tokenizer_keeps_native_vocab_tokens_out_of_hf_splitting(tmp_path: Path) -> None:
    _write_hf_rwkv_tokenizer_dir(tmp_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(tmp_path))

    assert tokenizer.encode("a\n\nb") == [1, 3, 2]
    assert tokenizer.convert_tokens_to_ids("\n\n") == 3
    assert tokenizer.get_added_vocab() == {"<|rwkv_tokenizer_end_of_text|>": 0}
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 3
    assert tokenizer.pad_token_id == 0
    assert tokenizer.all_special_ids == [0]
    assert tokenizer.vocab_size == 64


def test_rwkv_tokenizer_prioritizes_hf_added_token_boundaries(tmp_path: Path) -> None:
    vocab_path = _write_hf_rwkv_tokenizer_dir(tmp_path)
    with vocab_path.open("a", encoding="utf-8") as f:
        f.write("10 'a\\n' 2\n")
        f.write("11 '\\n' 1\n")

    tokenizer = RWKVTokenizer.from_pretrained(str(tmp_path))

    assert tokenizer.encode("a\n\nb") == [10, 11, 2]
    assert tokenizer.convert_tokens_to_ids("\n\n") == 3
    assert tokenizer.decode([10, 11, 2]) == "a\n\nb"


# ---------------------------------------------------------------------------
# Chat template / jinja rendering
# ---------------------------------------------------------------------------


def test_rwkv_tokenizer_uses_hf_chat_template_prefix(tmp_path: Path) -> None:
    _write_hf_rwkv_tokenizer_dir(tmp_path)
    tokenizer = RWKVTokenizer.from_pretrained(str(tmp_path))

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        add_generation_prompt=True,
    )

    assert rendered == "<|rwkv_tokenizer_end_of_text|>User: ab\n\nAssistant:"
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        tokenize=True,
    ) == [0, 4, 5, 6, 7, 8, 9, 1, 2, 3]


def test_rwkv_tokenizer_keeps_native_pipe_markers_in_raw_trie(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path))

    expected = {
        "<|im_start|>": 65530,
        "<|im_end|>": 65531,
        "<|endoftext|>": 65532,
        "<|think|>": 65533,
        "<|tool_call|>": 65534,
    }
    for token, token_id in expected.items():
        assert tokenizer.convert_tokens_to_ids(token) == token_id

    assert tokenizer.all_special_ids == []

    encoded = tokenizer.encode("<|think|><|tool_call|>")
    assert encoded == [65530, 65533, 65534, 65531]


def test_rwkv_tokenizer_uses_whole_string_trie_for_native_control_tokens(
    tmp_path: Path,
) -> None:
    _write_legacy_control_token_model_dir(tmp_path)
    tokenizer = RWKVTokenizer.from_pretrained(str(tmp_path))

    assert tokenizer.all_special_ids == []
    assert tokenizer.get_added_vocab() == {}
    assert tokenizer.bos_token_id == 65532
    assert tokenizer.eos_token_id == 65532
    assert tokenizer.pad_token_id == 65532
    assert tokenizer.convert_tokens_to_ids("<think>") == 65533

    expected = [5, 3, 4]
    assert tokenizer.encode("a<think>") == expected
    assert tokenizer(["a<think>", "a<think>"]).input_ids == [expected, expected]
    assert tokenizer.encode(".<think>") == [7, 3, 4]
    assert tokenizer.encode("a<think>") != [1, 65533]


def test_rwkv_tokenizer_renders_external_jinja_chat_template(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path), chat_template=_IM_TEMPLATE)

    messages = [
        {"role": "system", "content": "ab"},
        {"role": "user", "content": "ab"},
    ]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    assert rendered == (
        "<|im_start|>System: ab<|im_end|>\n"
        "<|im_start|>User: ab<|im_end|>\n"
        "<|im_start|>Assistant: <think>\n\n</think>\n\n"
    )

    overridden = tokenizer.apply_chat_template(
        messages,
        chat_template="{% for m in messages %}{{ m['content'] }}|{% endfor %}",
    )
    assert overridden == "ab|ab|"


def test_rwkv_tokenizer_auto_loads_chat_template_jinja_from_vocab_dir(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)
    (tmp_path / "chat_template.jinja").write_text(_IM_TEMPLATE, encoding="utf-8")

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path))

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        add_generation_prompt=True,
    )
    assert rendered == ("<|im_start|>User: ab<|im_end|>\n<|im_start|>Assistant: <think>\n\n</think>\n\n")


def test_rwkv_tokenizer_get_chat_template_returns_loaded_template(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path), chat_template=_IM_TEMPLATE)

    assert tokenizer.get_chat_template() == _IM_TEMPLATE
    assert tokenizer.get_chat_template("foo") == "foo"


def test_rwkv_tokenizer_accepts_conversation_alias(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path), chat_template=_IM_TEMPLATE)

    rendered = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "ab"},
            {"role": "user", "content": "ab"},
        ],
        add_generation_prompt=True,
    )

    assert rendered == (
        "<|im_start|>System: ab<|im_end|>\n"
        "<|im_start|>User: ab<|im_end|>\n"
        "<|im_start|>Assistant: <think>\n\n</think>\n\n"
    )


def test_rwkv_tokenizer_jinja_failure_falls_back_to_role_prefix(tmp_path: Path) -> None:
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    bad_template = "{% if not_a_var %}{{ raise_undefined }}"
    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path), chat_template=bad_template)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}],
        add_generation_prompt=True,
    )

    assert "User: ab" in rendered
    assert rendered.endswith("Assistant:")


def test_rwkv_tokenizer_legacy_role_prefix_fallback(tmp_path: Path) -> None:
    """Without a chat template, fall back to ``System:/User:/Assistant:``."""
    vocab_path = tmp_path / "rwkv_vocab_v20260603.txt"
    _write_im_vocab(vocab_path)

    tokenizer = RWKVTokenizer.from_pretrained(str(vocab_path))

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": "ab"},
        ],
        add_generation_prompt=True,
    )

    assert "System: hi\n\n" in rendered
    assert "User: ab\n\n" in rendered
    assert rendered.endswith("Assistant:")
