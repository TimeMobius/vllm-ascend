from pathlib import Path

from vllm_ascend.tokenizers.rwkv import RWKVTokenizer


def _write_vocab(path: Path) -> None:
    path.write_text(
        "0 b'a' 1\n1 b'b' 1\n2 b'ab' 2\n3 b' ' 1\n",
        encoding="utf-8",
    )


def test_rwkv_tokenizer_uses_longest_legacy_trie_match(tmp_path: Path):
    vocab_path = tmp_path / "rwkv_vocab_v20230424.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer(vocab_path)

    assert tokenizer.encode("ab a") == [2, 3, 0]
    assert tokenizer.decode([2, 3, 0]) == "ab a"


def test_rwkv_tokenizer_resolves_metadata_special_tokens(tmp_path: Path):
    vocab_path = tmp_path / "rwkv_vocab_v20230424.txt"
    _write_vocab(vocab_path)
    tokenizer = RWKVTokenizer(
        vocab_path,
        metadata={"bos_token": "<bos>", "eos_token": "<eos>"},
    )

    assert tokenizer.encode("<bos>a<eos>") == [4, 0, 5]
    assert tokenizer.bos_token_id == 4
    assert tokenizer.eos_token_id == 5


def test_rwkv_tokenizer_preserves_native_special_tokens(tmp_path: Path):
    """Native vocab control tokens remain trie tokens even when metadata lists them as special.

    When a token (e.g. '<bos>') exists in the native vocab AND is listed in metadata
    as a special token, the tokenizer should NOT promote it to a special token.
    Instead, it should remain encoded as a native trie token with its original ID.
    """
    vocab_path = tmp_path / "rwkv_vocab_v20230424.txt"
    # Vocab includes token ID 4 as '<bos>' (single byte 4) and ID 5 as '<eos>' (byte 5)
    vocab_path.write_text(
        "0 b'a' 1\n1 b'b' 1\n2 b'ab' 2\n3 b' ' 1\n4 b'\\x04' 1\n5 b'\\x05' 1\n",
        encoding="utf-8",
    )
    tokenizer = RWKVTokenizer(
        vocab_path,
        metadata={
            "bos_token": "\x04",  # Same as byte 4 in vocab
            "eos_token": "\x05",  # Same as byte 5 in vocab
        },
    )

    # Token \x04 should stay as native trie token with ID 4, not be promoted
    assert tokenizer.encode("\x04a\x05") == [4, 0, 5]
    # bos_token_id should be the native ID 4, not a new special token ID
    assert tokenizer.bos_token_id == 4
    # \x04 should NOT be in added vocab (it's native, not added)
    assert "\x04" not in tokenizer.get_added_vocab()


def test_rwkv_tokenizer_native_token_not_promoted_when_in_metadata(tmp_path: Path):
    """Tokens present in native vocab should not be promoted even if metadata lists them.

    This tests the case where a control character like \\x05 is both in the native
    vocab AND specified as eos_token in metadata. It should keep its native ID.
    """
    vocab_path = tmp_path / "rwkv_vocab_v20230424.txt"
    # ID 5 is byte 5 (enqiry character in latin-1)
    vocab_path.write_text(
        "0 b'a' 1\n1 b'b' 1\n5 b'\\x05' 1\n",
        encoding="utf-8",
    )
    tokenizer = RWKVTokenizer(
        vocab_path,
        metadata={"eos_token": "\x05"},
    )

    # Token \x05 should remain as native ID 5
    assert tokenizer.encode("a\x05") == [0, 5]
    assert tokenizer.eos_token_id == 5
    # Should NOT be in added vocab
    assert "\x05" not in tokenizer.get_added_vocab()
