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
