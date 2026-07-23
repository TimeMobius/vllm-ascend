from __future__ import annotations

import json
import re
from ast import literal_eval
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import BatchEncoding

from vllm.tokenizers.protocol import TokenizerLike


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.values: list[tuple[bytes, int]] = []

    def add(self, token: bytes, token_id: int, index: int = 0) -> None:
        if index == len(token):
            self.values.append((token, token_id))
            return
        child = self.children.setdefault(token[index], _TrieNode())
        child.add(token, token_id, index + 1)

    def longest(self, data: bytes, index: int) -> tuple[int, int]:
        node = self
        best: tuple[int, int] | None = None
        while index < len(data) and data[index] in node.children:
            node = node.children[data[index]]
            index += 1
            if node.values:
                best = (index, node.values[0][1])
        if best is None:
            raise ValueError("RWKV vocabulary cannot encode the input text.")
        return best


class RWKVTokenizer(TokenizerLike):
    _SPECIAL_KEYS = (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "additional_special_tokens",
    )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> RWKVTokenizer:
        del args
        kwargs.pop("trust_remote_code", False)
        kwargs.pop("revision", None)
        kwargs.pop("download_dir", None)
        root = Path(path_or_repo_id)
        vocab_path = root if root.is_file() else next(root.glob("rwkv_vocab*.txt"), None)
        if vocab_path is None:
            raise ValueError(f"No RWKV vocabulary found under {root}.")
        metadata = cls._load_metadata(vocab_path.parent)
        return cls(vocab_path, metadata=metadata)

    @staticmethod
    def _load_metadata(root: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        config_path = root / "tokenizer_config.json"
        if config_path.is_file():
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}
            if isinstance(loaded, dict):
                metadata.update(loaded)
        return metadata

    def __init__(
        self,
        vocab_path: Path,
        *,
        truncation_side: str = "left",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.vocab_path = vocab_path
        self._truncation_side = truncation_side
        self._id_to_bytes: dict[int, bytes] = {}
        self._bytes_to_id: dict[bytes, int] = {}
        for line in vocab_path.read_text(encoding="utf-8").splitlines():
            first_space = line.index(" ")
            last_space = line.rindex(" ")
            token_id = int(line[:first_space])
            raw_token = literal_eval(line[first_space:last_space])
            token = raw_token.encode("utf-8") if isinstance(raw_token, str) else raw_token
            self._id_to_bytes[token_id] = token
            self._bytes_to_id[token] = token_id
        self._root = _TrieNode()
        for token, token_id in self._bytes_to_id.items():
            self._root.add(token, token_id)

        metadata = metadata or {}
        self._special_tokens: dict[str, int] = {}
        next_id = max(self._id_to_bytes, default=-1) + 1
        for key in self._SPECIAL_KEYS:
            value = metadata.get(key)
            values = value if key == "additional_special_tokens" else [value]
            if not isinstance(values, list):
                values = [values]
            for token in values:
                if not isinstance(token, str):
                    continue
                token_bytes = token.encode("latin-1")
                if token_bytes in self._bytes_to_id:
                    continue
                self._special_tokens[token] = next_id
                next_id += 1
        self._special_id_to_token = {v: k for k, v in self._special_tokens.items()}
        self._token_to_id = {
            token.decode("latin-1"): token_id
            for token, token_id in self._bytes_to_id.items()
        }
        self._token_to_id.update(self._special_tokens)
        self._special_pattern = (
            re.compile("|".join(re.escape(t) for t in self._special_tokens))
            if self._special_tokens
            else None
        )
        self._chat_template = metadata.get("chat_template")
        self._default_id = self._token_to_id.get("<|endoftext|>", 0)
        self._bos_token_id = self._token_to_id.get(metadata.get("bos_token"), self._default_id)
        self._eos_token_id = self._token_to_id.get(metadata.get("eos_token"), self._default_id)
        self._pad_token_id = self._token_to_id.get(metadata.get("pad_token"), self._default_id)

    def _encode_plain(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        index = 0
        result: list[int] = []
        while index < len(data):
            index, token_id = self._root.longest(data, index)
            result.append(token_id)
        return result

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del add_special_tokens
        parts: list[str] = []
        if self._special_pattern is None:
            parts.append(text)
        else:
            last_end = 0
            for match in self._special_pattern.finditer(text):
                parts.append(text[last_end : match.start()])
                parts.append(match.group(0))
                last_end = match.end()
            parts.append(text[last_end:])
        result: list[int] = []
        for part in parts:
            if part in self._special_tokens:
                result.append(self._special_tokens[part])
            else:
                result.extend(self._encode_plain(part))
        if truncation and max_length is not None and len(result) > max_length:
            result = result[-max_length:] if self._truncation_side == "left" else result[:max_length]
        return result

    def __call__(self, text: str | list[str], **kwargs: Any) -> BatchEncoding:
        ids = [self.encode(item, **kwargs) for item in text] if isinstance(text, list) else self.encode(text, **kwargs)
        return BatchEncoding({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids] if isinstance(ids, list) and ids and isinstance(ids[0], list) else [1] * len(ids)})

    def decode(self, ids: Sequence[int] | int, skip_special_tokens: bool = False) -> str:
        values = [ids] if isinstance(ids, int) else list(ids)
        chunks: list[bytes] = []
        for token_id in values:
            if skip_special_tokens and token_id in self._special_id_to_token:
                continue
            chunks.append(self._id_to_bytes.get(token_id, self._special_id_to_token.get(token_id, "").encode()))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._special_tokens)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        return self._token_to_id.get(tokens, self._default_id)

    def convert_ids_to_tokens(self, ids: Sequence[int], skip_special_tokens: bool = False) -> list[str]:
        return [self.decode(token_id) for token_id in ids if not (skip_special_tokens and token_id in self._special_id_to_token)]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.decode(self.convert_tokens_to_ids(tokens))

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str | list[int]:
        rendered = "".join(f"{message.get('role', 'user').title()}: {message.get('content', '')}\n\n" for message in messages)
        if kwargs.get("add_generation_prompt"):
            rendered += "Assistant:"
        return self.encode(rendered) if kwargs.get("tokenize") else rendered

    def num_special_tokens_to_add(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._special_tokens)

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._special_id_to_token)

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return max(self._id_to_bytes, default=-1) + 1

    @property
    def max_token_id(self) -> int:
        return self.vocab_size - 1

    @property
    def max_chars_per_token(self) -> int:
        return max((len(value) for value in self._id_to_bytes.values()), default=0)

    @property
    def truncation_side(self) -> str:
        return self._truncation_side
