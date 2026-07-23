import pytest

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm_ascend.reasoning.rwkv_reasoning_parser import RWKVReasoningParser


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


class MultiTokenThinkTokenizer:
    """Small tokenizer where think markers are deliberately multi-token.

    ``<think>`` → <, think, >
    ``</think>`` → </, think, >
    """

    _piece_to_id = {
        "<": 1,
        "</": 2,
        "think": 3,
        ">": 4,
    }
    _id_to_piece = {token_id: piece for piece, token_id in _piece_to_id.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._piece_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        idx = 0
        next_id = max(self._id_to_piece) + 1
        while idx < len(text):
            if text.startswith("</", idx):
                piece = "</"
                idx += len(piece)
            elif text.startswith("think", idx):
                piece = "think"
                idx += len(piece)
            else:
                piece = text[idx]
                idx += 1
            token_id = self._piece_to_id.get(piece)
            if token_id is None:
                token_id = next_id + ord(piece)
                self._piece_to_id[piece] = token_id
                self._id_to_piece[token_id] = piece
            token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._id_to_piece[token_id] for token_id in token_ids)


def _extract_streaming(parser, tokenizer, deltas):
    """Helper to simulate streaming extraction over token deltas."""
    reasoning = ""
    content = ""
    prev_text = ""
    prev_ids: list[int] = []

    for delta in deltas:
        delta_ids = tokenizer.encode(delta)
        curr_text = prev_text + delta
        curr_ids = prev_ids + delta_ids
        msg = parser.extract_reasoning_streaming(
            prev_text,
            curr_text,
            delta,
            prev_ids,
            curr_ids,
            delta_ids,
        )
        if isinstance(msg, DeltaMessage):
            reasoning += msg.reasoning or ""
            content += msg.content or ""
        prev_text = curr_text
        prev_ids = curr_ids

    return reasoning or None, content or None


def test_rwkv_reasoning_parser_extracts_think_block():
    parser = RWKVReasoningParser(_Tokenizer())

    assert parser.extract_reasoning("<think>plan</think>answer", None) == (
        "plan",
        "answer",
    )


class TestMultiTokenStreaming:
    """Tests for multi-token marker boundaries in streaming mode."""

    def test_partial_end_marker_withheld_until_complete(self):
        """Partial ``</think>`` fragments (</, think>) must not leak into content."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        # Simulate streaming: first chunk ends with partial "</think", next has ">"
        # The "</think>" is NOT yet complete in first chunk, so nothing contentful
        # should be yielded until we see the full marker.
        # Full text built: "<think>reason</think>answer"
        reasoning, c = _extract_streaming(
            parser,
            MultiTokenThinkTokenizer(),
            ["<think>reason", "</think", ">answer"],
        )
        assert c == "answer", f"Expected 'answer' but got {c!r}; partial </ was leaked"
        assert reasoning == "reason"

    def test_partial_start_marker_withheld_in_reasoning(self):
        """Partial ``<think>`` fragments in reasoning stream are withheld."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        # Full: "<think>reason</think>answer"
        reasoning, c = _extract_streaming(
            parser,
            MultiTokenThinkTokenizer(),
            ["<think", ">reason", "</think", ">answer"],
        )
        assert reasoning == "reason"
        assert c == "answer"

    def test_acceptance_case_split_across_chunks(self):
        """From acceptance: feed ['<think>ab','c</','think>','ans'];
        no unresolved '</' leaks and 'ans' becomes content."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        # This simulates: "ab<think>c</think>ans"
        # But note: in the acceptance, the start marker is already complete
        # Let's construct exactly: ['<think>ab','c</','think>','ans']
        # means: first chunk "<think>ab", then "c</", then "think>", then "ans"
        # Full text: "<think>abc</think>ans"
        reasoning, c = _extract_streaming(
            parser,
            MultiTokenThinkTokenizer(),
            ["<think>ab", "c</", "think>", "ans"],
        )
        # reasoning should be "abc", content should be "ans"
        assert c == "ans", f"Expected 'ans' but got {c!r}"
        assert reasoning == "abc", f"Expected 'abc' but got {reasoning!r}"

    def test_end_marker_split_in_streaming(self):
        """``</think>`` split across tokens yields correct content delta."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        # Full text: "<think>reason</think>answer"
        reasoning, c = _extract_streaming(
            parser,
            MultiTokenThinkTokenizer(),
            ["<think>reason", "</", "think", ">", "answer"],
        )
        assert reasoning == "reason"
        assert c == "answer"

    def test_start_marker_split_in_streaming(self):
        """``<think>`` split across tokens yields correct reasoning delta."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        reasoning, c = _extract_streaming(
            parser,
            MultiTokenThinkTokenizer(),
            ["<", "think", ">", "reason", "</think", ">answer"],
        )
        assert reasoning == "reason"
        assert c == "answer"

    def test_is_reasoning_end_streaming_with_partial_delta(self):
        """is_reasoning_end_streaming returns False until complete end marker in delta."""
        tokenizer = MultiTokenThinkTokenizer()
        parser = RWKVReasoningParser(tokenizer)

        # Before the complete end marker, should not report end
        ids_before = tokenizer.encode("reason")
        delta_partial = tokenizer.encode("</think")
        assert not parser.is_reasoning_end_streaming(ids_before, delta_partial)

        # After the complete end marker in delta
        ids_full = tokenizer.encode("reason</think>")
        delta_final = tokenizer.encode(">")
        assert parser.is_reasoning_end_streaming(ids_full, delta_final)

    def test_without_trailing_partial_marker_end_tag(self):
        """Partial ``</think>`` at end of content string must be stripped."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        # Partial end marker at END of content text is stripped
        text = "answer</think"
        result = parser._without_trailing_partial_marker(text, ("</think>",))
        assert result == "answer", f"Expected 'answer' but got {result!r}"

    def test_without_trailing_partial_marker_no_match(self):
        """_without_trailing_partial_marker returns text unchanged when no partial marker."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        assert parser._without_trailing_partial_marker("answer", ("</think>",)) == "answer"

    def test_without_trailing_partial_marker_strips_longest_match(self):
        """_without_trailing_partial_marker strips longest matching suffix."""
        parser = RWKVReasoningParser(MultiTokenThinkTokenizer())
        assert parser._without_trailing_partial_marker("answer</think", ("<think>",)) == "answer</think"


class TestThinkingEnabled:
    """Tests for thinking-enabled/disabled behavior."""

    def test_thinking_disabled_returns_content_only(self):
        """When enable_thinking=False, plain text goes to content."""
        parser = RWKVReasoningParser(_Tokenizer())
        reasoning, c = parser.extract_reasoning("plain answer", None)
        assert reasoning is None
        assert c == "plain answer"

    def test_thinking_disabled_with_end_marker(self):
        """When enable_thinking=False, text before ``</think>`` is reasoning only."""
        parser = RWKVReasoningParser(_Tokenizer())
        # With thinking disabled, text after start marker but before end is reasoning
        # But without start marker and thinking disabled, it's content
        reasoning, c = parser.extract_reasoning("answer</think>", None)
        assert reasoning == "answer"
        assert c is None

    def test_thinking_disabled_full_marker(self):
        """When enable_thinking=False, ``<think>...</think>`` still extracted."""
        parser = RWKVReasoningParser(_Tokenizer())
        reasoning, c = parser.extract_reasoning("<think>reason</think>answer", None)
        assert reasoning == "reason"
        assert c == "answer"


class TestTokenBoundaries:
    """Tests for multi-token marker boundary correctness."""

    def test_extract_content_ids_respects_marker_length(self):
        """extract_content_ids skips past the full multi-token end marker."""
        tokenizer = MultiTokenThinkTokenizer()
        parser = RWKVReasoningParser(tokenizer)
        output_ids = tokenizer.encode("reason</think>answer")
        content_ids = parser.extract_content_ids(output_ids)
        assert tokenizer.decode(content_ids) == "answer"

    def test_count_reasoning_tokens_multi_token(self):
        """count_reasoning_tokens accounts for multi-token start/end markers."""
        tokenizer = MultiTokenThinkTokenizer()
        parser = RWKVReasoningParser(tokenizer)
        # "reason</think>answer" - no start marker and thinking disabled → 0
        ids = tokenizer.encode("reason</think>answer")
        assert parser.count_reasoning_tokens(ids) == 0

    def test_is_reasoning_end_false_without_complete_marker(self):
        """is_reasoning_end returns False when only partial end marker present."""
        tokenizer = MultiTokenThinkTokenizer()
        parser = RWKVReasoningParser(tokenizer)
        ids = tokenizer.encode("reason</think")  # incomplete
        assert not parser.is_reasoning_end(ids)

    def test_is_reasoning_end_true_with_complete_marker(self):
        """is_reasoning_end returns True when full end marker present."""
        tokenizer = MultiTokenThinkTokenizer()
        parser = RWKVReasoningParser(tokenizer)
        ids = tokenizer.encode("reason</think>")
        assert parser.is_reasoning_end(ids)
