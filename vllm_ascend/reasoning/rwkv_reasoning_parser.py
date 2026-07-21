from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike


class RWKVReasoningParser(ReasoningParser):
    start_token = "<think>"
    end_token = "</think>"
    reasoning_markers_require_text = True

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self.thinking_enabled = (kwargs.get("chat_template_kwargs") or {}).get(
            "enable_thinking"
        ) is True
        self.start_token_ids = tokenizer.encode(self.start_token, add_special_tokens=False)
        self.end_token_ids = tokenizer.encode(self.end_token, add_special_tokens=False)

    @staticmethod
    def _find(values: Sequence[int], pattern: Sequence[int]) -> int:
        width = len(pattern)
        return next(
            (index for index in range(len(values) - width + 1) if list(values[index : index + width]) == list(pattern)),
            -1,
        ) if width else -1

    @classmethod
    def _rfind(cls, values: Sequence[int], pattern: Sequence[int]) -> int:
        width = len(pattern)
        return next(
            (index for index in range(len(values) - width, -1, -1) if list(values[index : index + width]) == list(pattern)),
            -1,
        ) if width else -1

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        end = self._rfind(input_ids, self.end_token_ids)
        return end >= 0 and end > self._rfind(input_ids, self.start_token_ids)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Sequence[int]) -> bool:
        return self._find(delta_ids, self.end_token_ids) >= 0 or self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end = self._find(input_ids, self.end_token_ids)
        return input_ids[end + len(self.end_token_ids) :] if end >= 0 else []

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        del request
        start = model_output.find(self.start_token)
        if start >= 0:
            model_output = model_output[start + len(self.start_token) :]
        end = model_output.find(self.end_token)
        if end >= 0:
            return model_output[:end] or None, model_output[end + len(self.end_token) :] or None
        if start >= 0 or self.thinking_enabled:
            return model_output or None, None
        return None, model_output or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids
        previous = self.extract_reasoning(previous_text, None)
        current = self.extract_reasoning(current_text, None)
        reasoning = current[0][len(previous[0] or "") :] if current[0] else None
        content = current[1][len(previous[1] or "") :] if current[1] else None
        return DeltaMessage(reasoning=reasoning or None, content=content or None) if reasoning or content else None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        start = self._find(token_ids, self.start_token_ids)
        end = self._find(token_ids, self.end_token_ids)
        if start < 0 and not self.thinking_enabled:
            return 0
        begin = start + len(self.start_token_ids) if start >= 0 else 0
        return max(0, (end if end >= 0 else len(token_ids)) - begin)
