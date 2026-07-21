from vllm_ascend.reasoning.rwkv_reasoning_parser import RWKVReasoningParser


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


def test_rwkv_reasoning_parser_extracts_think_block():
    parser = RWKVReasoningParser(_Tokenizer())

    assert parser.extract_reasoning("<think>plan</think>answer", None) == (
        "plan",
        "answer",
    )
