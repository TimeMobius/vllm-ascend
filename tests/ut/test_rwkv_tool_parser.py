from vllm_ascend.tool_parsers.rwkv_tool_parser import RWKVToolParser


def test_rwkv_tool_parser_extracts_xml_invoke():
    parser = RWKVToolParser(None)
    request = type("Request", (), {"tools": []})()

    result = parser.extract_tool_calls(
        '<tool_call><invoke name="search"><parameter name="q">hello</parameter></invoke></tool_call>',
        request,
    )

    assert result.tools_called is True
    assert result.tool_calls[0].function.name == "search"
