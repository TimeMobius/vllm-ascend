import json
from types import SimpleNamespace

from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_ascend.tool_parsers.rwkv_tool_parser import RWKVToolParser


def _tool(name: str, properties: dict) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            parameters={"type": "object", "properties": properties},
        )
    )


def _request(*tools: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(tools=list(tools))


def _stream(
    parser: RWKVToolParser,
    previous_text: str,
    current_text: str,
    request: SimpleNamespace,
):
    return parser.extract_tool_calls_streaming(
        previous_text,
        current_text,
        current_text[len(previous_text) :],
        [],
        [],
        [],
        request,
    )


def test_rwkv_tool_parser_is_registered():
    assert ToolParserManager.get_tool_parser("rwkv") is RWKVToolParser


def test_rwkv_tool_parser_extracts_typed_arguments():
    parser = RWKVToolParser(None)
    request = _request(
        _tool(
            "typed",
            {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "items": {"type": "array"},
                "payload": {"type": "object"},
                "missing": {"type": "null"},
            },
        )
    )

    result = parser.extract_tool_calls(
        '<tool_call><invoke name="typed">'
        '<parameter name="count">7</parameter>'
        '<parameter name="ratio">2.0</parameter>'
        '<parameter name="enabled">yes</parameter>'
        '<parameter name="items">[\"a\", 2]</parameter>'
        '<parameter name="payload">{\"ok\": true}</parameter>'
        '<parameter name="missing">null</parameter>'
        "</invoke></tool_call>",
        request,
    )

    assert result.tools_called is True
    arguments = json.loads(result.tool_calls[0].function.arguments)
    assert arguments == {
        "count": 7,
        "ratio": 2,
        "enabled": True,
        "items": ["a", 2],
        "payload": {"ok": True},
        "missing": None,
    }


def test_rwkv_tool_parser_extracts_multiple_invokes_and_content():
    parser = RWKVToolParser(None)
    request = _request(
        _tool("search", {"query": {"type": "string"}}),
        _tool("save", {"value": {"type": "string"}}),
    )

    result = parser.extract_tool_calls(
        "before <tool_call>"
        '<invoke name="search"><parameter name="query">hello</parameter></invoke>'
        '<invoke name="save"><parameter name="value">world</parameter></invoke>'
        "</tool_call>after",
        request,
    )

    assert result.tools_called is True
    assert result.content == "before "
    assert [call.function.name for call in result.tool_calls] == ["search", "save"]


def test_rwkv_tool_parser_handles_schema_unions_and_enum_types():
    parser = RWKVToolParser(None)
    request = _request(
        _tool(
            "union",
            {
                "nullable_count": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}]
                },
                "mode": {"oneOf": [{"type": "boolean"}, {"type": "string"}]},
                "choice": {"allOf": [{"type": "number"}, {"enum": [1, 2]}]},
                "flag": {"enum": [True, False]},
            },
        )
    )

    result = parser.extract_tool_calls(
        '<tool_call><invoke name="union">'
        '<parameter name="nullable_count">null</parameter>'
        '<parameter name="mode">automatic</parameter>'
        '<parameter name="choice">2</parameter>'
        '<parameter name="flag">0</parameter>'
        "</invoke></tool_call>",
        request,
    )

    assert json.loads(result.tool_calls[0].function.arguments) == {
        "nullable_count": None,
        "mode": "automatic",
        "choice": 2,
        "flag": False,
    }


def test_rwkv_tool_parser_streams_complete_invokes_once():
    parser = RWKVToolParser(None)
    request = _request(_tool("search", {"query": {"type": "string"}}))
    first_text = (
        '<tool_call><invoke name="search">'
        '<parameter name="query">hello</parameter></invoke>'
    )

    first_delta = _stream(parser, "", first_text, request)
    assert first_delta is not None
    assert first_delta.tool_calls[0].function.name == "search"
    assert parser.prev_tool_call_arr[0]["name"] == "search"
    assert parser.streamed_args_for_tool[0] == '{"query": "hello"}'

    assert _stream(parser, first_text, first_text, request) is None

    second_text = (
        first_text
        + '<invoke name="search"><parameter name="query">again</parameter></invoke>'
    )
    second_delta = _stream(parser, first_text, second_text, request)
    assert second_delta is not None
    assert len(second_delta.tool_calls) == 1
    assert second_delta.tool_calls[0].index == 1

    reset_delta = _stream(parser, "", "new response", request)
    assert reset_delta is not None
    assert reset_delta.content == "new response"
    assert parser.prev_tool_call_arr == []
    assert parser.streamed_args_for_tool == []


def test_rwkv_tool_parser_streaming_holds_partial_tool_marker():
    parser = RWKVToolParser(None)
    request = _request()

    content_delta = _stream(parser, "", "before <tool", request)
    assert content_delta is not None
    assert content_delta.content == "before "
    assert (
        _stream(parser, "before <tool", "before <tool_call>", request) is None
    )


def test_rwkv_tool_parser_ignores_malformed_xml():
    parser = RWKVToolParser(None)
    request = _request()
    malformed = '<tool_call><invoke name="search"><parameter name="query">hello'

    result = parser.extract_tool_calls(malformed, request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == malformed
