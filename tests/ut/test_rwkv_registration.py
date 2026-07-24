"""
Focused registration tests for RWKV parser/renderer/tokenizer discoverability.

Tests that RWKV components are registered with the correct vLLM registries
and are discoverable in a fresh Python process.
"""

import subprocess
import sys


def test_rwkv_discoverable_in_fresh_process():
    # We need to test in a fresh Python process because registries are process-global
    # and may have been populated by previous tests/imports in the same process.
    code = """
from types import SimpleNamespace

from vllm.plugins import load_general_plugins
load_general_plugins()

# Verify ReasoningParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm_ascend.reasoning import RWKVReasoningParser
assert "rwkv" in ReasoningParserManager.list_registered(), \
    f"rwkv not in ReasoningParserManager: {ReasoningParserManager.list_registered()}"
parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
assert parser_cls is RWKVReasoningParser, f"Expected RWKVReasoningParser, got {parser_cls}"

# Verify ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm_ascend.tool_parsers import RWKVToolParser
assert "rwkv" in ToolParserManager.list_registered(), \
    f"rwkv not in ToolParserManager: {ToolParserManager.list_registered()}"
tool_cls = ToolParserManager.get_tool_parser("rwkv")
assert tool_cls is RWKVToolParser, f"Expected RWKVToolParser, got {tool_cls}"

from vllm.entrypoints.openai.api_server import validate_api_server_args
validate_api_server_args(SimpleNamespace(
    enable_auto_tool_choice=True,
    tool_call_parser="rwkv",
    structured_outputs_config=SimpleNamespace(reasoning_parser="rwkv"),
))

print("GENERAL_PLUGIN_REGISTRATIONS_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/mnt/data/Codes/vllm-ascend",
    )
    assert result.returncode == 0, (
        f"Fresh process check failed.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "GENERAL_PLUGIN_REGISTRATIONS_OK" in result.stdout, (
        f"Registration check did not pass.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_rwkv_registration_is_idempotent():
    """Verify repeated import/registration does not raise."""
    code = """
from vllm_ascend import register_rwkv

register_rwkv()
register_rwkv()

from vllm_ascend.reasoning import RWKVReasoningParser
from vllm_ascend.reasoning import RWKVReasoningParser as RWKVReasoningParser2

from vllm_ascend.tool_parsers import RWKVToolParser
from vllm_ascend.tool_parsers import RWKVToolParser as RWKVToolParser2

from vllm_ascend.renderers import RWKVRenderer
from vllm_ascend.renderers import RWKVRenderer as RWKVRenderer2

from vllm_ascend.tokenizers import RWKVTokenizer
from vllm_ascend.tokenizers import RWKVTokenizer as RWKVTokenizer2

print("IDEMPOTENT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/mnt/data/Codes/vllm-ascend",
    )
    assert result.returncode == 0, (
        f"Idempotency check failed.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "IDEMPOTENT_OK" in result.stdout, (
        f"Idempotency check did not pass.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_rwkv_listed_in_all_registries():
    """Verify rwkv appears in list_registered() for parsers."""
    # Import to trigger registration
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    from vllm_ascend.reasoning import RWKVReasoningParser  # noqa: F401
    from vllm_ascend.renderers import RWKVRenderer  # noqa: F401
    from vllm_ascend.tokenizers import RWKVTokenizer  # noqa: F401
    from vllm_ascend.tool_parsers import RWKVToolParser  # noqa: F401

    registered_reasoning = ReasoningParserManager.list_registered()
    registered_tools = ToolParserManager.list_registered()

    assert "rwkv" in registered_reasoning, (
        f"rwkv not in ReasoningParserManager.list_registered(): {registered_reasoning}"
    )
    assert "rwkv" in registered_tools, (
        f"rwkv not in ToolParserManager.list_registered(): {registered_tools}"
    )
    assert "rwkv" in RENDERER_REGISTRY.renderers, (
        f"rwkv not in RENDERER_REGISTRY.renderers: {list(RENDERER_REGISTRY.renderers.keys())}"
    )
    assert "rwkv" in TokenizerRegistry.tokenizers, (
        f"rwkv not in TokenizerRegistry.tokenizers: {list(TokenizerRegistry.tokenizers.keys())}"
    )
