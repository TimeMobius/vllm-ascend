# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 breakable-ACLGraph break-point tests.

The two registered RWKV7 custom ops must be valid break points for
``VLLM_USE_BREAKABLE_CUDAGRAPH`` capture, while default (breakable disabled)
behavior stays byte-for-byte identical: the ops still delegate to the eager
``RWKV7Attention``/``RWKV7Block`` submodule through ``forward_context``.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models import rwkv7

SOURCE_PATH = Path(__file__).parents[2] / "vllm_ascend" / "models" / "rwkv7.py"

ATTENTION_OP = "rwkv7_attention"
BLOCK_OP = "rwkv7_block_forward"
FAKE_OPS = ("rwkv7_attention_fake", "rwkv7_block_forward_fake")


def _function_decorator_names(func_name: str) -> list[str]:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == func_name)
    return [decorator.id for decorator in func.decorator_list if isinstance(decorator, ast.Name)]


def test_break_point_decorator_is_imported_from_breakable_module():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.compilation.breakable_cudagraph"
        and any(alias.name == "eager_break_during_capture" for alias in node.names)
        for node in tree.body
    )
    assert imported


def test_break_point_marker_present_on_registered_rwkv7_ops():
    for op_name in (ATTENTION_OP, BLOCK_OP):
        assert "eager_break_during_capture" in _function_decorator_names(op_name)


def test_break_point_marker_absent_on_fake_implementations():
    # Fakes are metadata-only kernels used for torch.compile fake tracing and
    # must not intercept capture; only the real kernels are break points.
    for op_name in FAKE_OPS:
        assert "eager_break_during_capture" not in _function_decorator_names(op_name)


def _make_attention_self() -> MagicMock:
    out = torch.randn(3, 8)
    shift = torch.randn(3, 8, dtype=torch.float32)
    recurrent = torch.randn(2, 4, 4, dtype=torch.float32)
    first_value = torch.randn(3, 8, dtype=torch.float32)
    layer = MagicMock()
    layer._forward.return_value = (out, shift, recurrent, first_value)
    return layer


def _make_attention_buffers() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(3, 8),
        torch.zeros(3, 8, dtype=torch.bfloat16),
        torch.zeros(2, 4, 4, dtype=torch.bfloat16),
        torch.zeros(3, 8, dtype=torch.bfloat16),
    )


def test_disabled_attention_still_delegates_to_eager_submodule(monkeypatch):
    # Module is imported with breakable disabled: the op must be the raw kernel.
    assert not hasattr(rwkv7.rwkv7_attention, "__wrapped__")

    hidden = torch.randn(3, 8)
    cached = torch.randn(3, 8)
    recurrent = torch.randn(2, 4, 4)
    v_first = torch.randn(3, 8)

    layer = _make_attention_self()
    monkeypatch.setattr(
        rwkv7,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.attn": layer}),
    )

    output, final_shift, final_recurrent, v_first_out = _make_attention_buffers()

    rwkv7.rwkv7_attention(
        hidden,
        cached,
        recurrent,
        v_first,
        output,
        final_shift,
        final_recurrent,
        v_first_out,
        "layer.attn",
    )

    layer._forward.assert_called_once()
    forwarded = layer._forward.call_args.args
    assert forwarded[0] is hidden
    assert forwarded[1] is cached
    assert forwarded[2] is recurrent
    assert forwarded[3] is v_first

    expected_out, expected_shift, expected_recurrent, expected_first = layer._forward.return_value
    torch.testing.assert_close(output, expected_out)
    torch.testing.assert_close(final_shift, expected_shift.to(final_shift.dtype))
    torch.testing.assert_close(final_recurrent, expected_recurrent.to(final_recurrent.dtype))
    torch.testing.assert_close(v_first_out, expected_first.to(v_first_out.dtype))


def test_disabled_attention_maps_empty_optional_states_to_none(monkeypatch):
    empty = torch.empty(0)
    layer = _make_attention_self()
    monkeypatch.setattr(
        rwkv7,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.attn": layer}),
    )

    output, final_shift, final_recurrent, v_first_out = _make_attention_buffers()
    rwkv7.rwkv7_attention(
        torch.randn(3, 8),
        empty,
        empty,
        empty,
        output,
        final_shift,
        final_recurrent,
        v_first_out,
        "layer.attn",
    )

    forwarded = layer._forward.call_args.args
    assert forwarded[1] is None
    assert forwarded[2] is None
    assert forwarded[3] is None


def test_disabled_block_forward_still_delegates_to_eager_submodule(monkeypatch):
    assert not hasattr(rwkv7.rwkv7_block_forward, "__wrapped__")

    hidden = torch.randn(3, 8)
    v_first = torch.randn(3, 8)
    expected_out = torch.randn(3, 8)
    expected_v_first = torch.randn(3, 8)

    layer = MagicMock()

    def _runtime(hidden_states, vf, out, vf_out):
        out.copy_(expected_out)
        vf_out.copy_(expected_v_first)

    layer._forward_runtime.side_effect = _runtime
    monkeypatch.setattr(
        rwkv7,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.block": layer}),
    )

    output = torch.zeros(3, 8)
    v_first_out = torch.zeros(3, 8)
    rwkv7.rwkv7_block_forward(hidden, v_first, output, v_first_out, "layer.block")

    layer._forward_runtime.assert_called_once()
    forwarded = layer._forward_runtime.call_args.args
    assert forwarded[0] is hidden
    assert forwarded[1] is v_first
    assert forwarded[2] is output
    assert forwarded[3] is v_first_out
    torch.testing.assert_close(output, expected_out)
    torch.testing.assert_close(v_first_out, expected_v_first)


class _RecordingCapture:
    """Minimal stand-in for BreakableCUDAGraphCapture."""

    _capturing = True

    def __init__(self) -> None:
        self.recorded: list = []

    def add_eager(self, fn):
        self.recorded.append(fn)
        return fn()


def _arm_breakable_capture(monkeypatch) -> "_RecordingCapture":
    import vllm.compilation.breakable_cudagraph as breakable

    monkeypatch.setattr(breakable, "is_breakable_cudagraph_enabled", lambda: True)

    capture = _RecordingCapture()
    monkeypatch.setattr(breakable.BreakableCUDAGraphCapture, "current", classmethod(lambda cls: capture))
    monkeypatch.setattr(breakable, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        breakable,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    )
    # The weak-ref tensor pool is irrelevant to break-point dispatch.
    monkeypatch.setattr(breakable, "weak_ref_tensor", lambda tensor: tensor)
    return capture


def test_enabled_attention_is_a_capture_break_point(monkeypatch):
    import vllm.compilation.breakable_cudagraph as breakable

    capture = _arm_breakable_capture(monkeypatch)
    decorated = breakable.eager_break_during_capture(rwkv7.rwkv7_attention)
    assert decorated is not rwkv7.rwkv7_attention

    layer = _make_attention_self()
    monkeypatch.setattr(
        rwkv7,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.attn": layer}),
    )
    output, final_shift, final_recurrent, v_first_out = _make_attention_buffers()

    decorated(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.randn(2, 4, 4),
        torch.randn(3, 8),
        output,
        final_shift,
        final_recurrent,
        v_first_out,
        "layer.attn",
    )

    assert len(capture.recorded) == 1
    assert callable(capture.recorded[0])
    layer._forward.assert_called_once()
    torch.testing.assert_close(output, layer._forward.return_value[0])


def test_enabled_decorator_preserves_custom_op_schema(monkeypatch):
    # When breakable is enabled the module-level decorator wraps the kernel
    # before direct_register_custom_op infers its schema; functools.wraps must
    # keep the signature and mutation annotations intact or import would fail.
    import vllm.compilation.breakable_cudagraph as breakable
    from torch.library import infer_schema

    monkeypatch.setattr(breakable, "is_breakable_cudagraph_enabled", lambda: True)
    decorated = breakable.eager_break_during_capture(rwkv7.rwkv7_attention)
    schema = infer_schema(
        decorated,
        mutates_args=["output", "final_shift_state", "final_recurrent_state", "v_first_out"],
    )
    assert "v_first_out" in schema
    assert "layer_name" in schema
    assert schema.endswith("-> ()")


def test_enabled_block_forward_is_a_capture_break_point(monkeypatch):
    import vllm.compilation.breakable_cudagraph as breakable

    capture = _arm_breakable_capture(monkeypatch)
    decorated = breakable.eager_break_during_capture(rwkv7.rwkv7_block_forward)
    assert decorated is not rwkv7.rwkv7_block_forward

    expected_out = torch.randn(3, 8)
    expected_v_first = torch.randn(3, 8)
    layer = MagicMock()
    layer._forward_runtime.side_effect = lambda hidden, vf, out, vf_out: (
        out.copy_(expected_out),
        vf_out.copy_(expected_v_first),
    )
    monkeypatch.setattr(
        rwkv7,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer.block": layer}),
    )

    output = torch.zeros(3, 8)
    v_first_out = torch.zeros(3, 8)
    decorated(torch.randn(3, 8), torch.randn(3, 8), output, v_first_out, "layer.block")

    assert len(capture.recorded) == 1
    assert callable(capture.recorded[0])
    layer._forward_runtime.assert_called_once()
    torch.testing.assert_close(output, expected_out)
    torch.testing.assert_close(v_first_out, expected_v_first)
