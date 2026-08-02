# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def _rwkv7_dynamic_arg_dims() -> dict[str, int | list[int] | dict[int, str]]:
    source_path = Path(__file__).parents[2] / "vllm_ascend" / "models" / "rwkv7.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7Model")
    decorator = next(
        node
        for node in model_class.decorator_list
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "support_torch_compile"
    )
    dynamic_arg_dims = next(keyword.value for keyword in decorator.keywords if keyword.arg == "dynamic_arg_dims")
    return ast.literal_eval(dynamic_arg_dims)


def test_rwkv7_piecewise_metadata_keeps_input_ids_static_and_shape_carrier_dynamic():
    dynamic_arg_dims = _rwkv7_dynamic_arg_dims()

    assert "input_ids" not in dynamic_arg_dims
    assert dynamic_arg_dims["positions"] == 0
    assert dynamic_arg_dims["intermediate_tensors"] == 0
    assert dynamic_arg_dims["inputs_embeds"] == 0
