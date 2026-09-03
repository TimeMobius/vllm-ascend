# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for RWKV7 epilogue row dispatch configuration."""

import pytest

from vllm_ascend.models.rwkv7 import (
    EPILOGUE_ROWS_CANDIDATES,
    select_epilogue_rows,
)


class TestSelectEpilogueRows:
    def test_empty_work_returns_one(self):
        assert select_epilogue_rows(num_tokens=0, num_heads=64) == 1

    @pytest.mark.parametrize(
        ("num_tokens", "expected_rows"),
        [(1, 2), (4, 4), (16, 8), (128, 8)],
    )
    def test_formula_matches_tuning_head_geometry(self, num_tokens, expected_rows):
        assert select_epilogue_rows(num_tokens=num_tokens, num_heads=16) == expected_rows

    @pytest.mark.parametrize(
        ("num_tokens", "expected_rows"),
        [(1, 4), (2, 4), (4, 8), (8, 8), (16, 8), (32, 8)],
    )
    def test_formula_selects_fastest_candidate(self, num_tokens, expected_rows):
        assert select_epilogue_rows(num_tokens=num_tokens, num_heads=64) == expected_rows

    def test_formula_returns_a_supported_compile_time_candidate(self):
        assert select_epilogue_rows(num_tokens=128, num_heads=64) in EPILOGUE_ROWS_CANDIDATES
