# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flash Linear Attention (FLA) operations for Ascend NPU.

This module provides Triton kernels for FLA operations adapted for Ascend NPUs,
including chunk, recurrent, and gating operations.
"""

from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
    rwkv7_lnx_rkvres_xg,
    rwkv7_lnx_rkvres_xg_reference,
)
from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import (
    rwkv7_kk_pre,
    rwkv7_kk_pre_reference,
)
from vllm_ascend.ops.triton.fla.rwkv7_mix6 import (
    rwkv7_mix6,
    rwkv7_mix6_reference,
)
from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
    fused_recurrent_rwkv7,
)
from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
    rwkv7_recurrent_reference_with_checkpoints,
)