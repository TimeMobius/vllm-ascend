from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import fused_recurrent_rwkv7
from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
    rwkv7_recurrent_reference,
    rwkv7_recurrent_reference_with_checkpoints,
)
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
from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 import (
    rwkv7_recurrent_t1,
    _rwkv7_recurrent_t1_reference,
)


def fused_mul_recurrent_rwkv7(*args, **kwargs):
    output, final_state, _ = fused_recurrent_rwkv7(*args, **kwargs)
    return output, final_state

__all__ = [
    "fused_recurrent_rwkv7",
    "fused_mul_recurrent_rwkv7",
    "rwkv7_recurrent_reference",
    "rwkv7_recurrent_reference_with_checkpoints",
    "rwkv7_lnx_rkvres_xg",
    "rwkv7_lnx_rkvres_xg_reference",
    "rwkv7_kk_pre",
    "rwkv7_kk_pre_reference",
    "rwkv7_mix6",
    "rwkv7_mix6_reference",
    "rwkv7_recurrent_t1",
    "_rwkv7_recurrent_t1_reference",
]
