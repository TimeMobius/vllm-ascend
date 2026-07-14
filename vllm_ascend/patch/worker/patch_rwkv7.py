#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from vllm.triton_utils import HAS_TRITON

if not HAS_TRITON:
    pass  # No patching needed without Triton
else:
    import torch

    from vllm_ascend.ops.triton.fla.rwkv7_mix6 import (
        rwkv7_mix6 as _mix6_ascend,
        rwkv7_mix6_reference as _mix6_ref_ascend,
    )
    from vllm_ascend.ops.triton.fla.rwkv7_kk_pre import (
        rwkv7_kk_pre as _kk_pre_ascend,
        rwkv7_kk_pre_reference as _kk_pre_ref_ascend,
    )
    from vllm_ascend.ops.triton.fla.rwkv7_epilogue import (
        rwkv7_lnx_rkvres_xg as _epilogue_ascend,
        rwkv7_lnx_rkvres_xg_reference as _epilogue_ref_ascend,
    )
    from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7 import (
        fused_recurrent_rwkv7 as _fused_rec_ascend,
    )
    from vllm_ascend.ops.triton.fla.fused_recurrent_rwkv7_ref import (
        rwkv7_recurrent_reference_with_checkpoints as _ref_checkpoint,
    )

    def _is_npu_or_cuda(tensor: torch.Tensor) -> bool:
        return tensor.device.type in ("npu", "cuda")

    def _fused_mul_recurrent_rwkv7(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        out, final_state, _ = _fused_rec_ascend(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        if output_final_state:
            return out, final_state
        return out

    def _fused_mul_recurrent_rwkv7_with_checkpoints(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        checkpoint_positions: torch.Tensor,
        checkpoint_offsets: torch.Tensor,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        out, final_state, checkpoint_states = _fused_rec_ascend(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            output_checkpoint_states=True,
        )
        assert checkpoint_states is not None
        return out, final_state, checkpoint_states

    def _patch_rwkv7_ops():
        import vllm.model_executor.layers.fla.ops.rwkv7 as ops_module
        import vllm.model_executor.layers.fla.ops as ops_namespace

        ops_module.rwkv7_mix6 = _mix6_ascend
        ops_module.rwkv7_mix6_reference = _mix6_ref_ascend
        ops_module.rwkv7_kk_pre = _kk_pre_ascend
        ops_module.rwkv7_kk_pre_reference = _kk_pre_ref_ascend
        ops_module.rwkv7_lnx_rkvres_xg = _epilogue_ascend
        ops_module.rwkv7_lnx_rkvres_xg_reference = _epilogue_ref_ascend
        ops_module.fused_mul_recurrent_rwkv7 = _fused_mul_recurrent_rwkv7
        ops_module.fused_mul_recurrent_rwkv7_with_checkpoints = (
            _fused_mul_recurrent_rwkv7_with_checkpoints
        )
        ops_module.rwkv7_recurrent_reference_with_checkpoints = _ref_checkpoint

        ops_namespace.rwkv7_mix6 = _mix6_ascend
        ops_namespace.rwkv7_mix6_reference = _mix6_ref_ascend
        ops_namespace.rwkv7_kk_pre = _kk_pre_ascend
        ops_namespace.rwkv7_kk_pre_reference = _kk_pre_ref_ascend
        ops_namespace.rwkv7_lnx_rkvres_xg = _epilogue_ascend
        ops_namespace.rwkv7_lnx_rkvres_xg_reference = _epilogue_ref_ascend
        ops_namespace.fused_mul_recurrent_rwkv7 = _fused_mul_recurrent_rwkv7
        ops_namespace.fused_mul_recurrent_rwkv7_with_checkpoints = (
            _fused_mul_recurrent_rwkv7_with_checkpoints
        )
        ops_namespace.rwkv7_recurrent_reference_with_checkpoints = _ref_checkpoint

    def _patch_rwkv7_model():
        import vllm.model_executor.models.rwkv7 as model_module

        RWKV7Attention = model_module.RWKV7Attention
        RWKV7FeedForward = model_module.RWKV7FeedForward

        _original_alt_recurrent = model_module._can_use_rwkv7_alt_recurrent

        def patched_mix_recurrent_inputs(self, hidden_states, delta):
            x_r = self.x_r.squeeze(0).squeeze(0)
            x_w = self.x_w.squeeze(0).squeeze(0)
            x_k = self.x_k.squeeze(0).squeeze(0)
            x_v = self.x_v.squeeze(0).squeeze(0)
            x_a = self.x_a.squeeze(0).squeeze(0)
            x_g = self.x_g.squeeze(0).squeeze(0)

            if self.perf_flags.use_fused_mix6 and _is_npu_or_cuda(hidden_states):
                return _mix6_ascend(
                    hidden_states=hidden_states,
                    delta=delta,
                    x_r=x_r,
                    x_w=x_w,
                    x_k=x_k,
                    x_v=x_v,
                    x_a=x_a,
                    x_g=x_g,
                )
            return _mix6_ref_ascend(
                hidden_states=hidden_states,
                delta=delta,
                x_r=x_r,
                x_w=x_w,
                x_k=x_k,
                x_v=x_v,
                x_a=x_a,
                x_g=x_g,
            )

        def patched_prepare_recurrent_key_terms(self, k, a):
            local_k_k = self.k_k[self.key_start : self.key_end].view(
                self.local_num_heads, self.head_dim
            )
            local_k_a = self.k_a[self.key_start : self.key_end].view(
                self.local_num_heads, self.head_dim
            )
            if self.perf_flags.use_fused_kk_pre and _is_npu_or_cuda(k):
                return _kk_pre_ascend(
                    k=k,
                    k_k=local_k_k.to(torch.float32),
                    a=a,
                    k_a=local_k_a.to(torch.float32),
                )
            return _kk_pre_ref_ascend(
                k=k,
                k_k=local_k_k.to(torch.float32),
                a=a,
                k_a=local_k_a.to(torch.float32),
            )

        def patched_finalize_attention_output(
            self, recurrent_output, r, k, v, g, hidden_dtype
        ):
            output = recurrent_output.reshape(-1, self.local_value_dim)
            if self.perf_flags.use_fused_lnx_rkvres_xg and _is_npu_or_cuda(
                recurrent_output
            ):
                weight = self.g_norm.weight[self.value_start : self.value_end].contiguous()
                bias = self.g_norm.bias[self.value_start : self.value_end].contiguous()
                local_r_k = self.r_k[
                    self.tp_rank * self.local_num_heads : (self.tp_rank + 1)
                    * self.local_num_heads
                ].to(torch.float32)
                output = _epilogue_ascend(
                    recurrent_output=recurrent_output.contiguous(),
                    r=r.contiguous(),
                    k=k.contiguous(),
                    v=v.contiguous(),
                    r_k=local_r_k.contiguous(),
                    weight=weight,
                    bias=bias,
                    g=g.contiguous(),
                    eps=self.g_norm.eps,
                    output_dtype=hidden_dtype,
                )
            else:
                output = self.g_norm(output)
                local_r_k = self.r_k[
                    self.tp_rank * self.local_num_heads : (self.tp_rank + 1)
                    * self.local_num_heads
                ].to(torch.float32)
                correction = (
                    (r * k * local_r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v
                ).reshape(-1, self.local_value_dim)
                output = (output + correction) * g.to(torch.float32)
                output = output.to(hidden_dtype)
            if self.perf_flags.use_direct_linear and model_module._can_use_rwkv7_direct_linear(
                self.o_proj, output
            ):
                return model_module._rwkv7_direct_linear(self.o_proj, output)
            output, _ = self.o_proj(output)
            return output

        def patched_can_use_fused_recurrent(hidden_states):
            if hidden_states.device.type not in ("npu", "cuda"):
                return False
            return model_module._rwkv7_packed_prefill_enabled()
        model_module._can_use_rwkv7_fused_recurrent = patched_can_use_fused_recurrent

        def patched_can_use_alt_recurrent(
            *, hidden_states, r, w, k, v, kk, a, recurrent_state, head_dim, head_v_dim
        ):
            if hidden_states.device.type == "cuda":
                return _original_alt_recurrent(
                    hidden_states=hidden_states,
                    r=r, w=w, k=k, v=v, kk=kk, a=a,
                    recurrent_state=recurrent_state,
                    head_dim=head_dim,
                    head_v_dim=head_v_dim,
                )
            if hidden_states.device.type != "npu":
                return False
            if head_dim != 64 or head_v_dim != 64:
                return False
            if hidden_states.numel() == 0:
                return False
            return False  # NPU uses fused path, not alt_recurrent
        model_module._can_use_rwkv7_alt_recurrent = patched_can_use_alt_recurrent

        RWKV7Attention._mix_recurrent_inputs = patched_mix_recurrent_inputs
        RWKV7Attention._prepare_recurrent_key_terms = patched_prepare_recurrent_key_terms
        RWKV7Attention._finalize_attention_output = patched_finalize_attention_output

    _patch_rwkv7_ops()
    _patch_rwkv7_model()