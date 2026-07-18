/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef RWKV7_ALT_RECURRENT_TORCH_ADPT_H
#define RWKV7_ALT_RECURRENT_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_rwkv7_alt_recurrent(
    const at::Tensor& r,
    const at::Tensor& w,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& kk,
    const at::Tensor& a,
    const c10::optional<at::Tensor>& initial_state)
{
    TORCH_CHECK(r.device().type() == c10::DeviceType::PrivateUse1, "`r` must be a NPU tensor.");
    TORCH_CHECK(r.scalar_type() == at::ScalarType::Float, "`r` must have dtype float32.");
    TORCH_CHECK(r.is_contiguous(), "`r` must be contiguous.");
    TORCH_CHECK(r.dim() == 4, "`r` must be 4D, got ", r.dim(), ".");

    const int64_t batch_size = r.size(0);
    const int64_t seq_len = r.size(1);
    const int64_t num_heads = r.size(2);
    const int64_t head_dim = r.size(3);

    TORCH_CHECK(head_dim == 64, "`r` must use head_dim=64, got ", head_dim, ".");

    // Check other tensors
    TORCH_CHECK(w.dim() == 4 && w.size(0) == batch_size && w.size(1) == seq_len &&
                    w.size(2) == num_heads && w.size(3) == 64,
                "`w` shape mismatch");
    TORCH_CHECK(k.dim() == 4 && k.size(0) == batch_size && k.size(1) == seq_len &&
                    k.size(2) == num_heads && k.size(3) == 64,
                "`k` shape mismatch");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == batch_size && v.size(1) == seq_len &&
                    v.size(2) == num_heads && v.size(3) == 64,
                "`v` shape mismatch");
    TORCH_CHECK(kk.dim() == 4 && kk.size(0) == batch_size && kk.size(1) == seq_len &&
                    kk.size(2) == num_heads && kk.size(3) == 64,
                "`kk` shape mismatch");
    TORCH_CHECK(a.dim() == 4 && a.size(0) == batch_size && a.size(1) == seq_len &&
                    a.size(2) == num_heads && a.size(3) == 64,
                "`a` shape mismatch");

    if (initial_state.has_value()) {
        const at::Tensor& h0 = *initial_state;
        TORCH_CHECK(h0.device().type() == c10::DeviceType::PrivateUse1, "`initial_state` must be a NPU tensor.");
        TORCH_CHECK(h0.scalar_type() == at::ScalarType::Float,
                    "`initial_state` must have dtype float32.");
        TORCH_CHECK(h0.is_contiguous(), "`initial_state` must be contiguous.");
        TORCH_CHECK(h0.dim() == 4 && h0.size(0) == batch_size && h0.size(1) == num_heads &&
                        h0.size(2) == 64 && h0.size(3) == 64,
                    "`initial_state` must have shape [", batch_size, ", ", num_heads, ", 64, 64]");
    }

    auto out = at::empty_like(v, v.options().dtype(at::ScalarType::Float));
    auto final_state = at::empty({batch_size, num_heads, 64, 64}, r.options().dtype(at::ScalarType::Float));

    EXEC_NPU_CMD(aclnnRWKV7AltRecurrent,
                 r,
                 w,
                 k,
                 v,
                 kk,
                 a,
                 initial_state,
                 out,
                 final_state);

    return {out, final_state};
}

} // namespace vllm_ascend
#endif