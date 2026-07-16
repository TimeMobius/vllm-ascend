/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_RWKV7_ALT_RECURRENT_H
#define OP_API_ACLNN_RWKV7_ALT_RECURRENT_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RWKV7AltRecurrent first-stage interface, calculates workspace size.
 * @param [in] r: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] w: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] k: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] v: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] kk: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] a: input tensor, data type support: float32, shape: [B, T, H, D].
 * @param [in] initial_state: optional input tensor, data type support: float32, shape: [B, H, D, D].
 * @param [out] out: output tensor, data type support: float32, shape: [B, T, H, D].
 * @param [out] final_state: output tensor, data type support: float32, shape: [B, H, D, D].
 * @param [out] workspaceSize: returns workspace size needed on NPU device.
 * @param [out] executor: returns op executor containing the operator calculation flow.
 * @return aclnnStatus: returns status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRWKV7AltRecurrentGetWorkspaceSize(
    const aclTensor *r, const aclTensor *w, const aclTensor *k, const aclTensor *v,
    const aclTensor *kk, const aclTensor *a, const aclTensor *initialState,
    aclTensor *out, aclTensor *finalState, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief RWKV7AltRecurrent second-stage interface, executes the operator.
 * @param [in] workspace: workspace memory address on NPU device.
 * @param [in] workspaceSize: workspace size on NPU device, obtained from first-stage interface.
 * @param [in] executor: op executor containing the operator calculation flow.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: returns status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRWKV7AltRecurrent(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_RWKV7_ALT_RECURRENT_H