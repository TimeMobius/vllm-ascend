/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rwkv7_alt_recurrent.cpp
 * \brief RWKV7AltRecurrent L0 operator registration.
 */
#include "../rwkv7_alt_recurrent.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RWKV7AltRecurrent);

RWKV7AltRecurrentOutput RWKV7AltRecurrent(const aclTensor *r, const aclTensor *w, const aclTensor *k,
                                          const aclTensor *v, const aclTensor *kk, const aclTensor *a,
                                          const aclTensor *initialState, aclOpExecutor *executor)
{
    L0_DFX(RWKV7AltRecurrent, r, w, k, v, kk, a, initialState);

    DataType outType = DataType::DT_FLOAT;
    Format format = Format::FORMAT_ND;

    auto out = executor->AllocTensor(outType, format, format);
    auto finalState = executor->AllocTensor(outType, format, format);

    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed.");
        return {nullptr, nullptr};
    }
    if (finalState == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "finalState AllocTensor failed.");
        return {nullptr, nullptr};
    }

    // infershape
    auto ret = INFER_SHAPE(
        RWKV7AltRecurrent,
        OP_INPUT(r, w, k, v, kk, a, initialState),
        OP_OUTPUT(out, finalState));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "RWKV7AltRecurrent InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RWKV7AltRecurrent,
        OP_INPUT(r, w, k, v, kk, a, initialState),
        OP_OUTPUT(out, finalState));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "RWKV7AltRecurrent ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }

    return {out, finalState};
}

} // namespace l0op