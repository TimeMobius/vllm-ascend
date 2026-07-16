/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_rwkv7_alt_recurrent.cpp
 * \brief RWKV7AltRecurrent ACLNN API implementation.
 */
#include <dlfcn.h>
#include "aclnn_rwkv7_alt_recurrent.h"
#include "../rwkv7_alt_recurrent.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t R_DIM_NUM = 4;
constexpr size_t W_DIM_NUM = 4;
constexpr size_t K_DIM_NUM = 4;
constexpr size_t V_DIM_NUM = 4;
constexpr size_t KK_DIM_NUM = 4;
constexpr size_t A_DIM_NUM = 4;
constexpr size_t STATE_DIM_NUM = 4;
constexpr size_t OUT_DIM_NUM = 4;
constexpr size_t FINAL_STATE_DIM_NUM = 4;

struct RWKV7AltRecurrentParams {
    // mandatory inputs
    const aclTensor *r {nullptr};
    const aclTensor *w {nullptr};
    const aclTensor *k {nullptr};
    const aclTensor *v {nullptr};
    const aclTensor *kk {nullptr};
    const aclTensor *a {nullptr};
    // optional input
    const aclTensor *initial_state {nullptr};
    // outputs
    const aclTensor *out {nullptr};
    const aclTensor *final_state {nullptr};
};

// support dtype
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FINAL_STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const RWKV7AltRecurrentParams &params)
{
    OP_CHECK_NULL(params.r, return false);
    OP_CHECK_NULL(params.w, return false);
    OP_CHECK_NULL(params.k, return false);
    OP_CHECK_NULL(params.v, return false);
    OP_CHECK_NULL(params.kk, return false);
    OP_CHECK_NULL(params.a, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.final_state, return false);

    return true;
}

static inline bool CheckDtypeValid(const RWKV7AltRecurrentParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.r, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.w, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.k, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.v, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.kk, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a, QKV_TYPE_SUPPORT_LIST, return false);

    if (params.initial_state != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.initial_state, STATE_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.final_state, FINAL_STATE_TYPE_SUPPORT_LIST, return false);

    return true;
}

static aclnnStatus CheckParams(RWKV7AltRecurrentParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("RWKV7AltRecurrent check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(RWKV7AltRecurrentParams &params)
{
    params.r->SetOriginalShape(params.r->GetViewShape());
    params.w->SetOriginalShape(params.w->GetViewShape());
    params.k->SetOriginalShape(params.k->GetViewShape());
    params.v->SetOriginalShape(params.v->GetViewShape());
    params.kk->SetOriginalShape(params.kk->GetViewShape());
    params.a->SetOriginalShape(params.a->GetViewShape());

    if (params.initial_state != nullptr) {
        params.initial_state->SetOriginalShape(params.initial_state->GetViewShape());
    }

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRWKV7AltRecurrentGetWorkspaceSize(
    const aclTensor *r, const aclTensor *w, const aclTensor *k, const aclTensor *v,
    const aclTensor *kk, const aclTensor *a, const aclTensor *initialState,
    aclTensor *out, aclTensor *finalState, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRWKV7AltRecurrent,
                   DFX_IN(r, w, k, v, kk, a, initialState),
                   DFX_OUT(out, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RWKV7AltRecurrentParams params {r, w, k, v, kk, a, initialState, out, finalState};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto r_ = l0op::Contiguous(r, uniqueExecutor.get());
    auto w_ = l0op::Contiguous(w, uniqueExecutor.get());
    auto k_ = l0op::Contiguous(k, uniqueExecutor.get());
    auto v_ = l0op::Contiguous(v, uniqueExecutor.get());
    auto kk_ = l0op::Contiguous(kk, uniqueExecutor.get());
    auto a_ = l0op::Contiguous(a, uniqueExecutor.get());

    const aclTensor *initialState_ = nullptr;
    if (initialState != nullptr) {
        initialState_ = l0op::Contiguous(initialState, uniqueExecutor.get());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());
    auto finalState_ = l0op::Contiguous(finalState, uniqueExecutor.get());

    // Call L0 interface
    auto result = l0op::RWKV7AltRecurrent(r_, w_, k_, v_, kk_, a_, initialState_, uniqueExecutor.get());
    if (result.out == nullptr || result.final_state == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // Copy view result to output
    auto viewCopyOut = l0op::ViewCopy(result.out, out_, uniqueExecutor.get());
    if (viewCopyOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyFinalState = l0op::ViewCopy(result.final_state, finalState_, uniqueExecutor.get());
    if (viewCopyFinalState == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // Get workspace size needed for computation
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRWKV7AltRecurrent(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRWKV7AltRecurrent);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif