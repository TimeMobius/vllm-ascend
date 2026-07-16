/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rwkv7_alt_recurrent_infershape.cpp
 * \brief RWKV7 alt recurrent infershape implementation.
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {

const size_t V_INDEX = 3;
const size_t INITIAL_STATE_INDEX = 6;
const size_t OUT_INDEX = 0;
const size_t FINAL_STATE_INDEX = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeRWKV7AltRecurrent(InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("RWKV7AltRecurrent", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeV = context->GetInputShape(V_INDEX);
    auto shapeInitialState = context->GetInputShape(INITIAL_STATE_INDEX);
    auto shapeOut = context->GetOutputShape(OUT_INDEX);
    auto shapeFinalState = context->GetOutputShape(FINAL_STATE_INDEX);

    if (shapeV == nullptr || shapeOut == nullptr || shapeFinalState == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    // out shape = v shape [B, T, H, D]
    shapeOut->SetDimNum(4);
    for (size_t i = 0; i < 4; ++i) {
        shapeOut->SetDim(i, shapeV->GetDim(i));
    }

    // final_state shape = [B, H, D, D] where D=64
    shapeFinalState->SetDimNum(4);
    shapeFinalState->SetDim(DIM_0, shapeV->GetDim(DIM_0));  // B
    shapeFinalState->SetDim(DIM_1, shapeV->GetDim(DIM_2));  // H
    shapeFinalState->SetDim(DIM_2, shapeV->GetDim(DIM_3));  // D (64)
    shapeFinalState->SetDim(DIM_3, shapeV->GetDim(DIM_3));  // D (64)

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRWKV7AltRecurrent(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RWKV7AltRecurrent)
    .InferShape(InferShapeRWKV7AltRecurrent)
    .InferDataType(InferDataTypeRWKV7AltRecurrent);
} // namespace ops