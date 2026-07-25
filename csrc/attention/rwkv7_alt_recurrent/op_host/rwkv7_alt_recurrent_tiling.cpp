/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rwkv7_alt_recurrent_tiling.cpp
 * \brief RWKV7 alt recurrent tiling implementation.
 */
#include "rwkv7_alt_recurrent_tiling.h"

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "error/ops_error.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RWKV7AltRecurrent, RWKV7AltRecurrentTiling, 0);

const size_t R_INDEX = 0;
const size_t W_INDEX = 1;
const size_t K_INDEX = 2;
const size_t V_INDEX = 3;
const size_t KK_INDEX = 4;
const size_t A_INDEX = 5;
const size_t INITIAL_STATE_INDEX = 6;

const size_t QKV_DIM_NUM = 4;
const size_t STATE_DIM_NUM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

void RWKV7AltRecurrentTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.vectorCoreNum = compileInfo_.aivNum;
}

ge::graphStatus RWKV7AltRecurrentTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; };

ge::graphStatus RWKV7AltRecurrentTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid context."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid dtypes."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid shapes."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "Invalid Format."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::DoOpTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::DoLibApiTiling()
{
    tilingKey_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t RWKV7AltRecurrentTiling::GetTilingKey() const { return tilingKey_; };

ge::graphStatus RWKV7AltRecurrentTiling::GetWorkspaceSize()
{
    constexpr int64_t sysWorkspaceSize = 16777216; // 16M
    workspaceSize_ = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto tilingDataSize = sizeof(RWKV7AltTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                           context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(R_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(R_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(W_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(W_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(K_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(K_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(V_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(V_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::AnalyzeDtype()
{
    auto rDtype = context_->GetInputDesc(R_INDEX)->GetDataType();
    auto wDtype = context_->GetInputDesc(W_INDEX)->GetDataType();
    auto kDtype = context_->GetInputDesc(K_INDEX)->GetDataType();
    auto vDtype = context_->GetInputDesc(V_INDEX)->GetDataType();
    auto kkDtype = context_->GetInputDesc(KK_INDEX)->GetDataType();
    auto aDtype = context_->GetInputDesc(A_INDEX)->GetDataType();

    OP_CHECK_IF(rDtype != ge::DT_FLOAT || wDtype != ge::DT_FLOAT || kDtype != ge::DT_FLOAT ||
                    vDtype != ge::DT_FLOAT || kkDtype != ge::DT_FLOAT || aDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "All input dtypes should be float32"), return ge::GRAPH_FAILED);

    if (context_->GetInputDesc(INITIAL_STATE_INDEX) != nullptr) {
        auto stateDtype = context_->GetInputDesc(INITIAL_STATE_INDEX)->GetDataType();
        OP_CHECK_IF(stateDtype != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "initial_state dtype should be float32"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool RWKV7AltRecurrentTiling::CheckDim(const gert::Shape shape, const size_t dim,
                                        const std::string& dimDesc)
{
    if (shape.GetDimNum() != dim) {
        OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu",
                dimDesc.c_str(), dim, shape.GetDimNum());
        return false;
    }
    return true;
}

ge::graphStatus RWKV7AltRecurrentTiling::CheckShapeDimAndRelation(
    const gert::Shape& rShape, const gert::Shape& wShape, const gert::Shape& kShape,
    const gert::Shape& vShape, const gert::Shape& kkShape, const gert::Shape& aShape,
    const gert::Shape& stateShape)
{
    // All QKV tensors [B, T, H, D] with D=64
    if (!CheckDim(rShape, QKV_DIM_NUM, "r") || !CheckDim(wShape, QKV_DIM_NUM, "w") ||
        !CheckDim(kShape, QKV_DIM_NUM, "k") || !CheckDim(vShape, QKV_DIM_NUM, "v") ||
        !CheckDim(kkShape, QKV_DIM_NUM, "kk") || !CheckDim(aShape, QKV_DIM_NUM, "a")) {
        return ge::GRAPH_FAILED;
    }

    // Check D == 64
    if (rShape.GetDim(DIM_3) != 64 || wShape.GetDim(DIM_3) != 64 || kShape.GetDim(DIM_3) != 64 ||
        vShape.GetDim(DIM_3) != 64 || kkShape.GetDim(DIM_3) != 64 || aShape.GetDim(DIM_3) != 64) {
        OP_LOGE(context_->GetNodeName(), "head_dim must be 64");
        return ge::GRAPH_FAILED;
    }

    // Check batch, seq_len, num_heads match
    if (rShape.GetDim(DIM_0) != wShape.GetDim(DIM_0) || rShape.GetDim(DIM_0) != kShape.GetDim(DIM_0) ||
        rShape.GetDim(DIM_0) != vShape.GetDim(DIM_0) || rShape.GetDim(DIM_1) != wShape.GetDim(DIM_1) ||
        rShape.GetDim(DIM_1) != kShape.GetDim(DIM_1) || rShape.GetDim(DIM_1) != vShape.GetDim(DIM_1) ||
        rShape.GetDim(DIM_2) != wShape.GetDim(DIM_2) || rShape.GetDim(DIM_2) != kShape.GetDim(DIM_2) ||
        rShape.GetDim(DIM_2) != vShape.GetDim(DIM_2)) {
        OP_LOGE(context_->GetNodeName(), "r, w, k, v dimension mismatch");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void RWKV7AltRecurrentTiling::FillTilingShapeData(const gert::Shape& rShape)
{
    tilingData_.batch = static_cast<uint32_t>(rShape.GetDim(DIM_0));
    tilingData_.seqLen = static_cast<uint32_t>(rShape.GetDim(DIM_1));
    tilingData_.numHeads = static_cast<uint32_t>(rShape.GetDim(DIM_2));
    tilingData_.headDim = static_cast<uint32_t>(rShape.GetDim(DIM_3));
    tilingData_.hasInitialState = (context_->GetInputDesc(INITIAL_STATE_INDEX) != nullptr) ? 1 : 0;
}

void RWKV7AltRecurrentTiling::UpdateDynamicBlockDimByTaskUnits()
{
    uint64_t taskUnits =
        static_cast<uint64_t>(tilingData_.batch) * static_cast<uint64_t>(tilingData_.numHeads);
    if (taskUnits == 0) {
        taskUnits = 1;
    }
    // The kernel processes exactly one work unit (one (batch, head) pair) per block,
    // so the launch grid must cover every task unit. Capping at the hardware core count
    // silently drops the remaining work units (e.g. batch*heads=64 with aivNum=40 leaves
    // heads 40..63 unprocessed and produces garbage in the output). Cap by the block-dim
    // launch limit instead so the runtime can time-slice across cores.
    constexpr uint64_t MAX_GRID_DIM = 65535;
    uint64_t selectedCoreNum = taskUnits > MAX_GRID_DIM ? MAX_GRID_DIM : taskUnits;
    tilingData_.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);
}

ge::graphStatus RWKV7AltRecurrentTiling::RuleCheckShapeDimAndRelation()
{
    const auto& rShape = context_->GetInputShape(R_INDEX)->GetOriginShape();
    const auto& wShape = context_->GetInputShape(W_INDEX)->GetOriginShape();
    const auto& kShape = context_->GetInputShape(K_INDEX)->GetOriginShape();
    const auto& vShape = context_->GetInputShape(V_INDEX)->GetOriginShape();
    const auto& kkShape = context_->GetInputShape(KK_INDEX)->GetOriginShape();
    const auto& aShape = context_->GetInputShape(A_INDEX)->GetOriginShape();
    gert::Shape stateShape;
    if (context_->GetInputShape(INITIAL_STATE_INDEX) != nullptr) {
        stateShape = context_->GetInputShape(INITIAL_STATE_INDEX)->GetOriginShape();
    }
    return CheckShapeDimAndRelation(rShape, wShape, kShape, vShape, kkShape, aShape, stateShape);
}

ge::graphStatus RWKV7AltRecurrentTiling::RuleFillTilingShapeData()
{
    const auto& rShape = context_->GetInputShape(R_INDEX)->GetOriginShape();
    FillTilingShapeData(rShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::RuleUpdateDynamicBlockDimByTaskUnits()
{
    UpdateDynamicBlockDimByTaskUnits();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RWKV7AltRecurrentTiling::AnalyzeShapes()
{
    struct RuleItem {
        const char* name;
        HostRuleFn fn;
    };
    const std::array<RuleItem, 3> shapeRules = {{
        {"RuleCheckShapeDimAndRelation", &RWKV7AltRecurrentTiling::RuleCheckShapeDimAndRelation},
        {"RuleFillTilingShapeData", &RWKV7AltRecurrentTiling::RuleFillTilingShapeData},
        {"RuleUpdateDynamicBlockDimByTaskUnits", &RWKV7AltRecurrentTiling::RuleUpdateDynamicBlockDimByTaskUnits},
    }};
    for (const auto& rule : shapeRules) {
        OP_CHECK_IF((this->*(rule.fn))() != ge::GRAPH_SUCCESS,
                    OP_LOGE(inputParams_.opName, "AnalyzeShapes rule failed: %s", rule.name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool RWKV7AltRecurrentTiling::CheckFormat(ge::Format format, const std::string& Desc)
{
    if (format == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
        return false;
    }
    return true;
}

ge::graphStatus RWKV7AltRecurrentTiling::AnalyzeFormat()
{
    if (!CheckFormat(context_->GetInputDesc(R_INDEX)->GetStorageFormat(), "r") ||
        !CheckFormat(context_->GetInputDesc(W_INDEX)->GetStorageFormat(), "w") ||
        !CheckFormat(context_->GetInputDesc(K_INDEX)->GetStorageFormat(), "k") ||
        !CheckFormat(context_->GetInputDesc(V_INDEX)->GetStorageFormat(), "v") ||
        !CheckFormat(context_->GetInputDesc(KK_INDEX)->GetStorageFormat(), "kk") ||
        !CheckFormat(context_->GetInputDesc(A_INDEX)->GetStorageFormat(), "a")) {
        return ge::GRAPH_FAILED;
    }
    if (context_->GetInputDesc(INITIAL_STATE_INDEX) != nullptr) {
        if (!CheckFormat(context_->GetInputDesc(INITIAL_STATE_INDEX)->GetStorageFormat(), "initial_state")) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void RWKV7AltRecurrentTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
    OP_LOGD(context_->GetNodeName(), "batch: [%u]", tilingData_.batch);
    OP_LOGD(context_->GetNodeName(), "seqLen: [%u]", tilingData_.seqLen);
    OP_LOGD(context_->GetNodeName(), "numHeads: [%u]", tilingData_.numHeads);
    OP_LOGD(context_->GetNodeName(), "headDim: [%u]", tilingData_.headDim);
}

static ge::graphStatus RWKV7AltRecurrentTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("RWKV7AltRecurrent", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForRWKV7AltRecurrent(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("RWKV7AltRecurrent", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<RWKV7AltRecurrentCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RWKV7AltRecurrent)
    .Tiling(RWKV7AltRecurrentTilingFunc)
    .TilingParse<RWKV7AltRecurrentCompileInfo>(TilingPrepareForRWKV7AltRecurrent);
} // namespace optiling