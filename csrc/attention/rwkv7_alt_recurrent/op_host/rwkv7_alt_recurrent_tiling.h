/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rwkv7_alt_recurrent_tiling.h
 * \brief RWKV7 alt recurrent tiling header.
 */
#ifndef __RWKV7_ALT_RECURRENT_TILING_H__
#define __RWKV7_ALT_RECURRENT_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/error_log.h"
#include "../op_kernel/rwkv7_alt_recurrent_tiling_data.h"

namespace optiling {
using namespace RWKV7AltRecurrent;

struct RWKV7AltRecurrentCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct RWKV7AltRecurrentInfo {
public:
    int64_t usedCoreNum = 0;
    const char* opName = "RWKV7AltRecurrent";
};

class RWKV7AltRecurrentTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RWKV7AltRecurrentTiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~RWKV7AltRecurrentTiling() override = default;

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();

    using HostRuleFn = ge::graphStatus(RWKV7AltRecurrentTiling::*)();

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus AnalyzeFormat();
    ge::graphStatus RuleCheckShapeDimAndRelation();
    ge::graphStatus RuleFillTilingShapeData();
    ge::graphStatus RuleUpdateDynamicBlockDimByTaskUnits();

    bool CheckDim(const gert::Shape shape, const size_t dim, const std::string& dimDesc);
    bool CheckFormat(ge::Format format, const std::string& Desc);

    ge::graphStatus CheckShapeDimAndRelation(const gert::Shape& rShape, const gert::Shape& wShape,
                                             const gert::Shape& kShape, const gert::Shape& vShape,
                                             const gert::Shape& kkShape, const gert::Shape& aShape,
                                             const gert::Shape& stateShape);
    void FillTilingShapeData(const gert::Shape& rShape);
    void UpdateDynamicBlockDimByTaskUnits();

    RWKV7AltRecurrentCompileInfo compileInfo_;
    RWKV7AltTilingData tilingData_;
    RWKV7AltRecurrentInfo inputParams_;
};

} // namespace optiling
#endif