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
 * \brief RWKV7 alt recurrent kernel entry for Ascend NPU.
 */
#include "rwkv7_alt_recurrent_tiling_data.h"
#include "rwkv7_alt_recurrent.h"

using namespace AscendC;
using namespace RWKV7AltRecurrent;

extern "C" __global__ __aicore__ void
rwkv7_alt_recurrent(GM_ADDR r, GM_ADDR w, GM_ADDR k, GM_ADDR v, GM_ADDR kk, GM_ADDR a,
                     GM_ADDR initialState, GM_ADDR out, GM_ADDR finalState,
                     GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(RWKV7AltTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    RWKV7AltKernel<float, float> op(&tilingData);
    RWKV7AltInitParams initParams{r, w, k, v, kk, a, initialState, out, finalState};
    op.Init(initParams, &pipe);
    op.Process();
}