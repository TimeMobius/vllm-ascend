/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rwkv7_alt_recurrent_tiling_data.h
 * \brief RWKV7 alt recurrent tiling data for Ascend NPU.
 */
#ifndef RWKV7_ALT_RECURRENT_TILING_DATA_H
#define RWKV7_ALT_RECURRENT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace RWKV7AltRecurrent {

#pragma pack(push, 8)
struct alignas(8) RWKV7AltTilingData {
    uint32_t vectorCoreNum;
    uint32_t batch;
    uint32_t seqLen;
    uint32_t numHeads;
    uint32_t headDim;
    uint32_t hasInitialState;  // 1 if initialState is provided, 0 otherwise
};
#pragma pack(pop)

} // namespace RWKV7AltRecurrent

#endif // RWKV7_ALT_RECURRENT_TILING_DATA_H