/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms
 * and conditions of the License. You may not use this file except in compliance with the License.
 * This software is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
 * PARTICULAR PURPOSE. See the License in the root of the software repository for details.
 */

/*!
 * \file rwkv7_alt_recurrent.h
 * \brief RWKV7 alt recurrent kernel for Ascend NPU.
 * Grid: (batch * numHeads), each block processes one (batch, head) pair.
 * Each block processes ALL 64 value_idx columns, maintaining independent state for each.
 *
 * Output layout: out [B, T, H, 64], final_state [B, H, D=64, V=64]
 *
 * Internal state representation:
 *   - The recurrence uses a transposed view: the kernel stores
 *     stateMatrix[vIdx][kIdx] (v outer axis, k inner axis) and writes
 *     it to the [B, H, D, V] allocation at offset stateBase + k*V + v.
 *   - When D == V == 64, that offset equals the [B, H, D=kIdx, V=vIdx]
 *     slot of a contiguous [B, H, D, V] tensor and the stored value
 *     equals the FP32 torch reference's state[k, v].
 *   - Therefore the native final_state tensor returned to Python is
 *     already [B, H, D, V]-semantic; no transpose is required at the
 *     call site (rwkv7_ascend.models.rwkvv7 decode path and the
 *     prefill recurrent-scan patch both treat the native layout as
 *     the reference layout).
 *
 * Matches /mnt/data/Codes/vllm/csrc/rwkv7_alt_recurrent.cu math.
 */

#ifndef __RWKV7_ALT_RECURRENT_KERNEL_H_
#define __RWKV7_ALT_RECURRENT_KERNEL_H_

#include "kernel_operator.h"
#include "rwkv7_alt_recurrent_tiling_data.h"

namespace RWKV7AltRecurrent {

using namespace AscendC;
constexpr int64_t kHeadDim = 64;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint32_t REPEAT_LENGTH = 64;
constexpr uint32_t MAX_REPEAT_TIME = 255;

struct RWKV7AltInitParams {
    GM_ADDR r;
    GM_ADDR w;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR kk;
    GM_ADDR a;
    GM_ADDR initialState;
    GM_ADDR out;
    GM_ADDR finalState;
};

template <typename inType, typename stateType>
class RWKV7AltKernel {
public:
    __aicore__ inline RWKV7AltKernel(const RWKV7AltTilingData* tilingData)
    {
        batch_ = tilingData->batch;
        seqLen_ = tilingData->seqLen;
        numHeads_ = tilingData->numHeads;
        headDim_ = tilingData->headDim;
        hasInitialState_ = (tilingData->hasInitialState == 1);
    }

    __aicore__ inline void Init(const RWKV7AltInitParams& initParams, TPipe* pipe)
    {
        blockIdx = GetBlockIdx();
        pipe_ = pipe;
        SetGlobalTensors(initParams);
        InitLocalBuffers();
    }

    __aicore__ inline void SetGlobalTensors(const RWKV7AltInitParams& initParams)
    {
        rGm_.SetGlobalBuffer((__gm__ inType*)initParams.r);
        wGm_.SetGlobalBuffer((__gm__ inType*)initParams.w);
        kGm_.SetGlobalBuffer((__gm__ inType*)initParams.k);
        vGm_.SetGlobalBuffer((__gm__ inType*)initParams.v);
        kkGm_.SetGlobalBuffer((__gm__ inType*)initParams.kk);
        aGm_.SetGlobalBuffer((__gm__ inType*)initParams.a);
        initStateGm_.SetGlobalBuffer((__gm__ stateType*)initParams.initialState);
        outGm_.SetGlobalBuffer((__gm__ float*)initParams.out);
        finalStateGm_.SetGlobalBuffer((__gm__ stateType*)initParams.finalState);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(rInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(wInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(kInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(kkInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(aInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(vInQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(expWQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(negKkQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(kkAQueue_, BUFFER_NUM, headDim_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, headDim_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t taskUnits = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(numHeads_);
        if (blockIdx >= taskUnits) {
            return;
        }
        uint64_t batchIdx = blockIdx / numHeads_;
        uint64_t headIdx = blockIdx % numHeads_;
        ProcessBatchHead(batchIdx, headIdx);
    }

private:
    __aicore__ inline void ProcessBatchHead(uint64_t batchIdx, uint64_t headIdx)
    {
        float stateMatrix[kHeadDim][kHeadDim];
        if (!hasInitialState_) {
            for (int64_t vIdx = 0; vIdx < kHeadDim; ++vIdx) {
                for (int64_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                    stateMatrix[vIdx][kIdx] = 0.0f;
                }
            }
        } else {
            uint64_t stateBase = ((batchIdx * numHeads_ + headIdx) * kHeadDim) * kHeadDim;
            for (int64_t vIdx = 0; vIdx < kHeadDim; ++vIdx) {
                for (int64_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                    uint64_t offset = stateBase + kIdx * kHeadDim + vIdx;
                    stateMatrix[vIdx][kIdx] = initStateGm_.GetValue(offset);
                }
            }
        }

        for (int64_t tokenIdx = 0; tokenIdx < seqLen_; ++tokenIdx) {
            ProcessToken(batchIdx, headIdx, tokenIdx, stateMatrix);
        }

        uint64_t stateBase = ((batchIdx * numHeads_ + headIdx) * kHeadDim) * kHeadDim;
        for (int64_t vIdx = 0; vIdx < kHeadDim; ++vIdx) {
            for (int64_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                uint64_t offset = stateBase + kIdx * kHeadDim + vIdx;
                finalStateGm_.SetValue(offset, stateMatrix[vIdx][kIdx]);
            }
        }
    }

    __aicore__ inline void ProcessToken(uint64_t batchIdx, uint64_t headIdx,
                                        int64_t tokenIdx, float stateMatrix[][kHeadDim])
    {
        uint64_t tensorBase = (((batchIdx * seqLen_ + tokenIdx) * numHeads_ + headIdx) * kHeadDim);

        LocalTensor<float> rLocal = rInQueue_.AllocTensor<float>();
        LocalTensor<float> wLocal = wInQueue_.AllocTensor<float>();
        LocalTensor<float> kLocal = kInQueue_.AllocTensor<float>();
        LocalTensor<float> kkLocal = kkInQueue_.AllocTensor<float>();
        LocalTensor<float> aLocal = aInQueue_.AllocTensor<float>();
        LocalTensor<float> vLocal = vInQueue_.AllocTensor<float>();

        DataCopyExtParams copyParams{1, static_cast<uint16_t>(headDim_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        DataCopyPad(rLocal, rGm_[tensorBase], copyParams, padParams);
        DataCopyPad(wLocal, wGm_[tensorBase], copyParams, padParams);
        DataCopyPad(kLocal, kGm_[tensorBase], copyParams, padParams);
        DataCopyPad(kkLocal, kkGm_[tensorBase], copyParams, padParams);
        DataCopyPad(aLocal, aGm_[tensorBase], copyParams, padParams);
        DataCopyPad(vLocal, vGm_[tensorBase], copyParams, padParams);

        rInQueue_.EnQue<float>(rLocal);
        wInQueue_.EnQue<float>(wLocal);
        kInQueue_.EnQue<float>(kLocal);
        kkInQueue_.EnQue<float>(kkLocal);
        aInQueue_.EnQue<float>(aLocal);
        vInQueue_.EnQue<float>(vLocal);

        rLocal = rInQueue_.DeQue<float>();
        wLocal = wInQueue_.DeQue<float>();
        kLocal = kInQueue_.DeQue<float>();
        kkLocal = kkInQueue_.DeQue<float>();
        aLocal = aInQueue_.DeQue<float>();
        vLocal = vInQueue_.DeQue<float>();

        LocalTensor<float> expWLocal = expWQueue_.AllocTensor<float>();
        Exp(expWLocal, wLocal, headDim_);
        expWQueue_.EnQue<float>(expWLocal);
        expWLocal = expWQueue_.DeQue<float>();

        LocalTensor<float> negKkLocal = negKkQueue_.AllocTensor<float>();
        LocalTensor<float> kkALocal = kkAQueue_.AllocTensor<float>();
        Muls(negKkLocal, kkLocal, -1.0f, headDim_);
        Mul(kkALocal, kkLocal, aLocal, headDim_);
        negKkQueue_.EnQue<float>(negKkLocal);
        kkAQueue_.EnQue<float>(kkALocal);
        negKkLocal = negKkQueue_.DeQue<float>();
        kkALocal = kkAQueue_.DeQue<float>();

        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

        for (int64_t vIdx = 0; vIdx < kHeadDim; ++vIdx) {
            float vVal = vLocal.GetValue(vIdx);
            float sa = 0.0f;
            for (int64_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                sa += stateMatrix[vIdx][kIdx] * negKkLocal.GetValue(kIdx);
            }

            float outVal = 0.0f;
            for (int64_t kIdx = 0; kIdx < kHeadDim; ++kIdx) {
                float newState = stateMatrix[vIdx][kIdx] * expWLocal.GetValue(kIdx) +
                                 kkALocal.GetValue(kIdx) * sa +
                                 kLocal.GetValue(kIdx) * vVal;
                stateMatrix[vIdx][kIdx] = newState;
                outVal += newState * rLocal.GetValue(kIdx);
            }
            outLocal.SetValue(vIdx, outVal);
        }

        outQueue_.EnQue<float>(outLocal);
        outLocal = outQueue_.DeQue<float>();

        DataCopyExtParams outCopyParams{1, static_cast<uint16_t>(headDim_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm_[tensorBase], outLocal, outCopyParams);

        rInQueue_.FreeTensor(rLocal);
        wInQueue_.FreeTensor(wLocal);
        kInQueue_.FreeTensor(kLocal);
        kkInQueue_.FreeTensor(kkLocal);
        aInQueue_.FreeTensor(aLocal);
        vInQueue_.FreeTensor(vLocal);
        expWQueue_.FreeTensor(expWLocal);
        negKkQueue_.FreeTensor(negKkLocal);
        kkAQueue_.FreeTensor(kkALocal);
        outQueue_.FreeTensor(outLocal);
    }

private:
    GlobalTensor<inType> rGm_;
    GlobalTensor<inType> wGm_;
    GlobalTensor<inType> kGm_;
    GlobalTensor<inType> vGm_;
    GlobalTensor<inType> kkGm_;
    GlobalTensor<inType> aGm_;
    GlobalTensor<stateType> initStateGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<stateType> finalStateGm_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> rInQueue_;
    TQue<QuePosition::VECIN, 1> wInQueue_;
    TQue<QuePosition::VECIN, 1> kInQueue_;
    TQue<QuePosition::VECIN, 1> vInQueue_;
    TQue<QuePosition::VECIN, 1> kkInQueue_;
    TQue<QuePosition::VECIN, 1> aInQueue_;
    TQue<QuePosition::VECIN, 1> expWQueue_;
    TQue<QuePosition::VECIN, 1> negKkQueue_;
    TQue<QuePosition::VECIN, 1> kkAQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    uint32_t batch_;
    uint32_t seqLen_;
    uint32_t numHeads_;
    uint32_t headDim_;
    uint64_t blockIdx;
    bool hasInitialState_;
};

}
#endif