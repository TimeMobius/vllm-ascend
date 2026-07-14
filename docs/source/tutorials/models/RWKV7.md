# RWKV7

## 1 Introduction

RWKV7 是线性注意力（Rwkv7）架构的最新版本，核心算子为 WKV7 循环算子和混合投影逻辑。vLLM 上游实现在 `vllm/model_executor/models/rwkv7.py` 和 `vllm/model_executor/layers/fla/ops/rwkv7.py`。

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Checkpoint

- **Path**: `/hikscale/models/RWKV/rwkv-step-12250-bf16-hf`
- **结构**: 61 层，hidden 4096，64 heads，head_dim 64，bf16，max_position 86016
- **Tokenizer**: 自定义 tokenizer
- **参考上游源码**: `/mnt/data/Codes/vllm`（只读参考实现，**不得直接修改**）
- **CUDA 语义引用**: RWKV-LM 仓库 `RWKV-v7/cuda` 分支
- **AscendC 语义引用**: gitcode `appleinsky/rwkv_Ascend` 分支

> **注意**: AscendC 实现方向依赖 `ascendc` 工具链可用性，在工具链可用前无法完成 kernel 验证。

### 3.2 环境

- **Conda 环境**: `rwkv7`
- **依赖**: CANN 9.0.0+, PyTorch 2.10.0 + torch-npu 2.10.0
- **环境变量**: `RWKV7_DISABLE_FUSED_RECURRENT=1` 强制使用 torch reference path

### 3.3 代码边界说明

| 来源 | 角色 | 说明 |
|------|------|------|
| `/mnt/data/Codes/vllm` | 只读参考 | vLLM 上游 CUDA 实现语义参考，**不可修改** |
| RWKV-LM `RWKV-v7/cuda` | 语义引用 | CUDA WKV7 循环算子算法语义 |
| gitcode `appleinsky/rwkv_Ascend` | 语义引用 | AscendC WKV7 实现方向参考 |

## 4 Verification Status

### 4.1 已验证（Operator 级）

以下 torch reference 实现在 NPU 上通过单元测试验证（`tests/ut/ops/test_rwkv7_npu.py`）：

- `rwkv7_recurrent_reference` — 核心循环
- `rwkv7_mix6_reference` — 混合投影
- `rwkv7_kk_pre_reference` — kk 预处理
- `rwkv7_lnx_rkvres_xg_reference` — 输出融合

> **说明**: 以上为 operator-level 验证，验证的是 torch fallback 在 NPU 上的数值正确性。

### 4.2 未验证

- **Triton-Ascend dispatch 尚未验证**: RWKV7 的 `mix6`/`kk_pre`/`lnx_rkvres_xg` 映射到 triton-ascend FLA 操作已实现 dispatch 逻辑（commit `7c3bc04c`），但**未经真卡 NPU kernel 执行验证**；当前 safe fallback 仍为 torch reference path
- **AscendC WKV7 kernel**: 独立 AscendC WKV7 kernel 未实现（Phase 3 内容）
- **Full serve / 真实权重推理**: 由于 vLLM 0.18.1 与 vllm-ascend 0.19.1 API 不匹配（`_approximate_gcd`），全量服务启动在当前版本组合下失败

### 4.3 当前已知阻塞

**版本不匹配**: vLLM 0.18.1 与 vllm-ascend 0.19.1 之间存在 `_approximate_gcd` API 不一致，导致 `vllm serve` 启动时服务 crash。此问题需等待版本对齐或上游修复。

## 5 Online Service Deployment

> **警告**: 由于版本不匹配阻塞（见 4.3），当前命令仅供文档参考。服务启动前请确保 vLLM 与 vllm-ascend 版本已对齐。

### 5.1 Recommended Startup Command

```bash
# 强制使用 torch reference path（避免 triton-ascend FLA 未验证问题）
export RWKV7_DISABLE_FUSED_RECURRENT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /hikscale/models/RWKV/rwkv-step-12250-bf16-hf \
    --served-model-name rwkv7 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85
```

> **保守说明**: `--max-model-len 32768` 为保守默认值，checkpoint 支持 max_position 86016，请根据实际输入长度调整。过高设置会导致 NPU 内存压力。

### 5.2 Device Gating 说明

上游通过 `hidden_states.device.type == "cuda"` 判断使用 fused kernel。vllm-ascend patch（commit `7c3bc04c`）在 NPU 上尝试启用 triton-ascend FLA dispatch，**当 dispatch 可用时优先使用，否则自动回退到 torch reference path**。环境变量 `RWKV7_DISABLE_FUSED_RECURRENT=1` 可显式禁用 fused path，强制使用 torch reference。

## 6 Functional Verification

> **注意**: 以下 curl 请求仅在版本对齐后有效。当前服务启动因 4.3 所述版本不匹配而失败。

### 6.1 Service Readiness

```bash
curl http://localhost:8000/v1/models
```

Expected: HTTP 200，返回模型列表。

### 6.2 Smoke Request

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "rwkv7",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "stream": false,
        "max_tokens": 64
    }'
```

Expected: HTTP 200，非空输出。

## 7 Design Reference

详细设计方案见 [RWKV7 Ascend 适配设计方案](../../developer_guide/rwkv7_ascend_design.md)。

涉及 commit：

- 设计: `c1a12b78`
- 隔离 kernel 实现: `a134b050`, `b33f5ee9`, `3de65355`, `95e135c2`
- 导出: `b40dad7b`
- Dispatch: `7c3bc04c`