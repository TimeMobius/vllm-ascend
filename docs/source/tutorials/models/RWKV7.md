# RWKV7

## 1 Introduction

RWKV7 是线性注意力（Rwkv7）架构的最新版本，核心算子为 WKV7 循环算子和混合投影逻辑。模型和配置由
`vllm-ascend` 独立提供，分别位于 `vllm_ascend/models/rwkv7.py` 和
`vllm_ascend/models/rwkv7_config.py`。

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Checkpoint

- **Path**: `/hikscale/models/RWKV/rwkv-step-12250-bf16-hf`
- **结构**: 61 层，hidden 4096，64 heads，head_dim 64，bf16，max_position 86016
- **Tokenizer**: 自定义 tokenizer
- **参考源码**: `/mnt/data/Codes/vllm`（只读语义和 API 参考，**不得直接修改**）
- **CUDA 语义引用**: RWKV-LM 仓库 `RWKV-v7/cuda` 分支
- **AscendC 语义引用**: gitcode `appleinsky/rwkv_Ascend` 分支

> **注意**: AscendC 实现方向依赖 `ascendc` 工具链可用性，在工具链可用前无法完成 kernel 验证。

### 3.2 环境

- **Conda 环境**: `rwkv7`
- **依赖**: CANN 9.0.0+, PyTorch 2.10.0 + torch-npu 2.10.0
- **vLLM 版本**: 以仓库根目录 `.github/vllm-release-tag.commit` 为准。运行前可执行
  `export VLLM_VERSION="$(tr -d '[:space:]' < .github/vllm-release-tag.commit)"`
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

### 4.2 验证状态

- **Triton-Ascend dispatch**: 已在真实 NPU 环境完成 dispatch 和 reference parity 验证
- **AscendC WKV7 kernel**: 已实现并作为可选 recurrent path 提供
- **Full serve / 真实权重推理**: 已在仓库 `.github/vllm-release-tag.commit` 指定的 vLLM 版本上完成真实权重加载和 HTTP smoke test

### 4.3 版本要求

不要在文档或命令中硬编码 vLLM 版本。vLLM Ascend 当前配套版本的唯一来源是
`.github/vllm-release-tag.commit`；如果手工切换 vLLM 提交，必须同步更新环境中的
`VLLM_VERSION`，并确认该提交与当前 vLLM Ascend 分支匹配。

## 5 Online Service Deployment

> **注意**: 启动前请使用 `.github/vllm-release-tag.commit` 设置 `VLLM_VERSION`，确保 vLLM 与
> vLLM Ascend 版本对齐。

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

本地 RWKV7 模型在 NPU 上直接调用 vllm-ascend 的 Triton-Ascend FLA dispatch，**当 dispatch 不可用时自动回退到 torch reference path**。环境变量 `RWKV7_DISABLE_FUSED_RECURRENT=1` 可显式禁用 fused path，强制使用 torch reference。

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
