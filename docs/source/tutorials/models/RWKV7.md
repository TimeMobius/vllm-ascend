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

- **Path**: `/hikscale/models/RWKV/Xiaoke-5-13B-2607`
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
- **recurrent backend**: `VLLM_ASCEND_RWKV7_RECURRENT_BACKEND` 选择循环后端；默认
  `auto`。要强制使用 torch reference path，直接设
  `VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=reference`（或用 `VLLM_ASCEND_RWKV7_PRESET=reference`）。
- **preset**: `VLLM_ASCEND_RWKV7_PRESET` 面向绝大多数部署用户，默认 `auto`。可选值
  `reference`（全部 reference）、`auto`（按 guard 选择）、`throughput`（C128 吞吐，
  优先 persistent-cache T=1 recurrent 并允许已验证的融合算子）。
- **observability**: `VLLM_ASCEND_RWKV7_OBSERVABILITY` 控制诊断，默认 `off`，可选
  `summary`（记录 hit / guard fallback / kernel exception 并在进程退出时打印汇总）。
- **operator overrides**: `VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES` 是严格 JSON，仅给
  kernel 开发与性能验证使用，优先级最高。key 为 `mix6`/`kk_pre`/`epilogue`/
  `block_norms`，value 为 `triton`/`reference`/`auto`；另可含 `recurrent` key 以便
  最高优先级覆盖循环后端。

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
- **END-TO-END decode throughput（910B3, 单卡, C128）**: 使用
  `triton_t1_cache`、`--mamba-cache-mode align` 和 `FULL_AND_PIECEWISE`，两次测量
  分别达到 **1214.300 tok/s** 和 **1213.850 tok/s**。这是面向吞吐量的并发配置，不能与
  单请求 latency 测量直接比较。

### 4.3 版本要求

不要在文档或命令中硬编码 vLLM 版本。vLLM Ascend 当前配套版本的唯一来源是
`.github/vllm-release-tag.commit`；如果手工切换 vLLM 提交，必须同步更新环境中的
`VLLM_VERSION`，并确认该提交与当前 vLLM Ascend 分支匹配。

## 5 Online Service Deployment

> **注意**: 启动前请使用 `.github/vllm-release-tag.commit` 设置 `VLLM_VERSION`，确保 vLLM 与
> vLLM Ascend 版本对齐。

### 5.1 Reference Startup Command

```bash
# 强制使用 torch reference path（避免使用任何 triton-ascend kernel）
export VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=reference
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /hikscale/models/RWKV/Xiaoke-5-13B-2607 \
    --served-model-name xiaoke-5 xiaoke-5-Ascend \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85
```

> **保守说明**: `--max-model-len 32768` 为保守默认值，checkpoint 支持 max_position
> 86016，请根据实际输入长度调整。过高设置会导致 NPU 内存压力。

### 5.2 Recommended Throughput Configuration (C128)

在 910B3 单卡上，以下配置使用 persistent-cache Triton T=1 recurrent backend。C128
测量的 decode throughput 为 1214.300 tok/s 和 1213.850 tok/s。该配置针对并发吞吐量，
不是单请求 latency preset。

```bash
# C128 吞吐 preset：优先 persistent-cache T=1 recurrent，并允许已验证的融合算子。
export VLLM_ASCEND_RWKV7_PRESET=throughput
export VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=triton_t1_cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /hikscale/models/RWKV/Xiaoke-5-13B-2607 \
    --served-model-name xiaoke-5 xiaoke-5-Ascend \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tokenizer-mode rwkv \
    --max-model-len 1M \
    --max-num-seqs 128 \
    --max-num-batched-tokens 32K \
    --mamba-cache-mode align \
    --gpu-memory-utilization 0.85 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,48,64,96,128]}'
```

> **约束**: 不要在该配置中加入 `--enforce-eager`，它会禁用图捕获。通过
> `VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES` 关闭已验证外的算子时要谨慎（例如关闭
> `epilogue` 会把吞吐量从 1214 降至约 970 tok/s）。

### 5.3 Device Gating 说明

本地 RWKV7 模型在 NPU 上直接调用 vllm-ascend 的 Triton-Ascend FLA dispatch，**当 dispatch 不可用时自动回退到 torch reference path**。要强制使用 torch reference，设置 `VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=reference`（仅循环路径）或 `VLLM_ASCEND_RWKV7_PRESET=reference`（全部算子）。

`VLLM_ASCEND_RWKV7_RECURRENT_BACKEND` 可选值为：

- `auto`：AscendC runtime guard 成功时使用 AscendC，否则使用 reference。
- `reference`：始终使用 reference。
- `ascendc`：尝试 AscendC，guard 失败时使用 reference。
- `triton_t1`：尝试非 persistent-cache Triton T=1 kernel，失败时使用 reference。
- `triton_t1_cache`：依次尝试 persistent-cache Triton T=1、非 cache Triton T=1 和
  reference。

## 6 Functional Verification

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
        "model": "xiaoke-5",
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
- FULL_DECODE_ONLY 修复链:
  - `c5201a706` fix(rwkv7_counters): make counter methods graph-safe under torch.compile fullgraph_capture
  - `edfbcfc7b` fix(attention): handle missing vllm_config in lru_cached helpers
  - `2988e2a41` fix(rwkv7): skip torch.compile in FULL/FULL_DECODE_ONLY graph mode
