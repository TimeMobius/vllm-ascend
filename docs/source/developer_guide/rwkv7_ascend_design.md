# RWKV7 Ascend 适配设计方案

## 1. 背景与目标

本设计文档描述 RWKV7 模型在 Ascend NPU 上的适配方案。RWKV7 是线性注意力（Rwkv7）架构的最新版本，其核心计算模式与 GDN（Gate Delta Net）有一定相似性，但包含独特的 WKV7 循环算子和混合投影逻辑。

**事实陈述：**

- RWKV7 的语义参考实现位于私有 vLLM 分支；可运行的模型实现位于
  `vllm_ascend/models/rwkv7.py`，配置位于 `vllm_ascend/models/rwkv7_config.py`
- RWKV7 上游 CUDA 实现位于 `/mnt/data/Codes/vllm/csrc/rwkv7_alt_recurrent.cu`（vLLM 上游 CUDA 实现，供模式参考）
- vLLM Ascend 已知参考路径（Path A）已在 `tests/ut/ops/test_rwkv7_npu.py` 中验证通过，使用 torch fallback 在 NPU 上运行
- RWKV7 支持通过 `RWKV7_DISABLE_FUSED_RECURRENT=1` 环境变量强制使用 reference path

**待验证假设：**

- 上游 RWKV7 的 state/cache 语义是否与 vLLM v1 attention metadata 完全兼容
- RWKV7 的循环算子是否有对应的 AscendC 实现路径（需独立开发或证明与现有 GDN kernel 等价）
- triton-ascend 上的 FLA 操作是否可映射到 RWKV7 的混合投影逻辑（mix6、kk_pre、lnx_rkvres_xg）

## 2. 源码边界与只读约定

### 2.1 私有 vLLM 源码 — 只读参考

`/mnt/data/Codes/vllm` 目录仅作为 RWKV7 语义和 API 参考，**不得直接修改**。主线发行版通过
`vllm-ascend` 的 general plugin 注册 RWKV7 模型和配置，不依赖上游包含 RWKV7。

相关上游文件（仅作语义参考）：

- `vllm/model_executor/models/rwkv7.py` — RWKV7 模型类定义
- `vllm/model_executor/layers/fla/ops/rwkv7.py` — FLA 操作实现
- `vllm/model_executor/layers/fla/ops/utils.py` — FLA 公共工具
- `vllm/attention/backends/fla_recurrent.py` — vLLM v1 FLA recurrent attention backend

### 2.2 外部代码引用边界

**CUDA 语义引用**：RWKV-LM 仓库 `RWKV-v7/cuda` 分支作为 CUDA WKV7 循环算子的算法语义参考。

**AscendC 语义引用**：gitcode 上的 `appleinsky/rwkv_Ascend` 分支作为 AscendC WKV7 实现方向的语义参考。

**vLLM 上游 CUDA 实现**：`/mnt/data/Codes/vllm/csrc/rwkv7_alt_recurrent.cu` 是 vLLM 上游的实际 CUDA 源码路径。

**License 合规**：上述外部代码不得直接复制到 vllm-ascend。License 条款需在使用前单独审核确认。

### 2.3 保留上游实现模式

本方案**不追求**用单一 WKV7 kernel 替换所有模型路径。正确做法是：

1. **Path A（Torch Reference）**：利用本地模型中的 torch reference 在 NPU 上运行
2. **Path B（Triton-Ascend FLA）**：将 RWKV7 的 triton FLA 操作（mix6、kk_pre、lnx_rkvres_xg、fused_mul_recurrent_rwkv7）映射到 triton-ascend 等价实现
3. **Path C（AscendC WKV7）**：开发独立的 AscendC WKV7 kernel（不预设与 GDN kernel 的复用）

## 3. 上游 vLLM RWKV7 执行路径

### 3.1 模型前向流程（已确认事实）

```
RWKV7Model.forward()
  └── RWKV7Block.forward()
        └── RWKV7Attention.forward()
              ├── rwkv7_mix6() 或 rwkv7_mix6_reference() — 混合投影
              ├── rwkv7_kk_pre() 或 rwkv7_kk_pre_reference() — kk 预处理
              ├── fused_mul_recurrent_rwkv7() 或 rwkv7_recurrent_reference() — 核心循环
              └── rwkv7_lnx_rkvres_xg() 或 rwkv7_lnx_rkvres_xg_reference() — 输出融合
```

### 3.2 设备门控逻辑（已确认事实）

上游通过以下条件决定使用哪个路径：

```python
# vllm/model_executor/models/rwkv7.py
use_fused = (hidden_states.device.type == "cuda" and _rwkv7_packed_prefill_enabled())
if use_fused:
    # Path: fused_mul_recurrent_rwkv7 + triton kernel
else:
    # Path: rwkv7_recurrent_reference (torch fallback)
```

在 NPU 上，`hidden_states.device.type == "npu" != "cuda"`，因此默认走 torch reference path。

### 3.3 状态与缓存语义（待验证假设）

RWKV7 的 state 语义与 GDN 的 SSM state 相似但不完全相同：

- **GDN state**：`[batch, heads, head_k_dim, head_v_dim]`，存储循环 hidden state
- **RWKV7 state**：`[batch, heads, head_dim, head_dim]`，存储 WKV7 循环的中间结果

假设（需测试验证）：vLLM v1 的 `AttentionMetadata` 可正确传递 RWKV7 的 state tensor，且 `chunk_gated_delta_rule` 的 `initial_state` 和 `final_state` 接口可复用于 RWKV7。

## 4. CUDA → AscendC 映射方案

### 4.1 WKV7 循环算子映射

**CUDA 原始实现**（vLLM 上游 `/mnt/data/Codes/vllm/csrc/rwkv7_alt_recurrent.cu`）：

- `rwkv7_recurrent_reference()` — torch 实现，数学基准
- `fused_mul_recurrent_rwkv7()` — CUDA fused kernel

**上游 FLA Triton ops**（`vllm/model_executor/layers/fla/ops/rwkv7.py`）：

- `rwkv7_mix6()` / `rwkv7_mix6_reference()` — 混合投影
- `rwkv7_kk_pre()` / `rwkv7_kk_pre_reference()` — kk 预处理
- `fused_mul_recurrent_rwkv7()` / `rwkv7_recurrent_reference()` — 核心循环
- `rwkv7_lnx_rkvres_xg()` / `rwkv7_lnx_rkvres_xg_reference()` — 输出融合

**AscendC 目标实现**：

上游 CUDA alt recurrent kernel 的功能需要映射到 AscendC 实现。**前提条件**：需先证明 WKV7 循环算子与现有 GDN `npu_recurrent_gated_delta_rule` 数学等价，或独立开发 AscendC WKV7 kernel。

映射步骤：

1. 验证 WKV7 的 `rwkv7_recurrent_reference` 与 GDN recurrent 形式的数学等价性
2. 若等价，评估接口适配成本
3. 若不等价，设计独立的 AscendC WKV7 kernel（Phase 3）

### 4.2 混合投影与 FLA 操作映射

**上游 Triton+FLA ops**：

- `rwkv7_mix6` / `rwkv7_mix6_reference` — 6 路混合投影（triton / torch）
- `rwkv7_kk_pre` / `rwkv7_kk_pre_reference` — kk 预处理（triton / torch）
- `rwkv7_lnx_rkvres_xg` / `rwkv7_lnx_rkvres_xg_reference` — 输出融合（triton / torch）

**Triton + FLA → Triton-Ascend + FLA 映射**：

现有 vllm-ascend triton FLA 操作位于 `vllm_ascend/ops/triton/fla/`，包括：

- `chunk_gated_delta_rule_fwd_h` — chunk 前向
- `chunk_fwd_o` — 输出融合
- `fused_qkvzba_split_reshape_cat` — QKV 融合

映射策略（待验证）：

1. 评估现有 triton FLA kernel 是否可复用于 RWKV7 的 mix6/kk_pre/lnx_rkvres_xg
2. 如不可复用，开发 RWKV7 专用的 triton-ascend kernel
3. 如 AscendC 实现不存在，先回退到 torch 实现（Path A）

**关键约束**：triton-ascend 对 RWKV7 的上述操作支持**目前未验证**，需要 Phase 2 开发确认可行性。

## 5. Reference Fallback 策略

### 5.1 当前已验证路径

`tests/ut/ops/test_rwkv7_npu.py` 验证了以下 Reference 实现可在 NPU 上工作：

- `rwkv7_recurrent_reference` — 核心循环
- `rwkv7_mix6_reference` — 混合投影
- `rwkv7_kk_pre_reference` — kk 预处理
- `rwkv7_lnx_rkvres_xg_reference` — 输出融合
- `fused_mul_recurrent_rwkv7` 在 NPU 上正确 fallback 到 reference

### 5.2 Fallback 使用场景

以下场景使用 Reference fallback：

1. **设备不兼容**：NPU 不满足 `use_fused` 条件
2. **Triton 不可用**：`triton.ops.flash_linear_attention` 不可用
3. **环境变量禁用**：`RWKV7_DISABLE_FUSED_RECURRENT=1`
4. **AscendC kernel 缺失**：Phase 2 开发完成前

### 5.3 性能考量

Reference fallback 在 NPU 上的性能表现需通过 Profiling 验证。根据 GDN 经验，torch reference 相比 AscendC kernel 有显著性能差距（预估 2-5x），因此 Phase 2 的 AscendC 开发是关键路径。

## 6. State/Cache 语义

### 6.1 RWKV7 State 管理

RWKV7 使用与 GDN 相似的 state 管理模式：

```python
# 上游调用方式（推测）
core_attn_out, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    initial_state=initial_state,
    output_final_state=True,
    cu_seqlens=cu_seqlens
)
# final_state 用于下一轮推理的初始 state
```

### 6.2 与 vLLM v1 Attention Metadata 的兼容性

**假设（待验证）**：

- `RWKV7Attention` 的 `state` 字段可通过 vLLM v1 的 `AttentionMetadata.spec_sequence_masks` 传递
- `prefill_state_indices` 和 `prefill_has_initial_state` 可正确处理 RWKV7 的 state 恢复

**风险**：如果上游 RWKV7 的 state 格式与 GDN 不兼容，可能需要在 vLLM v1 attention backend 中添加专门的 RWKV7 支持。

### 6.3 Checkpoint 格式

Checkpoint 路径：`/hikscale/models/RWKV/rwkv-step-12250-bf16-hf`

已确认事实：

- 61 层
- hidden size 4096
- 64 heads
- head_dim 64
- dtype: bf16
- 自定义 tokenizer
- max_position 86016

待验证假设：vLLM 的 RWKV7 模型加载器是否正确处理这些参数。

## 7. Checkpoint 事实清单

| 属性 | 值 | 状态 |
|------|-----|------|
| 路径 | `/hikscale/models/RWKV/rwkv-step-12250-bf16-hf` | 已确认 |
| 层数 | 61 | 已确认 |
| Hidden Size | 4096 | 已确认 |
| Head 数 | 64 | 已确认 |
| Head Dim | 64 | 已确认 |
| 数据类型 | bf16 | 已确认 |
| Tokenizer | 自定义 | 已确认 |
| Max Position | 86016 | 已确认 |

## 8. 提议的文件结构

```
vllm_ascend/
├── models/
│   ├── rwkv7.py                    # 独立 RWKV7 模型实现
│   └── rwkv7_config.py             # 独立 RWKV7 配置
├── patch/
│   └── worker/
│       └── patch_rwkv7.py          # 本地模型的 kernel dispatch patch
├── ops/
│   ├── rwkv7_attention.py          # Ascend RWKV7Attention wrapper
│   └── triton/
│       └── fla/
│           └── rwkv7_chunk.py      # RWKV7 专用 triton-ascend kernel
└── csrc/
    └── kernels/                    # AscendC kernel（遵循现有 csrc/kernels 目录结构）
        └── rwkv7_recurrent.cpp     # AscendC WKV7 kernel（Phase 3）
```

**注意**：

- 所有文件路径为提议，实际路径根据代码组织调整
- AscendC kernel 使用 `.cpp` 扩展名，遵循 vllm-ascend csrc 规范
- Phase 1 可能无需任何 worker patch，以运行时验证为准

## 9. 分阶段里程碑

### Phase 0: 环境确认（预计 1 天）

**目标**：确认开发环境和依赖完整性

**交付物**：

- 确认 `/mnt/data/Codes/vllm` 仅作为只读参考
- 确认 NPU 可用性和 CANN 版本（**待验证**）
- 确认环境变量 `RWKV7_DISABLE_FUSED_RECURRENT` 可用

**验收标准**：

- `torch.npu.is_available()` 返回 True（**待验证**）
- 上游 RWKV7 模型可成功加载（无需推理，**待验证**）
- triton-ascend 环境状态确认（**待验证**）

### Phase 1: Reference Path 验证（预计 3-5 天）

**目标**：验证 RWKV7 在 NPU 上使用 Reference path 端到端可运行

**交付物**：

- 端到端测试验证模型可推理
- 通过 `vllm_ascend/models/__init__.py` 注册 `RWKV7Config` 和 `RWKV7ForCausalLM`

**验收标准**：

- `GET /v1/models` 返回 200
- 单条 text 请求返回 200 且非空输出
- 模型加载日志无 fatal error

**一次 signed commit**：

```
feat(rwkv7): add initial RWKV7 reference path support

- Verify torch reference fallback works on Ascend NPU
- Add e2e test for RWKV7 inference on NPU
- Add worker patch only if runtime validation requires it
```

### Phase 2: Triton-Ascend FLA 映射（预计 1-2 周）

**目标**：将 RWKV7 的 triton FLA 操作映射到 triton-ascend

**交付物**：

- `vllm_ascend/ops/triton/fla/rwkv7_chunk.py`
- AscendC kernel 实现或回退到 torch 的 graceful degradation

**验收标准**：

- `RWKV7_DISABLE_FUSED_RECURRENT=0` 时使用 triton-ascend path
- 性能优于 Phase 1 的纯 torch reference（待 Profiling 验证）

**风险**：triton-ascend 的 RWKV7 支持可能需要较长开发周期。

**一次 signed commit**：

```
feat(rwkv7): add triton-ascend FLA path for RWKV7

- Map rwkv7_mix6 to triton-ascend FLA operations
- Map rwkv7_kk_pre to triton-ascend kernel
- Add graceful fallback to torch reference when AscendC unavailable
```

### Phase 3: AscendC WKV7 Kernel（预计 2-3 周，可选）

**目标**：开发专门的 AscendC WKV7 kernel

**交付物**：

- `csrc/kernels/rwkv7_recurrent.cpp` — AscendC WKV7 kernel（遵循 csrc/kernels 目录结构）
- Profiling 报告

**验收标准**：

- AscendC WKV7 kernel 正确性验证（与 torch reference 对比）
- 性能提升显著（目标：相比 Phase 2 提升 30%+）

**风险**：可能需要与华为 HCB 团队协作。

**一次 signed commit**（如 Phase 3 实施）：

```
perf(rwkv7): implement AscendC WKV7 recurrent kernel

- Add AscendC implementation of WKV7 recurrent operator
- Optimize memory layout for NPU
- Add benchmark and accuracy validation
```

## 10. 测试与验证

### 10.1 单元测试

**已有测试**（Phase 0-1 依赖）：

- `tests/ut/ops/test_rwkv7_npu.py` — Reference path 验证

**提议新增**：

- `tests/ut/ops/test_rwkv7_ascend.py` — AscendC kernel 正确性测试（Phase 2/3）
- `tests/ut/ops/test_rwkv7_attention.py` — RWKV7Attention wrapper 单元测试

### 10.2 端到端测试

**提议测试**：

- `tests/e2e/pull_request/one_card/test_rwkv7_inference.py` — RWKV7 端到端推理测试（**待验证：当前环境是否支持全模型推理**）

**验收标准**（**待运行时验证**）：

- 端到端推理无 crash
- 输出 token 分布合理（无 NaN/Inf）
- 日志无 unresolved architecture 错误

### 10.3 Profiling

**提议指标**（不预设具体数值）：

- Prefill throughput（tokens/s）
- Decode latency（ms/token）
- Memory usage（GB）

**验收标准**：

- AscendC kernel 性能优于 torch reference（具体倍数待实测）
- Memory 使用量在合理范围内

## 11. 环境要求

**专用环境变量**：

```bash
# 仅选择一个 recurrent backend：auto（默认）、reference、ascendc、triton_t1 或
# triton_t1_cache。
VLLM_ASCEND_RWKV7_RECURRENT_BACKEND=reference

# 覆盖所有 fused recurrent backend，并强制使用 reference path。
RWKV7_DISABLE_FUSED_RECURRENT=1
```

`triton_t1_cache` 依次尝试 persistent-cache Triton T=1、非 cache Triton T=1 和
reference path。`auto` 与 `ascendc` 都会先尝试 AscendC，失败后使用 reference。无效的
backend 值会在解析环境变量时直接失败，不会静默选择其他 backend。

**依赖项**：

- CANN 9.0.0+
- PyTorch 2.10.0 + torch-npu 2.10.0
- triton-ascend（用于 Phase 2+）

## 12. 风险清单

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| RWKV7 state 语义与 vLLM v1 不兼容 | 高 | Phase 1 验证时重点测试 state 传递 |
| triton-ascend FLA 映射复杂度超预期 | 中 | Phase 1 后重新评估，必要时跳过 Phase 2 |
| AscendC WKV7 kernel 开发周期长 | 中 | 优先保证 Reference path，Phase 3 作为可选优化 |
| Checkpoint 格式与 vLLM 加载器不兼容 | 低 | Phase 0 验证加载流程 |
| 外部代码 License 未审核 | 高 | 不直接复制外部代码，License 需单独审核 |

## 13. License 边界

### 13.1 上游 License

- **vLLM**：Apache 2.0（已确认）

### 13.2 合规要求

- **外部代码引用**（RWKV-LM `RWKV-v7/cuda`、gitcode `appleinsky/rwkv_Ascend`）：License 条款需在使用前单独审核确认，**不得直接复制**
- **可以**：参考其数学定义和算法逻辑，重新实现 AscendC 版本
- **应当**：在代码注释中注明参考来源（如 "Based on RWKV-LM implementation"）

### 13.3 依赖 License

- **vllm-ascend 现有 triton FLA 操作**：已有 Apache 2.0 + SPDX 头，可复用
- **Flash Linear Attention**（如引用）：需保留版权声明，License 条款单独确认

## 14. 明确非目标

以下内容**不在**本设计范围内：

1. **不实现** Python/C++/AscendC/Triton 代码（仅设计）
2. **不添加** placeholder YAML metrics（测试结果待实测后填充）
3. **不创建** todos 或 commits（设计文档阶段）
4. **不声称** triton-ascend RWKV7 支持已可用（Phase 2 才可能实现）
5. **不修改** `/mnt/data/Codes/vllm`（只读参考），模型和配置由 `vllm-ascend` 独立提供
6. **不替换** 所有上游路径为单一 WKV7 kernel（保留多路径fallback）
7. **不预设** benchmark 或 accuracy 具体数值（待实测）

## 15. 验收标准总结

### Phase 0 验收

- [ ] `/mnt/data/Codes/vllm` 保持只读
- [ ] NPU 可用（**待验证**）
- [ ] 环境变量 `rwkv7` 已确认可用（**待验证**）

### Phase 1 验收

- [ ] RWKV7 模型可端到端推理（**待验证**）
- [ ] `GET /v1/models` 返回 200（**待验证**）
- [ ] 单条 text 请求返回 200（**待验证**）
- [ ] 一次 signed commit 包含所有 Phase 1 改动

### Phase 2 验收

- [ ] triton-ascend path 可用（当 AscendC kernel 存在时）
- [ ] graceful fallback 到 torch reference（当 AscendC kernel 不存在时）
- [ ] 一次 signed commit 包含所有 Phase 2 改动

### Phase 3 验收（如实施）

- [ ] AscendC WKV7 kernel 正确性验证通过
- [ ] Profiling 显示性能提升
- [ ] 一次 signed commit 包含所有 Phase 3 改动

## 16. 参考资料

### 上游文档

- vLLM RWKV7 模型：`vllm/model_executor/models/rwkv7.py`（语义引用）
- vLLM FLA ops：`vllm/model_executor/layers/fla/ops/rwkv7.py`（语义引用）
- vLLM 上游 CUDA 实现：`/mnt/data/Codes/vllm/csrc/rwkv7_alt_recurrent.cu`（上游源码路径）

### 外部语义引用

- RWKV-LM CUDA：`RWKV-v7/cuda` 分支（CUDA 语义参考）
- AscendC 参考：`appleinsky/rwkv_Ascend`（AscendC 实现方向参考）

### vllm-ascend 先例

- Qwen3.5 GDN 适配：`vllm_ascend/patch/worker/patch_qwen3_5.py`
- GDN AscendC kernel：`vllm_ascend/ops/gdn.py`
- triton FLA 操作：`vllm_ascend/ops/triton/fla/`

### 测试参考

- 已有 RWKV7 测试：`tests/ut/ops/test_rwkv7_npu.py`

---

**文档版本**：v0.1（草案）
**创建日期**：2026-07-15
**状态**：待评审
