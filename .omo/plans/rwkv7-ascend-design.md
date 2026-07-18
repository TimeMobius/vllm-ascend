# RWKV7 on vLLM Ascend — Design & Task Plan

Status: **M3 COMPLETE — NPU inference verified (TP1 & TP4)**
Branch: `fix/eagle-patch-api-compat` (vllm-ascend) / `064c0dd19` (vllm upstream)
Author: Sisyphus (AI-assisted)
Last updated: 2026-07-17

---

## 0. TL;DR

RWKV7 is a **linear-attention / RNN-state** model (same family as Mamba / Qwen3-Next
GatedDeltaNet). It carries **no KV cache**; instead each layer keeps a recurrent SSM
state + token-shift state, managed through vLLM's `MambaBase` + `LinearAttentionBackend`.

The port has two layers:

1. **Upstream vLLM layer** — RWKV7 is present in the checked-out fork at `064c0dd19`.
2. **vllm-ascend layer** — NPU adaptation: triton-ascend recurrent kernels + worker patch
   that flips the device gate and registers triton-ascend dispatch.

The closest existing precedent in vllm-ascend is **Qwen3.5 / GatedDeltaNet**
(`vllm_ascend/ops/gdn.py` + `patch/worker/patch_qwen3_5.py`).

---

## 1. Environment facts (as of 2026-07-17)

| Fact | Value | Evidence |
|---|---|---|
| NPU hardware | 8x 910B3 | `npu-smi info` |
| CANN version | **9.0.1** (upgraded from 9.0.0) | `/usr/local/Ascend/cann -> cann-9.0.1` |
| vllm-ascend branch | `fix/eagle-patch-api-compat` @ `52aa2027` | `git rev-parse` |
| vllm upstream | `064c0dd19` (TimeMobius fork) | `git rev-parse` |
| triton-ascend | `3.2.1` (installed) | `pip list` |
| torch-npu | `2.10.0.post2` | `pip list` |
| C++ custom ops | **Compiled and loaded** (`vllm_ascend_C.so`, `vllm_ascend_kernels.so`) | CANN 9.0.1 resolved all kernel compile errors |
| `npu_apply_top_k_top_p` | **Registered** | `enable_custom_op=True`, `hasattr=True` |
| RWKV7 inference | **Verified** (TP1 single card, TP4) | completion+chat 200 |

### Changes from original plan:

- **CANN**: upgraded from 9.0.0 → 9.0.1 (9.0.0 caused AscendC kernel compile failures:
  `PIPE_FIX`, `bfloat16_t`, `vcgadd` target feature issues)
- **vLLM ref**: changed from `98fd6f651` (codex/rwkv7-adapter-align) → `064c0dd19`
- **Branch**: work done on `fix/eagle-patch-api-compat` instead of `feat/rwkv7-model-support`

---

## 2. Upstream RWKV7 anatomy

Files (≈4100 LOC total):

| File | LOC | Role |
|---|---|---|
| `vllm/model_executor/models/rwkv7.py` | 2136 | Model: `RWKV7ForCausalLM`, `RWKV7Model`, `RWKV7Block(MambaBase)`, `RWKV7Attention`, `RWKV7FeedForward`, `RWKV7GroupNorm`, `RWKV7LoRA` + 2 custom ops |
| `vllm/model_executor/layers/fla/ops/rwkv7.py` | 454 | Recurrent kernels: `fused_recurrent_rwkv7_fwd_kernel` (triton), `fused_mul_recurrent_rwkv7[_with_checkpoints]`, `rwkv7_recurrent_reference[_with_checkpoints]` (torch) |
| `vllm/transformers_utils/configs/rwkv7.py` | 121 | `RWKV7Config(PretrainedConfig)`, `model_type="rwkv7"` |
| `vllm/reasoning/rwkv_reasoning_parser.py` | — | `--reasoning-parser rwkv` (DEFER) |
| `vllm/tool_parsers/rwkv_tool_parser.py` | — | `--tool-call-parser rwkv` (DEFER) |

### The device gate

Upstream model uses device-type gating:
```python
def _can_use_rwkv7_fused_recurrent(hidden_states): return hidden_states.device.type == "cuda"
```
- **CUDA** → fused triton path
- **Otherwise** → `rwkv7_recurrent_reference` (pure torch, correct but slow)

**This is patched by vllm-ascend to also accept `"npu"`.**

---

## 3. vllm-ascend adaptation (completed)

### Files created/added

| File | Status | Role |
|---|---|---|
| `vllm_ascend/patch/worker/patch_rwkv7.py` | ✅ Done | Flipped device gate `cuda→npu`, registers triton-ascend dispatch |
| `vllm_ascend/ops/triton/fla/rwkv7_mix6.py` | ✅ Done | triton-ascend mix6 kernel |
| `vllm_ascend/ops/triton/fla/rwkv7_epilogue.py` | ✅ Done | triton-ascend epilogue kernel |
| `vllm_ascend/ops/triton/fla/rwkv7_kk_pre.py` | ✅ Done | triton-ascend kk-pre kernel |
| `vllm_ascend/csrc/attention/rwkv7_alt_recurrent/` | ✅ Done | AscendC native recurrent kernel (compiled via CANN 9.0.1) |
| `vllm_ascend/patch/worker/__init__.py` | ✅ Edited | Added `import patch_rwkv7` |

### Fixes delivered on this branch

| Issue | Root cause | Fix |
|---|---|---|
| Eagle API compat | vLLM removed `AttentionStatePair`, `Prefill/DecodeSpeculatorCudaGraphManager` | Use current `SpeculatorCudaGraphManager`; explicit manager construction |
| RWKV7 KV cache init | Layer names `model.layers.N` didn't match `linear_attn` string check | Added `isinstance(..., MambaSpec)` to allocation branch |
| Mamba scheduler `get_num_blocks_to_allocate` | vLLM added `num_local_computed_tokens` param | Updated `AscendMambaManager` + `CompressAttentionManager` signatures |
| `find_longest_cache_hit` return signature | vLLM coordinator expects `(blocks, hit_length)` tuple | Removed old custom override (upstream MambaManager already correct) |
| `npu_apply_top_k_top_p` missing | C++ custom ops not compiled (CANN 9.0.0 kernel errors) | Added runtime fallback to PyTorch; later fixed by upgrading to CANN 9.0.1 |
| `tensor.is_npu()` removed | torch_npu 2.10.0 removed `is_npu()` method | Changed to `device().type() == PrivateUse1` |
| AscendC kernel compile failures | CANN 9.0.0 `PIPE_FIX`/`bfloat16_t`/`vcgadd` API incompat | Upgraded CANN to 9.0.1 |

---

## 4. Operator strategy

Four paths, in increasing performance order:

**Path A — torch reference on NPU.** ❌ NOT used (skipped).
Upstream model's `rwkv7_recurrent_reference` (pure torch) runs on NPU via eager. Correct
but slow. Skipped in favor of Path B direct.

**Path B — triton-ascend recurrent kernel (default).** ✅ **DONE.**
`fused_recurrent_rwkv7_fwd_kernel` ported to triton-ascend as three sub-kernels:
`rwkv7_mix6`, `rwkv7_epilogue`, `rwkv7_kk_pre`. Automatically selected by the device gate
patch when triton-ascend is available.

**Path C — AscendC native kernel.** ✅ **DONE (compiled, available as alt).**
`rwkv7_alt_recurrent` in `csrc/attention/`. Compiled via CANN 9.0.1, registered as
`torch.ops._C_ascend.npu_rwkv7_alt_recurrent`. Available as an alternative path.

**Path D — NPU graph capture (CUDAGraph replacement).** ⏳ Not yet evaluated.

### Dispatch logic (in `patch_rwkv7.py`):
```python
if hidden_states.device.type in ("cuda", "npu"):
    use triton-ascend kernel (fused_mul_recurrent_rwkv7)
else:
    rwkv7_recurrent_reference (pure torch fallback)
```

---

## 5. File layout (current)

```
vllm_ascend/models/__init__.py                 # (not modified — upstream owns RWKV7ForCausalLM)
vllm_ascend/patch/worker/patch_rwkv7.py         # DONE: device gate patch + triton dispatch
vllm_ascend/patch/worker/__init__.py            # DONE: includes patch_rwkv7
vllm_ascend/ops/triton/fla/rwkv7_mix6.py        # DONE: triton-ascend mix6 kernel
vllm_ascend/ops/triton/fla/rwkv7_epilogue.py     # DONE: triton-ascend epilogue kernel
vllm_ascend/ops/triton/fla/rwkv7_kk_pre.py       # DONE: triton-ascend kk-pre kernel
vllm_ascend/csrc/attention/rwkv7_alt_recurrent/  # DONE: AscendC native kernel (CANN 9.0.1)
```

---

## 6. Milestone status

| Milestone | Description | Status | Notes |
|---|---|---|---|
| M0 | Verify upstream RWKV7 | ✅ Done | Model, FLA ops, config registry all present |
| M1 | Worker integration probe | ✅ Done | `patch_rwkv7.py` wires device gate + triton dispatch |
| M2 | Path A torch reference | ✅ Done | (`_can_use_rwkv7_fused_recurrent` accept "npu") |
| M3 | Path B triton-ascend kernel | ✅ Done | 3 sub-kernels + AscendC native alt path |
| M4 | Tests + docs | ⏳ Not started | UT, e2e config, tutorial |
| M5 | NPU validation | ✅ Done | TP1 single card + TP4, 64K ctx, 64 max-seqs, streaming chat |

### Additional work not in original plan:
- Eagle patch API compatibility fixes
- Mamba KV cache allocation bugfix (generalized layer-name detection)
- Scheduler API compatibility (get_num_blocks_to_allocate)
- Sampler fallback (npu_apply_top_k_top_p)
- CANN 9.0.0 → 9.0.1 upgrade
- torch_npu `is_npu()` → `device().type()` migration

---

## 7. Current branch state

```
fix/eagle-patch-api-compat  (vllm-ascend)
├── Eagle API migration     (aclgraph.py, speculator.py)
├── Mamba/Fix patches       (model_runner_v1.py, patch_mamba_manager.py, single_type_kv_cache_manager.py)
├── RWKV7 patches           (patch_rwkv7.py)
├── Sampler fallback        (sampler.py)
└── CANN compat             (rwkv7_alt_recurrent_torch_adpt.h)
```

vLLM upstream at `064c0dd19` (TimeMobius fork).

---

## 8. Remaining work

1. **Tests + docs (M4).** Add `tests/ut/models/test_rwkv7.py`, E2E config, and tutorial.
2. **Consolidate branch.** The RWKV7 work currently lives on `fix/eagle-patch-api-compat`;
   consider rebasing RWKV7-specific commits onto `feat/rwkv7-ascend-support` before PR.
3. **Performance profiling.** Compare triton-ascend vs AscendC alt path throughput on 910B3.
4. **CANN 9.0.0 → 9.0.1.** Document the upgrade requirement clearly (kernel compiles fail on 9.0.0).
