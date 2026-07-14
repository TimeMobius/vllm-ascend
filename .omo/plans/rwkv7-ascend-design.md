# RWKV7 on vLLM Ascend — Design & Task Plan

Status: EXECUTION PLAN (research complete; upstream RWKV7 and NPU runtime verified)
Author: Sisyphus (AI-assisted)
Date: 2026-07-14

---

## 0. TL;DR

RWKV7 is a **linear-attention / RNN-state** model (same family as Mamba / Qwen3-Next
GatedDeltaNet). It carries **no KV cache**; instead each layer keeps a recurrent SSM
state + token-shift state, managed through vLLM's `MambaBase` + `LinearAttentionBackend`.

The port has two layers:

1. **Upstream vLLM layer** — RWKV7 is already present and registered in the checked-out
   `codex/rwkv7-adapter-align` at `98fd6f651`; no upstream M0 is required.
2. **vllm-ascend layer** — a thin NPU adaptation: register the model, provide an NPU
   recurrent operator path (triton-ascend or torch fallback), and a worker patch that
   flips the device gate from `"cuda"` to `"npu"`.

The closest existing precedent in vllm-ascend is **Qwen3.5 / GatedDeltaNet**
(`vllm_ascend/ops/gdn.py` + `patch/worker/patch_qwen3_5.py`). We mirror its structure.

---

## 1. Environment facts (verified)

| Fact | Value | Evidence |
|---|---|---|
| This box has an NPU runtime | **YES** | 8x 910B3; torch-npu 2.10.0 reports available=True/count=8; NPU tensor smoke passed |
| vllm-ascend repo | `/mnt/data/Codes/vllm-ascend` @ `feat/rwkv7-model-support` | `git rev-parse` |
| vllm upstream repo | `/mnt/data/Codes/vllm` @ `codex/rwkv7-adapter-align` (`98fd6f651`) | `git rev-parse` |
| RWKV7 present on that vllm HEAD | **YES** (model, FLA ops, config, registry) | direct file/registry verification |
| RWKV7 reference branch | `origin/rwkv7-upstream-prep` (543 commits ahead, superset) | `git rev-list --left-right` |
| triton-ascend version target | `triton-ascend==3.2.1` (Huawei mirror) | `pyproject.toml`, `Dockerfile` |
| Existing FLA triton kernels in ascend | `vllm_ascend/ops/triton/fla/*` (chunk_gated_delta_rule etc.) | dir listing |
| Existing linear-attn precedent | Qwen3.5 GDN (`ops/gdn.py`), Bailing MoE linear attn | explore agent |

**Consequence:** dummy and real-weight validation can run on this machine. The runtime
was repaired with the complete CANN 9.0.0 package, 910B ops package, NNAL/ATB, and a
CANN 9.0.0 compatibility link `libhccl.so -> libhcomm.so` required by torch-npu 2.10.0.

---

## 2. Upstream RWKV7 anatomy (from `origin/rwkv7-upstream-prep`)

Files (≈4100 LOC total):

| File | LOC | Role |
|---|---|---|
| `vllm/model_executor/models/rwkv7.py` | 2136 | Model: `RWKV7ForCausalLM`, `RWKV7Model`, `RWKV7Block(MambaBase)`, `RWKV7Attention`, `RWKV7FeedForward`, `RWKV7GroupNorm`, `RWKV7LoRA` + 2 custom ops |
| `vllm/model_executor/layers/fla/ops/rwkv7.py` | 454 | Recurrent kernels: `fused_recurrent_rwkv7_fwd_kernel` (triton), `fused_mul_recurrent_rwkv7[_with_checkpoints]`, `rwkv7_recurrent_reference[_with_checkpoints]` (torch) |
| `vllm/transformers_utils/configs/rwkv7.py` | 121 | `RWKV7Config(PretrainedConfig)`, `model_type="rwkv7"` |
| `vllm/reasoning/rwkv_reasoning_parser.py` | — | `--reasoning-parser rwkv` (DEFER, per author) |
| `vllm/tool_parsers/rwkv_tool_parser.py` | — | `--tool-call-parser rwkv` (DEFER, per author) |
| `tests/model_executor/test_rwkv7.py` | 1371 | CPU-runnable correctness + regression tests |

Registration touch points (upstream):
- `registry.py`: `"RWKV7ForCausalLM": ("rwkv7", "RWKV7ForCausalLM")`
- `transformers_utils/config.py`: `rwkv7="RWKV7Config"`
- `fla/ops/__init__.py`: exports the 4 rwkv7 recurrent symbols
- `models/config.py`: uses generic `MambaModelConfig` (no RWKV7-specific policy)
- `v1/attention/backends/linear_attn.py`: `LinearAttentionBackend/Metadata/Builder`

### 2.1 Interfaces implemented by `RWKV7ForCausalLM`
`HasInnerState`, `IsAttentionFree`, `SupportsPP` (prefix-caching intentionally NOT
declared in upstream-prep first cut).

### 2.2 State model (`RWKV7Block.get_state_shape`)
Three per-layer state tensors (Mamba-style, NOT KV cache):
- `attn_shift_state`: `(hidden_size,)`
- `recurrent_state`: `(num_heads/tp, head_dim, value_dim/num_heads)`
- `ffn_shift_state`: `(hidden_size,)`
Runtime state dtype: **FP32** (`RWKV7_RUNTIME_DTYPE = torch.float32`, numerically load-bearing).

### 2.3 The device gate — THE key hook for NPU
```python
# vllm/model_executor/models/rwkv7.py: current branch device-gate helpers
def _rwkv7_packed_prefill_enabled(hidden_states): return hidden_states.device.type == "cuda"
def _can_use_rwkv7_fused_recurrent(hidden_states): return hidden_states.device.type == "cuda"
```
- On CUDA → fused triton path (`fused_mul_recurrent_rwkv7`).
- Otherwise → `rwkv7_recurrent_reference` (pure torch, slow but correct).

Two custom ops dispatch the CUDA compile path through `direct_register_custom_op`:
- `torch.ops.vllm.rwkv7_attention` (mutates output + final states + v_first)
- `torch.ops.vllm.rwkv7_block_forward`

**These two facts define the entire operator strategy** (§4).

---

## 3. vllm-ascend adaptation precedent (Qwen3.5 GDN)

`vllm_ascend/ops/gdn.py :: AscendGatedDeltaNetAttention(GatedDeltaNetAttention)`:
- Overrides `forward`: input proj → **core op** → output proj.
- Core op mixes **triton-ascend FLA kernels** (`vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule`)
  with **AscendC custom kernels** (`torch.ops._C_ascend.*`).
- `get_attn_backend()` returns `AscendGDNAttentionBackend`.

`patch/worker/patch_qwen3_5.py` wires it by **method-patching** the upstream base class:
```python
_GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend
```
Patch entry: `vllm_ascend/patch/worker/__init__.py` imports the patch module; `adapt_patch`
(in `vllm_ascend/utils.py`) triggers it per-worker.

Model registration precedent: `vllm_ascend/models/__init__.py :: register_model()` calls
`ModelRegistry.register_model(...)`.

---

## 4. Operator strategy for RWKV7 on NPU

RWKV7's core is `fused_mul_recurrent_rwkv7(r, w, k, v, kk, a, ...)` — a gated recurrent
(delta-rule-like) scan producing output + final recurrent state (+ optional checkpoints).

Three candidate paths, in ascending effort:

**Path A — torch reference on NPU (correctness-first).**
Flip the device gate to also accept `"npu"` but keep using `rwkv7_recurrent_reference`
(pure torch ops → run on NPU via torch_npu eager). Zero new kernels. Correct but slow;
no graph fusion. **Best first milestone: proves end-to-end wiring.**

**Path B — triton-ascend recurrent kernel (performance).**
Port `fused_recurrent_rwkv7_fwd_kernel` (triton) to triton-ascend, placed under
`vllm_ascend/ops/triton/fla/rwkv7.py`. triton-ascend==3.2.1 already runs the sibling FLA
chunk kernels, so the dialect is proven. Effort: adapt block sizes / `tl` intrinsics.

**Path C — AscendC native kernel (max perf, last resort).**
Follow `csrc/attention/recurrent_gated_delta_rule/` pattern. Highest effort; only if
Path B underperforms. Out of scope for first delivery.

**Decision:** ship **Path A first** (milestone gate = correctness), then **Path B** as a
perf follow-up. Path C deferred. 310P: torch fallback only (no triton), same as existing
`_310p/ops/fla`.

---

## 5. Proposed file layout (vllm-ascend side)

```
vllm_ascend/models/__init__.py                 # + register RWKV7ForCausalLM (edit)
vllm_ascend/models/rwkv7.py                     # AscendRWKV7ForCausalLM (thin) — only if overrides needed
vllm_ascend/patch/worker/patch_rwkv7.py         # NEW: flip device gate cuda→npu, patch op dispatch
vllm_ascend/patch/worker/__init__.py            # + import patch_rwkv7 (edit)
vllm_ascend/ops/triton/fla/rwkv7.py             # NEW (Path B): triton-ascend recurrent kernel
tests/ut/models/test_rwkv7.py                   # NEW: CPU import + wiring tests
tests/e2e/models/configs/RWKV7.yaml             # NEW: accuracy config (values filled after NPU run)
docs/source/tutorials/models/RWKV7.md           # NEW: tutorial
docs/source/tutorials/models/index.md           # + entry (edit)
```

Upstream RWKV7 is already present in the checked-out vLLM, so no upstream file changes
are needed in the delivery repo. Since the model's non-CUDA branches already use the
reference implementation, Path A should first prove whether a worker patch is needed
at all before adding an unnecessary monkey patch.

---

## 6. Task breakdown (milestones, each = 1 signed commit)

> Commit discipline: each milestone that produces a coherent, compiling unit gets ONE
> signed commit (`git commit -s`). Per AGENTS.md conventional-commit + sign-off.

- **M0 — Verify upstream RWKV7.** Confirm the current vLLM ref contains the model,
  recurrent FLA ops, config, registry, and linear-attention backend. No commit is needed
  in vllm-ascend.
- **M1 — Worker integration probe.** Add only the smallest worker import/registration
  hook if runtime probing proves vllm-ascend needs one; do not register a duplicate model
  class because upstream already owns `RWKV7ForCausalLM`. Commit.
- **M2 — Path A torch/reference correctness.** Run the upstream non-CUDA fallback on NPU
  first. Add a focused patch only for an observed metadata/state incompatibility; do not
  force NPU through CUDA custom ops. Commit.
- **M3 — Path B triton-ascend recurrent kernel.** Add `ops/triton/fla/rwkv7.py`; switch the
  NPU gate to use it when triton is available, else torch. Kernel-shape unit test. Commit.
- **M4 — Tests + docs.** `tests/ut/models/test_rwkv7.py`, `RWKV7.yaml`, tutorial + index.
  Commit.
- **M5 — NPU validation.** Two-stage gate (dummy → real weights), fill accuracy YAML,
  and record ACLGraph/eager evidence on the current 8x 910B3 host.

Milestones M0–M4 are implementation milestones; M5 runs on the current hardware.

---

## 7. Risks / unknowns

1. **Path A may require no model-code patch.** The upstream model intentionally falls back
   from CUDA-only fused paths to pure torch for non-CUDA devices; first prove this with an
   NPU forward before modifying vllm-ascend.
2. **triton-ascend recurrent-kernel parity** (Path B) must be tested on 910B3; Path A
   de-risks correctness first.
3. **FP32 runtime state** must be preserved on NPU (numerical stability); torch_npu FP32
   support assumed — verify on hardware.
4. Reasoning/tool parsers intentionally deferred (upstream author's guidance).

---

## 8. Confirmed implementation decisions

1. Delivery repo: `/mnt/data/Codes/vllm-ascend`, branch `feat/rwkv7-model-support`.
2. Upstream vLLM RWKV7 is already available at `98fd6f651`.
3. First-cut order: Path A (torch/reference correctness), then Path B (triton-ascend).
4. Every coherent milestone receives its own signed commit.
