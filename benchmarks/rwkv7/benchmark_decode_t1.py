# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 Decode T=1 three-path NPU benchmark.

Measures AscendC, Triton-Ascend, and Torch FP32 reference latency for the
fused RWKV7 T=1 recurrent step + reduce kernel at the single-token decode
batch sizes used by vLLM Ascend. Records parity against the Torch reference
and full timing-fence metadata.

Three backends, identical input distribution, identical timing boundary:

* ``torch_reference``  - pure PyTorch reference
                          ``vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1``
                          ``._rwkv7_recurrent_t1_reference``.
* ``triton_t1``        - current Triton-Ascend force-T1 kernel
                          ``vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1``
                          ``.rwkv7_recurrent_t1``.
* ``ascendc_alt_recurrent`` - AscendC custom op
                          ``torch.ops._C_ascend.npu_rwkv7_alt_recurrent``.

The shared recurrence is

    sa[h, v]      = sum_d state[h, d, v] * (-kk[h, d])
    new_state[h, d, v] = exp(w[h, d]) * state[h, d, v]
                       + (kk[h, d] * a[h, d]) * sa[h, v]
                       + k[h, d] * v[h, v]
    reduce_out[h, v]   = sum_d new_state[h, d, v] * r[h, d]

with rank-4 batched state ``[B, H, D, V]`` exclusively. The AscendC kernel
accepts ``[B, T=1, H, 64]`` for the per-token projections and ``[B, H, 64, 64]``
for the initial state (which matches the Triton/Torch layout ``[B, H, D, V]``
when ``D == V == 64``). Outputs are normalized so ``reduce_out`` is
``[B, H, V]`` and ``final_state`` is ``[B, H, D, V]`` for all three backends.

Timing fence (identical for every backend / shape / repetition):

    torch.npu.synchronize()                    # drain prior work
    s_warmup = s0.clone()                      # independent state
    for _ in range(WARMUP):  step_fn(s_warmup, ...)
    torch.npu.synchronize()                    # drain warmup
    s_timed = s0.clone()                       # fresh independent state
    torch.npu.synchronize()                    # defensive
    start_event.record()
    for _ in range(TIMED):    step_fn(s_timed, ...)   # NO sync / copy /
    end_event.record()                                   # item / log / cmp
    torch.npu.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

The timed chain contains 100 dependent state updates; ``w, kk, a, k, v, r``
are pre-allocated outside the timed window and held constant across the
chain to isolate the per-call kernel cost. The end-state is not compared
across backends - they legitimately diverge after the first step.

Parity / contract checks are run OUTSIDE the timing fence on a fresh single
step with the same seed-derived ``s0``, comparing each backend's
``new_state`` and ``reduce_out`` against the Torch reference at
``atol=rtol=2e-4``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H_HEADS = 32
D_HEAD_DIM = 64
V_VALUE_DIM = 64
DTYPE = torch.float32

BATCH_SIZES = (1, 2, 8, 16, 32, 64, 128, 256, 512)
WARMUP_CALLS = 20
TIMED_CALLS = 100
REPETITIONS = 3

BASE_SEED = 12345
SEED_STRIDE = 1009  # prime gap - guarantees disjoint RNG sequences per rep

ATOL = 2.0e-4
RTOL = 2.0e-4

TIMING_FENCE_CONTRACT = (
    "torch.npu.synchronize(); "
    "s_warmup = s0.clone(); "
    f"for _ in range({WARMUP_CALLS}): step_fn(s_warmup, ...); "
    "torch.npu.synchronize(); "
    "s_timed = s0.clone(); "
    "torch.npu.synchronize(); "
    "start_event.record(); "
    f"for _ in range({TIMED_CALLS}): step_fn(s_timed, ...); "
    "end_event.record(); "
    "torch.npu.synchronize(); "
    "elapsed_ms = start_event.elapsed_time(end_event)"
)

TIMED_CHAIN_RULES = (
    "no synchronize, no CPU copy, no .item(), no logging, no comparison "
    "between any backend's running state and the reference"
)

# Bound w so exp(w) stays well inside (0, 1); keeps the dependent 100-step
# chain numerically stable across all batch sizes.
W_SCALE = 0.05
W_OFFSET = -2.0

SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
OUTPUT_PATH = OUTPUT_DIR / "decode_t1_metrics.json"


# ---------------------------------------------------------------------------
# Backend step wrappers - return identical rank-4 contract for every path
# ---------------------------------------------------------------------------


def torch_ref_step(state, w, kk, a, k, v, r):
    """Pure PyTorch reference recurrence. Same contract as Triton T1."""
    from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 import (
        _rwkv7_recurrent_t1_reference,
    )

    return _rwkv7_recurrent_t1_reference(state, w, kk, a, k, v, r)


def triton_t1_step(state, w, kk, a, k, v, r):
    """Triton-Ascend force-T1 fused kernel. Same contract as reference."""
    from vllm_ascend.ops.triton.fla.rwkv7_recurrent_t1 import rwkv7_recurrent_t1

    return rwkv7_recurrent_t1(state, w, kk, a, k, v, r)


def ascendc_step(state, w, kk, a, k, v, r):
    """AscendC ``npu_rwkv7_alt_recurrent`` for T=1 decode.

    Returns ``(new_state, reduce_out)`` with shapes ``[B, H, D, V]`` and
    ``[B, H, V]`` to match the Triton / reference contract.
    """
    r_4d = r.unsqueeze(1)
    w_4d = w.unsqueeze(1)
    k_4d = k.unsqueeze(1)
    v_4d = v.unsqueeze(1)
    kk_4d = kk.unsqueeze(1)
    a_4d = a.unsqueeze(1)
    out_4d, final_state_4d = torch.ops._C_ascend.npu_rwkv7_alt_recurrent(
        r_4d, w_4d, k_4d, v_4d, kk_4d, a_4d, state
    )
    return final_state_4d, out_4d.squeeze(1)


BACKENDS = (
    ("torch_reference", torch_ref_step),
    ("triton_t1", triton_t1_step),
    ("ascendc_alt_recurrent", ascendc_step),
)


# ---------------------------------------------------------------------------
# Input generation - bounded so 100 dependent steps stay finite
# ---------------------------------------------------------------------------


def seed_for(base: int, rep_id: int) -> int:
    """Deterministic distinct seed per repetition within a fixed family."""
    return base + rep_id * SEED_STRIDE


def make_inputs(batch_size: int, seed: int, device: torch.device):
    """Build rank-4 batched inputs. Same distribution every repetition uses
    the same family; distinct seeds produce distinct inputs across reps."""
    torch.manual_seed(seed)
    gen_kwargs = {"device": device, "dtype": DTYPE}
    state = torch.randn(batch_size, H_HEADS, D_HEAD_DIM, V_VALUE_DIM, **gen_kwargs)
    base = torch.randn(batch_size, H_HEADS, D_HEAD_DIM, **gen_kwargs)
    w = W_OFFSET + W_SCALE * base
    kk = torch.randn(batch_size, H_HEADS, D_HEAD_DIM, **gen_kwargs)
    a = 0.1 * torch.randn(batch_size, H_HEADS, D_HEAD_DIM, **gen_kwargs)
    k = 0.1 * torch.randn(batch_size, H_HEADS, D_HEAD_DIM, **gen_kwargs)
    v = 0.1 * torch.randn(batch_size, H_HEADS, V_VALUE_DIM, **gen_kwargs)
    r = 0.1 * torch.randn(batch_size, H_HEADS, D_HEAD_DIM, **gen_kwargs)
    return state, w, kk, a, k, v, r


# ---------------------------------------------------------------------------
# Parity / contract checks - OUTSIDE the timing fence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractReport:
    shape_ok: bool
    dtype_ok: bool
    device_ok: bool
    finite_ok: bool
    max_abs_state: float
    max_abs_output: float
    max_rel_state: float
    max_rel_output: float
    passed: bool
    detail: str = ""


def parity_check(step_fn, s0, w, kk, a, k, v, r, ref_state, ref_out):
    """Run a single step with ``step_fn`` on a fresh clone of ``s0`` and
    compare against ``(ref_state, ref_out)``. Records shape/dtype/device/
    finite contract in addition to atol/rtol errors."""
    s = s0.clone()
    state, out = step_fn(s, w, kk, a, k, v, r)
    torch.npu.synchronize()

    shape_ok = (
        state.shape == ref_state.shape
        and out.shape == ref_out.shape
        and state.shape == s0.shape
        and out.shape == (s0.shape[0], s0.shape[1], V_VALUE_DIM)
    )
    dtype_ok = state.dtype == DTYPE and out.dtype == DTYPE
    device_ok = state.device.type == "npu" and out.device.type == "npu"
    finite_ok = bool(torch.isfinite(state).all().item() and torch.isfinite(out).all().item())

    diff_state = (state.float() - ref_state.float()).abs()
    diff_out = (out.float() - ref_out.float()).abs()
    max_abs_state = float(diff_state.max().item()) if diff_state.numel() else 0.0
    max_abs_output = float(diff_out.max().item()) if diff_out.numel() else 0.0

    ref_state_abs = ref_state.float().abs()
    ref_out_abs = ref_out.float().abs()
    denom_state = torch.clamp(ref_state_abs, min=1.0)
    denom_out = torch.clamp(ref_out_abs, min=1.0)
    max_rel_state = float((diff_state / denom_state).max().item()) if diff_state.numel() else 0.0
    max_rel_output = float((diff_out / denom_out).max().item()) if diff_out.numel() else 0.0

    abs_ok_state = max_abs_state <= ATOL + RTOL * float(ref_state_abs.max().item())
    abs_ok_out = max_abs_output <= ATOL + RTOL * float(ref_out_abs.max().item())
    passed = shape_ok and dtype_ok and device_ok and finite_ok and abs_ok_state and abs_ok_out
    detail = (
        f"shape_ok={shape_ok} dtype_ok={dtype_ok} device_ok={device_ok} "
        f"finite_ok={finite_ok} abs_ok(state)={abs_ok_state} abs_ok(out)={abs_ok_out}"
    )
    return ContractReport(
        shape_ok=shape_ok,
        dtype_ok=dtype_ok,
        device_ok=device_ok,
        finite_ok=finite_ok,
        max_abs_state=max_abs_state,
        max_abs_output=max_abs_output,
        max_rel_state=max_rel_state,
        max_rel_output=max_rel_output,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Timing fence - identical for every backend / shape / repetition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedSample:
    rep: int
    seed: int
    elapsed_ms: float
    per_step_ms: float


def timed_chain(step_fn, s0, w, kk, a, k, v, r) -> TimedSample:
    """Execute the timing-fence contract and return elapsed milliseconds
    measured by ``torch.npu.Event`` (device-side)."""
    # Drain any prior NPU work so warmup starts from a clean slate.
    torch.npu.synchronize()

    # Warmup chain uses an independent clone; its evolution is discarded.
    s_warmup = s0.clone()
    for _ in range(WARMUP_CALLS):
        new_state, _ = step_fn(s_warmup, w, kk, a, k, v, r)
        s_warmup = new_state
    torch.npu.synchronize()

    # Timed chain uses a second independent clone of the same s0.
    s_timed = s0.clone()
    torch.npu.synchronize()  # defensive fence after the clone

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    for _ in range(TIMED_CALLS):
        new_state, _ = step_fn(s_timed, w, kk, a, k, v, r)
        s_timed = new_state
    end_event.record()
    torch.npu.synchronize()

    elapsed_ms = float(start_event.elapsed_time(end_event))
    return TimedSample(
        rep=-1, seed=-1, elapsed_ms=elapsed_ms, per_step_ms=elapsed_ms / TIMED_CALLS
    )


def aggregate(samples):
    elapsed = [s.elapsed_ms for s in samples]
    per_step = [s.per_step_ms for s in samples]
    return {
        "elapsed_ms": {
            "mean": statistics.fmean(elapsed),
            "median": statistics.median(elapsed),
            "min": min(elapsed),
            "max": max(elapsed),
            "stdev": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
        },
        "per_step_ms": {
            "mean": statistics.fmean(per_step),
            "median": statistics.median(per_step),
            "min": min(per_step),
            "max": max(per_step),
            "stdev": statistics.pstdev(per_step) if len(per_step) > 1 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: str | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def collect_metadata(device_index: int) -> dict:
    vllm_ascend_version = "unknown"
    with contextlib.suppress(Exception):
        from vllm_ascend import __version__ as vllm_ascend_version

    vllm_version = "unknown"
    with contextlib.suppress(Exception):
        from vllm import __version__ as vllm_version

    git_sha = "unknown"
    git_branch = "unknown"
    with contextlib.suppress(Exception):
        git_sha = run(["git", "rev-parse", "HEAD"], cwd=str(SCRIPT_PATH.parents[2]))
        git_branch = run(["git", "branch", "--show-current"], cwd=str(SCRIPT_PATH.parents[2]))

    device_name = "unknown"
    with contextlib.suppress(Exception):
        device_name = torch.npu.get_device_name(device_index)
    device_cap = None
    with contextlib.suppress(Exception):
        device_cap = torch.npu.get_device_capability(device_index)

    torch_npu_version = "unknown"
    with contextlib.suppress(Exception):
        import torch_npu
        torch_npu_version = torch_npu.__version__

    return {
        "script": str(SCRIPT_PATH.relative_to(SCRIPT_PATH.parents[2])),
        "git_sha": git_sha,
        "git_branch": git_branch,
        "device_index": device_index,
        "device_name": device_name,
        "device_capability": list(device_cap) if device_cap is not None else None,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "service_conflict_check": {
            "service_port": 8000,
            "service_npu_index": 4,
            "benchmark_npu_index": device_index,
            "rationale": (
                "Port-8000 vLLM service binds NPU4 via ASCEND_RT_VISIBLE_DEVICES=4; "
                "benchmark selects a different physical NPU to avoid contention."
            ),
        },
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu_version,
        "vllm_version": vllm_version,
        "vllm_ascend_version": vllm_ascend_version,
        "dimensions": {
            "H": H_HEADS,
            "D": D_HEAD_DIM,
            "V": V_VALUE_DIM,
            "T": 1,
        },
        "dtype": str(DTYPE).split(".")[-1],
        "matrices": [
            "state[B,H,D,V]",
            "w[B,H,D]",
            "kk[B,H,D]",
            "a[B,H,D]",
            "k[B,H,D]",
            "v[B,H,V]",
            "r[B,H,D]",
        ],
        "batch_sizes": list(BATCH_SIZES),
        "seed_family": "linear(base + rep_id * stride)",
        "seed_base": BASE_SEED,
        "seed_stride": SEED_STRIDE,
        "seed_per_repetition_offset": SEED_STRIDE,
        "backends": [name for name, _ in BACKENDS],
        "warmup_calls": WARMUP_CALLS,
        "timed_calls": TIMED_CALLS,
        "repetitions": REPETITIONS,
        "timing_unit": "milliseconds",
        "timing_instrument": "torch.npu.Event(enable_timing=True)",
        "timing_fence_contract": TIMING_FENCE_CONTRACT,
        "timed_chain_rules": TIMED_CHAIN_RULES,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "w_distribution": f"Normal(mean={W_OFFSET}, std={W_SCALE}) -> exp(w) in (0,1)",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=int, default=0,
                        help="Physical NPU index to bind this benchmark to "
                             "(must not be the service's NPU).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Path to write the JSON metrics file.")
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("torch.npu not available; refusing to run", file=sys.stderr)
        return 2

    # Eagerly resolve custom op bindings so the AscendC kernel is loaded
    # before we enter any timing fence.
    import vllm_ascend.utils as _utils  # noqa: F401
    _utils.enable_custom_op()

    if not hasattr(torch.ops._C_ascend, "npu_rwkv7_alt_recurrent"):
        print("torch.ops._C_ascend.npu_rwkv7_alt_recurrent missing", file=sys.stderr)
        return 2

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    metadata = collect_metadata(args.device)
    results: list[dict] = []
    started_at = time.time()

    for batch_size in BATCH_SIZES:
        for backend_name, step_fn in BACKENDS:
            backend_row: dict = {
                "backend": backend_name,
                "batch_size": batch_size,
                "shape": {
                    "state": [batch_size, H_HEADS, D_HEAD_DIM, V_VALUE_DIM],
                    "w": [batch_size, H_HEADS, D_HEAD_DIM],
                    "kk": [batch_size, H_HEADS, D_HEAD_DIM],
                    "a": [batch_size, H_HEADS, D_HEAD_DIM],
                    "k": [batch_size, H_HEADS, D_HEAD_DIM],
                    "v": [batch_size, H_HEADS, V_VALUE_DIM],
                    "r": [batch_size, H_HEADS, D_HEAD_DIM],
                    "output": [batch_size, H_HEADS, V_VALUE_DIM],
                    "final_state": [batch_size, H_HEADS, D_HEAD_DIM, V_VALUE_DIM],
                },
                "repetitions": [],
                "aggregate": None,
                "parity": None,
            }

            for rep in range(REPETITIONS):
                seed = seed_for(BASE_SEED, rep)
                s0, w, kk, a, k, v, r = make_inputs(batch_size, seed, device)

                # Contract / parity: single-step, OUTSIDE the timing fence.
                ref_state, ref_out = torch_ref_step(
                    s0.clone(), w.clone(), kk.clone(), a.clone(),
                    k.clone(), v.clone(), r.clone(),
                )
                torch.npu.synchronize()

                if backend_name == "torch_reference":
                    # Reference vs reference is exact.
                    report = ContractReport(
                        shape_ok=True,
                        dtype_ok=True,
                        device_ok=True,
                        finite_ok=bool(
                            torch.isfinite(ref_state).all().item()
                            and torch.isfinite(ref_out).all().item()
                        ),
                        max_abs_state=0.0,
                        max_abs_output=0.0,
                        max_rel_state=0.0,
                        max_rel_output=0.0,
                        passed=True,
                        detail="reference backend compared against itself",
                    )
                else:
                    report = parity_check(
                        step_fn,
                        s0.clone(), w.clone(), kk.clone(), a.clone(),
                        k.clone(), v.clone(), r.clone(),
                        ref_state, ref_out,
                    )
                if not report.passed:
                    print(
                        f"PARITY FAIL backend={backend_name} B={batch_size} "
                        f"rep={rep} seed={seed} :: {report.detail}",
                        file=sys.stderr,
                    )

                # Timing fence on a third fresh clone.
                sample = timed_chain(
                    step_fn,
                    s0.clone(), w.clone(), kk.clone(), a.clone(),
                    k.clone(), v.clone(), r.clone(),
                )

                backend_row["repetitions"].append({
                    "rep": rep,
                    "seed": seed,
                    "elapsed_ms": sample.elapsed_ms,
                    "per_step_ms": sample.per_step_ms,
                    "parity_max_abs_state": report.max_abs_state,
                    "parity_max_abs_output": report.max_abs_output,
                    "parity_max_rel_state": report.max_rel_state,
                    "parity_max_rel_output": report.max_rel_output,
                    "parity_passed": report.passed,
                })

            samples = [
                TimedSample(rep=r["rep"], seed=r["seed"],
                            elapsed_ms=r["elapsed_ms"], per_step_ms=r["per_step_ms"])
                for r in backend_row["repetitions"]
            ]
            backend_row["aggregate"] = aggregate(samples)
            # Surface the worst-case parity across the 3 reps at row level.
            worst_abs_state = max(r["parity_max_abs_state"] for r in backend_row["repetitions"])
            worst_abs_output = max(r["parity_max_abs_output"] for r in backend_row["repetitions"])
            worst_rel_state = max(r["parity_max_rel_state"] for r in backend_row["repetitions"])
            worst_rel_output = max(r["parity_max_rel_output"] for r in backend_row["repetitions"])
            all_passed = all(r["parity_passed"] for r in backend_row["repetitions"])
            backend_row["parity"] = {
                "atol": ATOL,
                "rtol": RTOL,
                "max_abs_state": worst_abs_state,
                "max_abs_output": worst_abs_output,
                "max_rel_state": worst_rel_state,
                "max_rel_output": worst_rel_output,
                "passed": all_passed,
                "tolerance_check": (
                    f"max_abs_output <= {ATOL} + {RTOL} * max|ref_out| per rep"
                ),
            }
            results.append(backend_row)
            print(
                f"  backend={backend_name:<24} B={batch_size:>4} "
                f"elapsed_ms(reps)=[{', '.join(f'{r.elapsed_ms:.4f}' for r in samples)}] "
                f"per_step_us(median)={backend_row['aggregate']['per_step_ms']['median']*1000:.2f}",
                flush=True,
            )

    payload = {
        "metadata": metadata,
        "results": results,
        "wall_clock_seconds": round(time.time() - started_at, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
