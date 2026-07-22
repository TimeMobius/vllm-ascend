# RWKV7 Branch Integration Plan

## Goal

Put current-upstream-compatible RWKV7 Ascend kernel integration on
`feat/rwkv7-ascend-support`; keep `fix/eagle-patch-api-compat` as its descendant
with only version/API compatibility deltas; verify real RWKV7 serving on fix.

## Topology

1. Implement and test RWKV7 integration on `feat`.
2. Rebase the three existing fix commits (`c6d355b4`, `79714925`, `a40dade9`)
   onto the new feature tip.
3. Add compatibility-only fixes on `fix` only when required by the real service.
4. Prove `feat` is an ancestor of `fix` and run real service smoke tests on fix.

## Work

- Patch current upstream RWKV7 helper methods/functions, not removed legacy FLA
  module APIs.
- Cover mix6, kk-pre, recurrent varlen scan, epilogue, and safe reference
  fallback; keep decode single-step reference unless profiling proves fusion wins.
- Add integration-level parity/dispatch tests.
- Validate import, unit tests, service startup, completion/chat, streaming, and
  optimized-path selection on the fix branch.

## Constraints

- Preserve `.omo/run-continuation/` and all unrelated user changes.
- Do not modify `/mnt/data/Codes/vllm`.
- Do not add the external FLA package as a runtime dependency.
- Keep compatibility changes separate from feature changes.
