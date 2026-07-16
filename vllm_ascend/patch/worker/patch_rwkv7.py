#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
RWKV7 Ascend patch — DISABLED.

vLLM commit 064c0dd19 refactored the RWKV7 model to be fully self-contained.
The following APIs no longer exist and cannot be patched:

- ``vllm.model_executor.layers.fla.ops.rwkv7`` (removed entirely)
- ``RWKV7Attention._mix_recurrent_inputs`` (replaced by _project_recurrent_inputs)
- ``RWKV7Attention._prepare_recurrent_key_terms`` (integrated into _project_recurrent_inputs)
- ``_can_use_rwkv7_fused_recurrent``, ``_can_use_rwkv7_alt_recurrent`` (removed)
- ``perf_flags`` attribute (removed)

The current RWKV7 model uses inline _rwkv7_recurrent_scan and does not route
through any external ops module. The Ascend Triton kernels
(rwkv7_mix6, rwkv7_kk_pre, rwkv7_lnx_rkvres_xg, fused_recurrent_rwkv7) are
still valid standalone implementations, but there is no integration point in
the current upstream model to wire them in.

This patch is a no-op. The upstream reference path is used.
"""
# No-op: the old integration path is gone. The patch module exists solely to
# avoid import errors when this file is imported by patch/worker/__init__.py.
# All actual RWKV7 functionality falls through to the upstream reference path.
pass