#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Multi-stage build:
#   deps         - installs compilers + apt build deps (heavy, cached)
#   python-base  - clones vllm, installs vllm editable (cached unless vllm tag changes)
#   builder      - installs vllm-ascend editable + compiles CANN custom ops via build_aclnn.sh
#                  (cached unless csrc/** or setup.py change)
#   runtime      - minimal runtime image with CANN runtime libs + compiled .o binaries
#                  (rebuilt from scratch on vllm-ascend source edits but skips the
#                  ~30 min build_aclnn.sh compile thanks to COPY --from=builder)

ARG BASE_IMAGE=quay.io/ascend/cann:9.0.1-910b-ubuntu22.04-py3.12
ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# -----------------------------------------------------------------------------
# Stage 1: build dependencies
# Heavy: compilers, cmake, llvm-15, headers. Cached aggressively; rarely changes.
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS deps

ARG PIP_INDEX_URL

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git vim wget net-tools \
        gcc g++ cmake \
        numactl libnuma-dev libibverbs-dev libjemalloc2 libhiredis-dev \
        clang-15 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 20 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 20 && \
    pip config set global.index-url ${PIP_INDEX_URL} && \
    python3 -m pip install modelscope 'protobuf>3.20.0' && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: vllm editable install
# Cached unless VLLM_TAG / VLLM_COMMIT / requirements change.
# -----------------------------------------------------------------------------
FROM deps AS python-base

ARG MOONCAKE_TAG=0.3.11.post1
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=v0.25.1
ARG VLLM_COMMIT=""

WORKDIR /vllm-workspace

RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    python3 -m pip install mooncake-transfer-engine-npu==${MOONCAKE_TAG} --extra-index-url https://mirrors.aliyun.com/pypi/web/simple && \
    python3 -m pip cache purge

RUN if [ -n "$VLLM_COMMIT" ]; then \
      git init /vllm-workspace/vllm && \
      git -C /vllm-workspace/vllm fetch --depth 1 $VLLM_REPO "$VLLM_COMMIT" && \
      git -C /vllm-workspace/vllm checkout FETCH_HEAD; \
    else \
      git clone --depth 1 -b $VLLM_TAG $VLLM_REPO /vllm-workspace/vllm; \
    fi

# In x86, triton will be installed by vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it.
RUN VLLM_TARGET_DEVICE="empty" python3 -m pip install -e /vllm-workspace/vllm/[audio] --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton && \
    python3 -m pip cache purge

# -----------------------------------------------------------------------------
# Stage 3: vllm-ascend editable install + build_aclnn.sh compile
# Compiles 30+ CANN custom ops (~30 min). Cached when csrc/** is unchanged.
# -----------------------------------------------------------------------------
FROM python-base AS builder

ARG SOC_VERSION="ascend910b1"
ARG COMPILE_CUSTOM_KERNELS=1

ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION=$SOC_VERSION \
    TASK_QUEUE_ENABLE=1 \
    OMP_NUM_THREADS=1

WORKDIR /vllm-workspace

COPY . /vllm-workspace/vllm-ascend/
# csrc/build/ CMakeCache.txt hard-codes the host build path; must be wiped
# before build_aclnn.sh runs inside the container.
RUN rm -rf /vllm-workspace/vllm-ascend/csrc/build /vllm-workspace/vllm-ascend/csrc/build_out /vllm-workspace/vllm-ascend/csrc/output

RUN export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi" && \
    export VLLM_BATCH_INVARIANT=1 && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    python3 -m pip install -e /vllm-workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall "scipy==1.13.1" && \
    python3 -m pip uninstall -y triton triton-ascend && \
    python3 -m pip install --no-cache-dir --no-deps triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    python3 -m pip install --force-reinstall --no-deps triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    rm -rf /usr/local/python3.12.13/lib/python3.12/site-packages/numpy* && \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall "numpy==2.4.2" && \
    # CANN base image ships scipy 1.18 as a single-file _propack.cpython-312-*.so
    # (no __init__.py).  scipy 1.13 expects a PEP 420 namespace package
    # directory of sub-modules instead.  Wipe the stale single-file C
    # extension so the new install takes effect.
    rm -f /usr/local/python3.12.13/lib/python3.12/site-packages/scipy/sparse/linalg/_propack.cpython-312-aarch64-linux-gnu.so && \
    # CANN base image also pre-installs scipy 1.18.0 dist-info under
    # site-packages/, which shadows the new 1.13.1 dist-info and confuses
    # `pip show`.  Remove the stale dist-info so the new install is unambiguous.
    rm -f /usr/local/python3.12.13/lib/python3.12/site-packages/scipy-1.18.0.dist-info -r /usr/local/python3.12.13/lib/python3.12/site-packages/scipy.libs 2>/dev/null || true && \
    python3 -m pip cache purge

# -----------------------------------------------------------------------------
# Stage 4: final runtime image
# - Skips apt build deps (cmake, llvm-15-dev, headers) for a leaner base
# - Copies prebuilt vllm + vllm-ascend + compiled .o binaries from builder
# - Installs custom_transformer vendor into OPP and pins load_priority
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ARG PIP_INDEX_URL

# Runtime-only apt packages (build deps intentionally omitted)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git vim wget net-tools \
        numactl libnuma1 libibverbs1 libjemalloc2 libhiredis0.14 && \
    pip config set global.index-url ${PIP_INDEX_URL} && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Bring in everything we built (Python site-packages, vllm source tree, compiled CANN ops)
COPY --from=builder /usr/local/python3.12.13 /usr/local/python3.12.13
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /vllm-workspace /vllm-workspace

# Append libascend_hal.so path (devlib) and LD_LIBRARY_PATH for runtime
RUN echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" >> ~/.bashrc

# Required so aclnn loads custom_transformer via ASCEND_OPP_PATH
# without a host volume mount. Also pin load_priority=custom_transformer
# in opp/vendors/config.ini so the aclop runtime registers the .o binary
# (vllm-ascend's install.sh runs last for batch_invariant and would
# otherwise leave priority pointing at the wrong vendor).
RUN OPP_VENDOR_DIR=/usr/local/Ascend/cann-9.0.1/opp/vendors && \
    SRC=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer && \
    cp -al "$SRC" "$OPP_VENDOR_DIR/custom_transformer" || cp -r "$SRC" "$OPP_VENDOR_DIR/custom_transformer" && \
    [ -f "$OPP_VENDOR_DIR/custom_transformer/op_api/lib/libcust_opapi.so" ] && \
    printf 'load_priority=custom_transformer\n' > "$OPP_VENDOR_DIR/config.ini"

CMD ["/bin/bash"]