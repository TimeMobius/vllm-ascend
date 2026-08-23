#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""RWKV7 configuration model and resolver.

The four formal RWKV7 configuration knobs live under the single
``VLLM_ASCEND_RWKV7_*`` namespace:

- ``VLLM_ASCEND_RWKV7_PRESET``: deployment-facing policy (reference/auto/
  throughput). The full hierarchy is:

  .. code-block:: text

      default < PRESET expansion < RECURRENT_BACKEND explicit < OPERATOR_OVERRIDES["recurrent"]

- ``VLLM_ASCEND_RWKV7_RECURRENT_BACKEND``: expert override of the recurrent
  path (auto/reference/ascendc/triton_t1/triton_t1_cache). It only controls
  the core WKV7 recurrent dispatches; it has higher priority than the PRESET
  recurrent choice but lower than an explicit ``recurrent`` key inside
  ``OPERATOR_OVERRIDES``.
- ``VLLM_ASCEND_RWKV7_OBSERVABILITY``: off/summary. Replaces the boolean
  ``VLLM_ASCEND_RWKV7_PROFILE``.
- ``VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES``: strict JSON, highest precedence,
  developer-only. Keys: ``mix6``/``kk_pre``/``epilogue``/``block_norms`` with
  values ``triton``/``reference``/``auto``, plus an optional ``recurrent`` key
  whose value is one of the recurrent-backend enum values.

Every consumer snapshots the resolved config at import time, so it is never
re-parsed per decode step. The global ``VLLM_ASCEND_RWKV7_DISABLE_TRITON``
kill switch remains a separate, highest-priority safety valve applied by the
dispatch guards themselves (it is not part of this hierarchy).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum

_PRESET_ENV = "VLLM_ASCEND_RWKV7_PRESET"
_RECURRENT_BACKEND_ENV = "VLLM_ASCEND_RWKV7_RECURRENT_BACKEND"
_OBSERVABILITY_ENV = "VLLM_ASCEND_RWKV7_OBSERVABILITY"
_OPERATOR_OVERRIDES_ENV = "VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES"

# The per-operator keys accepted inside VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES.
_OPERATOR_KEYS: tuple[str, ...] = ("mix6", "kk_pre", "epilogue", "block_norms")

# The "recurrent" key inside OPERATOR_OVERRIDES is the highest-precedence
# recurrent selection. It takes the same values as RECURRENT_BACKEND.
_RECURRENT_KEY = "recurrent"

# Presets expand to a default recurrent backend + default per-operator mode.
preset_recurrent_defaults: dict[str, str] = {
    "reference": "reference",
    "auto": "auto",
    "throughput": "triton_t1_cache",
}
# reference: all Triton operators disabled; auto: guard-based; throughput:
# verified fused operators enabled.
preset_operator_defaults: dict[str, dict[str, str]] = {
    "reference": {key: "reference" for key in _OPERATOR_KEYS},
    "auto": {key: "auto" for key in _OPERATOR_KEYS},
    "throughput": {key: "triton" for key in _OPERATOR_KEYS},
}


class RWKV7OperatorOverride(str, Enum):
    """Per-operator override values accepted inside OPERATOR_OVERRIDES."""

    TRITON = "triton"
    REFERENCE = "reference"
    AUTO = "auto"


class RWKV7Preset(str, Enum):
    """Deployment-facing RWKV7 policy preset."""

    REFERENCE = "reference"
    AUTO = "auto"
    THROUGHPUT = "throughput"


class RWKV7Observability(str, Enum):
    """RWKV7 observability level (replaces the boolean PROFILE switch)."""

    OFF = "off"
    SUMMARY = "summary"


class RWKV7RecurrentBackend(str, Enum):
    """Mutually-exclusive RWKV7 recurrent backend selection."""

    AUTO = "auto"
    REFERENCE = "reference"
    ASCENDC = "ascendc"
    TRITON_T1 = "triton_t1"
    TRITON_T1_CACHE = "triton_t1_cache"

    # Use the AscendC ``npu_rwkv7_alt_recurrent`` kernel when its runtime
    # guard succeeds (auto/ascendc), otherwise fall back to the reference.
    @property
    def uses_ascendc(self) -> bool:
        return self in (RWKV7RecurrentBackend.AUTO,
                        RWKV7RecurrentBackend.ASCENDC)

    # Use the T=1 Triton decode kernel (non persistent-cache).
    @property
    def uses_triton_t1(self) -> bool:
        return self in (RWKV7RecurrentBackend.TRITON_T1,
                        RWKV7RecurrentBackend.TRITON_T1_CACHE)

    # Prefer the persistent-cache T=1 Triton decode kernel.
    @property
    def uses_persistent_cache(self) -> bool:
        return self is RWKV7RecurrentBackend.TRITON_T1_CACHE

    @classmethod
    def parse(cls, value: str, *, env: str = _RECURRENT_BACKEND_ENV) -> RWKV7RecurrentBackend:
        """Strictly parse a recurrent backend value, failing loudly."""
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(f"'{member.value}'" for member in cls)
            raise ValueError(
                f"Invalid value for {env}: {value!r}. Valid values are: {valid}."
            ) from exc


@dataclass(frozen=True, slots=True)
class _OperatorOverrides:
    """Strictly-parsed OPERATOR_OVERRIDES object.

    ``operator`` maps the four operator keys to a mode; ``recurrent`` is the
    optional highest-precedence recurrent selection (``None`` when omitted).
    """

    operator: dict[str, RWKV7OperatorOverride]
    recurrent: RWKV7RecurrentBackend | None

    @classmethod
    def parse(cls, raw: str | None) -> _OperatorOverrides | None:
        """Parse the raw env value, or ``None`` when unset/blank.

        Raises :class:`ValueError` naming ``VLLM_ASCEND_RWKV7_OPERATOR_OVERRIDES``
        for any violation of the strict JSON contract.
        """
        if raw is None or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON for {_OPERATOR_OVERRIDES_ENV}: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"{_OPERATOR_OVERRIDES_ENV} must be a JSON object, got "
                f"{type(parsed).__name__}"
            )
        operator: dict[str, RWKV7OperatorOverride] = {}
        recurrent: RWKV7RecurrentBackend | None = None
        valid_keys = ", ".join([*_OPERATOR_KEYS, _RECURRENT_KEY])
        for key, value in parsed.items():
            if key == _RECURRENT_KEY:
                if not isinstance(value, str):
                    raise ValueError(
                        f"Invalid value {value!r} for key {_RECURRENT_KEY!r} in "
                        f"{_OPERATOR_OVERRIDES_ENV}."
                    )
                recurrent = RWKV7RecurrentBackend.parse(
                    value, env=_OPERATOR_OVERRIDES_ENV
                )
                continue
            if key not in _OPERATOR_KEYS:
                raise ValueError(
                    f"Unknown key {key!r} in {_OPERATOR_OVERRIDES_ENV}; valid "
                    f"keys are: {valid_keys}."
                )
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid value {value!r} for key {key!r} in "
                    f"{_OPERATOR_OVERRIDES_ENV}; valid values are "
                    "'triton', 'reference', 'auto'."
                )
            try:
                operator[key] = RWKV7OperatorOverride(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value {value!r} for key {key!r} in "
                    f"{_OPERATOR_OVERRIDES_ENV}; valid values are "
                    "'triton', 'reference', 'auto'."
                ) from exc
        return cls(operator=operator, recurrent=recurrent)


@dataclass(frozen=True, slots=True)
class ResolvedRWKV7Config:
    """The resolved, snapshot-able RWKV7 configuration.

    ``operator`` uses the same keys as OPERATOR_OVERRIDES; each value is
    ``True`` when the operator's Triton candidate is allowed (runtime-guarded)
    and ``False`` when it is forced to the reference implementation.
    """

    preset: RWKV7Preset
    recurrent_backend: RWKV7RecurrentBackend
    operator: dict[str, bool]
    observability: RWKV7Observability

    def operator_enabled(self, key: str) -> bool:
        return self.operator[key]

    def recurrent_enabled(self) -> bool:
        """True when the resolved recurrent backend may use an accelerated path.

        ``reference`` disables all accelerated recurrent kernels; every other
        backend is guard-gated at the dispatch site.
        """
        return self.recurrent_backend is not RWKV7RecurrentBackend.REFERENCE

    def observability_enabled(self) -> bool:
        return self.observability is RWKV7Observability.SUMMARY


def _parse_preset(raw: str | None) -> RWKV7Preset:
    value = raw or "auto"
    try:
        return RWKV7Preset(value)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in RWKV7Preset)
        raise ValueError(
            f"Invalid value for {_PRESET_ENV}: {value!r}. Valid values are: "
            f"{valid}."
        ) from exc


def _parse_observability(raw: str | None) -> RWKV7Observability:
    value = raw or "off"
    try:
        return RWKV7Observability(value)
    except ValueError as exc:
        valid = ", ".join(f"'{member.value}'" for member in RWKV7Observability)
        raise ValueError(
            f"Invalid value for {_OBSERVABILITY_ENV}: {value!r}. Valid values "
            f"are: {valid}."
        ) from exc


def resolve_rwkv7_config() -> ResolvedRWKV7Config:
    """Resolve the RWKV7 configuration from the environment.

    Precedence (lowest to highest): built-in defaults, PRESET expansion,
    explicit RECURRENT_BACKEND env, then OPERATOR_OVERRIDES (for both operator
    modes and the optional ``recurrent`` key).
    """
    preset = _parse_preset(os.getenv(_PRESET_ENV))
    observability = _parse_observability(os.getenv(_OBSERVABILITY_ENV))

    # Recurrent: start from the preset expansion, then apply the explicit
    # RECURRENT_BACKEND override if the env var is set, then the highest-
    # precedence recurrent key in OPERATOR_OVERRIDES.
    recurrent_raw = preset_recurrent_defaults[preset.value]
    if os.getenv(_RECURRENT_BACKEND_ENV) is not None:
        recurrent_raw = os.environ[_RECURRENT_BACKEND_ENV]
    recurrent_backend = RWKV7RecurrentBackend.parse(
        recurrent_raw, env=_RECURRENT_BACKEND_ENV
    )

    # Operators: start from the preset expansion, then apply OPERATOR_OVERRIDES.
    operator_modes = dict(
        preset_operator_defaults[preset.value]
    )  # type: dict[str, str]
    overrides = _OperatorOverrides.parse(os.getenv(_OPERATOR_OVERRIDES_ENV))
    if overrides is not None:
        for key, mode in overrides.operator.items():
            operator_modes[key] = mode.value
        if overrides.recurrent is not None:
            recurrent_backend = overrides.recurrent

    operator = {
        key: operator_modes[key] != RWKV7OperatorOverride.REFERENCE.value
        for key in _OPERATOR_KEYS
    }
    return ResolvedRWKV7Config(
        preset=preset,
        recurrent_backend=recurrent_backend,
        operator=operator,
        observability=observability,
    )
