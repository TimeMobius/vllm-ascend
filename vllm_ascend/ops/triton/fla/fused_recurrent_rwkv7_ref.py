# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Reference implementation of RWKV7 recurrent operation.
# This is used as fallback when triton is not available.
# ruff: noqa: E501
# mypy: ignore-errors

import torch


def rwkv7_recurrent_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference implementation of RWKV7 recurrent operation.

    This is the pure PyTorch implementation that serves as:
    1. Ground truth for testing the triton kernel
    2. Fallback when triton is not available

    Args:
        r (torch.Tensor): Reception gate of shape [B, T, H, K]
        w (torch.Tensor): Forget gate of shape [B, T, H, K] (in log space)
        k (torch.Tensor): Key of shape [B, T, H, K]
        v (torch.Tensor): Value of shape [B, T, H, V]
        kk (torch.Tensor): Key normalization of shape [B, T, H, K]
        a (torch.Tensor): A gate of shape [B, T, H, K]
        initial_state (torch.Tensor, optional): Initial state of shape [N, H, K, V]
        output_final_state (bool): Whether to output final state
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths
        scale (float): Scaling factor for r

    Returns:
        tuple: (output, final_state)
            - output: [B, T, H, V] output tensor
            - final_state: [N, H, K, V] or None
    """
    out, final_state, _ = _rwkv7_recurrent_reference_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    return out, final_state


def rwkv7_recurrent_reference_with_checkpoints(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Reference implementation of RWKV7 recurrent with checkpoint support.

    Args:
        r (torch.Tensor): Reception gate of shape [B, T, H, K]
        w (torch.Tensor): Forget gate of shape [B, T, H, K] (in log space)
        k (torch.Tensor): Key of shape [B, T, H, K]
        v (torch.Tensor): Value of shape [B, T, H, V]
        kk (torch.Tensor): Key normalization of shape [B, T, H, K]
        a (torch.Tensor): A gate of shape [B, T, H, K]
        initial_state (torch.Tensor, optional): Initial state of shape [N, H, K, V]
        output_final_state (bool): Whether to output final state
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths
        checkpoint_positions (torch.Tensor, optional): Positions to save checkpoints
        checkpoint_offsets (torch.Tensor, optional): Offsets for each sequence's checkpoints
        output_checkpoint_states (bool): Whether to output checkpoint states
        scale (float): Scaling factor for r

    Returns:
        tuple: (output, final_state, checkpoint_states)
    """
    return _rwkv7_recurrent_reference_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        output_checkpoint_states=output_checkpoint_states,
        scale=scale,
    )


def _rwkv7_recurrent_reference_impl(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Internal implementation of RWKV7 recurrent."""
    if r.ndim != 4:
        raise ValueError(f"`r` must be 4D, got {r.ndim}.")
    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError("When `cu_seqlens` is provided, the batch size must be 1.")
    if output_checkpoint_states and (
        checkpoint_positions is None or checkpoint_offsets is None
    ):
        raise ValueError(
            "`checkpoint_positions` and `checkpoint_offsets` are required "
            "when `output_checkpoint_states=True`."
        )

    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    out = torch.empty_like(v)

    if output_final_state:
        if initial_state is None:
            final_state = torch.zeros(
                (N, H, K, V),
                device=r.device,
                dtype=torch.float32,
            )
        else:
            final_state = initial_state.to(torch.float32).clone()
    else:
        final_state = None

    if output_checkpoint_states:
        assert checkpoint_offsets is not None
        num_checkpoints = int(checkpoint_offsets[-1].item())
        checkpoint_states = torch.empty(
            (num_checkpoints, H, K, V),
            device=r.device,
            dtype=torch.float32,
        )
    else:
        checkpoint_states = None

    for seq_idx in range(N):
        batch_idx = 0 if cu_seqlens is not None else seq_idx
        if cu_seqlens is None:
            start = seq_idx * T
            end = start + T
        else:
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())

        if initial_state is None:
            state = torch.zeros((H, K, V), device=r.device, dtype=torch.float32)
        else:
            state = initial_state[seq_idx].to(torch.float32).clone()

        if output_checkpoint_states:
            assert checkpoint_positions is not None
            assert checkpoint_offsets is not None
            checkpoint_idx = int(checkpoint_offsets[seq_idx].item())
            checkpoint_end = int(checkpoint_offsets[seq_idx + 1].item())
        else:
            checkpoint_idx = 0
            checkpoint_end = 0

        for tok_idx in range(start, end):
            local_token_idx = tok_idx - start
            tensor_token_idx = tok_idx if cu_seqlens is not None else local_token_idx

            # RWKV7 recurrent state update
            # b_act_a = -kk
            # b_b = kk * a
            # state = exp(w) * state + b_b * sum(b_act_a * state) + k * v
            sa = (state * (-kk[batch_idx, tensor_token_idx]).unsqueeze(-1)).sum(dim=-2)
            state = (
                torch.exp(w[batch_idx, tensor_token_idx]).unsqueeze(-1) * state
                + (
                    kk[batch_idx, tensor_token_idx] * a[batch_idx, tensor_token_idx]
                ).unsqueeze(-1)
                * sa.unsqueeze(-2)
                + k[batch_idx, tensor_token_idx].unsqueeze(-1)
                * v[batch_idx, tensor_token_idx].unsqueeze(-2)
            )

            # Output: o = sum(state * r)
            out[batch_idx, tensor_token_idx] = (
                (state * (r[batch_idx, tensor_token_idx] * scale).unsqueeze(-1))
                .sum(dim=-2)
                .to(out.dtype)
            )

            # Checkpoint handling
            if (
                checkpoint_states is not None
                and checkpoint_idx < checkpoint_end
                and local_token_idx == int(checkpoint_positions[checkpoint_idx].item())
            ):
                checkpoint_states[checkpoint_idx] = state
                checkpoint_idx += 1

        if final_state is not None:
            final_state[seq_idx] = state

    return out, final_state, checkpoint_states