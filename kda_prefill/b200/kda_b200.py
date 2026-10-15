"""Kimi Delta Attention (KDA) forward / chunked prefill for NVIDIA B200 (sm_100a).

    from kda_b200 import kda_prefill
    out, final_state = kda_prefill(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens)

Shapes (varlen, packed): q, k, v, g [1, T, H, 128] bf16; beta [1, T, H] bf16 (logits); A_log [H] fp32;
dt_bias [H * 128] fp32; initial_state [N, H, 128 (V), 128 (K)] fp32, value-first; cu_seqlens [N + 1] int64 or None
(one sequence). Gate, beta sigmoid and the q / k L2 norm are computed in the kernel, as in FLA's
`chunk_kda(..., use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True,
safe_gate=True, lower_bound=-5.0, state_v_first=True)`. Per sequence and head, token by token:

    q, k <- L2-normalized;  beta <- sigmoid(beta);  a_t = -5 * sigmoid(exp(A_log) * (g_t + dt_bias))
    S <- S * exp(a_t);  S <- S + beta_t * k_t (v_t - k_t^T S);  o_t = scale * q_t^T S

Serving features (keyword arguments, FlashInfer `recurrent_kda` prefill contract as used by SGLang):
  ssm_state_indices          int32 [N]: initial_state is a state pool [P, H, 128, 128] (slot stride >= H*128*128,
                             16-byte aligned); sequence i reads and updates slot ssm_state_indices[i] in place, every
                             other slot stays bit-identical, and final_state is the pool itself.
  state_checkpoints          fp32 or bf16 [C, H, 128, 128] (contiguous; bf16 rows are the fp32 states rounded to nearest even), with checkpoint_cu_starts int64 [N + 1] and
  checkpoint_every_n_tokens  n (multiple of 64; 0 = off): sequence i writes row checkpoint_cu_starts[i] + j = its
                             state before token j * n, j = 0 .. ceil(len_i / n) - 1. The call then returns
                             (output, final_state, state_checkpoints).
  initial_state_source,      [P0, H, 128, 128] + int32 [N]: sequence i starts from the read-only slot
  initial_state_indices      initial_state_source[initial_state_indices[i]]; its final state still goes to its pool slot.

`prepare(...)` takes the same arguments plus `dynamic_layout` and returns a zero-argument `launch()` that re-reads the
current contents of every input buffer on each call. With `dynamic_layout=True` the launch can be captured in a CUDA
graph once and replayed after the contents of cu_seqlens, ssm_state_indices and checkpoint_cu_starts change (same
shapes, same total token count, every sequence non-empty): schedules are derived on the device.

The CUDA extensions are built with torch.utils.cpp_extension on first use (build directories next to the sources
under csrc/); each kernel variant is compiled the first time a call needs it.
"""
import importlib.util
import os
import sys

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
sys.path.insert(0, _CSRC)
_spec = importlib.util.spec_from_file_location("kda_b200_impl", os.path.join(_CSRC, "kernel.py"))
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

run = _impl.run
"""run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, **serving_kwargs) -> (output, final_state[, state_checkpoints])

The serving entry point: device-planned (no host reads of cu_seqlens), plans cached per layout."""

prepare = _impl.prepare
"""prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, *, dynamic_layout=False, **serving_kwargs) -> launch

dynamic_layout=False plans for the given layout on the host (the layout must not change between launches);
dynamic_layout=True returns the CUDA-graph-safe launch described in the module docstring."""


def status_bits(launch):
    """Table-validation bits of the most recent call of a launch from prepare() with serving features or
    dynamic_layout=True (0 = every caller table was valid; synchronizes): 1 = rejected cu_seqlens (every sequence treated
    as empty: states passed through, output zero-filled), 2 = bad / repeated ssm_state_indices entry, 4 = bad
    initial_state_indices entry, 8 = checkpoint_cu_starts rows outside state_checkpoints."""
    status = getattr(launch, "status", None)
    return 0 if status is None else _impl._w1.status_bits(status)


def kda_prefill(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens=None, scale=None, *,
                ssm_state_indices=None, state_checkpoints=None, checkpoint_cu_starts=None, checkpoint_every_n_tokens=0,
                initial_state_source=None, initial_state_indices=None):
    """Keyword-friendly wrapper of `run` (same semantics). scale defaults to 1 / sqrt(128).

    Returns (output [1, T, H, 128] bf16, final_state) or, with checkpoints, (output, final_state, state_checkpoints)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens,
               ssm_state_indices=ssm_state_indices, state_checkpoints=state_checkpoints,
               checkpoint_cu_starts=checkpoint_cu_starts, checkpoint_every_n_tokens=checkpoint_every_n_tokens,
               initial_state_source=initial_state_source, initial_state_indices=initial_state_indices)


__all__ = ["kda_prefill", "run", "prepare", "status_bits"]
