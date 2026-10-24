"""Launchers for the two decode baselines (AITER, vLLM HIP), on the benchmark's canonical inputs.

aiter : AITER fused_kda_decode (ae984f5, 2026-10-08) with the gfx950 speculative path of ROCm/aiter#5709
        (parallel-V recurrence + finalize kernel); the generic fused kernel otherwise. Triton. All shapes.
vllm  : vLLM's hand-written ROCm Kimi-K3 fused decode kernel (csrc/libtorch_stable/kimi_k3/
        fused_kda_decode_kernel_rocm.cu, vLLM 0f112d1). HIP. Plain decode only (S = 1), H in {12, 24, 48, 96}.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "aiter"))

from fused_kda_decode import fused_kda_decode  # noqa: E402

_vllm_ops = None


def _vllm():
    global _vllm_ops
    if _vllm_ops is None:
        torch.ops.load_library(os.path.join(HERE, "vllm_hip", "libkda_decode_sota.so"))
        _vllm_ops = torch.ops.kda_decode_sota
    return _vllm_ops


def vllm_supported(H, S):
    return S == 1 and H in (12, 24, 48, 96)


def run_aiter(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
              state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    H = A_log.numel()
    spec = num_accepted_tokens is not None
    fused_kda_decode(
        mixed_qkv, conv_state.transpose(1, 2), conv_weight, gate.unsqueeze(0), beta.unsqueeze(0), out_gate,
        A_log, dt_bias, ssm_state, state_indices, cu_seqlens, norm_weight, norm_eps, 128, H, lower_bound,
        num_accepted_tokens=num_accepted_tokens,
        conv_state_indices=state_indices[:, 0] if spec else None,
        out=out,
    )


def run_vllm(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
             state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    assert num_accepted_tokens is None
    T, H = beta.shape
    _vllm().fused_kda_decode(
        mixed_qkv, conv_weight, None, conv_state.transpose(1, 2), gate.unsqueeze(0), beta.unsqueeze(0),
        A_log, dt_bias, state_indices, ssm_state, out.view(1, T, H, 128), lower_bound,
        out_gate.view(T, H, 128), norm_weight, norm_eps,
    )


BASELINES = {"aiter": run_aiter, "vllm": run_vllm}
