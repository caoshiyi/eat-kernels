"""Kimi Delta Attention (KDA) forward for NVIDIA B200 (sm_100a), plain calls (benchmark track).

    from kda_b200_int21 import run, prepare
    out, final_state = run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)
    launch = prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)   # host-planned, fixed layout
    out, final_state = launch()

Shapes: q, k, v, g [1, T, H, 128] bf16; beta [1, T, H] bf16 (logits); A_log [H] fp32; dt_bias [H*128] fp32;
initial_state [N, H, 128 (V), 128 (K)] fp32, value-first, or None; cu_seqlens [N+1] int64 or None. Semantics match
FLA's chunk_kda(..., use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True,
safe_gate=True, lower_bound=-5.0, state_v_first=True). No state pool, checkpoints or graph-safe dynamic layouts:
use ../b200 (kda_b200) for serving.
"""
import importlib.util
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "kernel.py")
sys.path.insert(0, os.path.dirname(_SRC))
_spec = importlib.util.spec_from_file_location("kda_b200_int21_impl", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run = _mod.run
prepare = _mod.prepare

__all__ = ["run", "prepare"]
