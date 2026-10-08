"""Kimi Delta Attention (KDA) prefill for AMD MI350X / gfx950 (HIP).

    from kda_mi350x import kda_prefill
    meta = prepare(cu_seqlens)          # optional, untimed: per-batch metadata from cu_seqlens only
    out, final_state = kda_prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, cu_seqlens,
                                   scale=128 ** -0.5, lower_bound=-5.0, meta=meta)

Shapes (varlen, packed): q, k, v, raw_g [1, T, H, 128] bf16; raw_beta [1, T, H] bf16; A_log [H] f32;
dt_bias [H * 128] f32; initial_state [N, H, 128, 128] f32 (V-major, or an empty tensor for zeros);
cu_seqlens [N + 1] int64. Gate, beta sigmoid and q/k L2 norm are computed in the kernel, as in
fla's `chunk_kda(..., use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True,
safe_gate=True, state_v_first=True)`.

The extension is built on first import with torch.utils.cpp_extension (ROCm PyTorch, ~1-2 min) and cached.
"""
import hashlib
from pathlib import Path

from torch.utils.cpp_extension import load

_D = Path(__file__).resolve().parent / "csrc"
_FLAGS = ["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16"]
_srcs = {f: (_D / f).read_text() for f in ("kernel.h", "kernel.cu", "main.cpp")}
_dig = hashlib.sha1(("".join(_srcs.values()) + " ".join(_FLAGS)).encode()).hexdigest()[:12]
_b = Path(__file__).resolve().parent / "build" / _dig
_b.mkdir(parents=True, exist_ok=True)
for f, s in _srcs.items():
    if not (_b / f).exists() or (_b / f).read_text() != s:  # unchanged sources keep their mtime: no rebuild
        (_b / f).write_text(s)
_ext = load(name=f"kda_mi350x_{_dig}", sources=[str(_b / "main.cpp"), str(_b / "kernel.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=_FLAGS, build_directory=str(_b), verbose=False)

run = _ext.run
prepare = _ext.prepare


def kda_prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, cu_seqlens, scale=None, lower_bound=-5.0,
                meta=None):
    """Returns (output [1, T, H, 128] bf16, final_state [N, H, 128, 128] f32)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if meta is None:
        meta = prepare(cu_seqlens)
    r = run(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias, state=initial_state,
            cu_seqlens=cu_seqlens, scale=scale, lower_bound=lower_bound, meta=meta)
    return r["output"], r["new_state"]
