"""Production baselines for benchmark/eval_kda_mi350x.py, called with exactly the production contract.

prefill: vLLM's ROCm Kimi-K3 fused KDA prefill (vLLM 9521c60, `chunk_kda_prefill` with the fused backend:
         `fused_kda_prologue` + `fused_kda_chunk`, which read and write the paged state pool in place, honour
         `has_initial_state`, write the prefix-cache checkpoint rows, and read the strided beta directly).
         Python side follows prefill/vllm_reference/kda_chunk.py; kernels in prefill/fused_kda_chunk_kernel_rocm.cu.
decode:  the faster of AITER `fused_kda_decode` (ae984f5, gfx950 speculative path of ROCm/aiter#5709) and vLLM's ROCm
         fused decode kernel (vLLM 0f112d1; plain decode at H in {12, 24, 48, 96}), as in the phase-1 decode benchmark.
Build the two HIP libraries first: benchmark/build_baselines.sh.
"""
import os
import sys
from functools import cache

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "decode"))

from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets  # noqa: E402

CHUNK = 64
D = 128
_BLOCKS_P1, _BLOCKS_P2 = 2, 1
_MIN_LEN, _DEEP_LEN, _MAX_GROUPS = 4, 12, 32
_pre_ops = None


def _prefill_ops():
    global _pre_ops
    if _pre_ops is None:
        torch.ops.load_library(os.path.join(HERE, "prefill", "libkda_sota.so"))
        _pre_ops = torch.ops.kda_sota
    return _pre_ops


@cache
def _num_cus(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count


def _chunk_groups(chunks_per_seq, num_seqs, num_heads, dev):
    cus = _num_cus(dev)
    nh = max(num_seqs * num_heads, 1)
    fill1 = cus // (_BLOCKS_P1 * nh)
    fill2 = cus // (_BLOCKS_P2 * nh)
    cand = fill1
    if fill2 > 0 and chunks_per_seq // fill2 >= _DEEP_LEN:
        cand = fill2
    cand = min(cand, chunks_per_seq // _MIN_LEN, _MAX_GROUPS)
    return cand if cand >= 4 else 1


def prefill_prepare(cu_seqlens, num_heads, checkpoint_offsets=None, checkpoint_state_indices=None, device="cuda"):
    """Per-batch metadata vLLM's KDA backend builds once and shares across all KDA layers (not timed)."""
    cu = cu_seqlens.to(device=device, dtype=torch.int32)
    return dict(cu32=cu, chunk_indices=prepare_chunk_indices(cu.long(), CHUNK).to(torch.int32),
                chunk_offsets=prepare_chunk_offsets(cu, CHUNK))


def prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices,
            has_initial_state, cu_seqlens, out, checkpoint_offsets=None, checkpoint_state_indices=None, meta=None):
    ops = _prefill_ops()
    if meta is None:
        meta = prefill_prepare(cu_seqlens, q.shape[2], device=q.device)
    _, T, H, _ = q.shape
    N = cu_seqlens.numel() - 1
    dev = q.device
    ci = meta["chunk_indices"]
    nc = ci.shape[0]

    def like(last):
        return torch.empty(1, T, H, last, dtype=q.dtype, device=dev)

    qg, w, u = like(D), like(D), like(D)
    kg_t = torch.empty(nc, H, D, CHUNK, dtype=q.dtype, device=dev)
    aqk = like(CHUNK)
    decay = torch.empty(nc, H, D, dtype=torch.float32, device=dev)
    ops.fused_kda_prologue(q, k, v, raw_g, raw_beta, A_log.reshape(-1), dt_bias.reshape(-1), qg, w, u, kg_t, aqk,
                           decay, meta["cu32"], ci, None, None, None, None, scale, lower_bound)
    groups = _chunk_groups(nc // max(N, 1), N, H, dev.index or 0)
    gs = None
    if groups > 1:
        planes = groups * N * H
        gs = torch.empty(2 * planes * D * D + planes * D + (planes * D * D + 1) // 2, dtype=torch.float32, device=dev)
    ck = checkpoint_offsets is not None
    ops.fused_kda_chunk(qg, w, u, kg_t, aqk, decay, None, None, out, meta["cu32"], meta["chunk_offsets"], scale, gs,
                        groups, state_cache if ck else None, checkpoint_offsets if ck else None,
                        checkpoint_state_indices if ck else None, state_cache, state_indices, has_initial_state)


# ---------------------------------------------------------------------------------------------- decode
_dec = None


def _decode_mod():
    global _dec
    if _dec is None:
        import sota_decode
        _dec = sota_decode
    return _dec


def decode_baselines(H, S):
    m = _decode_mod()
    fns = {"aiter": m.run_aiter}
    if m.vllm_supported(H, S):
        fns["vllm"] = m.run_vllm
    return fns
