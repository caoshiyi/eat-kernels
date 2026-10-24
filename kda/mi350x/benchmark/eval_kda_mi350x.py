#!/usr/bin/env python3
"""
Evaluate the production KDA package (kda/kernel.py: prefill + decode) on AMD MI350X.

Score: mean_vs_base = sqrt(prefill_geomean x decode_geomean), each a geomean of (baseline latency / candidate latency)
over serving workloads called exactly as a serving engine calls them (vLLM's Kimi-K3 ROCm KDA layer):
  prefill  13 chunked-prefill steps drawn from the AgentX traces (H = 12 for Kimi-K3 TP8, H = 96 for TP1 / DP
           attention): post-conv q/k/v, raw gate, beta as a strided slice of the input projection, a paged fp32 state
           pool (vLLM page layout: conv and SSM state of a slot share one page) read and updated in place, per-sequence
           has_initial_state, one prefix-cache checkpoint per sequence written into the pool, output into a slice of
           the layer's output buffer. Eager call (vLLM breaks the KDA op out of graph capture), cache flushed, timed
           against vLLM's fused ROCm prefill on the same inputs.
  decode   10 decode / speculative-verify steps (Kimi-K3 DSpark, 7 drafts, accepted length 4.38 = SPEED-Bench coding;
           batch sizes over the AgentX concurrency range; H = 12 and 96), same page layout, strided beta / out_gate,
           HIP-graph replay, cache flushed, timed against the faster of AITER fused_kda_decode and vLLM's HIP kernel.
Baselines are timed live, interleaved with the candidate. Fresh random data every run (the batch layouts are fixed).

Hard gates (score 0 if any fails):
  1. every serving workload: output, every written pool row, the rest of the page buffer bit-identical, the output
     buffer outside the call's slice untouched; prefill again after every input and the pool are refilled in place,
     decode again after the captured graph is replayed with new slot indices, accepted counts and data;
  2. prefill contract: head counts 1..128, ragged / one-token / partial-chunk batches, 256 sequences, a 64K-token
     sequence, > 2^31-element tensors, meta=None, no host synchronization inside prefill(), prefill_prepare latency;
  3. numerics vs an fp64 recurrence: slow / saturated / spectrum gates, trained-layer statistics, aligned keys,
     rescaled inputs;
  4. decode contract: held-out shapes (H 8..96, S 1..8, B 1..512), contiguous and page layouts, graph replay with new
     indices;
  5. integration: prefill -> spec decode -> prefill on one pool, prefix hit from a checkpoint row;
  6. co-tenant: prefill and decode finish (and are correct) while another stream's kernel holds 3/4 of the CUs;
  7. no regression on the phase-1 workloads: the 10 amd_kda_mi350x prefill shapes and the 17 amd_kda_decode_mi350x
     shapes (their metrics, recorded baselines) must stay at or above the floors printed below.

Usage (from the repository root, one idle MI350X selected with HIP_VISIBLE_DEVICES; see benchmark/README.md):
    HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py                    # official: score + every gate
    HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --quick            # serving workloads only, fewer iterations
    HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --part prefill --workloads p12_loaded16k_1
    HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --gates contract,numerics   # gates only (see --help)
    HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --kernel path/to/other/kernel.py

Locations (defaults relative to this repository; override with the environment):
    KDA_REPO_ROOT       repository root: reference.py lives here, default kernel = $KDA_REPO_ROOT/kda/kernel.py
    KDA_BASELINES_DIR   vLLM / AITER baselines (sota_prod.py), default benchmark/baselines
    KDA_WORKLOADS_DIR   recorded phase-1 baselines (legacy_*_baseline_sota.json), default workloads/
    KDA_BENCH_CACHE     generated files (phase-1 reference outputs, co-tenant hog build), default benchmark/.cache
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import time
import zlib

import torch

BENCH_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KDA_REPO_ROOT", BENCH_DIR.parent)).resolve()
BASELINES = Path(os.environ.get("KDA_BASELINES_DIR", BENCH_DIR / "baselines")).resolve()
DATA = Path(os.environ.get("KDA_WORKLOADS_DIR", ROOT / "workloads")).resolve()
CACHE = Path(os.environ.get("KDA_BENCH_CACHE", BENCH_DIR / ".cache")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASELINES))

from reference import kda_decode_reference, kda_prefill_reference  # noqa: E402

D, W = 128, 4
LB = -5.0
SCALE = D ** -0.5
NORM_EPS = 1e-5
P_RTOL = 2e-2      # prefill: relative L2 error per sequence (output) and per written pool row
D_OUT_RTOL = 2e-2  # decode: relative L2 error of the bf16 output
D_ST_RTOL = 1e-2   # decode: relative L2 error of every written fp32 state slot
SPEC_S = 8         # Kimi-K3 DSpark: 7 drafts
SPEC_AL = 4.38     # DSpark accepted length on SPEED-Bench coding, 7 drafts (Inferact/Kimi-K3-DSpark model card)
PAGE_PAD = 4096    # bytes of padding per page (the hybrid allocator rounds pages up to the attention page size)
DEC_TOKENS_FIRST = 256  # decode tokens precede the prefill tokens in the layer's output buffer

# Floors of the phase-1 metrics (95 % of the starting package on this machine, measured by the operator).
LEGACY_PREFILL_FLOOR = float(os.environ.get("KDA_LEGACY_PREFILL_FLOOR", "0.932"))
LEGACY_DECODE_FLOOR = float(os.environ.get("KDA_LEGACY_DECODE_FLOOR", "1.500"))

# (H, [(new tokens, has_initial_state, checkpoint offset)]) per chunked-prefill step; gen_prefill_batches.py --seed 1.
PREFILL_WORKLOADS = {
    "p12_loaded8k_0": (12, [(7808, 1, 7808), (128, 1, 0)]),
    "p12_loaded8k_1": (12, [(3776, 1, 3584), (1280, 1, 896), (448, 1, 384), (2688, 1, 2688)]),
    "p12_loaded8k_2": (12, [(7936, 1, 7232)]),
    "p12_loaded16k_0": (12, [(5440, 1, 4992), (1920, 1, 1024), (8768, 0, 8192)]),
    "p12_loaded16k_1": (12, [(3008, 1, 3008), (1536, 1, 1536), (256, 1, 64), (128, 0, 0), (1600, 1, 832), (640, 1, 0),
                             (576, 1, 0), (512, 1, 0), (448, 1, 0), (2560, 1, 2560), (320, 1, 256), (4288, 1, 3712)]),
    "p12_loaded16k_2": (12, [(384, 1, 64), (320, 1, 128), (256, 1, 0), (7424, 1, 7104), (192, 1, 0), (256, 1, 128),
                             (1344, 1, 1280), (1280, 1, 320), (704, 1, 128), (2944, 0, 2048), (1280, 0, 1024)]),
    "p12_light8k_0": (12, [(5248, 1, 4608), (2944, 0, 2048)]),
    "p12_light8k_1": (12, [(640, 1, 0), (704, 1, 0), (960, 1, 960)]),
    "p12_light16k_0": (12, [(1600, 1, 640)]),
    "p12_light16k_1": (12, [(1664, 1, 768)]),
    "p96_loaded8k_0": (96, [(8192, 1, 7616)]),
    "p96_loaded8k_1": (96, [(384, 1, 0), (1344, 1, 896), (3008, 0, 2048), (2176, 1, 2048), (320, 1, 0), (832, 0, 0)]),
    "p96_light8k_0": (96, [(704, 1, 320), (1216, 1, 960), (512, 1, 320)]),
}
# name: (H, B, S)
DECODE_WORKLOADS = {
    "s8_h12_b1": (12, 1, 8), "s8_h12_b4": (12, 4, 8), "s8_h12_b16": (12, 16, 8), "s8_h12_b32": (12, 32, 8),
    "s8_h12_b64": (12, 64, 8), "s8_h12_b128": (12, 128, 8), "d_h12_b64": (12, 64, 1),
    "s8_h96_b1": (96, 1, 8), "s8_h96_b8": (96, 8, 8), "s8_h96_b32": (96, 32, 8),
}
QUICK_PREFILL = ["p12_loaded8k_1", "p12_loaded16k_1", "p12_light16k_0", "p96_loaded8k_1"]
QUICK_DECODE = ["s8_h12_b4", "s8_h12_b64", "d_h12_b64", "s8_h96_b8"]

# Phase-1 workloads (no-regression gate).
LEGACY_PREFILL = {"p_1x1k": [1024], "p_1x4k": [4096], "p_1x16k": [16384], "p_1x64k": [65536], "p_4x4k": [4096] * 4,
                  "p_8x2k": [2048] * 8, "p_16x1k": [1024] * 16, "p_32x512": [512] * 32, "p_64x128": [128] * 64,
                  "p_mixed": [8192, 1024, 777, 512, 256, 130, 64, 17]}
LEGACY_DECODE = {
    "d_k3_b1": (12, 1, 1), "d_k3_b32": (12, 32, 1), "d_k3_b128": (12, 128, 1),
    "s4_k3_b1": (12, 1, 4), "s4_k3_b16": (12, 16, 4), "s4_k3_b64": (12, 64, 4),
    "s8_k3_b1": (12, 1, 8), "s8_k3_b16": (12, 16, 8), "s8_k3_b64": (12, 64, 8),
    "d_glm16_b1": (16, 1, 1), "d_glm16_b64": (16, 64, 1), "d_glm16_b256": (16, 256, 1),
    "s6_glm16_b1": (16, 1, 6), "s6_glm16_b16": (16, 16, 6), "s6_glm16_b64": (16, 64, 6),
    "d_glm8_b128": (8, 128, 1), "s6_glm8_b32": (8, 32, 6),
}


# ================================================================================================ helpers
def load_kernel(path):
    spec = importlib.util.spec_from_file_location("kda_prod_cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("prefill_prepare", "prefill", "decode"):
        if not hasattr(mod, name):
            raise AttributeError(f"{path} does not define {name}()")
    return mod


_flush_buf = None


def flush():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    _flush_buf.zero_()


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 0.0


def timed_eager(fn):
    flush()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0  # us


def timed_graph(g):
    flush()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0


class Pages:
    """vLLM hybrid-allocator page layout: one page per slot holding the conv state ([taps, 3*H*128] bf16) followed by
    the SSM state ([H, 128, 128] fp32), pages padded; conv_state / ssm_state are strided views of one raw buffer."""

    def __init__(self, slots, H, taps, gen, contiguous=False):
        dim = H * D
        self.conv_bytes = taps * 3 * dim * 2
        ssm_bytes = H * D * D * 4
        if contiguous:  # separate dense pools (the phase-1 tasks' layout)
            self.raw = None
            self.conv = (torch.randn(slots, taps, 3 * dim, generator=gen, device="cuda")).bfloat16()
            self.ssm = torch.randn(slots, H, D, D, generator=gen, device="cuda") * 0.05
            return
        page = -(-(self.conv_bytes + ssm_bytes + PAGE_PAD) // 256) * 256
        self.raw = torch.randint(0, 256, (slots * page,), dtype=torch.uint8, generator=gen, device="cuda")
        self.conv = torch.as_strided(self.raw.view(torch.bfloat16), (slots, taps, 3 * dim), (page // 2, 3 * dim, 1), 0)
        self.ssm = torch.as_strided(self.raw.view(torch.float32), (slots, H, D, D), (page // 4, D * D, D, 1),
                                    self.conv_bytes // 4)
        self.conv.copy_(torch.randn(self.conv.shape, generator=gen, device="cuda").bfloat16())
        self.ssm.copy_(torch.randn(self.ssm.shape, generator=gen, device="cuda") * 0.05)

    def snapshot(self):
        return (self.raw.clone(),) if self.raw is not None else (self.conv.clone(), self.ssm.clone())

    def restore(self, snap):
        if self.raw is not None:
            self.raw.copy_(snap[0])
        else:
            self.conv.copy_(snap[0])
            self.ssm.copy_(snap[1])

    def clone_layout(self):
        """Independent copy with the same layout (for the baseline)."""
        p = Pages.__new__(Pages)
        p.conv_bytes = self.conv_bytes
        if self.raw is None:
            p.raw, p.conv, p.ssm = None, self.conv.clone(), self.ssm.clone()
        else:
            p.raw = self.raw.clone()
            p.conv = torch.as_strided(p.raw.view(torch.bfloat16), self.conv.shape, self.conv.stride(), 0)
            p.ssm = torch.as_strided(p.raw.view(torch.float32), self.ssm.shape, self.ssm.stride(),
                                     self.ssm.storage_offset())
        return p

    def check(self, snap, written, ref_ssm, ref_conv, rtol):
        """Written SSM rows within rtol of ref_ssm, conv equal to ref_conv (if given), everything else bit-identical.
        Returns (ok, worst written-row error, message)."""
        worst, bad = 0.0, []
        for r in written:
            if not torch.isfinite(self.ssm[r]).all():
                bad.append(f"row {r} non-finite")
                continue
            e = rel(self.ssm[r], ref_ssm[r])
            worst = max(worst, e)
            if e > rtol:
                bad.append(f"row {r} err {e:.1e}")
        if ref_conv is not None and not torch.equal(self.conv, ref_conv):
            bad.append("CONV-CACHE-MISMATCH")
        exp = Pages.__new__(Pages)
        exp.raw = None
        if self.raw is not None:
            e_raw = snap[0].clone()
            e_ssm = torch.as_strided(e_raw.view(torch.float32), self.ssm.shape, self.ssm.stride(),
                                     self.ssm.storage_offset())
            e_conv = torch.as_strided(e_raw.view(torch.bfloat16), self.conv.shape, self.conv.stride(), 0)
            if ref_conv is not None:
                e_conv.copy_(self.conv)
            for r in written:
                e_ssm[r] = self.ssm[r]
            if not torch.equal(self.raw, e_raw):
                bad.append("BYTES-OUTSIDE-WRITTEN-ROWS-MODIFIED")
        else:
            e_ssm = snap[1].clone()
            for r in written:
                e_ssm[r] = self.ssm[r]
            if not torch.equal(self.ssm, e_ssm):
                bad.append("UNWRITTEN-SLOT-MODIFIED")
            if ref_conv is None and not torch.equal(self.conv, snap[0]):
                bad.append("CONV-CACHE-MODIFIED")
        return not bad, worst, "; ".join(bad[:4])


def proj(T, H, gen):
    """Input projection [T, Wp] as vLLM's in_proj_qkvgfab: mixed_qkv | out_gate | f_a | beta | pad."""
    dim = H * D
    Wp = -(-(4 * dim + D + H) // 16) * 16
    P = torch.randn(T, Wp, generator=gen, device="cuda").bfloat16()
    return P, P[:, :3 * dim], P[:, 3 * dim:4 * dim], P[:, 4 * dim + D:4 * dim + D + H]


# ================================================================================================ prefill
class PrefillCase:
    def __init__(self, name, H, seqs, seed, pages=True, gates=None, dtype_ref=torch.float32):
        self.name, self.H, self.seqs, self.seed = name, H, seqs, seed
        self.gates = gates or {}
        self.dtype_ref = dtype_ref
        N = len(seqs)
        lens = [s[0] for s in seqs]
        self.N, self.T = N, sum(lens)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gen = gen
        slots = 2 * N + 9
        self.pages = Pages(slots, H, 2 + SPEC_S, gen, contiguous=not pages)
        perm = (torch.randperm(slots - 1, generator=gen, device="cuda") + 1).int()
        self.state_indices = perm[:N].contiguous()
        ck_rows = [int(perm[N + n]) if s[2] > 0 else -1 for n, s in enumerate(seqs)]
        self.has_init = torch.tensor([bool(s[1]) for s in seqs], device="cuda")
        cu = [0]
        for x in lens:
            cu.append(cu[-1] + x)
        self.cu_cpu = torch.tensor(cu, dtype=torch.int32)
        self.cu = self.cu_cpu.cuda()
        self.ck_off_cpu = torch.tensor([s[2] for s in seqs], dtype=torch.int32)
        self.ck_rows_cpu = torch.tensor(ck_rows, dtype=torch.int32)
        self.ck_off, self.ck_rows = self.ck_off_cpu.cuda(), self.ck_rows_cpu.cuda()
        self.written = sorted(set(self.state_indices.tolist()) | {r for r in ck_rows if r >= 0})
        T = self.T
        self.q = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        self.k, self.v, self.raw_g = torch.empty_like(self.q), torch.empty_like(self.q), torch.empty_like(self.q)
        self.P, _, _, beta = proj(T, H, gen)
        self.raw_beta = beta.unsqueeze(0)
        self.A_log = torch.empty(H, device="cuda")
        self.dt_bias = torch.empty(H * D, device="cuda")
        self.big_out = torch.empty(1, DEC_TOKENS_FIRST + T, H, D, dtype=torch.bfloat16, device="cuda")
        self.out = self.big_out[:, DEC_TOKENS_FIRST:]
        self.refill(seed)

    def refill(self, seed):
        """New values for every input and the pool, in place (shapes, layouts and metadata unchanged)."""
        g = torch.Generator(device="cuda").manual_seed(seed + 1)
        for t in (self.q, self.k, self.v, self.raw_g):
            t.copy_(torch.randn(t.shape, generator=g, device="cuda"))
        self.P.copy_(torch.randn(self.P.shape, generator=g, device="cuda"))
        self.A_log.copy_(torch.rand(self.H, generator=g, device="cuda") * 2 - 1)
        self.dt_bias.copy_(torch.randn(self.H * D, generator=g, device="cuda") * 0.5)
        self.pages.conv.copy_(torch.randn(self.pages.conv.shape, generator=g, device="cuda").bfloat16())
        self.pages.ssm.copy_(torch.randn(self.pages.ssm.shape, generator=g, device="cuda") * 0.05)
        gates = self.gates
        if "q_scale" in gates:
            self.q.mul_(gates["q_scale"])
        if "k_scale" in gates:
            self.k.mul_(gates["k_scale"])
        if "v_scale" in gates:
            self.v.mul_(gates["v_scale"])
        if "a_log" in gates:
            self.A_log.fill_(gates["a_log"])
        if "dt_range" in gates:
            lo, hi = gates["dt_range"]
            self.dt_bias.copy_(torch.linspace(lo, hi, D, device="cuda").repeat(self.H))
        if "beta_logit" in gates:
            self.raw_beta.fill_(gates["beta_logit"])
        if "aligned" in gates:  # q / k / v of a 64-token chunk share a direction (trained deep layers)
            T, H = self.T, self.H
            nc = -(-T // 64)
            for t in (self.q, self.k, self.v):
                base = torch.randn(nc, 1, H, D, generator=g, device="cuda")
                noise = torch.randn(nc, 64, H, D, generator=g, device="cuda") * gates["aligned"]
                t.copy_((base + noise).reshape(nc * 64, H, D)[:T].unsqueeze(0))
        if "state_scale" in gates:
            self.pages.ssm.mul_(gates["state_scale"] / 0.05)
        if "one_hot_k" in gates:
            idx = torch.randint(0, 4, (self.T, self.H), generator=g, device="cuda")
            self.k.copy_(torch.nn.functional.one_hot(idx, D).unsqueeze(0).to(self.k.dtype))
        self.big_out.fill_(float("nan"))
        for n, s in enumerate(self.seqs):  # slots of cold sequences and checkpoint rows are write-only: poison them
            if not s[1]:
                self.pages.ssm[int(self.state_indices[n])] = float("nan")
            if int(self.ck_rows_cpu[n]) >= 0:
                self.pages.ssm[int(self.ck_rows_cpu[n])] = float("nan")
        torch.cuda.synchronize()
        self.snap = self.pages.snapshot()
        self.ref = None

    def args(self, pages=None):
        p = pages or self.pages
        return (self.q, self.k, self.v, self.raw_g, self.raw_beta, self.A_log, self.dt_bias, SCALE, LB, p.ssm,
                self.state_indices, self.has_init, self.cu, self.out)

    def reference(self):
        if self.ref is None:  # on a copy: the candidate's writes in self.pages must survive for the check
            p = self.pages.clone_layout()
            p.restore(self.snap)
            o, s = kda_prefill_reference(self.q, self.k, self.v, self.raw_g, self.raw_beta, self.A_log, self.dt_bias,
                                         LB, SCALE, p.ssm, self.state_indices, self.has_init, self.cu,
                                         self.ck_off, self.ck_rows, dtype=self.dtype_ref)
            self.ref = (o, s)
        return self.ref

    def ck_kwargs(self):
        return dict(checkpoint_offsets=self.ck_off, checkpoint_state_indices=self.ck_rows)

    def verify(self, rtol=P_RTOL):
        o_ref, s_ref = self.reference()
        o = self.out
        bad, worst_o = [], 0.0
        if not torch.isfinite(o).all():
            bad.append("non-finite output")
        cu = self.cu_cpu.tolist()
        for n in range(self.N):
            e = rel(o[0, cu[n]:cu[n + 1]], o_ref[0, cu[n]:cu[n + 1]])
            worst_o = max(worst_o, e)
            if e > rtol:
                bad.append(f"seq {n} (len {cu[n + 1] - cu[n]}) out err {e:.1e}")
        head = self.big_out[:, :DEC_TOKENS_FIRST]
        if not torch.isnan(head.float()).all():
            bad.append("wrote outside its output slice")
        ok, worst_s, msg = self.pages.check(self.snap, self.written, s_ref, None, rtol)
        if not ok:
            bad.append(msg)
        return not bad, worst_o, worst_s, "; ".join(bad[:4])


def prefill_case_for(name):
    H, seqs = PREFILL_WORKLOADS[name]
    return H, seqs


def run_prefill_set(mod, names, seed, iters, trials, refill_check=True, log=print, baseline=True):
    import sota_prod
    rows, ratios, all_ok = [], [], True
    log(f"\n{'prefill workload':<17} {'H':>3} {'N':>3} {'T':>6} | {'vLLM us':>9} {'cand us':>9} {'speedup':>8} | correctness")
    for i, name in enumerate(names):
        H, seqs = PREFILL_WORKLOADS[name]
        s0 = seed * 1000 + zlib.crc32(name.encode()) % 997
        try:
            case = PrefillCase(name, H, seqs, s0)
            meta = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
            call = lambda: mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
            case.pages.restore(case.snap)
            call()
            torch.cuda.synchronize()
            ok, eo, es, msg = case.verify()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:300]}"
        if not ok:
            all_ok = False
            log(f"{name:<17} {H:3d} {len(seqs):3d} {sum(s[0] for s in seqs):6d} | {'':9} {'':9} {'':8} | FAIL ({msg})")
            continue
        bpages = case.pages.clone_layout()
        bmeta = sota_prod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
        bout = torch.empty_like(case.out)
        bargs = list(case.args(bpages))
        bargs[-1] = bout
        bcall = lambda: sota_prod.prefill(*bargs, **case.ck_kwargs(), meta=bmeta)
        tr = []
        for _ in range(trials):
            ct, bt = [], []
            for _ in range(2):
                call()
                bcall()
            for _ in range(iters):
                ct.append(timed_eager(call))
                bt.append(timed_eager(bcall))
            tr.append((statistics.median(bt), statistics.median(ct)))
        b_us = statistics.median(x[0] for x in tr)
        c_us = statistics.median(x[1] for x in tr)
        r = statistics.median(x[0] / x[1] for x in tr)
        detail = f"out {eo:.1e} pool {es:.1e}"
        if refill_check:
            case.refill(s0 + 77)
            meta2 = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
            case.pages.restore(case.snap)
            mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta2)
            torch.cuda.synchronize()
            ok2, eo2, es2, msg2 = case.verify()
            if not ok2:
                all_ok = False
                log(f"{name:<17} {H:3d} {case.N:3d} {case.T:6d} | {b_us:9.1f} {c_us:9.1f} {r:7.3f}x | "
                    f"FAIL after in-place refill ({msg2})")
                continue
            detail += f", after refill out {eo2:.1e} pool {es2:.1e}"
        ratios.append(r)
        rows.append((name, b_us, c_us, r))
        log(f"{name:<17} {H:3d} {case.N:3d} {case.T:6d} | {b_us:9.1f} {c_us:9.1f} {r:7.3f}x | PASS ({detail})")
        del case, bpages, bout, bargs
        torch.cuda.empty_cache()
    return gm(ratios), all_ok and len(ratios) == len(names), rows


# ================================================================================================ decode
class DecodeCase:
    def __init__(self, name, H, B, S, seed, pages=True, acc="dspark"):
        self.name, self.H, self.B, self.S = name, H, B, S
        dim, T = H * D, B * S
        self.T = T
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gen = gen
        taps = W - 1 if S == 1 else W - 2 + S
        slots = 2 * B * S + 16
        self.slots = slots
        self.pages = Pages(slots, H, taps, gen, contiguous=not pages)
        if pages:
            self.P, self.mixed_qkv, self.out_gate, self.beta = proj(T, H, gen)
        else:
            self.P = None
            self.mixed_qkv = torch.randn(T, 3 * dim, generator=gen, device="cuda").bfloat16()
            self.out_gate = torch.randn(T, dim, generator=gen, device="cuda").bfloat16()
            self.beta = torch.randn(T, H, generator=gen, device="cuda").bfloat16()
        self.gate = torch.randn(T, H, D, generator=gen, device="cuda").bfloat16()
        self.conv_weight = torch.randn(3, W, dim, generator=gen, device="cuda") * 0.4
        self.A_log = torch.rand(H, generator=gen, device="cuda") * 2 - 1
        self.dt_bias = torch.randn(dim, generator=gen, device="cuda") * 0.5
        self.norm_weight = 1 + torch.randn(D, generator=gen, device="cuda") * 0.1
        self.acc_mode = acc
        self.state_indices = torch.zeros((B,) if S == 1 else (B, S), dtype=torch.int32, device="cuda")
        self.nat = None if S == 1 else torch.ones(B, dtype=torch.int32, device="cuda")
        self.cu = torch.arange(0, T + 1, S, dtype=torch.long, device="cuda")
        self.out = torch.empty(T, dim, dtype=torch.bfloat16, device="cuda")
        self.new_indices(seed)

    def new_indices(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed + 3)
        B, S = self.B, self.S
        perm = (torch.randperm(self.slots - 1, generator=g, device="cuda")[: B * S] + 1).int()
        self.state_indices.copy_(perm.view(self.state_indices.shape))
        if S > 1:
            if self.acc_mode == "dspark" and S == SPEC_S:
                lo = int(SPEC_AL)
                hi = torch.rand(B, generator=g, device="cuda") < (SPEC_AL - lo)
                self.nat.copy_((lo + hi.int()).int())
            else:
                self.nat.copy_(torch.randint(1, S + 1, (B,), generator=g, device="cuda", dtype=torch.int32))
        torch.cuda.synchronize()
        self.snap = self.pages.snapshot()
        si = self.state_indices
        written = si.flatten() if S > 1 else si
        self.written = sorted(int(x) for x in written.tolist() if x > 0)
        self.ref = None

    def new_data(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed + 5)
        if self.P is not None:
            self.P.copy_(torch.randn(self.P.shape, generator=g, device="cuda"))
        else:
            for t in (self.mixed_qkv, self.out_gate, self.beta):
                t.copy_(torch.randn(t.shape, generator=g, device="cuda"))
        self.gate.copy_(torch.randn(self.gate.shape, generator=g, device="cuda"))
        self.pages.conv.copy_(torch.randn(self.pages.conv.shape, generator=g, device="cuda").bfloat16())
        self.pages.ssm.copy_(torch.randn(self.pages.ssm.shape, generator=g, device="cuda") * 0.05)
        self.new_indices(seed)

    def kwargs(self, pages=None, out=None):
        p = pages or self.pages
        return dict(mixed_qkv=self.mixed_qkv, conv_state=p.conv, conv_weight=self.conv_weight, gate=self.gate,
                    beta=self.beta, out_gate=self.out_gate, A_log=self.A_log, dt_bias=self.dt_bias,
                    norm_weight=self.norm_weight, ssm_state=p.ssm, state_indices=self.state_indices,
                    num_accepted_tokens=self.nat, cu_seqlens=self.cu, norm_eps=NORM_EPS, lower_bound=LB,
                    out=self.out if out is None else out)

    def reference(self):
        if self.ref is None:  # on a copy: the candidate's writes in self.pages must survive for the check
            p = self.pages.clone_layout()
            p.restore(self.snap)
            kw = self.kwargs(p)
            kw.pop("out")
            self.ref = kda_decode_reference(**kw)
        return self.ref

    def capture(self, fn, pages=None, out=None):
        kw = self.kwargs(pages, out)
        call = lambda: fn(**kw)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        torch.cuda.synchronize()
        return g

    def check(self, run):
        """run(): one call / replay on the current inputs. Pool restored first; returns (ok, message)."""
        o_ref, conv_ref, ssm_ref = self.reference()
        self.pages.restore(self.snap)
        self.out.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        if not torch.isfinite(self.out.float()).all():
            return False, "non-finite output"
        oe = rel(self.out, o_ref)
        ok, worst, msg = self.pages.check(self.snap, self.written, ssm_ref, conv_ref, D_ST_RTOL)
        ok = ok and oe <= D_OUT_RTOL
        return ok, f"out {oe:.1e} state {worst:.1e}" + (f" {msg}" if msg else "")


def run_decode_set(mod, names, seed, iters, trials, log=print, pool=None, replay_check=True, timed=True):
    import sota_prod
    pool = pool or DECODE_WORKLOADS
    rows, ratios, all_ok = [], [], True
    if timed:
        log(f"\n{'decode workload':<17} {'H':>3} {'B':>3} {'S':>2} | {'base us':>9} {'cand us':>9} {'speedup':>8} | correctness")
    for name in names:
        H, B, S = pool[name]
        s0 = seed * 1000 + zlib.crc32(name.encode()) % 997
        try:
            case = DecodeCase(name, H, B, S, s0)
            g = case.capture(mod.decode)
            ok, msg = case.check(g.replay)
            if ok and replay_check:
                case.new_data(s0 + 11)
                ok, msg2 = case.check(g.replay)
                msg += f", replay with new indices/data: {msg2}"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:300]}"
        if not ok:
            all_ok = False
            log(f"{name:<17} {H:3d} {B:3d} {S:2d} | {'':9} {'':9} {'':8} | FAIL ({msg})")
            continue
        if not timed:
            log(f"{name:<17} {H:3d} {B:3d} {S:2d} | PASS ({msg})")
            ratios.append(1.0)
            continue
        bgraphs = {}
        for which, fn in sota_prod.decode_baselines(H, S).items():
            bp = case.pages.clone_layout()
            bo = torch.empty_like(case.out)
            bgraphs[which] = (case.capture(fn, bp, bo), bp)
        tr = []
        for _ in range(trials):
            ct, bt = [], {k: [] for k in bgraphs}
            for _ in range(iters):
                ct.append(timed_graph(g))
                for k, (bg, _) in bgraphs.items():
                    bt[k].append(timed_graph(bg))
            bmed = min(statistics.median(v) for v in bt.values())
            tr.append((bmed, statistics.median(ct)))
        b_us = statistics.median(x[0] for x in tr)
        c_us = statistics.median(x[1] for x in tr)
        r = statistics.median(x[0] / x[1] for x in tr)
        ratios.append(r)
        rows.append((name, b_us, c_us, r))
        log(f"{name:<17} {H:3d} {B:3d} {S:2d} | {b_us:9.2f} {c_us:9.2f} {r:7.3f}x | PASS ({msg})")
        del case, bgraphs
        torch.cuda.empty_cache()
    return gm(ratios), all_ok and len(ratios) == len(names), rows


# ================================================================================================ gates
def gate_prefill_contract(mod, seed, log, heavy=True):
    """Contract cases (fp32 reference): head counts, ragged batches, edge checkpoints, big batches, meta=None."""
    rng = random.Random(seed)
    cases = []
    mix = [(777, 1, 704), (64, 0, 64), (1, 1, 0), (130, 0, 0), (1024, 1, 1024), (17, 1, 0), (255, 0, 192)]
    for H in (1, 4, 8, 16, 24, 32, 48, 128):
        cases.append((f"heads_h{H}", H, mix, {}))
    ragged = []
    for _ in range(64):
        n = rng.randint(1, 300)
        ck = 64 * rng.randint(1, n // 64) if n >= 64 and rng.random() < 0.6 else 0
        ragged.append((n, int(rng.random() < 0.8), ck))
    cases.append(("ragged64_h12", 12, ragged, {}))
    cases.append(("many256x64_h12", 12, [(64, int(rng.random() < 0.8), 64 * (rng.random() < 0.3)) for _ in range(256)], {}))
    cases.append(("tiny_h12", 12, [(1, 0, 0)], {}))
    cases.append(("ck_at_64_h12", 12, [(4096, 1, 64), (65, 0, 64), (128, 1, 128)], {}))
    cases.append(("meta_none_h12", 12, [(3000, 1, 2048), (100, 0, 64), (5000, 1, 0)], {"meta_none": True}))
    cases.append(("contiguous_pool_h12", 12, [(2000, 1, 1024), (300, 0, 256)], {"contiguous": True}))
    if heavy:
        cases.append(("long64k_h12", 12, [(65536, 1, 32768)], {}))
        cases.append(("int64_extent_h96", 96, [(1024, int(rng.random() < 0.8), 1024 * (rng.random() < 0.3)) for _ in range(176)], {}))
    bad = 0
    log(f"\n[gate] prefill contract ({len(cases)} cases, fp32 reference)")
    for i, (name, H, seqs, flags) in enumerate(cases):
        try:
            case = PrefillCase(name, H, seqs, seed * 7919 + i, pages=not flags.get("contiguous"))
            case.pages.restore(case.snap)
            if flags.get("meta_none"):
                mod.prefill(*case.args(), **case.ck_kwargs(), meta=None)
            else:
                meta = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
                mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
            torch.cuda.synchronize()
            ok, eo, es, msg = case.verify()
            detail = f"out {eo:.1e} pool {es:.1e}" + (f" {msg}" if msg else "")
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:240]}"
        bad += not ok
        log(f"  {name:<22} H={H:<3} N={len(seqs):<4} T={sum(s[0] for s in seqs):<7} {'PASS' if ok else 'FAIL'} ({detail})")
        case = None
        torch.cuda.empty_cache()
    return bad


def gate_prefill_host(mod, seed, log):
    """prefill() must not synchronize with the device (it runs eagerly, 69 layers per step); prefill_prepare runs
    once per step on a new layout and must stay within 1 ms + 5 us per sequence."""
    bad = 0
    log("\n[gate] prefill host behaviour")
    for name in ("p12_loaded16k_1", "p96_loaded8k_1"):
        H, seqs = PREFILL_WORKLOADS[name]
        case = PrefillCase(name, H, seqs, seed + 101)
        meta = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
        mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
        torch.cuda.synchronize()
        try:
            torch.cuda.set_sync_debug_mode("error")
            mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
            sync_ok, smsg = True, "no host sync"
        except RuntimeError as e:
            sync_ok, smsg = False, f"host sync inside prefill(): {str(e)[:160]}"
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        times = []
        for j in range(5):  # fresh layouts: jitter the lengths (same total, same N)
            lens = [s[0] for s in seqs]
            if len(lens) > 1:
                a, b = j % len(lens), (j + 1) % len(lens)
                d = min(64, lens[a] - 1)
                lens[a] -= d
                lens[b] += d
            cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
            offs = torch.tensor([(x // 1024) * 1024 if x >= 1024 else 0 for x in lens], dtype=torch.int32)
            rows = case.ck_rows_cpu.clone()
            rows[offs == 0] = -1
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mod.prefill_prepare(cu, H, offs, rows)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        limit = 1.0 + 0.005 * len(seqs)
        t_ok = statistics.median(times) <= limit
        ok = sync_ok and t_ok
        bad += not ok
        log(f"  {name:<18} {'PASS' if ok else 'FAIL'} ({smsg}; prepare on new layouts median {statistics.median(times):.3f} ms, "
            f"limit {limit:.3f} ms)")
    return bad


NUMERIC_PROBES = [
    # name, H, seqs, gate settings
    ("slow_decay_16k", 12, [(16384, 1, 8192)], {"a_log": 0.0, "dt_range": (-12.0, -6.0)}),
    ("spectrum_4x4k", 12, [(4096, 1, 2048)] * 4, {"a_log": 0.0, "dt_range": (-14.0, 2.0)}),
    ("saturated", 12, [(2048, 1, 1024), (777, 0, 0)], {"a_log": 2.0, "dt_range": (20.0, 30.0)}),
    ("deep_layer", 12, [(4096, 1, 3072), (1500, 1, 1024)], {"a_log": 0.0, "dt_range": (-12.0, -4.0), "beta_logit": 6.0,
                                                            "aligned": 0.3, "state_scale": 0.5}),
    ("deep_layer_h96", 96, [(2048, 1, 1024), (300, 0, 256)], {"a_log": 0.0, "dt_range": (-12.0, -4.0),
                                                              "beta_logit": 6.0, "aligned": 0.3, "state_scale": 0.5}),
    ("one_hot_keys", 12, [(1024, 1, 512), (200, 0, 0)], {"one_hot_k": True, "beta_logit": 4.0}),
    ("rescaled", 12, [(1024, 1, 1024), (513, 1, 512)], {"q_scale": 1e-3, "k_scale": 1e3, "v_scale": 1e3}),
]


def gate_numerics(mod, seed, log):
    import sota_prod
    bad = 0
    log("\n[gate] numerics vs fp64 recurrence (per-sequence relative L2 <= 2e-2; vLLM's error shown for reference)")
    for i, (name, H, seqs, gates) in enumerate(NUMERIC_PROBES):
        try:
            case = PrefillCase(name, H, seqs, seed * 31 + i, gates=gates, dtype_ref=torch.float64)
            case.pages.restore(case.snap)
            meta = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
            mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
            torch.cuda.synchronize()
            ok, eo, es, msg = case.verify()
            bp = case.pages.clone_layout()
            bp.restore(case.snap)
            bargs = list(case.args(bp))
            bout = torch.full_like(case.out, float("nan"))
            bargs[-1] = bout
            sota_prod.prefill(*bargs, **case.ck_kwargs())
            torch.cuda.synchronize()
            o_ref, s_ref = case.reference()
            beo = max(rel(bout[0, a:b], o_ref[0, a:b]) for a, b in zip(case.cu_cpu[:-1].tolist(), case.cu_cpu[1:].tolist()))
            bes = max(rel(bp.ssm[r], s_ref[r]) for r in case.written)
            detail = f"out {eo:.1e} pool {es:.1e} | vLLM out {beo:.1e} pool {bes:.1e}" + (f" {msg}" if msg else "")
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:240]}"
        bad += not ok
        log(f"  {name:<16} {'PASS' if ok else 'FAIL'} ({detail})")
        case = None
        torch.cuda.empty_cache()
    return bad


DECODE_HELDOUT = {
    "h_d_h12_b8": (12, 8, 1), "h_d_h12_b200": (12, 200, 1), "h_s2_h12_b8": (12, 8, 2), "h_s4_h12_b40": (12, 40, 4),
    "h_s8_h12_b100": (12, 100, 8), "h_s8_h12_b256": (12, 256, 8), "h_d_h24_b16": (24, 16, 1), "h_s5_h24_b33": (24, 33, 5),
    "h_s6_h16_b4": (16, 4, 6), "h_s6_h16_b48": (16, 48, 6), "h_d_h16_b512": (16, 512, 1), "h_s3_h8_b96": (8, 96, 3),
    "h_d_h8_b7": (8, 7, 1), "h_s8_h48_b12": (48, 12, 8), "h_d_h96_b64": (96, 64, 1), "h_s3_h96_b5": (96, 5, 3),
    "h_s7_h12_b1": (12, 1, 7),
}


def gate_decode_contract(mod, seed, log):
    log(f"\n[gate] decode contract: {len(DECODE_HELDOUT)} held-out shapes (page layout, uniform accepted counts, "
        "graph replay with new indices / data) + contiguous layout")
    bad = 0
    for i, (name, (H, B, S)) in enumerate(DECODE_HELDOUT.items()):
        for pages in (True, False) if i % 4 == 0 else (True,):
            try:
                case = DecodeCase(name, H, B, S, seed * 13 + i, pages=pages, acc="uniform")
                g = case.capture(mod.decode)
                ok, msg = case.check(g.replay)
                if ok:
                    case.new_data(seed * 13 + i + 1)
                    ok, msg2 = case.check(g.replay)
                    msg += f"; replay {msg2}"
                if ok:  # eager call too (vLLM's ROCm K3 layer runs the KDA op outside graph capture)
                    ok, msg3 = case.check(lambda: mod.decode(**case.kwargs()))
                    msg += f"; eager {msg3}"
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:240]}"
            bad += not ok
            log(f"  {name:<15} H={H:<3} B={B:<4} S={S} {'pages' if pages else 'dense'} {'PASS' if ok else 'FAIL'} ({msg})")
            case = None
        torch.cuda.empty_cache()
    return bad


def gate_integration(mod, seed, log):
    """One pool, one conv cache: prefill (cold, warm, checkpoint) -> 2 spec-verify steps -> next-turn prefill resuming
    from the accepted slot -> a new request resuming from a checkpoint row (prefix hit). Checked against the
    composition of the reference operators."""
    log("\n[gate] integration (prefill -> spec decode -> prefill on one pool)")
    H, S, B = 12, SPEC_S, 3
    g = torch.Generator(device="cuda").manual_seed(seed + 555)
    slots = 64
    pages = Pages(slots, H, 2 + S, g)
    ref_ssm = pages.ssm.clone()
    ref_conv = pages.conv.clone()
    dim = H * D
    A_log = torch.rand(H, generator=g, device="cuda") * 2 - 1
    dt_bias = torch.randn(dim, generator=g, device="cuda") * 0.5
    conv_w = torch.randn(3, W, dim, generator=g, device="cuda") * 0.4
    norm_w = 1 + torch.randn(D, generator=g, device="cuda") * 0.1
    spec_idx = torch.arange(1, 1 + B * S, dtype=torch.int32, device="cuda").view(B, S)
    ck_rows_all = torch.arange(40, 40 + B, dtype=torch.int32, device="cuda")
    msgs, ok_all = [], True

    def prefill_step(lens, has_init, slots_n, ck_off, ck_rows, tag):
        nonlocal ref_ssm, ok_all
        T = sum(lens)
        q, k, v, rg = (torch.randn(1, T, H, D, generator=g, device="cuda").bfloat16() for _ in range(4))
        P, _, _, beta = proj(T, H, g)
        rb = beta.unsqueeze(0)
        cu_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
        hi = torch.tensor(has_init, device="cuda")
        si = torch.tensor(slots_n, dtype=torch.int32, device="cuda")
        co, cr = torch.tensor(ck_off, dtype=torch.int32), torch.tensor(ck_rows, dtype=torch.int32)
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        meta = mod.prefill_prepare(cu_cpu, H, co, cr)
        mod.prefill(q, k, v, rg, rb, A_log, dt_bias, SCALE, LB, pages.ssm, si, hi, cu_cpu.cuda(), out,
                    checkpoint_offsets=co.cuda(), checkpoint_state_indices=cr.cuda(), meta=meta)
        o_ref, ref_ssm = kda_prefill_reference(q, k, v, rg, rb, A_log, dt_bias, LB, SCALE, ref_ssm, si, hi,
                                               cu_cpu.cuda(), co.cuda(), cr.cuda())
        torch.cuda.synchronize()
        e = max(rel(out[0, a:b], o_ref[0, a:b]) for a, b in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()))
        rows = set(slots_n) | {r for r in ck_rows if r >= 0}
        es = max(rel(pages.ssm[r], ref_ssm[r]) for r in rows)
        good = e <= P_RTOL and es <= P_RTOL
        ok_all &= good
        msgs.append(f"{tag} out {e:.1e} pool {es:.1e}{'' if good else ' FAIL'}")
        pages.ssm[list(rows)] = ref_ssm[list(rows)]  # continue from the reference state

    def decode_step(acc, tag):
        nonlocal ref_ssm, ref_conv, ok_all
        T = B * S
        P, mq, og, beta = proj(T, H, g)
        gate = torch.randn(T, H, D, generator=g, device="cuda").bfloat16()
        nat = torch.tensor(acc, dtype=torch.int32, device="cuda")
        cu = torch.arange(0, T + 1, S, dtype=torch.long, device="cuda")
        out = torch.empty(T, dim, dtype=torch.bfloat16, device="cuda")
        kw = dict(mixed_qkv=mq, conv_state=pages.conv, conv_weight=conv_w, gate=gate, beta=beta, out_gate=og,
                  A_log=A_log, dt_bias=dt_bias, norm_weight=norm_w, ssm_state=pages.ssm, state_indices=spec_idx,
                  num_accepted_tokens=nat, cu_seqlens=cu, norm_eps=NORM_EPS, lower_bound=LB)
        rkw = dict(kw, conv_state=ref_conv, ssm_state=ref_ssm)
        o_ref, ref_conv, ref_ssm = kda_decode_reference(**rkw)
        mod.decode(**kw, out=out)
        torch.cuda.synchronize()
        e = rel(out, o_ref)
        rows = [int(x) for x in spec_idx.flatten().tolist()]
        es = max(rel(pages.ssm[r], ref_ssm[r]) for r in rows)
        conv_ok = torch.equal(pages.conv, ref_conv)
        good = e <= D_OUT_RTOL and es <= D_ST_RTOL and conv_ok
        ok_all &= good
        msgs.append(f"{tag} out {e:.1e} state {es:.1e}{'' if conv_ok else ' CONV-MISMATCH'}{'' if good else ' FAIL'}")
        pages.ssm[rows] = ref_ssm[rows]

    try:
        heads = spec_idx[:, 0].tolist()
        prefill_step([1500, 64, 777], [False, True, True], heads, [1024, 64, 0], [40, 41, -1], "prefill-1")
        decode_step([1, 1, 1], "verify-1")
        decode_step([5, 3, 8], "verify-2")
        resume = [int(spec_idx[n, a - 1]) for n, a in enumerate([4, 1, 6])]
        decode_step([4, 1, 6], "verify-3")
        prefill_step([640, 1300, 90], [True, True, True], resume, [0, 1024, 64], [-1, 42, 43], "prefill-2")
        pages.ssm[50] = pages.ssm[40]  # prefix hit: the caller copies a checkpoint row into the new request's slot
        ref_ssm[50] = ref_ssm[40]
        prefill_step([2048], [True], [50], [2048], [44], "prefill-3 (prefix hit)")
    except Exception as e:  # noqa: BLE001
        ok_all = False
        msgs.append(f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:240]}")
    log(f"  {'PASS' if ok_all else 'FAIL'}: " + "; ".join(msgs))
    return 0 if ok_all else 1


def gate_cotenant(kernel_path, seed, log, timeout=600):
    log("\n[gate] co-tenant: another stream's kernel holds 3/4 of the CUs (all their LDS) for 4 s")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--cotenant-worker", "--kernel", str(kernel_path),
           "--seed", str(seed)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + p.stderr).strip().splitlines()
        lines = [l for l in out if l.startswith("  ")]
        for l in lines:
            log(l)
        ok = p.returncode == 0
        if not ok and not lines:
            log("  " + " | ".join(out[-5:]))
    except subprocess.TimeoutExpired:
        ok = False
        log(f"  FAIL: no progress within {timeout} s (hang)")
    return 0 if ok else 1


HOG_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
__global__ void hog(unsigned long long ticks, int* started) {
  extern __shared__ int lds[];
  if (threadIdx.x == 0) { lds[0] = 1; atomicAdd(started, 1); }
  unsigned long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) { __builtin_amdgcn_s_sleep(8); }
  if (threadIdx.x == 0 && lds[0] == 2) started[1] = 1;
}
void launch(int64_t blocks, int64_t lds_bytes, double seconds, torch::Tensor started) {
  int rate = 0;
  hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);
  hipFuncSetAttribute((const void*)hog, hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds_bytes);
  unsigned long long ticks = (unsigned long long)(seconds * rate * 1000.0);
  hipLaunchKernelGGL(hog, dim3(blocks), dim3(64), lds_bytes, at::hip::getCurrentHIPStream().stream(), ticks,
                     started.data_ptr<int>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""


def cotenant_worker(mod, seed):
    from torch.utils.cpp_extension import load_inline
    bdir = CACHE / "build_hog"
    bdir.mkdir(parents=True, exist_ok=True)
    hog = load_inline("kda_hog", cpp_sources="", cuda_sources=HOG_SRC, extra_cuda_cflags=["-O2", "--offload-arch=gfx950"],
                      build_directory=str(bdir), verbose=False)
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    lds = 160 * 1024
    blocks = cus * 3 // 4
    side = torch.cuda.Stream()
    bad = 0
    pre = PrefillCase("p12_loaded16k_1", *PREFILL_WORKLOADS["p12_loaded16k_1"], seed + 9)
    dec = DecodeCase("s8_h12_b64", 12, 64, 8, seed + 10)
    dg = dec.capture(mod.decode)
    meta = mod.prefill_prepare(pre.cu_cpu, pre.H, pre.ck_off_cpu, pre.ck_rows_cpu)
    tests = [("prefill p12_loaded16k_1", lambda: mod.prefill(*pre.args(), **pre.ck_kwargs(), meta=meta), pre),
             ("decode s8_h12_b64 eager", lambda: mod.decode(**dec.kwargs()), dec),
             ("decode s8_h12_b64 graph", dg.replay, dec)]
    for name, fn, case in tests:
        if isinstance(case, PrefillCase):
            case.pages.restore(case.snap)
        else:
            case.pages.restore(case.snap)
            case.out.fill_(float("nan"))
        started = torch.zeros(2, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            hog.launch(blocks, lds, 4.0, started)
        t_wait = time.perf_counter()
        while int(started[0].item()) < blocks and time.perf_counter() - t_wait < 2.0:
            time.sleep(0.01)
        t0 = time.perf_counter()
        fn()
        torch.cuda.current_stream().synchronize()
        dt = time.perf_counter() - t0
        side.synchronize()
        if isinstance(case, PrefillCase):
            ok, eo, es, msg = case.verify()
            detail = f"out {eo:.1e} pool {es:.1e}" + (f" {msg}" if msg else "")
        else:
            o_ref, conv_ref, ssm_ref = case.reference()
            oe = rel(case.out, o_ref)
            okp, worst, msg = case.pages.check(case.snap, case.written, ssm_ref, conv_ref, D_ST_RTOL)
            ok = okp and oe <= D_OUT_RTOL
            detail = f"out {oe:.1e} state {worst:.1e}" + (f" {msg}" if msg else "")
        fast = dt < 1.5
        good = ok and fast
        bad += not good
        print(f"  {name:<26} {'PASS' if good else 'FAIL'} (finished {dt * 1e3:.0f} ms after launch beside the hog"
              f"{'' if fast else ': waited for the co-tenant to leave (needs all workgroups resident)'}; {detail})",
              flush=True)
    return bad


# ================================================================================================ legacy
def legacy_prefill(mod, log):
    """The 10 amd_kda_mi350x workloads (seed-0 inputs, contiguous state) through prefill(); metric of that task:
    mean(recorded vLLM latency / latency), eager, cache flushed, median of 50."""
    base = json.loads((DATA / "legacy_prefill_baseline_sota.json").read_text())["latency_ms"]
    ref_dir = CACHE / "legacy_ref"  # fp32 reference outputs, computed and saved on first use (~600 MB)
    ref_dir.mkdir(parents=True, exist_ok=True)
    vs, ok_all = [], True
    log(f"\n[gate] no regression, phase-1 prefill workloads (contiguous state, recorded vLLM latencies)")
    for name, lens in LEGACY_PREFILL.items():
        g = torch.Generator("cuda").manual_seed(0)
        T, N, H = sum(lens), len(lens), 12
        rn = lambda *s, sc=1.0: torch.randn(*s, generator=g, device="cuda") * sc
        q, k, v, rg = rn(1, T, H, D).bfloat16(), rn(1, T, H, D).bfloat16(), rn(1, T, H, D).bfloat16(), rn(1, T, H, D).bfloat16()
        rb = rn(1, T, H).bfloat16()
        A_log = torch.rand(H, generator=g, device="cuda") * 2 - 1
        dt_bias = rn(H * D, sc=0.5)
        state = rn(N, H, D, D, sc=0.05)
        cu_cpu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
        cu = cu_cpu.cuda()
        si = torch.arange(N, dtype=torch.int32, device="cuda")
        hi = torch.ones(N, dtype=torch.bool, device="cuda")
        path = ref_dir / f"ref_{name}.pt"
        if path.exists():
            r = torch.load(path, map_location="cuda")
            o_ref, s_ref = r["output"], r["new_state"]
        else:
            o_ref, s_ref = kda_prefill_reference(q, k, v, rg, rb, A_log, dt_bias, LB, SCALE, state, si, hi, cu)
            torch.save({"output": o_ref, "new_state": s_ref}, path)
        pool = state.clone()
        out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device="cuda")
        try:
            meta = mod.prefill_prepare(cu_cpu, H)
            call = lambda: mod.prefill(q, k, v, rg, rb, A_log, dt_bias, SCALE, LB, pool, si, hi, cu, out, meta=meta)
            call()
            torch.cuda.synchronize()
            eo = max(rel(out[0, a:b], o_ref[0, a:b]) for a, b in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()))
            es = rel(pool, s_ref)
            ok = eo <= P_RTOL and es <= P_RTOL
            if ok:
                for _ in range(10):
                    call()
                ts = sorted(timed_eager(call) for _ in range(50))
                us = ts[25]
        except Exception as e:  # noqa: BLE001
            ok, eo, es = False, float("nan"), float("nan")
            log(f"  {name:<10} RUNTIME_ERROR {type(e).__name__}: {str(e)[:200]}")
        if not ok:
            ok_all = False
            log(f"  {name:<10} FAIL (out {eo:.1e} state {es:.1e})")
            continue
        r = base[name] * 1000.0 / us
        vs.append(r)
        log(f"  {name:<10} {us:9.1f} us  vLLM(recorded) {base[name] * 1000:9.1f} us  {r:6.3f}x  (out {eo:.1e} state {es:.1e})")
        torch.cuda.empty_cache()
    m = sum(vs) / len(vs) if vs else 0.0
    return m, ok_all and len(vs) == len(LEGACY_PREFILL)


def legacy_decode(mod, log):
    """The 17 amd_kda_decode_mi350x workloads (their canonical inputs, dense pools) through decode(); metric of that
    task: mean(recorded best-baseline latency / latency), graph replay, cache flushed, median of 200."""
    base = json.loads((DATA / "legacy_decode_baseline_sota.json").read_text())["latency_us"]
    vs, ok_all = [], True
    log(f"\n[gate] no regression, phase-1 decode workloads (dense pools, recorded baselines)")
    for name, (H, B, S) in LEGACY_DECODE.items():
        g = torch.Generator("cuda").manual_seed(zlib.crc32(name.encode()))
        rn = lambda *s, sc=1.0: torch.randn(*s, generator=g, device="cuda") * sc
        dim, T = H * D, B * S
        taps = W - 1 if S == 1 else W - 2 + S
        slots = 2 * B * S + 16
        perm = torch.randperm(slots - 1, generator=g, device="cuda")[: B * S].int() + 1
        si = perm if S == 1 else perm.view(B, S).contiguous()
        acc = None if S == 1 else torch.randint(1, S + 1, (B,), generator=g, device="cuda", dtype=torch.int32)
        inp = dict(mixed_qkv=rn(T, 3 * dim).bfloat16(), conv_state=rn(slots, taps, 3 * dim).bfloat16(),
                   conv_weight=rn(3, W, dim, sc=0.4), gate=rn(T, H, D).bfloat16(), beta=rn(T, H).bfloat16(),
                   out_gate=rn(T, dim).bfloat16(), A_log=torch.rand(H, generator=g, device="cuda") * 2 - 1,
                   dt_bias=rn(dim, sc=0.5), norm_weight=1 + rn(D, sc=0.1), ssm_state=rn(slots, H, D, D, sc=0.05),
                   state_indices=si, num_accepted_tokens=acc,
                   cu_seqlens=torch.arange(0, T + 1, S, dtype=torch.long, device="cuda"), norm_eps=NORM_EPS,
                   lower_bound=LB)
        conv0, ssm0 = inp["conv_state"].clone(), inp["ssm_state"].clone()
        o_ref, conv_ref, ssm_ref = kda_decode_reference(**inp)
        written = (si.flatten() if S > 1 else si).long()
        out = torch.empty(T, dim, dtype=torch.bfloat16, device="cuda")
        try:
            call = lambda: mod.decode(**inp, out=out)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                call()
            inp["conv_state"].copy_(conv0)
            inp["ssm_state"].copy_(ssm0)
            gr.replay()
            torch.cuda.synchronize()
            oe = rel(out, o_ref)
            se = rel(inp["ssm_state"][written], ssm_ref[written])
            ok = oe <= D_OUT_RTOL and se <= D_ST_RTOL and torch.equal(inp["conv_state"], conv_ref)
            if ok:
                ts = sorted(timed_graph(gr) for _ in range(200))
                us = ts[100]
        except Exception as e:  # noqa: BLE001
            ok, oe, se = False, float("nan"), float("nan")
            log(f"  {name:<13} RUNTIME_ERROR {type(e).__name__}: {str(e)[:200]}")
        if not ok:
            ok_all = False
            log(f"  {name:<13} FAIL (out {oe:.1e} state {se:.1e})")
            continue
        r = base[name] / us
        vs.append(r)
        log(f"  {name:<13} {us:8.2f} us  best baseline(recorded) {base[name]:8.2f} us  {r:6.3f}x")
        torch.cuda.empty_cache()
    m = sum(vs) / len(vs) if vs else 0.0
    return m, ok_all and len(vs) == len(LEGACY_DECODE)


# ================================================================================================ main
GATE_NAMES = ["contract", "host", "numerics", "decode", "integration", "cotenant", "legacy"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=str(ROOT / "kda" / "kernel.py"))
    ap.add_argument("--quick", action="store_true", help="4 + 4 serving workloads, 1 trial, no gates")
    ap.add_argument("--part", choices=["prefill", "decode", "both"], default="both")
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument("--gates", default=None, help="run only these gates (comma list of: " + ", ".join(GATE_NAMES) + ")")
    ap.add_argument("--no-gates", action="store_true", help="serving workloads only (not an official score)")
    ap.add_argument("--seed", type=int, default=None, help="default: fresh random seed every run")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--heldout", default=None, help="file with a dict of extra prefill compositions (unscored)")
    ap.add_argument("--cotenant-worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if "HIP_VISIBLE_DEVICES" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("No GPU selected: set HIP_VISIBLE_DEVICES to ONE idle MI350X (the timings assume the GPU is not shared).")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        mod = load_kernel(args.kernel)
    except Exception as e:  # noqa: BLE001
        print("LOAD / COMPILE ERROR DETAILS:")
        print(str(e)[-6000:])
        print("  mean_vs_base = 0.000 (candidate failed to load)")
        return 1

    if args.cotenant_worker:
        with torch.inference_mode():
            return 1 if cotenant_worker(mod, seed) else 0

    print(f"[OK] {torch.cuda.get_device_name(0)} | {args.kernel} | seed {seed}")
    if args.heldout:
        import ast
        global PREFILL_WORKLOADS
        PREFILL_WORKLOADS = {k: (v[0], v[1]) for k, v in ast.literal_eval(Path(args.heldout).read_text()).items()}
        with torch.inference_mode():
            pg, pok, _ = run_prefill_set(mod, list(PREFILL_WORKLOADS), seed, args.iters or 20, args.trials or 1)
        print(f"\nheld-out prefill geomean vs vLLM = {pg:.3f} ({'all pass' if pok else 'FAILURES'}) [unscored]")
        return 0 if pok else 1

    official = not (args.quick or args.workloads or args.gates or args.no_gates or args.part != "both" or
                    args.iters or args.trials)
    p_iters, d_iters, trials = (10, 50, 1) if args.quick else (30, 100, 3)
    if args.iters:
        p_iters = d_iters = args.iters
    if args.trials:
        trials = args.trials
    pg = dg = 1.0
    pok = dok = True
    prows, drows = [], []
    run_serving = args.gates is None
    with torch.inference_mode():
        if run_serving and args.part in ("prefill", "both"):
            names = [w for w in (args.workloads or (QUICK_PREFILL if args.quick else PREFILL_WORKLOADS)) if w in PREFILL_WORKLOADS]
            if names:
                pg, pok, prows = run_prefill_set(mod, names, seed, p_iters, trials)
                print(f"prefill_geomean_vs_vllm = {pg:.3f} ({'all pass' if pok else 'FAILURES'})")
        if run_serving and args.part in ("decode", "both"):
            names = [w for w in (args.workloads or (QUICK_DECODE if args.quick else DECODE_WORKLOADS)) if w in DECODE_WORKLOADS]
            if names:
                dg, dok, drows = run_decode_set(mod, names, seed, d_iters, trials)
                print(f"decode_geomean_vs_best_baseline = {dg:.3f} ({'all pass' if dok else 'FAILURES'})")
        score = math.sqrt(pg * dg) if (pok and dok) else 0.0
        if not official and args.gates is None and (args.quick or args.no_gates or args.workloads or args.part != "both"
                                                     or args.iters or args.trials):
            print(f"\n  mean_vs_base = {score:.3f} [partial: quick / subset / no gates; not an official score]")
            return 0 if (pok and dok) else 1

        gates = args.gates.split(",") if args.gates else GATE_NAMES
        failed = {}
        if "contract" in gates:
            failed["prefill contract"] = gate_prefill_contract(mod, seed, print, heavy=official or "contract" in gates)
        if "host" in gates:
            failed["prefill host"] = gate_prefill_host(mod, seed, print)
        if "numerics" in gates:
            failed["numerics"] = gate_numerics(mod, seed, print)
        if "decode" in gates:
            failed["decode contract"] = gate_decode_contract(mod, seed, print)
        if "integration" in gates:
            failed["integration"] = gate_integration(mod, seed, print)
        if "cotenant" in gates:
            failed["co-tenant"] = gate_cotenant(args.kernel, seed, print)
        lp = ld = None
        if "legacy" in gates:
            lp, lp_ok = legacy_prefill(mod, print)
            ld, ld_ok = legacy_decode(mod, print)
            failed["phase-1 workload correctness"] = int(not lp_ok) + int(not ld_ok)
            print(f"  legacy prefill mean_vs_base = {lp:.3f} (floor {LEGACY_PREFILL_FLOOR:.3f}); "
                  f"legacy decode mean_vs_base = {ld:.3f} (floor {LEGACY_DECODE_FLOOR:.3f})")
            if lp_ok and lp < LEGACY_PREFILL_FLOOR:
                failed["phase-1 prefill regression"] = 1
            if ld_ok and ld < LEGACY_DECODE_FLOOR:
                failed["phase-1 decode regression"] = 1

    print(f"\n{'=' * 72}")
    if run_serving:
        for tag, rows in (("prefill", prows), ("decode", drows)):
            items = [f"{n} {r:.2f}x" for n, _, _, r in rows]
            for j in range(0, len(items), 5):
                print(f"  {tag if j == 0 else '':<8}" + ", ".join(items[j:j + 5]))
        print(f"  prefill_geomean_vs_vllm = {pg:.3f}   decode_geomean_vs_best_baseline = {dg:.3f}")
    problems = [f"{k} ({v})" for k, v in failed.items() if v]
    if run_serving and not (pok and dok):
        problems.insert(0, "serving workload correctness")
    if problems:
        print("GATE FAILURE: " + "; ".join(problems))
        if run_serving:
            print("  mean_vs_base = 0.000 (a gate failed)")
        print("=" * 72)
        return 1
    if not run_serving:
        print("  all selected gates pass [gates only; not an official score]")
        print("=" * 72)
        return 0
    print(f"  mean_vs_base = {score:.3f} (sqrt(prefill geomean x decode geomean) over "
          f"{len(PREFILL_WORKLOADS)} + {len(DECODE_WORKLOADS)} serving workloads)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
