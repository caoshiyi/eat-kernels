#!/usr/bin/env python3
"""
Benchmark kda_b200 against FlashKDA on B200: the eval the kernel was optimized against (EAT task
b200_kda_serving_clean), with the same input recipe, correctness gate and timing protocol.

Serving set (scored): 26 workloads, each at H = 12 and H = 16, called the way a serving engine calls prefill:
the state is a pool of N + 8 fp32 slots with a random slot permutation (ssm_state_indices), prefix-cache
checkpoints are written every 64 tokens, and the timed call is the launch from prepare(..., dynamic_layout=True)
(the CUDA-graph-safe path). FlashKDA gets the same initial states as a contiguous tensor and writes no
checkpoints, i.e. it does strictly less work.
Int21 set: six plain calls at H = 64 / 96 (8192 tokens), static prepare().

Baseline: FlashKDA (MoonshotAI, fused CUTLASS forward) through FLA's chunk_kda with the final state requested,
timed live in the same process. Correctness (before timing and again after every input is refilled in place):
output and updated state vs FLA's Triton chunk_kda, every element within half the tensor's RMS or 5% of its own
magnitude, relative L2 <= 0.03. Timing: CUPTI kernel time (cupti_timer.py), cold L2, the call captured in a CUDA
graph, median of 30 iterations, median of 3 trials; fresh random seed per run unless --seed is given.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python bench.py            # serving (26) + Int21 (6)
    CUDA_VISIBLE_DEVICES=<gpu> python bench.py --quick    # 1 trial x 10 iterations
    CUDA_VISIBLE_DEVICES=<gpu> python bench.py --sets serving --heads 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

D = 128
LOWER_BOUND = -5.0
EVERY = 64
ATOL_RMS, RTOL, MAX_REL_L2 = 0.5, 0.05, 0.03

AGENT = [256, 512, 900, 1600, 1600, 2400, 4000, 8700]
_varlen = tuple(random.Random(16).randint(17, 255) for _ in range(16))
FAMILIES = {
    "long": {"1x8192": (8192,), "1x32768": (32768,), "2x8192": (8192,) * 2, "3000+13384": (3000, 13384),
             "4x4096": (4096,) * 4, "8x1088": (1088,) * 8},
    "resid": {"4x64": (64,) * 4, "16x64": (64,) * 16, "64x64": (64,) * 64, "16x256": (256,) * 16,
              "16x17..255": _varlen},
    "agent": {"agent8": tuple(AGENT), "agent16": tuple(AGENT * 2)},
}
MIXED = [1300, 547, 2048, 963, 271, 3063]
INT21 = {
    "h96-fixed": (96, [8192]), "h96-mixed_varlen": (96, MIXED), "h96-uniform_varlen": (96, [1024] * 8),
    "h64-fixed": (64, [8192]), "h64-mixed_varlen": (64, MIXED), "h64-uniform_varlen": (64, [1024] * 8),
}


def serving_workloads(heads, families):
    return {f"h{H}-{fam}-{name}": (H, list(lens)) for H in heads for fam in families for name, lens in FAMILIES[fam].items()}


# ----------------------------------------------------------------------------- inputs (judge recipe)
def make_inputs(heads, seq_lens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(seq_lens)

    def randn(shape, scale, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=gen, device="cuda") * scale).to(dtype)

    A_log = torch.log(torch.empty(heads, device="cuda").uniform_(1.0, 16.0, generator=gen))
    dt = torch.exp(torch.rand(heads * D, generator=gen, device="cuda") * (math.log(0.1) - math.log(0.001))
                   + math.log(0.001)).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    q, k, v, g = (randn((1, T, heads, D), 0.5) for _ in range(4))
    beta = randn((1, T, heads), 0.5)
    initial_state = randn((len(seq_lens), heads, D, D), 0.25, torch.float32)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = cu_of(seq_lens)
    return [q, k, v, g, beta, A_log, dt_bias, 1.0 / math.sqrt(D), initial_state, cu_seqlens]


def refill_(inputs, seed):
    """New values in the same buffers (shapes, strides and addresses unchanged)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for t, sc in ((inputs[0], 0.5), (inputs[1], 0.5), (inputs[2], 0.5), (inputs[3], 0.5), (inputs[4], 0.5),
                  (inputs[8], 0.25)):
        t.copy_(torch.randn(t.shape, generator=gen, device="cuda") * sc)


def cu_of(lens):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int64, device="cuda")


def pool_of(P, H, gen):
    return torch.randn(P, H, D, D, generator=gen, device="cuda") * 0.25


def ck_meta(lens, every):
    counts = [-(-n // every) for n in lens]
    return torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int64, device="cuda"), sum(counts)


# ----------------------------------------------------------------------------- baseline / reference
def fla_chunk_kda(flash_kda, q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens):
    """flash_kda=True: FlashKDA (the timed baseline); False: FLA's Triton kernel (the correctness reference)."""
    os.environ["FLA_FLASH_KDA"] = "1" if flash_kda else "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    return chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=float(scale), use_qk_l2norm_in_kernel=True,
                     use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True,
                     safe_gate=True, lower_bound=LOWER_BOUND, A_log=A_log, dt_bias=dt_bias,
                     initial_state=initial_state, output_final_state=True, cu_seqlens=cu_seqlens)


def check(got, want):
    x, y = got.float(), want.float()
    err = (x - y).abs()
    ratio = torch.minimum(err / (ATOL_RMS * y.pow(2).mean().sqrt()), err / (RTOL * y.abs() + 1e-30))
    rel_l2 = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y).clamp_min(1e-30))
    worst = float(torch.nan_to_num(ratio, nan=float("inf")).max())
    return worst <= 1.0 and rel_l2 <= MAX_REL_L2 and bool(torch.isfinite(x).all()), worst, rel_l2


def as_tuple(res):
    return (res,) if torch.is_tensor(res) else tuple(res)


# ----------------------------------------------------------------------------- timing
def cupti_ms(fn, warmup, iters, use_cuda_graph):
    from cupti_timer import bench_gpu_time_with_cupti

    return statistics.median(bench_gpu_time_with_cupti(fn, warmup, iters, cold_l2_cache=True, use_cuda_graph=use_cuda_graph))


def time_launch(launch, warmup, iters):
    try:
        return cupti_ms(launch, warmup, iters, use_cuda_graph=True)
    except Exception:  # noqa: BLE001 - a launch that replays its own CUDA graph is timed as issued
        torch.cuda.synchronize()
        return cupti_ms(launch, warmup, iters, use_cuda_graph=False)


# ----------------------------------------------------------------------------- workloads
def serving_case(mod, H, lens, seed, warmup, iters, trials):
    inputs = make_inputs(H, lens, seed)
    if inputs[9] is None:
        inputs[9] = cu_of(lens)
    N = len(lens)
    gen = torch.Generator(device="cuda").manual_seed(seed + 7)
    pool = pool_of(N + 8, H, gen)
    idx = torch.randperm(N + 8, generator=gen, device="cuda")[:N].to(torch.int32)
    ccs, C = ck_meta(lens, EVERY)
    ck = torch.zeros((C, H, D, D), device="cuda")
    kw = dict(ssm_state_indices=idx, state_checkpoints=ck, checkpoint_cu_starts=ccs, checkpoint_every_n_tokens=EVERY)

    def reference():
        base_inputs = inputs[:8] + [pool[idx.long()].clone(), inputs[9]]
        return base_inputs, fla_chunk_kda(False, *[x.clone() if torch.is_tensor(x) else x for x in base_inputs])

    def verify(ref):
        res = as_tuple(launch())
        torch.cuda.synchronize()
        ok_o, w_o, l_o = check(res[0], ref[0])
        ok_s, w_s, l_s = check(pool[idx.long()], ref[1])
        return ok_o and ok_s, f"out worst {w_o:.2f} relL2 {l_o:.4f}, pool worst {w_s:.2f} relL2 {l_s:.4f}"

    with torch.no_grad():
        base_inputs, ref = reference()
        launch = mod.prepare(*inputs[:8], pool, inputs[9], dynamic_layout=True, **kw)
        ok, detail = verify(ref)
        if not ok:
            return None, detail
        base_ms, cand_ms = [], []
        for _ in range(trials):
            base_ms.append(cupti_ms(lambda: fla_chunk_kda(True, *base_inputs), warmup, iters, True))
            cand_ms.append(time_launch(launch, warmup, iters))
        refill_(inputs, seed + 500)
        pool.copy_(pool_of(N + 8, H, torch.Generator(device="cuda").manual_seed(seed + 501)))
        _, ref2 = reference()
        ok2, detail2 = verify(ref2)
    if not ok2:
        return None, f"after in-place refill: {detail2}"
    return (statistics.median(base_ms), statistics.median(cand_ms)), detail


def int21_case(mod, H, lens, seed, warmup, iters, trials):
    inputs = make_inputs(H, lens, seed)

    def verify(ref):
        res = as_tuple(launch())
        torch.cuda.synchronize()
        (ok_o, w_o, l_o), (ok_s, w_s, l_s) = check(res[0], ref[0]), check(res[1], ref[1])
        return ok_o and ok_s, f"out worst {w_o:.2f} relL2 {l_o:.4f}, state worst {w_s:.2f} relL2 {l_s:.4f}"

    with torch.no_grad():
        ref = fla_chunk_kda(False, *[x.clone() if torch.is_tensor(x) else x for x in inputs])
        launch = mod.prepare(*inputs)
        ok, detail = verify(ref)
        if not ok:
            return None, detail
        base_ms, cand_ms = [], []
        for _ in range(trials):
            base_ms.append(cupti_ms(lambda: fla_chunk_kda(True, *inputs), warmup, iters, True))
            cand_ms.append(time_launch(launch, warmup, iters))
        refill_(inputs, seed + 500)
        ref2 = fla_chunk_kda(False, *[x.clone() if torch.is_tensor(x) else x for x in inputs])
        ok2, detail2 = verify(ref2)
    if not ok2:
        return None, f"after in-place refill: {detail2}"
    return (statistics.median(base_ms), statistics.median(cand_ms)), detail


def run_set(title, case_fn, mod, workloads, seed, warmup, iters, trials, rows):
    print(f"\n{title:26s} {'N':>3s} {'T':>6s} {'FlashKDA ms':>11s} {'kernel ms':>9s} {'speedup':>8s}  correctness")
    speedups, ok_all = [], True
    for i, (name, (H, lens)) in enumerate(workloads.items()):
        t, detail = case_fn(mod, H, lens, seed * 1000 + i, warmup, iters, trials)
        if t is None:
            ok_all = False
            print(f"{name:26s} {len(lens):3d} {sum(lens):6d} {'':11s} {'':9s} {'':8s}  FAIL ({detail})", flush=True)
            continue
        b, c = t
        speedups.append(b / c)
        rows[name] = {"H": H, "lens": lens, "flashkda_ms": b, "kernel_ms": c, "speedup": b / c}
        print(f"{name:26s} {len(lens):3d} {sum(lens):6d} {b:11.4f} {c:9.4f} {b / c:7.3f}x  PASS ({detail})", flush=True)
        torch.cuda.empty_cache()
    gm = math.exp(sum(map(math.log, speedups)) / len(speedups)) if speedups else 0.0
    return gm, ok_all and len(speedups) == len(workloads)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="serving,int21", help="comma list of: serving, int21")
    ap.add_argument("--heads", default="12,16", help="serving head counts")
    ap.add_argument("--families", default=",".join(FAMILIES), help="serving families: " + ", ".join(FAMILIES))
    ap.add_argument("--quick", action="store_true", help="1 trial x 10 iterations")
    ap.add_argument("--seed", type=int, default=None, help="default: fresh random seed every run")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--json", default=None, help="write per-workload results to this file")
    args = ap.parse_args(argv)
    if args.quick:
        args.iters, args.trials = 10, 1
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")

    import kda_b200

    sets = args.sets.split(",")
    print(f"[OK] {torch.cuda.get_device_name()} | torch {torch.__version__} | seed {seed} | {args.trials} x {args.iters} iters")
    rows, summary, ok = {}, {}, True
    if "serving" in sets:
        wl = serving_workloads([int(h) for h in args.heads.split(",")], args.families.split(","))
        gm, s_ok = run_set("serving workload", serving_case, kda_b200, wl, seed, args.warmup, args.iters, args.trials, rows)
        summary["serving_geomean_vs_flashkda"] = gm
        ok &= s_ok
    if "int21" in sets:
        gm, i_ok = run_set("Int21 workload", int21_case, kda_b200, INT21, seed + 1, args.warmup, args.iters, args.trials, rows)
        summary["int21_geomean_vs_flashkda"] = gm
        ok &= i_ok
    print("\n" + "=" * 64)
    for k, v in summary.items():
        print(f"  {k} = {v:.3f}")
    print(f"  correctness: {'all pass' if ok else 'FAILURES'}")
    print("=" * 64)
    if args.json:
        Path(args.json).write_text(json.dumps({"device": torch.cuda.get_device_name(), "seed": seed, "summary": summary,
                                               "all_pass": ok, "workloads": rows}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
