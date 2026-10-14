#!/usr/bin/env python3
"""Correctness of kda_b200 against the fp64 token-by-token recurrence (reference.py).

Covers the Int21 shapes (H = 64 / 96, 8192 tokens; run() and static prepare()), varlen serving calls with a state
pool, prefix-cache checkpoints and a committed-state source pool, CUDA-graph replays of a dynamic-layout launch with
changing batch layouts, other head counts and tiny sequences, and the API error contract.
Each case: finite, every element within half the tensor's RMS or 5% of its own magnitude, relative L2 < 1e-2;
untouched pool slots and the source pool bit-identical.

    CUDA_VISIBLE_DEVICES=<gpu> python test_kda_b200.py      # or: pytest -q test_kda_b200.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kda_b200  # noqa: E402
from bench import ck_meta, cu_of, make_inputs  # noqa: E402
from reference import kda_reference  # noqa: E402

D = 128
REL_L2_MAX = 1e-2
MIXED = [1300, 547, 2048, 963, 271, 3063]


def close(got, gold, what):
    x, y = got.float(), gold.float()
    err = (x - y).abs()
    ratio = torch.minimum(err / (0.5 * y.pow(2).mean().sqrt()), err / (0.05 * y.abs() + 1e-30))
    worst = float(torch.nan_to_num(ratio, nan=float("inf")).max())
    l2 = float(torch.linalg.vector_norm(got.double() - gold.double()) / torch.linalg.vector_norm(gold.double()).clamp_min(1e-30))
    assert bool(torch.isfinite(x).all()), f"{what}: non-finite values"
    assert worst <= 1.0 and l2 < REL_L2_MAX, f"{what}: worst element ratio {worst:.2f}, relative L2 {l2:.2e}"


def reference(q, k, v, g, beta, A_log, dt_bias, scale, init, lens, every=0):
    """fp64 recurrence per sequence; with every > 0 also the checkpoint rows (state before token j * every), obtained by
    chaining reference.py over the checkpoint segments."""
    outs, finals, cks, t0 = [], [], [], 0
    for n, L in enumerate(lens):
        S = init[n:n + 1]
        seg = every if every else max(L, 1)
        for s in range(0, L, seg):
            e = min(L, s + seg)
            if every:
                cks.append(S[0])
            sl = slice(t0 + s, t0 + e)
            o, S = kda_reference(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], A_log, dt_bias, scale, S,
                                 torch.tensor([0, e - s], device=q.device))
            outs.append(o)
        finals.append(S[0])
        t0 += L
    return torch.cat(outs, 1), torch.stack(finals), (torch.stack(cks) if every else None)


def _gates_slow(inputs, gen):
    """Slow decay (long memory): exercises the cross-chunk state hand-off much harder than the random gates."""
    H = inputs[5].numel()
    inputs[5].fill_(0.0)
    inputs[6].copy_(torch.rand(H * D, generator=gen, device="cuda") * 6.0 - 12.0)


# ----------------------------------------------------------------------------- plain calls
def _plain(H, lens, seed, slow=False, static=False):
    inputs = make_inputs(H, lens, seed)
    if slow:
        _gates_slow(inputs, torch.Generator(device="cuda").manual_seed(seed))
    q, k, v, g, beta, A_log, dt_bias, scale, init, cu = inputs
    with torch.no_grad():
        res = kda_b200.prepare(*inputs)() if static else kda_b200.run(*inputs)
        torch.cuda.synchronize()
        out, fin, _ = reference(q, k, v, g, beta, A_log, dt_bias, scale, init, lens)
    close(res[0], out, f"H={H} lens={lens[:8]} output")
    close(res[1], fin, f"H={H} lens={lens[:8]} final_state")


def test_int21_shapes():
    for i, (H, lens) in enumerate([(96, [8192]), (96, MIXED), (96, [1024] * 8), (64, [8192]), (64, MIXED), (64, [1024] * 8)]):
        _plain(H, lens, 100 + i, static=True)
        _plain(H, lens, 200 + i, static=False)


def test_serving_head_counts_varlen():
    for i, (H, lens) in enumerate([(12, [8192]), (16, [3000, 13384]), (12, [256, 512, 900, 1600, 1600, 2400, 4000, 8700]),
                                   (16, [64] * 16), (12, [random.Random(3).randint(17, 255) for _ in range(16)])]):
        _plain(H, lens, 300 + i)


def test_slow_decay():
    _plain(12, [16384], 400, slow=True)
    _plain(64, [3000, 1, 2048, 777, 64, 33], 401, slow=True)


def test_other_heads_and_tiny():
    for i, (H, lens) in enumerate([(1, [4096]), (4, list(range(1, 16))), (8, [37]), (32, [700, 1300, 45]), (96, [1]),
                                   (128, [2048]), (12, [1, 17, 33, 1, 1023, 64, 1, 2])]):
        _plain(H, lens, 500 + i)


def test_single_sequence_no_cu():
    inputs = make_inputs(16, [5000], 600)
    assert inputs[9] is None
    q, k, v, g, beta, A_log, dt_bias, scale, init, _ = inputs
    with torch.no_grad():
        res = kda_b200.run(*inputs)
        out, fin, _ = reference(q, k, v, g, beta, A_log, dt_bias, scale, init, [5000])
    close(res[0], out, "cu_seqlens=None output")
    close(res[1], fin, "cu_seqlens=None final_state")


# ----------------------------------------------------------------------------- serving contract
def _serving(H, lens, seed, every=64, use_source=False, pad=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    inputs = make_inputs(H, lens, seed)
    q, k, v, g, beta, A_log, dt_bias, scale, _, _ = inputs
    cu = cu_of(lens)
    N, P = len(lens), len(lens) + 5
    slot = H * D * D
    pool = (torch.randn(P, slot + pad, generator=gen, device="cuda") * 0.25).as_strided((P, H, D, D), (slot + pad, D * D, D, 1))
    idx = torch.randperm(P, generator=gen, device="cuda")[:N].to(torch.int32)
    start = pool[idx.long()].clone()
    kw = dict(ssm_state_indices=idx)
    src = None
    if use_source:
        src = torch.randn(P + 3, H, D, D, generator=gen, device="cuda") * 0.25
        sidx = torch.randperm(P + 3, generator=gen, device="cuda")[:N].to(torch.int32)
        kw.update(initial_state_source=src, initial_state_indices=sidx)
        start = src[sidx.long()].clone()
    src_before = None if src is None else src.clone()
    ck = None
    if every:
        ccs, C = ck_meta(lens, every)
        ck = torch.full((C, H, D, D), float("nan"), device="cuda")
        kw.update(state_checkpoints=ck, checkpoint_cu_starts=ccs, checkpoint_every_n_tokens=every)
    pool_before = pool.clone()
    with torch.no_grad():
        res = kda_b200.kda_prefill(q, k, v, g, beta, A_log, dt_bias, pool, cu, scale, **kw)
        torch.cuda.synchronize()
        out, fin, gck = reference(q, k, v, g, beta, A_log, dt_bias, scale, start, lens, every)
    tag = f"H={H} lens={lens[:8]} every={every} source={use_source} pad={pad}"
    close(res[0], out, tag + " output")
    close(pool[idx.long()], fin, tag + " pool slots")
    keep = torch.ones(P, dtype=torch.bool, device="cuda")
    keep[idx.long()] = False
    assert torch.equal(pool[keep], pool_before[keep]), tag + ": other pool slots modified"
    if src is not None:
        assert torch.equal(src, src_before), tag + ": source pool modified"
    if every:
        assert len(res) == 3 and res[2] is ck
        close(ck, gck, tag + " checkpoints")


def test_pool_checkpoints():
    _serving(12, [300, 64, 1, 1000, 77], 700)
    _serving(16, [4096] * 4, 701)
    _serving(12, [8192], 702, every=256)
    _serving(16, [2048, 513, 77], 703, use_source=True, pad=64)
    _serving(12, [1000, 200, 3], 704, every=0)


def test_dynamic_layout_graph_replay():
    for H, T, N, every in ((12, 2048, 6, 64), (16, 4096, 3, 64), (12, 1024, 16, 0)):
        rng = random.Random(H * T + N)
        gen = torch.Generator(device="cuda").manual_seed(H + T + N)

        def partition():
            cuts = sorted(rng.sample(range(1, T), N - 1))
            return [b - a for a, b in zip([0, *cuts], [*cuts, T])]

        lens0 = partition()
        q, k, v, g, beta, A_log, dt_bias, scale, _, _ = make_inputs(H, lens0, 800 + H + N)
        cu = cu_of(lens0)
        P = N + 4
        pool = torch.randn(P, H, D, D, generator=gen, device="cuda") * 0.25
        idx = torch.randperm(P, generator=gen, device="cuda")[:N].to(torch.int32)
        kw = dict(dynamic_layout=True, ssm_state_indices=idx)
        ck = ccs = None
        if every:
            ccs, _ = ck_meta(lens0, every)
            ck = torch.full((T // every + N, H, D, D), float("nan"), device="cuda")
            kw.update(state_checkpoints=ck, checkpoint_cu_starts=ccs, checkpoint_every_n_tokens=every)
        pool0 = pool.clone()
        with torch.no_grad():
            launch = kda_b200.prepare(q, k, v, g, beta, A_log, dt_bias, scale, pool, cu, **kw)
            launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                res = launch()
            for r, lens in enumerate([partition(), partition(), [T - (N - 1)] + [1] * (N - 1)]):
                cu.copy_(cu_of(lens))
                for t in (q, k, v, g, beta):
                    t.copy_(torch.randn(t.shape, generator=gen, device="cuda") * 0.5)
                pool.copy_(pool0)
                idx.copy_(torch.randperm(P, generator=gen, device="cuda")[:N].to(torch.int32))
                if every:
                    new_ccs, C = ck_meta(lens, every)
                    ccs.copy_(new_ccs)
                    ck.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert kda_b200.status_bits(launch) == 0
                out, fin, gck = reference(q, k, v, g, beta, A_log, dt_bias, scale, pool0[idx.long()], lens, every)
                tag = f"replay {r} H={H} T={T} N={N}"
                close(res[0], out, tag + " output")
                close(pool[idx.long()], fin, tag + " pool slots")
                if every:
                    close(ck[:C], gck, tag + " checkpoints")


def test_wrapper_matches_run():
    inputs = make_inputs(12, [1000, 3000, 17], 900)
    q, k, v, g, beta, A_log, dt_bias, scale, init, cu = inputs
    with torch.no_grad():
        a = kda_b200.run(*inputs)
        b = kda_b200.kda_prefill(q, k, v, g, beta, A_log, dt_bias, init, cu)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_api_errors():
    inputs = make_inputs(12, [100, 200, 300, 400], 950)
    q, k, v, g, beta, A_log, dt_bias, scale, _, cu = inputs
    pool = torch.zeros(8, 12, D, D, device="cuda")
    idx = torch.arange(8, dtype=torch.int32, device="cuda")[::2]   # non-contiguous slot table
    try:
        kda_b200.run(q, k, v, g, beta, A_log, dt_bias, scale, pool, cu, ssm_state_indices=idx)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-contiguous ssm_state_indices must raise ValueError")
    try:
        kda_b200.run(q, k, v, g, beta, A_log[:6], dt_bias, scale, pool[:4], cu)
    except ValueError:
        pass
    else:
        raise AssertionError("a short A_log must raise ValueError")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)
        torch.cuda.empty_cache()
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
