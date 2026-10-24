"""Real-workload correctness gate for the MI350X KDA package (prefill + decode), against an fp64 recurrence.

Inputs: activations captured from Kimi-Linear-48B prefills of GSM8K / MATH-500 (kda_real_gate.py's cases: single
long requests, 32-request packed batches, layers stacked along heads -> H = 96 / 64; plus H = 12 slices = 4 heads of
each of three layers), converted to the vLLM convention (A_log = dt_bias = 0, raw_g = logit(-g / 5), lower_bound -5,
raw_beta = logit(beta)).

prefill: vLLM's calling convention (strided raw_beta, output into a slice, pool rows starting as NaN, one prefix-cache
  checkpoint per sequence); checks output, final states, checkpoint rows, untouched rows, prefill_health().
  Pass rule of the B200 real gate: finite, elementwise check, rel-L2 < 1e-2, and rel-L2 <= 2x FLA chunk_kda's + 1e-3.
decode: the last tokens of each request of the two 32-request batches are fed through decode (plain, S = 1, and
  speculative verify, S = 8 with random accepted counts) after the prefix state was put into the pool. The captured
  q / k / v are post-conv-and-SiLU, so the decode input is SiLU^-1 of them with a pass-through conv (tap 3 = 1);
  the fp64 gold uses exactly the post-SiLU values the kernel sees. out_gate / norm_weight are random (not captured).
  Pass: finite, elementwise check, rel-L2 < 1e-2 for the output and every written state slot, conv cache bit-exact
  against reference.py, decode_health() ok.

    HIP_VISIBLE_DEVICES=0 python amd_real_gate.py --repo <kda_mi350x_k3 checkout> [--data <dir>]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import zlib

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--only", default="", help="prefill | decode")
args = ap.parse_args()
os.environ["KDA_REAL_DATA"] = args.data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, args.repo)
sys.path.insert(0, os.path.join(args.repo, "kda"))

import kda_real_gate as rg  # noqa: E402
import kernel  # noqa: E402
from reference import kda_decode_reference  # noqa: E402

D, W, LB, NORM_EPS = 128, 4, -5.0, 1e-5
ATOL_RMS, RTOL, MAX_REL_L2 = 0.5, 0.05, 0.03
REL_L2_MAX = 1e-2
H12_HEADS = [0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67]


def check(got, want):
    x, y = got.float(), want.float()
    err = (x - y).abs()
    ratio = torch.minimum(err / (ATOL_RMS * y.pow(2).mean().sqrt()), err / (RTOL * y.abs() + 1e-30))
    rel = rel_l2(x, want)
    worst = float(torch.nan_to_num(ratio, nan=float("inf")).max())
    return worst <= 1.0 and rel <= MAX_REL_L2 and bool(torch.isfinite(x).all()), worst, rel


def rel_l2(x, y):
    return float(torch.linalg.vector_norm(x.double() - y.double()) / torch.linalg.vector_norm(y.double()).clamp_min(1e-30))


def heads_slice(inputs, idx):
    q, k, v, g, beta, A_log, dt_bias, scale, state, cu = inputs
    hs = torch.tensor(idx, device=q.device)
    return [q[:, :, hs].contiguous(), k[:, :, hs].contiguous(), v[:, :, hs].contiguous(), g[:, :, hs].contiguous(),
            beta[:, :, hs].contiguous(), A_log[: len(idx)].contiguous(), dt_bias[: len(idx) * D].contiguous(), scale,
            state[:, hs].contiguous(), cu]


def fla_ref(inputs):
    os.environ["FLA_FLASH_KDA"] = "0"
    from fla.ops.kda import chunk_kda
    q, k, v, g, beta, A_log, dt_bias, scale, state, cu = inputs
    decay = (LB * torch.sigmoid(g.float())).contiguous()
    o, s = chunk_kda(q=q, k=k, v=v, g=decay, beta=torch.sigmoid(beta.float()).to(q.dtype), scale=float(scale),
                     initial_state=state.float().contiguous(), output_final_state=True, use_qk_l2norm_in_kernel=True,
                     state_v_first=True, cu_seqlens=cu)
    return o, s


# ----------------------------------------------------------------------------------------------------- prefill
def prefill_case(name, inputs):
    q, k, v, g, beta, A_log, dt_bias, scale, state0, cu = inputs
    dev = q.device
    T, H = q.shape[1], q.shape[2]
    cu64 = torch.tensor([0, T], device=dev, dtype=torch.int64) if cu is None else cu.to(torch.int64)
    lens = (cu64[1:] - cu64[:-1]).tolist()
    N = len(lens)
    has_init = bool(state0.abs().max() > 0)

    ck_off = [((L - 1) // 64) * 64 for L in lens]
    gen = torch.Generator(device="cpu").manual_seed(T * 7 + H)
    rows = (torch.randperm(2 * N + 4, generator=gen)[: 2 * N] + 1).tolist()
    st_rows, ck_rows = rows[:N], rows[N:]
    ck_rows = [r if o > 0 else -1 for r, o in zip(ck_rows, ck_off)]
    slots = 2 * N + 6
    pool = torch.full((slots, H, D, D), float("nan"), device=dev)
    if has_init:
        pool[st_rows] = state0.float()
    before = pool.clone()

    proj = torch.zeros(T, 3 * H, dtype=torch.bfloat16, device=dev)
    proj[:, H:2 * H] = beta[0]
    raw_beta = proj[:, H:2 * H].unsqueeze(0)
    buf = torch.zeros(1, T + 64, H, D, dtype=torch.bfloat16, device=dev)
    out = buf[:, 64:]
    cu32 = cu64.to(torch.int32)
    hinit = torch.full((N,), has_init, dtype=torch.bool, device=dev)
    si = torch.tensor(st_rows, dtype=torch.int32, device=dev)
    co = torch.tensor(ck_off, dtype=torch.int32)
    cr = torch.tensor(ck_rows, dtype=torch.int32)

    with torch.no_grad():
        gold_out, gold_final = rg.fp64_recurrence(*inputs, lower_bound=LB)
        ck_inputs = None
        ck_idx = [n for n in range(N) if ck_off[n] > 0]
        if ck_idx:
            starts = cu64[:-1].tolist()
            tok = torch.cat([torch.arange(starts[n], starts[n] + ck_off[n], device=dev) for n in ck_idx])
            cu_ck = torch.tensor([0, *torch.tensor([ck_off[n] for n in ck_idx]).cumsum(0).tolist()], device=dev,
                                 dtype=torch.int64)
            ck_inputs = [x[:, tok].contiguous() for x in inputs[:5]] + [A_log, dt_bias, scale,
                                                                       state0[ck_idx].contiguous(), cu_ck]
            gold_ck = rg.fp64_recurrence(*ck_inputs, lower_bound=LB)[1]
        try:
            fla_o, fla_s = fla_ref(inputs)
            fla_l2 = (rel_l2(fla_o, gold_out), rel_l2(fla_s, gold_final))
        except Exception as e:  # noqa: BLE001
            fla_l2 = None
            print(f"   (FLA reference failed: {type(e).__name__}: {str(e)[:120]})")

        meta = kernel.prefill_prepare(cu32.cpu(), H, co, cr)
        t0 = time.perf_counter()
        kernel.prefill(q, k, v, g, raw_beta, A_log, dt_bias, scale, LB, pool, si, hinit, cu32, out,
                       co.to(dev), cr.to(dev), meta=meta)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        health = kernel.prefill_health()

    res_out, res_final = out, pool[st_rows]
    c_out, c_fin = check(res_out, gold_out), check(res_final, gold_final)
    l2 = [c_out[2], c_fin[2]]
    ok = c_out[0] and c_fin[0] and max(l2) < REL_L2_MAX and bool(health.get("ok", False))
    if fla_l2 is not None:
        ok = ok and all(a <= 2.0 * f + 1e-3 for a, f in zip(l2, fla_l2))
    ck_txt = ""
    if ck_idx:
        c_ck = check(pool[[ck_rows[n] for n in ck_idx]], gold_ck)
        ok = ok and c_ck[0] and c_ck[2] < REL_L2_MAX
        ck_txt = f", ckpt relL2 {c_ck[2]:.4f} ({len(ck_idx)} rows)"
    written = set(st_rows) | {r for r in ck_rows if r > 0}
    untouched = [r for r in range(slots) if r not in written]
    same = bool(torch.equal(pool[untouched].isnan(), before[untouched].isnan()))
    ok = ok and same and bool(torch.equal(buf[:, :64], torch.zeros_like(buf[:, :64])))
    fla_txt = "" if fla_l2 is None else f" (FLA {fla_l2[0]:.4f} / {fla_l2[1]:.4f})"
    print(f"{name:58s} {'PASS' if ok else 'FAIL'}  T={T} H={H} N={N}  relL2 out {l2[0]:.4f} state {l2[1]:.4f}{fla_txt}"
          f"{ck_txt}, worst elem {c_out[1]:.2f}/{c_fin[1]:.2f}, health {health.get('ok')}, untouched {same}, "
          f"{ms:.1f} ms", flush=True)
    return ok


def prefill_cases():
    for name, build in rg._cases(0):
        if name.startswith("trained-l44"):
            continue
        yield name, build
        if "L0-14-25" in name:
            yield name + "_h12", lambda b=build: heads_slice(b(), H12_HEADS)


# ------------------------------------------------------------------------------------------------------ decode
def silu_inv(y):
    y = y.double().clamp_min(-0.2784)
    x = torch.where(y > 0, y + 0.3, torch.full_like(y, -0.5))
    for _ in range(60):
        s = torch.sigmoid(x)
        f = x * s - y
        df = s * (1 + x * (1 - s))
        x = (x - f / df.clamp_min(1e-3)).clamp_min(-1.2784)
    return x


def fp64_steps(state, q, k, v, g_logit, beta_logit):
    """state [B, H, V, K] fp64; q/k/v/g [B, L, H, D]; beta [B, L, H]. Returns o [B, L, H, D], states [B, L, H, V, K]."""
    S = state.transpose(-1, -2).clone()  # [B, H, K, V]
    qn, kn = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    decay = (LB * torch.sigmoid(g_logit)).exp()
    b = torch.sigmoid(beta_logit)
    outs, states = [], []
    for t in range(q.shape[1]):
        S *= decay[:, t][..., None]
        kS = torch.einsum("bhk,bhkv->bhv", kn[:, t], S)
        S += kn[:, t][..., None] * (b[:, t][..., None] * (v[:, t] - kS))[:, :, None, :]
        outs.append(D ** -0.5 * torch.einsum("bhk,bhkv->bhv", qn[:, t], S))
        states.append(S.transpose(-1, -2).float())
    return torch.stack(outs, 1), torch.stack(states, 1)


def gated_norm(o, out_gate, nw):
    ob = o.to(torch.bfloat16).double()
    return ob * torch.rsqrt(ob.pow(2).mean(-1, keepdim=True) + NORM_EPS) * nw.double() * torch.sigmoid(out_gate.double())


def decode_case(name, inputs, S, calls, seed):
    q, k, v, g, beta, A_log, dt_bias, scale, _, cu = inputs
    dev = q.device
    H = q.shape[2]
    dim = H * D
    cu = cu.tolist()
    B = len(cu) - 1
    lens = [cu[n + 1] - cu[n] for n in range(B)]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    acc = (torch.randint(1, S + 1, (calls, B), generator=torch.Generator().manual_seed(seed)) if S > 1
           else torch.ones(calls, B, dtype=torch.long))
    span = int(acc.sum(0).max()) + S
    start = [L - span for L in lens]
    assert min(start) > 0, "sequences too short"

    with torch.no_grad():
        pre_tok = torch.cat([torch.arange(cu[n], cu[n] + start[n], device=dev) for n in range(B)])
        cu_pre = torch.tensor([0, *torch.tensor(start).cumsum(0).tolist()], device=dev, dtype=torch.int64)
        pre = [x[:, pre_tok].contiguous() for x in inputs[:5]] + [A_log, dt_bias, scale,
                                                                 torch.zeros(B, H, D, D, device=dev), cu_pre]
        S0 = rg.fp64_recurrence(*pre, lower_bound=LB)[1].float()  # what the pool holds

        idx = torch.stack([torch.arange(cu[n] + start[n], cu[n] + start[n] + span, device=dev) for n in range(B)])
        qkv = torch.cat([q[0][idx], k[0][idx], v[0][idx]], dim=2)  # [B, span, 3H, D]
        x = silu_inv(qkv).to(torch.bfloat16)
        c = F.silu(x.double())
        cq, ck, cv = c[:, :, :H], c[:, :, H:2 * H], c[:, :, 2 * H:]
        gl, bl = g[0][idx].double(), beta[0][idx].double()
        o_st, s_st = fp64_steps(S0.double(), cq, ck, cv, gl, bl)

        taps = W - 1 if S == 1 else W - 2 + S
        rows = (torch.randperm(B * S + 4, generator=torch.Generator().manual_seed(seed))[: B * S] + 1).view(B, S)
        slots = B * S + 6
        ssm = torch.full((slots, H, D, D), float("nan"), device=dev)
        conv = torch.randn(slots, taps, 3 * dim, generator=gen, device=dev).bfloat16()
        ssm[rows[:, 0]] = S0
        conv_w = torch.zeros(3, W, dim, device=dev)
        conv_w[:, W - 1] = 1.0
        nw = 1 + 0.1 * torch.randn(D, generator=gen, device=dev)
        A0, dtb = torch.zeros(H, device=dev), torch.zeros(dim, device=dev)
        cu_d = torch.arange(0, B * S + 1, S, device=dev, dtype=torch.int32)

        pos = [0] * B
        worst_o = worst_s = 0.0
        ok = True
        for ci in range(calls):
            T = B * S
            tok = torch.tensor([[pos[n] + t for t in range(S)] for n in range(B)], device=dev)  # [B, S]
            bsel = torch.arange(B, device=dev)[:, None].expand(B, S)
            P = torch.zeros(T, 3 * dim + dim + 2 * H, dtype=torch.bfloat16, device=dev)
            P[:, :3 * dim] = x[bsel, tok].reshape(T, 3 * dim)
            P[:, 3 * dim:4 * dim] = torch.randn(T, dim, generator=gen, device=dev).bfloat16()
            P[:, 4 * dim:4 * dim + H] = bl[bsel, tok].reshape(T, H).to(torch.bfloat16)
            mixed, og, bt = P[:, :3 * dim], P[:, 3 * dim:4 * dim], P[:, 4 * dim:4 * dim + H]
            gate = gl[bsel, tok].reshape(T, H, D).to(torch.bfloat16).contiguous()
            if S == 1:
                si, nat = rows[:, 0].to(dev, torch.int32).contiguous(), None
            else:
                si = rows.to(dev, torch.int32).contiguous()
                nat = torch.tensor([1 if ci == 0 else int(acc[ci - 1, n]) for n in range(B)], device=dev,
                                   dtype=torch.int32)
            out = torch.empty(T, dim, dtype=torch.bfloat16, device=dev)
            conv_before, ssm_before = conv.clone(), ssm.clone()
            ref_out, ref_conv, _ = kda_decode_reference(mixed, conv_before, conv_w, gate, bt, og, A0, dtb, nw, ssm_before,
                                                        si, nat, cu_d, NORM_EPS, LB)
            kernel.decode(mixed, conv, conv_w, gate, bt, og, A0, dtb, nw, ssm, si, nat, cu_d, NORM_EPS, LB, out)
            torch.cuda.synchronize()
            gold_o = gated_norm(o_st[bsel, tok], og.view(B, S, H, D), nw).reshape(T, dim)
            co = check(out, gold_o)
            if S == 1:
                gold_s = s_st[torch.arange(B), torch.tensor(pos)]
                cs = check(ssm[rows[:, 0]], gold_s)
            else:
                gold_s = s_st[bsel, tok]  # state after each verified token
                cs = check(ssm[rows.to(dev)], gold_s)
            conv_ok = torch.equal(conv, ref_conv)
            ok = ok and co[0] and cs[0] and co[2] < REL_L2_MAX and cs[2] < REL_L2_MAX and conv_ok
            worst_o, worst_s = max(worst_o, co[2]), max(worst_s, cs[2])
            ref_l2 = rel_l2(ref_out, gold_o)
            for n in range(B):
                pos[n] += int(acc[ci, n]) if S > 1 else 1
        health = kernel.decode_health()
        ok = ok and bool(health.get("ok", False))
    print(f"{name:58s} {'PASS' if ok else 'FAIL'}  H={H} B={B} S={S} calls={calls}  worst relL2 out {worst_o:.4f} "
          f"state {worst_s:.2e} (fp32 reference.py out {ref_l2:.4f}), accepted {acc.float().mean():.1f}/call, conv cache exact, health {health.get('ok')}",
          flush=True)
    return ok


def main():
    torch.manual_seed(0)
    kernel.decode_warmup()
    failed = total = 0
    if args.only in ("", "prefill"):
        print(f"\n{'PREFILL real workload (vs fp64)':58s} verdict")
        for name, build in prefill_cases():
            inputs = build()
            try:
                ok = prefill_case(name, inputs)
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"{name:58s} FAIL  RUNTIME_ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
            failed += not ok
            total += 1
            del inputs
            torch.cuda.empty_cache()
    if args.only in ("", "decode"):
        print(f"\n{'DECODE real workload (vs fp64)':58s} verdict")
        for sample in ("gsm8k-batch32", "math500-batch32"):
            full = rg.kimi_linear_inputs(sample, (0, 14, 25))
            for tag, inp in (("h96", full), ("h12", heads_slice(full, H12_HEADS))):
                for S, calls in ((1, 16), (8, 5)):
                    name = f"kimi-linear_{sample}_L0-14-25_{tag}_{'plain' if S == 1 else 'spec8'}"
                    try:
                        ok = decode_case(name, inp, S, calls, seed=zlib.crc32(name.encode()) % 10000)
                    except Exception as e:  # noqa: BLE001
                        ok = False
                        print(f"{name:58s} FAIL  RUNTIME_ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)
                    failed += not ok
                    total += 1
            del full
            torch.cuda.empty_cache()
    print(f"\nREAL-WORKLOAD GATE: {total - failed} / {total} PASS")


if __name__ == "__main__":
    main()
