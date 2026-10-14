"""Real-workload correctness gate for the KDA forward eval (not timed, not scored; any failure scores 0).

Two sets, each checked against a token-by-token fp64 recurrence of the operator:

* trained-l44: synthetic inputs drawn with the statistics of a trained Kimi-K3 deep layer (layer 44): per-head
  decay rate exp(A_log) near 1, deep negative dt_bias, beta logits up to sigmoid ~0.99, q / k / v sharing a
  direction inside a chunk (mean pairwise cos 0.90 / 0.93 / 0.80), nonzero initial state with sparse peaks.
  Fresh seed every run. Isotropic randn inputs leave the intra-chunk Gram matrix near diagonal; this regime
  does not (it is the FlashInfer CAKE KDA acceptance fixture, tests/kda/test_kda_prefill_trained_gate_distribution.py).
* kimi-linear: activations captured at the input of the KDA layers of Kimi-Linear-48B-A3B-Instruct prefilling
  GSM8K / MATH-500 prompts (public dataset humanfia-lab/kda-datasets, folder kda-forward/; single 1.6k-8k-token
  requests and 32-request packed batches), converted to this ABI (A_log = dt_bias = 0,
  g = logit(clamp(-g / 5)), beta logits = logit(beta), layers stacked along heads), plus one case whose initial
  state is the fp64 state of the first half of the 8k request. Real gates decay far deeper than the timed
  workloads' (p99 at the lower bound, ~231 bits per 32-token chunk).

A case passes when output and final state are finite, every element is within the eval's element-wise
tolerance of the fp64 result, relative L2 vs fp64 < 1e-2, and relative L2 <= 2x FLA chunk_kda's + 1e-3.
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

D = 128
EPS = 1e-6
REL_L2_MAX = 1e-2
DATA = Path(os.environ.get("KDA_REAL_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
REAL_CASES = [(s, ls) for s in ("math500-multi-8192", "math500-multi-4096", "gsm8k-8shot", "math500-batch32",
                                "gsm8k-batch32") for ls in ((0, 14, 25), (5, 10, 18))] + [
    ("math500-multi-8192", (10, 18)), ("gsm8k-batch32", (0, 25))]
CARRY = ("math500-multi-8192", (0, 14, 25), 4096)
TRAINED_CASES = [
    ((64,), 12), ((128,), 12), ((64,) * 64, 12), ((128,) * 64, 12), ((2048,), 12), ((2241,), 12), ((1024,) * 4, 12),
    ((8192,), 96), ((1300, 547, 2048, 963, 271, 3063), 96), ((1024,) * 8, 64), ((128,) * 64, 64),
]
_A_LOG_PER_HEAD = (0.0089, 0.4862, 0.0643, 0.2757, -0.0391, 0.1571, 0.6433, 0.4869, 0.1956, 0.3648, 0.7447, 0.5918)
_BETA_LOGIT_MEAN_PER_HEAD = (3.65, -1.82, -2.68, -1.39, 2.99, -0.22, -5.11, -4.32, -0.41, -3.29, -5.56, -7.12)
_DT_BIAS_QUANTILES = ((0.000, -8.2005), (0.001, -7.9175), (0.010, -7.1049), (0.100, -6.5265), (0.500, -4.6909),
                      (0.900, -2.8283), (0.990, -1.9639), (0.999, -1.3914), (1.000, -1.2937))
_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I32": torch.int32, "I64": torch.int64}


def _load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = bytearray(f.read())
    out = {}
    for name, m in header.items():
        if name == "__metadata__":
            continue
        s, e = m["data_offsets"]
        dt = _DT[m["dtype"]]
        out[name] = torch.frombuffer(buf, dtype=dt, count=(e - s) // dt.itemsize, offset=s).reshape(m["shape"]).cuda()
    return out


def kimi_linear_inputs(sample, layers):
    ls = [_load_st(DATA / sample / f"layer{i:02d}.safetensors") for i in layers]

    def cat(name):
        return torch.cat([x[name] for x in ls], dim=2)

    q, k, v = cat("q").contiguous(), cat("k").contiguous(), cat("v").contiguous()
    g = torch.logit((-cat("g").float() / 5.0).clamp(EPS, 1 - EPS)).to(torch.bfloat16).contiguous()
    beta = torch.logit(cat("beta").float().clamp(EPS, 1 - EPS)).to(torch.bfloat16).contiguous()
    H = q.shape[2]
    cu = ls[0].get("cu_seqlens")
    cu = cu.to(torch.int64).contiguous() if cu is not None and cu.numel() > 2 else None
    N = 1 if cu is None else cu.numel() - 1
    return [q, k, v, g, beta, torch.zeros(H, device="cuda"), torch.zeros(H * D, device="cuda"), 1.0 / math.sqrt(D),
            torch.zeros(N, H, D, D, device="cuda"), cu]


def trained_inputs(lengths, heads, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lengths)
    shape = (1, T, heads, D)

    def randn(*size, scale=1.0, shift=0.0):
        return torch.randn(size, generator=gen, device="cuda") * scale + shift

    def correlated(rms, cos):
        spread = math.sqrt((1.0 - cos) / (cos * D))
        x = F.normalize(randn(1, 1, heads, D), dim=-1) + spread * randn(*shape)
        return (x * (rms * math.sqrt(D) / math.sqrt(1.0 + D * spread * spread))).bfloat16()

    def per_head(vals):
        return torch.tensor([vals[h % len(vals)] for h in range(heads)], dtype=torch.float32, device="cuda")

    q, k = correlated(0.09, 0.90), correlated(0.22, 0.93)
    v = correlated(0.06, 0.80).clamp(-1.0, 1.0)
    g = randn(*shape, scale=0.88, shift=-0.62).bfloat16()
    beta = (randn(1, T, heads, scale=1.5) + per_head(_BETA_LOGIT_MEAN_PER_HEAD)).bfloat16()
    ps = torch.tensor([p for p, _ in _DT_BIAS_QUANTILES], device="cuda")
    vs = torch.tensor([x for _, x in _DT_BIAS_QUANTILES], device="cuda")
    prob = torch.linspace(0.0, 1.0, heads * D, device="cuda")
    up = torch.searchsorted(ps, prob, right=True).clamp_(1, len(ps) - 1)
    lo = up - 1
    dt = vs[lo] + (prob - ps[lo]) / (ps[up] - ps[lo]) * (vs[up] - vs[lo])
    dt_bias = dt[torch.randperm(heads * D, generator=gen, device="cuda")].contiguous()
    state = randn(len(lengths), heads, D, D, scale=0.014)
    peaks = torch.rand(state.shape, generator=gen, device="cuda") < 2e-4
    state = torch.where(peaks, randn(*state.shape, scale=0.5), state).contiguous()
    cu = None
    if len(lengths) > 1:
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int64)
    return [q, k, v, g, beta, per_head(_A_LOG_PER_HEAD).contiguous(), dt_bias, 1.0 / math.sqrt(D), state, cu]


def fp64_recurrence(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, lower_bound=-5.0):
    """reference.py's recurrence with fp64 results kept, all sequences stepped together. States are [N, H, V, K]."""
    H, T = q.shape[2], q.shape[1]
    qn = F.normalize(q[0].double(), dim=-1, eps=1e-6)
    kn = F.normalize(k[0].double(), dim=-1, eps=1e-6)
    vf = v[0].double()
    decay = (lower_bound * torch.sigmoid(A_log.double().exp().view(H, 1) * (g[0].double() + dt_bias.double().view(H, D)))).exp()
    b = torch.sigmoid(beta[0].double())
    cu = [0, T] if cu_seqlens is None else cu_seqlens.tolist()
    lens = [e - s for s, e in zip(cu[:-1], cu[1:])]
    order = sorted(range(len(lens)), key=lambda n: -lens[n])  # active sequences at step t are a prefix
    starts = torch.tensor([cu[n] for n in order], device=q.device)
    S = initial_state[order].double().transpose(-1, -2).contiguous()  # [N, H, K, V]
    out = torch.empty_like(vf)
    sorted_lens = [lens[n] for n in order]
    m = len(order)
    for t in range(max(lens, default=0)):
        while m and sorted_lens[m - 1] <= t:
            m -= 1
        tok = starts[:m] + t
        Sa = S[:m]
        Sa *= decay[tok][..., None]
        kt = kn[tok]
        kS = torch.einsum("nhk,nhkv->nhv", kt, Sa)
        Sa += kt[..., None] * (b[tok][..., None] * (vf[tok] - kS))[:, :, None, :]
        out[tok] = scale * torch.einsum("nhk,nhkv->nhv", qn[tok], Sa)
    final = torch.empty(len(lens), H, D, D, dtype=torch.float64, device=q.device)
    final[order] = S.transpose(-1, -2)
    return out[None], final


def _rel_l2(x, y):
    return float(torch.linalg.vector_norm(x.double() - y) / torch.linalg.vector_norm(y).clamp_min(1e-30))


def _cases(seed):
    for i, (lengths, heads) in enumerate(TRAINED_CASES):
        tag = f"{len(lengths)}x{lengths[0]}" if len(set(lengths)) == 1 else f"mixed{len(lengths)}"
        yield f"trained-l44_h{heads}_{tag}", lambda lengths=lengths, heads=heads, i=i: trained_inputs(lengths, heads, seed * 131 + i)
    for sample, layers in REAL_CASES:
        yield f"kimi-linear_{sample}_L{'-'.join(map(str, layers))}", lambda s=sample, ls=layers: kimi_linear_inputs(s, ls)
    sample, layers, cut = CARRY

    def carry():
        full = kimi_linear_inputs(sample, layers)
        first = [x[:, :cut].contiguous() if i < 5 else x for i, x in enumerate(full)]
        second = [x[:, cut:].contiguous() if i < 5 else x for i, x in enumerate(full)]
        second[8] = fp64_recurrence(*first)[1].float().contiguous()
        return second
    yield f"kimi-linear_{sample}_L{'-'.join(map(str, layers))}_state@{cut}", carry


def run_real_gate(module, seed, make_launch, check, fla_chunk_kda, lower_bound):
    """Returns the number of failed cases, or None when the captured data is not available."""
    if not (DATA / CARRY[0] / "layer00.safetensors").exists():
        print(f"\nREAL-WORKLOAD GATE: captured data not found at {DATA}; skipped (not an official result)")
        return None
    print(f"\n{'real workload (vs fp64 recurrence)':52s} correctness")
    failed = 0
    for name, build in _cases(seed):
        inputs = build()
        with torch.no_grad():
            gold = fp64_recurrence(*inputs, lower_bound=lower_bound)
            fla = fla_chunk_kda(False, *[x.clone() if torch.is_tensor(x) else x for x in inputs])
            fla_l2 = [_rel_l2(x, y) for x, y in zip(fla, gold)]
            try:
                res = make_launch(module, [x.clone() if torch.is_tensor(x) else x for x in inputs])()
                torch.cuda.synchronize()
                res = (res,) if torch.is_tensor(res) else tuple(res)
                if len(res) != 2:
                    raise ValueError("no final_state")
                l2 = [_rel_l2(x, y) for x, y in zip(res, gold)]
                elem = [check(x, y.float()) for x, y in zip(res, gold)]
                finite = all(bool(torch.isfinite(x.float()).all()) for x in res)
                ok = (finite and all(e[0] for e in elem) and max(l2) < REL_L2_MAX
                      and all(a <= 2.0 * f + 1e-3 for a, f in zip(l2, fla_l2)))
                detail = (f"relL2 out {l2[0]:.4f} state {l2[1]:.4f} (FLA {fla_l2[0]:.4f} / {fla_l2[1]:.4f}), "
                          f"worst elem {elem[0][1]:.2f} / {elem[1][1]:.2f}{'' if finite else ', NON-FINITE'}")
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"RUNTIME_ERROR {type(e).__name__}: {str(e)[:300]}"
        failed += not ok
        print(f"{name:52s} {'PASS' if ok else 'FAIL'} ({detail})", flush=True)
        del inputs, gold, fla
        torch.cuda.empty_cache()
    return failed
