"""[architect4] DUAL-CHAIN launch for the CP tail: ONE launch runs, per (sequence, head) task, two chains over one shared prep:
chain L (v, S0 = s0L) -> outL, fsL and chain T (v treated as 0, S0 = s0T) -> outT, fsT. Kernel: cake_dual/ (plain M128 with
chain T in TMEM cols 256-511, in-place decay scale fused into C1). Per chain bitwise identical to the plain M128 kernel.
Whole tasks, LPT over the given CTA budget (no pieces)."""
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None
_LAYOUT_FLAGS = int(open(os.path.join(_HERE, "cake", "layout_flags")).read())


def _module():
    global _ext
    if _ext is None:
        spec = importlib.util.spec_from_file_location("kda_dual_build", os.path.join(_HERE, "cake_dual", "build.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ext = mod.build()
    return _ext


def lpt_lists(costs, n):
    """LPT: task ids per bin, bins in decreasing-load order."""
    import heapq
    heap = [(0.0, b) for b in range(max(1, n))]
    lists = [[] for _ in range(max(1, n))]
    for t in sorted(range(len(costs)), key=lambda i: -costs[i]):
        load, b = heapq.heappop(heap)
        lists[b].append(t)
        heapq.heappush(heap, (load + costs[t], b))
    return [l for l in lists if l]


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, s0L, s0T, outL, outT, fsL, fsT, n_ctas, l_zero=False, t_eye=False, pieces=None, piece_cost=(1.40, 6.4)):
    """q..beta: [1, T, H, 128] (one sequence); s0L/s0T/fsL/fsT: [1, H, 128, 128] fp32; outL/outT: [1, T, H, 128] bf16.
    l_zero / t_eye: the caller guarantees s0L == 0 / s0T == I (CP tail): the kernel then writes them into TMEM instead of loading."""
    B, T, H, D = q.shape
    assert B == 1 and D == 128
    dev = q.device
    ext = _module()
    if pieces is None:
        pieces = [T]
    offs = [0]
    for c in pieces[:-1]:
        offs.append(offs[-1] + 32 * c)
    offs.append(T)
    m = len(pieces)
    cu = torch.tensor(offs, dtype=torch.int64, device=dev)
    order = torch.zeros(m, dtype=torch.int32, device=dev)
    costs = [piece_cost[0] * c + piece_cost[1] for c in pieces]
    tc = [(pi * H + h, costs[pi]) for pi in range(m) for h in range(H)]
    ll = lpt_lists([c for _, c in tc], n_ctas)
    lists = [[tc[t][0] for t in l] for l in ll]
    task_list = torch.tensor([t for l in lists for t in l], dtype=torch.int32, device=dev)
    ptr = [0]
    for l in lists:
        ptr.append(ptr[-1] + len(l))
    cta_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
    Hp = (H + 7) // 8 * 8
    Tp = max(T, 32)
    beta2 = beta.reshape(T, H)
    beta_pad = None
    if Hp != H or Tp != T:
        beta_pad = torch.zeros(Tp, Hp, dtype=beta.dtype, device=dev)
        beta_tma = beta_pad
    else:
        beta_tma = beta2
    for t in (outL, outT, fsL, fsT, s0L, s0T):
        assert t.is_contiguous()
    maps = ext.encode_maps(q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D), beta_tma,
                           outL.view(T, H, D), outT.view(T, H, D), 128, _LAYOUT_FLAGS).to(dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    sc, lb = float(scale), -5.0

    def launch():
        if beta_pad is not None:
            beta_pad[:T, :H].copy_(beta2)
        ext.launch(q, k, v, g, beta, A, dtb, cu, order, s0L, outL, fsL, s0T, outT, fsT, maps, task_list, cta_ptr, H, sc, lb,
                   0 if l_zero else 1, 0 if t_eye else 1)

    return launch
