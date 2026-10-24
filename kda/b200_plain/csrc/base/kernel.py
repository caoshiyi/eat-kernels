"""KDA forward for B200: fused chunked kernels (FlashInfer-snapshot CAKE M64/M128 schedules, fp32 state I/O).

cake_v7 = cake_v6 kernels + split_tail() in prepare(): when the LPT packing leaves the busiest CTA above the
  others, whole tasks are cut into a front piece (runs first on a lightly loaded CTA, hands its fp32 state to the next
  piece through an L2 scratch slot + release flag) and a back piece (runs last on another CTA), priced by the measured
  piece-boundary cost; only used with the overlapped-I/O kernel, which implements the pieces.
cake_v6 = cake_v5 + (gen/gen.sh EXTRA="vmma wpx2 rst2 rbal", EXTRA64="wpx2 rst2 rbal"):
  - vmma (M128): the residual U = S.kd - v is formed on the tensor core (two extra SS MMAs, A = the v tile as an MN-major
    SW128 operand, B = a constant -I); the chain step C2 is then u = -beta*U without per-token smem loads of v;
  - wpx2: work pass with packed f32x2 math and vector smem accesses;
  - rst2: integer midpoint exponent c = rint(lb*log2e*16) for the in-chunk decay products, so the qd/kd restore is an exact
    power-of-two bf16x2 multiply on the packed values;
  - rbal: restore of all 32 rows on prep warps 2/3 (warp 1 only does its inverse step).

Three kernels, chosen per call by the makespan model in plan():
  - M64: two CTAs per (sequence, head), value rows split (wins when there are few, long tasks, e.g. one 8k sequence
    at H=64 on 148 SMs);
  - persistent M128 (cake/): LPT-packed (sequence, head) tasks, one or more per CTA;
  - persistent M128 with overlapped state I/O (cake_cio/): double-buffered TMEM state, the next task's initial state is
    staged in and the previous task's final state written out during the current task.  It costs a little per chunk
    and saves ~15k cycles per task boundary, so it is used when the plan puts more than one task on some CTA.

ABI: run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None) -> (output, final_state)
     prepare(same args) -> launch()   (planning + descriptor encoding outside the timed region)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cake"))
_ext = None
_ext_cio = None


def _module_cio():
    global _ext_cio
    if _ext_cio is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kda_cio_build", os.path.join(_HERE, "cake_cio", "build.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ext_cio = mod.build()
    return _ext_cio


_ext_ppc = None


def _module_ppc():
    global _ext_ppc
    if _ext_ppc is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kda_ppc_build", os.path.join(_HERE, "cake_ppc", "build.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ext_ppc = mod.build()
    return _ext_ppc


def _module():
    global _ext
    if _ext is None:
        from build import build
        _ext = build()
    return _ext


def _num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _makespan(costs, n_sms):
    """Greedy list scheduling of CTAs in launch order (the hardware dispatches CTAs in blockIdx order)."""
    import heapq
    free = [0.0] * min(n_sms, len(costs))
    heapq.heapify(free)
    end = 0.0
    for c in costs:
        t = heapq.heappop(free) + c
        end = max(end, t)
        heapq.heappush(free, t)
    return end


_LAYOUT_FLAGS = int(open(os.path.join(_HERE, "cake", "layout_flags")).read())  # written by mkvar.sh from the patch set
_CIO_MIN_CHUNKS = 17       # see use_cio in prepare()
# [architect_cio iter 15] modelled cio-over-plain gain needed before the cio kernel replaces the plain plan.  The model over-prices cio: measured
# (v41, eval gen, 3 x A/B, separate processes) a modelled gain of 5.1% (h96-mixed wrap plan) ran +6.0% SLOWER than the plain plan, 5.9% (h64-mixed
# wrap plan) ran 1.8% slower (7.6% on trained gates, 8.6% bursty), 3.7% measured -0.9% earlier; every other layout / held-out shape that uses cio models >= 25%.
_CIO_BAR = 1.10
_CIO_BOUNDARY_SAVING = 0.63  # chunk-steps saved per task boundary vs plain (cake_v22 nslope: plain 4.85 us, cio 3.92 us per task)
_CIO_CHUNK_OVERHEAD = 0.0   # relative per-chunk cost of the overlapped-I/O kernel (cake_v20: 1.436 vs plain 1.439 us/chunk)

# Cost model (units: one 32-token chunk step of an M128 CTA, measured ~3.4k cycles on B200).
_M64_CHUNK_COST = 0.93     # an M64 CTA (half the value rows, full prep) per chunk
_TASK_COST_PERSIST = 1.0   # per task in a persistent M128 CTA (state load/store; pipeline keeps running)
_TASK_COST_LAUNCH = 8.0    # per CTA in the one-task-per-CTA M64 kernel (launch, pipeline fill and drain)
# Refitted to per-CTA end-time stamps (work_analyst/it4/fitplain.py, 9 layouts): the plain M128 kernel pays 1.435 us per chunk and
# 4.58 us per task (its blocking 64 KB state store + load) = 3.2 chunk-steps per task.  Used for the LPT packing and every plain-vs-cio /
# plain-vs-pieces decision; the M64-vs-M128 choice keeps the model it was calibrated with (_TASK_COST_PERSIST).
_PLAIN_TASK = 3.4   # cake_v22 nslope (H=148, N equal tasks/CTA): 4.85 us per task boundary / 1.416 us per chunk
_PLAIN_SHORT = -0.5  # a task shorter than _PRELOAD_MIN costs less at its boundaries (fitplain: -0.7..-1.1 us)


def _plain_task(n):
    return n + _PLAIN_TASK + (_PLAIN_SHORT if n < 17 else 0.0)

# The plain-with-pieces build (cake_ppc) measured on cake_v19 (it4/logs_ppc_v19.txt): +6.5% per chunk vs plain, +5.6 chunk-steps per piece.
_PPC_EPS = 1.10    # the ppc build has neither deferral nor the later epilogue patches (v19: +6.5%; plain got faster since)
_PPC_PIECE = 5.6


def _lpt(costs, n_bins):
    """Longest-processing-time-first assignment; returns (makespan, bins of task indices)."""
    import heapq
    order = sorted(range(len(costs)), key=lambda i: -costs[i])
    heap = [(0.0, b) for b in range(n_bins)]
    bins = [[] for _ in range(n_bins)]
    for i in order:
        load, b = heapq.heappop(heap)
        bins[b].append(i)
        heapq.heappush(heap, (load + costs[i], b))
    return max(l for l, _ in heap), bins


_PACK_MAX_TASKS = 2500   # planning-time bound of the packing refinement (tens of ms)


def _refine_bins(costs, bins, max_iter=2000):
    """[architect_cio] Local search on top of LPT: repeatedly swap a task of the busiest bin with a smaller task of another bin whenever that
    lowers the larger of the two loads (strictly lowers the makespan or the busiest bin; ends when no swap helps).  Deterministic.
    Bins keep the LPT convention (largest task first)."""
    loads = [sum(costs[i] for i in b) for b in bins]
    for _ in range(max_iter):
        k = max(range(len(bins)), key=lambda i: loads[i]); mk = loads[k]
        best = None
        for j in range(len(bins)):
            if j == k:
                continue
            for a in bins[k]:
                for b in bins[j]:
                    d = costs[a] - costs[b]
                    if d > 1e-9 and loads[j] + d < mk - 1e-9:
                        cand = max(loads[j] + d, loads[k] - d)
                        if best is None or cand < best[0]:
                            best = (cand, a, j, b)
        if best is None:
            break
        _, a, j, b = best
        bins[k].remove(a); bins[j].remove(b); bins[k].append(b); bins[j].append(a)
        d = costs[a] - costs[b]; loads[k] -= d; loads[j] += d
    return [sorted(b, key=lambda i: -costs[i]) for b in bins]


def plan(seq_lens, H, n_sms):
    """Choose between the persistent M128 kernel (tasks = (sequence, head), LPT-packed onto <= n_sms CTAs) and the
    M64 kernel (two CTAs per task, value rows split, hardware-scheduled waves) by modelled makespan."""
    chunks = [(L + 31) // 32 for L in seq_lens]
    tasks = [(n, h) for n in range(len(seq_lens)) for h in range(H)]
    costs = [chunks[n] + _TASK_COST_PERSIST for n, _ in tasks]
    n_cta = max(1, min(n_sms, len(tasks)))
    m128, _ = _lpt(costs, n_cta)
    pcosts = [_plain_task(chunks[n]) for n, _ in tasks]
    _, bins = _lpt(pcosts, n_cta)   # packing with the refitted per-task cost
    if len(tasks) <= _PACK_MAX_TASKS:
        bins = _refine_bins(pcosts, bins)
    order = sorted(range(len(seq_lens)), key=lambda i: -seq_lens[i])
    m64_tasks = [chunks[i] * _M64_CHUNK_COST + _TASK_COST_LAUNCH for i in order for _ in range(2 * H)]
    m64 = _makespan(m64_tasks, n_sms)
    if m64 < m128:
        return 64, order, None
    task_list, cta_ptr = [], [0]
    for b in bins:
        task_list += [tasks[i][0] * H + tasks[i][1] for i in b]
        cta_ptr.append(len(task_list))
    return 128, order, (task_list, cta_ptr)


_SPLIT_BOUNDARY = 4.0  # chunk-steps per blocking piece boundary (fp32 state to / from L2 scratch), measured in iteration 1


def split_tail(task_chunks, bins, n_cta, boundary=_SPLIT_BOUNDARY):
    """Even out an LPT packing by splitting whole tasks into a front piece and a back piece with exact state handoff.

    The busiest CTA gives up its smallest whole task t; its first s chunks run FIRST on the least-loaded CTA x (front piece,
    writes its fp32 state to an L2 scratch slot and releases a flag) and the remaining chunks run LAST on the next
    least-loaded CTA y (back piece, waits for the flag and loads the scratch state).  A split is kept only if the busiest
    CTA's modelled load drops without creating a new maximum (each piece pays `boundary` chunk-steps) and the back piece's modelled start is after the front
    piece's modelled end (so a co-resident persistent grid cannot deadlock).  Returns per-CTA lists of
    (task, c0, c1, in_slot, out_slot) and the number of scratch slots.
    """
    cost = [c + 1.0 for c in task_chunks]
    whole = [list(b) for b in bins]
    front = [[] for _ in range(n_cta)]  # (task, c0, c1, slot)
    back = [[] for _ in range(n_cta)]
    load = [sum(cost[t] for t in b) for b in whole]
    nslot = 0
    for _ in range(4 * n_cta):
        k = max(range(n_cta), key=lambda i: load[i])
        if not whole[k]:
            break
        cand = [i for i in whole[k] if task_chunks[i] >= _PRELOAD_MIN]   # tasks shorter than the preload window are never cut
        if not cand:                                                   # (their pieces would pay blocking handoffs; stamps)
            break
        t = min(cand, key=lambda i: cost[i])
        n = task_chunks[t]
        order = sorted((i for i in range(n_cta) if i != k), key=lambda i: load[i])
        if len(order) < 2:
            break
        x, y = min(order[0], order[1]), max(order[0], order[1])  # front piece on the lower CTA index (dispatched first)
        s = int(round((load[y] - load[x] + n) / 2.0))
        s = max(1, min(n - 1, s))
        new_x, new_y = load[x] + s + 1.0 + boundary, load[y] + (n - s) + 1.0 + boundary
        if max(new_x, new_y, load[k] - cost[t]) >= load[k] - 0.5:  # the busiest CTA's load must strictly drop
            break
        # the front piece's state is written out (and its flag released) while x runs its next task's first 16 chunks
        a_end = sum(e - b + boundary for _, b, e, _ in front[x]) + s + boundary + 17
        if load[y] < a_end + boundary:
            break
        whole[k].remove(t)
        load[k] -= cost[t]
        front[x].append((t, 0, s, nslot))
        back[y].append((t, s, n, nslot))
        load[x], load[y] = new_x, new_y
        nslot += 1
    if max(load) > max(sum(cost[t] for t in b) for b in bins) - 1.0:  # no makespan gain: keep whole tasks
        front = [[] for _ in range(n_cta)]
        back = [[] for _ in range(n_cta)]
        whole = [list(b) for b in bins]
        nslot = 0
    lists = []
    for i in range(n_cta):
        lst = [(t, c0, c1, -1, sl) for t, c0, c1, sl in front[i]]
        lst += [(t, 0, task_chunks[t], -1, -1) for t in whole[i]]
        lst += [(t, c0, c1, sl, -1) for t, c0, c1, sl in back[i]]
        lists.append(lst)
    return lists, nslot


_PRELOAD_MIN = 17        # the cio kernel stages the next task's state over the last 17 chunks of a task (else a blocking load)
_SHORT_FRONT = 3.0       # chunk-steps of blocking state load after a front piece shorter than _PRELOAD_MIN (measured ~10k cycles)


# cio list cost model (chunk-steps of the plain kernel), fitted to per-CTA end-time stamps of the cio kernel over wrap plans of
# 4 layouts (work_analyst/it4/logs_bp1.txt, residual <= 0.4 us): per chunk 1.49 us (= 1.04 plain chunk-steps), per list entry
# 3.31 us (state I/O at the boundary), per front piece +0.68, per back piece +0.11 (its state is preloaded from the scratch slot
# once the front has published), per entry shorter than _PRELOAD_MIN that is followed by another (blocking next load) +1.0.
_CIO_EPS = 1.0     # cake_v20: cio per chunk = plain per chunk (deferral removed most of the cio scale overhead)
_CIO_ENTRY = 2.77   # cake_v22 nslope: 3.92 us per task boundary (u1 stamp fit on v18: 3.31 us)
_CIO_FRONT = 0.48
_CIO_BACK = 0.08
_CIO_SHORT = 0.70
_CIO_BACK_BLOCK = 2.4      # back piece whose predecessor is shorter than _PRELOAD_MIN (no preload: blocking L2 scratch load; v18 fit 3.44 us)
_CIO_SHORT_STORE = 0.2     # entry shorter than _PRELOAD_MIN after another entry (blocking store of the previous state; [64]x128 calibration)


def _entry_cost(n, front, back, short):
    return n * _CIO_EPS + _CIO_ENTRY + (_CIO_FRONT if front else 0.0) + (_CIO_BACK if back else 0.0) + (_CIO_SHORT if short else 0.0)


def _list_cost(lst):
    """Modelled cost (chunk-steps) of one CTA's cio task list [(task, c0, c1, in_slot, out_slot)]."""
    c, prev = 0.0, None
    for i, (_, c0, c1, si, so) in enumerate(lst):
        n = c1 - c0
        c += _entry_cost(n, so >= 0, si >= 0, i + 1 < len(lst) and n < _PRELOAD_MIN)
        if si >= 0 and (prev is None or prev < _PRELOAD_MIN):
            c += _CIO_BACK_BLOCK - _CIO_BACK
        if prev is not None and n < _PRELOAD_MIN:
            c += _CIO_SHORT_STORE
        prev = n
    return c


def wrap_plan(task_chunks, n_cta):
    """McNaughton wrap-around schedule with pieces (preemptive makespan bound): tasks, longest first, fill CTAs from the
    highest index down to a common capacity L; the task that overflows a CTA is cut into a back piece (its last chunks, at the
    END of that CTA) and a front piece (its first chunks, at the START of the next lower CTA, dispatched first).  The front
    piece hands its fp32 state to the back piece through an L2 scratch slot + flag (exact, existing cio piece path).  L is the
    smallest capacity for which every task fits and every back piece starts (modelled) well after its front piece's state is
    published.  Returns (lists, nslot) or None."""
    order = sorted(range(len(task_chunks)), key=lambda i: -task_chunks[i])
    total = sum(_entry_cost(n, False, False, False) for n in task_chunks)
    lo = int(total / n_cta)
    for L4 in range(4 * lo, 4 * lo + 256):   # capacity in quarter chunk-steps
        L = L4 / 4.0
        lists = [[] for _ in range(n_cta)]
        heads = [[] for _ in range(n_cta)]   # front pieces (start of a CTA)
        head_cost = [0.0] * n_cta
        j, load, nslot, ok = n_cta - 1, 0.0, 0, True
        for t in order:
            n = task_chunks[t]
            while True:
                if j < 0:
                    ok = False
                    break
                w = _entry_cost(n, False, False, 0 < n < _PRELOAD_MIN)
                if load + w <= L:
                    lists[j].append((t, 0, n, -1, -1)); load += w
                    break
                prev = lists[j][-1] if lists[j] else (heads[j][-1] if heads[j] else None)   # entry before the back piece on CTA j
                bb = _CIO_BACK if (prev is not None and prev[2] - prev[1] >= _PRELOAD_MIN) else _CIO_BACK_BLOCK
                a = int((L - load - _CIO_ENTRY - bb) / _CIO_EPS)        # chunks of the back piece that fit on CTA j
                b = n - a                                               # front piece on CTA j - 1
                front_cost = _entry_cost(b, True, False, b < _PRELOAD_MIN)
                if n >= _PRELOAD_MIN and a >= 1 and b >= 1 and j >= 1 and front_cost + 16 + 8 <= L - (_entry_cost(a, False, False, False) + bb):
                    lists[j].append((t, b, n, nslot, -1))
                    heads[j - 1].append((t, 0, b, -1, nslot))
                    head_cost[j - 1] = front_cost
                    nslot += 1
                    j -= 1
                    load = front_cost
                    break
                j -= 1                                                   # close CTA j, retry the task on the next one
                load = head_cost[j] if j >= 0 else 0.0
            if not ok:
                break
        if not ok:
            continue
        out = []
        for i in range(n_cta):
            backs = [e for e in lists[i] if e[3] >= 0]
            out.append(heads[i] + [e for e in lists[i] if e[3] < 0] + backs)
        return out, nslot
    return None


def schedule(seq_lens, H, n_sms):
    """Host-side plan: (variant, seq order, persistent sched, use_cio, cio (ids, lists, nslot) or None)."""
    variant, order, sched = plan(seq_lens, H, n_sms)
    # The overlapped-I/O kernel hides a task boundary (~4 chunk-steps, measured) only when the current task has
    # >= _CIO_MIN_CHUNKS chunks, and costs ~2% per chunk (measured).  Use it when, on the busiest CTA of the plan, the
    # hidden boundaries outweigh that overhead and every task is long enough to hide its boundary.
    use_cio = False
    if sched is not None and min((L + 31) // 32 for L in seq_lens) >= _CIO_MIN_CHUNKS:
        loads = []
        for i in range(len(sched[1]) - 1):
            ts = sched[0][sched[1][i]:sched[1][i + 1]]
            loads.append((sum((seq_lens[t // H] + 31) // 32 for t in ts), len(ts)))
        busy_chunks, busy_tasks = max(loads)
        use_cio = (busy_tasks - 1) * _CIO_BOUNDARY_SAVING > _CIO_CHUNK_OVERHEAD * busy_chunks
    # Also take the cio kernel (which supports split pieces) when the split planner's modelled makespan, including the cio kernel's
    # per-chunk overhead, beats the whole-task plan by more than 5% - independent of the boundary-hiding heuristic above.  This is what
    # lets layouts with few tasks per CTA (e.g. two long sequences) use idle SMs.  (5% bar: a 3.7% modelled gain measured -0.9%.)
    if sched is not None and not use_cio:
        n_cta0 = len(sched[1]) - 1
        bins0 = [sched[0][sched[1][i]:sched[1][i + 1]] for i in range(n_cta0)]
        ids0 = sorted(set(sched[0])); pos0 = {t: i for i, t in enumerate(ids0)}
        tch0 = [(seq_lens[t // H] + 31) // 32 for t in ids0]
        whole0 = max(sum(_plain_task(tch0[pos0[t]]) for t in b) for b in bins0)
        lists0, ns0 = split_tail(tch0, [[pos0[t] for t in b] for b in bins0], n_cta0)
        sp0 = max(_list_cost(l) for l in lists0)
        if ns0 > 0 and sp0 * _CIO_BAR < whole0:   # switching from the plain kernel needs a _CIO_BAR modelled gain (short entries are priced loosely)
            use_cio = True
    wrapped = None
    if sched is not None:
        n_ctw = len(sched[1]) - 1
        idw = sorted(set(sched[0]))
        tchw = [(seq_lens[t // H] + 31) // 32 for t in idw]
        wp = wrap_plan(tchw, n_ctw)
        if wp is not None:
            wcost = max(_list_cost(l) for l in wp[0])
            if use_cio:
                binsw = [sched[0][sched[1][i]:sched[1][i + 1]] for i in range(n_ctw)]
                posw = {t: i for i, t in enumerate(idw)}
                ls_, ns_ = split_tail(tchw, [[posw[t] for t in b] for b in binsw], n_ctw)
                base_cost = max(_list_cost(l) for l in ls_)
            else:   # plain persistent kernel: whole tasks, no cio overhead
                pw = {t: i for i, t in enumerate(idw)}
                base_cost = max(sum(_plain_task(tchw[pw[t]]) for t in sched[0][sched[1][i]:sched[1][i + 1]]) for i in range(n_ctw))
            if wcost * (1.02 if use_cio else _CIO_BAR) < base_cost:   # vs plain: _CIO_BAR, as for the split path
                wrapped = (idw, wp[0], wp[1])
                use_cio = True
    cio_plan = None
    if use_cio:
        if wrapped is not None:
            ids, lists, nslot = wrapped
        else:
            n_cta = len(sched[1]) - 1
            bins_t = [sched[0][sched[1][i]:sched[1][i + 1]] for i in range(n_cta)]
            ids = sorted(set(sched[0]))
            pos = {t: i for i, t in enumerate(ids)}
            tchunks = [(seq_lens[t // H] + 31) // 32 for t in ids]
            lists, nslot = split_tail(tchunks, [[pos[t] for t in b] for b in bins_t], n_cta)
        cio_plan = (ids, lists, nslot)
    return variant, order, sched, use_cio, cio_plan


def plain_pieces(seq_lens, H, sched):
    """split_tail refinement of the plain kernel's whole-task LPT plan; returns (ids, lists, nslot) or None (keep whole tasks)."""
    n_ct = len(sched[1]) - 1
    ids = sorted(set(sched[0])); pos = {t: i for i, t in enumerate(ids)}
    tch = [(seq_lens[t // H] + 31) // 32 for t in ids]
    bins = [[pos[t] for t in sched[0][sched[1][i]:sched[1][i + 1]]] for i in range(n_ct)]
    whole = max(sum(_plain_task(tch[t]) for t in b) for b in bins)
    lists, nslot = split_tail(tch, bins, n_ct)
    if nslot == 0:
        return None
    cost = max(sum((c1 - c0) * _PPC_EPS + _PLAIN_TASK + (_PPC_PIECE if (si >= 0 or so >= 0) else 0.0) for _, c0, c1, si, so in l) for l in lists)
    if cost * 1.02 < whole:
        return ids, lists, nslot
    return None


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, out=None, final_state=None, n_sms=None):
    """out / final_state: optional caller-owned output buffers (shapes of q / [N, H, D, D]); n_sms: SM budget for the plan (CP groups)."""
    B, T, H, D = q.shape
    assert D == 128
    dev = q.device
    if cu_seqlens is None:
        assert B == 1
        cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
        seq_lens = [T]
    else:
        cu = cu_seqlens if cu_seqlens.dtype == torch.int64 else cu_seqlens.to(torch.int64)
        cl = cu.tolist()
        seq_lens = [cl[i + 1] - cl[i] for i in range(len(cl) - 1)]
    N = len(seq_lens)
    ext = _module()
    variant, order, sched, use_cio, cio_plan = schedule(seq_lens, H, n_sms if n_sms is not None else _num_sms(dev))
    seq_order = torch.tensor(order, dtype=torch.int32, device=dev)
    pp = None if (sched is None or use_cio) else plain_pieces(seq_lens, H, sched)
    if sched is None:
        task_list = cta_ptr = torch.zeros(1, dtype=torch.int32, device=dev)
    elif pp is None:
        task_list = torch.tensor(sched[0], dtype=torch.int32, device=dev)
        cta_ptr = torch.tensor(sched[1], dtype=torch.int32, device=dev)
    else:   # plain kernel with pieces: (task, c0, c1, slots) entries + L2 handoff scratch / flags
        ext_pp = _module_ppc()
        pp_ids, pp_lists, p_slot = pp
        flat, ptr = [], [0]
        for lst in pp_lists:
            for ti, c0, c1, si, so in lst:
                flat += [pp_ids[ti], c0, c1, (si + 1) | ((so + 1) << 16)]
            ptr.append(len(flat) // 4)
        task_list = torch.tensor(flat, dtype=torch.int32, device=dev)
        cta_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
        p_scratch = torch.zeros(p_slot * 128 * 128, dtype=torch.float32, device=dev)
        p_flags = torch.zeros(p_slot, dtype=torch.int32, device=dev)
    if out is None:
        out = torch.empty_like(q)
    if final_state is None:
        final_state = torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    assert out.is_contiguous() and final_state.is_contiguous()
    Hp = (H + 7) // 8 * 8
    Tp = max(T, 32)
    beta2 = beta.reshape(T, H)
    beta_pad = None
    if Hp != H or Tp != T:
        beta_pad = torch.zeros(Tp, Hp, dtype=beta.dtype, device=dev)
        beta_tma = beta_pad
    else:
        beta_tma = beta2
    maps = ext.encode_maps(q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D), beta_tma,
                           out.view(T, H, D), variant, _LAYOUT_FLAGS).to(dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    s0 = initial_state.contiguous()
    s0_src = None
    if s0.data_ptr() % 16 != 0:   # [flow] cio cp.async state pieces need 16 B alignment: stage a misaligned view in an aligned buffer (re-copied every launch)
        s0_src, s0 = s0, torch.empty_like(s0)
    sc, lb = float(scale), -5.0

    if use_cio:
        cio = _module_cio()
        ids, lists, nslot = cio_plan
        flat, ptr = [], [0]
        for lst in lists:
            for ti, c0, c1, si, so in lst:
                flat += [ids[ti], c0, c1, (si + 1) | ((so + 1) << 16)]
            ptr.append(len(flat) // 4)
        desc = torch.tensor(flat, dtype=torch.int32, device=dev)
        cta_ptr = torch.tensor(ptr, dtype=torch.int32, device=dev)
        dummy_f = torch.zeros(max(1, nslot) * 128 * 128, dtype=torch.float32, device=dev)  # state handoff scratch
        dummy_i = torch.zeros(max(1, nslot), dtype=torch.int32, device=dev)                 # handoff flags (consumer resets)

    def launch():
        if s0_src is not None:
            s0.copy_(s0_src)
        if beta_pad is not None:
            beta_pad[:T, :H].copy_(beta2)
        if use_cio:
            cio.launch(q, k, v, g, beta, A, dtb, cu, seq_order, s0, out, final_state, maps, desc, cta_ptr, dummy_f, dummy_i, H, sc, lb)
        else:
            if pp is not None:
                ext_pp.launch(variant, q, k, v, g, beta, A, dtb, cu, seq_order, s0, out, final_state, maps, task_list, cta_ptr, p_scratch, p_flags, N, H, sc, lb)
            else:
                ext.launch(variant, q, k, v, g, beta, A, dtb, cu, seq_order, s0, out, final_state, maps, task_list, cta_ptr, N, H, sc, lb)
        return out, final_state

    return launch


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    return prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)()
