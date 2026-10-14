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
PLAN_DBG = None   # tools: int64 [16] device tensor receiving the lean plain planner's globaltimer stamps
_feat = {}   # [contract] KDA_FEAT builds of the plain / cio kernels (state pool, source pool, prefix-cache checkpoints), built on first use


def _build_dir(sub, name, feat, ck=False, bd=False, rw=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, sub, "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(feat=True, ck=ck, bd=True) if (feat and bd) else mod.build(feat=True, ck=ck, rw=True) if (feat and rw) else mod.build(feat=True, ck=ck) if feat else (mod.build(bd=True) if bd else mod.build())


def _feat_module(kind, sub, ck, rw=False, bd=False):
    """KDA_FEAT build (pool / source-pool addressing) of a kernel, or KDA_FEAT + KDA_CK (also prefix-cache checkpoints: the prep rebase hook and the
    checkpoint store sit in the chunk loops, so they are compiled only when a checkpoint buffer is given)."""
    key = (kind, bool(ck), bool(rw), bool(bd))
    if key not in _feat:
        _feat[key] = _build_dir(sub, f"kda_{kind}_build_f{int(bool(ck))}{'r' if rw else ''}{'b' if bd else ''}", True, bool(ck), bd=bool(bd), rw=bool(rw))
    return _feat[key]


def _module_cio(feat=False, ck=False):
    global _ext_cio
    if feat:
        return _feat_module("cio", "cake_cio", ck)
    if _ext_cio is None:
        _ext_cio = _build_dir("cake_cio", "kda_cio_build", False)
    return _ext_cio


_ext_ppc = None


_DYN_PLAIN_CHUNKS = 17   # [planner_graph] dynamic path: plain kernel (whole tasks, next-task L2 prefetch) when the mean sequence is shorter than this many chunks
_dev_plan = None
_dev_plan_init = set()


_DYN_PLAIN_TASKS_PER_SM = 6.5   # [dynpath] many tasks per SM: a whole-task deal is balanced to ~1 / x and the plain kernel runs ~4-5% faster than cio on whole tasks
_DYN_PLAIN_MANY_CHUNKS = 36   # ... for sequences shorter than this many chunks on average (measured 2-5% gain at 700-1000 tokens x 16-24 sequences, none at 2048)


def _dyn_use_plain(T, N, H, n_sms):
    """Shape-only choice of the dynamic kernel: plain (whole tasks, snake deal) for short mean sequences or many tasks per SM, else cio (wrap plan with cuts)."""
    if N * H <= n_sms:   # [contract2] at most one task per CTA: no task boundary to hide, so the cio kernel's per-chunk overhead (and, with checkpoints, its slower store) is pure loss
        return True
    return T < _DYN_PLAIN_CHUNKS * 32 * N or (N * H >= _DYN_PLAIN_TASKS_PER_SM * n_sms and T < _DYN_PLAIN_MANY_CHUNKS * 32 * N)


_dev_max_seq = {}


def dev_planner_max_seq(dev_index=0):
    """Largest number of sequences the device planner can handle (shared-memory bound of its single CTA)."""
    if dev_index not in _dev_max_seq:
        with torch.cuda.device(dev_index):   # [multi-GPU] max_seq() reads the CURRENT device's opt-in shared memory
            _dev_max_seq[dev_index] = int(_dev_planner(dev_index).max_seq())
    return _dev_max_seq[dev_index]


def _dev_planner(dev_index=0):
    """[planner_graph] device-side planner extension (work_planner_graph/dyn/plan_dev.cu): cu_seqlens -> task tables on the GPU."""
    global _dev_plan
    if _dev_plan is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kda_pg_plan_build", os.path.join(_HERE, "..", "..", "dyn", "build.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _dev_plan = mod.build()
    if dev_index not in _dev_plan_init:   # per-device opt-in of the planner's dynamic shared memory, outside of stream capture
        with torch.cuda.device(dev_index):
            _dev_plan.init()
        _dev_plan_init.add(dev_index)
    return _dev_plan


def _module_ppc(feat=False, ck=False):
    global _ext_ppc
    if feat:
        return _feat_module("ppc", "cake_ppc", ck)
    if _ext_ppc is None:
        _ext_ppc = _build_dir("cake_ppc", "kda_ppc_build", False)
    return _ext_ppc


_ext_bd = None


def _module_bd():
    """[tiny] plain M128 build that reads beta directly (H % 8 != 0: no beta TMA map, no padded copy launch)"""
    global _ext_bd
    if _ext_bd is None:
        _ext_bd = _build_dir("cake", "kda_plain_bd_build", False, bd=True)
    return _ext_bd


_ext_tiny = None


def _module_tiny():
    """[tiny2] bespoke one-chunk kernel (tiny/kda_tiny.cu): one CTA per (sequence, head) task, mma.sync + fp32 register state, no pipeline machinery"""
    global _ext_tiny
    if _ext_tiny is None:
        _ext_tiny = _build_dir("tiny", "kda_tiny_build", False)
    return _ext_tiny


# [tiny2] Cost model of the one-chunk kernel against the persistent plan (us, B200, judge protocol: cold L2, graph replay; fitted on h4 / h96 x {1, 16, 2 x 16, 32} tokens):
# the tiny kernel runs one wave per n_sms tasks (one CTA per SM, 288 threads, 84 KB smem): _TINY_FIX_US launch + start-up and _TINY_WAVE_US[n8 token tiles] per wave (<= 16 tokens: 2 tiles, <= 32: 4);
# the persistent plain kernel pays _MAIN_ENTRY_US once and _plain_task(1) chunk-steps (5.6 us) per task of its busiest CTA.
_TINY_MAX_TOKENS = 32
_TINY_MAX_SEQ = 256        # layout table passed in the kernel parameters (SeqTab)
_TINY_FIX_US = 1.8          # [shortmid] refit with the next-wave L2 prefetch (96 heads x {1..5} x 16 tokens: 7.30 / 12.8 / 18.7 / 23.7 us; 32 tokens: 9.9 / 17.7 / 27.0)
_TINY_WAVE_US = {2: 5.5, 4: 8.3}
_TINY_DYN_FIX_US = 1.6      # dynamic launches (cu_seqlens read on the device) do not know the next wave's layout: no prefetch, the pre-prefetch constants
_TINY_DYN_WAVE_US = {2: 5.9, 4: 8.6}
_MAIN_ENTRY_US = 5.7
_MAIN_CHUNK_US = 1.435      # us per chunk step of the plain kernel (cake_v22 nslope: 1.416-1.439)
_TINY_BAR = 0.97           # the tiny launch must model at least 3% faster than the persistent plan


def tiny_cost_us(seq_lens, H, n_sms):
    """Modelled time of the one-chunk kernel, or None when it does not apply (a sequence longer than one chunk, too many sequences)."""
    N = len(seq_lens)
    if N == 0 or N > _TINY_MAX_SEQ or max(seq_lens) > _TINY_MAX_TOKENS or max(seq_lens) == 0:
        return None
    tiles = 2 if max(seq_lens) <= 16 else 4
    waves = -(-(N * H) // n_sms)
    return _TINY_FIX_US + _TINY_WAVE_US[tiles] * waves


def main_cost_us(seq_lens, H, n_sms):
    """Modelled time of the persistent plain plan for sequences of at most one chunk each (one task = one chunk): entry + the busiest CTA's tasks."""
    per_cta = -(-(len(seq_lens) * H) // n_sms)
    return _MAIN_ENTRY_US + _MAIN_CHUNK_US * _plain_task(1) * per_cta


def tiny_prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, seq_lens, n_sms):
    """[tiny2] Launch of the one-chunk kernel for a static layout (cost-model gated), or None.  The layout (bos, length per sequence) is baked into the kernel parameters:
    a prepared launch is tied to its layout like the host plans.  Inputs are the normalised (contiguous, 16 B aligned) buffers of _prepare_plain; the 32-byte state accesses
    need a 32-byte aligned initial_state."""
    if q.shape[0] != 1 or q.shape[3] != 128 or initial_state.data_ptr() % 32:
        return None
    H = q.shape[2]
    tc = tiny_cost_us(seq_lens, H, n_sms)
    if tc is None or tc >= _TINY_BAR * main_cost_us(seq_lens, H, n_sms):
        return None
    dev = q.device
    N = len(seq_lens)
    bos, c = [], 0
    for L in seq_lens:
        bos.append(c)
        c += L
    out = torch.empty_like(q)
    final_state = torch.empty(N, H, 128, 128, dtype=torch.float32, device=dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    cu = torch.tensor(bos + [c], dtype=torch.int64, device=dev)   # not read by the static kernel (the layout is in the parameters); a valid pointer
    ext = _module_tiny()
    s0 = initial_state
    sc = float(scale)
    lens = list(seq_lens)
    nostat = torch.empty(0, dtype=torch.int32, device=dev)
    wave_n = n_sms if N * H > n_sms else 0   # [shortmid] next-wave L2 prefetch (the inputs and state of the task this SM's successor CTA will run)

    def launch():
        ext.launch(q, k, v, g, beta, A, dtb, cu, s0, out, final_state, N * H, H, sc, -5.0, bos, lens, N, q.shape[1], nostat, wave_n)
        return out, final_state

    return launch


_TINY_DYN_EXTRA_US = 0.5   # the dynamic launch reads cu_seqlens first (one more dependent cold load) and validates the table in every CTA (measured +0.35 us)
_DYN_PLANNER_US = 4.5      # dynamic persistent path = static plan + ~4.5 us (device planner launch, measured: h96 2 x 16 tokens 16.99 -> 21.54 us)


def tiny_dyn_prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu, N, n_sms, status):
    """[tiny2] Dynamic-layout launch of the one-chunk kernel (cu_seqlens read and validated on the device, int64 [N + 1], same N and T on every replay), or None.  Applies when
    the shape guarantees one chunk per sequence (T <= 32 in total), N <= 256 and the cost model beats planner + persistent plan.  A rejected table (must start at 0, be
    non-decreasing, end at <= T) turns the launch into a no-op as in the planner path: every sequence empty (state passes through), the output zero-filled, status bit 1."""
    if q.shape[0] != 1 or q.shape[3] != 128 or initial_state.data_ptr() % 32:
        return None
    T, H = q.shape[1], q.shape[2]
    if T < 1 or T > _TINY_MAX_TOKENS or N > _TINY_MAX_SEQ or N < 1:
        return None
    tiles = 2 if T <= 16 else 4
    waves = -(-(N * H) // n_sms)
    tc = _TINY_DYN_FIX_US + _TINY_DYN_EXTRA_US + _TINY_DYN_WAVE_US[tiles] * waves
    if tc >= _TINY_BAR * (_DYN_PLANNER_US + main_cost_us([32] * N, H, n_sms)):
        return None
    dev = q.device
    out = torch.empty_like(q)
    final_state = torch.empty(N, H, 128, 128, dtype=torch.float32, device=dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    ext = _module_tiny()
    sc = float(scale)

    def launch():
        ext.launch(q, k, v, g, beta, A, dtb, cu, initial_state, out, final_state, N * H, H, sc, -5.0, [], [], N, T, status, 0)
        return out, final_state

    return launch


def _module(feat=False, ck=False, rw=False, bd=False):
    global _ext
    if feat:
        return _feat_module("plain", "cake", ck, rw, bd)
    if _ext is None:
        _ext = _build_dir("cake", "kda_plain_build", False)
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
PAD_COPY_FACTORY = None   # [portfolio] set by w3/kernel.py: returns a (src [T, H], dst [Tp, Hp]) bf16 padded-copy launcher (contract_aux.cu) or raises
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


def plan(seq_lens, H, n_sms, allow_m64=True):
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
    if allow_m64 and m64 < m128:
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


def schedule(seq_lens, H, n_sms, feat=False):
    """Host-side plan: (variant, seq order, persistent sched, use_cio, cio (ids, lists, nslot) or None)."""
    variant, order, sched = plan(seq_lens, H, n_sms, allow_m64=True)
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


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, out=None, final_state=None, n_sms=None, feat=None, dynamic=False, val=None):
    """out / final_state: optional caller-owned output buffers (shapes of q / [N, H, D, D]); n_sms: SM budget for the plan (CP groups).
    feat [contract]: None (the v44 kernels: sequence n at row n of contiguous [N, H, D, D] state buffers), or a dict that selects the KDA_FEAT kernels
      {"src": (idx int32 [N] | None, slot_stride), "dst": (idx | None, slot_stride), "ck": (buffer, ck_cu int64 [N+1], n // 32) | None}:
    sequence n reads its initial state at initial_state + slot * src_stride + head * 16384 (slot = src idx[n], or n), writes its final state to final_state
    likewise (dst), and writes the state before token j * n of each head to row ck_cu[n] + j of the checkpoint buffer.  initial_state / final_state are then
    arbitrary (padded) pools of [., H, D, D] slots (M128 plain / cio / ppc and M64 have these builds).
    dynamic [contract]: the plan depends on shapes only (N, H, T, SM count), never on the contents of cu_seqlens: plain M128 kernel, task t = sequence * H + head dealt
    round-robin to the CTAs; the kernel reads cu_seqlens (and the slot / checkpoint tables) on the device, so a captured graph can be replayed with a new layout
    (same N, same T).  No host read of any device table (no cu.tolist()).  Load balance is the round-robin's."""
    B, T, H, D = q.shape
    assert D == 128
    dev = q.device
    if cu_seqlens is None:
        assert B == 1
        cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
        seq_lens = [T]
    elif dynamic:
        assert cu_seqlens.dtype == torch.int64 and cu_seqlens.is_contiguous(), "dynamic layout: cu_seqlens must be a contiguous int64 device tensor"
        cu, seq_lens = cu_seqlens, [0] * (cu_seqlens.numel() - 1)
    else:
        cu = cu_seqlens if cu_seqlens.dtype == torch.int64 else cu_seqlens.to(torch.int64)
        cl = cu.tolist()
        seq_lens = [cl[i + 1] - cl[i] for i in range(len(cl) - 1)]
    N = len(seq_lens)
    use_ck = feat is not None and feat["ck"] is not None
    if val is None:
        val = feat.get("val") if feat is not None else None   # [contract2] dynamic path: table-validation arguments (contract_aux.cu / validate_dev.cuh), run by the planner launch
    pk = {"val": val} if val else {}
    cu_k = val["cu_out"] if val and val["cu_out"] is not None else cu   # the kernels read the sanitized cu_seqlens when the planner produced one
    ext = _module(feat is not None, use_ck)
    dyn_plain = ident = False
    if dynamic:   # [planner_graph] device-planned launch (cio with pieces, or plain for short tasks): the task tables are produced on the GPU from cu_seqlens inside the graph
        dyn_plain = _dyn_use_plain(T, N, H, n_sms if n_sms is not None else _num_sms(dev))
        ident = dyn_plain and val is not None and val.get("ident_validate") is not None and N * H <= (n_sms if n_sms is not None else _num_sms(dev))
        variant, order, sched, use_cio, cio_plan = 128, list(range(N)), None, not dyn_plain, None
    else:
        variant, order, sched, use_cio, cio_plan = schedule(seq_lens, H, n_sms if n_sms is not None else _num_sms(dev), feat is not None)
    seq_order = torch.tensor(order, dtype=torch.int32, device=dev)
    pp = None if (sched is None or use_cio or dynamic) else plain_pieces(seq_lens, H, sched)
    # [pdlp] planned plain launches (more tasks than SMs) of a pool / checkpoint call read beta directly (no padded-beta copy blocks holding SMs next to the planner); identity launches keep the copy (it runs beside the validation, off the critical path)
    bd_planned = bool(dynamic and dyn_plain and not ident and feat is not None and val is not None and N <= 256 and N * H > (n_sms if n_sms is not None else _num_sms(dev)))
    if dyn_plain:
        ncta = max(1, min(n_sms if n_sms is not None else _num_sms(dev), N * H))
        dplan = _dev_planner(dev.index if dev.index is not None else torch.cuda.current_device())
        pdlp, nlate = False, 0
        if not ident and val is not None and feat is not None and N <= 256 and N * H > ncta:   # [pdlp] plain launch PDL-chained behind the lean planner: the grid leaves the planner / pad blocks' SMs free
            nsm_ = n_sms if n_sms is not None else _num_sms(dev)
            nb_ = 2 + (min((T * H + 1023) // 1024, 128) if ((H + 7) // 8 * 8 != H and not bd_planned) else 0)
            if nsm_ - nb_ >= nsm_ * 0.8:
                pdlp, nlate = True, nb_
        if ident:   # [pdl] one task per CTA: the plan is the identity (task b -> CTA b), fixed at prepare time; the launch only validates the caller's tables
            task_list = torch.arange(N * H + 2 * ncta, dtype=torch.int32, device=dev)
            cta_ptr = torch.arange(ncta + 1, dtype=torch.int32, device=dev)
        else:
            task_list = torch.zeros(N * H + 2 * ncta, dtype=torch.int32, device=dev)
            cta_ptr = torch.zeros(ncta + 1, dtype=torch.int32, device=dev)
    elif sched is None:
        task_list = cta_ptr = torch.zeros(1, dtype=torch.int32, device=dev)
    elif pp is None:
        task_list = torch.tensor(sched[0], dtype=torch.int32, device=dev)
        cta_ptr = torch.tensor(sched[1], dtype=torch.int32, device=dev)
    else:   # plain kernel with pieces: (task, c0, c1, slots) entries + L2 handoff scratch / flags
        ext_pp = _module_ppc(feat is not None, use_ck)
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
    if val:
        val["out"] = out   # [dynpath2] the planner's validation block zero-fills it when cu_seqlens is rejected
    if final_state is None:
        final_state = torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    assert out.is_contiguous() and (feat is not None or final_state.is_contiguous())
    Hp = (H + 7) // 8 * 8
    bd = Hp != H and (feat is None or bd_planned) and variant == 128 and not use_cio and pp is None   # [tiny] plain M128 launch with H % 8 != 0: beta read directly, no padded copy launch
    if bd:
        ext = _module_bd() if feat is None else _module(True, use_ck, bd=True)
    Tp = T if Hp == H else max(T, 32)   # [portfolio] T < 32 needs no padded copy when H % 8 == 0 (TMA zero-fills the box beyond the extent, as for q, k, v)
    beta2 = beta.reshape(T, H)
    beta_pad = None
    if bd:
        beta_tma = torch.zeros(32, 8, dtype=beta.dtype, device=dev)   # the beta map slot is unused by the KDA_BETA_DIRECT build
    elif Hp != H or Tp != T:
        beta_pad = torch.zeros(Tp, Hp, dtype=beta.dtype, device=dev)
        beta_tma = beta_pad
    else:
        beta_tma = beta2
    maps = ext.encode_maps(q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D), beta_tma,
                           out.view(T, H, D), variant, _LAYOUT_FLAGS).to(dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    if feat is not None:   # [contract] pools are addressed in place (slot stride), never copied: they must already be 16 B aligned
        assert initial_state.stride()[1:] == (D * D, D, 1) and initial_state.data_ptr() % 16 == 0 and final_state.data_ptr() % 16 == 0
        s0, s0_src = initial_state, None
        (src_idx, src_stride), (dst_idx, dst_stride), ck = feat["src"], feat["dst"], feat["ck"]
        fargs = (src_idx, src_stride, dst_idx, dst_stride, ck[0] if ck else None, ck[1] if ck else None, ck[2] if ck else 0)
    else:
        s0 = initial_state.contiguous()
        fargs = ()
    s0_src = None
    if feat is None and s0.data_ptr() % 16 != 0:   # [flow] cio cp.async state pieces need 16 B alignment: stage a misaligned view in an aligned buffer (re-copied every launch)
        s0_src, s0 = s0, torch.empty_like(s0)
    sc, lb = float(scale), -5.0

    if use_cio:
        cio = _module_cio(feat is not None, use_ck)
        if dynamic:
            ncta = max(1, min(n_sms if n_sms is not None else _num_sms(dev), N * H))
            dplan = _dev_planner(dev.index if dev.index is not None else torch.cuda.current_device())
            desc = torch.empty(4 * (N * H + 2 * ncta), dtype=torch.int32, device=dev)
            cta_ptr = torch.zeros(ncta + 1, dtype=torch.int32, device=dev)
            dummy_f = torch.zeros(ncta * 128 * 128, dtype=torch.float32, device=dev)   # state hand-off slots: at most one cut per CTA boundary
            dummy_i = torch.zeros(ncta, dtype=torch.int32, device=dev)
        else:
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

    _lean = dynamic and N <= 256 and (n_sms if n_sms is not None else _num_sms(dev)) <= 256 and (T * H * 100 // 32 + N * H * 400 + 400 * 256) < (1 << 30)   # lean planner (plan_cp.cu plan_lean_*): <= 256 sequences, 32-bit cost units (< 2^30)

    pad_fn = None
    if beta_pad is not None and PAD_COPY_FACTORY is not None:
        try:
            pad_fn = PAD_COPY_FACTORY()   # [portfolio] built at prepare time (never inside a graph capture)
        except Exception:
            pad_fn = None

    pad_skip = [False]
    if dynamic and val is not None and beta_pad is not None:   # [long] the planner launch carries the copy (extra blocks): no graph node of its own
        val["pad_src"], val["pad_dst"] = beta2, beta_pad
        pad_skip[0] = True

    def pad_only():   # [shortmid] the padded-beta copy alone (H % 8 != 0, no beta-direct build): _prepare_impl's launch_eager orders it against the concurrent big launches (NOTES: TPC sharing)
        if pad_fn is not None:
            pad_fn(beta2, beta_pad)
        else:
            beta_pad[:T, :H].copy_(beta2)

    def launch(pad=True, pad_done=None):
        if s0_src is not None:
            s0.copy_(s0_src)
        if beta_pad is not None:
            if pad and not pad_skip[0]:
                pad_only()
            if pad_done is not None:   # a sibling launch waits for this event (launch_eager)
                pad_done.record(torch.cuda.current_stream())
        if use_cio:
            if dynamic:
                if _lean:
                    dplan.plan_lean(cu, H, ncta, desc, cta_ptr, 0, None, **pk)
                else:
                    dplan.plan_wrap(cu, H, ncta, desc, cta_ptr, **pk)
            cio.launch(q, k, v, g, beta, A, dtb, cu_k, seq_order, s0, out, final_state, maps, desc, cta_ptr, dummy_f, dummy_i, H, sc, lb, *fargs)
        else:
            if dyn_plain and ident:
                val["ident_validate"](dict(val, noop=1))
            elif dyn_plain:
                if _lean:
                    dplan.plan_lean(cu, H, ncta, task_list, cta_ptr, 1 | (nlate << 8), PLAN_DBG, **pk)
                else:
                    dplan.plan_wrap(cu, H, ncta, task_list, cta_ptr, True, **pk)
            if pp is not None:
                ext_pp.launch(variant, q, k, v, g, beta, A, dtb, cu_k, seq_order, s0, out, final_state, maps, task_list, cta_ptr, p_scratch, p_flags, N, H, sc, lb, *fargs)
            elif ident:
                ext.set_pdl(1)   # [pdl] identity plan: the kernel's prologue overlaps the validation launch (griddepcontrol.wait before the first table read)
                try:
                    ext.launch(variant, q, k, v, g, beta, A, dtb, cu_k, seq_order, s0, out, final_state, maps, task_list, cta_ptr, N, H, sc, lb, *fargs)
                finally:
                    ext.set_pdl(0)   # [audit E1] a launch exception must not leave this thread in PDL mode for the next (different) prepared launch
            else:
                if dyn_plain and pdlp: ext.set_pdl(2)   # PDL behind the planner: the task tables are written by it (the kernel prefetches the first state only after the wait)
                try:
                    ext.launch(variant, q, k, v, g, beta, A, dtb, cu_k, seq_order, s0, out, final_state, maps, task_list, cta_ptr, N, H, sc, lb, *fargs)
                finally:
                    if dyn_plain and pdlp: ext.set_pdl(0)   # [audit E1]
        return out, final_state

    launch.has_pad = beta_pad is not None
    launch.pad_only = pad_only
    launch.pad_info = (beta2, beta_pad) if beta_pad is not None else None
    launch.pad_skip = pad_skip
    return launch


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    return prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)()
