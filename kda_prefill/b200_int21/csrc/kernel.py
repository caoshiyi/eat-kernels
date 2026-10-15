"""KDA forward: exact context-parallel (CP) wrapper (analyst) around base/ (kernels and planners: see base/kernel.py and RECIPE).

The recurrence is affine in the state: S_t = A_t S_{t-1} + b_t.  When the (sequence, head) tasks leave SMs idle (few long
sequences), one sequence j (one of the two longest; the cost model picks) is split at a chunk boundary x into a head part and a tail:
  P : sequences 0..j with j cut at x, run normally                     -> outputs in place, S_A = state of j at x
  S : sequences j+1.. (if any), run normally on their own SM budget   -> outputs and final states in place
  L : the tail from S = 0                                              -> o_loc written in place into the output, S_loc
  Tr: the tail with v = 0 and S0 = I (the value index becomes the transition's column index)
                                                                       -> Qt_t = scale P_t q_t (bf16), P = transition over the tail
  fixup (cp_fix.cu): out[x + t] = o_loc[t] + bf16(S_A) Qt_t ;  S_fin = S_A P + S_loc (fp32-class: bf16 hi/lo split operands)
[architect it7] MULTI-PIECE tails: the dual launch runs tasks (piece, head); when it has spare CTAs (e.g. H = 64: 84 CTAs for 64 whole-tail tasks) the planner cuts every head's tail
into two pieces (s1, s2 chunks; s1 a multiple of 4 = one 128-token fixup tile) and packs the 2H piece tasks on all of the dual CTAs.  The
fixup then chains the states: S_in(1) = S_A P_0 + S_loc_0, S_fin = S_in(1) P_1 + S_loc_1, and the CTA that owns a head also writes that head's
piece-1 tiles (no flags, no inter-CTA waiting).  Chosen only when the model gains >= _MULTI_MARGIN and the launch averages <= 3 tasks per CTA.
L and Tr run either as two base launches or (architect4, base/dual.py + base/cake_dual/) as ONE dual-chain launch in which every
tail task runs both chains over a single shared prep (per chain bitwise identical to the base plain kernel; S0 = 0 / I written in
kernel); cp_plan prices both options and keeps the cheaper.
P, S and the tail groups run concurrently on disjoint SM budgets (graph branches on private streams, persistent grids sized by the base
planner).  Whether and where to split is chosen by a cost model (chunk counts, SM budgets, fixup bytes), never by a table of layouts.


  Numeric range.  The decay of S is deferred: a key channel's state is held as S / 2^R with R in [-80, 0] (rebased to R = 0 by an
  exact scale pass when the accumulated log2-decay of any channel passes -80 within a chunk group, at a chunk boundary that is a
  function of g only).  Scaling by 2^R is exact in fp32, so the result is the same operator; the only cost is range: the stored
  state is up to 2^80 larger than the true state.  The kernel is therefore exact for |S|, |v| below ~5e14 (measured sharp
  threshold at mid decay: x5e14 passes, x1e15 .. x3e16 overflow) and above that overflows to inf/NaN LOUDLY (never a finite wrong
  answer; fine sweep it8/logs31/magfine80.txt).  Slow-decay inputs (no rebase) keep fp32's full range.  Real Kimi-Linear prefill: v is a silu'd short-conv
  projection of RMS-normed hidden states (|v| <~ 1e2-1e3), S0 is a previous state of that same operator (|S| <~ 1e3-1e4), q/k are
  L2-normed, beta in (0,1), a in [-5, 0): 10 orders of magnitude below the limit.
"""
import os
import sys
import importlib.util

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_base = _load("kda_cp_base_kernel", os.path.join(_HERE, "base", "kernel.py"))
_dual = _load("kda_cp_dual_launch", os.path.join(_HERE, "base", "dual.py"))  # [architect4] L + Tr as one dual-chain launch
_fix = None


_streams = {}


def _branch_streams(fix, dev):
    """Three private non-blocking streams per device, created once (cudaStreamCreate in cp_fix.cu)."""
    key = torch.device(dev).index
    if key not in _streams:
        _streams[key] = [torch.cuda.ExternalStream(fix.new_stream(), device=dev) for _ in range(3)]
    return _streams[key]


_fs = None


def _fixsched():
    global _fs
    if _fs is None:
        _fs = _load("kda_cp_fix_sched", os.path.join(_HERE, "fix_sched.py"))
    return _fs


def _fixmod():
    global _fix
    if _fix is None:
        _fix = _load("kda_cp_fix_build", os.path.join(_HERE, "build_fix.py")).build()
    return _fix


# ---------------------------------------------------------------- cost model (units: one 32-token chunk step of an M128 CTA)
_STEP_US = 1.70            # measured: one chunk step ~3.15k cycles at ~1.85 GHz
_TASK_COST = 1.0           # per task (state load/store, pipeline fill)
_CONTENTION = 1.0          # [launch it13] was 1.06: in situ the tail (dual) groups finish BEFORE the head (h64-fixed: 194 vs 209 us), i.e. the model overpriced them; 1.0 balances the CP split (h64 [8192] -1.9%, h80 -2.0%, h32 -2.6%, held-out 3000/12288/32768/31-seq -1.7..-5.1%; 0.97 / 0.94 mixed or worse)
_FIX_BW = 4.5e6            # fixup effective bytes/us (output: read o_loc + Qt, write out)
_FIX_STATE_US = 0.08       # fixup per-head state composition (192 KB read + 64 KB written per head)
_FIX_FIXED_US = 6.0        # launch, prologue and tail of the fixup (measured: below ~50 chunk-steps of work the fixed costs decide)
_FIX_OWNER_TILES = 8.0     # [architect it7] first state item of a head with a chained (multi-piece) tail, in output tiles (in situ sweep 6-12: 8)
_FIX_CHAIN_TILES = 4.0     # [architect it7] each further chained state item of that head
_FORCE_PIECES = None       # [architect it7] tests: list of tail piece sizes (chunks) overriding the planner's pieces
_FIX_STATE_TILES = 6.0     # a head's state item costs ~6 output tiles of the fixup grid (in-situ sweep 2-10 on cake_v15: 6 best)
_SHORT_CHUNKS = 17         # tasks shorter than this take the base's blocking state-I/O path (measured 13-17k cycles per task)
_SHORT_TASK_COST = 5.0     # chunk-steps per such task (vs _TASK_COST for long tasks)
_MIN_GAIN = 0.04           # required modelled gain over the direct plan
_DUAL_F = 1.40             # [architect4] dual-chain task (L + Tr over one prep): chunk-steps per chunk (measured, see work_architect4)
_DUAL_TASK = 6.4           # [architect4] dual task boundary: two states in, two out (plain blocking I/O, 2 x 3.2 chunk-steps, analyst)


_MULTI_MARGIN = 0.05      # [architect it7] a two-piece tail must beat the whole-task plan by this much in the model (h96-fixed: model +1%, measured -3%)
_PIECE_TASKS_PER_CTA = 3  # [architect it7] pieces only while the dual launch averages <= 3 tasks per CTA (h96: 3.7-4.8 tasks per CTA measured +2..3.4% slower)
_PIECE_MIN = 8            # [architect it7] shortest dual piece (chunks); pieces other than the last are multiples of 4 chunks (= one 128-token fixup tile)
_FIX_CHAIN_US = 0.07      # [architect it7] fixup: extra state item per head per extra tail piece (us per head, in situ: 2 pieces on 64 heads = +4.5 us with the tile bytes)


def _dual_pieces(tcx, H, n_D):
    """[architect it7] Multi-piece dual tail: cut every head's tail (tcx chunks) into two consecutive dual pieces (s1, s2), s1 a multiple of 4
    chunks, s2 <= s1; the 2H piece tasks are packed (LPT) on the n_D CTAs of the dual launch, which a whole-task-per-head plan leaves
    partly idle (H < n_D) or unbalanced.  Returns (makespan, (s1, s2)) or None.  The fixup chains the heads' states over the pieces."""
    if n_D < 2 or tcx < 2 * _PIECE_MIN or 2 * H > _PIECE_TASKS_PER_CTA * n_D:
        return None
    best = None
    lo = _PIECE_MIN
    hi = tcx // 2
    step = max(4, ((hi - lo) // 10) // 4 * 4)
    cands = set(range(lo, hi + 1, step)) | {hi}
    done = set()
    while cands:
        s2 = cands.pop()
        if s2 in done or s2 < lo or s2 > hi:
            continue
        done.add(s2)
        s1 = tcx - s2
        if s1 % 4:
            s1 -= s1 % 4
            s2 = tcx - s1
            if s2 > s1 or s2 in done:
                continue
            done.add(s2)
        mk = _lpt_makespan([_DUAL_F * s1 + _DUAL_TASK] * H + [_DUAL_F * s2 + _DUAL_TASK] * H, n_D)
        if best is None or mk < best[0]:
            best = (mk, (s1, s2))
    if best is not None and step > 4:        # refine around the coarse optimum
        s2c = best[1][1]
        for s2 in range(max(lo, s2c - step), min(hi, s2c + step) + 1, 4):
            if s2 in done:
                continue
            done.add(s2)
            s1 = tcx - s2
            if s1 % 4 or s2 > s1:
                continue
            mk = _lpt_makespan([_DUAL_F * s1 + _DUAL_TASK] * H + [_DUAL_F * s2 + _DUAL_TASK] * H, n_D)
            if mk < best[0]:
                best = (mk, (s1, s2))
    return best


def _lpt_makespan(costs, n):
    import heapq
    heap = [0.0] * max(1, n)
    for c in sorted(costs, reverse=True):
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + c)
    return max(heap)


_gc_cache = {}


def _group_cost(seq_lens, H, n_sms):
    """Modelled makespan (chunk-steps) of one base launch on n_sms CTAs, from the base's own planner: its cio piece lists
    (split / wrap planners) when it has them, else the M64 or whole-task LPT plan."""
    key = (tuple(seq_lens), H, n_sms)
    if key in _gc_cache:
        return _gc_cache[key]
    chunks = [(L + 31) // 32 for L in seq_lens]
    cio_plan = None
    if hasattr(_base, "schedule"):
        variant, order, sched, _, cio_plan = _base.schedule(seq_lens, H, n_sms)
    else:
        variant, order, sched = _base.plan(seq_lens, H, n_sms)
    if cio_plan is not None and hasattr(_base, "_list_cost"):
        c = max(_base._list_cost(l) for l in cio_plan[1])
    elif variant == 64:
        c = _base._makespan([chunks[i] * _base._M64_CHUNK_COST + _base._TASK_COST_LAUNCH for i in order for _ in range(2 * H)], n_sms)
    else:
        c = _lpt_makespan([chunks[n] + _TASK_COST for n in range(len(seq_lens)) for _ in range(H)], n_sms)
    if any(0 < ch < _SHORT_CHUNKS for ch in chunks):
        # the base planner prices every task boundary alike; short tasks really cost _SHORT_TASK_COST each (blocking I/O path)
        c = max(c, _lpt_makespan([ch + (_SHORT_TASK_COST if ch < _SHORT_CHUNKS else _TASK_COST)
                                  for ch in chunks if ch > 0 for _ in range(H)], n_sms))
    _gc_cache[key] = c
    return c


def _direct_cost(seq_lens, H, n_sms):
    chunks = [(L + 31) // 32 for L in seq_lens]
    lpt = _group_cost(seq_lens, H, n_sms)
    # the base may also cut tasks into sequential pieces on other CTAs (split planners): it cannot beat the longest task or the
    # evenly spread total, so take that bound as the direct cost (CP is only chosen where whole idle SMs remain)
    bound = max(max(chunks) + _TASK_COST, sum(c + _TASK_COST for c in chunks) * H / n_sms)
    return min(lpt, bound)


def _fix_us(tail, H):
    """Fixup time (measured in situ, cp_fix.cu): tile bytes (read o_loc + Qt, write out) at the in-situ rate + per-head state work."""
    return _FIX_FIXED_US + tail * H * 128 * 2 * 3 / _FIX_BW + _FIX_STATE_US * H


def _split_budget(w1, w2, n):
    """SMs for two concurrent persistent launches with work w1, w2 sharing a budget of n."""
    if w2 <= 0:
        return n, 0
    if w1 <= 0:
        return 0, n
    n2 = min(n - 1, max(1, int(round(n * w2 / (w1 + w2)))))
    return n - n2, n2


def _even_split(n_p, n_s):
    """[analyst it6] Group boundaries on even SM indices: concurrent groups that share a TPC (two SMs) at an odd boundary run 5-15%
    slower in situ (forced-plan sweeps, work_analyst/it4/v29_*, BOARD 23:5x).  Round the P / S budget split to an even n_P."""
    if n_s <= 0 or n_p % 2 == 0:
        return n_p, n_s
    tot = n_p + n_s
    if n_p >= 3:
        n_p -= 1
    else:
        n_p += 1
    return n_p, tot - n_p


def _side_cost(seq_lens, H, n_sms):
    """Cost of the P / S groups: the base planner for small groups; for many tasks the fluid bound max(longest, total / SMs)
    (keeps planning time bounded for layouts with hundreds of sequences)."""
    if len(seq_lens) * H <= 4 * n_sms:
        return _group_cost(seq_lens, H, n_sms)
    cs = [(L + 31) // 32 for L in seq_lens]
    tc = [c + (_SHORT_TASK_COST if c < _SHORT_CHUNKS else _TASK_COST) for c in cs if c > 0]
    return max(max(tc), sum(tc) * H / n_sms) if tc else 0.0


def cp_plan(seq_lens, H, n_sms):
    """Returns None (run direct) or (j, x, n_P, n_S, n_B, pieces, p_first): split sequence j at x tokens from its start (multiple of 32).
    Groups (run concurrently): P = sequences 0..j with j cut at x (n_P SMs), S = sequences j+1.. (n_S SMs, may be 0),
    L and Tr = the tail of j (n_B SMs each).  Modelled cost = contention * max(groups) + fixup (all from chunk counts)."""
    N = len(seq_lens)
    chunks = [(L + 31) // 32 for L in seq_lens]
    tcost = lambda ch: ch + (_SHORT_TASK_COST if ch < _SHORT_CHUNKS else _TASK_COST) if ch > 0 else 0.0
    # CP only pays when the longest task leaves SMs idle: cheap exit when it does not exceed the evenly spread work
    if max(chunks) + _TASK_COST <= (1.0 + _MIN_GAIN) * sum(tcost(c) for c in chunks) * H / n_sms:
        return None
    direct = _direct_cost(seq_lens, H, n_sms)
    best = None
    for j in sorted(range(N), key=lambda i: -seq_lens[i])[:2]:           # the two longest sequences
        T = seq_lens[j]
        if T < 32 * 8:
            continue
        pre_lens, suf_lens = seq_lens[:j], seq_lens[j + 1:]
        w_suf = sum(tcost(chunks[n]) for n in range(j + 1, N)) * H
        w_pre = sum(tcost(chunks[n]) for n in range(j)) * H
        tc = (T + 31) // 32

        def cost_at(xc, n_A, multi=True):
            n_B = (n_sms - n_A) // 2
            n_B -= n_B % 2 if n_B > 1 else 0          # [analyst it6] even group boundaries (L / Tr start on even SM indices)
            n_P, n_S = _even_split(*_split_budget(w_pre + (xc + _TASK_COST) * H, w_suf, n_A))
            if n_P < 1 or (w_suf > 0 and n_S < 1):
                return None
            a = max(_side_cost(pre_lens + [32 * xc], H, n_P), _side_cost(suf_lens, H, n_S) if w_suf > 0 else 0.0)
            b = _group_cost([T - 32 * xc], H, n_B)
            n_D = n_sms - n_A                       # [architect4] dual: L + Tr as one launch of H whole tasks on n_D CTAs
            b_d = _lpt_makespan([_DUAL_F * (tc - xc) + _DUAL_TASK] * H, n_D)
            fix = _fix_us(T - 32 * xc, H) / _STEP_US
            opt = (_CONTENTION * max(a, b_d) + fix, (n_P, n_S, -n_D, None, a >= b_d)) if b_d < b else \
                (_CONTENTION * max(a, b) + fix, (n_P, n_S, n_B, None, a >= b))
            if multi:
                mp = _dual_pieces(tc - xc, H, n_D)
                if mp is not None:
                    cm = max(a, _CONTENTION * mp[0]) + fix + _FIX_CHAIN_US * H / _STEP_US
                    if cm < opt[0] * (1.0 - _MULTI_MARGIN):
                        opt = (cm, (n_P, n_S, -n_D, mp[1], a >= _CONTENTION * mp[0]))
            return opt

        for n_A in sorted(set([H, n_sms // 2, (2 * n_sms) // 3, H + (n_sms - H) // 4, H + (n_sms - H) // 3,
                               n_sms - H, n_sms - (H + 1) // 2])):   # [architect4] + dual groups with 1 or 2 whole tasks per CTA
            n_A -= n_A % 2 if n_A > 2 else 0          # [analyst it6] the dual / L groups start on an even SM index
            if n_A < 1 or (n_sms - n_A) // 2 < 1:
                continue
            step = max(4, (tc - 7) // 24)
            coarse = [(cost_at(xc, n_A), xc) for xc in range(4, tc - 3, step)]  # coarse grid (<= ~24 points), then refine
            coarse = [(c, xc) for c, xc in coarse if c is not None]
            if not coarse:
                continue
            _, xc0 = min(coarse, key=lambda t: t[0][0])
            for xc in range(max(4, xc0 - step + 1), min(tc - 3, xc0 + step)):
                r = cost_at(xc, n_A)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], j, 32 * xc) + r[1]
    if best is None or best[0] > direct * (1.0 - _MIN_GAIN):
        return None
    return best[1:]


def _prepare_impl(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    B, T, H, D = q.shape
    dev = q.device
    if D != 128 or B != 1:
        raise ValueError(f"KDA kernel: expected q of shape [1, T, H, 128], got {tuple(q.shape)}")
    if cu_seqlens is None:
        cl = [0, T]
    else:
        cl = [int(c) for c in cu_seqlens.tolist()]
        if len(cl) < 2 or cl[0] != 0 or cl[-1] > T or any(b < a for a, b in zip(cl, cl[1:])):
            raise ValueError("KDA kernel: cu_seqlens must be non-decreasing, start at 0 and not exceed T")
    if tuple(initial_state.shape) != (len(cl) - 1, H, D, D):
        raise ValueError(f"KDA kernel: initial_state must be [N, H, 128, 128] with N = {len(cl) - 1}, got {tuple(initial_state.shape)}")
    seq_lens = [cl[i + 1] - cl[i] for i in range(len(cl) - 1)]
    N = len(seq_lens)
    n_sms = _base._num_sms(dev)
    plan = cp_plan(seq_lens, H, n_sms)
    if plan is None:
        return _base.prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)
    j, xl, n_P, n_S, n_B, pieces, p_first = plan
    if _FORCE_PIECES:                              # tests / experiments: force the tail pieces (chunks per piece), keeping the planner's SM budgets
        pieces = list(_FORCE_PIECES)
        xl = seq_lens[j] - 32 * sum(pieces)
        n_B = -(n_sms - n_P)
    x = cl[j] + xl                                     # absolute token index of the split
    e = cl[j + 1]                                      # end of the split sequence
    tail = e - x
    out = torch.empty_like(q)
    final_state = torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    sA = torch.empty(j + 1, H, D, D, dtype=torch.float32, device=dev)  # prefix launch final states (entry j = S_A)
    m_p = len(pieces) if pieces else 1
    tph = (tail + 127) // 128
    if pieces:
        assert sum(pieces) == (tail + 31) // 32 and all(c % 4 == 0 and c > 0 for c in pieces[:-1]) and pieces[-1] > 0, (pieces, tail)
        pt = [0]
        for c in pieces[:-1]:
            pt.append(pt[-1] + c // 4)
        pt.append(tph)
        assert all(pt[i + 1] > pt[i] for i in range(m_p)), pt
    else:
        pt = [0, tph]
    sL = torch.empty(m_p, H, D, D, dtype=torch.float32, device=dev)
    sP = torch.empty(m_p, H, D, D, dtype=torch.float32, device=dev)
    sin_ = torch.empty(max(m_p - 1, 0), H, D, D, dtype=torch.float32, device=dev)
    qt = torch.empty(1, tail, H, D, dtype=q.dtype, device=dev)
    vz = torch.zeros(1, tail, H, D, dtype=v.dtype, device=dev)
    s0z = torch.zeros(m_p, H, D, D, dtype=torch.float32, device=dev)
    eye = torch.eye(D, dtype=torch.float32, device=dev).expand(m_p, H, D, D).contiguous()
    i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
    hd = lambda t: t[:, :x]
    tl = lambda t: t[:, x:e]
    sf = lambda t: t[:, e:]
    LP = _base.prepare(hd(q), hd(k), hd(v), hd(g), hd(beta), A_log, dt_bias, scale, initial_state[:j + 1],
                       i64(cl[:j + 1] + [x]) if j > 0 else None, out=hd(out), final_state=sA, n_sms=n_P)
    LS = None
    suffix_empty = j + 1 < N and cl[N] == e                 # only zero-length sequences after j: final state = initial state
    if j + 1 < N and not suffix_empty:
        LS = _base.prepare(sf(q), sf(k), sf(v), sf(g), sf(beta), A_log, dt_bias, scale, initial_state[j + 1:],
                           i64([c - e for c in cl[j + 1:]]) if N - j - 1 > 1 else None, out=sf(out), final_state=final_state[j + 1:],
                           n_sms=n_S)
    if n_B < 0:   # [architect4] dual-chain tail: L (v, S0 = 0) and Tr (v = 0, S0 = I) share one prep in one launch on -n_B CTAs
        LL = _dual.prepare(tl(q), tl(k), tl(v), tl(g), tl(beta), A_log, dt_bias, scale, s0z, eye, tl(out), qt, sL, sP, -n_B,
                           l_zero=True, t_eye=True, pieces=pieces, piece_cost=(_DUAL_F, _DUAL_TASK))
        LT = None
    else:
        LL = _base.prepare(tl(q), tl(k), tl(v), tl(g), tl(beta), A_log, dt_bias, scale, s0z, None,
                           out=tl(out), final_state=sL, n_sms=n_B)
        LT = _base.prepare(tl(q), tl(k), vz, tl(g), tl(beta), A_log, dt_bias, scale, eye, None,
                           out=qt, final_state=sP, n_sms=n_B)
    fix = _fixmod()
    fs = _fixsched()
    seg_l, segp_l = fs.fix_schedule(H, tail, n_sms, _FIX_STATE_TILES if m_p == 1 else _FIX_OWNER_TILES, pt, _FIX_CHAIN_TILES)
    seg_t = torch.tensor(seg_l, dtype=torch.int32, device=dev)
    segp_t = torch.tensor(segp_l, dtype=torch.int32, device=dev)
    # private streams (not from torch's pool: a pool stream can coincide with the capturing stream and serialize the branches)
    streams = _branch_streams(fix, dev)
    e0 = torch.cuda.Event()
    ev = [torch.cuda.Event() for _ in range(3)]
    out_tail = out[0, x:e]

    def launch_eager():
        cur = torch.cuda.current_stream()
        e0.record(cur)
        if p_first:     # the group with the longest modelled makespan is launched first (graph branches start in capture order, ~3 us apart)
            LP()
        for s in streams:
            s.wait_event(e0)
        for s, ee, fn in zip(streams, ev, (LL, LT, LS)):
            if fn is not None:
                with torch.cuda.stream(s):
                    fn()
                    ee.record(s)
        if not p_first:
            LP()
        if j > 0:
            final_state[:j].copy_(sA[:j])
        if suffix_empty:
            final_state[j + 1:].copy_(initial_state[j + 1:])
        for ee, fn in zip(ev, (LL, LT, LS)):
            if fn is not None:
                cur.wait_event(ee)
        fix.fixup(out_tail, qt, sA[j], sP, sL, sin_, final_state[j], seg_t, segp_t, H, tail, m_p)

    def launch():
        launch_eager()
        return out, final_state

    return launch


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    """[analyst it6] API normalization.  The kernels read raw contiguous bf16 / fp32 buffers (TMA maps, flat pointers).  An input that is
    not contiguous (a view of a fused projection), not of the API dtype, or not 16 B aligned (a contiguous view at an odd element offset:
    TMA maps and vector loads need 16 B) is copied into a private contiguous buffer on EVERY launch
    (launch() must read the current contents); contiguous inputs of the right dtype are used in place, at zero cost."""
    spec = ((q, torch.bfloat16), (k, torch.bfloat16), (v, torch.bfloat16), (g, torch.bfloat16), (beta, torch.bfloat16),
            (A_log, torch.float32), (dt_bias, torch.float32), (initial_state, torch.float32))
    ins, pairs = [], []
    for t, dt in spec:
        if t.dtype != dt or not t.is_contiguous() or t.data_ptr() % 16:
            c = torch.empty(t.shape, dtype=dt, device=t.device)
            pairs.append((c, t))
            t = c
        ins.append(t)
    inner = _prepare_impl(*ins[:7], scale, ins[7], cu_seqlens)
    if not pairs:
        return inner

    def launch():
        for c, t in pairs:
            c.copy_(t)
        return inner()

    return launch


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    return prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)()
