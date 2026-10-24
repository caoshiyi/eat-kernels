"""KDA forward: exact context-parallel (CP) wrapper (analyst) around base/ (kernels and planners: see base/kernel.py and RECIPE).

[contract] Production contract (program.md): run / prepare take ssm_state_indices (state pool, in place), state_checkpoints + checkpoint_cu_starts +
checkpoint_every_n_tokens (prefix-cache checkpoints), initial_state_source + initial_state_indices (committed-state source pool) and dynamic_layout.
Without any of them the benchmark path below is used unchanged (default kernel builds, SASS-identical to cake_v44).  With them the same kernel sources run as
their KDA_FEAT / KDA_CK builds (see work_contract/NOTES.md): state addressing through device slot tables, checkpoints written by the chunk loops (and, for a CP
tail, by dual-kernel snapshots composed in the fixup), host-planned CP / M64 / cio / ppc plans as before.  dynamic_layout=True: shape-only plan (no host reads).

[contract2] Table validation (no host sync).  The kernels never read the caller's cu_seqlens / ssm_state_indices / initial_state_indices / checkpoint_cu_starts: one device pass at the head of every
launch (a second block of the dynamic path's planner launch, contract_aux.cu in the static path) writes sanitized private copies.  A slot that is negative, >= P or (destination) a repeat is redirected to a private
trash window addressed as a pool slot (the pool is never touched, the sequence's outputs are undefined but finite); checkpoint rows that do not fit in the buffer are skipped (row -1); a cu_seqlens that is not
non-decreasing inside [0, T] makes the dynamic launch a no-op (all sequences empty); the static path (prepare / run above ~4300 sequences) raises ValueError at prepare time for any rejected table entry.  `launch.status` / `last_status` (bits 1 cu, 2 dst slot, 4 src slot, 8 checkpoint rows; status_bits()) report
the last launch, KDA_CHECK=1 raises ValueError after each launch.  checkpoint_every_n_tokens: any multiple of 32 (exact); other values, wrong dtypes / shapes raise ValueError.  run() with more sequences than the
device planner holds (single-CTA shared memory, ~4300) takes the host-planned path.  Details and measurements: work_contract2/NOTES.md, REPORT.md.

[portfolio] Few-head shapes (H <= 37 and H <= n_sms / 4: the one / two-piece plan leaves most SMs idle, e.g. H = 1: one chain of T / 32 chunk steps on one CTA).  The tail is cut into m >> 2 pieces (mp_plan: closed-form
us model of head chain / dual pieces / fixup chain / output tiles, calibrated on measured rates; chosen only when it beats the best one / two-piece plan of the same model by 10%), the fixup runs as a state-chain launch
(cp_fix.cu fixup_mp mode 1, or the hierarchical fixup_h pair when the model says it is shorter: group totals Q / R in parallel, then the chain over the totals, [portfolio] NOTES.md) and one output-tile launch (mode 2) over all SMs.
Several long sequences on few heads (e.g. two 524288-token sequences at H = 1) run as independent CP instances on their own SM budgets and private streams (_multi_plan / _prepare_multi).  A tail that carries checkpoints keeps the
one / two-piece plan; the state pool is supported (staged final state + slot copy).  KDA_MP=0 / KDA_HIER=0 switch the additions off (A/B); with them off the launches are cake_v46's.

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
import functools
import threading
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
_dyncp = _load("kda_pg_dyncp", os.path.join(_HERE, "..", "dyn", "dyncp.py"))   # [planner_graph] device-planned context-parallel path (graph-safe)
_dynmp = _load("kda_pg_dynmp", os.path.join(_HERE, "..", "dyn", "dynmp.py"))   # [dynfew] device-planned many-piece context-parallel path for few-head multi-sequence launches
_fix = None
_fix_ck = None
_fix_ck_bf = None


def _dev_index(dev):
    """Device ordinal of a tensor's / torch.device (an index-less torch.device('cuda') means the current device)."""
    i = torch.device(dev).index
    return torch.cuda.current_device() if i is None else i


def _new_stream_on(fix, dev):
    """[multi-GPU] a private stream of device dev: cudaStreamCreate* binds the stream to the CURRENT device, so create it under that device."""
    idx = _dev_index(dev)
    with torch.cuda.device(idx):
        return torch.cuda.ExternalStream(fix.new_stream(), device=idx)


_streams = {}


def _branch_streams(fix, dev, idx=0):
    """Three private non-blocking streams per device (and per concurrent instance idx, [portfolio] multi-sequence CP), created once (cudaStreamCreate in cp_fix.cu)."""
    key = (_dev_index(dev), idx)
    if key not in _streams:
        _streams[key] = [_new_stream_on(fix, dev) for _ in range(3)]
    return _streams[key]


_fs = None


def _fixsched():
    global _fs
    if _fs is None:
        _fs = _load("kda_cp_fix_sched", os.path.join(_HERE, "fix_sched.py"))
    return _fs


def _fixmod(ck=False):
    global _fix, _fix_ck, _fix_ck_bf
    if ck:   # [contract] cp_fix_ck.cu: the fixup with checkpoint items (the default fixup is the v44 kernel, unchanged)
        if _base.ck_bf16():   # [bf16ckpt] bf16 checkpoint rows: own flavor
            if _fix_ck_bf is None:
                _fix_ck_bf = _load("kda_cp_fix_build_ck_bf", os.path.join(_HERE, "build_fix.py")).build(ck=True, bf=True)
            return _fix_ck_bf
        if _fix_ck is None:
            _fix_ck = _load("kda_cp_fix_build_ck", os.path.join(_HERE, "build_fix.py")).build(ck=True)
        return _fix_ck
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
_SNAP_STEPS = 3.0           # [contract] chunk-steps per tail checkpoint snapshot of a dual task (two 64 KB state stores of the compute warps; plain final store ~1.5 us)
_FIX_ITEM_US = 6.0          # [contract] fixup: one checkpoint item (S_in P + S_loc: 3 hi/lo MMA passes, 192 KB read, 64 KB TMA store), us on its owner CTA (fitted: H96 [8192] n=256 direct 421 us vs CP 428 us, H64 n=256 CP 392 vs direct 429, n=1024 CP wins both)
_SNAP_BYTES_MAX = 3 << 30  # [contract] snapshot scratch (S_loc + P per tail checkpoint per head) above which a CP plan is not considered
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


_gc_cache = {}    # bounded: cleared when it reaches _GC_MAX entries (one entry per distinct layout seen by the static planner)
_GC_MAX = 4096


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
    if len(_gc_cache) >= _GC_MAX:
        _gc_cache.clear()
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


def cp_plan(seq_lens, H, n_sms, ck_n=0):
    """Returns None (run direct) or (j, x, n_P, n_S, n_B, pieces, p_first): split sequence j at x tokens from its start (multiple of 32).
    Groups (run concurrently): P = sequences 0..j with j cut at x (n_P SMs), S = sequences j+1.. (n_S SMs, may be 0),
    L and Tr = the tail of j (n_B SMs each).  Modelled cost = contention * max(groups) + fixup (all from chunk counts).
    ck_n [contract]: checkpoint interval in chunks (0 = none): the tail runs as the dual launch only (S_loc / P snapshots at its checkpoint boundaries,
    composed by the fixup), priced by _SNAP_STEPS per snapshot and _FIX_ITEM_US per fixup item."""
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
            rows = max(0, (tc - 1) // ck_n - xc // ck_n) if ck_n else 0          # [contract] tail checkpoint boundaries beyond the cut
            if rows * H * 2 * 65536 > _SNAP_BYTES_MAX:
                return None
            snap = rows * _SNAP_STEPS
            b_d = _lpt_makespan([_DUAL_F * (tc - xc) + _DUAL_TASK + snap] * H, n_D)
            fix = (_fix_us(T - 32 * xc, H) + ((H + n_sms - 1) // n_sms) * rows * _FIX_ITEM_US) / _STEP_US
            if ck_n:
                b = float("inf")                    # the base-launch tail has no snapshot path
            opt = (_CONTENTION * max(a, b_d) + fix, (n_P, n_S, -n_D, None, a >= b_d)) if b_d < b else \
                (_CONTENTION * max(a, b) + fix, (n_P, n_S, n_B, None, a >= b))
            if multi:
                mp = _dual_pieces(tc - xc, H, n_D)
                if mp is not None:
                    cm = max(a, _CONTENTION * (mp[0] + snap)) + fix + _FIX_CHAIN_US * H / _STEP_US
                    if cm < opt[0] * (1.0 - _MULTI_MARGIN):
                        opt = (cm, (n_P, n_S, -n_D, mp[1], a >= _CONTENTION * (mp[0] + snap)))
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


# ---------------------------------------------------------------- [portfolio] many-piece CP plan for few-head shapes (closed-form cost model, microseconds)
# The dual launch of a tail has H * m piece tasks; when H is small most CTAs would idle (H = 1: 1 task per sequence), so the tail is cut into m >> 2 pieces
# and the fixup chains the m states of a head (cp_fix.cu mode 1: ~3.3 us per item, one CTA per head) before one output-tile launch on all SMs (mode 2).
# Every term is a measured rate (work_portfolio/NOTES.md): head chain 1.42 us / 32-token chunk (M64: 2 CTAs per head) or 1.70 (M128: 1 CTA), dual chunk 1.80 us
# (2.38 once more than ~112 CTAs share HBM), fixup chain 8 + 3.3 m us, output tiles tail * H * 768 B at 5.8 TB/s + 15 us, 6 us of launch gaps.
_MP_HEAD_US = {64: 1.42, 128: 1.70}
_MP_HEAD_FIXED = 12.0
_MP_DUAL_US, _MP_DUAL_US_BUSY, _MP_DUAL_BUSY_CTAS, _MP_DUAL_FIXED = 1.80, 2.38, 112, 8.0
_MP_CHAIN_FIXED, _MP_CHAIN_ITEM = 8.0, 3.3
_MP_DUAL_ITEM, _MP_HIER_ITEM, _MP_HIER_GAP = 4.3, 3.6, 3.0       # hierarchical chain (cp_fix.cu fixup_h): a two-accumulator group item, and the launch gap between its two stages
# [planref] the rates above are the v47 constants; `_MPK` holds the constant sets the planner actually reads: "single" = ONE sequence planned alone on the whole device (in-situ rates of work_dynfew:
# head 1.38 us / chunk for both variants, dual 1.87 / 1.90, chain item 4.5; 2-11% faster plans than the v47 constants), "multi" = one CP instance among several that run concurrently on their own SM budgets
# (_multi_plan / _prepare_multi); a two-instance plan measured 4.5% worse with the single-sequence set, so it keeps its own (see work_planref/NOTES.md).
_MPK = {
    "single": dict(head={64: 1.38, 128: 1.38}, head128_pad=1.38, head_fixed=12.0, dual=1.87, dual_busy=1.90, busy_ctas=112, dual_fixed=8.0, chain_fixed=8.0, chain_item=4.5, dual_item=4.3, hier_item=3.6, hier_gap=3.0),
    "multi": dict(head=dict(_MP_HEAD_US), head128_pad=_MP_HEAD_US[128], head_fixed=_MP_HEAD_FIXED, dual=_MP_DUAL_US, dual_busy=_MP_DUAL_US_BUSY, busy_ctas=_MP_DUAL_BUSY_CTAS, dual_fixed=_MP_DUAL_FIXED, chain_fixed=_MP_CHAIN_FIXED,
                  chain_item=_MP_CHAIN_ITEM, dual_item=_MP_DUAL_ITEM, hier_item=_MP_HIER_ITEM, hier_gap=_MP_HIER_GAP),
}
_HIER_MIN_PIECES = 8


def _chain_us(m, H, n_sms, kc="single"):
    """(us, group size s) of the state chain of m pieces per head: flat (s == 0: one chain of m items per head, fixup_mp mode 1) or hierarchical (groups of s pieces:
    stage 1 = group totals / the head group's true chain, stage 3 = group-entry chain over the totals + the group's own pieces)."""
    K = _MPK[kc]
    waves = -(-H // min(H, n_sms))
    best = (waves * (K['chain_fixed'] + K['chain_item'] * m), 0)
    if m >= _HIER_MIN_PIECES:
        for sz in range(2, m):
            G = -(-m // sz)
            if G < 3:
                continue
            last = m - (G - 1) * sz
            w1 = -(-(H * (G - 1)) // n_sms)
            t = w1 * (K['chain_fixed'] + K['dual_item'] * sz) + K['hier_gap'] + w1 * (K['chain_fixed'] - 2.0 + K['hier_item'] * (G - 2 + max(sz, last)))
            if t < best[0] * 0.95:
                best = (t, sz)
    return best
_MP_TILE_BW, _MP_TILE_FIXED, _MP_GAPS = 5.8e6, 15.0, 6.0
_MP_MARGIN = 0.10           # a many-piece plan must beat the best one / two-piece plan of the same model by this fraction
for _k in _MPK.values():
    _k.update(tile_bw=_MP_TILE_BW, tile_fixed=_MP_TILE_FIXED, gaps=_MP_GAPS, margin=_MP_MARGIN)
_MP_XQ = 4      # head length granularity (chunks)
_MP_MAX_HEADS = 37          # H <= n_sms / 4: beyond that the dual launch is filled by the head count alone and the 2-piece planner is the model of record


@functools.lru_cache(maxsize=1024)
def _m64_from(H, n_P):
    """Smallest head length (chunks) from which the base planner runs a head group of H tasks on n_P CTAs with the two-CTA M64 kernel (it picks M128 below: a short head cannot pay M64's launch cost)."""
    lo, hi = 1, 1 << 20
    if _base.plan([32 * hi], H, n_P)[0] != 64:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _base.plan([32 * mid], H, n_P)[0] == 64:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _mp_cost(tc, xc, H, n_sms, n_P, variant, pieces_n, kc="single"):
    """(us, pieces list) of a plan that runs the head x = xc chunks on n_P CTAs and cuts the remaining tail into m = pieces_n equal-ish dual pieces."""
    tail = tc - xc
    n_D = n_sms - n_P
    m = pieces_n
    q4, r4 = divmod(tail // 4, m)                    # pieces are multiples of 4 chunks (one fixup tile) except the last: tail // 4 quanta spread evenly, the remainder chunks on the last piece
    if q4 < 1:
        return None
    pieces = [4 * (q4 + (1 if i < r4 else 0)) for i in range(m)]
    pieces[-1] += tail % 4
    c = max(pieces)
    waves = -(-(H * m) // n_D)
    if waves > _PIECE_TASKS_PER_CTA:
        return None
    K = _MPK[kc]
    dual_us = K['dual'] if min(H * m, n_D) <= K['busy_ctas'] else K['dual_busy']
    dual = waves * (dual_us * c + K['dual_fixed'])
    hv = 64 if (variant == 64 and xc >= _m64_from(H, n_P)) else 128      # the head launch uses whatever variant the base planner picks for it
    head = (K['head128_pad'] if (hv == 128 and H % 8) else K['head'][hv]) * xc + K['head_fixed']
    chain = _chain_us(m, H, n_sms, kc)[0] if m > 2 else K['chain_fixed'] + K['chain_item'] * m
    tiles = K['tile_fixed'] + 32 * tail * H * 768 / K['tile_bw']
    fix = (chain + tiles + K['gaps']) if m > 2 else (chain * 0.6 + tiles)
    return max(head, dual) + fix, pieces, head >= dual


@functools.lru_cache(maxsize=4096)
def _mp_search(L, H, n_sms, kc="single"):
    """(best many-piece candidate, best one / two-piece candidate) of the closed-form model for ONE sequence of L tokens on n_sms SMs; candidate = (us, xc, n_P, m, pieces, p_first)."""
    tc = (L + 31) // 32
    best, best_few = None, None
    if H > _MP_MAX_HEADS or H > n_sms // 4 or tc < 64:
        return best, best_few
    for variant, n_P in ((128, H + H % 2), (64, 2 * H)):
        n_D = n_sms - n_P
        if n_D < 2:
            continue
        for m in sorted(set(list(range(1, 33)) + [int(32 * 1.15 ** k) for k in range(1, 30)])):
            if H * m > _PIECE_TASKS_PER_CTA * n_D:
                break
            lo, hi = 4, tc - 4 * m                    # bisection on the head length: head(xc) rises, the dual piece falls
            if hi <= lo:
                continue
            while hi - lo > 2:
                mid = (lo + hi) // 2
                r = _mp_cost(tc, mid, H, n_sms, n_P, variant, m, kc)
                if r is None or r[2]:
                    hi = mid
                else:
                    lo = mid
            for xc in range(max(4, lo - 3), min(tc - 4 * m, hi + 3) + 1):
                if xc % _MP_XQ:
                    continue
                r = _mp_cost(tc, xc, H, n_sms, n_P, variant, m, kc)
                if r is None:
                    continue
                cand = (r[0], xc, n_P, m, r[1], r[2])
                if m <= 2:
                    if best_few is None or cand[0] < best_few[0]:
                        best_few = cand
                elif best is None or cand[0] < best[0]:
                    best = cand
    return best, best_few


def mp_plan(seq_lens, H, n_sms, kc="single"):
    """Plan tuple (as cp_plan) with m >= 3 tail pieces for ONE sequence (N == 1) on few heads, or None (keep the one / two-piece planner)."""
    if len(seq_lens) != 1:
        return None
    best, best_few = _mp_search(seq_lens[0], H, n_sms, kc)
    if best is None or (best_few is not None and best[0] > best_few[0] * (1.0 - _MPK[kc]['margin'])):
        return None
    _, xc, n_P, m, pieces, p_first = best
    return (0, 32 * xc, n_P, 0, -(n_sms - n_P), tuple(pieces), p_first)


_MI_MIN_CHUNKS = 256        # [portfolio] multi-sequence CP: a sequence is cut only when it is at least this long (chunks of 32 tokens)
_MI_MAX_CUTS = 4
# [planref] a host-known layout can take the frozen device many-piece plan (dyn/dynmp.py: the planner runs once at prepare) instead of the host plans (multi-instance, two-piece, direct).  The device model's
# pred_us (its fixed cost `ovh` = planner + empty launches included; in situ it overrates a frozen plan by ~15%) must stay below _STATIC_MP_BAR x the best host-model cost (direct, multi-instance tau; the host
# models underrate in situ by ~10%).  51 layouts (work_planref/NOTES.md 2.3): regret against the better of the two 0.15% on average, 4.3% worst (always-device 1.16 / never 10.8 as sum of log ratios); bar and offset were
# fitted on 25 layouts and reproduce on 26 other layouts.  One sequence without checkpoints is judged against the host many-piece model (bar 1.05: 25 shapes, wins 1 x 1M -8%, 37 : 100000 -18%, no losses).
_STATIC_MP_BAR, _STATIC_MP_BAR1 = 1.26, 1.05


def _static_mp_wins(pred_us, seq_lens, H, n_sms, ck_n, tau):
    if len(seq_lens) == 1 and not ck_n:   # one sequence, no checkpoints: the host many-piece plan (mp_plan) is the competitor; the device plan wins only where it models within 5% of it (1 x 1M tokens -8%, 37 : 100000 -19%)
        cs = [c[0] for c in _mp_search(seq_lens[0], H, n_sms, "single") if c is not None]
        return bool(cs) and pred_us < _STATIC_MP_BAR1 * min(cs)
    cands = [_direct_cost(seq_lens, H, n_sms) * _STEP_US]
    if tau is not None:
        cands.append(tau)
    return pred_us < _STATIC_MP_BAR * min(cands)


_MI_MARGIN = 0.10           # must beat the one-cut bound (see _multi_plan) by this fraction


def _mi_cost(L, H, b):
    """Modelled us of ONE long sequence (cut by the closed-form planner) on a budget of b SMs."""
    cands = [c for c in _mp_search(L, H, b, "multi") if c is not None]
    if cands:
        return min(c[0] for c in cands)
    return _MPK["multi"]["head"][64 if b >= 2 * H else 128] * ((L + 31) // 32) + _MPK["multi"]["head_fixed"]     # no cut possible: the whole chain


def _multi_plan(seq_lens, H, n_sms):
    """[portfolio] Several long sequences on few heads: the one-cut planner leaves the second longest sequence a single chain.  Cut the k longest sequences, each as its own
    CP instance on its own SM budget (closed-form planner of that sequence); the sequences between them run as direct launches on their budget.  Returns
    [(first seq, end seq, SM budget, cut)] in sequence order or None.  Budgets minimise the modelled makespan (smallest tau such that the budgets that reach tau fit)."""
    _multi_plan.last_tau = None
    N = len(seq_lens)
    if H > _MP_MAX_HEADS or H > n_sms // 4 or N > 64 or min(seq_lens) <= 0:
        return None
    chunks = [(L + 31) // 32 for L in seq_lens]
    order = sorted(range(N), key=lambda i: -chunks[i])
    if chunks[order[0]] < _MI_MIN_CHUNKS:
        return None
    # what the one-cut planners cannot beat: the longest sequence cut into one / two pieces on ALL SMs, and the second longest run whole (M64 chain rate)
    few = _mp_search(seq_lens[order[0]], H, n_sms, "multi")[1]
    ref = max(few[0] if few is not None else 0.0, _MPK["multi"]["head"][64] * chunks[order[1]] + _MPK["multi"]["head_fixed"])
    budgets = sorted(set([2, 4, 8, 16, 24, 32] + [max(2, (n_sms * i // 16) // 2 * 2) for i in range(1, 17)]))
    best = None
    for k in range(1, min(N, _MI_MAX_CUTS) + 1):
        cut = set(order[:k])
        if chunks[order[k - 1]] < _MI_MIN_CHUNKS:
            break
        segs, i = [], 0
        while i < N:
            if i in cut:
                segs.append((i, i + 1, True)); i += 1
            else:
                j = i
                while j < N and j not in cut:
                    j += 1
                segs.append((i, j, False)); i = j
        tab = []        # per segment: [(budget, us)]
        for a, b_, is_cut in segs:
            row = []
            for bud in budgets:
                if is_cut:
                    row.append((bud, _mi_cost(seq_lens[a], H, bud)))
                else:
                    row.append((bud, _group_cost(seq_lens[a:b_], H, bud) * _STEP_US))
            tab.append(row)
        for tau in sorted(set(c for row in tab for _, c in row)):
            need = []
            for row in tab:
                ok = [bud for bud, c in row if c <= tau]
                if not ok:
                    break
                need.append(min(ok))
            else:
                if sum(need) <= n_sms:
                    if best is None or tau < best[0]:
                        spare = n_sms - sum(need)
                        best = (tau, [(a, b_, nb, c) for (a, b_, c), nb in zip(segs, need)], spare)
                    break
    _multi_plan.last_tau = None if best is None else best[0]      # modelled us of the multi-instance plan (tools / the choice against the frozen device plan)
    if best is None or best[0] > ref * (1.0 - _MI_MARGIN):
        return None
    tau, segs, spare = best
    out = []
    for idx, (a, b_, nb, c) in enumerate(segs):           # spare SMs go to the cut sequences (the longest chains), round robin in 2 SM steps
        out.append([a, b_, nb, c])
    cuts = [x for x in out if x[3]]
    while spare >= 2 and cuts:
        for x in cuts:
            if spare >= 2:
                x[2] += 2; spare -= 2
    return [tuple(x) for x in out]


def _prepare_multi(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cl, seq_lens, mi):
    """One CP instance per segment of mi (see _multi_plan): own SM budget, own private stream set, results written in place into slices of one out / final_state.
    All instances start from the launch stream and the launch stream joins them at the end (fork / join events: graph capture friendly)."""
    B, T, H, D = q.shape
    dev = q.device
    out = torch.empty_like(q)
    final_state = torch.empty(len(seq_lens), H, D, D, dtype=torch.float32, device=dev)
    insts = []
    for idx, (a, b_, nb, is_cut) in enumerate(mi):
        s0, s1 = cl[a], cl[b_]
        sl = lambda t: t[:, s0:s1]
        cu = None if b_ - a == 1 else torch.tensor([c - s0 for c in cl[a:b_ + 1]], dtype=torch.int64, device=dev)
        fn = _prepare_impl(sl(q), sl(k), sl(v), sl(g), sl(beta), A_log, dt_bias, scale, initial_state[a:b_], cu, None, nb, sl(out), final_state[a:b_], 1 + idx)
        insts.append((fn, _main_stream(dev, idx)))
    e0 = torch.cuda.Event()
    evs = [torch.cuda.Event() for _ in insts]

    padders = [getattr(fn, "pad_all", None) or getattr(fn, "pad_only", None) if getattr(fn, "has_pad", False) else None for fn, _ in insts]

    def launch():
        cur = torch.cuda.current_stream()
        for pf in padders:   # [shortmid] the padded-beta copies of every instance run before any big launch of any instance (see _prepare_impl: TPC sharing).
            if pf is not None:   # (Concurrent copies on the instance streams + cross-stream event waits were 1-9% slower than this sequential form: NOTES)
                pf()
        e0.record(cur)
        for (fn, st), ee, pf in zip(insts, evs, padders):
            st.wait_event(e0)
            with torch.cuda.stream(st):
                fn(pad=False) if pf is not None else fn()
            ee.record(st)
        for ee in evs:
            cur.wait_event(ee)
        return out, final_state

    return launch


_main = {}


def _main_stream(dev, idx):
    """Private stream idx of a multi-sequence launch (the instance's launch stream; its branch streams are _branch_streams(.., 1 + idx))."""
    key = (_dev_index(dev), idx)
    if key not in _main:
        _main[key] = _new_stream_on(_fixmod(), dev)
    return _main[key]


def _prepare_impl(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, fx=None, n_sms=None, out=None, final_state=None, stream_set=0):
    """fx [contract]: None (benchmark path: sequence n = row n of contiguous [N, H, D, D] states), or the production-contract context
    {"src": (initial-state tensor or pool, idx | None, slot stride), "dst": (pool | None, idx | None, slot stride), "ck": (buffer, ck_cu, n // 32) | None}:
    the P / S launches address the pool in place through slices of the slot tables; the cut sequence's head state S_A and the fixup result go through
    private scratch and are scattered into their pool slots at the end (index_copy)."""
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
    if fx is None and tuple(initial_state.shape) != (len(cl) - 1, H, D, D):
        raise ValueError(f"KDA kernel: initial_state must be [N, H, 128, 128] with N = {len(cl) - 1}, got {tuple(initial_state.shape)}")
    seq_lens = [cl[i + 1] - cl[i] for i in range(len(cl) - 1)]
    N = len(seq_lens)
    n_sms = n_sms or _base._num_sms(dev)    # [portfolio] n_sms / out / final_state / stream_set: one instance of a multi-sequence CP launch (SM budget, slices of the shared result buffers)
    if fx is None and out is None and final_state is None and stream_set == 0 and n_sms == _base._num_sms(dev):   # [tiny2] every sequence fits one chunk: the bespoke one-chunk kernel (cost-model gated)
        tl = _base.tiny_prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, seq_lens, n_sms)
        if tl is not None:
            return tl
    ck_n = fx["ck"][2] if fx is not None and fx["ck"] is not None else 0

    def static_mp():
        # host-known layout: the device many-piece planner runs once here (frozen plan, no planner kernel per launch); None = it does not cut this layout, the host plans apply
        if out is not None or not _dynmp.static_eligible(N, H, T, n_sms, ck_n):
            return None
        if fx is None:
            src, dst, ck = (initial_state, None, H * D * D), (None, None, H * D * D), None
        else:
            src, dst, ck = fx["src"], fx["dst"], fx["ck"]
        inner = _dynmp.prepare(_base, _dual, q, k, v, g, beta, A_log, dt_bias, scale, src, dst, None, val=None, ck=ck, frozen=True, cu_host=cl)
        return inner

    mi, tau = None, None
    if fx is None and N >= 2 and out is None and n_sms == _base._num_sms(dev):
        mi = _multi_plan(seq_lens, H, n_sms)
        tau = _multi_plan.last_tau
    inner = static_mp()   # [planref] frozen device many-piece plan when its model beats the host plans'
    if inner is not None and _static_mp_wins(inner.pred_us[0], seq_lens, H, n_sms, ck_n, tau):
        return inner
    if mi is not None:
        return _prepare_multi(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cl, seq_lens, mi)
    plan = None
    if not ck_n:   # [portfolio] many-piece tail (state chain + tile launches), few-head single-sequence shapes
        plan = mp_plan(seq_lens, H, n_sms, "multi" if out is not None else "single")
    if plan is None:
        plan = cp_plan(seq_lens, H, n_sms, ck_n)
    if plan is None:
        if fx is None:
            return _base.prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, nopart=True, **({"n_sms": n_sms, "out": out, "final_state": final_state} if out is not None else {}))
        return _base.prepare(q, k, v, g, beta, A_log, dt_bias, scale, fx["src"][0], cu_seqlens, final_state=fx["dst"][0] if fx["dst"][0] is not None else None,
                             feat={"src": fx["src"][1:], "dst": fx["dst"][1:], "ck": fx["ck"]})
    j, xl, n_P, n_S, n_B, pieces, p_first = plan
    if ck_n and n_B >= 0:
        n_B = -(n_sms - n_P - n_S)
    if _FORCE_PIECES:                              # tests / experiments: force the tail pieces (chunks per piece), keeping the planner's SM budgets
        pieces = list(_FORCE_PIECES)
        xl = seq_lens[j] - 32 * sum(pieces)
        n_B = -(n_sms - n_P)
    x = cl[j] + xl                                     # absolute token index of the split
    e = cl[j + 1]                                      # end of the split sequence
    tail = e - x
    if out is None:
        out = torch.empty_like(q)
    if final_state is None:
        final_state = torch.empty(N, H, D, D, dtype=torch.float32, device=dev) if fx is None or fx["dst"][0] is None else fx["dst"][0]
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
    split_fix = m_p > 2 and not ck_n   # [portfolio] many-piece tail: state chain launch + output-tile launch (cp_fix.cu modes 1 / 2)
    sin_ = torch.empty(max(m_p - 1, 0), H, D, D, dtype=torch.bfloat16 if split_fix else torch.float32, device=dev)   # [portfolio] many-piece tails: bf16(S_in) staging of the split fixup
    qt = torch.empty(1, tail, H, D, dtype=q.dtype, device=dev)
    vz = torch.zeros(1, tail, H, D, dtype=v.dtype, device=dev)
    s0z = torch.zeros(m_p, H, D, D, dtype=torch.float32, device=dev)
    eye = torch.eye(D, dtype=torch.float32, device=dev).expand(m_p, H, D, D).contiguous()
    i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)
    hd = lambda t: t[:, :x]
    tl = lambda t: t[:, x:e]
    sf = lambda t: t[:, e:]
    if fx is None:
        fP = fS = None
        src_all = initial_state
        sl_src = lambda a, b: initial_state[a:b]
    else:
        pool_dst = fx["dst"][0] is not None
        sl = lambda t, a, b: None if t is None else t[a:b]
        s_t, s_idx, s_stride = fx["src"]
        d_t, d_idx, d_stride = fx["dst"]
        ck = fx["ck"]
        fP = {"src": (sl(s_idx, 0, j + 1), s_stride), "dst": (None, H * D * D), "ck": (ck[0], ck[1][:j + 1], ck[2]) if ck else None}
        fS = {"src": (sl(s_idx, j + 1, N), s_stride), "dst": (sl(d_idx, j + 1, N), d_stride) if pool_dst else (None, H * D * D),
              "ck": (ck[0], ck[1][j + 1:], ck[2]) if ck else None}
        sl_src = (lambda a, b: s_t) if s_idx is not None else (lambda a, b: s_t[a:b])   # no slot table: sequence n = row n, so a sub-launch needs the row-sliced view
    LP = _base.prepare(hd(q), hd(k), hd(v), hd(g), hd(beta), A_log, dt_bias, scale, sl_src(0, j + 1),
                       i64(cl[:j + 1] + [x]) if j > 0 else None, out=hd(out), final_state=sA, n_sms=n_P, feat=fP, nopart=(fx is None))
    LS = None
    suffix_empty = j + 1 < N and cl[N] == e                 # only zero-length sequences after j: final state = initial state
    if j + 1 < N and not suffix_empty:
        LS = _base.prepare(sf(q), sf(k), sf(v), sf(g), sf(beta), A_log, dt_bias, scale, sl_src(j + 1, N),
                           i64([c - e for c in cl[j + 1:]]) if (N - j - 1 > 1 or cl[N] != T) else None, out=sf(out),
                           final_state=final_state if (fx is not None and pool_dst) else final_state[j + 1:], n_sms=n_S, feat=fS, nopart=(fx is None))
    # [contract] checkpoints inside the tail: boundary chunk b = k * ck_n of the sequence, k0 <= k <= kmax.  b == xc is the head state S_A and a piece start is the
    # chained S_in(p): plain copies after the fixup; every other boundary is a snapshot (S_loc, P) of the dual launch composed by a fixup item.
    snap_items, copy_sA, copy_sin, snapL, snapP, cki_t = [], [], [], None, None, None
    if ck_n:
        xc, tc_j = xl // 32, (seq_lens[j] + 31) // 32
        k0, kmax = -(-xc // ck_n), (tc_j - 1) // ck_n
        pb = [xc]
        for c in (pieces or [tc_j - xc]):
            pb.append(pb[-1] + c)
        for kk in range(k0, kmax + 1):
            b = kk * ck_n
            p = max(i for i in range(len(pb) - 1) if pb[i] <= b)
            if b == xc:
                copy_sA.append(kk)
            elif b == pb[p]:
                copy_sin.append((kk, p - 1))
            else:
                snap_items.append((kk - k0, p, kk, 0))   # rows are relative to the cut sequence's first checkpoint row (read on the device: replays may change the table)
        if snap_items:
            snapL = torch.empty(kmax - k0 + 1, H, D, D, dtype=torch.float32, device=dev)
            snapP = torch.empty_like(snapL)
            cki_t = torch.tensor(snap_items, dtype=torch.int32, device=dev)
    ck_copy_t = torch.tensor([(r_, -1) for r_ in copy_sA] + [(r_, pi_) for r_, pi_ in copy_sin], dtype=torch.int32, device=dev) if (copy_sA or copy_sin) else None
    ck_base = fx["ck"][1][j:j + 1] if ck_n else None
    if n_B < 0:   # [architect4] dual-chain tail: L (v, S0 = 0) and Tr (v = 0, S0 = I) share one prep in one launch on -n_B CTAs
        LL = _dual.prepare(tl(q), tl(k), tl(v), tl(g), tl(beta), A_log, dt_bias, scale, s0z, eye, tl(out), qt, sL, sP, -n_B,
                           l_zero=True, t_eye=True, pieces=pieces, piece_cost=(_DUAL_F, _DUAL_TASK),
                           ck=(snapL, snapP, ck_n, xc, k0) if snap_items else None, nopart=(fx is None))
        LT = None
    else:
        LL = _base.prepare(tl(q), tl(k), tl(v), tl(g), tl(beta), A_log, dt_bias, scale, s0z, None,
                           out=tl(out), final_state=sL, n_sms=n_B, nopart=(fx is None))
        LT = _base.prepare(tl(q), tl(k), vz, tl(g), tl(beta), A_log, dt_bias, scale, eye, None,
                           out=qt, final_state=sP, n_sms=n_B, nopart=(fx is None))
    use_fixck = (bool(snap_items) or (fx is not None and pool_dst)) and not split_fix      # [contract] the feature fixup: checkpoint items and / or the pool slot written directly (a split fixup stages the final state: sfix + slot copy)
    fix = _fixmod(use_fixck)
    fs = _fixsched()
    seg_l, segp_l = fs.fix_schedule(H, tail, n_sms, _FIX_STATE_TILES if m_p == 1 else _FIX_OWNER_TILES, pt, _FIX_CHAIN_TILES,
                                   extra=len(snap_items) * _FIX_STATE_TILES)
    assert not (split_fix and use_fixck)
    if split_fix:   # mode 1 = state chain (one CTA per head up to the SM count), mode 2 = output tiles on all SMs
        seg_l, segp_l = fs.tile_schedule(H, pt, n_sms)
        chain_ptr = torch.zeros(min(H, n_sms) + 1, dtype=torch.int32, device=dev)
        hier_s = _chain_us(m_p, H, n_sms, "multi" if out is not None else "single")[1]
        if hier_s:
            hier_G = -(-m_p // hier_s)
            qtot = torch.empty(hier_G, H, D, D, dtype=torch.float32, device=dev)
            rtot = torch.empty(hier_G, H, D, D, dtype=torch.float32, device=dev)
            e1 = torch.empty(H, D, D, dtype=torch.float32, device=dev)
    seg_t = torch.tensor(seg_l, dtype=torch.int32, device=dev)
    segp_t = torch.tensor(segp_l, dtype=torch.int32, device=dev)
    # private streams (not from torch's pool: a pool stream can coincide with the capturing stream and serialize the branches)
    streams = _branch_streams(fix, dev, stream_set)
    e0 = torch.cuda.Event()
    ev = [torch.cuda.Event() for _ in range(3)]
    out_tail = out[0, x:e]
    if fx is not None:
        sfix = torch.empty(H, D, D, dtype=torch.float32, device=dev) if pool_dst else None      # TMA-store target of the (unused) final-state item when the pool slot is written directly

    # [shortmid] padded-beta copies (H % 8 != 0 launches without a beta-direct build): the copy is a 296-CTA kernel; while it still drains, the concurrent big launches
    # (1024 threads, one whole SM each) place their CTAs scattered, so a CTA of one kernel ends up on the second SM of a TPC whose first SM runs a CTA of ANOTHER kernel,
    # and both run ~2x slower (per-CTA stamps, NOTES section 3): plan times varied by 15-40% with the launch order (the "x % 4" effect).
    # Mode 1 (the head has no copy and exactly one sibling has): the head launches first (p_first) or waits for the sibling's copy.  Mode 2: every copy of the launch runs
    # before any big launch (also when the caller, `_prepare_multi`, runs the copies of all its instances first: launch(pad=False)).  H % 8 == 0 launches have no copy and
    # keep the model's order unchanged.
    pad_sibs = [fn for fn in (LL, LT, LS) if fn is not None and getattr(fn, "has_pad", False)]
    pad_head = getattr(LP, "has_pad", False)
    pad_fns = ([LP] if pad_head else []) + pad_sibs
    pad_mode = 0 if not pad_fns else (1 if (not pad_head and len(pad_sibs) == 1) else 2)
    epads = [torch.cuda.Event() for _ in pad_sibs]

    def pad_all():
        for fn in pad_fns:
            fn.pad_only()

    def launch_eager(pad=True):
        cur = torch.cuda.current_stream()
        mode, kw = pad_mode, {}
        if pad_mode and not pad:
            mode = 2
        if mode == 2:
            if pad:
                pad_all()
            kw = {"pad": False}
        e0.record(cur)
        if p_first:     # the group with the longest modelled makespan is launched first (graph branches start in capture order, ~3 us apart)
            LP(**kw)
        for s in streams:
            s.wait_event(e0)
        for s, ee, fn in zip(streams, ev, (LL, LT, LS)):
            if fn is not None:
                with torch.cuda.stream(s):
                    if mode == 1 and not p_first and fn in pad_sibs:
                        fn(pad_done=epads[pad_sibs.index(fn)])
                    else:
                        fn(**kw)
                    ee.record(s)
        if not p_first:
            if mode == 1:
                for ep in epads:
                    cur.wait_event(ep)
            LP(**kw)
        if fx is None:
            if j > 0:
                final_state[:j].copy_(sA[:j])
            if suffix_empty:
                final_state[j + 1:].copy_(initial_state[j + 1:])
        else:
            if j > 0:
                if pool_dst:   # [contract2] slot-indexed copy (a rejected slot points at the private trash window, which index_copy_ cannot address)
                    _caux().copy_rows(sA, H * D * D, None, 0, final_state, d_stride, d_idx[:j], 0, j, H)
                else:
                    final_state[:j].copy_(sA[:j])
            if suffix_empty:   # only zero-length sequences follow: their final state = their initial state (identical slot of the pool: nothing to do)
                if fx["src"][0] is not final_state or fx["src"][1] is not fx["dst"][1]:
                    _caux().copy_rows(s_t, s_stride, s_idx[j + 1:] if s_idx is not None else None, j + 1, final_state, d_stride if pool_dst else H * D * D,
                                      d_idx[j + 1:] if pool_dst else None, j + 1, N - j - 1, H)
        for ee, fn in zip(ev, (LL, LT, LS)):
            if fn is not None:
                cur.wait_event(ee)
        if use_fixck:
            fix.fixup(out_tail, qt, sA[j], sP, sL, sin_, sfix if pool_dst else final_state[j], seg_t, segp_t, H, tail, m_p,
                      fx["ck"][0] if snap_items else None, snapP, snapL, cki_t, final_state if pool_dst else None, d_idx if pool_dst else None, d_stride if pool_dst else 0, j, ck_base)
        elif split_fix:
            fin = sfix if (fx is not None and pool_dst) else final_state[j]
            if hier_s:
                fix.fixup_h(sA[j], sP, sL, qtot, rtot, e1, sin_, fin, H, m_p, hier_s, hier_G, 1, n_sms)
                fix.fixup_h(sA[j], sP, sL, qtot, rtot, e1, sin_, fin, H, m_p, hier_s, hier_G, 3, n_sms)
            else:
                fix.fixup_mp(out_tail, qt, sA[j], sP, sL, sin_, fin, seg_t, chain_ptr, H, tail, m_p, 1)
            fix.fixup_mp(out_tail, qt, sA[j], sP, sL, sin_, fin, seg_t, segp_t, H, tail, m_p, 2)
            if fx is not None and pool_dst:
                _caux().copy_rows(sfix, H * D * D, None, 0, final_state, d_stride, d_idx[j:j + 1], 0, 1, H)
        else:
            fix.fixup(out_tail, qt, sA[j], sP, sL, sin_, final_state[j], seg_t, segp_t, H, tail, m_p)
        if ck_copy_t is not None:   # checkpoint rows that are plain copies of S_A / a chained piece state: one launch, row = ck_cu[j] + relative row (device)
            _caux().copy_ck_rows(sA[j], sin_, fx["ck"][0], ck_base, ck_copy_t, H)

    def launch(pad=True):
        launch_eager(pad)
        return out, final_state

    launch.has_pad = pad_mode != 0
    launch.pad_all = pad_all
    return launch


def _prepare_plain(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
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


_caux_mod = None
last_status = None   # [contract2] int32 [1] device tensor of the most recent contract prepare(): table-validation bits of its last launch (see contract_aux.cu)


def _caux():
    global _caux_mod
    if _caux_mod is None:
        _caux_mod = _load("kda_contract_aux_build", os.path.join(_HERE, "build_caux.py")).build()
    return _caux_mod


_cfrozen_mod = None


def _cfrozen():   # [final6] D2: validation of the frozen one-sequence plans (own module, see contract_frozen.cu)
    global _cfrozen_mod
    if _cfrozen_mod is None:
        _cfrozen_mod = _load("kda_contract_frozen_build", os.path.join(_HERE, "build_cfrozen.py")).build()
    return _cfrozen_mod


_base.PAD_COPY_FACTORY = _dual.PAD_COPY_FACTORY = lambda: _caux().pad_copy   # [portfolio] beta padding (H % 8 != 0): one trivial kernel instead of torch's strided copy


_trash_cache = {}
_trash_lock = threading.Lock()


def _trash_slot(base, stride, H, dev):
    """A private H x 128 x 128 fp32 state window addressable as slot s of the pool `base` (address = base + s * stride floats, s an integer, possibly negative):
    rejected table entries are redirected there, so a bad slot can corrupt no pool slot and fault on no address.  Returns (s, buffer).
    The window only depends on (base address, slot stride, H): cached, so a serving loop over the same pool allocates it once."""
    key = (_dev_index(dev), base.data_ptr(), stride, H)
    hit = _trash_cache.get(key)
    if hit is not None:
        return hit
    step = stride * 4
    buf = torch.zeros((step + H * D_ * D_ * 4) // 4 + 16, dtype=torch.float32, device=dev)
    s = -(-(buf.data_ptr() - base.data_ptr()) // step)
    if abs(s) >= 2 ** 31:
        raise ValueError("KDA kernel: the state pool is too far from device scratch memory to be addressed by an int32 slot")
    with _trash_lock:   # [recover] concurrent prepare() calls: evict + insert atomically (next(iter(d)) raises when another thread inserts in between)
        if len(_trash_cache) >= 16:
            _trash_cache.pop(next(iter(_trash_cache)), None)
        _trash_cache[key] = (int(s), buf)
        return _trash_cache[key]


D_ = 128


def status_bits(status):
    """Bit mask of the table validation of the last launch (status tensor of a contract launch): 1 bad cu_seqlens (the launch then treats every sequence as empty),
    2 bad / repeated ssm_state_indices entry, 4 bad initial_state_indices entry, 8 checkpoint_cu_starts rows outside state_checkpoints."""
    return int(status[0]) | int(status[1])


def _table(t, name, N, live=False):
    if t is None:
        return None
    if t.dtype not in (torch.int32, torch.int64) or t.dim() != 1 or t.numel() != N or t.device.type != "cuda":
        raise ValueError(f"KDA kernel: {name} must be a 1-D CUDA int32 / int64 tensor with {N} entries, got {tuple(t.shape)} {t.dtype}")
    if live and not t.is_contiguous():   # [finalrev] a dynamic_layout launch re-reads the table at every replay: a private .contiguous() copy would be stale
        raise ValueError(f"KDA kernel: {name} must be contiguous with dynamic_layout=True (the launch reads it in place at every replay)")
    return t if t.is_contiguous() else t.contiguous()


def _cp2_on(N, H, T, n_sms):
    """[dynpath2] device-planned context parallelism for N >= 2 in dynamic launches: auto (default) = only when the most a cut could ever save (one sequence holding almost all tokens: its chunk count + 1
    minus the fluid share of all tasks, chunk-steps) exceeds 250 steps (425 us), i.e. a long total length on few heads: there the fixed cost (~45 us: planner, empty tail launches, cio head) is <= ~6% and the saving on
    skewed layouts is 8-35%; shorter totals lose 5-17% on every layout that is not cut (work_dynpath2 REPORT.md section 4)."""
    c = T / 32.0
    return (c + 1.0) - (c + N) * H / n_sms >= 250.0


def _single_seq_layout(cu_seqlens, T):
    """[dynpath] dynamic_layout with N == 1: replays keep the total token count, so the layout is cu = [0, T] -- fully determined by the shape.  Checked once at
    prepare time (not at launch); a padded cu_seqlens (cu[-1] < T) or a read that is not possible (prepare under capture) keeps the device-planned path."""
    if cu_seqlens is None:
        return True
    try:
        return [int(c) for c in cu_seqlens.tolist()] == [0, T]
    except Exception:
        return False


def _span(t):
    if t is None or t.numel() == 0:
        return None
    lo = t.data_ptr()
    return lo, lo + (sum((n - 1) * abs(st) for n, st in zip(t.shape, t.stride())) + 1) * t.element_size()


def _check_overlap(written, read_only):
    """[finalrev] A written buffer (state pool, checkpoint buffer) that shares memory with another operand (the committed-state source pool, the inputs, the other written buffer) made
    the result order dependent (sequence A's destination slot == sequence B's source slot) with no status bit: rejected on the host (storage ranges, exact)."""
    spans = [(n, _span(t)) for n, t in written]
    for i, (n, sp) in enumerate(spans):
        if sp is None:
            continue
        others = [(m, _span(t)) for m, t in read_only] + spans[:i] + spans[i + 1:]
        for m, so in others:
            if so is not None and sp[0] < so[1] and so[0] < sp[1]:
                raise ValueError(f"KDA kernel: {n} overlaps the memory of {m}: written buffers must not alias any other operand")


def _prepare_feat(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, ssm_state_indices, state_checkpoints, checkpoint_cu_starts,
                  checkpoint_every_n_tokens, initial_state_source, initial_state_indices, dynamic_layout, dynamic_fallback=False, host_plan_ok=True):
    """[contract] production-contract path: state pool addressed in place (slot stride, per-sequence slot tables read on the device), committed-state
    source pool, prefix-cache checkpoints written by the kernels at chunk boundaries (see base/kernel.py prepare(feat=...)).
    [contract2] Every launch starts with ONE device validation kernel (contract_aux.cu) that turns the caller's tables (cu_seqlens, ssm_state_indices,
    initial_state_indices, checkpoint_cu_starts; int32 or int64) into sanitized private copies; the kernels only ever see those.  No host sync."""
    B, T, H, D = q.shape
    dev = q.device
    _base._tls.bf = bool(checkpoint_every_n_tokens) and torch.is_tensor(state_checkpoints) and state_checkpoints.dtype == torch.bfloat16   # [bf16ckpt] picks the KDA_CKBF kernel flavors for everything prepared below (this thread)
    N = 1 if cu_seqlens is None else cu_seqlens.numel() - 1      # shapes only: dynamic_layout must not read the contents of any table on the host
    if N < 1:
        raise ValueError("KDA kernel: cu_seqlens must describe at least one sequence")
    pool = ssm_state_indices is not None
    _check_overlap([("the state pool", initial_state if pool else None), ("state_checkpoints", state_checkpoints if checkpoint_every_n_tokens else None)],
                   [("initial_state_source", initial_state_source), ("initial_state", None if pool else initial_state)] + list(zip("q k v g beta A_log dt_bias".split(), (q, k, v, g, beta, A_log, dt_bias))))
    init_user = None
    if initial_state is None:   # source-pool-only call: the destination is a fresh [N, H, 128, 128] buffer
        if initial_state_source is None or pool:
            raise ValueError("KDA kernel: initial_state is required unless only initial_state_source is given")
        initial_state = torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    if pool:
        if not (initial_state.dtype == torch.float32 and initial_state.dim() == 4 and tuple(initial_state.shape[1:]) == (H, D, D)):
            raise ValueError(f"KDA kernel: the state pool must be fp32 [P, {H}, 128, 128], got {tuple(initial_state.shape)} {initial_state.dtype}")
        if not (initial_state.stride()[1:] == (D * D, D, 1) and initial_state.stride(0) >= H * D * D and initial_state.stride(0) % 4 == 0
                and initial_state.data_ptr() % 16 == 0):
            raise ValueError("KDA kernel: pool slots must be dense [H, 128, 128] blocks, slot stride >= H*128*128 and a multiple of 4 floats, 16-byte aligned")
    else:
        if tuple(initial_state.shape) != (N, H, D, D):
            raise ValueError(f"KDA kernel: initial_state must be [N, H, 128, 128] with N = {N}, got {tuple(initial_state.shape)}")
        if not initial_state.is_contiguous():
            raise ValueError("KDA kernel: initial_state must be contiguous without ssm_state_indices")
        if initial_state.data_ptr() % 16:   # [finalrev] the cp.async state loads need 16 B alignment: stage a misaligned view (read-only here) in an aligned buffer, re-copied every launch
            init_user = initial_state
            initial_state = torch.empty_like(init_user)
    d_in = _table(ssm_state_indices, "ssm_state_indices", N, dynamic_layout)
    s_in = _table(initial_state_indices, "initial_state_indices", N, dynamic_layout)
    src_t, trash = initial_state, []
    if initial_state_source is not None:
        src_t = initial_state_source
        if not (src_t.dtype == torch.float32 and src_t.dim() == 4 and tuple(src_t.shape[1:]) == (H, D, D) and src_t.stride()[1:] == (D * D, D, 1)
                and src_t.stride(0) > 0 and src_t.stride(0) % 4 == 0 and src_t.data_ptr() % 16 == 0):
            raise ValueError("KDA kernel: initial_state_source must be fp32 [P0, H, 128, 128] with dense, 16-byte aligned slots")
        if s_in is None:
            raise ValueError("KDA kernel: initial_state_source needs initial_state_indices")
    elif s_in is not None:
        raise ValueError("KDA kernel: initial_state_indices without initial_state_source")
    d_idx = s_idx = None
    d_trash = s_trash = 0
    if pool:
        d_trash, tb = _trash_slot(initial_state, initial_state.stride(0), H, dev)
        trash.append(tb)
        d_idx = torch.empty(N, dtype=torch.int32, device=dev)
        s_idx = d_idx                    # in place: the source is the destination slot (a rejected slot is the trash window for both)
    if initial_state_source is not None:
        s_trash, tb = _trash_slot(src_t, src_t.stride(0), H, dev)
        trash.append(tb)
        s_idx = torch.empty(N, dtype=torch.int32, device=dev)
    dst_t = initial_state if pool else None
    ck = ck_in = ck_safe = ck_user = None
    if checkpoint_every_n_tokens:
        n = int(checkpoint_every_n_tokens)
        if n < 32 or n % 32:
            raise ValueError(f"KDA kernel: checkpoint_every_n_tokens must be a multiple of 32 (chunk size; the contract uses multiples of 64), got {n}")
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError("KDA kernel: checkpoint_every_n_tokens needs state_checkpoints and checkpoint_cu_starts")
        if not (state_checkpoints.dtype in (torch.float32, torch.bfloat16) and state_checkpoints.is_contiguous() and state_checkpoints.dim() == 4
                and tuple(state_checkpoints.shape[1:]) == (H, D, D)):
            raise ValueError("KDA kernel: state_checkpoints must be a contiguous fp32 or bf16 [C, H, 128, 128] tensor")
        if state_checkpoints.data_ptr() % 32:   # the checkpoint stores are 256-bit: a buffer that is not 32-byte aligned is staged through an aligned private copy (copied in and out around the launch)
            ck_user = state_checkpoints
            state_checkpoints = torch.empty(ck_user.shape, dtype=ck_user.dtype, device=dev)
        ck_in = _table(checkpoint_cu_starts, "checkpoint_cu_starts", N + 1, dynamic_layout)
        ck_safe = torch.empty(N + 1, dtype=torch.int64, device=dev)
        ck = (state_checkpoints, ck_safe, n // 32)
    cu_in = None
    if cu_seqlens is not None:
        cu_in = _table(cu_seqlens, "cu_seqlens", N + 1, dynamic_layout)
    cu_safe = torch.empty(N + 1, dtype=torch.int64, device=dev) if cu_in is not None else None
    status = torch.zeros(2, dtype=torch.int32, device=dev)   # [0]: table validation kernel, [1]: device planner (cu_seqlens); see status_bits()
    global last_status
    last_status = status
    n_tok = 32 * ck[2] if ck else 1
    C_rows = state_checkpoints.shape[0] if ck else 0
    aux = _caux()

    if dynamic_layout and N > _base.dev_planner_max_seq(_dev_index(dev)):
        # more sequences than the single-CTA device planner can hold: run() falls back to the host-planned path (host sync, any N); a graph-captured launch cannot
        if not dynamic_fallback:
            raise ValueError(f"KDA kernel: dynamic_layout=True supports at most {_base.dev_planner_max_seq(_dev_index(dev))} sequences per launch, got {N}")
        dynamic_layout = False
    cu_raw = cu_in
    if dynamic_layout and cu_in is not None and cu_in.dtype != torch.int64:   # the device planner reads an int64 table: convert (one tiny copy per launch)
        cu_raw = torch.empty(N + 1, dtype=torch.int64, device=dev)
    val = dict(N=N, T=T, cu=cu_raw, cu_out=cu_safe, d=d_in, d_out=d_idx, P_d=initial_state.shape[0] if pool else 0, d_trash=d_trash,
               s=s_in, s_out=s_idx if s_in is not None else None, P_s=src_t.shape[0] if (initial_state_source is not None) else 0, s_trash=s_trash,
               ck=ck_in, ck_out=ck_safe, C=C_rows, ck_tokens=n_tok, status=status)

    def validate():
        aux.validate_tables(val)

    if not dynamic_layout:
        validate()   # prepare-time pass: fills the sanitized tables the host planners below may read (the static path reads cu_seqlens / checkpoint rows on the host)
    fx = {"src": (src_t, s_idx, src_t.stride(0)), "dst": (dst_t, d_idx, initial_state.stride(0) if pool else H * D * D), "ck": ck}
    if dynamic_layout:   # [planner_graph] device-planned launch (base/kernel.py prepare(dynamic=True): cio / plain; dyn/dyncp.py: + context-parallel tail), no host reads
        inner = None
        host_validate = False
        # The table validation runs as a second block of the device planner's launch (concurrent with the planner, which sanitizes cu_seqlens itself): no extra launch.
        planner_extra = dict(val, pstat=status[1:], ident_validate=aux.validate_tables, fast=1)
        if not pool and initial_state_source is None and ck is None:
            # [tiny2] T <= 32 in total: one chunk per sequence.  One sequence: its layout is [0, T] by the shape -> the static one-chunk kernel; N >= 2: the bespoke kernel reads and validates
            # cu_seqlens itself (no planner launch)
            if N == 1 and _single_seq_layout(cu_seqlens, T):
                inner = _base.tiny_prepare(q, k, v, g, beta, A_log, dt_bias, scale, src_t, [T], _base._num_sms(dev))
            elif cu_raw is not None:
                inner = _base.tiny_dyn_prepare(q, k, v, g, beta, A_log, dt_bias, scale, src_t, cu_raw, N, _base._num_sms(dev), status[1:])
        if (N == 1 and ck is not None and _dynmp.eligible(1, H, T, _base._num_sms(dev), ck[2], N_min=1) and _single_seq_layout(cu_seqlens, T)):
            # [dynfew] one sequence with checkpoints on few heads: the many-piece plan (pieces replayed from their chained entering states) is a function of the shape alone: planned once, no planner kernel per launch;
            # the table validation (slot / checkpoint tables) is its own launch.  None = the planner never cuts this shape: the host plan below.
            inner = _dynmp.prepare(_base, _dual, q, k, v, g, beta, A_log, dt_bias, scale, fx["src"], fx["dst"], cu_raw, val=None, ck=ck, frozen=True)
            host_validate = inner is not None
        if inner is None and N == 1 and host_plan_ok and _single_seq_layout(cu_seqlens, T):
            nf1 = not pool and initial_state_source is None and ck is None
            # [dynpath] one sequence: the only possible layout is [0, T] (same total token count) -> host cost-model plan (incl. CP).  No planner kernel to carry the table validation:
            # with slot tables it runs as its own launch (host_validate); without any table there is nothing to validate (the kernels never read the caller's cu).
            inner = _prepare_impl(q, k, v, g, beta, A_log, dt_bias, scale, src_t, None, None if nf1 else fx)
            host_validate = not nf1
        skip_cp = False
        if (inner is None and N == 1 and not host_plan_ok
                and _dyncp.eligible(N, H, T, _base._num_sms(dev), ck[2] if ck else 0, dev) and _single_seq_layout(cu_seqlens, T)):
            # [dynpath2] run(), one sequence ([0, T] by the shape): the context-parallel plan is a function of the shape alone -> computed once per shape by the planner kernel, then every launch
            # of that shape carries no planner (table validation, when tables are in use, is its own launch as in the host-planned N == 1 path); None = this shape is never cut
            inner = _dyncp.prepare(_base, _dual, q, k, v, g, beta, A_log, dt_bias, scale, fx["src"], fx["dst"], cu_raw, val=None, ck=ck, frozen=True)
            if inner is not None:
                host_validate = pool or initial_state_source is not None or ck is not None
            else:   # this shape is never cut: the host-scheduled non-CP launch (M64 / M128 / cio by the base cost model; one sequence = a cheap plan), tables validated by their own launch
                skip_cp = True
                nf1 = not pool and initial_state_source is None and ck is None
                if _base.schedule([T], H, _base._num_sms(dev), not nf1)[0] == 64:   # the M64 kernel (two CTAs per task) wins for few heads: only the host-scheduled launch has it
                    inner = _base.prepare(q, k, v, g, beta, A_log, dt_bias, scale, fx["src"][0], None, final_state=fx["dst"][0] if fx["dst"][0] is not None else None,
                                          feat=None if nf1 else {"src": fx["src"][1:], "dst": fx["dst"][1:], "ck": ck})
                    host_validate = not nf1
        if (inner is None and not skip_cp and _dynmp.eligible(N, H, T, _base._num_sms(dev), ck[2] if ck else 0)):
            # [dynfew] N >= 2 on few heads: one launch serves every cut sequence with a device-planned many-piece plan (dyn/dynmp.py); with checkpoints the tail is replayed from the chained piece states;
            # the two-piece device CP path below keeps the other shapes
            inner = _dynmp.prepare(_base, _dual, q, k, v, g, beta, A_log, dt_bias, scale, fx["src"], fx["dst"], cu_raw, val=planner_extra, ck=ck)
        if (inner is None and not skip_cp and _dyncp.eligible(N, H, T, _base._num_sms(dev), ck[2] if ck else 0, dev)
                and (N == 1 or _cp2_on(N, H, T, _base._num_sms(dev))) and (ck is None or N == 1)):
            # [dynpath2] N >= 2: the device-planned CP path is opt-in (KDA_DYN_CP2=1): its fixed cost (general planner 15-50 us, empty dual / fixup launches, cio head) and its mediocre head balance cost
            # 5-17% on every layout that is not cut, more than the 5-22% it saves on strongly skewed ones (16 held-out layouts, work_dynpath2 REPORT.md); one sequence has a shape-only plan and keeps CP (frozen)
            # checkpoints: the device-planned CP path (planner + empty tail launches on every replay, ~+10% when no cut is chosen) is only worth it for one sequence (non-frozen: padded cu); N >= 2 keeps the plain / cio launch
            inner = _dyncp.prepare(_base, _dual, q, k, v, g, beta, A_log, dt_bias, scale, fx["src"], fx["dst"], cu_raw, val=planner_extra, ck=ck)
        if inner is None:
            nofeat = not pool and initial_state_source is None and ck is None
            # [dynpath] no contract feature in use: the default (benchmark) builds read the same device task tables and have no slot-table loads (FEAT builds cost ~1.5-2%)
            inner = _base.prepare(q, k, v, g, beta, A_log, dt_bias, scale, src_t, cu_raw, final_state=dst_t,
                                  feat=None if nofeat else {"src": fx["src"][1:], "dst": fx["dst"][1:], "ck": ck}, dynamic=True, val=planner_extra)
    else:
        torch.cuda.synchronize()
        st = status_bits(status)
        if st:   # the static (host-planned) path reads cu_seqlens and the checkpoint rows on the host: reject bad tables loudly right here
            raise ValueError(f"KDA kernel: invalid cu_seqlens / slot / checkpoint tables (status bits {st}: 1 cu_seqlens, 2 ssm_state_indices out of range or repeated, "
                             "4 initial_state_indices out of range, 8 checkpoint_cu_starts rows outside state_checkpoints)")
        inner = _prepare_impl(q, k, v, g, beta, A_log, dt_bias, scale, src_t, cu_in, fx)   # [finalrev] the validated contiguous device table (was the caller's tensor: a strided / CPU view was read through its raw pointer)

    if dynamic_layout and host_validate and N == 1 and cu_raw is not None:
        cfz = _cfrozen()
        def validate():   # [final6] D2: the frozen one-sequence plans never read cu_seqlens: their validation launch rejects a replay whose cu != [0, T] (a padded bucket) instead of letting the plan run on it
            cfz.validate_tables_frozen(val)
    pi = getattr(inner, "pad_info", None) if dynamic_layout else None
    if pi is not None and host_validate:   # [long] the standalone validation launch carries the inner launch's padded-beta copy
        val["pad_src"], val["pad_dst"] = pi
        inner.pad_skip[0] = True

    def launch():
        if init_user is not None:
            initial_state.copy_(init_user)
        if cu_raw is not cu_in:   # int32 cu_seqlens: the device planner / kernels read an int64 table
            cu_raw.copy_(cu_in)
        if (not dynamic_layout or host_validate):   # the device-planned dynamic path validates inside its planner kernel (one launch fewer)
            validate()
        if ck_user is not None:
            state_checkpoints.copy_(ck_user)
        out, fin = inner()
        if ck_user is not None:
            ck_user.copy_(state_checkpoints)
        return (out, fin, ck_user if ck_user is not None else state_checkpoints) if checkpoint_every_n_tokens else (out, fin)

    launch.status = status
    launch.keepalive = trash
    launch.tables = getattr(inner, "tables", None)   # [dynfew] tools: the many-piece planner's tables
    launch.pred_us = getattr(inner, "pred_us", None)   # [long] tools: frozen many-piece plans: (modelled us of the plan, of no cut)
    return launch


def _bind_device(launch, idx):
    """[multi-GPU] launch() runs with device idx current (torch current stream / events, cudaGetDevice() of the extensions, streams created lazily), whatever the caller's current device is;
    the caller's device is restored afterwards.  Attributes of launch (status, keepalive, tables) are kept.
    Cost: one torch._C._cuda_exchangeDevice call (~1.3 us from Python) per launch, so it is only installed when the process can see more than one device
    (the C++ launch functions hold their own CUDAGuard in every case)."""
    if torch.cuda.device_count() <= 1:
        return launch
    xd = torch._C._cuda_exchangeDevice

    def bound():
        prev = xd(idx)
        try:
            return launch()
        finally:
            if prev != idx:
                xd(prev)

    bound.__dict__.update(launch.__dict__)
    return bound


def prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, **kw):
    """Public entry: _prepare_api() with the tensors' device current, both at prepare time (scratch buffers, streams, planner init, syncs) and in the returned launch()."""
    idx = _dev_index(q.device)
    for t in (k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, *[x for x in kw.values() if torch.is_tensor(x)]):   # [multi-GPU] every CUDA operand must live on q's device
        if t is not None and t.is_cuda and t.device != q.device:
            raise ValueError(f"KDA kernel: all CUDA tensors must be on the device of q (cuda:{idx}), got one on {t.device}")
    with torch.cuda.device(idx):
        launch = _prepare_api(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, **kw)
    return _bind_device(launch, idx)


def _check_args(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, dynamic_layout):
    """[finalrev] ValueErrors for operands the kernels index without looking: every tensor is addressed with q's T / H (a short A_log / dt_bias / k / beta was read out of bounds); a
    non-fp32 initial_state raised a RuntimeError deep inside; a CPU / strided cu_seqlens of a STATIC launch was read through its raw device pointer (now a private contiguous device copy:
    a static launch's layout is fixed at prepare); a dynamic launch re-reads cu_seqlens at every replay, so it must be a contiguous CUDA tensor.  Returns the cu_seqlens to use."""
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("A_log", A_log), ("dt_bias", dt_bias)):
        if not torch.is_tensor(t) or t.device.type != "cuda":
            raise ValueError(f"KDA kernel: {name} must be a CUDA tensor")
    if q.dim() != 4 or q.shape[0] != 1 or q.shape[3] != 128:
        raise ValueError(f"KDA kernel: expected q of shape [1, T, H, 128], got {tuple(q.shape)}")
    T, H = q.shape[1], q.shape[2]
    for name, t in (("k", k), ("v", v), ("g", g)):
        if tuple(t.shape) != tuple(q.shape):
            raise ValueError(f"KDA kernel: {name} must have the shape of q {tuple(q.shape)}, got {tuple(t.shape)}")
    if beta.dim() < 2 or beta.shape[0] != 1 or beta.shape[1] != T or beta.numel() != T * H:
        raise ValueError(f"KDA kernel: beta must be [1, T, H] = [1, {T}, {H}], got {tuple(beta.shape)}")
    if A_log.numel() != H or dt_bias.numel() != H * 128:
        raise ValueError(f"KDA kernel: A_log must have H = {H} entries and dt_bias H * 128 = {H * 128}, got {A_log.numel()} and {dt_bias.numel()}")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError(f"KDA kernel: initial_state / the state pool must be fp32, got {initial_state.dtype}")
    if cu_seqlens is not None:
        if not torch.is_tensor(cu_seqlens) or cu_seqlens.dim() != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("KDA kernel: cu_seqlens must be a 1-D int32 / int64 tensor")
        if cu_seqlens.device != q.device or not cu_seqlens.is_contiguous():
            if dynamic_layout:
                raise ValueError("KDA kernel: cu_seqlens must be a contiguous tensor on q's device with dynamic_layout=True (it is read in place at every replay)")
            cu_seqlens = cu_seqlens.to(q.device).contiguous()
    return cu_seqlens


def _prepare_api(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, *, ssm_state_indices=None, state_checkpoints=None,
            checkpoint_cu_starts=None, checkpoint_every_n_tokens=0, initial_state_source=None, initial_state_indices=None, dynamic_layout=False, dynamic_fallback=False, _host_plan_ok=True):
    """Serving entry (program.md "Production contract").  Without contract features this is the benchmark path, unchanged; with a state pool /
    checkpoints / source pool the same kernels run as their KDA_FEAT builds.  Inputs are normalised as in _prepare_plain (private contiguous copy per launch)."""
    cu_seqlens = _check_args(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, dynamic_layout)
    if ssm_state_indices is None and not checkpoint_every_n_tokens and initial_state_source is None and not dynamic_layout:
        return _prepare_plain(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)
    if q.shape[1] == 0:   # [contract2] no tokens at all: every sequence is empty (states copied through, no checkpoint rows, empty output).  The kernels' TMA maps need a
        # non-empty token axis, so the launch runs on 32 zero tokens with all-empty sequences and returns the empty slice.
        q, k, v, g, beta = [torch.zeros((t.shape[0], 32) + tuple(t.shape[2:]), dtype=t.dtype, device=t.device) for t in (q, k, v, g, beta)]
        cu_seqlens = torch.zeros(2 if cu_seqlens is None else cu_seqlens.numel(), dtype=torch.int64, device=q.device)
        inner0 = prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, ssm_state_indices=ssm_state_indices, state_checkpoints=state_checkpoints,
                         checkpoint_cu_starts=checkpoint_cu_starts, checkpoint_every_n_tokens=checkpoint_every_n_tokens, initial_state_source=initial_state_source,
                         initial_state_indices=initial_state_indices, dynamic_layout=dynamic_layout, dynamic_fallback=dynamic_fallback)

        def launch0():
            res = inner0()
            return (res[0][:, :0],) + tuple(res[1:])

        launch0.status = getattr(inner0, "status", None)
        return launch0
    spec = ((q, torch.bfloat16), (k, torch.bfloat16), (v, torch.bfloat16), (g, torch.bfloat16), (beta, torch.bfloat16),
            (A_log, torch.float32), (dt_bias, torch.float32))
    ins, pairs = [], []
    for t, dt in spec:
        if t.dtype != dt or not t.is_contiguous() or t.data_ptr() % 16:
            c = torch.empty(t.shape, dtype=dt, device=t.device)
            pairs.append((c, t))
            t = c
        ins.append(t)
    inner = _prepare_feat(*ins, scale, initial_state, cu_seqlens, ssm_state_indices, state_checkpoints, checkpoint_cu_starts,
                          checkpoint_every_n_tokens, initial_state_source, initial_state_indices, dynamic_layout, dynamic_fallback, _host_plan_ok)
    if not pairs:
        return inner

    def launch():
        for c, t in pairs:
            c.copy_(t)
        return inner()

    launch.status = getattr(inner, "status", None)
    return launch


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, **prod):
    return prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, **prod)()
