"""[dynfew] Graph-safe MANY-PIECE context-parallel launch for few-head shapes (dynamic_layout=True / run()): N >= 2 sequences, or one sequence with checkpoints.

One planner kernel (plan_mp.cu: closed-form cost model from cu_seqlens on the GPU; for one sequence it runs once per shape) decides which sequences are cut, where, and into how many dual
pieces, and writes every table the launches below read; the graph never changes between replays:
  planner -> [head group: cio launch over (sequence, head) tasks: the first x chunks of every cut sequence hand their state S_A to scratch slot (first piece id) * H + head, uncut sequences run whole]
          || [tail group: ONE dual-chain launch over the pieces of ALL cut sequences (cu_tail = piece list with dummy gap pieces)]
          -> state chains (cp_fix_mpd.cu: flat chain per (cut sequence, head) = mode 1, hierarchical = fixup_hd stages 1 and 3)
          -> no checkpoints: output tiles of every piece (mode 2: o_loc + Qt . bf16(S_in), final state direct-stored into the pool slot by the chain);
             checkpoints: the fp32 entering state of every piece is stored by the chain and ONE plain (FEAT + CK) launch replays the pieces as pseudo-sequences (exact outputs, checkpoint rows, final state).
Without a cut in a replay every tail table is empty and the extra launches exit at once.  State pool / source pool supported."""
import os, importlib.util
import threading
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PHL = True   # the chain kernels (cp_fix_mpd.cu) read P as bf16 hi / lo planes
KC_MAX = 32           # cut sequences (= sequences) per launch
MAX_HEADS = 37        # model validity: H <= n_sms / 4 (the static many-piece planner's bound)
MAX_TASKS = 1280      # sequences x heads the planner holds
PAR_DEFAULT = dict(a_head=1.38, f_task=4.5, f_headl=10.0, a_dual=1.87, a_dual_busy=1.90, f_dual=10.0, ch_fixed=8.0, ch_item=4.5, tile_bw=5.8e6, tile_fixed=15.0, gaps=6.0,
                   ovh=15.0, margin=0.10, whole_pack=1.0, hier_item=4.2, dual_item=4.3, hier_gap=3.0, ck_row_us=1.9, dual_busy=112, hier_min=8, hier_on=1, force_x=0, force_cp=0, wrap=1, wrap_pen=5.0, wrap_tau=1.0, stages=2, split=1.0)
CK_CHAIN_ITEM, CK_CHAIN_FIXED = 2.9, 10.0   # [replay] flat checkpoint-mode chain (cp_fix_mpd v2): item period / fixed us; passed to the planner in the tile_bw / tile_fixed slots (unused in checkpoint mode)   # rates measured on B200 (work_dynfew/tools/rates.py)
_PAR_KEYS = list(PAR_DEFAULT)


CK_PAR = {"a_dual": 1.55, "a_dual_busy": 1.70}   # [replay] checkpoint-mode overrides of the cost-model constants (state-only dual, flat chain): see calibration notes in NOTES.md E12


def _par(ck=False):
    p = dict(PAR_DEFAULT)
    if ck:
        p.update(CK_PAR)
    if ck:   # [replay] checkpoint mode: the planner reads the flat-chain constants from tile_bw / tile_fixed (the parameter struct must not grow)
        p["tile_bw"] = float(CK_CHAIN_ITEM); p["tile_fixed"] = float(CK_CHAIN_FIXED)
    return [float(p[k]) for k in _PAR_KEYS]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mods = {}


def _m(name):
    if name not in _mods:
        _mods[name] = _load("kda_pg_dmp_" + name, os.path.join(_HERE, {"plan": "build_planmp.py", "fix": "build_fixmpd.py", "fixdyn": "build_fix.py"}[name])).build()
    return _mods[name]


_streams = {}
_cstreams = {}
_frozen_cache = {}


def _chain_streams(key):
    """[long] two more private streams per device for the concurrent chain launches (stage 3 of the hierarchical chain, the flat chain + checkpoint tables)."""
    if key not in _cstreams:
        with torch.cuda.device(key):
            _cstreams[key] = [torch.cuda.ExternalStream(_m("fixdyn").new_stream(), device=key) for _ in range(3)]
    return _cstreams[key]
DBG = None   # tools: int64 [4] device tensor receiving the planner's model costs and clock


_trash_cache = {}
_trash_lock = threading.Lock()


def _trash_slot(base, stride, H, dev):
    """A private H x 128 x 128 fp32 state window addressable as slot s of `base` (address = base + s * stride floats): pieces whose final state is not a sequence's final state are redirected there
    (same mechanism as w3/kernel.py _trash_slot, kept per (base, stride, H))."""
    key = (torch.device(dev).index, base.data_ptr(), stride, H)
    hit = _trash_cache.get(key)
    if hit is not None:
        return hit
    step = stride * 4
    buf = torch.zeros((step + H * 128 * 128 * 4) // 4 + 16, dtype=torch.float32, device=dev)
    s = -(-(buf.data_ptr() - base.data_ptr()) // step)
    if abs(s) >= 2 ** 31:
        raise ValueError("KDA kernel: the state pool is too far from device scratch memory to be addressed by an int32 slot")
    with _trash_lock:   # [recover] concurrent prepare() calls: evict + insert atomically (next(iter(d)) raises when another thread inserts in between)
        if len(_trash_cache) >= 16:
            _trash_cache.pop(next(iter(_trash_cache)), None)
        _trash_cache[key] = (int(s), buf)
        return _trash_cache[key]


def eligible(N, H, T, n_sms, ck_n, N_min=2):
    """Shape-only gate: few heads, several (<= 32) sequences, enough tokens for pieces to matter, and the potential saving of a cut (the longest chain minus the fluid share)
    large against the fixed cost of the structure (~50 us).  Checkpoints (ck_n chunks per row): pieces are whole numbers of checkpoint intervals and the tail is replayed (see prepare)."""
    if N < N_min or N > KC_MAX or H > MAX_HEADS or H > n_sms // 4 or N * H > (n_sms if ck_n else MAX_TASKS) or n_sms < 16 or n_sms > 256 or T * H * 128 * 2 > (3 << 30):
        return False     # with checkpoints the replay pass makes a cut pay only when the tasks are well below the SM count (and the plain kernel of the old path stores rows faster than the cio head here): N * H <= 0.35 n_sms
    c = T / 32.0
    if N * H > n_sms and (c + N) * H * PAR_DEFAULT["a_head"] / n_sms < 400.0:
        return False     # beyond the SM count (wrap plan of the uncut tasks): the structure's fixed cost (~30-40 us: planner, empty launches) must stay a few % of a balanced batch's time (modelled fluid time)
    return (c + 1.0) - (c + N) * H / n_sms >= 300.0


def static_eligible(N, H, T, n_sms, ck_n):
    """prepare() of a host-known layout (frozen plan): the shape limits of the planner kernel only (the layout decides everything else: the planner returns no plan when it does not cut)."""
    return (1 <= N <= KC_MAX and H <= MAX_HEADS and H <= n_sms // 4 and N * H <= MAX_TASKS and 8 <= n_sms <= 256 and T * H * 128 * 2 <= (3 << 30))


def prepare(base, dual, q, k, v, g, beta, A_log, dt_bias, scale, src, dst, cu_seqlens, n_sms=None, val=None, ck=None, frozen=False, cu_host=None):
    """src = (initial-state tensor or pool, idx int32 [N] | None, slot stride); dst = (pool | None, idx | None, slot stride); cu_seqlens int64 [N + 1] (device);
    ck = (checkpoint buffer [C, H, 128, 128], sanitised checkpoint_cu_starts int64 [N + 1], chunks per checkpoint) | None.
    Checkpoint mode: the tail of a cut sequence is NOT fixed up by output tiles.  Dual pieces + the state chain give the entering state S_in(p) of every piece (fp32, stored next to S_A),
    then ONE plain launch (FEAT + CK build) runs every piece as a pseudo-sequence started from S_in(p): exact outputs, the checkpoint rows inside the piece (pieces are whole numbers of
    checkpoint intervals, so the piece-local chunk index mod ck_n is the global one), and, for the last piece of a sequence, its final state straight into the pool slot.
    frozen (one sequence: cu = [0, T] by the shape; or a host-known layout, cu_host = its cu_seqlens as a list: static prepare()): the plan is made once here by the planner kernel and the launches carry no planner
    (None when the layout is never cut)."""
    B, T, H, D = q.shape
    dev = q.device
    s_t, s_idx, s_stride = src
    d_t, d_idx, d_stride = dst
    n_sms = n_sms or base._num_sms(dev)
    cu = cu_seqlens
    if frozen:
        assert cu_host is not None or cu_seqlens is None or cu_seqlens.numel() == 2
        cu = torch.tensor(cu_host if cu_host is not None else [0, T], dtype=torch.int64, device=dev)    # private: the layout is fixed by the shape (or known on the host)
    assert cu.dtype == torch.int64 and cu.is_contiguous()
    cu_k = val["cu_out"] if val and val["cu_out"] is not None else cu
    if frozen:
        val = None
    pk = {"val": val} if val else {}
    N = cu.numel() - 1
    planner, fixm = _m("plan"), _m("fix")
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(key):
        planner.init()
    if key not in _streams:
        with torch.cuda.device(key):   # [multi-GPU] the stream belongs to the device that is current when it is created
            _streams[key] = torch.cuda.ExternalStream(_m("fixdyn").new_stream(), device=key)
    dstream = _streams[key]
    if ('h', key) not in _streams:
        with torch.cuda.device(key):
            _streams[('h', key)] = torch.cuda.ExternalStream(_m("fixdyn").new_stream(), device=key)
    hstream = _streams[('h', key)]
    if ('h2', key) not in _streams:
        with torch.cuda.device(key):
            _streams[('h2', key)] = torch.cuda.ExternalStream(_m("fixdyn").new_stream(), device=key)
    h2stream = _streams[('h2', key)]
    KC = min(N, KC_MAX)
    ck_n = ck[2] if ck else 0
    ncta0 = max(1, min(n_sms, N * H))
    P_cap = 3 * n_sms // H + KC + 2                       # piece ids (real pieces <= 3 n_sms / H, plus one gap piece per cut sequence)
    fkey = (key, T, H, n_sms, ck_n, tuple(cu_host) if cu_host is not None else None)
    fz = _frozen_cache.get(fkey) if frozen else None       # the plan of a one-sequence shape (shared read-only by every launch of that shape)
    if fz is not None:
        desc, cptr, dtask, dptr, cu_tail, gs, geom, seg, segp = (fz[n] for n in ("desc", "cptr", "dtask", "dptr", "cu_tail", "gs", "geom", "seg", "segp"))
    else:
        desc = torch.zeros(4 * (N * H + ncta0) + 16 + N * H, dtype=torch.int32, device=dev)      # head table (cio int4 entries) | the plain task list of launch B2
        cptr = torch.zeros(4 * (ncta0 + 2), dtype=torch.int32, device=dev)       # [0:n+1] the whole head table; [n+1:2(n+1)] launch A (cut heads only); [2(n+1):3(n+1)] launch B (whole tasks only)
        dtask = torch.zeros(3 * n_sms + 16, dtype=torch.int32, device=dev)
        dptr = torch.zeros(n_sms + 1, dtype=torch.int32, device=dev)
        cu_tail = torch.zeros(P_cap + 1, dtype=torch.int64, device=dev)
        gs = torch.zeros(16 * KC + 16, dtype=torch.int32, device=dev)
        geom = torch.zeros(16, dtype=torch.int32, device=dev)
        seg = torch.zeros(4 * (4 * n_sms + 16), dtype=torch.int32, device=dev)
        segp = torch.zeros(n_sms + 1, dtype=torch.int32, device=dev)
    cptr_all, n1_ = cptr, ncta0 + 2                          # planner outputs of the head tables (sizes fixed by the shape): section stride ncta0 + 2
    cptr_a, cptr_b, cptr_b2 = cptr_all[n1_:n1_ + ncta0 + 1], cptr_all[2 * n1_:2 * n1_ + ncta0 + 1], cptr_all[3 * n1_:3 * n1_ + ncta0 + 1]
    cptr = cptr_all[:ncta0 + 1]
    tlb = desc[4 * (N * H + ncta0) + 16:4 * (N * H + ncta0) + 16 + N * H]
    G_cap = P_cap // 2 + 2                                   # hierarchical groups (>= 3 per slot, groups of >= 2 pieces): at most half the pieces
    qtot = torch.empty(G_cap * H, 128, 128, dtype=torch.float32, device=dev)
    rtot = torch.empty_like(qtot)
    e1 = torch.empty(KC * H, 128, 128, dtype=torch.float32, device=dev)
    slot0 = P_cap * H            # wrap plan hand-off slots (one per head-group CTA) live behind the piece slots
    scratch = torch.zeros((P_cap * H + (0 if slot0 == 0 else n_sms)) * 16384, dtype=torch.float32, device=dev)   # S_A of the cut sequences: slot (first piece id) * H + head (checkpoint mode: also S_in(p) of every piece at slot (piece id) * H + head)
    flags = torch.zeros(P_cap * H + n_sms + 8, dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    if val:
        val["out"] = out
    final_state = d_t if d_t is not None else torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    seq_order = torch.zeros(max(N, 1), dtype=torch.int32, device=dev)
    qt = torch.empty(T, H, 128, dtype=q.dtype, device=dev)
    sL = torch.empty(P_cap * H, 128, 128, dtype=torch.float32, device=dev)
    sP = torch.empty_like(sL)
    sin_ = torch.empty(P_cap * H, 128, 128, dtype=torch.bfloat16, device=dev)
    zero_s = torch.zeros(16, dtype=torch.float32, device=dev)
    Hp = (H + 7) // 8 * 8
    Tp = max(T, 32)
    beta2 = beta.reshape(T, H)
    beta_pad = None
    if Hp != H or Tp != T:
        beta_pad = torch.zeros(Tp, Hp, dtype=beta.dtype, device=dev)
        beta_tma = beta_pad
    else:
        beta_tma = beta2
    feat = ck is not None or not (s_idx is None and d_idx is None and d_t is None)
    cio = base._module_cio(feat, ck is not None)
    bext = base._module(feat, ck is not None)      # plain kernel of launch B2
    dext = dual._module(phl=True, noout=bool(ck)) if PHL else dual._module(dyn_exit=True)   # [long] PHL: the dual launch writes the transition matrices P as bf16 hi / lo planes (same bytes as the fp32 sP)
    maps = dext.encode_maps(q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D), beta_tma, out.view(T, H, D), qt.view(T, H, D), 128, base._LAYOUT_FLAGS).to(dev)
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    assert s_t.stride()[1:] == (D * D, D, 1) and s_t.data_ptr() % 16 == 0 and final_state.data_ptr() % 16 == 0
    fargs = (s_idx, s_stride, d_idx, d_stride, ck[0], ck[1], ck[2]) if ck else ((s_idx, s_stride, d_idx, d_stride, None, None, 0) if feat else ())
    sc, lb = float(scale), -5.0
    par = _par(ck=bool(ck))
    split = not frozen
    par[_PAR_KEYS.index("split")] = 1.0 if split else 0.0
    fin_stride = d_stride if d_t is not None else H * D * D
    fin_idx = d_idx if d_t is not None else None
    hier = bool(par[_PAR_KEYS.index("hier_on")]) and P_cap >= 8
    if ck:   # replay launch: pieces as pseudo-sequences (cu_tail), initial states = scratch slots, checkpoint rows / destination slots per piece from tables built by the chain launch
        ck_pc = torch.zeros(P_cap, dtype=torch.int64, device=dev)
        d_pc = torch.zeros(P_cap, dtype=torch.int32, device=dev)
        trash, trash_keep = _trash_slot(final_state, d_stride if d_t is not None else H * D * D, H, dev)
        rext = base._module(True, True, rw=True)   # [long] KDA_RWAIT build: a task waits for rwait[task] (set by the chain kernels) before it loads its entering state
        rargs = (None, H * D * D, d_pc, d_stride if d_t is not None else H * D * D, ck[0], ck_pc, ck[2])
        seq_order_r = torch.zeros(max(P_cap, 1), dtype=torch.int32, device=dev)
    e_plan, e_dual, e_hb, e_hb2 = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    e_go, e_s3, e_s4 = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    cf_koff = KC * H
    rstart = torch.zeros(4, dtype=torch.int32, device=dev)   # chain CTAs resident (counted by the chain kernels, consumed by chain_gate)
    e_gate = torch.cuda.Event()
    rflags = torch.zeros(P_cap * H + 16, dtype=torch.int32, device=dev)   # [long] per piece task: entering state complete (set by the chain kernels, cleared by the replay task that consumed it)
    e_h1, e_tab, e_head = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    cflags = torch.zeros(cf_koff + G_cap * H + 1, dtype=torch.int32, device=dev)   # chain flags: E_1 per (slot, head), (Q_g, R_g) per (group, head), launch counter (last element)
    pad_skip = [False]
    if beta_pad is not None and val is not None:   # [long] the planner launch carries the padded-beta copy as extra blocks (no graph node of its own)
        val["pad_src"], val["pad_dst"] = beta2, beta_pad
        pad_skip[0] = True
    n_chain = max(1, min(n_sms, KC * H))
    s3, s4, s5 = _chain_streams(key)
    flat_any = hier_any = True
    if frozen:   # one plan per shape, computed once (outside of any capture)
        if fz is None:
            dbg = torch.zeros(32, dtype=torch.int64, device=dev)   # [audit A1] the planner's second search stage writes dbg[16..19]: a 16-element buffer was an out-of-bounds write (padding of the 512 B allocator block, so it never corrupted anything)
            planner.plan_mp(cu, H, ncta0, n_sms, ck_n, desc, cptr_all, dtask, dptr, cu_tail, gs, geom, seg, segp, par, dbg, None, slot0)
            torch.cuda.synchronize()
            pred = dbg[:2].tolist()      # modelled us x 100 of the chosen plan and of the no-cut plan
            g0 = geom.tolist()
            rows = gs.view(-1, 16)[:g0[0]].tolist()
            fz = dict(desc=desc, cptr=cptr, dtask=dtask, dptr=dptr, cu_tail=cu_tail, gs=gs, geom=geom, seg=seg, segp=segp, g0=g0, rows=rows, pred=(pred[0] / 100.0, pred[1] / 100.0))
            if len(_frozen_cache) > 256:
                _frozen_cache.clear()
            _frozen_cache[fkey] = fz
        g0, rows = fz["g0"], fz["rows"]
        if g0[0] == 0:
            return None
        flat_any, hier_any = any(r[6] == 0 for r in rows), any(r[6] != 0 for r in rows)
        n_chain = max(1, min(n_sms, g0[0] * H))
        hier = hier and hier_any
        ncta0 = g0[2]; cptr = cptr[:ncta0 + 1]          # launch grids = the group budgets (idle CTAs of a wider grid cost setup time and shuffle the SM placement)
        dptr = dptr[:g0[3] + 1]

    replay_ee = True
    conc = hier and n_sms >= 2 * 8
    overlap_replay = True
    n_chain_f = min(n_chain, n_sms - 8) if flat_any else 0   # chain CTAs + the gate CTA must always fit next to each other
    def _launch():
        if beta_pad is not None and not pad_skip[0]:
            beta_pad[:T, :H].copy_(beta2)
        cur = torch.cuda.current_stream()
        if not frozen:
            planner.plan_mp(cu, H, ncta0, n_sms, ck_n, desc, cptr_all, dtask, dptr, cu_tail, gs, geom, seg, segp, par, DBG, pk.get("val"), slot0)
        e_plan.record(cur)
        dstream.wait_event(e_plan)

        def head():
            cio.launch(q, k, v, g, beta, A, dtb, cu_k, seq_order, s_t, out, final_state, maps, desc, cptr, scratch, flags, H, sc, lb, *fargs)

        def b2_launch():    # [split] launch B2: when nothing is cut every task is a whole task and the plain kernel (faster than cio, no hand-off machinery) runs them; empty otherwise.
            # Launched FIRST: a launch that is empty in a replay still passes every CTA through an SM slot (~4 us, all SMs), and the launch that is real when nothing is cut must not queue behind those.
            h2stream.wait_event(e_plan)
            with torch.cuda.stream(h2stream):
                bext.set_ee(True)      # CTAs without tasks (every CTA when something is cut) exit before the prologue
                try:
                    bext.launch(128, q, k, v, g, beta, A, dtb, cu_k, seq_order, s_t, out, final_state, maps, tlb, cptr_b2, N, H, sc, lb, *fargs)
                finally:
                    bext.set_ee(False)   # [audit E1] exception-safe (thread_local flag)
                e_hb2.record(h2stream)

        def head_split():   # [split] launch A: the cut heads (they feed the state chain); B (own stream): the whole tasks of the uncut sequences of a launch with cuts, which only have to finish by the end of the launch
            cio.launch(q, k, v, g, beta, A, dtb, cu_k, seq_order, s_t, out, final_state, maps, desc, cptr_a, scratch, flags, H, sc, lb, *fargs)
            hstream.wait_event(e_plan)
            with torch.cuda.stream(hstream):     # cio: wrap-plan hand-offs
                cio.launch(q, k, v, g, beta, A, dtb, cu_k, seq_order, s_t, out, final_state, maps, desc, cptr_b, scratch, flags, H, sc, lb, *fargs)
                e_hb.record(hstream)

        def tail():
            with torch.cuda.stream(dstream):      # tail (dual-chain) launch, high priority: all its CTAs exit at once when this replay has no cut
                dext.launch(q, k, v, g, beta, A, dtb, cu_tail, seq_order, zero_s, out, sL, zero_s, qt, sP, maps, dtask, dptr, H, sc, lb, 0, 0, *((P_cap * H * 16384,) if PHL else ()))
                e_dual.record(dstream)

        head_fn = head_split if split else head
        if split:
            b2_launch()
        tail(); head_fn()
        if ck and overlap_replay:   # [long] replay launch OVERLAPPED with the chain: its tasks wait for per-piece flags (rflags) set by the chain kernels as each entering state completes
            mpd_args = (out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 1)
            e_head.record(cur)
            # checkpoint tables (rows / destination slot of every piece) only need the planner: one tiny launch on s4, concurrent with the head and dual launches
            s4.wait_event(e_plan)
            with torch.cuda.stream(s4):
                fixm.fixup_mpd(*mpd_args, 0, None, None, 0, 4, cu_k, ck[1], d_idx if d_t is not None else None, ck_pc, d_pc, cu_tail, ck[2], trash)
                e_tab.record(s4)
            # chain kernels first (their CTAs must be resident before the replay CTAs that wait for them): stage 1 on the (high priority) tail stream, stage 3 on s3, the flat chain on s4.
            # Launch order = priority (~2.5 us per graph launch): the hierarchical chain first when the (frozen) plan has one, else the flat chain
            def _hd1():
                dstream.wait_event(e_head)
                with torch.cuda.stream(dstream):
                    fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 1, n_sms, None, None, 0, 1, cflags=cflags, cf_koff=cf_koff, cf_total=2 * n_sms, rflags=rflags, rstart=rstart)
                    e_h1.record(dstream)

            def _hd3():
                s3.wait_event(e_dual); s3.wait_event(e_head)
                with torch.cuda.stream(s3):
                    fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 3, n_sms, None, None, 0, 1, cflags=cflags, cf_koff=cf_koff, cf_total=2 * n_sms, rflags=rflags, rstart=rstart)
                    e_s3.record(s3)

            def _mpd():
                s4.wait_event(e_dual); s4.wait_event(e_head)
                with torch.cuda.stream(s4):
                    if n_chain_f > 0:
                        fixm.fixup_mpd(*mpd_args, n_chain_f, None, None, 0, 3, cu_k, ck[1], d_idx if d_t is not None else None, ck_pc, d_pc, cu_tail, ck[2], trash, rflags=rflags, rstart=rstart)
                    e_s4.record(s4)
            use_hd = (not frozen) or hier_any   # a frozen plan without hierarchical groups needs no stage 1 / 3 launches
            hd_fuse = use_hd and n_sms >= 16

            def _hdf():   # [hdfuse] stage 1 + stage 3 in one launch (njobs CTAs each, <= n_sms - 4 together: the planner bounds K H (2 G - 1) <= n_sms - 4)
                dstream.wait_event(e_head)
                with torch.cuda.stream(dstream):
                    fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 13, n_sms - 4, None, None, 0, 1, cflags=cflags, cf_koff=cf_koff, cf_total=n_sms - 4, rflags=rflags, rstart=rstart)
                    e_h1.record(dstream)
            if hd_fuse:
                if frozen and hier_any: _hdf(); _mpd()
                else: _mpd(); _hdf()
            elif frozen and hier_any:
                _hd1(); _hd3(); _mpd()
            elif use_hd:
                _mpd(); _hd1(); _hd3()
            else:
                _mpd()
            # the replay launch (its CTAs spin on the flags) is released once every working chain CTA is resident
            s5.wait_event(e_dual); s5.wait_event(e_head)
            with torch.cuda.stream(s5):
                fixm.chain_gate(geom, rstart, H, n_chain_f, 1 if use_hd else 0)
                e_gate.record(s5)
            cur.wait_event(e_dual); cur.wait_event(e_tab); cur.wait_event(e_gate)
            rext.set_ee(replay_ee)   # [ee] replay CTAs without pieces (all of them in a launch with no cut) leave before the prologue
            try:
                rext.launch(128, q, k, v, g, beta, A, dtb, cu_tail, seq_order_r, scratch, out, final_state, maps, dtask, dptr, P_cap, H, sc, lb, *rargs, rflags)
            finally:
                rext.set_ee(False)   # [audit E1]
            if use_hd:
                cur.wait_event(e_h1)
                if not hd_fuse: cur.wait_event(e_s3)
            cur.wait_event(e_s4)
            if split:   # join the whole-task launches (B / B2) of the head group
                cur.wait_event(e_hb); cur.wait_event(e_hb2)
            return out, final_state
        cur.wait_event(e_dual)
        if conc:   # [long] the flat chain / checkpoint tables (stream s4), hierarchical stage 3 (stream s3) and stage 1 (this stream) run CONCURRENTLY: stage 3 waits for its inputs through flags
            e_go.record(cur)
            mpd_args = (out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 1, n_chain if (flat_any or not ck) else 0)
            s3.wait_event(e_go); s4.wait_event(e_go)
            def hd1():
                fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 1, n_sms, None, None, 0, 1 if ck else 0, cflags=cflags, cf_koff=cf_koff, cf_total=2 * n_sms)

            def hd3():
                with torch.cuda.stream(s3):
                    fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 3, n_sms, None if ck else final_state, None if ck else fin_idx, 0 if ck else fin_stride, 1 if ck else 0,
                                  cflags=cflags, cf_koff=cf_koff, cf_total=2 * n_sms)
                    e_s3.record(s3)

            def mpd():
                with torch.cuda.stream(s4):
                    if ck:
                        fixm.fixup_mpd(*mpd_args, None, None, 0, 1, cu_k, ck[1], d_idx if d_t is not None else None, ck_pc, d_pc, cu_tail, ck[2], trash)
                    else:
                        fixm.fixup_mpd(*mpd_args, final_state, fin_idx, fin_stride, 0, None, None, None, None, None, None, 0, 0)
                    e_s4.record(s4)
            # every launch costs ~2.5 us of graph launch latency: the chain that gates the replay goes first.  Known shapes (frozen plans): the hierarchical chain when the plan has one, else the flat chain;
            # device-planned layouts (unknown): flat chains are the common case (hierarchical groups need >= 8 pieces of one sequence)
            if frozen and hier_any:
                hd1(); hd3(); mpd()
            else:
                mpd(); hd1(); hd3()
            cur.wait_event(e_s3); cur.wait_event(e_s4)
            if ck:
                rext.launch(128, q, k, v, g, beta, A, dtb, cu_tail, seq_order_r, scratch, out, final_state, maps, dtask, dptr, P_cap, H, sc, lb, *rargs, None)
                if split:   # join the whole-task launches (B / B2) of the head group
                    cur.wait_event(e_hb); cur.wait_event(e_hb2)
                return out, final_state
            fixm.fixup_mpd(out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 2, 0, None, None, 0, 0, None, None, None, None, None, None, 0, 0)   # [hardening] writes only rows inside the cut sequences' tails (store_tail_rows): runs concurrently with B / B2
            if split:   # join the whole-task launches (B / B2) of the head group
                cur.wait_event(e_hb); cur.wait_event(e_hb2)
            return out, final_state
        if ck:
            fixm.fixup_mpd(out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 1, n_chain if flat_any else 0, None, None, 0, 1, cu_k, ck[1], d_idx if d_t is not None else None, ck_pc, d_pc, cu_tail, ck[2], trash)
            if hier:
                fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 1, n_sms, None, None, 0, 1)
                fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 3, n_sms, None, None, 0, 1)
            rext.launch(128, q, k, v, g, beta, A, dtb, cu_tail, seq_order_r, scratch, out, final_state, maps, dtask, dptr, P_cap, H, sc, lb, *rargs, None)
            if split:   # join the whole-task launches (B / B2) of the head group
                cur.wait_event(e_hb); cur.wait_event(e_hb2)
            return out, final_state
        fixm.fixup_mpd(out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 1, n_chain, final_state, fin_idx, fin_stride, 0, None, None, None, None, None, None, 0, 0)
        if hier:
            fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 1, n_sms, None, None, 0, 0)
            fixm.fixup_hd(scratch, sP, sL, qtot, rtot, e1, sin_, gs, geom, H, 3, n_sms, final_state, fin_idx, fin_stride, 0)
        fixm.fixup_mpd(out, qt, scratch, sP, sL, sin_, seg, segp, gs, geom, H, T, 2, 0, None, None, 0, 0, None, None, None, None, None, None, 0, 0)
        if split:   # join the whole-task launches (B / B2) of the head group
            cur.wait_event(e_hb)
            cur.wait_event(e_hb2)
        return out, final_state

    def launch():
        # [latent] item 2: the inter-launch flag arrays (rflags / rstart / cflags / cio wrap flags) are consumer-cleared and carry no epoch: a launch that is abandoned between a producer and its consumer by a HOST exception
        # (any BaseException while the launches are enqueued) would leave stale flags that release a later launch of this prepared object early (stale S_in, no hang).  The abandoned launch's queued work is drained and the
        # flag arrays are re-zeroed before the exception propagates.  (A device-side trap / kernel fault kills the context: nothing can launch afterwards.  Stream capture executes nothing: no cleanup.)
        try:
            return _launch()
        except BaseException:
            if not torch.cuda.is_current_stream_capturing():
                try:
                    for s_ in (dstream, hstream, h2stream, s3, s4, s5):
                        s_.synchronize()
                    cur_ = torch.cuda.current_stream()
                    cur_.synchronize()
                    for t_ in (rflags, cflags, rstart, flags):
                        t_.zero_()
                    cur_.synchronize()
                except BaseException:
                    pass   # a device error here is reported by the original exception / the next call
            raise

    launch.pad_info = (beta2, beta_pad) if beta_pad is not None else None   # [long] a caller whose validation launch carries the copy sets pad_skip[0] = True
    launch.pad_skip = pad_skip
    launch.pred_us = fz["pred"] if frozen else None      # frozen plans: (modelled us of the plan incl. the model's fixed cost `ovh`, of no cut)
    launch.keep = (trash_keep,) if ck else ()    # the private trash window must outlive the launch (the module cache evicts)
    launch.tables = dict(cptr_all=cptr_all, tlb=tlb, qtot=qtot, rtot=rtot, e1=e1, desc=desc, cptr=cptr, dtask=dtask, dptr=dptr, cu_tail=cu_tail, gs=gs, geom=geom, seg=seg, segp=segp, scratch=scratch, sL=sL, sP=sP, sin_=sin_, qt=qt)
    return launch
