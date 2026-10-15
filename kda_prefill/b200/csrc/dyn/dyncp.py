"""Graph-safe context-parallel dynamic path (device-planned): head group (cio, the cut sequence's head hands S_A to scratch slots) || dual-chain tail launch,
then the fixup (patched cp_fix: device tail geometry).  One planner kernel (plan_cp.cu) decides cut / budgets / pieces / tables from cu_seqlens on the GPU
with the host cp_plan cost model; with no CP in a replay every tail table is empty and the extra launches exit immediately.
Used for shapes where CP is conceivable (few tasks: N*H <= 2*n_sms), no checkpoints."""
import os, importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mods = {}


def _m(name, bf=False):
    key = name + ("_bf" if bf else "")
    if key not in _mods:
        ck = name == "fixck"
        _mods[key] = _load("kda_pg_" + key, os.path.join(_HERE, {"plan": "build.py", "fix": "build_fix.py", "fixck": "build_fix.py", "aux": "build_aux.py"}[name])).build(**({"ck": True, **({"bf": True} if bf else {})} if ck else {}))
    return _mods[key]


_streams = {}
DBG = None   # tools: int64 device tensor receiving the planner's clock64 stamps (read when the launch is captured)
_QT_MAX_BYTES = 3 << 30


_SNAP_BYTES_DYN = 1 << 30   # [dynpath2] snapshot scratch (S_loc + P per tail checkpoint per head) a device-planned CP replay may use: allocated once per prepared launch


def ck_rows_cap(T, H, ck_n):
    """Tail checkpoint rows (per cut) the snapshot buffers hold: the most any cut can need ((T / 32 - 1) // ck_n + 1) bounded by _SNAP_BYTES_DYN; 0 when no checkpoints."""
    if not ck_n:
        return 0
    return max(0, min(((T + 31) // 32 - 1) // ck_n + 1, _SNAP_BYTES_DYN // (2 * H * 65536)))


def eligible(N, H, T, n_sms, ck_n, dev):
    if T < 2048:      # tiny problems: the extra planner work and launches would dominate and no tail is long enough to cut
        return False
    """CP can only pay when few tasks leave SMs idle (host cp_plan exit test needs the longest task to exceed the fluid share): shape-only gate."""
    return (ck_n == 0 or ck_rows_cap(T, H, ck_n) >= 1) and N * H <= 2 * n_sms and N <= 256 and n_sms >= 8 and T * H * 128 * 2 <= _QT_MAX_BYTES


_n1_cache = {}
_CP_OVH = 40.0   # chunk-steps (1.7 us each) a cut must save beyond the host rule on a device-planned replay: planner (~30-50 us when it cuts) + dual / fixup launches + head imbalance (measured: D(CP) vs D(no CP) over 16 held-out layouts)


def _lpt2(n, H, t1, t2):
    q, r = divmod(H, n)
    lhi, llo = (q + 1) * t1, q * t1
    remaining, clo, chi, klo, khi, last = H, n - r, r, 1, 1, 0.0
    while remaining > 0:
        vlo = llo + klo * t2 if clo > 0 else 1e300
        vhi = lhi + khi * t2 if chi > 0 else 1e300
        if vlo <= vhi:
            remaining -= min(remaining, clo); last = vlo; klo += 1
        else:
            remaining -= min(remaining, chi); last = vhi; khi += 1
    return max(lhi if r > 0 else llo, last)


def _head_first(H, c, g, cu_tail, ck_n, n_sms):
    """Launch order of the two concurrent groups = the host planner's p_first: the group with the longer modelled makespan (head: fluid bound of its H whole tasks; tail: dual tasks, one or two pieces) goes first
    (the first launched group gets measurably better SMs: ~2% on the one-sequence CP shapes)."""
    xc, n_A, n_D, m = g[6], g[4], g[5], g[1]
    wc = lambda n: n + 2.77 + (0.7 if n < 17 else 0.0)
    a = max((H * wc(xc) + 3.33 * (n_A - 1)) / n_A, wc(xc))
    tcx = c - xc
    rows = max(0, (c - 1) // ck_n - xc // ck_n) if ck_n else 0
    snap = rows * 3.0
    if m == 2:
        s1 = (cu_tail[1] - cu_tail[0]) // 32
        b = _lpt2(n_D, H, 1.4 * s1 + 6.4, 1.4 * (tcx - s1) + 6.4) + snap
    else:
        b = -(-H // n_D) * (1.4 * tcx + 6.4 + snap)
    return a >= b


def n1_tables(planner, T, H, n_sms, ck_n, rows_cap, dev):
    """[dynpath2] One sequence (cu = [0, T], fixed by the shape): the context-parallel plan is a pure function of (T, H, SM count, checkpoint interval).  It is computed once per key by the
    single-sequence planner kernel (the same one the graph-safe path runs every launch), read back, and shared read-only by every launch of that shape: the launches then carry no planner kernel.
    Returns a dict of device tables + geometry, or None when the planner never cuts this shape (the caller uses the non-CP launch)."""
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), T, H, n_sms, ck_n)
    if key in _n1_cache:
        return _n1_cache[key]
    ncta0 = max(1, min(n_sms, H))
    t = dict(desc=torch.zeros(H + 2 * n_sms, dtype=torch.int32, device=dev), cptr=torch.zeros(n_sms + 1, dtype=torch.int32, device=dev), dtask=torch.zeros(4 * H + 16, dtype=torch.int32, device=dev),
             dptr=torch.zeros(n_sms + 1, dtype=torch.int32, device=dev), cu_tail=torch.zeros(3, dtype=torch.int64, device=dev), seg=torch.zeros(4 * (2 * H + n_sms + 16), dtype=torch.int32, device=dev),
             segp=torch.zeros(n_sms + 1, dtype=torch.int32, device=dev), geom=torch.zeros(16, dtype=torch.int32, device=dev), cu_a=torch.zeros(2, dtype=torch.int64, device=dev),
             cki=torch.zeros(4 * (rows_cap + 2), dtype=torch.int32, device=dev), ccopy=torch.zeros(4, dtype=torch.int32, device=dev))
    cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
    planner.plan_n1(cu, H, ncta0, t["desc"], t["cptr"], None, t["dtask"], t["dptr"], t["cu_tail"], t["seg"], t["segp"], t["geom"], n_sms, t["cu_a"], None, ck_n, rows_cap, t["cki"], t["ccopy"])
    g = t["geom"].tolist()
    if g[1] == 0:
        res = None
    else:
        c = (T + 31) // 32
        t["m"] = g[1]; t["e"] = g[2]; t["ncopy"] = g[10]; t["rows"] = max(1, (c - 1) // ck_n - g[9] + 1) if ck_n else 0
        t["head_first"] = _head_first(H, c, g, t["cu_tail"].tolist(), ck_n, n_sms)
        res = t
    if len(_n1_cache) > 512:
        _n1_cache.clear()
    _n1_cache[key] = res
    return res


def prepare(base, dual, q, k, v, g, beta, A_log, dt_bias, scale, src, dst, cu_seqlens, n_sms=None, val=None, ck=None, frozen=False):
    """src = (initial-state tensor or pool, idx int32 [N] | None, slot stride); dst = (pool | None, idx | None, slot stride);
    ck [dynpath2] = (checkpoint buffer [C, H, 128, 128], sanitized checkpoint_cu_starts int64 [N + 1], chunks per checkpoint) | None."""
    B, T, H, D = q.shape
    dev = q.device
    s_t, s_idx, s_stride = src
    d_t, d_idx, d_stride = dst
    n_sms = n_sms or base._num_sms(dev)
    cu = cu_seqlens if cu_seqlens is not None else torch.tensor([0, T], dtype=torch.int64, device=dev)
    assert cu.dtype == torch.int64 and cu.is_contiguous()
    pk = {"val": val} if val else {}   # [contract2] table validation (a second block of the planner launch); the planner also writes the sanitized cu_seqlens (the kernels below read cu_k)
    cu_k = val["cu_out"] if val and val["cu_out"] is not None else cu
    N = cu.numel() - 1
    planner, fixdyn, aux = _m("plan"), _m("fixck" if ck else "fix", bool(ck) and base.ck_bf16()), _m("aux")
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(key):
        planner.init()
        aux.init()
    if key not in _streams:
        with torch.cuda.device(key):   # [multi-GPU] the stream belongs to the device that is current when it is created
            _streams[key] = torch.cuda.ExternalStream(fixdyn.new_stream(), device=key)
    dstream = _streams[key]
    ncta0 = max(1, min(n_sms, N * H))
    plain_a = N == 1   # one sequence: the head launch is the plain kernel (cu_A = [0, x]); its final state IS S_A
    lean_ok = N >= 2 and N <= 32 and not plain_a and (T * H * 100 // 32 + N * H * 400 + 400 * 256) < (1 << 30)
    n1plan = plain_a
    plan_mode = (1 if plain_a else 0) | (4 if lean_ok else 0) | 8    # bit 2: lean non-CP plan when the cp_plan early exit applies (plan_cp.cu)
    ck_n = ck[2] if ck else 0
    rows_cap = ck_rows_cap(T, H, ck_n)
    tabs = None
    if frozen:   # one sequence, shape-only plan (n1_tables): shared read-only tables, no planner launch
        assert N == 1 and plain_a and n1plan
        tabs = n1_tables(planner, T, H, n_sms, ck_n, rows_cap, dev)
        if tabs is None:
            return None
        desc, cptr, dtask, dptr, cu_tail, seg, segp, geom, cu_a, cki, ccopy = (tabs[n] for n in ("desc", "cptr", "dtask", "dptr", "cu_tail", "seg", "segp", "geom", "cu_a", "cki", "ccopy"))
        rows_cap = tabs["rows"]
        n_A, n_D = tabs["geom"][4:6].tolist() if "nA" not in tabs else (tabs["nA"], tabs["nD"])
        tabs["nA"], tabs["nD"] = n_A, n_D
        cptr, dptr = cptr[:n_A + 1], dptr[:n_D + 1]   # launch grids = the group budgets (as the host-planned launches): idle CTAs of a wider grid cost setup time and shuffle the SM placement
    else:
        desc = torch.empty(4 * (N * H + 2 * n_sms), dtype=torch.int32, device=dev)
        cptr = torch.zeros(n_sms + 1, dtype=torch.int32, device=dev)
        dtask = torch.zeros(4 * H + 16, dtype=torch.int32, device=dev)
        dptr = torch.zeros(n_sms + 1, dtype=torch.int32, device=dev)
        cu_tail = torch.zeros(3, dtype=torch.int64, device=dev)
        seg = torch.zeros(4 * (2 * H + n_sms + 16), dtype=torch.int32, device=dev)
        segp = torch.zeros(n_sms + 1, dtype=torch.int32, device=dev)
        geom = torch.zeros(16, dtype=torch.int32, device=dev)   # [8] ns, [9] k0, [10] plain-copy rows, [12:15] the dual kernel's ck_geom {xc, k0, x}
        cu_a = torch.zeros(2, dtype=torch.int64, device=dev) if plain_a else None
    if frozen:   # head state S_A only (plain head launch: no wrap hand-off slots, no flags)
        scratch, flags = torch.empty(H * 128 * 128, dtype=torch.float32, device=dev), None
    else:
        scratch = torch.zeros((H + n_sms) * 128 * 128, dtype=torch.float32, device=dev)   # slots [0, H): S_A of the cut sequence; [H, H + n_sms): wrap hand-off
        flags = torch.zeros(H + n_sms, dtype=torch.int32, device=dev)
    sA = scratch[:H * 16384].view(H, 128, 128)
    out = torch.empty_like(q)
    if val:
        val["out"] = out   # [dynpath2] the planner's validation block zero-fills it when cu_seqlens is rejected
    final_state = d_t if d_t is not None else torch.empty(N, H, D, D, dtype=torch.float32, device=dev)
    seq_order = torch.zeros(max(N, 1), dtype=torch.int32, device=dev)
    qt = torch.empty(T, H, 128, dtype=q.dtype, device=dev)     # rows [e, e + 128) are zeroed on the device (zero_qt_tail), the tail rows are written by the dual launch
    if frozen:   # the tail end is fixed by the shape: zero the rows once (only rows [x, e) are ever written)
        qt[tabs["e"]:tabs["e"] + 128].zero_()
    sL = torch.empty(2, H, 128, 128, dtype=torch.float32, device=dev)
    sP = torch.empty_like(sL)
    sin_ = torch.empty(1, H, 128, 128, dtype=torch.float32, device=dev)
    sfix = torch.empty(H, 128, 128, dtype=torch.float32, device=dev)      # fixup final-state staging (sa must not alias sfin), scattered into the state slot of the cut sequence
    zero_s = torch.zeros(16, dtype=torch.float32, device=dev)          # L / Tr initial states are written by the dual kernel itself (l_zero / t_eye)
    Hp = (H + 7) // 8 * 8
    Tp = max(T, 32)
    beta2 = beta.reshape(T, H)
    beta_pad = None
    if Hp != H or Tp != T:
        beta_pad = torch.zeros(Tp, Hp, dtype=beta.dtype, device=dev)
        beta_tma = beta_pad
    else:
        beta_tma = beta2
    feat = ck is not None or not (s_idx is None and d_idx is None and d_t is None)   # [dynpath2] no pool / source tables: the default builds (same ABI, no slot-table loads, ~1.5-2% faster)
    ext = base._module(feat, ck is not None)
    cio = None if plain_a else base._module_cio(feat, ck is not None)
    dext = dual._module(ck_dev=True) if ck else dual._module(dyn_exit=True)
    dmaps = dext.encode_maps(q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D), beta_tma, out.view(T, H, D), qt.view(T, H, D), 128,
                             base._LAYOUT_FLAGS).to(dev)
    maps = dmaps   # the dual build's 7 maps (q k v g beta out qt) start with the head launch's 6 (same layout flags, same out tensor): one encode serves both
    A = A_log.contiguous().float()
    dtb = dt_bias.contiguous().float().view(-1)
    assert s_t.stride()[1:] == (D * D, D, 1) and s_t.data_ptr() % 16 == 0 and final_state.data_ptr() % 16 == 0
    fargs = (s_idx, s_stride, d_idx, d_stride, ck[0], ck[1], ck[2]) if ck else ((s_idx, s_stride, d_idx, d_stride, None, None, 0) if feat else ())
    fargs_head = (s_idx, s_stride, None, H * 128 * 128, ck[0], ck[1], ck[2]) if ck else ((s_idx, s_stride, None, H * 128 * 128, None, None, 0) if feat else ())   # frozen: the head's state goes to scratch row 0
    if ck:
        snapL = torch.empty(rows_cap + 2, H, 128, 128, dtype=torch.float32, device=dev)   # S_loc / P snapshots of the tail's checkpoint boundaries (row = boundary - k0)
        snapP = torch.empty_like(snapL)
        if not frozen:
            cki = torch.zeros(4 * (rows_cap + 2), dtype=torch.int32, device=dev)
            ccopy = torch.zeros(4, dtype=torch.int32, device=dev)
        ck_geom = geom[12:15]
    ckargs = dict(ck_n=ck_n, ck_rows_cap=rows_cap, cki=cki, ccopy=ccopy) if ck else {}
    sc, lb = float(scale), -5.0
    swap_streams = frozen and not tabs["head_first"]   # the longer group is launched first; the head is on the side stream when the tail is the longer one
    e_plan, e_dual = torch.cuda.Event(), torch.cuda.Event()
    pad_skip = [False]
    if beta_pad is not None and val is not None and not frozen:   # [long] the planner launch carries the padded-beta copy (extra blocks)
        val["pad_src"], val["pad_dst"] = beta2, beta_pad
        pad_skip[0] = True

    def launch():
        if beta_pad is not None and not pad_skip[0]:
            beta_pad[:T, :H].copy_(beta2)
        cur = torch.cuda.current_stream()
        if frozen:
            pass   # the plan is read-only data (n1_tables): no planner launch
        elif n1plan:   # one sequence: the single-sequence planner (same tables as plan_wrap, fewer stages)
            planner.plan_n1(cu, H, ncta0, desc, cptr, DBG, dtask, dptr, cu_tail, seg, segp, geom, n_sms, cu_a, pk.get("val"), ck_n, rows_cap, ckargs.get("cki"), ckargs.get("ccopy"), _CP_OVH)
        else:
            planner.plan_wrap(cu, H, ncta0, desc, cptr, plan_mode, DBG, dtask, dptr, cu_tail, seg, segp, geom, n_sms, H, cu_a, **pk, **ckargs, cp_ovh=_CP_OVH)
        e_plan.record(cur)
        dstream.wait_event(e_plan)
        dual_args = (q, k, v, g, beta, A, dtb, cu_tail, seq_order, zero_s, out, sL, zero_s, qt, sP, dmaps, dtask, dptr, H, sc, lb, 0, 0, *((snapL, snapP, ck_n, ck_geom) if ck else ()))
        if swap_streams:   # experiment: the head launch goes to the side stream
            with torch.cuda.stream(dstream):
                ext.launch(128, q, k, v, g, beta, A, dtb, cu_a, seq_order, s_t, out, sA.view(1, H, 128, 128), maps, desc, cptr, N, H, sc, lb, *fargs_head)
                e_dual.record(dstream)
            dext.launch(*dual_args)
        else:
          with torch.cuda.stream(dstream):      # tail (dual-chain) launch, high priority: all its CTAs exit at once when this replay has no CP
            dext.launch(*dual_args)
            e_dual.record(dstream)
        if not frozen:
            aux.zero_qt_tail(qt, geom, T, H)       # rows [e, e + 128): disjoint from the rows [x, e) the dual launch writes
        if swap_streams:
            pass
        elif plain_a and frozen:   # the head's final state is S_A and goes to the scratch state (the cut is certain for this shape): the fixup then writes the final state of the sequence into its slot
            ext.launch(128, q, k, v, g, beta, A, dtb, cu_a, seq_order, s_t, out, sA.view(1, H, 128, 128), maps, desc, cptr, N, H, sc, lb, *fargs_head)
        elif plain_a:
            ext.launch(128, q, k, v, g, beta, A, dtb, cu_a, seq_order, s_t, out, final_state, maps, desc, cptr, N, H, sc, lb, *fargs)
        else:
            cio.launch(q, k, v, g, beta, A, dtb, cu_k, seq_order, s_t, out, final_state, maps, desc, cptr, scratch, flags, H, sc, lb, *fargs)
        cur.wait_event(e_dual)
        ckx = (ck[0], snapP, snapL, cki, ck[1]) if ck else ()   # checkpoint items of the fixup (device-planned): buffer, snapshots, item table, ck_cu
        if plain_a and frozen:
            fixdyn.fixup_dyn(out, qt, sA, sP, sL, sin_, sfix, seg, segp, geom, H, T, None, None, 0, final_state, d_idx, d_stride, *ckx)
            if ck and tabs["ncopy"]:
                aux.copy_ck_dyn(sA, None, None, 0, sin_, ck[0], ck[1], ccopy, geom, H)
        elif plain_a:
            fixdyn.fixup_dyn(out, qt, sA, sP, sL, sin_, sfix, seg, segp, geom, H, T, final_state, d_idx, d_stride, None, None, 0, *ckx)
            if ck:   # S_A is still in the pool slot here (the scatter below overwrites it with the final state)
                aux.copy_ck_dyn(sA, final_state, d_idx, d_stride, sin_, ck[0], ck[1], ccopy, geom, H)
            aux.scatter_state(sfix, final_state, d_idx, d_stride, geom, H)
        else:   # cio head: S_A is in scratch, so the fixup stores the final state of the cut sequence directly into its slot (no scatter launch)
            fixdyn.fixup_dyn(out, qt, sA, sP, sL, sin_, sfix, seg, segp, geom, H, T, None, None, 0, final_state, d_idx, d_stride, *ckx)
            if ck:
                aux.copy_ck_dyn(sA, None, None, 0, sin_, ck[0], ck[1], ccopy, geom, H)
        return out, final_state

    launch.pad_info = (beta2, beta_pad) if beta_pad is not None else None
    launch.pad_skip = pad_skip
    return launch
