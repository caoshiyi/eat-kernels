"""v14_b (v75) = v74 + two rules of the per-XCC model of ASYMMETRY.md, behind one guard.

event = F + max over physical XCC x of (S_x + W_x): XCC_START_US = S_x (start offsets), XCC_WRITE_RATIO = how much longer XCC x
writes on a single-round spec grid. Rules (each launches its own kernel instantiation; NOTES_v14.md has every number):
  (X) xcc_fill():  v74's X1 generalised. A latency-bound plain grid runs on the m XCCs with the smallest start offsets, with the
                   smallest m for which the busiest of them carries <= 16 blocks (X1 is m = 1; m <= 4).
  (U) xcc_items(): in a write-bound single-round spec grid with spare CUs the XCC with the slowest write path gives its last
                   K = round(0.08 N / 8) items to blocks behind the grid on the two XCCs that start first.
(A third rule of the model, a per-XCC read-first hold on the plain state-first hold grids, gains 0.1-0.2 us in a process that
allocates its flush buffer before the pools and nothing in the eval's own allocation order. Not part of this kernel; the form is
in kernel_v14_a.py / kda_v14_a.cu, NOTES_v14 item H.)
Structure: this module is a layer over kernel_v13_a (v74). The cost model of v74 (choose(), prologue_first(), hold_ticks(), ...) and
its two builds of kda_v13_a.cu are used as they are, so every kernel v74 can launch is the same binary at the same place in the
same code object as in v74 (a one-source build of everything, with every reachable kernel instruction-identical and every rule
off, measured 0.0011-0.0018 slower than v74 on the full 17: NOTES_v14 "the build term"). The per-XCC kernels live in kda_v14_b.cu,
built twice more with -DKDA_XONLY (default scheduler: the XF masks; max-ILP: UQ). All four code objects are uploaded at the first
eager call, v74's first (_upload(): device memory taken later, between a caller's pool allocations, moves memory-bound grids).
Guard (HW_RESEARCH.md card B): the tables are MI350X's, so a rule needs _xcc_tab() (8 XCCs by the HIP device attribute, 256 CUs,
compiled XCC constants equal to the tables), a stream without a CU mask, and a dispatch that is an exact round-robin over the XCCs
(a driver / firmware contract whose origin rotates per queue: the kernels read HW_REG_XCC_ID, never blockIdx alone), probed once
per device outside any capture; a failed probe is sticky and warns, xcc_reprobe() repeats it. Without the guard the grid is v74's
symmetric one. v74's X1 kernel is launched from v74's build, behind this guard.

Names this module does not define (FORCE_*, the mailbox workspaces, reset_workspace(), ...) are v74's: attribute access falls
through to kernel_v13_a, and assignments to v74's switches are forwarded to it."""
import hashlib
import os
import sys
import types
import warnings

import torch
from torch.utils.cpp_extension import load

import kernel_v13_a as V

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kda_v14_b.cu")
_xmods = {}

# ---- per-XCC model (ASYMMETRY.md): properties of MI350X (8 XCCs, 256 CUs), only used on a device that _xcc_tab() accepts
XCC_START_US = (0.45, 0.0, 0.62, 0.45, 0.90, 0.62, 0.80, 0.57)  # S_x: start offset of XCC x through a ~700-instruction kernel, relative to
# XCC 1 (ASYMMETRY A.1: X1 launched on each target XCC, 18 fresh-process pairs x 3 shapes on HIP 0 / 2 / 3, +- 0.05 us; same order on HIP 6)
XCC_WRITE_RATIO = (1.0, 1.17, 1.18, 1.33, 1.0, 1.17, 1.0, 1.17)  # token-loop duration of XCC x relative to XCCs 0 / 4 / 6 on single-round
# spec grids (ASYMMETRY A.4: stamped, two HSA queues, 12,16,4 / 16,16,6 / 12,16,8: odd 1.12-1.22, XCC 2 1.12-1.25, XCC 3 1.22-1.44)
X1_XCC = min(range(8), key=XCC_START_US.__getitem__)  # = 1: the XCC that starts first (UNDERFLOOR.md E5: 88-99 % of the replays on four GPUs)
X1_XCC_BW = V.X1_XCC_BW  # bytes per us through one XCC's memory path (v74)
X1_SPAN_US = V.X1_SPAN_US  # us: memory phase of a latency-bound plain block (v74); 0 switches xcc_fill() off, X1 included
XF_MAX = 4  # xcc_fill(): most XCCs a latency-bound plain grid is placed on (1 = X1 only, v74). Measured (NOTES_v14 item X): m = 2 / 3 / 4
# -0.11..-0.29 us on twelve shapes of 17-64 blocks; m = 5 / 6 (the next XCCs start 0.62 us late, 0.28 us before the last) -0.16..+0.06 on eight
UQ_FRAC = 0.92  # xcc_items(): share of its rows the slowest-writing XCC keeps (1 = off, v74): 22 of 24 (ASYMMETRY B.6: 23 gives half, 21 no more)
_x1 = {}  # device -> probed: the dispatch is an exact round-robin over 8 XCCs (False is sticky)
_tab = {}  # device -> the per-XCC tables apply (8 XCCs, 256 CUs, compiled constants as the tables)
_X1_OK = False  # this call: tables apply, the stream has no CU mask and the round-robin was probed exact (xcc_fill(), xcc_items())


def _xext(ilp=False):
    """The builds of kda_v14_b.cu with -DKDA_XONLY: only the per-XCC kernels (ilp=False: the XF masks and the host queries; ilp=True:
    UQ, max-ILP scheduler as every spec kernel). Flags otherwise as v74's builds (kernel_v13_a._ext())."""
    if ilp not in _xmods:
        rocm = V._rocm_version()
        if rocm != V._AUDITED_ROCM:
            warnings.warn(f"kda decode kernel: building with ROCm {rocm}, audited on {V._AUDITED_ROCM}: rerun scratch_final/validate_all.sh (asm checkers)")
        digest = hashlib.sha1(open(_SRC, "rb").read() + rocm.encode()).hexdigest()[:10] + "_x" + ("_ilp" if ilp else "")
        bdir = os.path.join(_HERE, "build", "v14b_" + digest)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        xf = ["-mllvm", "-amdgpu-sched-strategy=max-ilp", "-DKDA_SPEC_ONLY"] if ilp else []
        try:
            _xmods[ilp] = load(name=f"kda_v14b_{digest}", sources=[_SRC], extra_cflags=["-O3"],
                               extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16", "-DSPOL_SPEC=17",
                                                  "-DSPOL_PLAIN=17", "-DPFSTRIDE=" + str(2 << 20), "-DSTAG=150", "-DSTAGL=3", "-DBLD=2", "-DKDA_XONLY"] + xf,
                               build_directory=bdir, verbose=False)
        finally:
            if arch0 is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch0
    return _xmods[ilp]


def _ext(ilp=False):
    """v74's build (kernel_v13_a._ext()); also makes sure the per-XCC build of the same kind exists (validation builds both first)."""
    _xext(ilp)
    return V._ext(ilp)


_uploaded = set()


def _upload(dev):
    """First eager call on a device: upload the code objects of all four builds at once, v74's first and in v74's order (default,
    max-ILP), then the two per-XCC builds, by one 8-workgroup launch of the 19-instruction probe kernel from each.
    Why: the runtime allocates device memory when a build's first kernel is launched. Taken lazily, in the middle of a caller's
    own allocations, that memory changes where the caller's later pools land, and memory-bound grids follow the pool's place: with
    the per-XCC max-ILP build first used by the eval's 5th workload, the plain 1024-block hold workloads (v74's kernel, v74's
    binary) read +0.09 +- 0.03 / +0.17 +- 0.06 us; with it uploaded ahead of v74's max-ILP build, s4_k3_b1 +0.14 +- 0.05 (the place of
    its code). With everything taken here, v74's objects first, every workload on a v74 kernel is within +- 0.05 us of v74
    (NOTES_v14 "the allocation term"). A capture that precedes every eager call cannot do this (no launch inside a capture that is
    not the kernel's): the builds are then loaded when first used, as in v74."""
    if dev in _uploaded or torch.cuda.is_current_stream_capturing():
        return
    _uploaded.add(dev)
    buf = torch.full((8,), -1, dtype=torch.int32, device=f"cuda:{dev}")
    with torch.cuda.device(dev):
        for m in (V._ext(False), V._ext(True), _xext(False), _xext(True)):
            m.xcc_probe(buf, 8, 1)
        torch.cuda.synchronize(dev)


def _xcc_order():
    """XCCs by start offset (ties by id): 1, 0, 3, 7, 2, 5, 6, 4."""
    return sorted(range(8), key=lambda x: (XCC_START_US[x], x))


def _xf_mask(m):
    return sum(1 << x for x in _xcc_order()[:m])


def _uq_consts():
    """(slow XCC, receiver mask) of xcc_items(): the XCC with the largest write ratio, and the two XCCs that start first outside its
    write-path pair {2k, 2k + 1} (MEASURE.md 2.2)."""
    slow = max(range(8), key=XCC_WRITE_RATIO.__getitem__)
    rcv = [x for x in _xcc_order() if x >> 1 != slow >> 1][:2]
    assert rcv == sorted(rcv, reverse=True), "the kernel hands the moved items to the receivers in descending id order"
    return slow, sum(1 << x for x in rcv)


def _xcc_tab(dev):
    """The per-XCC tables are MI350X's: 8 XCCs as the HIP runtime reports them (hipDeviceAttributeNumberOfXccs; a CU count alone would
    also pass on another part, HW_RESEARCH.md card B), 256 CUs, a build whose compiled XCC constants are the tables', and v74's X1
    kernel on the XCC the table names. Host queries only (no launch): valid inside a capture."""
    if dev not in _tab:
        m = _xext(False)
        with torch.cuda.device(dev):
            c = list(m.xcc_consts())
            _tab[dev] = (m.num_xcc() == 8 and V._num_cu(dev) == 256 and c[0] == X1_XCC and V._ext(False).x1_xcc() == X1_XCC
                         and c[-2:] == list(_uq_consts()) and all(_xf_mask(i + 2) == v for i, v in enumerate(c[1:-2])) and len(c) - 3 >= XF_MAX - 1)
    return _tab[dev]


def _stream_ok(dev):
    """The current stream has no CU mask: a masked queue is another queue type than the one the device was probed on."""
    with torch.cuda.device(dev):
        return bool(_xext(False).stream_unmasked())


def _rr_exact(x, n_xcc=8):
    """x[i] = XCC that workgroup i of one dispatch ran on (linear index). True iff the dispatch is an exact round-robin: workgroup i on
    XCC (x[0] + i) % n_xcc, every XCC id in range. Pure function of the probe result (tested on synthetic violations: chunked
    dispatch, fewer XCCs, permutations, a single misplaced workgroup; scratch_v14/xcc_tests.py)."""
    x = [int(v) for v in x]
    return len(x) >= n_xcc and all(0 <= v < n_xcc for v in x) and all((v - i) % n_xcc == x[0] % n_xcc for i, v in enumerate(x))


_PROBE_GRIDS = ((96, 1), (128, 2), (64, 3), (100, 3), (12, 17), (16, 9))  # X1 / XF forms (8 H' x B), a 1-D grid, UQ forms (H x (B + extra))
_probe_hook = None  # test hook: maps a probe result to what the checker sees (fault injection, scratch_v14/xcc_tests.py)


def _probe_now(dev):
    ok = _xcc_tab(dev)
    m = V._ext(False)  # v74's probe kernel
    for gx, gy in _PROBE_GRIDS if ok else ():
        buf = torch.full((gx * gy,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        with torch.cuda.device(dev):
            m.xcc_probe(buf, gx, gy)
        x = buf.cpu().tolist()
        if _probe_hook is not None:
            x = _probe_hook(x)
        ok = ok and _rr_exact(x)
    if not ok and _tab.get(dev):
        warnings.warn(f"kda decode kernel: the workgroup dispatch on cuda:{dev} is not an exact round-robin over 8 XCCs"
                      + (" any more: graphs captured on this device with the per-XCC launch forms are invalid and must be re-captured" if _x1.get(dev) else "")
                      + "; using the symmetric launch forms")
    _x1[dev] = ok
    return ok


def _x1_probe(dev):
    """Once per device, outside any capture: launch the probe kernel on grids of the launch forms that depend on the workgroup -> XCC
    mapping and check that every dispatch is an exact round-robin over the 8 XCCs (_rr_exact()). A capture that comes before the
    probe leaves the device unprobed: the symmetric grid is used (no host work inside a capture). A failure is sticky and warns."""
    if dev not in _x1:
        if torch.cuda.is_current_stream_capturing():
            return False
        _probe_now(dev)
    return _x1[dev]


def xcc_reprobe(dev=None):
    """Repeat the round-robin probe on a device (default: the current one), outside any capture. Returns the verdict; if a device that
    passed before fails now, the per-XCC launch forms are off from here on and the warning says that graphs captured with them
    must be re-captured. For operators after a driver / partition-mode change, and for the fault-injection tests."""
    dev = torch.cuda.current_device() if dev is None else dev
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("kda decode: xcc_reprobe() inside a graph capture")
    if _x1.get(dev) is False:
        return False
    return _probe_now(dev)


def _plain_lb(B, H, S, spec, ns, nth):
    if spec or S != 1 or ns != 1 or nth != 1024:
        return False
    return bool(V.prologue_first(B, H, S, spec, ns, nth)) and not V.next_round(B, H, S, spec, ns, nth) and not V.hold_ticks(B, H, S, spec, ns, nth)


def xcc_fill(B, H, S, spec, ns, nth):
    """Plain latency-bound whole-state grid (1024 threads, prologue-first, one round): XCC mask to run it on (0 = every XCC, the
    symmetric grid). The grid is launched as 8 ceil(H / m) x B blocks; the block on the r-th XCC of the mask does head
    (blockIdx.x / 8) m + r, the others return at entry (m = 1 is v74's X1 kernel). The event is F + max over the used XCCs of
    S_x + W_x, so the grid goes on the m XCCs with the smallest start offsets, with the smallest m for which the busiest of them
    (B ceil(H / m) blocks: a row of 8 workgroups serves m heads of one sequence) carries its state bytes (64 KB in + 64 KB out per
    block) inside a block's own memory phase: B ceil(H / m) x 128 KB / X1_XCC_BW <= X1_SPAN_US, i.e. 16 blocks per XCC (X1 vs v73:
    8 / 12 / 16 blocks -0.47 / -0.40 / -0.41 us, 24 blocks -0.07 / -0.14, 32 blocks +0.27; UNDERFLOOR.md E8b), up to XF_MAX XCCs
    (beyond it the last XCC's start offset is nearly the grid's own: m = 5 / 6 measured -0.16..+0.06 us)."""
    if not _X1_OK or not _plain_lb(B, H, S, spec, ns, nth):
        return 0
    for m in range(1, XF_MAX + 1):
        if B * -(-H // m) * 2 * 128 * 128 * 4 / X1_XCC_BW <= X1_SPAN_US:
            return _xf_mask(m)
    return 0


def xcc_first(B, H, S, spec, ns, nth):
    """1 = the grid is launched in an XCC-masked form (xcc_fill())."""
    return int(xcc_fill(B, H, S, spec, ns, nth) != 0)


def xcc_items(B, H, S, spec, ns, nth):
    """Single-round spec whole-state grid (1024 threads, prologue-first) with spare CUs: N | K << 9 (0 = off). The grid ends on the
    XCC with the slowest write path (XCC 3: its token loop lasts 1.22-1.44 x the even XCCs'; slack <= 0.15 us on nine shapes,
    ASYMMETRY A.2 / A.4), so that XCC keeps UQ_FRAC of its N / 8 rows and its last K items run on blocks behind the grid on the two
    XCCs that start first outside its write-path pair (never XCC 2: NOTES_v13 item 3). UQ_FRAC = 0.92: 22 of 24 items; 23 of 24 gives
    half, 21 of 24 or other receivers no more (ASYMMETRY B.6). Needs (i) a spare CU for every moved item: the launched grid must be one
    round (a moved item on a full grid is a second round: +4.8..+6.0 us, NOTES_v14 item U2); (ii) a write-bound grid, i.e. the
    predicate of wave_offset(): the state-write phase at the HBM write ceiling outlasts the block's own token loop by WS_US (below
    it the grid is latency-bound and ends on the XCCs that start last: 108 blocks at S = 2 +0.14 us, NOTES_v14 item U), so the grid
    is in the W1 or the SB class; (iii) the exact round-robin (_x1_probe()). Measured against v74 (24 pairs per cell): 22 write-bound
    shapes of 120-248 blocks -0.33..+0.07 us, mean -0.13, none slower beyond CI."""
    if UQ_FRAC >= 1 or not _X1_OK or not spec or ns != 1 or nth != 1024 or S < 2:
        return 0
    if not V.prologue_first(B, H, S, spec, ns, nth) or V.next_round(B, H, S, spec, ns, nth):
        return 0
    N = B * H
    K = int((1 - UQ_FRAC) * (N // 8) + 0.5)
    if K < 1 or K > 7 or H > 255 or N > 511:
        return 0
    rows = -(-N // 8) + -(-K // 2)  # the grid's rows of 8 workgroups + the receivers' rows (two receivers per row)
    if H * -(-8 * rows // H) > V._NUM_CU:
        return 0
    if N * S * 128 * 128 * 4 / V.WS_HBM_W - S * V.WS_TOK_US < V.WS_US:
        return 0  # not write-bound (the predicate of wave_offset()): the grid does not end on the slow write path
    return N | K << 9


def hold_arg(B, H, S, spec, ns, nth):
    """The launch's multi-purpose argument, as v74's: read-first hold ticks | state rows behind the barrier (1) | first nontemporal
    tail block | -1 wave offset | -2 XCC-masked plain grid (each kernel class reads at most one of them)."""
    return (V.hold_ticks(B, H, S, spec, ns, nth) or V.state_after_prologue(B, H, S, spec, ns, nth) or V.tail_nt(B, H, S, spec, ns, nth)
            or -V.wave_offset(B, H, S, spec, ns, nth) or -2 * xcc_first(B, H, S, spec, ns, nth))


def xv_arg(B, H, S, spec, ns, nth):
    """The launch's per-XCC argument: xcc_fill() mask (plain) | xcc_items() N | K << 9 (spec)."""
    return xcc_items(B, H, S, spec, ns, nth) if spec else xcc_fill(B, H, S, spec, ns, nth)


def _guard(dev, B, H, S, spec, ns, nth):
    """Sets _X1_OK for this call. The host queries and the probe run only for a call a per-XCC rule could apply to (so a process that
    never meets such a shape does exactly what v74 does)."""
    global _X1_OK
    _X1_OK = True
    cand = bool(xv_arg(B, H, S, spec, ns, nth))
    _X1_OK = cand and _xcc_tab(dev) and _stream_ok(dev) and _x1_probe(dev)


def run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
        state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    B = cu_seqlens.shape[0] - 1
    T = mixed_qkv.shape[0]
    S = T // max(B, 1)
    H = A_log.numel()
    spec = num_accepted_tokens is not None
    if S > V._SMAX or (not spec and S != 1) or not V._validate(B, T, S, H, spec, mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias,
                                                              norm_weight, ssm_state, state_indices, num_accepted_tokens, out):
        import kernel_aiter_fallback as fb
        return fb.run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight,
                      ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
    dev = mixed_qkv.device.index
    _upload(dev)
    V._NUM_CU = V._num_cu(dev)  # v74's cost model is evaluated for this call's device
    ns, nth = V.choose(B, H, S, spec)
    vec = V.use_vec(B, H, S, ns)
    _guard(dev, B, H, S, spec, ns, nth)
    hold = hold_arg(B, H, S, spec, ns, nth)
    xv = xv_arg(B, H, S, spec, ns, nth)
    args = (mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
            state_indices, num_accepted_tokens, out, V._workspace(mixed_qkv.device, B * H, vec), float(norm_eps),
            float(lower_bound), ns, nth, vec, V.prologue_first(B, H, S, spec, ns, nth), hold, V.next_round(B, H, S, spec, ns, nth))
    if xv and (spec or xv != 1 << X1_XCC):
        _xext(spec).run(*args, xv)  # a per-XCC kernel of this module's builds
    else:
        V._ext(spec).run(*args)  # a v74 kernel from v74's build (X1 included: hold = -2)


_FWD = {k for k in vars(V) if k.startswith("FORCE_")} | {"HOLD_MIN", "HOLD_CU_US", "HOLD_BW2", "HOLD_BW_NT", "V16_US_PER_TOKEN", "V16_FIXED_US",
                                                          "TN_US_PER_TOKEN", "TN_FIXED_US", "WS_HBM_W", "WS_TOK_US", "WS_US", "_NUM_CU"}


def __getattr__(k):
    """Names this module does not define are v74's."""
    return getattr(V, k)


class _Layer(types.ModuleType):
    """Assignments to v74's switches go to v74 (the tools set them on the kernel module)."""

    def __setattr__(self, k, v):
        if k in _FWD:
            setattr(V, k, v)
        else:
            super().__setattr__(k, v)


_me = sys.modules.get(__name__)
if _me is not None and getattr(_me, "__file__", None) == __file__:
    _me.__class__ = _Layer
