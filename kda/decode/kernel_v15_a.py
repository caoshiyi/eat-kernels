"""v15_a (v76) = v75 with the placement of its kernel descriptors chosen instead of drawn.

What a dispatch reads before its first instruction: on every XCC the command processor fetches the kernel descriptor (64 B in the
code object's .rodata) and the head of the kernel arguments. MI350X interleaves 4 KiB pages over its two I/O dies by a parity of
physical address bits (13, 21 and 28: NOTES_v15.md 1.0); a line on the other die costs an XCC 0.08-0.12 us more, and a dispatch ends on
the XCCs that start last (XCC_START_US: 4 and 6, i.e. the quad 4..7, which is one die). Measured on v75 in 600 fresh processes with
the descriptor page probed in each (scratch_v15/draw.py): descriptor on the die of XCCs 4-7 against the other die: B = 1 spec
-0.15..-0.19 us, mid-B spec and plain hold -0.04..-0.13, plain B = 1 -0.03..-0.04; never slower. Where a code object lands is the
driver's buddy allocator's choice per process: v75's descriptors are on the fast die in 28 % (max-ILP build) / 54 % (default build)
of the eval's processes, and differently on every GPU (1 of 30 on one, 13 of 30 on another).

Mechanism (a property of the binary and of one measurement per device; nothing about shapes, callers or call order):
  - kda_v15_a.cu holds every kernel of kda_v14_b.cu twice (the same source in two namespaces: identical instruction streams), with
    filler descriptors so that copy 1's descriptor table is exactly 8 KiB behind copy 0's, both inside the object's first buddy
    piece. Two lines 8 KiB apart in one piece differ in address bit 13 and in nothing above it: they are on different dies in every
    process, whatever the piece's physical address.
  - _place(dev), at the first eager call on a device (outside any capture): both objects are uploaded, then one wave on XCC 1 and
    one on XCC 5 time single cold loads at filler lines of the two descriptor windows (72 launches per object, each behind a
    512 MB fill of a buffer the extension allocates for the probe and frees after it: no torch tensor, nothing left behind). The copy whose window is nearer to the quad that starts last is used from then on; an unclear reading, a layout
    that is not the expected one, a device that is not an 8-XCC MI350X, or a capture before any eager call leave copy 0 (which is
    what v75 is: a draw).
One build per scheduler now (v74's kernels and the per-XCC kernels of v75 in the same object), so two code objects instead of four.

Everything else is v75: this module is a layer over private instances of kernel_v14_b / kernel_v13_a (cost model, guard, mailboxes;
the shared modules are not touched), with their build hooks pointed at the builds of kda_v15_a.cu. Names this module does not
define are theirs."""
import hashlib
import math
import math
import importlib.util
import os
import statistics
import struct
import sys
import types
import warnings

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kda_v14_b.cu")  # the kernels (v75's source, unchanged: what the asm checkers of validate_all.sh audit)
_SRC_W = os.path.join(_HERE, "kda_v15_a.cu")  # the build: that source twice + filler descriptors + the probe kernels


def _private(fname, name, shadow=None):
    """A private instance of a sibling module (so that redirecting its build hooks does not change the module other code imports)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in (shadow or {})}
    sys.modules[name] = m
    sys.modules.update(shadow or {})
    try:
        spec.loader.exec_module(m)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m


V = _private("kernel_v13_a.py", "kernel_v15_a__v13_a")
K = _private("kernel_v14_b.py", "kernel_v15_a__v14_b", {"kernel_v13_a": V})
assert K.V is V

# ---- builds: kda_v15_a.cu = 2 x kda_v14_b.cu + filler descriptors + probe kernels
_PADS = {False: (0, 120, 32, 0), True: (1, 66, 32, 0)}  # per build: filler descriptors (in front of copy 0, between the copies = 128 - kernels
# per copy, behind copy 1), and bytes of .bss filler. The filler is 0: with 1796096 / 505856 bytes both loaded objects are exactly 2 MiB and
# the driver serves them from its 2 MiB blocks instead of its small pieces, which moves the caller's first torch segments instead of the
# runtime's kernarg pools (NOTES_v15 2.4: state_indices on the fast die in 25 % instead of 57 % of the processes, score -0.0013). Not used.
if os.environ.get("KDA_EXPERIMENT") == "1" and os.environ.get("KDA_V15_BSS"):
    _PADS = {False: (0, 120, 32, 1796096), True: (1, 68, 32, 505856)}
_KALIGN = {False: 0, True: 0}
if os.environ.get("KDA_EXPERIMENT") == "1" and os.environ.get("KDA_V15_KALIGN"):
    _KALIGN = {False: int(os.environ["KDA_V15_KALIGN"].split(",")[0]), True: int(os.environ["KDA_V15_KALIGN"].split(",")[-1])}
_mods = {}
_lay = {}


def _ext(ilp=False):
    """ilp=False: default machine scheduler (plain kernels). ilp=True: max-ILP scheduler, spec kernels only. Flags as v74's builds."""
    if ilp not in _mods:
        rocm = V._rocm_version()
        if rocm != V._AUDITED_ROCM:
            warnings.warn(f"kda decode kernel: building with ROCm {rocm}, audited on {V._AUDITED_ROCM}: rerun scratch_final/validate_all.sh (asm checkers)")
        p0, p1, p2, bss = _PADS[ilp]
        # test builds only (fault injection of the bounded mailbox poll, e.g. KDA_CG="MB_BOUND=1 MB_FI=1"): extra -D switches
        cg = os.environ.get("KDA_CG", "").split() if os.environ.get("KDA_EXPERIMENT") == "1" else []
        digest = hashlib.sha1(open(_SRC_W, "rb").read() + open(_SRC, "rb").read() + rocm.encode() + repr((_PADS[ilp], _KALIGN[ilp])).encode()
                              + " ".join(cg).encode()).hexdigest()[:10] + ("_ilp" if ilp else "")
        bdir = os.path.join(_HERE, "build", "v15a_" + digest)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        xf = ["-mllvm", "-amdgpu-sched-strategy=max-ilp", "-DKDA_SPEC_ONLY"] if ilp else []
        try:
            m = load(name=f"kda_v15a_{digest}", sources=[_SRC_W], extra_cflags=["-O3"], extra_include_paths=[_HERE],
                     extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16", "-DSPOL_SPEC=17", "-DSPOL_PLAIN=17",
                                        "-DPFSTRIDE=" + str(2 << 20), "-DSTAG=150", "-DSTAGL=3", "-DBLD=2",
                                        f"-DKDA_V15_PAD0={p0}", f"-DKDA_V15_PAD1={p1}", f"-DKDA_V15_PAD2={p2}", f"-DKDA_V15_BSS={bss}", f"-DKDA_V15_KALIGN={_KALIGN[ilp]}"] + xf + ["-D" + f for f in cg],
                     build_directory=bdir, verbose=False)
        finally:
            if arch0 is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch0
        _mods[ilp] = m
        try:
            _lay[ilp] = _layout(m.__file__)
        except Exception as e:  # an unreadable binary only switches the choice off
            _lay[ilp] = dict(ok=False, why=f"layout not readable: {e!r}")
        if not _lay[ilp]["ok"]:
            warnings.warn(f"kda decode kernel: descriptor layout of the {'max-ILP' if ilp else 'default'} build is not the expected one "
                          f"({_lay[ilp]['why']}): descriptor placement stays a draw (copy 0)")
    return _mods[ilp]


_SRC_2 = os.path.join(_HERE, "kda_v17_a.cu")  # third code object: the 2-way V-split (V.split2()), kda_vec<64, S> of the same source
_mod2 = []


def _ext2():
    """The build of kda_v17_a.cu (max-ILP scheduler, flags as the spec build). Its own code object: the two objects above are
    byte-identical with or without it."""
    if not _mod2:
        rocm = V._rocm_version()
        cg = os.environ.get("KDA_CG", "").split() if os.environ.get("KDA_EXPERIMENT") == "1" else []  # test builds, as _ext()
        digest = hashlib.sha1(open(_SRC_2, "rb").read() + open(_SRC, "rb").read() + rocm.encode() + " ".join(cg).encode()).hexdigest()[:10] + "_ilp"
        bdir = os.path.join(_HERE, "build", "v17a_" + digest)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        try:
            _mod2.append(load(name=f"kda_v17a_{digest}", sources=[_SRC_2], extra_cflags=["-O3"], extra_include_paths=[_HERE],
                              extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16", "-DSPOL_SPEC=17", "-DSPOL_PLAIN=17",
                                                 "-DPFSTRIDE=" + str(2 << 20), "-DSTAG=150", "-DSTAGL=3", "-DBLD=2",
                                                 "-mllvm", "-amdgpu-sched-strategy=max-ilp", "-DKDA_SPEC_ONLY"] + ["-D" + f for f in cg],
                              build_directory=bdir, verbose=False))
        finally:
            if arch0 is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch0
    return _mod2[0]


_SRC_3 = os.path.join(_HERE, "kda_v18_a.cu")  # fourth code object: the claim grid (claim_grid()), kda_main<128, S, spec, 1024, PF, NX> of the same source with CLM = 1
_mod3 = []


def _ext3():
    """The build of kda_v18_a.cu (max-ILP scheduler, flags as the spec build but without KDA_SPEC_ONLY: that switch also instantiates
    every S = 8 spec kernel; this object holds the four claim-grid kernels only). Its own code object, as _ext2()'s."""
    if not _mod3:
        rocm = V._rocm_version()
        cg = os.environ.get("KDA_CG", "").split() if os.environ.get("KDA_EXPERIMENT") == "1" else []  # test builds, as _ext()
        digest = hashlib.sha1(open(_SRC_3, "rb").read() + open(_SRC, "rb").read() + rocm.encode() + " ".join(cg).encode()).hexdigest()[:10] + "_ilp"
        bdir = os.path.join(_HERE, "build", "v18a_" + digest)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        try:
            _mod3.append(load(name=f"kda_v18a_{digest}", sources=[_SRC_3], extra_cflags=["-O3"], extra_include_paths=[_HERE],
                              extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16", "-DSPOL_SPEC=17", "-DSPOL_PLAIN=17",
                                                 "-DPFSTRIDE=" + str(2 << 20), "-DSTAG=150", "-DSTAGL=3", "-DBLD=2",
                                                 "-mllvm", "-amdgpu-sched-strategy=max-ilp"] + ["-D" + f for f in cg],
                              build_directory=bdir, verbose=False))
        finally:
            if arch0 is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch0
    return _mod3[0]


V._ext = _ext  # the private v74 / v75 layers query and probe through these builds
K._xext = _ext
K._ext = _ext

# ---- layout of a built code object (host-side ELF reading, no device)
_WIN = 8192  # bytes: lines this far apart in one buddy piece are on different I/O dies (address bit 13)
_NLINE = 8  # filler descriptors a window must hold (the probe reads lines no dispatch reads)


def _code_object(so):
    """The gfx950 code object inside an extension: the .hip_fatbin section is a clang offload bundle."""
    b = open(so, "rb").read()
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    for s in sh:
        if b[stro + s[0]:b.index(b"\0", stro + s[0])] == b".hip_fatbin":
            fb = b[s[4]:s[4] + s[5]]
            break
    else:
        raise ValueError("no .hip_fatbin")
    if fb[:24] != b"__CLANG_OFFLOAD_BUNDLE__":
        raise ValueError("not an offload bundle")
    n, = struct.unpack_from("<Q", fb, 24)
    p = 32
    for _ in range(n):
        o, s, tl = struct.unpack_from("<QQQ", fb, p)
        p += 24
        t = fb[p:p + tl]
        p += tl
        if b"gfx950" in t:
            return fb[o:o + s]
    raise ValueError("no gfx950 entry")


def _symbols(co):
    """(name, value, size, type) of every defined symbol of an ELF64 object, and its loaded size."""
    if co[:4] != b"\x7fELF" or co[4] != 2:
        raise ValueError("not ELF64")
    phoff, shoff = struct.unpack_from("<QQ", co, 0x20)
    phentsize, phnum, shentsize, shnum, _ = struct.unpack_from("<HHHHH", co, 0x36)
    mem = 0
    for i in range(phnum):
        t, _, _, va, _, _, msz, _ = struct.unpack_from("<IIQQQQQQ", co, phoff + i * phentsize)
        if t == 1:
            mem = max(mem, va + msz)
    sh = [struct.unpack_from("<IIQQQQIIQQ", co, shoff + i * shentsize) for i in range(shnum)]
    tabs = [s for s in sh if s[1] == 2] or [s for s in sh if s[1] == 11]  # .symtab, else .dynsym
    out = []
    for s in tabs[:1]:
        stro = sh[s[6]][4]
        for j in range(s[5] // 24):
            nm, info, _, shndx, val, size = struct.unpack_from("<IBBHQQ", co, s[4] + j * 24)
            if shndx:  # defined symbols only
                out.append((co[stro + nm:co.index(b"\0", stro + nm)].decode(), val, size, info & 15))
    return out, -(-mem // 4096) * 4096


def _layout(so):
    """Descriptor layout of a build: ok iff the two copies' descriptor tables are pairwise exactly _WIN apart, each inside one
    _WIN-aligned window, both windows inside the object's first buddy piece (the largest power of two <= the loaded size: the
    driver allocates descending power-of-two pieces, each aligned to its own size, PLACEMENT_RESEARCH.md 2.4), and each window
    holds at least _NLINE filler descriptors. Returns the probe's inputs: where = (address, size) of kda_where's code, lines =
    (filler line addresses in copy 0's window, in copy 1's window)."""
    syms, mem = _symbols(_code_object(so))
    kd = [(n, v) for n, v, _, _ in syms if n.endswith(".kd")]
    c = [sorted(v for n, v in kd if f"3kc{i}" in n and "kda_xcc_probe" not in n) for i in (0, 1)]
    pad = sorted(v for n, v in kd if "4kpad" in n)
    wh = [(v, s) for n, v, s, t in syms if "kda_where" in n and t == 2]  # STT_FUNC
    piece = 1 << (mem.bit_length() - 1)
    why = None
    if len(wh) != 1 or wh[0][1] <= 0 or wh[0][1] >= 4096:
        why = "no kda_where"
    elif not c[0] or len(c[0]) != len(c[1]):
        why = "the copies differ in kernel count"
    elif any(b - a != _WIN for a, b in zip(c[0], c[1])):
        why = "descriptor tables not 8 KiB apart"
    elif len({a // _WIN for a in c[0]}) != 1:
        why = "a descriptor table straddles a window"
    elif piece < 4 * _WIN or c[1][-1] + 64 > piece:
        why = "descriptors outside the first buddy piece"
    w0 = c[0][0] // _WIN if c[0] else 0
    lines = ([a for a in pad if a // _WIN == w0], [a for a in pad if a // _WIN == w0 + 1])
    if why is None and min(map(len, lines)) < _NLINE:
        why = "too few filler lines in a window"
    return dict(ok=why is None, why=why, where=wh[0] if wh else None, lines=lines, n=len(c[0]), kd0=c[0][0] if c[0] else None, mem=mem, piece=piece)


# ---- the choice, once per device
PLACE_MIN_US = 0.03  # us: the two quads' readings of a window must differ by at least this, with opposite signs in the two windows.
# The die difference of a cold load is 0.08 us from XCCs 1 / 5 (NOTES_v14 item Cc: 0.32 / 0.40 us; 0.36 / 0.44 with the bit-14 term), the
# probe's resolution 0.01 us; measured |difference| on four GPUs: 0.06-0.10 us (NOTES_v15 section 2).
PLACE_REPS = 16  # readings per (window, quad). No verdict (-> copy 0) in 7.6 % of 328 probes with 8 readings, 2.5 % of 320 with 16; a wrong verdict in 0 of 648
PLACE_WARM = 2  # rounds in front that are not kept
PLACE_FLUSH_BYTES = 512 << 20  # the probe's own flush buffer (twice the 256 MB memory-side cache), allocated and freed inside _place()
PLACE_COLD_US = 0.2  # us: a reading below this is not a cold load (a warm line reads < 0.1 us): no verdict
FORCE_COPY = None  # tests / A-B arms: 0 or 1 = that copy everywhere; "far" = the opposite of the choice; "off" = no probe (copy 0: a draw)
_copy = {}  # (device, ilp) -> copy in use
_place_log = {}  # (device, ilp) -> what the probe read (diagnostics: place_report())
_placed = set()
_place_hook = None  # test hook: maps the probe's readings {(window, xcc): [us]} to what the rule sees (fault injection)


def _late_quad():
    """(an XCC of the quad that starts first, the same position in the quad that starts last), from v75's start-offset table. The two
    quads {0..3} and {4..7} are the two I/O dies' sides (NOTES_v14 item Cc, HW_RESEARCH.md card C)."""
    a = K.X1_XCC
    late = max(range(8), key=K.XCC_START_US.__getitem__)
    return (a, a ^ 4) if (late >> 2) != (a >> 2) else None


def _verdict(r):
    """r[(window, quad)] = cold-load latencies (us) of filler lines of descriptor window 0 / 1 read from the early quad (0) / the late
    quad (1). Returns the copy whose window is nearer to the late quad, or None when the readings do not say so clearly: each window
    needs >= 3 readings per quad, every median a cold load, the early-minus-late difference >= PLACE_MIN_US in one window and
    <= -PLACE_MIN_US in the other (two lines 8 KiB apart are on different dies: one window alone is not trusted). Pure function."""
    try:
        if any(len(r[(w, q)]) < 3 for w in (0, 1) for q in (0, 1)):
            return None
        m = {k: statistics.median(v) for k, v in r.items()}
    except (KeyError, TypeError):
        return None
    if any(not (PLACE_COLD_US <= v < 5.0) for v in m.values()):
        return None
    d = [m[(w, 0)] - m[(w, 1)] for w in (0, 1)]  # > 0: the window is far from the early quad, i.e. on the late quad's die
    if d[0] >= PLACE_MIN_US and d[1] <= -PLACE_MIN_US:
        return 0
    if d[1] >= PLACE_MIN_US and d[0] <= -PLACE_MIN_US:
        return 1
    return None


def _probe_place(dev, ilp):
    """Upload the build's code object (first launch) and read its two descriptor windows. Eager, synchronous, outside any capture."""
    m, lay = _ext(ilp), _lay[ilp]
    out = torch.zeros(2, dtype=torch.int64, device=f"cuda:{dev}")
    with torch.cuda.device(dev):
        m.where(out)  # the first launch from this build: the runtime loads the code object here
        pc = int(out[0].item())
        q = _late_quad()
        if not lay["ok"] or q is None or not K._xcc_tab(dev):
            return None, dict(why=lay["why"] or "not an 8-XCC MI350X")
        base = (pc - lay["where"][0]) & ~4095
        if not 0 <= pc - lay["where"][0] - base < lay["where"][1]:
            return None, dict(why="code object not found")
        if not m.flush_open(PLACE_FLUSH_BYTES):
            return None, dict(why="no memory for the probe's flush buffer")
        r = {}
        try:
            for rep in range(-PLACE_WARM, PLACE_REPS):  # the first PLACE_WARM rounds are not kept (first touches of the pages from each XCC)
                for w in (0, 1) if rep % 2 == 0 else (1, 0):
                    a = lay["lines"][w][(rep // 2) % len(lay["lines"][w])]
                    for qi in (0, 1) if (rep // 2) % 2 == 0 else (1, 0):
                        m.flush_fill()
                        out.zero_()
                        m.pp(base + a, out, q[qi])
                        t = int(out[0].item())
                        if t > 0 and rep >= 0:
                            r.setdefault((w, qi), []).append(t / 100.0)
        finally:
            torch.cuda.synchronize(dev)
            m.flush_close()
    if _place_hook is not None:
        r = _place_hook(r)
    return _verdict(r), dict(base=base, r=r)


def _place(dev):
    """First eager call on a device: upload both code objects (default build first) and choose each one's copy (module docstring).
    Inside a capture nothing can be launched: the copy stays 0 and the objects are loaded when first used, until an eager call."""
    if dev in _placed or torch.cuda.is_current_stream_capturing():
        return
    _placed.add(dev)
    for ilp in (False, True):
        c, log = (None, dict(why="off")) if FORCE_COPY == "off" else _probe_place(dev, ilp)
        if FORCE_COPY == "off":
            _ext(ilp).where(torch.zeros(2, dtype=torch.int64, device=f"cuda:{dev}"))
        _place_log[(dev, ilp)] = dict(log, verdict=c)
        _copy[(dev, ilp)] = 0 if c is None else c
    torch.cuda.synchronize(dev)


def copy_of(dev, ilp):
    """The copy a launch on this device uses now."""
    c = _copy.get((dev, bool(ilp)), 0)
    if FORCE_COPY in (0, 1):
        return FORCE_COPY
    if FORCE_COPY == "far" and _place_log.get((dev, bool(ilp)), {}).get("verdict") is not None:
        return 1 - c
    return c


def place_report():
    """{(device, max-ILP build?): (copy in use, verdict or None, early-minus-late median difference in us per window)}."""
    out = {}
    for k, v in _place_log.items():
        r = v.get("r") or {}
        d = tuple(round(statistics.median(r[(w, 0)]) - statistics.median(r[(w, 1)]), 3) if r.get((w, 0)) and r.get((w, 1)) else None for w in (0, 1))
        out[k] = (copy_of(*k), v.get("verdict"), d, v.get("why"))
    return out


K._upload = _place  # v75's eager upload of its four builds is this module's _place()


# ---- NTA (production page layout): nontemporal agent-scope state stores in the blocks behind the first generation of a spec multi-round
# grid whose written states are not 64 KiB-aligned. A state is 64 KB; the kernel's cached write-through (sc0 sc1) stores of a state that
# straddles a 64 KiB boundary run ~10 % slower than of an aligned one, sc1 nt stores ~3 % (pure 805 MB stream, 12 x 128 x 8: 144 us aligned,
# 159 / 149 us unaligned; work_decode_throughput/NOTES.md). The phase-1 pools (H x 64 KiB slots at a 2 MiB-aligned base) are aligned; vLLM's
# page pool is not (page = conv cache + states + padding: 31 of 32 states of the Kimi-K3 page are unaligned). On aligned states sc1 nt loses
# 1-2 % (the read side competes). The first generation keeps sc0 sc1 (kda_v14_b.cu ST_NTAG: nt stores of the blocks that start together
# starve XCCs 2 / 3 for the rest of the grid; work_layout_builder/NOTES.md section 2), so the model counts the bytes behind it.
NTA_GAIN = 0.075  # share of the stream time of the generations behind the first that sc1 nt stores give back on a pool of mixed unaligned phases
# (fit, 34 page / padded-stride grids of 1.3-12 rounds, H 8-96, S 2-8, three fresh processes each: gain = 0.071..0.084 x stream - 1.6..2.3 us)
NTA_BW = 5.0e6  # bytes per us: state traffic (read + written) of a multi-round spec grid on unaligned states with write-through stores
NTA_FIXED_US = 2.0  # us lost whatever the grid (fit intercept; 8.5 when the first generation was nt too)
NTA_MARGIN_US = 0.6  # us of predicted net gain required: NTA from 34.7 us of stream behind the first generation (59 boundary grids of 23-72 us, H 8-96,
# S 2-8, three fresh processes each: every grid from 35 us on gains or is neutral (-0.1..+3.1 us), below it H = 96 loses (96,8,4 -0.3, 96,5,8 -0.7 us))
NTA_ALIGN = 4096  # bytes: a state whose distance to a 64 KiB boundary is below this counts partly (256 B: free; 4 KiB: the full penalty)
NTA_ONE_PHASE = 0.5  # a pool whose states all share ONE unaligned phase (a dense pool at an offset) gains this share of what mixed phases gain
# (12 x 128 x 8: 4.6-8.9 us at 4 KiB / 16 KiB / 32 KiB / 60 KiB against 8.7-11.3 mixed; 3-round grids: -0.9..+1.1 against +1.5..+3.1)
FORCE_NTA = None  # experiments: 0 = never, 1 = on every spec multi-round grid of more than 2 rounds, 2 = of more than 1 round
_unal = {}


def _phases(ssm_state):
    """(share of the pool's states that are not 64 KiB-aligned, number of distinct phases), from the layout alone (address of slot 0 and slot
    stride; the H states of a slot are 64 KiB apart, so they share a phase)."""
    key = (ssm_state.data_ptr() & 65535, (ssm_state.stride(0) * 4) & 65535)
    u = _unal.get(key)
    if u is None:
        base, sb = key
        n = 65536 // math.gcd(sb, 65536) if sb else 1  # distinct phases the slots take
        tot = 0.0
        for i in range(min(n, 4096)):  # (more than 4096 phases: a uniform sample of them)
            ph = (base + (i * (n // min(n, 4096))) * sb) & 65535
            tot += min(1.0, min(ph, 65536 - ph) / NTA_ALIGN)
        u = _unal[key] = (tot / min(n, 4096), n)
        if len(_unal) > 4096:
            _unal.clear()
    return u


def unaligned_share(ssm_state):
    """Share of the pool's states that are not 64 KiB-aligned: 0.0 for the phase-1 pools, 0.97 for the Kimi-K3 page, 1.0 for a dense pool at an offset."""
    return _phases(ssm_state)[0]


def nta_weight(ssm_state):
    """How much of NTA_GAIN the pool's layout gives: the gain is not proportional to the unaligned share (half of the slots aligned, half at
    32 KiB: the full gain, 12 x 128 x 8 +10.4 us against +10.8 on pages), and a single shared phase gains less (NTA_ONE_PHASE)."""
    share, n = _phases(ssm_state)
    return min(1.0, 2.0 * share) * (NTA_ONE_PHASE if n == 1 else 1.0)


def nt_all(B, H, S, spec, ns, nth, ssm_state):
    """Spec multi-round grids: True = the state stores of every block behind the first generation sc1 nt (the kernel's NTA copy; the TN tail
    stays). Cost model: predicted gain = NTA_GAIN x nta_weight(layout) x (state bytes of the blocks behind the first generation) / NTA_BW
    - NTA_FIXED_US >= NTA_MARGIN_US (at S = 8 on pages: from 550 blocks = 2.15 rounds on)."""
    nxt = V.next_round(B, H, S, spec, ns, nth)
    if not spec or S < 2 or not nxt or B * H <= nxt or B * H >= 1 << 30:
        return False
    if FORCE_NTA is not None:
        return FORCE_NTA == 2 or (bool(FORCE_NTA) and B * H > 2 * nxt)
    gain = NTA_GAIN * nta_weight(ssm_state) * (B * H - nxt) * (S + 1) * 128 * 128 * 4 / NTA_BW - NTA_FIXED_US
    return gain >= NTA_MARGIN_US


# ---- claim grid (iteration 8, pull_builder; kernel: "Claim grid" in kda_v14_b.cu kda_main, object kda_v18_a.cu)
# Measured on the plain grid (work_pull_builder/NOTES.md section 1): the runtime gives every XCC the same number of blocks, a block on an
# odd XCC takes 12-15 % longer, so the even XCCs are done 9 % of the call earlier; the memory system is the global bottleneck, so the odd
# XCCs speed up when they are alone, but the last blocks of a grid are latency-bound stragglers. When the lag exceeds about one block span
# (from ~5 rounds on) the even XCCs can take whole items off the odd ones: the launch carries CLM_ROWS x B extra rows of blocks and the
# blocks of the last generation(s) and of the extra rows take their items from a per-capture self-wrapping counter. Measured against the
# plain grid on shared pools (us; pages with NTA / dense): 12,107,8 +4.5 / +2.3; 12,128,8 +3.0 / +1.7; 96,16,8 +4.9 / +2.3; 12,256,8
# +10..12 / +10; 96,32,8 +11.6 / +12.5; 12,512,8 +28 / +27; 16,128,6 +4.0 / +1.8; below: 12,64,8 +0.3 / -1.5..-2.5; 12,85,8 0 / -0.1..-1;
# 16,64,6 dense -1.0; 16,96,6 dense 0.0. The claim costs a block ~1 us (an atomic and a block barrier in front of its first load), paid
# by the claiming generations only.
CLAIM = True  # False: never (the plain grid; what the co-tenant modes of kernel_v16f_a select)
_claim_off = False  # set per call by kernel_v16f_a._run(): a co-tenant mode is on
FORCE_CLM = None  # experiments: 0 = never, 1 = on every grid the kernel exists for
CLM_GAIN = 0.042  # share of the stream time behind the first generation the claim grid gives back
CLM_FIXED_US = 2.9  # us lost whatever the grid: cached state stores (dense / aligned pools); on from 105 us of stream behind the first generation
# (forced-on survey of 21 dense grids of 60-151 us, three fresh processes, one pool per grid, work_xcc_balance_builder/logs/dense_bound_*.log: every
# grid from 105 us on +1.0..+4.1 %; 94-100 us: S = 8 +1.0..+2.7 %, S = 5 +0.1 %; 90 us +0.2 %; 3-4 rounds (60-70 us) -0.8..0.0 %)
CLM_FIXED_NTA_US = 2.2  # ... with the NTA copy (nt_all(): unaligned pools); on from 88 us (forced-on survey of 22 page grids of 83-110 us, three
# processes, work_pull_builder/logs/mp_bound_*.log: every grid from 88 us on +1.8..+6.3 %, 83-86 us +0.1..+3.1 %)
CLM_MARGIN_US = 1.5  # us of predicted net gain required
# The tail term (iteration 9, work_xcc_balance_builder NOTES section 7). A CU runs an integer number of blocks, so the even XCCs' lead (they keep
# their rate in a saturated XCC pair, the odd member takes the shortfall: ~12 % of a round per round) can only be used in whole items. When the
# last generation is PARTIAL, its items are such whole items: on the plain grid an odd XCC starts its share of them when its last full generation
# ends, on the claim grid the CUs that free up first (the even XCCs') take them. That pays from three full generations in front of the tail (the
# lead must exceed what a tail block loses by running beside the odd XCCs' last full generation) and while the even XCCs can absorb the tail
# (tail share f of a generation <= 0.5; nothing left at f >= 0.75). Forced-on survey, three fresh processes, one pool per grid (logs/tail_*):
# 34 grids of 3.1-4.5 rounds, H 8-96, S 5-8: f <= 0.56: pages +0.5..+4.2 us (mean +2.4), dense -0.3..+3.1 (mean +1.7); f 0.62-0.69: +0.6..+1.3 /
# -0.8..+1.1; f >= 0.75: -0.2..+1.4 / -0.6..+0.8 (H 48 / 96: +3). Two full generations in front (2.1-2.9 rounds): +0.5 us mean: off.
CLM_TAIL = 0.06  # tail gain as a share of the stream time of ONE generation (S = 8: 1.8 us)
CLM_TAIL_ROUNDS = 3  # full generations in front of the partial one from which the tail term counts
CLM_TAIL_F0, CLM_TAIL_F1 = 0.5, 0.75  # tail share of a generation up to which the term is full / from which it is zero
CLM_ROWS = 0.06  # extra rows of blocks per row of the grid (3 % .. 12 % measured equal within 0.5 %)
CLM_GEN_ROUNDS = 12.0  # one claiming generation per this many rounds (the odd XCCs' lag is ~9 % of the call: 1 generation at 12 rounds, 2 at 24)


def claim_grid(B, H, S, spec, ns, nth, nta):
    """Spec multi-round whole-state grids: None = the plain grid, or (extra rows, c0) = the claim grid: blocks from grid index c0 on take
    their item from the counter. Cost model: gain = CLM_GAIN x (stream time of the blocks behind the first generation, as nt_all()) -
    CLM_FIXED[_NTA]_US + the tail term (a partial last generation behind >= 3 full ones) >= CLM_MARGIN_US (S = 8: from 3.9 rounds on with NTA,
    from 4.5 rounds on without; with a partial last generation of up to ~0.6 of a full one from 3 / 4 rounds)."""
    nxt = V.next_round(B, H, S, spec, ns, nth)
    if not CLAIM or _claim_off or not spec or S < 5 or S > 8 or ns != 1 or nth != 1024 or not nxt or B * H <= nxt or H > 4095:
        return None
    if FORCE_CLM is not None:
        if not FORCE_CLM:
            return None
    else:
        gain = CLM_GAIN * (B * H - nxt) * (S + 1) * 128 * 128 * 4 / NTA_BW - (CLM_FIXED_NTA_US if nta else CLM_FIXED_US)
        k0 = (B * H - 1) // nxt  # full generations in front of the last one
        if k0 >= CLM_TAIL_ROUNDS:
            f = (B * H - k0 * nxt) / nxt  # the last generation's share of a full one (1.0: not partial)
            gain += CLM_TAIL * nxt * (S + 1) * 128 * 128 * 4 / NTA_BW * min(1.0, max(0.0, (CLM_TAIL_F1 - f) / (CLM_TAIL_F1 - CLM_TAIL_F0)))
        if gain < CLM_MARGIN_US:
            return None
    rows = max(1, int(math.ceil(CLM_ROWS * B)))
    c0 = max(nxt, B * H - nxt * max(1, int(math.ceil(B * H / nxt / CLM_GEN_ROUNDS))))
    if rows > 4095 or c0 >= 1 << 19 or (B + rows) * H >= 1 << 30:
        return None
    return rows, c0


# The counter words. One workspace (4 KiB: 63 words, each on its own 64 B line) per (capture, stream), released with the graph exactly
# as v74's V-split workspaces are (V._capture_key(): calls of one capture on one stream are chained, never in flight together; a HIP
# user object retained by the graph reports its destruction through v74's cell registry); eager calls share one per (device, stream). Inside a workspace a word
# belongs to one (claimers, items) pair: launches of the same grid may share a word (each leaves it where it found it), launches of
# different grids must not (the second would start misaligned: slower, never wrong).
# A capture can allocate and fill nothing, so its workspace comes from spares that the eager calls of claim grids keep filled (and
# warmup()); a capture that finds none (no eager claim-grid call before it in the process), a runtime that cannot name the capture,
# or a 64th grid in one workspace: the plain grid, silently. Any content of a word is valid for the kernel; the sentinel (>= every
# top) or 0 is the aligned start.
_CLM_WORDS = 64
_CLM_NSPARE = 16  # as v74's _NSPARE: a capture cannot make one (4 KiB each; made at the first eager call of a claim grid or by warmup())
_clm_eager = {}  # (device, stream) -> [workspace, {(claimers, items): word index}]
_clm_cap = {}  # capture key -> the same
_clm_spare = {}  # device -> sentinel-filled workspaces for the next captures
_clm_free = {}  # device -> workspaces of dead graphs (refilled before reuse)


def _clm_reap():
    """Workspaces of captures whose graph is gone (v74's cell registry says so: V._retain() / V._reap()) go to the free list."""
    V._reap()
    for key in [k for k in _clm_cap if k not in V._cell]:
        ent = _clm_cap.pop(key)
        if len(_clm_free.setdefault(key[0], [])) < 8:
            _clm_free[key[0]].append(ent[0])


def _clm_retain(key, graph):
    """Tie `key` to the life of the graph being captured, through v74's registry (a user object whose destroy callback is libc's
    time() on a never-freed cell: no Python runs when the runtime fires it)."""
    return V._hip_lib() is not None and bool(V._retain(key, graph))


def _clm_spares(dev):
    """Eager only: keep the device's spares filled (dead graphs' workspaces first, refilled here)."""
    sp = _clm_spare.setdefault(dev, [])
    while len(sp) < _CLM_NSPARE:
        fr = _clm_free.get(dev)
        if fr:
            t = fr.pop()
            V._whole(t).fill_(V._SENT)  # through a normal view of the storage: the workspace may be an inference tensor and this call may run outside inference_mode (final_auditor A6)
        else:
            t = torch.full((16 * _CLM_WORDS,), V._SENT, dtype=torch.int32, device=torch.device("cuda", dev))
        sp.append(t)
    return sp


def _clm_word(dev, claimers, items):
    """The counter word of this launch (a view starting at the word), or None: use the plain grid."""
    _clm_reap()
    if torch.cuda.is_current_stream_capturing():
        try:
            key, graph = V._capture_key(dev)
        except Exception:  # noqa: BLE001
            key = None
        if key is None:
            return None
        key = key + ("claim",)  # this layer's entry in v74's registry (v74's own key is the bare triple)
        ent = _clm_cap.get(key)
        if ent is None:
            sp = _clm_spare.get(dev)
            try:
                if not sp or not _clm_retain(key, graph):
                    return None
            except Exception:  # noqa: BLE001
                return None
            ent = _clm_cap[key] = [sp.pop(), {}]
    else:
        _clm_spares(dev)
        key = (dev, torch.cuda.current_stream(dev).cuda_stream)
        ent = _clm_eager.get(key)
        if ent is None:
            ent = _clm_eager[key] = [torch.full((16 * _CLM_WORDS,), V._SENT, dtype=torch.int32, device=torch.device("cuda", dev)), {}]
    i = ent[1].get((claimers, items))
    if i is None:
        if len(ent[1]) >= _CLM_WORDS - 1:
            return None
        i = ent[1][(claimers, items)] = len(ent[1]) + 1
    return ent[0][16 * i:16 * i + 1]


def reset_workspace():
    """v74's reset_workspace() plus the claim counters (back to the sentinel: the aligned start)."""
    V.reset_workspace()
    _clm_reap()
    for t in ([e[0] for e in list(_clm_eager.values()) + list(_clm_cap.values())] + [t for l in list(_clm_spare.values()) + list(_clm_free.values()) for t in l]):
        V._whole(t).fill_(V._SENT)  # (see _clm_spares: legal in every autograd mode)
    torch.cuda.synchronize()


def workspace_health(heal=False):
    """v74's workspace_health() plus the claim counters (iteration 10, final_auditor's finding A2). A counter word rests at the sentinel (never
    used) or at its top = claimers - 1 (where every completed launch leaves it). Any other value is what an INCOMPLETE claim-grid launch
    leaves (a device error, a killed launch): results stay correct (any content of the word is a valid start), but every later launch of
    that grid starts misaligned, +1..6 % for the life of the graph. "claim_misaligned" counts such words (it does not change "ok": no
    result is wrong); heal=True returns the workspaces that hold one to the sentinel ("claim_healed"), as reset_workspace() does for all."""
    rep = V.workspace_health(heal)
    _clm_reap()
    ents = [e for e in list(_clm_eager.values()) + list(_clm_cap.values()) if e[1]]
    for d in {e[0].device for e in ents}:
        torch.cuda.synchronize(d)
    mis = healed = 0
    for e, vals in [(e, e[0][::16].tolist()) for e in ents]:
        n = sum(1 for (claimers, _items), i in e[1].items() if vals[i] != V._SENT and vals[i] != claimers - 1)
        mis += n
        if n and heal:
            V._whole(e[0]).fill_(V._SENT)  # (see _clm_spares: legal in every autograd mode)
            healed += 1
    if healed:
        for d in {e[0].device for e in ents}:
            torch.cuda.synchronize(d)
    rep["claim_misaligned"], rep["claim_healed"] = mis, healed
    return rep


def workspace_report():
    """v74's workspace_report() plus the claim-counter workspaces (4 KiB each)."""
    r = V.workspace_report()
    _clm_reap()
    r.update(claim_captured=len(_clm_cap), claim_eager=len(_clm_eager), claim_spares=sum(len(v) for v in _clm_spare.values()),
             claim_free=sum(len(v) for v in _clm_free.values()))
    return r


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
        return fb.run(mixed_qkv, conv_state, conv_weight, gate, beta.contiguous(), out_gate.contiguous(), A_log, dt_bias, norm_weight,
                      ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
    dev = mixed_qkv.device.index
    _place(dev)
    V._NUM_CU = V._num_cu(dev)  # v74's cost model is evaluated for this call's device
    nb = V.halves(B, H, S, spec)
    if nb:  # a little more than one block per CU: two launches over disjoint sequence ranges, each a single-round grid (V.halves())
        for lo, hi in ((0, nb), (nb, B)):
            run(mixed_qkv[lo * S:hi * S], conv_state, conv_weight, gate[lo * S:hi * S], beta[lo * S:hi * S], out_gate[lo * S:hi * S], A_log, dt_bias,
                norm_weight, ssm_state, state_indices[lo:hi], num_accepted_tokens[lo:hi], cu_seqlens[:hi - lo + 1], norm_eps, lower_bound,
                out[lo * S:hi * S])
        return
    ns, nth = V.choose(B, H, S, spec)
    vec = V.use_vec(B, H, S, ns)
    if ns == 2:  # the 2-way split: its own code object (kda_v17_a.cu), no per-XCC rule, no copy choice
        return _ext2().run2(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
                            state_indices, num_accepted_tokens, out, V._workspace(mixed_qkv.device, B * H, 1), float(norm_eps),
                            float(lower_bound))
    K._guard(dev, B, H, S, spec, ns, nth)
    hold = K.hold_arg(B, H, S, spec, ns, nth)
    xv = K.xv_arg(B, H, S, spec, ns, nth)
    nta = nt_all(B, H, S, spec, ns, nth, ssm_state)
    if nta:  # NTA: -(first TN tail block), or -(B H + 1) when there is no tail
        hold = -(hold if hold > 0 else B * H + 1)
    cg = None if (xv or vec) else claim_grid(B, H, S, spec, ns, nth, nta)
    if cg is not None:
        word = _clm_word(dev, (B + cg[0]) * H - cg[1], B * H)
        if word is not None:  # (None: no workspace for this capture, the plain grid below)
            return _ext3().run3(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
                                state_indices, num_accepted_tokens, out, word, float(norm_eps), float(lower_bound), hold,
                                V.next_round(B, H, S, spec, ns, nth), cg[0], cg[1])
    if not spec and xv == 1 << K.X1_XCC:
        xv = 0  # m = 1 is v74's X1 kernel (hold = -2), as in v75
    _ext(spec).run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
                   state_indices, num_accepted_tokens, out, V._workspace(mixed_qkv.device, B * H, vec), float(norm_eps),
                   float(lower_bound), ns, nth, vec, V.prologue_first(B, H, S, spec, ns, nth), hold, V.next_round(B, H, S, spec, ns, nth),
                   xv, copy_of(dev, spec))


_OWN = {"_mod2", "_mod3", "CLAIM", "_claim_off", "FORCE_CLM", "CLM_GAIN", "CLM_FIXED_US", "CLM_FIXED_NTA_US", "CLM_MARGIN_US", "CLM_TAIL", "CLM_TAIL_ROUNDS", "CLM_TAIL_F0", "CLM_TAIL_F1", "CLM_ROWS", "CLM_GEN_ROUNDS", "_CLM_WORDS", "_CLM_NSPARE", "_clm_eager", "_clm_cap", "_clm_spare", "_clm_free",  "FORCE_NTA", "NTA_ONE_PHASE", "NTA_GAIN", "NTA_BW", "NTA_FIXED_US", "NTA_MARGIN_US", "NTA_ALIGN", "_unal", "FORCE_COPY", "PLACE_MIN_US", "PLACE_COLD_US", "PLACE_REPS", "PLACE_WARM", "PLACE_FLUSH_BYTES", "_place_hook", "_copy", "_placed", "_place_log", "_mods", "_lay", "_PADS"}


def __getattr__(k):
    """Names this module does not define are v75's (which passes on to v74's)."""
    return getattr(K, k)


class _Layer(types.ModuleType):
    """Assignments to v75's / v74's switches go to the private layers (the tools set them on the kernel module)."""

    def __setattr__(self, k, v):
        if k in _OWN or k.startswith("__") or not hasattr(K, k):
            super().__setattr__(k, v)
        else:
            setattr(K, k, v)


_me = sys.modules.get(__name__)
if _me is not None and getattr(_me, "__file__", None) == __file__:
    _me.__class__ = _Layer
