"""v16f_a (v77) = v76 with the host-side defects of REDTEAM16.md closed. Same builds, same kernels, same cost model, same dispatch.

Nothing here runs at a graph replay: every change is in what run() does on the host around the launch.

  F1  the guard's verdict (kernel_v14_b._X1_OK) and the CU count (kernel_v13_a._NUM_CU) are per-call decisions kept in module
      globals: two threads calling run() swapped them (hardware: 119 of 300 000 calls on a rejected stream launched a per-XCC form).
      run() now holds one process-wide re-entrant lock from the first host decision to the (asynchronous) launch.
  F2  the placement probe's flush buffer is one global per extension: first calls on two devices from two threads faulted the
      context (hardware: 5 of 5 processes). Closed by the same lock.
  F4  the launch went to the CURRENT device's stream whatever device the tensors are on (hardware: context fault). The host part
      and the launch now run under torch.cuda.device(<the tensors' device>) when the two differ.
  F5  the mailboxes that captured graphs address by raw pointer were owned by module globals of a private module instance:
      importlib.reload() freed them under live graphs (hardware: the tensor is freed). The registries (and the per-device probe
      verdicts) now live on a process-wide holder, sys.modules["_kda_decode_state"], that a reload re-attaches.
  F6 / F11  an exception inside a probe propagated out of the caller's first run() and could leave 512 MiB allocated. The probes are
      now best-effort: any failure is a warning and the fallback the probe already had (copy 0 / symmetric launch forms); the flush
      buffer is released on every path. A first eager run() beside ANOTHER thread's capture also invalidated that capture (HIP's
      global capture mode): the probes now run with this thread in hipStreamCaptureModeThreadLocal (_own_captures_only()); a failure
      that still names a capture is retried at the next eager calls (_PLACE_TRIES), anything else is final for the process.
  F9  the round-robin probe ran at the first eager call of a shape a per-XCC rule applies to, so the rules depended on which shape
      a caller happened to run eagerly first. It now runs with the placement probe, at the first eager call on a device (in the
      eval's own order that is the same call: nothing moves there). warmup() does all first-call work on request.
  F12 the placement probe issues loads at raw addresses computed from a loader-layout assumption. It now runs only on the audited
      ROCm, only when /proc/self/maps shows the probed lines inside one device mapping, and a reading counts only if the dword it
      loaded is the dword the ELF has at that address.
  F13 _X1_OK is reset after every call (an exception in a query used to leave it True); assignments to a name kernel_v15_a itself
      defines go to that module, not past it to the v75 layer.

Layering: this module is a layer over a private instance of kernel_v15_a.py (v76), which is itself a layer over private instances of
kernel_v14_b.py (v75) / kernel_v13_a.py (v74). Names this module does not define are theirs. See NOTES_final16.md."""
import contextlib
import ctypes
import importlib.util
import os
import struct
import sys
import threading
import types
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kda_v14_b.cu")  # the kernels (what the asm checkers of validate_all.sh audit; unchanged since v75)

# ---- process-wide state: survives importlib.reload() of this module and is shared by every instance of it in the process
_STATE = sys.modules.get("_kda_decode_state")
if _STATE is None:
    _STATE = types.ModuleType("_kda_decode_state")
    _STATE.lock = threading.RLock()
    _STATE.reg = {}
    sys.modules["_kda_decode_state"] = _STATE
_LOCK = _STATE.lock


def _private(fname, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M = _private("kernel_v15_a.py", "kernel_v16f_a__v15_a")  # v76
K, V = M.K, M.V  # its private v75 / v74 layers


def _set(mod, k, v):
    types.ModuleType.__setattr__(mod, k, v)  # on that module itself, past its forwarding __setattr__


# F5: what must outlive a module instance. Mailboxes (captured graphs point at them), and what was learned about a device.
for _mod, _names in ((V, ("_mbox", "_captured", "_spare", "_spare2", "_retired", "_cnt", "_cap", "_free", "_dead", "_uo")), (K, ("_x1", "_tab")),
                     (M, ("_copy", "_place_log", "_placed"))):
    for _n in _names:
        _set(_mod, _n, _STATE.reg.setdefault(("v16f_a", _n) + ((V._MB_REC,) if _mod is V else ()), vars(_mod)[_n]))
_tries = _STATE.reg.setdefault(("v16f_a", "_tries"), {})  # device -> probe attempts that ended in a transient failure

HOST_PROBES = True  # False (set before the first call): no probe, no synchronisation, no transient allocation at the first call: copy 0
# and the symmetric launch forms, i.e. what v73 launches. For callers that cannot have a blocking call in their first eager run().
_PLACE_TRIES = 3  # eager calls at which the probes are attempted when they fail because another thread is capturing

# ---- co-tenant mode (iteration 7, backlog D): an operator's switch for a GPU shared with kernels that fill whole XCCs (their CUs' LDS).
# The dispatcher hands workgroups to the 8 XCCs round-robin; an XCC with fewer free CUs than a launch puts on it makes the launch wait
# for the co-tenant (platform rule, AITER's kernels too), and a CU without free LDS is not free for a kernel that needs LDS. The default
# launch forms trade some of that exposure for latency; each level below gives part of it back (costs: work_decode_gatekeeper/NOTES.md D):
#   ""           default.
#   "symmetric"  no per-XCC forms (X1 / XF put 8 workgroups per head on the device where the symmetric grid puts one; UQ adds rows).
#   "nosplit"    + no V-split (2 / 4 / 8 workgroups per (n, h) pair and a mailbox exchange): one whole-state block per pair.
#   "lds_free"   every call on the AITER kernel (no LDS at all: never waits for LDS; the baseline's latency).
# Set with set_cotenant_mode() / kernel.decode_cotenant_mode(), or KDA_DECODE_COTENANT in the environment at import. It applies to calls
# made (and graphs captured) afterwards; graphs captured earlier keep their launch forms.
_COTENANT_MODES = ("", "symmetric", "nosplit", "lds_free")
COTENANT_MODE = os.environ.get("KDA_DECODE_COTENANT", "")
if COTENANT_MODE not in _COTENANT_MODES:
    warnings.warn(f"kda decode kernel: unknown KDA_DECODE_COTENANT={COTENANT_MODE!r} (one of {_COTENANT_MODES[1:]}): ignored")
    COTENANT_MODE = ""
_nosplit_saved = []


def set_cotenant_mode(mode=""):
    """Select the co-tenant mode (see above) for the calls and captures that follow. Returns the previous mode."""
    global COTENANT_MODE
    if mode not in _COTENANT_MODES:
        raise ValueError(f"kda decode: co-tenant mode must be one of {_COTENANT_MODES}, got {mode!r}")
    with _LOCK:
        old, COTENANT_MODE = COTENANT_MODE, mode
        _apply_cotenant_mode()
    return old


def _apply_cotenant_mode():
    """Under the lock. nosplit and above: v74's FORCE_NSPLIT = 1 (choose() then never splits, the 2-way split included); restored when
    the mode is taken back."""
    if COTENANT_MODE in ("nosplit", "lds_free"):
        if not _nosplit_saved:
            _nosplit_saved.append(V.FORCE_NSPLIT)
        V.FORCE_NSPLIT = 1
    elif _nosplit_saved:
        V.FORCE_NSPLIT = _nosplit_saved.pop()


_hip = []


def _hip_runtime():
    """The HIP runtime this process already runs on (the libamdhip64 in its address space: never a second copy), or None."""
    try:
        with open("/proc/self/maps") as f:
            paths = {l.split()[5] for l in f if "libamdhip64" in l and len(l.split()) > 5}
        return ctypes.CDLL(sorted(paths)[0]) if len(paths) == 1 else None
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def _own_captures_only():
    """F6: a blocking HIP call from a thread in the default (global) capture mode fails while ANY thread captures, and invalidates
    that capture (hardware: the first eager run() beside another thread's capture raised and broke the capture). For the duration of
    the probes this thread is put in hipStreamCaptureModeThreadLocal (hipThreadExchangeStreamCaptureMode, the API meant for this:
    only this thread's own captures restrict it; the probes touch nothing another thread's capture owns), then restored. Measured
    (scratch_final16/exp_mode.py): in that mode hipMalloc / hipFree / hipMemsetAsync / .item() / a stream synchronise beside another
    thread's capture succeed and leave the capture valid; a DEVICE synchronise fails and invalidates it in every mode, so the probes
    synchronise their own stream only. If the runtime library cannot be reached nothing is changed (best-effort as below)."""
    old = None
    try:
        if not _hip:
            _hip.append(_hip_runtime())
        mode = ctypes.c_int(1)  # hipStreamCaptureModeThreadLocal
        if _hip[0] is not None and _hip[0].hipThreadExchangeStreamCaptureMode(ctypes.byref(mode)) == 0:
            old = mode.value
    except Exception:  # noqa: BLE001
        old = None
    try:
        yield
    finally:
        if old is not None:
            _hip[0].hipThreadExchangeStreamCaptureMode(ctypes.byref(ctypes.c_int(old)))


def _capturing_error(e):
    return "captur" in repr(e).lower()


def _elf_dword(ilp, a):
    """The 32-bit word the built code object holds at ELF address a (None if it is not in a loaded segment)."""
    cache = _STATE.reg.setdefault(("v16f_a", "_elf"), {})
    so = M._ext(ilp).__file__
    if so not in cache:
        co = M._code_object(so)
        phoff, = struct.unpack_from("<Q", co, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", co, 0x36)
        segs = []
        for i in range(phnum):
            t, _, off, va, _, fsz, _, _ = struct.unpack_from("<IIQQQQQQ", co, phoff + i * phentsize)
            if t == 1:
                segs.append((va, off, fsz))
        cache[so] = (co, segs)
    co, segs = cache[so]
    for va, off, fsz in segs:
        if va <= a and a + 4 <= va + fsz:
            return struct.unpack_from("<I", co, off + a - va)[0]
    return None


def _mapped(lo, hi):
    """[lo, hi) lies inside one device mapping of this process (the runtime maps device memory at the address the GPU uses). None when
    the process map cannot be read (then the check is not made)."""
    try:
        with open("/proc/self/maps") as f:
            for l in f:
                p = l.split()
                if len(p) > 5 and ("renderD" in p[5] or "kfd" in p[5]):
                    a, b = (int(x, 16) for x in p[0].split("-"))
                    if a <= lo and hi <= b:
                        return True
        return False
    except OSError:
        return None


def _probe_place(dev, ilp):
    """kernel_v15_a._probe_place() with the guards of F11 / F12: audited toolchain only, probed lines inside a device mapping, every
    reading's loaded dword compared with the ELF's, the flush buffer released on every path."""
    m, lay = M._ext(ilp), M._lay[ilp]
    out = torch.zeros(2, dtype=torch.int64, device=f"cuda:{dev}")
    with torch.cuda.device(dev):
        m.where(out)  # the first launch from this build: the runtime loads the code object here
        pc = int(out[0].item())
        q = M._late_quad()
        if not lay["ok"] or q is None or not K._xcc_tab(dev):
            return None, dict(why=lay["why"] or "not an 8-XCC MI350X")
        if V._rocm_version() != V._AUDITED_ROCM:
            return None, dict(why="not the audited ROCm: the loader layout is not assumed")
        base = (pc - lay["where"][0]) & ~4095
        if not 0 <= pc - lay["where"][0] - base < lay["where"][1]:
            return None, dict(why="code object not found")
        lines = [a for w in (0, 1) for a in lay["lines"][w]]
        want = {a: _elf_dword(ilp, a) for a in lines}
        if any(v is None for v in want.values()) or _mapped(base + min(lines), base + max(lines) + 64) is False:
            return None, dict(why="the probed lines are not where the layout says")
        if not m.flush_open(M.PLACE_FLUSH_BYTES):
            return None, dict(why="no memory for the probe's flush buffer")
        r = {}
        bad = 0
        try:
            for rep in range(-M.PLACE_WARM, M.PLACE_REPS):
                for w in (0, 1) if rep % 2 == 0 else (1, 0):
                    a = lay["lines"][w][(rep // 2) % len(lay["lines"][w])]
                    for qi in (0, 1) if (rep // 2) % 2 == 0 else (1, 0):
                        m.flush_fill()
                        out.zero_()
                        m.pp(base + a, out, q[qi])
                        t, v = out.tolist()
                        if t > 0 and v != want[a]:
                            bad += 1
                        if t > 0 and rep >= 0:
                            r.setdefault((w, qi), []).append(t / 100.0)
        finally:
            try:
                torch.cuda.current_stream(dev).synchronize()
            finally:
                m.flush_close()
    if bad:
        return None, dict(base=base, r=r, why=f"{bad} readings loaded other bytes than the code object's")
    if M._place_hook is not None:
        r = M._place_hook(r)
    return M._verdict(r), dict(base=base, r=r)


def _place(dev):
    """First eager call on a device (or warmup()): the placement probe of both builds (kernel_v15_a._place()) and the round-robin probe
    (kernel_v14_b._x1_probe(), F9), both best-effort (F6 / F11). Inside a capture nothing can be launched: nothing happens, copy 0 and
    the symmetric forms are used until an eager call."""
    if dev in M._placed or torch.cuda.is_current_stream_capturing():
        return
    M._placed.add(dev)
    try:  # backlog E: every code object the dispatch can select is BUILT with the first eager call (host only: compile on a cold
        M._ext2()  # build cache, dlopen otherwise; no device memory, nothing launched), so that no later call of a class pays a JIT build
        M._ext3()  # (the claim grid's object, likewise)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"kda decode kernel: the 2-way split build failed ({e!r:.200}); it will be retried at the first call that needs it")
    if not HOST_PROBES:
        M._ext(False)
        M._ext(True)
        for ilp in (False, True):
            M._place_log[(dev, ilp)] = dict(why="host probes off", verdict=None)
            M._copy[(dev, ilp)] = 0
        K._x1.setdefault(dev, False)
        return
    with _own_captures_only():
        transient = False
        for ilp in (False, True):
            try:
                if M.FORCE_COPY == "off":
                    c, log = None, dict(why="off")
                    M._ext(ilp).where(torch.zeros(2, dtype=torch.int64, device=f"cuda:{dev}"))
                else:
                    c, log = _probe_place(dev, ilp)
            except Exception as e:  # noqa: BLE001  a probe that cannot run is a probe without a verdict, not a failed call
                c, log = None, dict(why=f"probe failed: {e!r}"[:300])
                transient = transient or _capturing_error(e)
                warnings.warn(f"kda decode kernel: the descriptor placement probe on cuda:{dev} failed ({e!r:.200}): descriptor placement stays a draw (copy 0)")
            M._place_log[(dev, ilp)] = dict(log, verdict=c)
            M._copy[(dev, ilp)] = 0 if c is None else c
        try:
            torch.cuda.current_stream(dev).synchronize()
            if K._xcc_tab(dev) and K._stream_ok(dev):
                K._x1_probe(dev)  # F9: with the first eager call, not with the first eager call of a candidate shape
        except Exception as e:  # noqa: BLE001
            transient = transient or _capturing_error(e)
            if dev not in K._x1:
                warnings.warn(f"kda decode kernel: the round-robin probe on cuda:{dev} could not run ({e!r:.200})")
    if transient:  # a capture forbids the probes' blocking calls after all: try again at a later eager call
        _tries[dev] = _tries.get(dev, 0) + 1
        if _tries[dev] < _PLACE_TRIES:
            M._placed.discard(dev)
            if K._x1.get(dev) is not True:
                K._x1.pop(dev, None)
        elif dev not in K._x1:
            K._x1[dev] = False


_set(M, "_place", _place)
_set(M, "_probe_place", _probe_place)
K._upload = _place


def _upload2(d):
    """Load the 2-way split's code object on the device (the runtime loads an object at its first launch) by one real launch of its
    kernel on a one-pair batch of zeros (H = 1, B = 1, S = 2, a 3-slot pool of its own); also makes the 128-pair spare workspaces."""
    z = lambda *shape, dt=torch.bfloat16: torch.zeros(*shape, dtype=dt, device=d)  # noqa: E731
    f32 = torch.float32
    idx = torch.tensor([[1, 2]], dtype=torch.int32, device=d)
    acc = torch.ones(1, dtype=torch.int32, device=d)
    out = z(2, 128)
    mb = V._mailbox(d, 128)
    M._ext2().run2(z(2, 384), z(3, 4, 384), z(3, 4, 128, dt=f32), z(2, 1, 128), z(2, 1), z(2, 128), z(1, dt=f32), z(128, dt=f32),
                   torch.ones(128, dtype=f32, device=d), z(3, 1, 128, 128, dt=f32), idx, acc, out, mb, 1e-5, -5.0)
    with _own_captures_only():
        torch.cuda.current_stream(d.index).synchronize()
    if not bool(torch.isfinite(out.float()).all()):
        raise RuntimeError("the 2-way split kernel's self-test wrote non-finite values")


def _upload3(d):
    """Load the claim grid's code object on the device by one real launch of its kernel (H = 1, B = 1, S = 5, one extra row, every
    block claiming, a 6-slot pool of its own) and make the spare counter workspaces of later captures."""
    z = lambda *shape, dt=torch.bfloat16: torch.zeros(*shape, dtype=dt, device=d)  # noqa: E731
    f32 = torch.float32
    idx = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32, device=d)
    acc = torch.ones(1, dtype=torch.int32, device=d)
    out = z(5, 128)
    word = torch.full((16,), V._SENT, dtype=torch.int32, device=d)
    M._clm_spares(d.index)
    M._ext3().run3(z(5, 384), z(6, 7, 384), z(3, 4, 128, dt=f32), z(5, 1, 128), z(5, 1), z(5, 128), z(1, dt=f32), z(128, dt=f32),
                   torch.ones(128, dtype=f32, device=d), z(6, 1, 128, 128, dt=f32), idx, acc, out, word, 1e-5, -5.0, 0, 256, 1, 0)
    with _own_captures_only():
        torch.cuda.current_stream(d.index).synchronize()
    if not bool(torch.isfinite(out.float()).all()):
        raise RuntimeError("the claim grid kernel's self-test wrote non-finite values")


def warmup(device=None):
    """Everything any later call could need once per device (builds of the three code objects, their upload, placement probe,
    round-robin probe, spare workspaces of both sizes), on request: call it once per device at start-up, outside any capture and while
    no other thread captures. It blocks (~30 ms on a warm build cache; the JIT builds, about a minute each, on a cold one) and takes
    512 MiB of device memory for that time. After it no decode call compiles or uploads anything, whatever its shape (the AITER fallback
    for S > 8 or strided index tensors excepted: Triton specialises per shape at first use). Without it the builds and probes happen
    inside the first eager run() and the 2-way split's object is uploaded at the first call of that class; a process whose first run()
    is inside a capture gets copy 0 and the symmetric launch forms until its first eager call."""
    dev = torch.cuda.current_device() if device is None else torch.device(device).index if not isinstance(device, int) else device
    with _LOCK, torch.cuda.device(dev):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("kda decode: warmup() inside a graph capture")
        _place(dev)
        d = torch.device("cuda", dev)
        V._workspace(d, 1, False)
        V._mailbox(d, 1)
        with _own_captures_only():
            torch.cuda.current_stream(dev).synchronize()
        try:
            _upload2(d)
        except Exception as e:  # noqa: BLE001  best-effort, as the probes
            warnings.warn(f"kda decode kernel: warm-up of the 2-way split on cuda:{dev} failed ({e!r:.200})")
        try:
            _upload3(d)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"kda decode kernel: warm-up of the claim grid on cuda:{dev} failed ({e!r:.200})")
    return M.place_report()


def _run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
         state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    M._claim_off = bool(COTENANT_MODE)  # every co-tenant mode: the plain multi-round grid (no claim grid, kernel_v15_a.claim_grid())
    if not COTENANT_MODE:
        try:
            return M.run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
                         state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
        finally:
            K._X1_OK = False  # F13: the verdict is this call's, whatever happened in it
    # co-tenant modes (operator switch, see COTENANT_MODE)
    _apply_cotenant_mode()
    if COTENANT_MODE == "lds_free":
        import kernel_aiter_fallback as fb
        B = cu_seqlens.shape[0] - 1
        V._validate(B, mixed_qkv.shape[0], mixed_qkv.shape[0] // max(B, 1), A_log.numel(), num_accepted_tokens is not None, mixed_qkv,
                    conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state, state_indices, num_accepted_tokens, out)
        return fb.run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight,  # (AITER's kernel reads the
                      ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)  # strided slices itself)
    dev = mixed_qkv.device.index
    M._place(dev)  # the one-time probes first (they record the device's own verdict), then this call without a per-XCC form
    x1 = K._x1.get(dev)
    K._x1[dev] = False  # (restored below: the mode can be taken back)
    try:
        return M.run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
                     state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
    finally:
        K._X1_OK = False
        if x1 is None:
            K._x1.pop(dev, None)
        else:
            K._x1[dev] = x1


def run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
        state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    d = mixed_qkv.device
    if d.type != "cuda":
        raise TypeError(f"kda decode: the tensors must be on a HIP device, got {d}")
    a = (mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
         state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
    with _LOCK:  # F1 / F2: the launcher's per-call decisions live in module globals of the layers below
        if d.index != torch.cuda.current_device():  # F4: guard, probes, mailbox and launch all on the tensors' device
            with torch.cuda.device(d.index):
                return _run(*a)
        return _run(*a)


_OWN = {"HOST_PROBES", "_PLACE_TRIES", "COTENANT_MODE"}
_DUAL = {"_place", "_probe_place"}  # installed into kernel_v15_a by this module: a test that replaces one replaces both


def __getattr__(k):
    """Names this module does not define are v76's (which passes on to v75's and v74's)."""
    return getattr(M, k)


class _Layer(types.ModuleType):
    """Assignments go to the layer that defines the name (F13): this module's own switches here, a name kernel_v15_a defines on
    kernel_v15_a itself, anything else through kernel_v15_a's forwarding to the v75 / v74 switches."""

    def __setattr__(self, k, v):
        if k in _DUAL:
            super().__setattr__(k, v)
            _set(M, k, v)
        elif k in _OWN or k.startswith("__") or k in vars(self):
            super().__setattr__(k, v)
        elif k in vars(M):
            _set(M, k, v)
        else:
            setattr(M, k, v)


_me = sys.modules.get(__name__)
if _me is not None and getattr(_me, "__file__", None) == __file__:
    _me.__class__ = _Layer
