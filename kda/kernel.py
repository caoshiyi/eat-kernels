"""KDA for serving on MI350X: the package the eval (and a serving engine) imports. See program.md for the contract.

    meta = prefill_prepare(cu_seqlens, num_heads, checkpoint_offsets=None, checkpoint_state_indices=None)
    prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices,
            has_initial_state, cu_seqlens, out, checkpoint_offsets=None, checkpoint_state_indices=None, meta=None)
    decode(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
           state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)

Start of the production run: the two phase-1 kernels behind thin adapters.
  prefill: the amd_kda_mi350x final kernel (prefill/kernel.cu + main.cpp, ~3.0x vLLM on contiguous-state batches).
           The adapter gathers the initial states from the pool (zeros where has_initial_state is false), runs the
           kernel, copies the output into `out` and scatters the final states back. With checkpoints it runs the
           batch twice with every checkpointed sequence split at its offset: pass 1 yields the checkpoint states,
           pass 2 resumes the second segments from them.
  decode:  the amd_kda_decode_mi350x final kernel (v77, decode/kernel_v16f_a.py), already on the production contract
           except that it needs contiguous beta / out_gate, which the adapter copies.
"""
import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_D = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ prefill
_FLAGS = ["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16"]
_P = _D / "prefill"
_srcs = {f: (_P / f).read_text() for f in ("kernel.h", "kernel.cu", "main.cpp")}
_dig = hashlib.sha1(("".join(_srcs.values()) + " ".join(_FLAGS)).encode()).hexdigest()[:12]
_b = _P / "build" / _dig
_b.mkdir(parents=True, exist_ok=True)
for _f, _s in _srcs.items():
    if not (_b / _f).exists() or (_b / _f).read_text() != _s:
        (_b / _f).write_text(_s)
_pre = load(name=f"kda_prefill_{_dig}", sources=[str(_b / "main.cpp"), str(_b / "kernel.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=_FLAGS, build_directory=str(_b), verbose=False)


import os
# schedule metadata: 3 = relay (fused prologue + recurrence items), 1 = prologue + chunk walk with per-sequence groups,
# 2 = the same with uniform groups (vLLM's rule)
_MODE = int(os.environ.get("KDA_PROD_MODE", "3"))


# Micro sequences (integrator): a sequence with fewer than _MICRO (token, head) outputs is recomputed with the exact fp32
# recurrence of reference.py (a handful of torch ops on [m, H, 128, 128], no host sync) and its output tokens and final
# state row are written over the kernel's. Reason: the first tokens' output is beta * v * (k^ . q^) + (initial-state term);
# when k^ . q^ happens to be ~0 the result is a small difference and the 16-bit Gram operands of the kernels (ours on both
# paths, and vLLM's) leave an absolute error of ~7e-5 in the dot product. Over many (token, head) pairs that averages out;
# on a 1-token sequence at H = 1 it is the whole output: relative error > 2e-2 for 1.3 % of random draws (warm) / 10 % (cold),
# measured over 300 / 200 seeds (work_integrator/logs/h1_sweep_06.log, micro_sweep_06.log; vLLM: 0.7 % / 8.5 %). At 8 or
# more outputs the worst case over 200 seeds is 1.1e-2. The rule depends only on the layout (length x heads), never fires
# for H >= _MICRO (so not for Kimi-K3's 12 / 96 heads), and such sequences are too short to carry a checkpoint (64 tokens).
_MICRO = 8


def _micro_plan(cu_seqlens, num_heads, dev):
    if num_heads >= _MICRO:
        return None
    cu = cu_seqlens.detach().to("cpu", torch.int64)
    lens = cu[1:] - cu[:-1]
    sel = torch.nonzero((lens > 0) & (lens * num_heads < _MICRO)).flatten()
    if sel.numel() == 0:
        return None
    ln, st = lens[sel], cu[:-1][sel]
    t = torch.arange(int(ln.max()))
    valid = t[None, :] < ln[:, None]                      # [m, L]
    pos = st[:, None] + torch.where(valid, t[None, :], torch.zeros_like(t)[None, :])
    mi, ti = torch.nonzero(valid, as_tuple=True)          # host tensors: the valid (sequence, step) pairs
    return (sel.to(dev), pos.to(dev), valid.to(dev), pos[valid].to(dev), mi.to(dev), ti.to(dev))


def _micro_run(micro, q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices,
               has_initial_state):
    """Exact fp32 recurrence (reference.py) of the micro sequences; reads their initial rows, writes nothing."""
    sel, pos, valid, _, mi, ti = micro
    H, D = q.shape[2], q.shape[3]
    f32 = torch.float32
    qn = torch.nn.functional.normalize(q[0][pos].to(f32), dim=-1, eps=1e-6)      # [m, L, H, D]
    kn = torch.nn.functional.normalize(k[0][pos].to(f32), dim=-1, eps=1e-6)
    vf = v[0][pos].to(f32)
    gl = lower_bound * torch.sigmoid(A_log.to(f32).view(H, 1).exp() * (raw_g[0][pos].to(f32) + dt_bias.to(f32).view(H, D)))
    vm = valid[:, :, None]
    dec = torch.where(vm[..., None], gl.exp(), torch.ones_like(gl))             # padded steps: decay 1, beta 0
    bt = torch.where(vm, torch.sigmoid(raw_beta[0][pos].to(f32)), torch.zeros((), dtype=f32, device=q.device))
    rows = state_indices[sel].long()
    S = state_cache[rows].to(f32)                                               # [m, H, V, K]
    S = torch.where(has_initial_state[sel].bool()[:, None, None, None], S, torch.zeros_like(S)).transpose(-1, -2)
    o = torch.empty_like(vf)
    for t in range(pos.shape[1]):
        S = S * dec[:, t, :, :, None]
        kS = torch.einsum("nhk,nhkv->nhv", kn[:, t], S)
        S = S + kn[:, t, :, :, None] * (bt[:, t, :, None] * (vf[:, t] - kS))[:, :, None, :]
        o[:, t] = scale * torch.einsum("nhk,nhkv->nhv", qn[:, t], S)
    return rows, S.transpose(-1, -2), o[mi, ti]  # integer indices from the host plan (a mask index would sync)


def prefill_prepare(cu_seqlens, num_heads, checkpoint_offsets=None, checkpoint_state_indices=None, device=None):
    """Per-batch schedule metadata from host copies of the batch layout (shared by every KDA layer of the step)."""
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    # device tables are copied to `dev` inside prepare_prod (one H2D copy from a warm host scratch buffer)
    di = (dev.index if dev.index is not None else torch.cuda.current_device()) if dev.type == "cuda" else -1
    # Watchdog backstop: a relay fault of the previous step on this (device, current stream) that nobody has been told
    # about raises here, before the new step touches anything (host memory read, no sync). See prefill_health().
    if di >= 0:
        _pre.relay_check(di)
    d, h = _pre.prepare_prod(cu_seqlens, int(num_heads), _MODE, checkpoint_offsets, checkpoint_state_indices, di)
    return (d.to(dev), h, _micro_plan(cu_seqlens, int(num_heads), dev))


def prefill_range_guard(on=None):
    """Range guard of the relay path (process-wide; default ON, KDA_RANGE_GUARD=0 in the environment turns it off).

    The relay's intermediates are fp16 (6e-5 .. 65504). Inputs or states above that window (any |v| element or state
    element above ~6.5e4) give non-finite outputs and pool rows with the guard off. With it on, every V slice (16 value
    channels of a sequence-head) is judged where its chain ends (split sequences: in the scan) and a non-finite slice is
    recomputed in fp32 from its untouched initial state: correct, at ~5 us per token for the affected slices, and
    ~1.5 % slower on the relay path for everyone while the guard is on.
    The small side (magnitudes below ~1e-5, beta below ~1e-4 on a cold sequence: inaccurate, never non-finite) is only
    watched in a build with KDA_RX_FULL=1 (prefill/kernel.cu), which costs more and recomputes tiny slices every step.
    KDA_RELAY_OFF=1 (chunk-walk path) is the other range-safe setting (~2.4x slower).
    on=None: return the current setting; otherwise set it and return the previous one."""
    prev = bool(_pre.get_range_guard())
    if on is not None:
        _pre.set_range_guard(1 if on else 0)
    return prev


def prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices,
            has_initial_state, cu_seqlens, out, checkpoint_offsets=None, checkpoint_state_indices=None, meta=None):
    if meta is None:
        meta = prefill_prepare(cu_seqlens, q.shape[2], checkpoint_offsets, checkpoint_state_indices, q.device)
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and raw_g.is_contiguous()):
        q, k, v, raw_g = q.contiguous(), k.contiguous(), v.contiguous(), raw_g.contiguous()
    micro = meta[2] if len(meta) > 2 else None
    if micro is not None:
        with torch.no_grad():
            m_rows, m_state, m_out = _micro_run(micro, q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound,
                                                state_cache, state_indices, has_initial_state)
    _pre.run_prod(q, k, v, raw_g, raw_beta, A_log, dt_bias, state_cache, state_indices, has_initial_state, out,
                  checkpoint_offsets, checkpoint_state_indices, scale, lower_bound, meta[0], meta[1])
    if micro is not None:
        with torch.no_grad():
            out[0].index_copy_(0, micro[3], m_out.to(out.dtype))
            state_cache.index_copy_(0, m_rows, m_state.to(state_cache.dtype))


_RELAY_STATE = {0: "on", 1: "off: switched off", 2: "off: XCD probe failed", 3: "off: watchdog fault"}


def prefill_health(device=None, sync=False, acknowledge=False):
    """Relay watchdog status of a device (default: the current one). Never raises.

    The relay path hands state between workgroups; if a handoff wait times out (it cannot hang: bounded wait), the launch
    completes with INVALID results and records a fault. This reads the fault words (host memory; no device sync unless
    sync=True, which waits for the device so that the answer covers every prefill issued so far).
    Returns {"ok", "relay", "relay_state", "serial", "faults"}: ok is False while a fault has not been reported
    (by a prefill() / prefill_prepare() exception or acknowledge=True); `faults` lists every fault of the device:
    {"device", "stream" (torch.cuda.Stream.cuda_stream), "code", "kind", "launch_serial", "xcd", "hop", "pending"}.
    Scope of a fault: every prefill() issued on that (device, stream) since its last clean prefill_health(sync=True);
    `serial` (relay launches of the process so far) brackets a step: a fault with launch_serial <= the serial read at
    the start of the step belongs to an earlier step. acknowledge=True: the caller handles the fault itself; prefill()
    will not raise for it. After a fault the relay stays off on that device (chunk-walk path, correct, ~2.4x slower)."""
    idx = torch.cuda.current_device() if device is None else (device if isinstance(device, int) else torch.device(device).index)
    if sync:
        torch.cuda.synchronize(idx)
    state, serial, recs = _pre.relay_health(idx, bool(acknowledge))
    faults = []
    for dv, stream, code, pending in recs:
        wait = bool(code >> 62 & 1)
        faults.append({"device": dv, "stream": stream, "code": code, "kind": "wait_timeout" if wait else "xcd",
                       "launch_serial": (code & ~(3 << 61)) >> 24 if wait else None,
                       "xcd": code >> 20 & 0xf if wait else code >> 32 & 0xf, "hop": code & 0xfffff if wait else None,
                       "pending": bool(pending)})
    return {"ok": not any(f["pending"] for f in faults), "relay": "on" if state == 0 else "off",
            "relay_state": _RELAY_STATE[state], "serial": serial, "faults": faults}


# ------------------------------------------------------------------------------------------------ decode
sys.path.insert(0, str(_D / "decode"))
from kernel_v16f_a import run as _decode_run  # noqa: E402
import kernel_v16f_a as _decode_mod  # noqa: E402


def decode_warmup(device=None):
    """Once per device at start-up, outside any capture: builds and uploads every decode code object, runs the one-time probes and
    makes the spare workspaces, so that no later decode call (eager or captured, any shape) compiles, uploads or probes anything."""
    return _decode_mod.warmup(device)


def decode_cotenant_mode(mode=""):
    """Operator switch for a GPU shared with kernels that fill whole XCCs: "" (default), "symmetric", "nosplit", "lds_free"
    (decode/kernel_v16f_a.py COTENANT_MODE). Applies to the calls and graph captures that follow; returns the previous mode."""
    return _decode_mod.set_cotenant_mode(mode)


def decode_health(heal=False):
    """Health of the decode workspaces (V-split mailboxes, leave flags, tombstones), host side and synchronous: call it between steps.
    {"ok": True, ...} = every workspace is as a completed in-contract launch leaves it. ok False = something outside the contract
    happened (a graph replayed concurrently with itself, a killed launch): "stale_words" / "raised_flags" count leftovers a later launch
    could still consume, "ghost_events" launches that did consume some (the kernel stops that by itself within two launches of a
    workspace, but those results were wrong): treat the steps since the last clean check as suspect. heal=True refills the dirty
    workspaces (decode/kernel_v13_a.py workspace_health(); kernel._decode_mod.reset_workspace() refills everything).
    "ghost_events" also counts the sticky marks tombstones leave (kept until a refill, also after the kernel has healed itself).
    "claim_misaligned" = claim-grid counter words that an incomplete launch left off their rest value: correct results, but that grid
    is 1-6 % slower until heal=True ("claim_healed") or reset_workspace(); it does not change "ok" (decode/kernel_v15_a.py)."""
    return _decode_mod.workspace_health(heal)


def decode(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
           state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    _decode_run(mixed_qkv, conv_state, conv_weight, gate if gate.is_contiguous() else gate.contiguous(), beta, out_gate,
                A_log, dt_bias, norm_weight, ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps,
                lower_bound, out)
