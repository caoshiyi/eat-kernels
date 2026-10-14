# KDA for Kimi-K3 serving on AMD Instinct MI350X — handover for the serving engineer

This is Part A (sections 1-10) of the final handover written at the end of the optimisation run, plus the final-numbers
section B0 of the operator part. Content unchanged except as noted here.

- **Package** = `kda/` in this repository (`kda/kernel.py`, `kda/prefill/`, `kda/decode/`), byte-identical to the shipped
  snapshot `33_claim_fill_any_mode`. Snapshot labels such as `14_h0s_twx`, `27_kernarg_lines`, `31_flag_early_clock`,
  `32_health_sticky_claim_heal` and "package NN" name earlier package versions of the run; they are version labels, not files.
- Paths below are relative to `kda/` when they name package sources (`prefill/kernel.cu`, `decode/kernel_v15_a.py`, ...).
  The eval is `benchmark/eval_kda_mi350x.py` (originally `tools/eval_amd_kda_prod_mi350x.py`), the contract `CONTRACT.md`
  (originally `program.md`).
- **[internal]** replaces a reference to a test tool, log, patch or note of the development workspace (per-role work
  directories, the team board, the results table) that is not part of this repository. The measurements quoted next to
  such a marker were produced by it; the tools and logs can be provided on request.
- Role names (`release_closer`, the auditor, gatekeepers, builders) are agents of the team that built and tested the package.
- "HIP 0 / 2 / 3 / 6" are the four MI350X GPUs of the development machine (HIP numbering). Every number was measured on
  that machine (MI350X, gfx950, ROCm 7.2.0, torch 2.14.0+rocm7.2), mostly with other users' jobs on the host.

# Part A — for the serving engineer

## 1. What it is

One Python module, `kernel.py`, with the two KDA (Kimi Delta Attention) operators of a Kimi-K3 layer on vLLM's exact calling conventions:

- **`prefill`**: chunked prefill of a packed batch (one layer). Reads and writes vLLM's paged state pool in place (strided page views, per-sequence
  `has_initial_state`, prefix-cache checkpoints written into pool rows, strided `raw_beta`, output into a slice). Runs eagerly, never synchronises with the
  device. HIP C++ (`prefill/kernel.cu`, JIT-built at import).
- **`decode`**: fused decode / speculative verify (conv + recurrence + gated RMSNorm in one kernel launch). HIP-graph capturable; a captured graph is
  correct when replayed with new slot indices, accepted counts and data. HIP C++ with hand-written parts (`decode/*.cu`, four code objects JIT-built at
  the first decode call or by `decode_warmup()`), plus AITER's Triton kernel as a fallback for shapes outside the fast path.

Math: `reference.py` (`kda_prefill_reference`, `kda_decode_reference`). Head dim 128, fp32 state, bf16 activations. Contract: `CONTRACT.md`.

Speed against what vLLM ships today, official eval (`benchmark/eval_kda_mi350x.py`), ten runs of the final package (section 9):
**prefill 3.04x vLLM's fused ROCm prefill, decode 1.69x the best of AITER / vLLM HIP decode, combined 2.27 (`mean_vs_base` 2.267, 2.263–2.271 over ten runs; 2.261–2.275 over the four GPUs).**
What changed for a caller since 27 (iterations 9 and 10): one new entry point, `kernel.decode_health(heal=False)` (section 4.5: semantics, cost, tested recipe), which since 32 also reports and realigns claim-grid counters and since 33 works in every autograd mode; a defect closed that 27 had (section 6 D9) and one that 28–31 had (the health check could read clean behind wrong launches, section 4.5); the calling conventions are unchanged.

## 2. Minimal integration

```python
import kernel                                   # JIT-builds / loads the prefill extension (first time ~45 s, then cached under prefill/build/; takes a lock file there at EVERY load; initialises the HIP device)
kernel.decode_warmup()                          # once per device at start-up, outside any graph capture (first time ~105 s of JIT, then 0.15 s)

# per scheduler step, once, on HOST tensors; shared by all 69 KDA layers of the step
meta = kernel.prefill_prepare(cu_seqlens_cpu, num_heads, checkpoint_offsets_cpu, checkpoint_state_indices_cpu)   # pass device= on a multi-GPU process
# per layer, eager, no host sync inside
kernel.prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices, has_initial_state,
               cu_seqlens, out, checkpoint_offsets, checkpoint_state_indices, meta=meta)
# per layer, inside the engine's captured decode graph (or eagerly)
kernel.decode(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state, state_indices,
              num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)

# after every step that had prefill tokens, once the step's outputs are synchronised and BEFORE committing them (host memory read, no device sync):
h = kernel.prefill_health()
if not h["ok"]: ...                             # section 4.2: drop the step, recompute those requests, invalidate checkpoint rows written in it

# between steps, after anything abnormal (a cancelled / timed-out step, a device error, a graph that may have been replayed twice at once) and BEFORE the next decode replay;
# also periodically (not per step: 0.1 ms per live small-speculative graph). ALWAYS from the thread that captures / replays, never while a capture is open (rule 2):
d = kernel.decode_health(heal=True)             # section 4.5: synchronous; ok False = decode results since the last clean check are suspect
```

Rules a caller must respect (all are also torch / HIP rules for any custom kernel in a graph):
1. Call `decode_warmup()` (or make one eager decode call) on a device **before the first capture** on it, and call `decode_warmup()` again before capturing more than 16 graphs of one kind in a row:
   16 graphs that contain B × H ≤ 82 speculative calls, AND 16 graphs that contain claim grids (S 8: H 12 B 65–79 and ≥ 84, H 96 B 9 and ≥ 11). The 17th of the first kind refills its workspace at every replay
   (+4.1 µs = +22–25 %), the 17th of the second kind silently runs the plain grid (+1.6–3.4 %); `kernel._decode_mod.workspace_report()` shows the spares (section 4.1; [internal]).
   A server that captures one graph per batch size should call `decode_warmup()` between every 16 captures.
2. Do not replay one graph concurrently with itself (if it can have happened: `decode_health(heal=True)`, section 4.5). Do not destroy a graph while one of its replays is in flight.
   **Never call `prefill_prepare()` (or `prefill(meta=None)`, which calls it), `decode_health()`, `reset_workspace()`, `prefill_health(sync=True)` or `torch.cuda.synchronize()` from another thread while a graph
   capture can be in progress on the device: the capture is lost and the process can abort (section 6, D14). These calls belong to the thread that captures and replays, or behind ONE engine lock that also
   surrounds every capture.** `prefill_prepare()` is the per-step call of the prefill path: for an engine with a prefill thread and a decode thread it matters more than the health calls.
   `prefill(meta=...)`, `prefill_health()` without `sync` and `kernel._decode_mod.workspace_report()` are safe from any thread.
3. A captured graph bakes pointers, strides, the pool's slot count, B / S / H, the launch form and the co-tenant mode. Slot indices ≤ 0 or beyond the
   pool are padding (row skipped, nothing written). Re-capture after re-allocating a pool.
4. Slot 0 of the pool is reserved. Checkpoint rows are distinct and never alias `state_indices`.
5. One package per process (the decode modules are imported by name; two copies of the package in one process share a registry).
6. Set none of the debug environment variables of section 5.3.
7. Keep the package directory writable and at the path it was built at; import the package in the process that uses the GPU, not in a pre-fork parent (section 6, B3 / B4).
8. Do not let speculative decode graphs overlap KDA prefill on ANOTHER stream or process unless decode latency may wait for the prefill (section 6, D1): keep a step's prefill and decode on one stream, or use `lds_free` graphs.
9. Every tensor argument of `prefill()` must be on the device (only `prefill_prepare()` takes host tensors): a host `state_indices`, `has_initial_state`, `checkpoint_state_indices` or `A_log` is a device fault that
   takes the process's device context with it, not an exception (section 6, P7). Pass each `meta` only with the layout it was prepared for (P8).

## 3. Entry points

| function | what it does | notes |
|---|---|---|
| `prefill_prepare(cu_seqlens, num_heads, checkpoint_offsets=None, checkpoint_state_indices=None, device=None)` | schedule metadata from the batch layout, host only, one blocking H2D copy (**not from another thread while a graph capture is open: section 6, D14**) | median 0.07 ms (gate: 1 ms + 5 µs per sequence). Raises once if a relay fault of the previous step on this (device, current stream) is unreported. `device=None` = current device |
| `prefill(..., meta=None)` | one layer's chunked prefill on the current stream of `q.device` | **Every tensor argument must be on the device** (only `prefill_prepare` takes host tensors): a host `state_indices`, `has_initial_state`, `checkpoint_state_indices` or `A_log` is a device fault that takes the process's HIP context with it (section 6, P7). `meta=None` computes it inline. Non-contiguous q / k / v / raw_g are copied. H < 8 sequences with fewer than 8 (token, head) outputs take an exact fp32 torch side path (never for Kimi-K3's 12 / 96 heads) |
| `decode(...)` | one layer's decode (S = 1, `num_accepted_tokens=None`, `state_indices` [B]) or speculative verify (`state_indices` [B, S], `num_accepted_tokens` [B]) | one kernel launch per call (two for speculative batches of 257–312 (sequence, head) pairs). `beta`, `out_gate`, `mixed_qkv`, pools may be row-strided; `out` and `A_log` must be contiguous (`ValueError`); index tensors must be contiguous int32 on the device (CPU tensors: `TypeError`; int64: `ValueError`). Only the LENGTH of `cu_seqlens` is used (B = len − 1, S = T / B). B = 0: a plain call is a no-op, a speculative one raises `ValueError`. Out-of-range accepted counts and slot indices are memory-safe (77 + 77 cases, [internal]) |
| `decode_warmup(device=None)` | builds and uploads the four decode code objects, runs the one-time probes, makes spare workspaces | section 4.1 |
| `decode_cotenant_mode(mode="")` | `""`, `"symmetric"`, `"nosplit"`, `"lds_free"`; returns the previous mode | section 4.3; process-wide; applies to calls and captures made afterwards |
| `prefill_health(device=None, sync=False, acknowledge=False)` | relay watchdog status, never raises | section 4.2 |
| `decode_health(heal=False)` | state of the decode workspaces (leftovers of out-of-contract events, marks of consumed ones, claim counters off their rest value); synchronous, between steps, any autograd mode | section 4.5 |
| `prefill_range_guard(on=None)` | get / set the fp16 range guard of the relay path (default ON) | section 4.4 |

Operator tools on `kernel._decode_mod` (the module `decode/kernel_v16f_a.py`; **attributes set on `kernel` itself do nothing**):
`workspace_report()` (live / free / spare decode workspaces), `place_report()` (what the placement probe chose, also returned by `decode_warmup`),
`reset_workspace()` (refill every decode workspace unconditionally; `kernel.decode_health(heal=True)` refills only the dirty ones and says whether there were any; synchronises the current device),
`xcc_reprobe()` (repeat the round-robin probe after a driver / partition change; can only switch the per-XCC forms off).
On `kernel._pre` (the prefill extension): `set_relay_off(1)` (run-time form of `KDA_RELAY_OFF`).

## 4. Optional interfaces: what each one trades

### 4.1 `decode_warmup()` — default: not called
Without it, the same work happens inside the first eager decode call of the process (30 ms and a 512 MiB transient device allocation for the probes; on a cold
build cache four JIT builds, ~105 s), and a process whose FIRST decode call is inside a graph capture runs the slower symmetric launch forms until re-captured.
With it ([internal], 23 captures of every dispatch class behind one warm-up, no eager call at all): no build and no upload later, every
capture ≤ 10 ms, first replay 32–61 µs, capture-only graphs as fast as graphs captured after eager calls (0.98–1.02).
- **Spare workspaces.** A speculative call with B × H ≤ 82 (the "V-split" forms, e.g. H 12 with B ≤ 6) needs a 401 kB workspace (803 kB for 65–82 pairs) that must
  exist before the capture (HIP allows no eager work while a stream captures). Warm-up and every eager V-split call keep 16 spares per size class per device
  (19 MB). A capture that finds none still works but its graph refills its workspace at every replay: +4 µs, i.e. +20–27 % on such a call. One workspace serves
  all calls of one capture (a 69-layer model graph takes one), and returns when the graph is destroyed.
- **Claim-grid counters.** Speculative multi-round grids run as a "claim grid" from about four rounds on (S = 8 on page pools: B × H ≥ 1008, e.g. H 12 B ≥ 84, H 96 B ≥ 11; on dense pools from about 4.5 rounds: H 12 B ≥ 86), and also
  with a PARTIAL last generation behind at least three full ones (H 12 S 8: B 65–79; H 96: B 9; rule `decode/kernel_v15_a.py` `claim_grid()`, promotions 26 / 29 / 30): the launch carries 6 % extra
  blocks and the blocks of its last generation take their item from a per-capture counter word, so the faster XCCs take more of the remaining work (+2.5–5.5 % on grids of five rounds and more, +0.4–4.4 % on the 3.1–4.9-round grids).
  One capture can hold 63 different claim grids; further ones run the plain grid (correct, without the gain). The word lives in a 4 KiB workspace per
  capture; warm-up and eager claim-grid calls keep 16 spares per device. A capture that finds none runs the plain grid (correct, without the gain). Any content of the word is valid (a launch's claims are a
  permutation whatever it starts from); the co-tenant modes use the plain grid; `kernel._decode_mod.M.CLAIM = False` switches the form off.
  The rule's newest part, the TAIL term (claim grid for a partial last generation behind at least three full ones: H 12 S 8 B 65–79, H 96 B 9; promotion 30) has its own run-time switch:
  `kernel._decode_mod.M.CLM_TAIL = 0.0` before the captures it should apply to (process-wide; a captured graph keeps its form). Tested by [internal]: 7 tail grids claim -> plain -> claim,
  5 control grids unchanged, every graph correct ([internal]). The term gains +0.4–4.4 % on those batch sizes and nothing on the scored ones; see section 6 D12 for what is known about its variance.
- Not covered by warm-up: the Triton fallback (S > 8, strided index tensors, every call in mode `lds_free`) compiles per shape at first use; with
  `kernel._decode_mod.HOST_PROBES = False` the two main code objects are built but uploaded at their first launch; a second device needs its own call.
- `kernel._decode_mod.HOST_PROBES = False` before the first decode call: no probe, no 512 MiB transient, no blocking; costs the per-XCC forms and the code-object
  placement (phase-1 legacy shapes −0.5 %, plain B × H ≤ 16 calls −3 %).

### 4.2 `prefill_health()` — the relay watchdog (prefill)
The default prefill path ("relay") hands state between workgroups. Every wait is bounded: if a handoff does not arrive (never seen outside fault injection;
it needs a workgroup that is not dispatched for ~0.8 s), the launch **completes with invalid but finite results** after a 0.79–0.83 s stall (several broken links; a single broken link returns after 0.3 s) and records a fault.
`prefill_health()` returns `{"ok", "relay", "relay_state", "serial", "faults"}` from host memory (no device sync unless `sync=True`).
- `ok` False = a fault nobody was told about. Scope of a fault: every `prefill()` issued on that (device, stream) since the last clean synchronised check.
  Required reaction: `prefill_health(acknowledge=True)`, drop the step, recompute every request that had prefill tokens on that (device, stream), invalidate the
  checkpoint rows written in it. There is no per-request abort.
- If the engine does not check, the next `prefill_prepare()` (or non-relay `prefill()`) raises `RuntimeError` once — after one invalid step was consumed.
- After a fault the relay stays off on that device for the life of the process: the chunk-walk path (correct, no inter-workgroup waits, ~2.4x slower).
- In the 16000-step and the co-tenant / CU-mask soaks of section 7: 0 faults, relay on throughout.

### 4.3 `decode_cotenant_mode()` / `KDA_DECODE_COTENANT` — a GPU shared with other kernels
The fast decode forms use LDS (14–28 KiB per workgroup for speculative calls) and, for small calls, launch forms tuned per XCC. Costs on an idle GPU
([internal], re-measured on `27_kernarg_lines`, 2026-10-12, [internal]; every mode also gives up the claim grid: that is 1–3 % on the two
scored grids that take it and is included) and what each mode is for:

| mode | scored decode geomean | phase-1 shapes | use it when |
|---|---|---|---|
| `""` (default) | 1.000 | 1.000 | the engine owns the GPU, or shares it with ordinary kernels (GEMM, attention, copies, this package's own decode): default stays the fastest mode beside all of those (section 6, D2) |
| `"symmetric"` | 0.998 | 0.999 (plain B × H ≤ 16: 0.96–0.97; claim grids 0.97–0.99) | plain B = 1 calls beside a co-tenant that keeps one XCC busy: up to 40 % faster than default there (41.4 → 29.6 µs beside an attention loop) |
| `"nosplit"` | 0.957 | 0.967 (B = 1 speculative 0.76–0.88) | no inter-workgroup exchange at all for small speculative calls, if an operator wants none. Neither `nosplit` nor `symmetric` helps beside a KDA prefill (H 12 B 1 S 8 behind our H 96 prefill: 363–368 µs like default; `lds_free` 58; section 6 D1) |
| `"lds_free"` | 0.571 | 0.623 | AITER's kernel for every call (no LDS). **The mode for a GPU on which ANOTHER PROCESS runs KDA prefill concurrently** (this package's or vLLM's): speculative decode is then 2–5x faster than default (section 6, D1) |

The mode is baked into a graph at capture. **Selecting it automatically:** the package cannot (nothing runs on the host at a replay, and a workgroup that is not
dispatched cannot notice anything); a device-side telemetry word would cost every small call 0.2–0.3 µs. An engine can: keep a default and an `lds_free`
graph per batch size and choose per step from their measured latencies with non-blocking event queries ([internal], section 6 D3).
A generic "is LDS available" probe is NOT a usable signal ([internal]: it reads 121x its idle value beside an attention loop, where default is
still 1.8x faster than `lds_free`, and 10x beside vLLM's prefill, where `lds_free` is faster).

### 4.4 Prefill range guard and relay switch
- `prefill_range_guard()` / `KDA_RANGE_GUARD` — **default ON**. The relay's intermediates are fp16. With the guard on, a value slice whose chain overflowed
  (any |v| or state element above ~6.5e4) is recomputed in fp32 from its untouched initial state: correct, at 2.05 µs per token of the affected slices
  (latency cliff, section 6 P2). Guard off (`KDA_RANGE_GUARD=0`): ~1.0–1.2 % faster prefill, non-finite outputs and pool rows on such data.
- `KDA_RELAY_OFF=1` (latched at the first relay use of the process) / `kernel._pre.set_relay_off(1)`: every prefill takes the chunk-walk path: no
  inter-workgroup waits, no fp16 handoff, no latency cliff, no small-magnitude limitation (P1) — and ~2.4x slower prefill (still about 1.2x vLLM).

### 4.5 `decode_health()` — leftovers of out-of-contract events in the decode workspaces
Small speculative calls (B × H ≤ 82, the "V-split" forms) exchange partial sums between workgroups through a per-capture workspace. **Inside the contract every launch returns every word and flag of it to the sentinel**, also when
its workgroups leave / claim beside a co-tenant (57400 serving steps with the check on in iteration 9, undisturbed, beside co-tenants, under a CU mask and with forced leaves: 590 of 590 checks clean; iteration 10: section 7).
Something OUTSIDE the contract can leave words behind that the NEXT launches accept as their own — measured cause: one graph replayed concurrently with itself beside a co-tenant
([internal]); a killed launch is the other candidate.

**What the kernel does by itself (since `28_ghost_stamps`):** at most the first two launches of a (sequence, head) pair after the event are wrong, then it is clean, whatever the co-tenant does. Measured again in iteration 10:
over 72 runs (9 V-split shapes, co-tenant on XCCs 0-3 / 4-7, 3 or 4 overlapping replays, 30 clean replays beside the co-tenant each) 41 wrong replays of 2160, **none with an index above 1**, 0 wrong plain replays behind them
([internal]). (On 27 and before: every later replay beside a co-tenant, 30 of 30, until `reset_workspace()`.)

**What the kernel cannot do:** prevent those first one or two wrong launches. A block left over from the overlapping replays is indistinguishable from an early block of the next launch without a launch identity in every
exchanged word, and that costs the normal path of the most latency-sensitive kernels (argument: [internal] 1.4, re-checked in [internal] 1.1). The engine closes it:

```python
# RECIPE (tested: [internal] health-cost tool, part B, three V-split shapes beside a co-tenant: 0 wrong of 90 with it; 1, 1, 2 wrong of 30 without it)
# after ANYTHING abnormal — a cancelled / timed-out / failed step, a device error, a graph that may have been replayed while one of its replays was in flight —
# and BEFORE the next decode replay:
h = kernel.decode_health(heal=True)      # synchronous; refills what is dirty
if not h["ok"]:                          # leftovers were there: nothing consumed them yet IF this ran before the next replay
    ...                                  # results of the steps since the last clean check are suspect: recompute them
# periodically (e.g. once per second), if such events cannot be ruled out — from the thread that captures / replays, never while a capture is open ("Threading" below):
h = kernel.decode_health()               # ok False (stale_words / raised_flags / ghost_events > 0) = an event happened since the last clean check
```
- **Return value:** `{"ok", "workspaces", "dirty", "stale_words", "raised_flags", "ghost_events", "healed", "claim_misaligned", "claim_healed"}`. `stale_words` / `raised_flags` = leftovers that no launch has consumed yet.
  `ghost_events` = launches DID consume leftovers (their results were wrong): tombstones and the sticky marks they leave in the workspace, kept until a refill. `heal=True` refills the dirty workspaces (`healed`).
  `claim_misaligned` = claim-grid counter words (section 4.1) that an INCOMPLETE launch left off their rest value: results stay correct, but that grid is slower (+1.7..+4.3 % measured, +0.9..+6 % in the auditor's run) for the
  life of the graph; it does not change `ok`; `heal=True` realigns them (`claim_healed`). `kernel._decode_mod.reset_workspace()` refills everything unconditionally.
- **Threading: it synchronises the DEVICE.** HIP permits no device synchronise while any stream of the device is capturing: a `decode_health()` (or `reset_workspace()`, `prefill_health(sync=True)`,
  `torch.cuda.synchronize()`) from a second thread during another thread's `torch.cuda.graph(...)` raises in both threads, the capture is invalidated, and when the engine drops the broken graph torch aborts the process
  (section 6, D14). Call it from the capturing thread between steps, or behind the lock that guards captures.
- **Cost per call: 93 µs per live workspace, linear** ([internal], idle device, median of 40): 0.38 ms with one live V-split graph, 0.93 ms with 4, 3.2 ms with 16, 9.2 ms with 64;
  `heal=True` costs the same when nothing is dirty. One workspace serves all calls of one capture (a 69-layer model graph has one). For scale: one replay of the smallest graph is 24 µs. So: not at every step.
  It synchronises the device: call it between steps, with no decode call of the process in flight.
- **What a clean periodic check does and does not prove.** Up to `31_flag_early_clock` the tombstone could be overwritten by the kernel's own healing (measured: an event on a 2-way-split graph, two wrong replays, then
  `ok True, ghost_events 0`). The sticky mark of `32_health_sticky_claim_heal` closes that for every leftover that still had its leave flag raised when it was consumed ([internal]: 8 of 8 and 9 of 9 runs with a wrong replay reported afterwards, masks 0x0f / 0xf0; [internal]: 3 of 3). A leftover whose
  flag was already gone is consumed without any mark (not observed in 17 of 17 runs with a wrong replay, but possible by the code): **`ok True` at a periodic check is not a proof that no launch was wrong; only the check right after the event, before the next replay, is.**
- **Autograd modes.** Workspaces are torch tensors made lazily in the mode of the calling thread; made inside `torch.inference_mode()` (a model forward) they are inference tensors. Since `33_claim_fill_any_mode` every
  maintenance call works in every mode (the auditor's [internal] in three orderings: 0 raises; [internal]). On 26–32 `reset_workspace()` raised
  `RuntimeError: Inplace update to inference tensor outside InferenceMode` when called outside inference_mode after claim-grid calls inside it, `decode_warmup()` warned, an eager claim-grid call could raise after a claim
  graph had died, and on 32 `decode_health(heal=True)` raised when it had a claim counter to realign.
- **It reports live workspaces only** (eager per (device, stream), captured, retired). The workspace of a destroyed graph goes to a free list (refilled before reuse) and leaves the report: check BEFORE dropping graphs.
- The kernel recognises a foreign record by a hash of the launch's INPUT rows and slot indices: a later launch with byte-identical inputs and indices does not see the difference.

## 5. Environment variables

### 5.1 Operator switches
| variable | default | effect |
|---|---|---|
| `KDA_DECODE_COTENANT` | unset | `symmetric` / `nosplit` / `lds_free` at import (section 4.3); an unknown value warns and is ignored |
| `KDA_RANGE_GUARD` | on | `0` turns the prefill range guard off (section 4.4) |
| `KDA_RELAY_OFF` | unset | `1`: prefill chunk-walk path for the whole process (section 4.4) |
| `KDA_PROD_MODE` | `3` | prefill schedule: 3 relay; 1 chunk walk with per-sequence groups; 2 chunk walk with uniform groups (vLLM's rule). Read at import. Leave at 3 |

### 5.2 Set by the build
`PYTORCH_ROCM_ARCH` is rewritten to `gfx950` in `os.environ` during each decode JIT build and restored afterwards. `TRITON_CACHE_DIR` (Triton's own) decides
where the fallback kernel's cache lives.

### 5.3 Debug / experiment variables — a serving environment must set NONE of these
- Prefill planner overrides read by plain `getenv` on every call, with no guard: `KDA_RELAY`, `KDA_RELAY_G`, `KDA_VSPLIT`, `KDA_ORD`, `KDA_R32`, `KDA_TW`,
  `KDA_VCK`, `KDA_FDO`, `KDA_TWM`, `KDA_UN`, `KDA_IC`, `KDA_FT` (`prefill/kernel.cu`). `KDA_FDO` / `KDA_FT` are read at prepare AND at launch: changing one in
  between desynchronises tables and launch. `KDA_R32` / `KDA_TW` select relay variants that were never routed in production. `KDA_RX_TINY` is read but inert.
- Decode: `KDA_EXPERIMENT=1` arms `KDA_CG` (extra `-D` for fault-injection builds: `MB_BOUND=1`, `MB_FI=..`), `KDA_V15_BSS`, `KDA_V15_KALIGN`. Without
  `KDA_EXPERIMENT=1` they are ignored.
- Not environment variables (compile-time, source edits): `KDA_RX_FULL` (small-magnitude watch, P1), `KDA_WD_*`, `MB_BOUND`, `MB_EPOCH`, `SLOT_RANGE`.

## 6. Known limitations, with trigger and measured numbers

**Prefill**
- **P1. Small magnitudes are inaccurate in relative terms (never non-finite, not detected).** Trigger: the whole state of a head below ~1e-5 in magnitude
  (`v` ≥ 1e5 times smaller than ordinary for a whole sequence) or beta below ~5e-5 from a cold start (a dead head). Relative pool-row error then exceeds the
  2e-2 tolerance (`v` × 1e-5: 3.6e-2; × 1e-6: 0.31; beta 6e-6: 5.5e-2) while the absolute error stays ≤ 2.8e-7 per state element and ≤ 8e-9 on any output
  (vLLM: 2.4e-3 relative). Covered only by `KDA_RELAY_OFF=1` or a source build with `#define KDA_RX_FULL 1` (measured −5.5 % prefill on an earlier snapshot).
- **P2. Recomputation latency cliff on out-of-range data.** Trigger: any element above ~6.5e4 (fp16). 1 × 1600 tokens, H 12: 37 µs in range → 3330 µs with one
  such `v` element (vLLM 85 µs); 1 × 7936 tokens: 16.5 ms (vLLM 0.2 ms); if all 69 layers of a model are hit: 0.23 s per 1600-token chunk, 1.1 s per 8K chunk.
- **P3. A relay fault is a failed step, not a repair** (section 4.2): 0.8 s stall, invalid finite results, relay off on the device until restart.
- **P4. 64 relay flag regions per process over all devices** (one per distinct (device, stream) that ran a relay prefill; 8 MB each, never freed). The 65th
  falls back to the chunk walk (~2.4x slower on that stream). Not reachable on one device through `torch.cuda.Stream()` (torch hands out at most 32 pooled streams per device: 90 live "new" streams all ran
  the relay, [internal]); it needs external streams or three devices in one process.
- **P5.** No prefill warm-up entry: the first `prefill` on a device uploads its code object and runs a stripe probe (once).
- **P6.** Sequences with ≥ 65536 planes take the chunk walk by layout.
- **P7. A HOST tensor handed to `prefill()` is a device fault, not an exception** (iteration 10, auditor's finding A5; [internal]). `prefill/main.cpp`
  `run_prod()` checks dtype, shape and contiguity of every tensor but the device of the meta only. `state_indices`, `has_initial_state`, `checkpoint_state_indices` or `A_log` on the CPU:
  `AcceleratorError: CUDA error: an illegal memory access was encountered`, and every later device call of the process fails the same way (all requests of that engine). A host `checkpoint_offsets` alone or `cu_seqlens`
  is accepted (taken from the prepared meta); wrong dtypes raise a clean `RuntimeError`. The slip is natural because `prefill_prepare()` REQUIRES host tensors for the same-named arguments. Not fixed: prefill is frozen
  (the fix is four `is_cuda()` terms in `run_prod`; first thing to do if the prefill phase is reopened, section B5). Until then the caller must guarantee it (an `assert t.is_cuda` in the engine's wrapper costs nothing).
  The decode entry point has this check (`TypeError`). Reproduced on the final package ([internal]): fault for `state_indices`, `has_initial_state`,
  `checkpoint_state_indices` (alone or with the offsets) and `A_log`; a good call afterwards fails the same way. **No operator switch avoids it:** with `KDA_RELAY_OFF=1` the same five fault and a host
  `checkpoint_offsets` alone faults too ([internal]: the chunk walk reads the offsets from the tensor).
- **P8. Outside the checkpoint contract nothing is reported** (auditor, [internal]): a checkpoint offset that is not a multiple of 64, or beyond the sequence length, silently writes NO
  checkpoint row (the reference rejects such layouts); a `meta` prepared for ANOTHER layout with the same H, N and T is accepted and gives wrong results (no byte outside the batch's rows is touched; a meta of another
  size raises). Pass the meta of the step it was prepared for.

**Decode**
- **D1. Speculative decode in flight together with KDA prefill — another process OR another stream of the same process.** Prefill workgroups hold 136–163 KiB of a CU's 160 KiB LDS; a speculative decode
  workgroup needs 14–28 KiB and is not dispatched until a prefill workgroup leaves: the decode call WAITS for the prefill. (AITER's kernel uses no LDS and runs beside it.)
  - *Two processes* ([internal], 2026-10-12; hot replays, p50 µs, default / `lds_free`): beside this package's H 96 prefill 12,1,8: 279 / 56; 12,16,8: 294 / 86; 96,8,8: 440 / 183;
    beside vLLM's prefill 93 / 71, 328 / 103, 578 / 236. Plain decode (12,64,1) is not affected (45 / 74).
  - *One process, two streams* (corrected in iteration 10; the ship report had this as "rare outliers"; [internal], 150 launches per cell, the decode graph replayed
    on stream D right behind a `prefill()` on stream P): **every speculative replay that overlaps a prefill ends after that prefill ends** (95–100 % of the launches). p50 µs, default / `lds_free`:

    | decode graph | alone | beside our p12_loaded16k_1 (123 µs) | beside our p96_loaded8k_1 (377 µs) | beside our 1 × 32K H 12 (271 µs) | beside vLLM's p96 prefill (1130 µs) |
    |---|---|---|---|---|---|
    | H 12 B 1 S 8 | 15.5 / 27.8 | 93.7 / 48.6 | 363.8 / 56.2 | 202.5 / 52.3 | 804.5 / 62.4 |
    | H 12 B 4 S 8 | 18.0 / 33.5 | 35.7 / 56.3 | 40.9 / 66.5 | 38.8 / 54.6 | 48.6 / 75.9 |
    | H 12 B 16 S 8 | 29.3 / 48.3 | 104.1 / 74.9 | 379.9 / 78.7 | 219.9 / 78.0 | 830.5 / 108.6 |
    | H 12 B 64 S 8 | 90.7 / 136.1 | 173.8 / 172.1 | 453.3 / 180.3 | 333.3 / 180.1 | 1031 / 213 |
    | H 12 B 128 S 8 | 174.2 / 275.0 | 256.7 / 307.0 | 533.4 / 314.0 | 412.8 / 315.9 | 1224 / 351 |
    | H 96 B 8 S 8 | 90.5 / 133.9 | 174.2 / 173.7 | 459.9 / 180.6 | 333.9 / 180.5 | 1028 / 212 |
    | H 12 B 64 S 1 (plain) | 29.4 / 33.4 | 44.5 / 64.5 | 48.3 / 71.0 | 46.0 / 71.9 | 58.2 / 95.7 |

    Only H 12 B 4 S 8 and plain decode are unaffected. The other side of the same table: the PAIR finishes earlier in default mode, because beside AITER's decode the prefill is the one that slows down
    (p12_loaded16k_1 123 -> 392 µs beside AITER B 128; p96 377 -> 654): time until both are done, default / `lds_free`: 257 / 392 µs (B 128 + p12 16k), 533 / 654 (B 128 + p96), 376 / 383 (B 1 + p96).
    **Default mode = best throughput, worst decode latency; `lds_free` = decode first, prefill pays.** Correctness in the same runs: shared-pool rounds (prefill on P and a spec graph on D in flight together on disjoint
    rows of one page pool, 4 launch orders, 9 layout pairs × 8 rounds) all pass, `prefill_health` and `decode_health` clean. Re-run on the final package (33; [internal]):
    H 12 B 1 S 8 alone 15.3 / 27.5, behind our p96 prefill 360 / 55, behind vLLM's 806 / 63; H 12 B 4 S 8 unaffected (40.6 / 65.4); shared-pool rounds all pass.
  - *What to do:* within ONE engine whose steps are serial (prefill part, then decode part, on one stream) none of this occurs. An engine that issues decode graphs concurrently with chunked prefill (a second model or
    engine in the process, an async scheduler, a second replica on the GPU) needs `lds_free` graphs for latency-critical speculative decode (or the selector of D3), or must keep prefill and decode on ONE stream.
- **D2. Other co-tenants** (same tool, default / `lds_free`, 12,1,8 and 96,8,8): none 15.6 / 27.4 and 93 / 141; attention loop 44 / 79 and 200 / 212; 1 GiB
  copies 25 / 57 and 153 / 226; this package's decode 31 / 55 and 156 / 189; this package's H 12 prefill 47 / 50 and 147 / 171; 8192² GEMM 21 / 42 and
  117 / 1120. Default is the right mode beside all of these.
- **D3. Mode selection is manual** (section 4.3). Engine-side selector, measured: mean µs per hot replay, default / `lds_free` / selector at 12,1,8 ([internal]): idle 17.1 / 26.9 / 16.8; attention 34 / 69 / 34; copies 24 / 55 / 24; our decode 29 / 75 / 30; our H 12 prefill 83 / 49 / 49; our H 96 prefill 338 / 54 / 59; vLLM's prefill 150 / 64 / 74. It is a prototype: beside a GEMM that stalls everything in ~1 ms bursts its averages flap (use medians and hysteresis), and at 96,8,8 it needs more than 600 replays to settle (1.25–1.43 x the best mode in that window).
- **D4. Co-tenants that leave fewer than 4 free CUs per XCC**: a platform rule — an XCC starts only as many workgroups of a dispatch as it has free CUs and
  does not reuse them until the co-tenant leaves; every kernel waits (AITER too), forms with more workgroups (small plain calls on few XCCs, V-split) earlier.
  Never a hang: every device-side wait is bounded. This is the stress suite's intermittent `cotenant` WARN / FAIL line (section 7).
- **D5.** The leave / claim safety machinery of the V-split forms costs small speculative calls about 0.5–1.5 % (phase-1 B = 1 shapes). Since `31_flag_early_clock` the leave flags are read EARLY (behind a block's own "alive"
  stores, not behind its completed poll: +1.0–1.5 % on those calls). Iteration 8 had rejected the first form of that because it was correct only while no dispatched workgroup stalls (an injected 0.3 s stall left output rows
  unwritten). The shipped form has no such assumption: a block that is older than 1 ms at the end of its poll re-reads the flags there, and a block that would leave while its missing siblings are known to be alive waits for
  them at least 8 ms; both compare the device's real-time clock (`s_memrealtime`), which is assumed to be ONE clock for all eight XCCs. The 0.3 s and 3 s stall injections, a stall between the alive stores and the flag load, and
  entry-late splits pass (0 failures; [internal]).
- **D6.** S > 8, strided `state_indices` / `num_accepted_tokens`, plain calls with S ≠ 1 and mode `lds_free` run AITER's Triton kernel (baseline speed, per-shape
  compile at first use).
- **D7.** Eager calls keep one 401 kB workspace per (device, stream handle) for the life of the process (handles recycle). A failed round-robin probe is sticky
  for the process. A process-wide lock serialises the host part of `decode()` across threads (microseconds).
- **D8.** Multi-round speculative grids on the page layout pay 3–6 % against dense pools (a property of the MI350X memory system; one pool position in device
  memory can be faster by as much); latency of B = 1 calls depends on the GPU (2 % between two of ours) and on host load (+0.1–0.3 µs).
- **D9. After a graph was replayed concurrently with itself** (outside the contract; beside a co-tenant) later CLEAN replays of small speculative calls can be wrong: at most the first two per (sequence, head) pair, then the
  kernel is clean by itself; `kernel.decode_health(heal=True)` between steps, before the next replay, prevents even those (section 4.5: recipe, cost, and what a clean periodic check does not prove). (On 27 and before: every later replay beside a co-tenant, 30 of 30, until `reset_workspace()`.)
  After such an event one launch can also take 8 ms longer once (a block waiting for a sibling that only LOOKS alive).
- **D11. Large speculative grids beside co-tenant PROCESSES** ([internal], 2026-10-12; p50 µs, ours / best baseline, 12,128,8 and 96,32,8): none 182 / 281 and 339 / 624; 1 GiB copies 310 / 418 and
  539 / 913; this package's decode 335 / 359 and 628 / 694; **attention loop 390 / 351 and 762 / 680** (0.90 of AITER); **this package's H 12 prefill 382 / 325 and 717 / 659; H 96 prefill 1326 / 330 and 2637 / 671; vLLM's prefill
  1176 / 375 and 2451 / 722** (D1's LDS wait on large grids: 3.5–4x). The claim grid (6 % more workgroups) is not more exposed than the plain grid of the same package: 0.97–1.10 of it beside every one of these, p99 alike.
  Mode `lds_free` (= AITER) is the faster choice for multi-round grids beside an attention-heavy or prefill-heavy co-tenant process.
- **D10. Several engines on several streams of one process beside a kernel that occupies 3/4 or more of the CUs:** only one of the concurrently issued KDA kernels proceeds, the others finish when that co-tenant kernel
  ends (measured 400 ms = its duration; single calls finish in 0.1–0.4 ms). vLLM's prefill and AITER's decode behave the same: platform queueing.

- **D12. Latency spread on the batch sizes between the scored ones (H 12 B 65–79, H 96 B 9, ...: the claim grid's tail term, promotion 30).** Iteration 9 left "sporadic lows" open: a three-package A/B read some of these
  claim graphs 2–5 % slow now and then. Measured in iteration 10 by the auditor ([internal], team board): **no slow mode of the package.** Fixed-slot graphs, 6 fresh processes × 3
  pools + 2 hot: per-(process, pool) medians of the claim grid fill a 1.7–3.1 % range with no second cluster (12,65,8 97.2–99.5 µs; 12,70,8 102.6–105.0; 12,79,8 113.9–116.7; 96,9,8 101.5–103.2; 48,19,8 107.4–110.7); the plain
  grid's range on the same pools is wider (2.7–5.5 %); on the same pool the claim grid is never more than 0.2 % behind the plain grid (plain / claim 0.998–1.053). What a harness can read as a "low": the pool's position in
  device memory (2–3 %), the first 5–25 flushed replays of every new graph (4–10 % slow, claim or plain), and stretches of 10–25 replays at +4–8 % that hit the claim and the plain graph of one pool at the same instant.
  Serving-shaped soak, single engine, hot, rows moving every step: 5–13 % (H 12) and 9–20 % (H 96 B 9) of the replays are 5–10 % above the median, with the plain-grid control (`M.CLAIM = False`, same seed) showing the same
  shoulder; paired medians claim vs plain: 1.0 / 3.1 / 2.3 / 2.9 % faster at B 65 / 70 / 79 / H 96 B 9, p99 equal. The tail term stays on; its run-time switch is in section 4.1. Deterministic slower cases that do exist:
  a 17th claim graph captured behind one warm-up runs the plain grid (+1.6–3.4 %; call `decode_warmup()` again, section 4.1), and D13.
- **D13. A claim-grid counter word left off its rest value by an incomplete launch** (device error, killed launch; not reachable inside the contract: unchanged after replays on other streams, with padding rows, and after
  200 self-concurrent replay pairs, [internal]) keeps that grid 1–6 % slower for the life of the graph; results stay correct. `decode_health()` reports it as `claim_misaligned` and
  `heal=True` realigns it (since `32_health_sticky_claim_heal`; before: only `reset_workspace()` did) ([internal]).

- **D14. A device synchronise from another thread during a graph capture kills the capture, and then the process** (auditor's finding A7; [internal]; same code on 31–33). Thread A captures / replays / drops decode graphs; thread B polls `kernel.decode_health()`: within seconds B gets `CUDA error: operation not permitted when stream is capturing`,
  A gets `operation failed due to a previous error during capture`, and when A drops the invalidated graph: `terminate called after throwing an instance of 'c10::AcceleratorError' ... from ~CUDAGraph`, SIGABRT.
  It is the platform's rule (any `torch.cuda.synchronize()` or blocking host-to-device copy of the engine does the same); the package makes it easy to break by offering a periodic health call. Affected entry points:
  `decode_health()`, `kernel._decode_mod.reset_workspace()`, `prefill_health(sync=True)`, **and `prefill_prepare()`** (its blocking H2D copy; so also `prefill(meta=None)`): a thread that calls ONLY `prefill_prepare()`
  kills another thread's capture immediately ([internal]). Safe from any thread: `prefill(meta=...)` (273 K calls beside captures, 0 errors), `prefill_health()` (no sync),
  `workspace_report()` (4.0 M polls beside 1015 capture cycles, 0 errors); control with an idle second thread: 1460 capture cycles clean. Estimated exposure
  of a careless poller: about 1e-3 per capture at one poll per 10 s, ~20 % for a 128-graph start-up under a running poller. **Not guarded in the package** (it cannot see another thread's capture; a guard that
  refuses while one of ITS OWN decode calls is being captured would cover part of the window; not built: found 70 minutes before the end of the last session). The rule is rule 2 of section 2.
  Reproduced on the final package (33) in the first seconds of a 20 s run ([internal]). Feasibility of a guard, measured
  ([internal]): a second thread CAN ask HIP whether another thread's stream is capturing (`hipStreamGetCaptureInfo_v2` on its handle: active in
  200 of 200 queries during the capture, inactive after it) without disturbing that capture (graph correct). A guard built on it would turn most collisions into a clean `RuntimeError` in the maintenance thread; it
  could not close the race between its check and a capture that begins a microsecond later, nor see a capture on a stream the package has never seen, so the caller rule stays in any case.
  Two things that do NOT help, both measured on 33: (1) a prototype of that guard ([internal], never promoted): under the auditor's reproducer the collision still
  happened after 5 capture cycles instead of 0 ([internal]: the capture began between the check and the synchronise); (2) capturing with
  `torch.cuda.graph(..., capture_error_mode="thread_local")`: on this HIP runtime a `decode_health()` from another thread fails and invalidates the capture exactly as in the default mode
  ([internal]: 20 of 20 polls raise, the capture is lost, in both modes). **Only mutual exclusion in the engine works: one lock around every
  capture and every health / reset / synchronise call, or all of them on one thread.**
  By code reading the same three entry points and `workspace_report()` also touch the capture registries without the lock that `decode()` holds for its host part (D7's lock is not a promise that the
  package is thread-safe as a whole); a race there was not reproduced (the 4.0 M polls above). The same rule covers it: maintenance calls from the thread that issues the decode calls.

**Both**
- **B1. MI350X only** (8 XCCs, 256 CUs, compile-time constants; hand-written asm audited on ROCm 7.2.0; another toolchain warns at build and the decode
  placement probe turns itself off).
- **B2. One GPU per process is the tested configuration** (section 8).
- **B3. The package directory must be writable at every start, and the toolchain present if the path changes** ([internal], finding A4). torch's JIT takes a lock file in `prefill/build/<digest>/` at every
  load, also when nothing is rebuilt: on a read-only copy `import kernel` fails with `PermissionError: ... prefill/build/<digest>/lock`. A copy of the package at ANOTHER path rebuilds even if its build directories were
  copied (the ninja files hold absolute paths: 34 s for prefill, ~105 s for the four decode objects). For a container image: build at image-build time at the final path, keep `prefill/build` and `decode/build` writable
  for the serving user, ship `hipcc` / `ninja` if the path can change.
- **B4. `import kernel` initialises the HIP device** (torch's JIT loader queries the visible cards unless `PYTORCH_ROCM_ARCH` is set). A parent that imports the package and forks: every worker fails with torch's
  `Cannot re-initialize CUDA in forked subprocess`. A pre-fork server must import the package in the workers, or set `PYTORCH_ROCM_ARCH=gfx950` in the environment (then the import leaves the device alone and forked
  workers run prefill + decode correctly: [internal]).
- **B5. Tail latency of the smallest plain calls.** On plain decode with B ≤ 4 our p99 sits AT vLLM's p99 (p99 / p50 1.25–1.30 against vLLM's 1.17–1.20): the package wins the median by 7 %, not the tail. Host load moves our
  p99 a little more than vLLM's (ours / vLLM p99: 0.91–1.00 quiet, 0.98–1.05 beside 32 compile processes; 0 FAIL of the suite's `tail_latency` test (line 1.10) in 19 runs of the audit; the one logged FAIL of iteration 9 had a
  much noisier host).

## 7. What was tested, and how to rerun it

The tests below ran with the team's internal tools ([internal]) on the development machine, every GPU command under
its GPU scheduler with a timeout. What this repository contains for re-running them: the official eval and its gates,
the smoke test and the workload tables (`benchmark/README.md`); the stress suites, soaks, probes and fault-injection
builds named below are not included.

**State at the end of iteration 10.** The table below is the ship report's (results up to package 31, with the commands). What iteration 10 ran, on which package (33 = 32 + three host lines, every kernel symbol
identical; 32 = 31 + the sticky mark in the leave / claim blocks + the claim counters in the health check):

| what | package | result | log |
|---|---|---|---|
| officials | 33 | section 9 | [internal] |
| every gate (prefill gates included) over new seeds: a full eval per `--seed` | 33 | seeds 691–730: **0 of 40 with a gate failure**; prefill geomean 3.033–3.054, decode 1.674–1.703, phase-1 decode 1.559–1.602 | [internal] |
| decode gates over new seeds ([internal]) | 32, 33 | 631–660 on 32, 661–690 on 33: 0 failing of 60 (seeds 1–330, 371–690 used by now; 691–730 by the full sweep above; next free: 731) | [internal] |
| decode stress suite, now 45 tests (+ `health_recipe`, `claim_heal`: the two defects of 28–31 that iteration 10 fixed; both FAIL by design on 31) | 33 | **41 PASS / 4 WARN / 0 FAIL** (`cotenant`, `starved_1cu`, `beyond_contract`, `tail_latency`); the auditor's independent run on 33: 42 PASS / 3 WARN / 0 FAIL; 32: 40 / 3 / 0 of 43 + the two new PASS | [internal] |
| serving soak with `decode_health()` at every check, five engines | 32, 33 | undisturbed 7500 steps and beside the 3/4 co-tenant 7500 steps on each, + 3200 steps on an `MB_BOUND=1` build of 32 beside the co-tenant: 0 failures in 33200 steps, health clean at 340 of 340 checks, relay on | [internal] |
| fault builds of the small-speculative leave / claim protocol | 32 (kernels of 33 identical) | forced leaves 17.3 K + 412.6 K + 30.1 K: 0 failures; poisoned flags: correct results, the tool's 33 sentinel lines as on 31; [internal] 0 wrong; 0.3 s stall of either split (three builds): 0 failures, 5904 claims each; entry-late split 0: 13488 leaves = 13488 claims; `MB_BOUND=1` suite subset 14 PASS / 1 WARN | [internal] |
| out-of-contract event, then 30 clean replays beside a co-tenant | 32 kernels | `ghost_probe` masks 0x0f / 0xf0 / 0x55: PASS × 3; `bound_probe` 0x0f / 0xf0: 36 runs, 26 wrong replays of 1080, largest index 1, 17 of 17 runs with a wrong replay reported by `decode_health()` afterwards | [internal] |
| `decode_health()` recipe; claim counters; autograd modes | 33 | [internal] RECIPE PASS 6 / 6; [internal] PASS; [internal] PASS (22 steps), the auditor's [internal] 0 raises in three orderings | [internal] |
| alternation against the previous package (officials 5 pairs, legacy 8 rounds, A/B 4 processes) | 32 vs 31, 33 vs 32 | officials 2.2702 → 2.2670 and 2.2596 → 2.2626; legacy 0.9999 (A/A 0.9978) and 0.9993 (A/A 0.9998); A/B scored 0.9995 (A/A 0.9985) and 1.0015 (A/A 0.9999): within noise both times | [internal] |
| lifecycle probes (warm-up coverage, exit with live graphs, 300 live captures, claim probe) | 32, 33 | PASS | [internal] |
| **independent audit** ([internal]: re-run matrix, fresh-eyes probes, gate-failure risk) | 31, re-runs on 33 | on 31, identical to the last known results: prefill stress suite 30 PASS / 3 WARN / 1 FAIL (P1, as on 14), prefill gate sweep 100 new seeds 0 failures + 690 seeds of the thinnest contract cases 0 failures, relay hazards, watchdog recovery 14 / 14, `slow_split_soak` 645 runs, `test_guard_layouts` 274 cases, `test_prepare_latency` 0 of 278 over, decode suite 39 / 4 / 0, forced leaves, soaks; new: cross-phase on one shared pool in 32 switch combinations (all correct; latency = D1), serving soak at tail batch sizes, co-tenant gate worker 24 / 24 (prefill finishes 4–16 ms after launch, limit 1500), decode edges 77 + 77 + 37 cases, prefill empty / zero-length batches 65 cases, threads 60 s, exit with launches in flight, fork. **On 33:** decode suite 42 / 3 / 0 of 45, [internal] 0 raises, [internal], `ghost_probe`, soak 700 ticks, soak variants `KDA_RELAY_OFF=1` and `lds_free`, `test_output_order` 90 / 90, `test_range_repair` 90 / 120 (the 30 = P1), three officials 2.266 / 2.274 / 2.269, spawned workers, 200 pool create / capture / drop rounds with flat memory, cold cache of both phases (33 s + 104 s, then first calls 0.1–9.4 ms), prefill shapes H 1–160 and one 100 K-token sequence (56 cases), 5.8 GiB pools with B to 4096 (14 cases), value ranges (52 cases): no FAIL and no unexplained WARN. **The auditor's verdict ([internal] team board): fit to hand to a serving team under the conditions of rules 1–9 of section 2** | [internal] |

| what | command | time | result as of the ship report of iteration 9 (packages up to 31; "final package" in this column = 31) |
|---|---|---|---|
| official eval: 13 + 10 timed workloads, all hard gates | `HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py` | ~1 min | section 9 |
| decode gates over seeds | [internal] | 8 min per 30 | seeds 1–330 and 371–630 in the gatekeepers' and builders' sweeps of the four decode iterations: 0 failures (511–630 on 28–31; next free: 631) |
| **end-to-end serving soak**: 4 engines (H 12 S 8, H 96 S 8, H 12 S 1, H 16 S 6) in one process, one page pool and two streams each; prefill steps (cold, prefix hit from a checkpoint row, resumed chunks, next-turn from the accepted slot, checkpoints) and decode steps (graphs of several B replayed on alternating streams, eager calls) launched back to back; long-memory data; every step against the reference on a shadow pool that is never resynchronised; untouched bytes, `prefill_health`, memory, latency every 20 ticks; graphs re-captured | [internal] | 10 min | **Final package (31) and each of 28 / 29 / 30 as candidates: 1500 ticks = 7500 steps with five engines (the four above + H 12 graphs of B 22 / 64 / 128), 0 failures, relay on, RSS flat, `decode_health()` clean at 75 of 75 checks between steps (the soak of [internal] checks it).** 27: 10000 steps, 0 failures. Earlier snapshots: 24, 4000 ticks = 16000 steps (5931 prefill steps with 12857 sequence chunks, 10069 decode steps, 104 re-captures): **0 failures**; relay on, 0 health events; worst state error 8.7e-4, output 5.2e-3, no growth; host RSS flat; workspaces constant; latency last third / first third 0.93–1.07 per step kind. ~93 % of the prefill steps take the relay. Also clean (2000–6000 steps each, [internal]): `KDA_RELAY_OFF=1`, modes `symmetric` / `nosplit` / `lds_free`, `KDA_RANGE_GUARD=0`, and a fault-injection build in which every V-split block attempts to leave (`MB_BOUND=1`) |
| the same beside the eval's co-tenant kernel and under CU masks | [internal] | 15 min | **Final package (31; [internal]) and 28 ([internal]), each: 3/4 co-tenant 4000 steps, 15/16 co-tenant 2000, CU mask of 128 CUs 3000, forced leaves in every V-split launch (`MB_BOUND=1` build) beside the co-tenant 3200: 0 failures, `decode_health()` clean at 130 of 130 checks.** 27: 0 failures in all five (20400 steps), as on 25: 3/4-CU co-tenant every 10th tick; 15/16-CU co-tenant; process CU masks of 128 CUs, 3 of every 8 CUs, and an odd tail mask + co-tenant. Note: with two or more engines' kernels in flight beside the co-tenant the tick finishes when the co-tenant's kernel ends (400 ms in the test); with one engine every step finishes in 0.1–0.4 ms. vLLM's prefill + AITER's decode behave identically in the same test ([internal]): platform queueing, section 6 D4 |
| soak self-check (it can fail) | [internal] (state / byte injection) | 15 s | both detected |
| decode stress suite, 43 tests (iteration 9 added `ghost_chain`: the event of section 4.5 and 20 replays beside the co-tenant after it, and `claim_redteam`: claim graphs in flight together, two claim calls in one capture, other streams, threads, 24 live graphs, half-full co-tenants): allocation tails, every dispatch boundary, 400 replays per class, 22 live graphs, 69-layer model graph, streams / threads / processes, killed neighbours, decode beside prefill, co-tenant shapes, CU masks, extreme shapes, padding rows, value ranges, long memory, workspace lifetime, NTA layouts, exit with live graphs, capture-only graphs, the claim grid's counter (poisoned words, shared eager stream, co-tenants, padding rows) | [internal] (`--soak`: 26 min) | 13 min | 43 tests. 28 and 29: 40 PASS / 3 WARN / 0 FAIL; 30: 39 / 4 / 0 (`proc_modes` WARN on a loaded machine); **31 (final): 39 / 3 / 1, the FAIL being `tail_latency` on a plain B = 1 shape whose kernel did not change (p99 20.0 vs vLLM 17.8 µs) while JIT builds ran on the host; `tail_latency` and `proc_modes` PASS on the rerun without them.** 27: 38 / 2 / 1 of 41 (the intermittent `cotenant` line explained below this table) |
| forced leaves of the V-split protocol (fault-injection build on a COPY of the package) | [internal] (= [internal] + the stall reproducers) | 20 min | on 28: 0 wrong results in 16.2 K + 39.6 K + 27.5 K forced leaves (39.6 K + 27.5 K claims, each through the launch-identity check), 15-test suite subset 14 PASS / 1 WARN, [internal] 0 wrong, a 0.3 s stall of either split 5904 leaves = 5904 claims |
| out-of-contract event, then clean replays beside a co-tenant (section 4.5) | [internal] (`--mode heal`: with `decode_health(heal=True)` after the event) | 4 min | 27: 30 of 30 wrong on two of five shapes, unbounded. 28: 1 / 0 / 0 / 2 / 1 of 30, none after the second; with heal 0 of 250 |
| claim grid as an attacker; large grids beside co-tenant processes | [internal] | 2 min; 8 min | PASS (20 checks); section 6 D11 |
| after an out-of-contract event: the largest index of a wrong clean replay, whether `decode_health()` reports every run that had one, event counters of the leave / claim blocks (hook build) | [internal] (`--counters` on a copy built with `KDA_EXPERIMENT=1 KDA_CG="MB_FI=50"` and the hook of [internal]) | 7 min | section 4.5 |
| `decode_health()`: cost per call for 1 / 4 / 16 / 64 live graphs, and the engine recipe (event, check, heal, 30 clean replays beside a co-tenant; counter-example without the heal) | [internal] | 3 min | section 4.5 |
| run-time switch of the claim grid's tail term | [internal] | 1 min | PASS |
| what changed in the binary of the V-split kernels between two packages | [internal] (both built) | 10 s | — |
| warm-up coverage / capture-only graphs | [internal] | 1 min | PASS (0.98–1.02) |
| exit with live graphs; 300 live captures | [internal] | 1 min | exit 0; PASS |
| cold build cache, first-call latency | [internal] (it works on cold copies) | 5 min | PASS on 26: four builds at start-up (105 s), worst first call afterwards 8.9 ms |
| co-tenant processes: modes, probe, selector | [internal] | 5 min each | sections 4.3, 6 |
| phase-1 (dense-pool) decode metric, alternated on one GPU | [internal] | 13 s per reading | section 9 |
| fresh-process A/B of two decode packages on 44 shapes | [internal] | 10 min | — |
| prefill stress suite, 34 tests (relay hazards, watchdog, co-tenants, CU masks, range, layouts) | [internal] | 8 min | 30 PASS / 3 WARN / 1 FAIL = `numerics_range` small side (P1), as on 14 (prefill frozen since) |
| prefill split path on data with memory | [internal] | 5 min | 0 failures on 14 |
| prefill watchdog fault injection, range repair, output ordering | [internal] (each [internal] under [internal]) | 2–28 min | 14 / 14; 90 / 120 (30 = small side, P1); 90 / 90 (on 14) |

The stress suite's `cotenant` test prints WARN (occasionally FAIL) for one line: small V-split or plain per-XCC shapes (e.g. H 12, B 2, S 2) beside a co-tenant
that holds 15/16 of the CUs with all their LDS. It is limitation D4: whether the 16 free CUs are spread 2 per XCC or bunched decides which kernels fit, per
launch; the suite retries three times and compares with the baseline beside the same co-tenant (FAIL = ours waited 3 of 3 and the baseline 0 of 3; seen twice
in eleven suite runs of the last two iterations, WARN in the others, the baseline waits 2.5 s in the same test on other launches). The other standing WARNs: `starved_1cu` (the kernel's
own LDS need when one CU per XCC is left, D1's mechanism), `beyond_contract` (int64 indices raise `ValueError`), `tail_latency` (p99 of one plain shape within
0.2 µs of vLLM's).

## 8. Never tested
- **Two GPUs in one process** ([internal] exposes one GPU per command; no agent was allowed to give a process two). Status at the end of iteration 10: still not run; the package did not change in a way that touches it
  (section B0), so the code-level reading of iterations 8 / 9 stands ([internal] section 4, [internal] section 4; re-read against the final sources by `release_closer`,
  [internal] section 4). **Until the checklist below has passed on real hardware: one process per GPU (vLLM's worker model).**
  - *Prefill, by code:* flag regions are keyed (device, stream); stripe-probe verdict and relay-off-after-fault are per device; `prefill_prepare(..., device=)` builds per-device tables and a meta of another device is
    refused: **a two-GPU caller must pass `device=`**. Process-wide: `KDA_RELAY_OFF`, the range-guard switch, the table of 64 flag regions, and ONE host-mapped status block (probe results and fault words) allocated under
    whichever device runs the first relay launch. That block is the one thing the code cannot settle: if it is not visible from the second device, the relay silently switches off there (correct, ~2.4x slower) or the first
    relay launch on it faults.
  - *Decode, by code:* every registry is keyed by device (eager workspaces (device, stream); captured workspaces (device, stream, HIP capture id); spare V-split workspaces, claim-grid counter workspaces and their free lists;
    probes and placement); the callback cells are host memory keyed by the capture key (which contains the device); launch-identity stamps and tombstones live inside each workspace; `decode_health()` walks the workspaces of
    ALL devices and synchronises each device that owns one; `reset_workspace()` synchronises the current device only; the cost model's CU count is set per call from the call's device under the process-wide launcher lock.
    Each device needs its own `decode_warmup(device)`.
  - *Known platform blocker (phase 1, measured on v77, never re-run):* on ROCm 7.2.0 a GRAPH that holds any custom kernel faults on a second device of a process once that kernel was captured on the first (AITER's kernel
    too); eager calls on several devices worked. If it still holds, decode graphs need one process per GPU whatever this package does.
  - *First run on multi-GPU hardware, in this order:* (1) [internal] (alternating / threaded / in-flight-together prefill on two devices, health on both, foreign meta refused);
    (2) `kernel.prefill_health(device=1)["relay_state"] == "on"` after the first prefill on device 1 (else the status block is not shared); (3) a forced watchdog fault on device 1
    ([internal]): the fault word arrives, blame names device 1, device 0 keeps its relay; (4) `kernel.decode_warmup(1)` and the stress suite's `grid,values,replay_many` with the
    pools on device 1, eager first; (5) one decode graph on device 0, then one on device 1 (the platform fault above; if it faults: one process per GPU); (6) [internal] with engines
    spread over the devices (it has no `--devices` option: every engine is created on the current device); (7) exit with live graphs on both devices ([internal]): status 0; (8) `kernel.decode_health()` after
    decode calls on both devices: `workspaces` = the sum over both, `ok`; (9) [internal] under `torch.cuda.device(1)`;
    (10) [internal] A / E with the pools on device 1 (`kernel._decode_mod.M._clm_spare[1]` must shrink, not `[0]`).
- Any GPU other than MI350X, any ROCm other than 7.2.0, a driver whose dispatch is not striped over 8 XCCs or changes while the process runs.
- A real vLLM server end to end (the soak drives the package with a serving-shaped schedule, not vLLM itself); a real model's activations (numerics gates
  use synthetic trained-layer statistics); runs longer than ~10 minutes / 16000 steps.
- A `KDA_RX_FULL 1` build on the final sources; the prefill debug variables of section 5.3; `kernel._decode_mod.V.SHARE_PER_CAPTURE = False`.
- More than 64 distinct (device, stream) pairs with relay prefill beyond the fallback being taken (not reachable on one device through torch's stream pool: P4).
- `torch.autocast` contexts around the calls; S > 8 beyond two fallback shapes; maintenance calls from a second thread beyond what D14 reports; a package-side guard for D14.
- The health check's miss rate on 31 as a sweep (it rests on one reproducer run and the mechanism; on 32 / 33: 17 of 17 runs with a wrong replay reported).

## 9. Official numbers and their spread
<!-- S9 -->
Final package (`33_claim_fill_any_mode`), `benchmark/eval_kda_mi350x.py` with no options, **ten runs one at a time on a quiet machine** (HIP 0, 2026-10-13 05:05-05:14), every gate passes in each
([internal]; tables by [internal]):

| | mean | min | max | sd | runs |
|---|---|---|---|---|---|
| `mean_vs_base` | **2.267** | 2.263 | 2.271 | 0.003 | 2.271 2.263 2.271 2.267 2.269 2.264 2.268 2.264 2.264 2.268 |
| prefill geomean vs vLLM's fused ROCm prefill (13 workloads) | **3.038** | 3.035 | 3.043 | 0.003 | 3.036 3.038 3.036 3.035 3.040 3.042 3.043 3.035 3.041 3.038 |
| decode geomean vs the best of AITER / vLLM HIP decode (10 workloads) | **1.691** | 1.684 | 1.699 | 0.005 | 1.699 1.685 1.699 1.693 1.693 1.684 1.691 1.690 1.685 1.694 |
| phase-1 decode shapes (gate floor 1.500) | **1.590** | 1.585 | 1.604 | 0.006 | 1.589 1.585 1.600 1.604 1.588 1.587 1.586 1.588 1.586 1.586 |
| phase-1 prefill shapes (gate floor 0.932) | **2.808** | 2.791 | 2.837 | 0.014 | 2.803 2.824 2.809 2.837 2.791 2.801 2.812 2.814 2.795 2.791 |

Per workload, mean of the ten runs (baseline = vLLM's fused ROCm prefill for the p rows, the faster of AITER / vLLM HIP decode for the others):

| workload | baseline us | ours us | ratio | [min, max] |
|---|---|---|---|---|
| p12_loaded8k_0 | 243.0 | 80.22 | 3.027 | [3.020, 3.032] |
| p12_loaded8k_1 | 248.8 | 74.76 | 3.325 | [3.297, 3.369] |
| p12_loaded8k_2 | 202.7 | 80.84 | 2.509 | [2.496, 2.519] |
| p12_loaded16k_0 | 397.3 | 130.36 | 3.048 | [3.036, 3.063] |
| p12_loaded16k_1 | 463.9 | 113.17 | 4.098 | [4.087, 4.115] |
| p12_loaded16k_2 | 625.6 | 124.26 | 5.035 | [5.001, 5.052] |
| p12_light8k_0 | 215.3 | 81.30 | 2.651 | [2.633, 2.675] |
| p12_light8k_1 | 82.0 | 34.08 | 2.401 | [2.381, 2.431] |
| p12_light16k_0 | 83.4 | 36.74 | 2.271 | [2.249, 2.285] |
| p12_light16k_1 | 83.9 | 37.94 | 2.214 | [2.207, 2.229] |
| p96_loaded8k_0 | 1232.0 | 339.17 | 3.624 | [3.613, 3.646] |
| p96_loaded8k_1 | 1137.8 | 352.25 | 3.230 | [3.201, 3.253] |
| p96_light8k_0 | 369.7 | 117.13 | 3.163 | [3.146, 3.176] |
| s8_h12_b1 | 32.2 | 15.41 | 2.087 | [2.064, 2.111] |
| s8_h12_b4 | 36.8 | 18.32 | 2.010 | [1.975, 2.031] |
| s8_h12_b16 | 57.4 | 31.80 | 1.804 | [1.764, 1.828] |
| s8_h12_b32 | 89.3 | 54.48 | 1.641 | [1.603, 1.680] |
| s8_h12_b64 | 149.6 | 94.29 | 1.585 | [1.568, 1.618] |
| s8_h12_b128 | 280.1 | 176.01 | 1.591 | [1.568, 1.603] |
| d_h12_b64 | 30.9 | 28.58 | 1.081 | [1.058, 1.099] |
| s8_h96_b1 | 44.4 | 23.00 | 1.929 | [1.913, 1.941] |
| s8_h96_b8 | 146.1 | 91.96 | 1.588 | [1.560, 1.611] |
| s8_h96_b32 | 616.4 | 333.78 | 1.847 | [1.831, 1.871] |

Legacy decode per shape: d_k3_b1 1.228, d_k3_b32 1.124, d_k3_b128 1.203, s4_k3_b1 1.723, s4_k3_b16 1.804, s4_k3_b64 1.585, s8_k3_b1 2.092, s8_k3_b16 1.877, s8_k3_b64 1.627, d_glm16_b1 1.413, d_glm16_b64 1.514, d_glm16_b256 1.378, s6_glm16_b1 1.865, s6_glm16_b16 1.738, s6_glm16_b64 1.610, d_glm8_b128 1.529, s6_glm8_b32 1.717 (ratios to the recorded phase-1 baselines).

**Expected official score: 2.268 (fifteen officials on HIP 0: 2.263-2.274; the ten below: 2.267), run to run ±0.005 on one GPU (sd 0.003; ten runs 2.263-2.271), GPU to GPU 2.261-2.275.** Spread:
- **Run to run on one GPU:** `mean_vs_base` 2.263-2.271; decode geomean 1.684-1.699 (±0.5 %: the baselines are timed live, and their timing is half of it:
  in alternated officials OUR latencies per workload repeat within ±0.5 %), prefill 3.035-3.043 (±0.15 %); single decode workloads ±1-2 % (the 1.5-round grid S 8 H 12 B 32: ±3 %).
- **Five more officials after everything else had ended** (05:45-05:49, HIP 0, none of the team's jobs on the machine; the host's load average was still 12-14 from other users, as during the ten runs;
  [internal]): 2.272 2.274 2.265 2.271 2.265 (mean 2.269), decode 1.686-1.698, phase-1 decode 1.592-1.603, every gate passing. **Fifteen officials on HIP 0 in all: mean 2.268, 2.263-2.274.**
- **GPU to GPU** (four officials started together on the four GPUs, twice, 05:14-05:16; all gates pass in all eight; [internal]): HIP 0 2.269 / 2.268 (decode 1.694 / 1.694, phase-1 decode 1.594 / 1.589); HIP 2 2.267 / 2.275 (decode 1.689 / 1.695, phase-1 decode 1.594 / 1.586); HIP 3 2.271 / 2.269 (decode 1.697 / 1.694, phase-1 decode 1.598 / 1.598); HIP 6 2.261 / 2.261 (decode 1.678 / 1.679, phase-1 decode 1.559 / 1.561).
  HIP 6 is the slow GPU for B = 1 latencies (it has read ~2 % lower on the phase-1 decode number for every package of the last four iterations).
- **All 18 officials of the final package in this section:** `mean_vs_base` 2.261-2.275, phase-1 decode 1.559-1.604; plus the two promotion evals (2.271 on 32, 2.268 on 33) and 20 alternated officials
  of 32 / 33 during the protocol (2.256-2.276): **every gate passing in every run.**
- **Host load:** other jobs on the machine cost B = 1 calls 0.1-0.3 µs. Measured by the audit ([internal]): the phase-1 decode number beside 32 endless compile processes -0.4 %, beside 96 (host load 40-63) -1.3 %;
  phase-1 prefill: no effect.
- **Margin of the phase-1 decode gate (floor 1.500):** lowest reading of the final package 1.559 (HIP 6, 3.9 % above the floor); the metric loses 0.063 per µs added to every shape, so the floor is +0.9 µs on every
  small call on HIP 6 and +1.4 µs on the other GPUs; 96 compile processes cost ~0.3 µs. A failure needs the slow GPU AND an oversubscribed host (about three times busier than 96 compile processes), or a GPU 4 % slower than HIP 6.
- **Every gate over 40 new seeds** (a full eval per `--seed`, 691-730: 13 + 10 workloads with refill / re-replay, prefill contract and host gates, fp64 numerics, decode held-out shapes, integration, co-tenant, both phase-1 floors; [internal]): **0 runs with a gate failure or a FAIL line**; prefill geomean 3.033-3.054, decode 1.674-1.703, mean 2.258-2.276, phase-1 prefill 2.77-2.84, phase-1 decode 1.559-1.602 (three runs at a time on three GPUs beside the legacy alternation: a loaded machine). Prefill's behaviour did not drift with the shared `kernel.py`: its gates and its geomean are those of 14.
- **Phase-1 decode metric of the decode snapshot chain, alternated on one GPU** ([internal], HIP 0, 8 rounds, fresh process per reading, order rotated; copies of the snapshots in [internal]): final package (33) **1.601**, 27_kernarg_lines **1.593**, 24_nta_margin **1.586**, 19_slot_range **1.591**, 14_h0s_twx **1.603**. Relative to the final package: 27 0.9950, 24 0.9901, 19 0.9934, 14 1.0010. Small speculative shapes, same order: s8_k3_b1 2.115 / 2.072 / 2.065 / 2.083 / 2.136; s4_k3_b1 1.721 / 1.699 / 1.685 / 1.706 / 1.737; s6_glm16_b1 1.891 / 1.868 / 1.852 / 1.868 / 1.908; plain d_k3_b1 1.233 / 1.229 / 1.209 / 1.215 / 1.225; large grid s8_k3_b64 1.631 / 1.632 / 1.629 / 1.631 / 1.634. Reading: 14 -> 19 paid for the bounded poll (the leave / claim safety of the small speculative calls), 19 -> 24 for the epoch tag, 24 -> 27 collected the kernel-argument fixes, 27 -> 33 the early flag load: **the final package is back within 0.1 % of 14 on this metric while carrying every safety mechanism 14 did not have** (bounded waits, epoch tags, launch stamps, health check). The dense-pool metric does not see the NTA / claim-grid gains of the page layout (those are in the scored decode geomean: 1.65 on 14, 1.69 now).
- Against the previous packages by alternation (one GPU, order rotated or paired): 31 -> 32 officials 2.2702 -> 2.2670 (our latencies per workload 0.995-1.005), legacy 0.9999 (A/A 0.9978); 32 -> 33 officials 2.2596 -> 2.2626,
  legacy 0.9993 (A/A 0.9998), every kernel symbol identical; 27 -> 31 (iteration 9): decode +0.27 %, legacy +0.24 %. Six officials of 31 three hours earlier read 2.269 (2.268-2.270): the same package speed within noise.
- Comparing two packages: alternate them on one GPU ([internal], which rotates the order; [internal] do NOT rotate, and the last package of a round reads ~1 % slow on the
  1.5-round grid); consecutive officials on different GPUs or hours apart prove nothing below 1-2 %.
<!-- /S9 -->

## 10. Deployment checklist
1. One engine process per GPU. `import kernel`; `kernel.decode_warmup()`; then capture graphs (≤ 16 small-speculative graphs per warm-up).
2. GPU exclusively the engine's, or shared with ordinary kernels: default mode. KDA prefill in flight together with speculative decode — from another process on the GPU, or from another stream / engine of the SAME
   process (section 6, D1: every overlapping speculative replay waits for the prefill): `lds_free` graphs for latency-critical decode (or both graphs + the selector), or one stream for both.
3. After each step with prefill tokens: `prefill_health()`; handle `ok == False` as a failed step. Assert in the engine that every tensor passed to `prefill()` is on the device (section 6, P7: a host index tensor kills the process's device context).
4. On shutdown nothing is required (graphs may be alive at exit).
5. After any device error, aborted or cancelled launch, or a graph that may have been replayed concurrently with itself, BEFORE the next decode replay: `kernel.decode_health(heal=True)` (it also realigns claim-grid
   counters; `kernel._decode_mod.reset_workspace()` refills everything unconditionally) and `kernel.prefill_health(sync=True, acknowledge=True)`. Call `kernel.decode_health()` periodically between steps (not per step:
   93 µs per live workspace) if such events cannot be ruled out, and treat `ok == False` as "decode results since the last clean check are suspect". A clean periodic check is not a proof (section 4.5).
   **All of these calls, and `prefill_prepare()`, from the thread that captures and replays (or behind one engine lock that also surrounds every capture), never while a capture is open on the device (rule 2, D14).**
7. Container images: build at the final path, keep `prefill/build` and `decode/build` writable (section 6, B3); import in the worker process (B4).
6. If activations can exceed ~6.5e4 or whole heads can be numerically dead (< 1e-5), read P1 / P2 before choosing `KDA_RELAY_OFF`.

---
# Part B (excerpt) — final numbers

## B0. Final numbers
<!-- B0 -->
**Package: `33_claim_fill_any_mode`** = the task dir (`diff -r` clean; prefill and the prefill part of `kernel.py` identical to `14_h0s_twx`). Iteration 10 changed no speed: two robustness promotions
(32: the decode health check could read clean behind wrong launches, and did not see claim counters; 33: maintenance calls legal in every autograd mode), both within noise by alternation (section 9, last bullets).

| | ten officials, one at a time, HIP 0, quiet (05:05-05:14) | all 18 officials incl. the per-GPU spread |
|---|---|---|
| `mean_vs_base` | **2.267** [2.263, 2.271], sd 0.003 | 2.261-2.275 |
| prefill geomean | 3.038 [3.035, 3.043] | 3.035-3.055 |
| decode geomean | 1.691 [1.684, 1.699] | 1.678-1.699 |
| phase-1 decode shapes (floor 1.500) | 1.590 [1.585, 1.604] | 1.559-1.604 |
| phase-1 prefill shapes (floor 0.932) | 2.81 [2.79, 2.84] | 2.78-2.84 |
| gate failures | 0 of 10 | 0 of 18 |

- **Expected official score: 2.268; run to run on one GPU 2.263-2.274 (fifteen officials on HIP 0: the ten of the table, mean 2.267, sd 0.003, and five more at 05:45-05:49, mean 2.269); GPU to GPU 2.261 (HIP 6) to 2.275 (HIP 2).** Per GPU, two officials each: HIP 0 2.269 / 2.268 (decode 1.694 / 1.694, phase-1 decode 1.594 / 1.589); HIP 2 2.267 / 2.275 (decode 1.689 / 1.695, phase-1 decode 1.594 / 1.586); HIP 3 2.271 / 2.269 (decode 1.697 / 1.694, phase-1 decode 1.598 / 1.598); HIP 6 2.261 / 2.261 (decode 1.678 / 1.679, phase-1 decode 1.559 / 1.561).
- **Every gate over 40 new seeds** (a full eval per `--seed`, 691-730: 13 + 10 workloads with refill / re-replay, prefill contract and host gates, fp64 numerics, decode held-out shapes, integration, co-tenant, both phase-1 floors; [internal]): **0 runs with a gate failure or a FAIL line**; prefill geomean 3.033-3.054, decode 1.674-1.703, mean 2.258-2.276, phase-1 prefill 2.77-2.84, phase-1 decode 1.559-1.602 (three runs at a time on three GPUs beside the legacy alternation: a loaded machine). Prefill's behaviour did not drift with the shared `kernel.py`: its gates and its geomean are those of 14.
- **Phase-1 decode metric of the decode snapshot chain, alternated on one GPU** ([internal], HIP 0, 8 rounds, fresh process per reading, order rotated; copies of the snapshots in [internal]): final package (33) **1.601**, 27_kernarg_lines **1.593**, 24_nta_margin **1.586**, 19_slot_range **1.591**, 14_h0s_twx **1.603**. Relative to the final package: 27 0.9950, 24 0.9901, 19 0.9934, 14 1.0010. Small speculative shapes, same order: s8_k3_b1 2.115 / 2.072 / 2.065 / 2.083 / 2.136; s4_k3_b1 1.721 / 1.699 / 1.685 / 1.706 / 1.737; s6_glm16_b1 1.891 / 1.868 / 1.852 / 1.868 / 1.908; plain d_k3_b1 1.233 / 1.229 / 1.209 / 1.215 / 1.225; large grid s8_k3_b64 1.631 / 1.632 / 1.629 / 1.631 / 1.634. Reading: 14 -> 19 paid for the bounded poll (the leave / claim safety of the small speculative calls), 19 -> 24 for the epoch tag, 24 -> 27 collected the kernel-argument fixes, 27 -> 33 the early flag load: **the final package is back within 0.1 % of 14 on this metric while carrying every safety mechanism 14 did not have** (bounded waits, epoch tags, launch stamps, health check). The dense-pool metric does not see the NTA / claim-grid gains of the page layout (those are in the scored decode geomean: 1.65 on 14, 1.69 now).
- **Gate margins** (worst measured; the audit's sweeps on 31 = the same prefill and the same decode fast path, [internal] section 4, plus this section's runs):

  | gate | worst margin measured | evidence |
  |---|---|---|
  | workload correctness, prefill | out 4.3e-3 against 2e-2 (4.7x), pool rows 3.9e-4 (51x) | 100-seed sweep, all officials |
  | workload correctness, decode | out 4.6e-4 (43x), state 2e-7 | all officials, 680 decode-gate seeds (1-330, 371-690), 0 failures |
  | prefill contract | `heads_h8` 8.5e-3 against 2e-2 (2.4x; p99 6.8e-3), `tiny_h12` 7.6e-3 (2.6x) | 790 seeds of these cases, 0 failures: the thinnest correctness margin of the eval |
  | prefill host | `prefill_prepare` median 0.07-0.09 ms against 1.03-1.06 (11-14x); no device sync (deterministic) | 100 seeds; 278 layouts |
  | numerics vs fp64 | `rescaled` 5.3e-3 (3.8x) | 300 seeds |
  | decode contract (17 shapes x layouts) | out 5.8e-4 (34x) | officials, 680 seeds |
  | integration | prefill out 4.3e-3 (4.7x) | 100 seeds |
  | co-tenant | prefill finishes 4-16 ms after launch beside the hog, decode 0 ms; limit 1500 ms (90x) | 8 seeds run directly (the official log never prints the prefill line of this gate: the eval strips the worker's first line; its verdict counts) |
  | phase-1 prefill floor 0.932 | 2.78 (3.0x) | every official |
  | **phase-1 decode floor 1.500** | **1.559 on HIP 6 (3.9 %)**; 1.585-1.604 on the others (5.7-7 %) | every official; the narrowest gate of the package: section 9 for what moves it |

  The audit's estimate for a single official of this package returning 0: below 1 % on a quiet or normally shared host on any of the four GPUs; the realistic way to a 0 is environmental (an oversubscribed host on the slow
  GPU, a slower fifth GPU, a cold build cache against the co-tenant worker's timeout), not the package.
- What the score does not show (Part A, section 6): the decode numbers are for a GPU the engine owns; speculative decode that overlaps a KDA prefill on another stream or process waits for it (D1); the phase-1 prefill
  limitations P1 / P2 / P7 are data- or caller-dependent and outside the gates.
<!-- /B0 -->
