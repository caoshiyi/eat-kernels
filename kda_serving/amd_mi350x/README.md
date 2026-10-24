# KDA kernels for Kimi-K3 serving on AMD Instinct MI350X

Two operators for the KDA (Kimi Delta Attention) layers of Kimi-K3, written for MI350X (gfx950) and called with vLLM's
exact calling conventions for its ROCm Kimi-K3 KDA layer (drop-in for that layer's prefill and decode calls):

- **`prefill`**: chunked prefill of a packed batch, one layer. Reads and updates vLLM's paged state pool in place
  (strided page views, per-sequence `has_initial_state`), writes prefix-cache checkpoint rows into the pool, reads the
  strided `raw_beta` slice of the input projection, writes its output into a slice of the layer's output buffer. Runs
  eagerly, never synchronises with the device. HIP C++ (`kda/prefill/`), JIT-built at import.
- **`decode`**: fused decode / speculative verify (short conv + delta-rule recurrence + gated RMSNorm in one launch), plain
  decode (S = 1) and DSpark verify (S up to 8). HIP-graph capturable; a captured graph stays correct when replayed
  with new slot indices, accepted counts and data. HIP C++ (`kda/decode/*.cu`, JIT-built at first use), with AITER's
  Triton kernel as the fallback for shapes outside the fast path.

Head dim 128, bf16 activations, fp32 state. Math: `reference.py`. Calling convention, shapes, layouts and acceptance
gates: `CONTRACT.md`. Integration details, every optional interface, limitations with measurements: `docs/HANDOVER.md`.

## Numbers

`benchmark/eval_kda_mi350x.py` (official run: 13 chunked-prefill steps from AgentX Kimi-K3 coding-session traces, 10
decode / speculative-verify steps; baselines timed live on the same inputs; all hard gates must pass, else the score is 0):

| | final package, 10 officials on one GPU | 18 officials over 4 GPUs |
|---|---|---|
| prefill, geomean vs vLLM's fused ROCm Kimi-K3 prefill | **3.04x** (3.035–3.043) | 3.035–3.055 |
| decode, geomean vs the faster of AITER `fused_kda_decode` / vLLM's HIP decode | **1.69x** (1.684–1.699) | 1.678–1.699 |
| `mean_vs_base` = sqrt(prefill × decode) | **2.267** (2.263–2.271) | 2.261–2.275 |

Gates over 40 new seeds (a full eval each): 0 failures. On real activations (Kimi-Linear-48B on GSM8K / MATH-500, H = 12 / 64 / 96,
single long requests and 32-request batches, prefill with checkpoints and a carried state, plain and speculative decode)
both operators pass a correctness gate against fp64: 27 / 27 cases, prefill output within 0.17–0.28 % relative L2
(FLA `chunk_kda` on the same inputs: 0.23–0.35 %) (`benchmark/real_gate/`). Per-workload ratios range from 2.2x to 5.0x (prefill) and from
1.07x (plain decode, H 12 B 64) to 2.1x (speculative B = 1) (`docs/HANDOVER.md`, section 9). A release check from a
fresh clone of this repository read 2.224–2.259 (prefill 3.02–3.03, decode 1.64–1.69) on a day when the live AITER
baseline ran faster on some workloads; the package's own latencies matched the handover's (`benchmark/README.md`).

## Requirements

AMD Instinct MI350X (gfx950; 8 XCCs / 256 CUs are compile-time assumptions), ROCm 7.2.0 (`hipcc`, `ninja`), Python 3.12,
torch 2.14.0+rocm7.2, triton-rocm 3.8.0 (`requirements.txt`). Other GPUs and toolchains are untested; the decode build
warns on a ROCm other than 7.2.0. One GPU per process is the tested configuration.

## Integration

```python
import sys; sys.path.insert(0, "<repo>/kda")
import kernel                                   # JIT-builds / loads the prefill extension (first time ~35-45 s, then cached under kda/prefill/build/)
kernel.decode_warmup()                          # once per device at start-up, outside any graph capture (first time ~105 s of JIT, then 0.15 s)

# per scheduler step, once, on HOST tensors; shared by all 69 KDA layers of the step
meta = kernel.prefill_prepare(cu_seqlens_cpu, num_heads, checkpoint_offsets_cpu, checkpoint_state_indices_cpu)   # pass device= on a multi-GPU process
# per layer, eager, no host sync inside
kernel.prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices, has_initial_state,
               cu_seqlens, out, checkpoint_offsets, checkpoint_state_indices, meta=meta)
# per layer, inside the engine's captured decode graph (or eagerly)
kernel.decode(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state, state_indices,
              num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)

# after every step that had prefill tokens, once its outputs are synchronised and BEFORE committing them (host read, no device sync):
h = kernel.prefill_health()
if not h["ok"]: ...                             # drop the step, recompute those requests, invalidate checkpoint rows written in it

# between steps, after anything abnormal (cancelled / timed-out step, device error, a graph possibly replayed twice at once)
# and BEFORE the next decode replay; from the thread that captures / replays, never while a capture is open:
d = kernel.decode_health(heal=True)             # synchronous; ok False = decode results since the last clean check are suspect
```

Key caller rules (full list with reasons: `docs/HANDOVER.md`, section 2):
1. Call `decode_warmup()` before the first graph capture on a device, and again before capturing more than 16
   small-speculative or claim-grid graphs in a row.
2. Every tensor passed to `prefill()` must be on the device; only `prefill_prepare()` takes host tensors.
3. `prefill_prepare()`, `prefill(meta=None)`, `decode_health()`, `reset_workspace()`, `prefill_health(sync=True)` synchronise
   the device: call them from the thread that captures and replays (or under one engine lock that also covers every
   capture), never while another thread has a capture open.
4. Do not replay a graph concurrently with itself. A captured graph bakes pointers, strides, pool size and B / S / H:
   re-capture after re-allocating a pool. Slot 0 is reserved; checkpoint rows never alias `state_indices`.
5. One copy of the package per process; keep `kda/` writable at the path it was built at (JIT lock files and build
   caches live there); import it in the process that uses the GPU, not in a pre-fork parent.
6. Keep a step's prefill and speculative decode on one stream (limitation 1 below), and set none of the debug
   environment variables (`docs/HANDOVER.md`, section 5.3).

## Known limitations (short list; all of them with numbers in `docs/HANDOVER.md`, section 6)

1. **Speculative decode behind KDA prefill on another stream or process waits for the prefill** (D1). Prefill
   workgroups hold most of each CU's LDS; a speculative decode workgroup is not dispatched until they leave. Measured:
   H 12 B 1 S 8 15 µs alone, 364 µs beside our H 96 prefill. One stream for prefill and decode (the normal serial
   engine step) avoids it; otherwise capture latency-critical decode in `decode_cotenant_mode("lds_free")` (AITER path,
   no LDS, about 0.57x of default speed when alone).
2. **A host tensor passed to `prefill()` is a device fault, not an exception** (P7): `state_indices`,
   `has_initial_state`, `checkpoint_state_indices` or `A_log` on the CPU raises an illegal memory access and the
   process's device context is lost. The engine wrapper must assert `t.is_cuda`. Decode checks this (`TypeError`).
3. **Phase-1 decode floor margin**: the no-regression gate on the phase-1 decode shapes (floor 1.500) is the narrowest
   gate: lowest reading 1.559 on the slowest of four GPUs (about 4 % margin). A heavily oversubscribed host on a slow GPU
   is the realistic way to fail it.
4. Prefill relay path: fp16 intermediates. Values above ~6.5e4 are recomputed in fp32 (correct, but a latency cliff);
   whole heads below ~1e-5 lose relative accuracy (P1 / P2). `KDA_RELAY_OFF=1` avoids both at about 2.4x slower prefill.
5. MI350X / ROCm 7.2.0 only; one GPU per process tested (a two-GPU process is untested, `docs/HANDOVER.md` section 8).

## Validate

`benchmark/README.md`: setup (`bash benchmark/build_baselines.sh`), the full official eval
(`HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py`, about 1 min warm), the quick correctness test
(`benchmark/smoke_test.py`), single workloads, expected output and expected numbers with their spread. Workload
definitions: `workloads/README.md`.

## Repository layout

| path | what |
|---|---|
| `kda/` | the package as shipped: `kernel.py` (entry points), `prefill/` (HIP C++), `decode/` (HIP C++ + Python launchers, AITER Triton fallback; `decode/FINAL.md` = the decode kernel's design notes) |
| `reference.py` | math reference (`kda_prefill_reference`, `kda_decode_reference`) |
| `CONTRACT.md` | calling convention, shapes, dtypes, page layout, workloads, hard gates |
| `benchmark/` | the eval, smoke test, vendored vLLM / AITER baselines, vLLM's reference layer; `real_gate/`: correctness on real Kimi-Linear activations |
| `workloads/` | workload tables as data, the prefill batch generator and its trace-derived lengths, recorded phase-1 baselines |
| `docs/HANDOVER.md` | serving-engineer guide: interfaces, environment variables, limitations, what was tested, official numbers |

## Provenance

The kernels were optimised and hardened by teams of AI agents (EAT) on one MI350X machine, in two phases: first on
simplified interfaces (`amd_kda_mi350x` prefill, `amd_kda_decode_mi350x` decode), then on vLLM's production calling
convention (this package). `kda/prefill/kernel.cu` started from vLLM's fused ROCm Kimi-K3 chunk kernel (its header is
kept); the decode fallback is AITER's `fused_kda_decode`. Every number in this repository was measured on that machine.
The source comments in `kda/` were written during that work and refer to internal notes and experiments not included
here. Contact: Shiyi Cao (shicao@berkeley.edu).

License: Apache-2.0 (`LICENSE`). Third-party code (vLLM, Apache-2.0; AITER, MIT) keeps its headers; see `NOTICE`.
