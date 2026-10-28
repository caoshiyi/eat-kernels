# KDA prefill — NVIDIA B200 (sm_100a)

Kimi Delta Attention (chunked gated delta rule with per-channel gates) forward / chunked prefill for Kimi-K3
serving on B200: H = 12 (TP8) or 16 heads per GPU, head dim 128, varlen batches, a paged state pool, prefix-cache
checkpoints every 64 tokens, and one CUDA graph replayed across batch layouts. Written in CUDA C++ (tcgen05 / TMEM /
TMA, inline PTX) with Python host glue. On the 26 serving workloads below it is **2.57× faster than FlashKDA**
(geomean) with bf16 checkpoint rows and 2.40× with fp32 rows, and **3.39×** on the six Int21 workloads (H = 64 / 96,
plain calls), while doing strictly more work than FlashKDA in the serving set (pool gather / scatter and checkpoint
writes). This is the run's final production kernel; the benchmark-track kernel for plain calls (3.44× on Int21) is
in [`../b200_plain`](../b200_plain).

## Use

```python
from kda_b200 import kda_prefill, prepare, run

# plain or serving call (device-planned, plans cached per layout)
out, final_state = kda_prefill(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens)

# serving: state pool + prefix-cache checkpoints, captured once in a CUDA graph and replayed for every batch layout
launch = prepare(q, k, v, g, beta, A_log, dt_bias, 128 ** -0.5, pool, cu_seqlens, dynamic_layout=True,
                 ssm_state_indices=slots, state_checkpoints=ckpt, checkpoint_cu_starts=ck_starts,
                 checkpoint_every_n_tokens=64)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out, pool, ckpt = launch()
# ... write the next batch into q / k / v / g / beta / cu_seqlens / slots / ck_starts in place, then
graph.replay()
```

Shapes: q, k, v, g `[1, T, H, 128]` bf16; beta `[1, T, H]` bf16 (logits); A_log `[H]` fp32; dt_bias `[H*128]` fp32;
initial_state `[N, H, 128 (V), 128 (K)]` fp32, value-first (or a pool `[P, H, 128, 128]` with `ssm_state_indices`);
cu_seqlens `[N+1]` int64 or `None`. Semantics match FLA's `chunk_kda(..., use_gate_in_kernel=True,
use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True, safe_gate=True, lower_bound=-5.0, state_v_first=True)`;
`reference.py` is the fp64 token-by-token definition. The serving keywords follow FlashInfer's `recurrent_kda` prefill
contract as used by SGLang (`ssm_state_indices`, `state_checkpoints` / `checkpoint_cu_starts` /
`checkpoint_every_n_tokens`, `initial_state_source` / `initial_state_indices`); see the docstring of `kda_b200.py`.

- `run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, **serving)`: the serving entry point.
  Schedules are derived on the device; it meets the contract's host-latency limits (wall clock with a device sync:
  <= 1 ms + 0.01 ms per sequence on a repeated layout, <= 2 ms + 0.025 ms per sequence on a new one).
- `prepare(..., dynamic_layout=False)`: host-planned launch for a fixed layout (the fastest path for a static batch).
- `prepare(..., dynamic_layout=True)`: CUDA-graph-safe launch. Replays may change the contents of cu_seqlens,
  ssm_state_indices and checkpoint_cu_starts (same shapes, same total token count, every sequence non-empty).
- `status_bits(launch)`: device-side table validation of the last call (0 = all caller tables valid).

Requirements: an NVIDIA B200 (sm_100a), CUDA 13.x (`nvcc` on PATH; built with 13.2, driver 595.91), PyTorch with
CUDA 13 (tested: 2.12.1+cu130, Python 3.12). Extensions build on first use with `torch.utils.cpp_extension` into
`build*/` directories next to the sources; each kernel variant compiles the first time a call needs it (the tests
build about 20 variants, about 15 minutes on a 192-core host; later imports load the cached builds). `bench.py`
additionally needs `flash-linear-attention` (0.5.2) with FlashKDA installed (MoonshotAI FlashKDA, commit 7afb9f4)
and `cupti-python` >= 13.

```bash
CUDA_VISIBLE_DEVICES=0 python test_kda_b200.py     # correctness vs the fp64 recurrence (16 min from an empty build cache)
CUDA_VISIBLE_DEVICES=0 python bench.py             # 26 serving + 6 Int21 workloads vs FlashKDA (about 8 min once built)
CUDA_VISIBLE_DEVICES=0 python bench.py --ckpt-dtype bf16   # the same with bf16 checkpoint rows
CUDA_VISIBLE_DEVICES=0 python bench.py --quick     # 1 trial x 10 iterations
```

## Results (NVIDIA B200, torch 2.12.1+cu130, CUDA 13.2)

All numbers in this section were measured on the sources in this directory.

Geomean speedup over FlashKDA (fused CUTLASS forward, called through FLA's `chunk_kda` with the final state
requested, timed live in the same process). Timing: CUPTI kernel time, cold L2, the call captured in a CUDA graph,
median of 30 iterations, median of 3 trials, fresh random inputs each run. Serving workloads: the state is a pool of
N + 8 slots with a random slot permutation, checkpoints are written every 64 tokens, and the timed launch is
`prepare(..., dynamic_layout=True)`; FlashKDA gets contiguous initial states and writes no checkpoints. Int21
workloads are plain static calls. Every workload is checked against FLA's Triton `chunk_kda` (output and updated
state) before timing and after an in-place refill.

| measurement | serving, bf16 checkpoints | serving, fp32 checkpoints | Int21 |
|---|---|---|---|
| `bench.py --ckpt-dtype bf16` (seed 16708478) and `bench.py` (seed 6614966) | 2.573× | 2.403× | 3.392× / 3.387× |
| the run's official evaluation (all gates), pointed at this directory | 2.585× | 2.407× | 3.392× |

Per workload (FlashKDA time from the bf16 run; FlashKDA writes no checkpoints, so its time does not depend on the
checkpoint dtype):

| workload | H | sequence lengths | FlashKDA (ms) | bf16 ckpt (ms) | speedup | fp32 ckpt (ms) | speedup |
|---|---|---|---|---|---|---|---|
| long-1x8192 | 12 | 8192 | 0.7782 | 0.1313 | 5.93× | 0.1432 | 5.43× |
| long-1x32768 | 12 | 32768 | 3.0467 | 0.3645 | 8.36× | 0.4035 | 7.55× |
| long-2x8192 | 12 | 8192 × 2 | 0.8187 | 0.2239 | 3.66× | 0.2504 | 3.27× |
| long-3000+13384 | 12 | 3000, 13384 | 1.2684 | 0.2363 | 5.37× | 0.2555 | 4.97× |
| long-4x4096 | 12 | 4096 × 4 | 0.4596 | 0.2272 | 2.02× | 0.2473 | 1.86× |
| long-8x1088 | 12 | 1088 × 8 | 0.1617 | 0.0824 | 1.96× | 0.0886 | 1.83× |
| resid-4x64 | 12 | 64 × 4 | 0.0297 | 0.0165 | 1.80× | 0.0168 | 1.77× |
| resid-16x64 | 12 | 64 × 16 | 0.0421 | 0.0276 | 1.53× | 0.0283 | 1.48× |
| resid-64x64 | 12 | 64 × 64 | 0.1062 | 0.0614 | 1.73× | 0.0628 | 1.69× |
| resid-16x256 | 12 | 256 × 16 | 0.0819 | 0.0554 | 1.48× | 0.0576 | 1.42× |
| resid-16x17..255 | 12 | 109 × 16 | 0.0579 | 0.0381 | 1.52× | 0.0379 | 1.53× |
| agent8 | 12 | 256, 512, 900, 1600, 1600, 2400, 4000, 8700 | 0.8854 | 0.2891 | 3.06× | 0.3123 | 2.84× |
| agent16 | 12 | the agent8 lengths × 2 | 1.3225 | 0.5462 | 2.42× | 0.6368 | 2.08× |
| long-1x8192 | 16 | 8192 | 0.7905 | 0.1530 | 5.17× | 0.1644 | 4.81× |
| long-1x32768 | 16 | 32768 | 3.0952 | 0.4510 | 6.86× | 0.5016 | 6.17× |
| long-2x8192 | 16 | 8192 × 2 | 0.8443 | 0.3033 | 2.78× | 0.3201 | 2.64× |
| long-3000+13384 | 16 | 3000, 13384 | 1.2987 | 0.2824 | 4.60× | 0.3114 | 4.17× |
| long-4x4096 | 16 | 4096 × 4 | 0.4851 | 0.2551 | 1.90× | 0.2841 | 1.71× |
| long-8x1088 | 16 | 1088 × 8 | 0.1755 | 0.0847 | 2.07× | 0.0904 | 1.94× |
| resid-4x64 | 16 | 64 × 4 | 0.0305 | 0.0181 | 1.69× | 0.0183 | 1.67× |
| resid-16x64 | 16 | 64 × 16 | 0.0446 | 0.0286 | 1.56× | 0.0290 | 1.53× |
| resid-64x64 | 16 | 64 × 64 | 0.1345 | 0.0700 | 1.92× | 0.0741 | 1.82× |
| resid-16x256 | 16 | 256 × 16 | 0.0900 | 0.0558 | 1.61× | 0.0586 | 1.53× |
| resid-16x17..255 | 16 | 109 × 16 | 0.0621 | 0.0385 | 1.62× | 0.0395 | 1.57× |
| agent8 | 16 | 256, 512, 900, 1600, 1600, 2400, 4000, 8700 | 0.9172 | 0.3563 | 2.57× | 0.3922 | 2.34× |
| agent16 | 16 | the agent8 lengths × 2 | 1.3874 | 0.5465 | 2.54× | 0.6364 | 2.18× |
| **serving geomean** | | | | | **2.573×** | | **2.403×** |
| Int21 fixed | 96 | 8192 | 1.0365 | 0.3034 | 3.42× | 0.3036 | 3.41× |
| Int21 mixed_varlen | 96 | 1300, 547, 2048, 963, 271, 3063 | 0.9117 | 0.2492 | 3.66× | 0.2522 | 3.63× |
| Int21 uniform_varlen | 96 | 1024 × 8 | 0.7406 | 0.2524 | 2.93× | 0.2541 | 2.92× |
| Int21 fixed | 64 | 8192 | 0.9391 | 0.2394 | 3.92× | 0.2413 | 3.89× |
| Int21 mixed_varlen | 64 | 1300, 547, 2048, 963, 271, 3063 | 0.6841 | 0.1836 | 3.73× | 0.1817 | 3.77× |
| Int21 uniform_varlen | 64 | 1024 × 8 | 0.5023 | 0.1767 | 2.84× | 0.1768 | 2.85× |
| **Int21 geomean** | | | | | **3.392×** | | **3.387×** |

(The eval names the 16-sequence residual workload "17..255"; its generator re-seeds per draw, so all 16 lengths are 109.)

**bf16 checkpoints.** `state_checkpoints` may be bf16. Each row is the fp32 state rounded to nearest even,
bit-identical to rounding the rows of the same call with an fp32 buffer (output and pool are unchanged); bf16 rows
are stored as full 128 B lines, half the bytes of an fp32 row.

**Aligned static calls.** A static `prepare` without pool or checkpoints whose sequence lengths are all multiples
of 32 (all six Int21 workloads) uses a build of the M128 kernels without the partial-chunk handling (about 1% on
Int21); any other call uses the general build. The kernel traps if the precondition is violated.

**Correctness.** On this directory, every check of the run's official evaluation passes: the 26 serving workloads
(bf16 rows bit-exact against the fp32-checkpoint call) and 12 extra checkpoint cases (static / `run()` / dynamic,
every 64 / 256, padded pool, source pool), the Int21 workloads, fp64 probes off the timed distribution, a
real-workload gate with Kimi-Linear prefill activations, and the production contract (pool, checkpoints, source
pool, graph replays with changing layouts, host latency, FlashInfer's 29-shape production portfolio from 1 token to
1M tokens and 256 packed sequences). `test_kda_b200.py` passes (11 tests, against the fp64 recurrence).

**Robustness testing during the run.** The agents' gatekeepers stress-tested the run's own sources of this kernel
(before the release froze its defaults, below): 510 threaded hammer runs (8 threads overlapping prepared launches,
300 iterations each) and a soak of 24.2 M CUDA-graph replays for the fp32 path; for the aligned build another 510
hammer runs and 6.8 M soak replays; for bf16 rows 602 hammer runs, 7.0 M soak replays and about 700 differential
fuzz cases. All with 0 mismatches, 0 hangs, 0 traps; compute-sanitizer memcheck / synccheck clean (racecheck reports
only hazards mapped to benign classes).

## Production defaults were frozen

The run's sources carried about 45 environment-variable switches (`KDA_*` in Python, one `getenv` in a planner) for
A/B experiments, plus compile-time debug hooks (schedule-perturbation fuzz points, timer stamps, audit records,
debug builds). For this release every switch was replaced by its production default and the debug hooks were
removed; the remaining compile-time macros are the variants the host code selects (`KDA_FEAT`, `KDA_CK`, `KDA_CKBF`,
`KDA_NOPART`, `KDA_RWAIT`, ...). The inter-CTA / watchdog bound is the constant `KDA_BOUND_NS` = 60 s. After the
freeze, the fp32 paths were checked bit for bit against the run's sources on 178 layouts (407 result hashes: the 26
serving layouts with dynamic launch, in-place replay and `run()`, 120 held-out serving layouts, the 6 Int21 layouts,
and 26 layouts aimed at the remaining paths); the bf16 and aligned variants were then added as compile-time flavors
of the same sources and checked with the evaluation and tests above.

## Limitations

- **Hardware:** sm_100a only (B200; tuned on B200). Head dim 128, bf16 inputs, fp32 state; any H >= 1 works (tuned for
  H = 12 / 16 / 64 / 96). Scheduling follows cost models fitted on B200 over (H, sequence lengths, checkpoint interval).
- **Dynamic replays with N = 1:** a one-sequence `dynamic_layout=True` launch uses a plan frozen for cu = [0, T]. A
  replay with any other cu (a padded bucket) is a safe no-op: status bit 1, the state passes through, no checkpoint
  rows. With N >= 2, padded cu (cu[-1] < T) is handled exactly.
- **Input range:** q / k rows are L2-normalized exactly up to a row norm of 6.2e29; above that the output token (q)
  or that sequence-head's state (k) becomes NaN, loudly (never a silent wrong value). Subnormal q / k rows (norm
  below ~1e-38) are flushed to zero. Real activations are many orders of magnitude inside this window.
- **Tables:** caller tables are validated on the device (bad entries set status bits instead of faulting);
  a non-contiguous `ssm_state_indices` (or other dynamic table) raises `ValueError`.
- **Concurrency:** a prepared launch owns its buffers; overlapping launches of different prepared objects (from
  several threads / streams) are bit-exact, but the same prepared launch must not be replayed concurrently with
  itself. Tested with one process using the kernel and GPUs of one type.
- **Bounded waits:** cooperating CTAs assume co-residency. Every inter-CTA wait, chain-kernel barrier wait and a
  per-CTA lifetime watchdog trap after 60 s, so a lost hand-off becomes `unspecified launch failure` instead of a
  hang. A CTA descheduled for over a minute (a debugger, a GPU time-sliced for that long) would trap falsely.
- **Tested shapes:** the 26 serving and 6 Int21 workloads, 120 held-out serving layouts (H = 8 / 12 / 16 / 32 and
  other packings at H = 64 / 96; none slower than FlashKDA), the 29-shape production portfolio, probes and real
  Kimi-Linear captures. Other clocks or co-tenant load can shift the cost-model dispatch.

## Provenance

The starting point was the first SM100-family KDA prefill kernels in FlashInfer (the CAKE team's `flashkda_*` M64 /
M128 fused kernels, FlashInfer snapshot of Aug 13, 2026, Apache-2.0); the derived kernel files keep their NVIDIA
copyright headers (see NOTICE). An EAT supervisor-evolved agent team rewrote and extended them in three phases on
one git branch: phase 1 optimized plain calls on the six Int21 workloads (Oct 7-10, 2026: from about 2× for the
FlashInfer kernels to 3.48× over FlashKDA), phase 2 adapted the kernel to serving (Oct 10-13: 2.10× -> 2.41× on the
serving set), and phase 3 (Oct 14-15) added the aligned Int21 build and bf16 checkpoint rows (2.41× -> 2.58×). It
was a clean-room run: the agents had no network access, and no access to FlashInfer's KDA kernels newer than that
snapshot or to any other team's KDA kernels.

This directory is the run's final kernel, `solutions/cake_v64_bf16` (Oct 15, 2026), with the production defaults
frozen as described above. The harness is [EAT](https://github.com/caoshiyi/EAT); the traces (every supervisor and
agent transcript, the team configurations, the notes, and the kernel at every iteration) are in
[eva98/eat-evolution-traces](https://huggingface.co/datasets/eva98/eat-evolution-traces) (`b200_kda_clean/`).
