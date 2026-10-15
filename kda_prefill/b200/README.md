# KDA prefill — NVIDIA B200 (sm_100a)

Kimi Delta Attention (chunked gated delta rule with per-channel gates) forward / chunked prefill for Kimi-K3
serving on B200: H = 12 (TP8) or 16 heads per GPU, head dim 128, varlen batches, a paged state pool, prefix-cache
checkpoints every 64 tokens, and one CUDA graph replayed across batch layouts. Written in CUDA C++ (tcgen05 / TMEM /
TMA, inline PTX) with Python host glue. On the 26 serving workloads below it is **2.58x faster than FlashKDA**
(geomean) with bf16 checkpoint rows and 2.41x with fp32 rows, and on the six Int21 workloads (H = 64 / 96, plain
calls) **3.34x**, while doing strictly more work than FlashKDA in the serving set (pool gather / scatter and
checkpoint writes).

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

Geomean speedup over FlashKDA (fused CUTLASS forward, called through FLA's `chunk_kda` with the final state
requested, timed live in the same process). Timing: CUPTI kernel time, cold L2, the call captured in a CUDA graph,
median of 30 iterations, median of 3 trials, fresh random inputs each run. Serving workloads: the state is a pool of
N + 8 slots with a random slot permutation, checkpoints are written every 64 tokens, and the timed launch is
`prepare(..., dynamic_layout=True)`; FlashKDA gets contiguous initial states and writes no checkpoints. Every workload
is checked against FLA's Triton `chunk_kda` (output and updated state) before timing and after an in-place refill.

One `bench.py` run with fp32 checkpoint rows, from a fresh checkout with empty build caches (seed 10049429; this
table was measured before bf16 checkpoints were added, and the fp32 path is unchanged):

| workload | H | sequence lengths | this (ms) | FlashKDA (ms) | speedup |
|---|---|---|---|---|---|
| long-1x8192 | 12 | 8192 | 0.1423 | 0.7798 | 5.48x |
| long-1x32768 | 12 | 32768 | 0.3971 | 3.0470 | 7.67x |
| long-2x8192 | 12 | 8192 x 2 | 0.2478 | 0.8187 | 3.30x |
| long-3000+13384 | 12 | 3000, 13384 | 0.2544 | 1.2712 | 5.00x |
| long-4x4096 | 12 | 4096 x 4 | 0.2468 | 0.4601 | 1.86x |
| long-8x1088 | 12 | 1088 x 8 | 0.0884 | 0.1625 | 1.84x |
| resid-4x64 | 12 | 64 x 4 | 0.0170 | 0.0293 | 1.73x |
| resid-16x64 | 12 | 64 x 16 | 0.0281 | 0.0426 | 1.52x |
| resid-64x64 | 12 | 64 x 64 | 0.0610 | 0.1063 | 1.74x |
| resid-16x256 | 12 | 256 x 16 | 0.0570 | 0.0820 | 1.44x |
| resid-16x17..255 | 12 | 109 x 16 | 0.0381 | 0.0573 | 1.50x |
| agent8 | 12 | 256, 512, 900, 1600, 1600, 2400, 4000, 8700 | 0.3229 | 0.8868 | 2.75x |
| agent16 | 12 | the agent8 lengths x 2 | 0.6364 | 1.3244 | 2.08x |
| long-1x8192 | 16 | 8192 | 0.1650 | 0.7922 | 4.80x |
| long-1x32768 | 16 | 32768 | 0.5060 | 3.0963 | 6.12x |
| long-2x8192 | 16 | 8192 x 2 | 0.3208 | 0.8443 | 2.63x |
| long-3000+13384 | 16 | 3000, 13384 | 0.3124 | 1.3003 | 4.16x |
| long-4x4096 | 16 | 4096 x 4 | 0.2837 | 0.4854 | 1.71x |
| long-8x1088 | 16 | 1088 x 8 | 0.0905 | 0.1763 | 1.95x |
| resid-4x64 | 16 | 64 x 4 | 0.0180 | 0.0307 | 1.71x |
| resid-16x64 | 16 | 64 x 16 | 0.0286 | 0.0446 | 1.56x |
| resid-64x64 | 16 | 64 x 64 | 0.0733 | 0.1348 | 1.84x |
| resid-16x256 | 16 | 256 x 16 | 0.0581 | 0.0902 | 1.55x |
| resid-16x17..255 | 16 | 109 x 16 | 0.0387 | 0.0625 | 1.61x |
| agent8 | 16 | 256, 512, 900, 1600, 1600, 2400, 4000, 8700 | 0.3886 | 0.9187 | 2.36x |
| agent16 | 16 | the agent8 lengths x 2 | 0.6368 | 1.3902 | 2.18x |
| **serving geomean** | | | | | **2.416x** |
| Int21 fixed | 96 | 8192 | 0.3040 | 1.0367 | 3.41x |
| Int21 mixed_varlen | 96 | 1300, 547, 2048, 963, 271, 3063 | 0.2530 | 0.9120 | 3.60x |
| Int21 uniform_varlen | 96 | 1024 x 8 | 0.2566 | 0.7409 | 2.89x |
| Int21 fixed | 64 | 8192 | 0.2576 | 0.9403 | 3.65x |
| Int21 mixed_varlen | 64 | 1300, 547, 2048, 963, 271, 3063 | 0.1828 | 0.6845 | 3.74x |
| Int21 uniform_varlen | 64 | 1024 x 8 | 0.1792 | 0.5023 | 2.80x |
| **Int21 geomean** | | | | | **3.329x** |

(The eval names the 16-sequence residual workload "17..255"; its generator re-seeds per draw, so all 16 lengths are 109.)

### bf16 checkpoints

`state_checkpoints` may be bf16. Each row is the fp32 state rounded to nearest even, bit-identical to rounding the
rows of the same call with an fp32 buffer (output and pool are unchanged); the rows are stored as full 128 B lines,
half the bytes of an fp32 row. One `bench.py --ckpt-dtype bf16` run (seed 7168497), serving workloads:

| workload | H = 12 (ms) | speedup | H = 16 (ms) | speedup |
|---|---|---|---|---|
| long-1x8192 | 0.1333 | 5.83x | 0.1544 | 5.11x |
| long-1x32768 | 0.3622 | 8.39x | 0.4594 | 6.72x |
| long-2x8192 | 0.2205 | 3.71x | 0.2998 | 2.81x |
| long-3000+13384 | 0.2369 | 5.37x | 0.2778 | 4.67x |
| long-4x4096 | 0.2214 | 2.07x | 0.2536 | 1.91x |
| long-8x1088 | 0.0840 | 1.92x | 0.0858 | 2.04x |
| resid-4x64 | 0.0165 | 1.77x | 0.0171 | 1.73x |
| resid-16x64 | 0.0271 | 1.57x | 0.0283 | 1.59x |
| resid-64x64 | 0.0619 | 1.74x | 0.0701 | 1.90x |
| resid-16x256 | 0.0544 | 1.50x | 0.0563 | 1.60x |
| resid-16x17..255 | 0.0369 | 1.57x | 0.0380 | 1.62x |
| agent8 | 0.2987 | 2.96x | 0.3560 | 2.57x |
| agent16 | 0.5453 | 2.43x | 0.5419 | 2.56x |
| **serving geomean** | | | | **2.579x** |

Int21 geomean in the same run: 3.341x.

Repeat measurements of the same sources with the official eval:

| run | serving geomean | Int21 geomean |
|---|---|---|
| this directory, through the official eval (all gates) | 2.414 | 3.348 |
| this directory, `bench.py` from a fresh checkout (table above) | 2.416 | 3.329 |
| the run's final kernel, official evals (3 runs) | 2.414 / 2.415 / 2.418 | 3.341 / 3.341 / 3.345 |
| this directory with bf16 checkpoints, through the phase-3 official eval (all gates) | 2.582 (bf16 rows), 2.406 (fp32 rows) | 3.372 |

Correctness and robustness: all **117 correctness gates** of the official eval pass on this directory, re-run after
freezing the defaults: serving and Int21 workloads, fp64 probes off the timed distribution, a real-workload gate with
Kimi-Linear prefill activations, and the production contract: pool, checkpoints, source pool, graph replays with
changing layouts, host latency, FlashInfer's 29-shape production portfolio from 1 token to 1M tokens and 256 packed
sequences. On the run's final sources, which this directory reproduces bit for bit: **510 hammer runs** (8 threads
overlapping prepared launches, 300 iterations each) and a **soak of 24.2 M graph replays** (10 serving configurations
on two GPUs, plus threads, SM burners and NaN injection; 140 GPU-minutes) with **0 mismatches, 0 hangs, 0 traps**;
compute-sanitizer memcheck / synccheck clean (racecheck reports only hazards mapped to benign classes).

## Production defaults were frozen

The run's sources carried about 45 environment-variable switches (`KDA_*` in Python, one `getenv` in a planner) for
A/B experiments, plus compile-time debug hooks (schedule-perturbation fuzz points, timer stamps, audit records,
debug builds). For this release every switch was replaced by its production default and the debug hooks were
removed; the remaining compile-time macros are the feature variants the host code selects (`KDA_FEAT`, `KDA_CK`,
`KDA_RWAIT`, ...). The inter-CTA / watchdog bound is the constant `KDA_BOUND_NS` = 60 s. Outputs, final states, pool
slots and checkpoint rows are bit-identical to the run's final kernel on 178 layouts (407 result hashes): the 26
serving layouts (dynamic launch, in-place replay and `run()`), 120 held-out serving layouts (launch and replay), the
6 Int21 layouts (static `prepare` and `run()`), and 26 layouts aimed at the remaining paths (one-chunk sequences,
H % 8 != 0, few heads, source pool, checkpoints every 256, up to 256 sequences).

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
copyright headers (see NOTICE). An EAT supervisor-evolved agent team rewrote and extended them in three phases: phase 1
optimized plain calls on the six Int21 workloads (37 iterations, Oct 7-10, 2026: from about 2x for the FlashInfer
kernels to 3.48x over FlashKDA), phase 2 adapted the kernel to serving (20 iterations, Oct 10-13, 2026: 2.10x -> 2.41x
on the serving set, Int21 kept above its 3.27 floor), and phase 3 (Oct 14-15, 2026) added bf16 checkpoint rows
(2.41x -> 2.58x on the serving set). It was a clean-room run: the agents had no network access, and
no access to FlashInfer's KDA kernels newer than that snapshot or to any other team's KDA kernels.

The harness is [EAT](https://github.com/caoshiyi/EAT); the complete traces of phases 1 and 2 are in
[eva98/eat-evolution-traces](https://huggingface.co/datasets/eva98/eat-evolution-traces) (`b200_kda_clean/`):
- every supervisor and agent transcript;
- the team configurations;
- the notes, and the kernel at every iteration.

This directory corresponds to `solutions/phase2_serving/cake_v62` of that run (phase 2 closeout, Oct 13, 2026), with the production
defaults frozen as described above, plus the bf16 checkpoint flavor of phase 3 (`cake_v64_bf16`). Phase 3 also
added a build without the partial-chunk handling for static plain calls whose sequence lengths are all multiples of
32 (about 1% on the Int21 workloads); it is not included.
