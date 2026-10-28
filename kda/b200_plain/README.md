# KDA forward — NVIDIA B200 (sm_100a), plain calls (benchmark track)

Kimi Delta Attention (chunked gated delta rule with per-channel gates) forward for plain calls: initial state in,
output and final state out, varlen batches, head dim 128. Written in CUDA C++ (tcgen05 / TMEM / TMA, inline PTX) with
Python host glue. On the six Int21 workloads (H = 64 / 96, 8192 tokens) it is **3.44× faster than FlashKDA**
(geomean). For serving (state pool, prefix-cache checkpoints, graph-safe dynamic layouts) use the production
kernel in [`../b200`](../b200), which reaches 3.39× on the same workloads.

## Use

```python
from kda_b200_int21 import run, prepare

out, final_state = run(q, k, v, g, beta, A_log, dt_bias, 128 ** -0.5, initial_state, cu_seqlens)

launch = prepare(q, k, v, g, beta, A_log, dt_bias, 128 ** -0.5, initial_state, cu_seqlens)   # host-planned, fixed layout
out, final_state = launch()
```

Shapes: q, k, v, g `[1, T, H, 128]` bf16; beta `[1, T, H]` bf16 (logits); A_log `[H]` fp32; dt_bias `[H*128]` fp32;
initial_state `[N, H, 128 (V), 128 (K)]` fp32, value-first, or `None`; cu_seqlens `[N+1]` int64 or `None`. Semantics
match FLA's `chunk_kda(..., use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True,
safe_gate=True, lower_bound=-5.0, state_v_first=True)`; `../b200/reference.py` is the fp64 token-by-token definition.

Requirements: as for [`../b200`](../b200) (B200, CUDA 13.x, PyTorch with CUDA 13; `bench.py` also needs
`flash-linear-attention` 0.5.2 with FlashKDA 7afb9f4 and `cupti-python`). `bench.py` reuses `../b200/bench.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python bench.py     # 6 Int21 workloads vs FlashKDA, correctness vs FLA chunk_kda
```

## Results (NVIDIA B200, torch 2.12.1+cu130, CUDA 13.2)

Measured on the sources in this directory. Timing as in `../b200`: CUPTI kernel time, cold L2, the call captured in
a CUDA graph, median of 30 iterations, median of 3 trials; FlashKDA is called through FLA's `chunk_kda` with the
final state requested. Each workload is checked against FLA's Triton `chunk_kda` before timing and after an
in-place refill.

One `bench.py` run (seed 2984520):

| workload | H | sequence lengths | FlashKDA (ms) | this (ms) | speedup |
|---|---|---|---|---|---|
| fixed | 96 | 8192 | 1.0372 | 0.2976 | 3.49× |
| mixed_varlen | 96 | 1300, 547, 2048, 963, 271, 3063 | 0.9163 | 0.2452 | 3.74× |
| uniform_varlen | 96 | 1024 × 8 | 0.7404 | 0.2511 | 2.95× |
| fixed | 64 | 8192 | 0.9402 | 0.2422 | 3.88× |
| mixed_varlen | 64 | 1300, 547, 2048, 963, 271, 3063 | 0.6856 | 0.1784 | 3.84× |
| uniform_varlen | 64 | 1024 × 8 | 0.5027 | 0.1740 | 2.89× |
| **Int21 geomean** | | | | | **3.439×** |

The run's phase-1 official evaluation (the version at the tag: Int21 timing plus correctness checks against the
fp64 reference, probes off the timed distribution, trained-gate and Kimi-Linear real-workload gates) passes every
check on this directory and scores **3.428×**; in the run, the same kernel scored 3.446× when it was promoted.

## Limitations

- Plain calls only: no state pool (`ssm_state_indices`), no prefix-cache checkpoints, no source pool, and no
  graph-safe dynamic layouts (a prepared launch is for one fixed layout). The current evaluation's production
  contract checks therefore do not apply.
- sm_100a only, head dim 128, bf16 inputs, fp32 state; tuned for H = 64 / 96.

## Provenance

This is `solutions/cake_v44` of the clean-room EAT run described in [`../b200`](../b200), at the git tag
`bench-cake_v44` (Oct 9, 2026): the phase-1 kernel promoted in iteration 18 and frozen as the benchmark-track
result, unchanged (it reads no environment switches and has no debug hooks). The production kernel in `../b200`
was developed from it. The derived kernel files keep their NVIDIA copyright headers (see NOTICE).
