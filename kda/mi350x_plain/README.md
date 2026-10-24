# KDA prefill — AMD Instinct MI350X (gfx950)

Kimi Delta Attention (chunked gated delta rule with per-channel gates) prefill for Kimi-K3 at TP8 (H = 12 heads,
head dim 128, varlen batches), written in HIP C++. On the 10 production-shaped workloads below it is
**3.0x faster than vLLM's hand-written ROCm KDA kernels** and **7.3x faster than flash-linear-attention's Triton
`chunk_kda`**, with output and final state within 5e-3 relative L2 of an fp32 reference.

This is the phase-1 kernel on a simplified interface. The production version on vLLM's calling convention
(paged state pool, prefix-cache checkpoints, plus fused decode) is [`../mi350x`](../mi350x).

## Use

```python
from kda_mi350x import kda_prefill, prepare
meta = prepare(cu_seqlens)   # optional; depends only on cu_seqlens, so it can be built once per batch layout
out, final_state = kda_prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, cu_seqlens,
                               scale=128 ** -0.5, lower_bound=-5.0, meta=meta)
```

Inputs and semantics match fla's `chunk_kda(..., use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
use_qk_l2norm_in_kernel=True, safe_gate=True, state_v_first=True)`; see the docstring of `kda_mi350x.py`.
Requirements: ROCm 7.x PyTorch on an MI350X/MI355X; the extension builds on first import (about 1-2 minutes).

```bash
HIP_VISIBLE_DEVICES=0 python bench.py            # correctness + latency on the 10 workloads (needs sota/build.sh and fla for baselines)
HIP_VISIBLE_DEVICES=0 python test_slowdecay.py   # correctness under slow-decay (long-memory) gates
```

## Results (AMD Instinct MI350X, torch 2.14.0+rocm7.2)

Median of 50 runs, Infinity Cache flushed before each; latency includes the launch from Python.
`prepare` is untimed for both this kernel and vLLM (whose own host metadata is also built outside the timed region).

| workload | sequence lengths | this (ms) | vLLM ROCm (ms) | vs vLLM | vs fla |
|---|---|---|---|---|---|
| p_1x1k | 1024 | 0.0302 | 0.0629 | 2.09x | 11.7x |
| p_1x4k | 4096 | 0.0504 | 0.1222 | 2.42x | 9.1x |
| p_1x16k | 16384 | 0.1522 | 0.3988 | 2.62x | 7.3x |
| p_1x64k | 65536 | 0.5331 | 1.3895 | 2.61x | 8.2x |
| p_4x4k | 4096 x 4 | 0.1137 | 0.3805 | 3.35x | 6.0x |
| p_8x2k | 2048 x 8 | 0.1041 | 0.3388 | 3.25x | 6.0x |
| p_16x1k | 1024 x 16 | 0.1025 | 0.3160 | 3.08x | 6.2x |
| p_32x512 | 512 x 32 | 0.1084 | 0.3202 | 2.96x | 5.7x |
| p_64x128 | 128 x 64 | 0.0666 | 0.1920 | 2.88x | 6.5x |
| p_mixed | 8192, 1024, 777, 512, 256, 130, 64, 17 | 0.1062 | 0.4914 | 4.63x | 6.8x |
| **mean** | | | | **2.99x** | **7.3x** |

The table is the run's last official eval. I re-measured the kernel independently on an idle GPU with three full
evals: 2.975x, 3.018x and 3.037x (mean 3.01x). The slow-decay correctness test (`test_slowdecay.py`) passes.

## Limitations

- **Hardware:** built and tuned for gfx950 (MI350X); MI355X is the same ISA but untested.
- **Shapes:** bf16 inputs, head dim 128, fp32 state. `prepare()` builds its scheduling metadata for H = 12. With other head
  counts, `run` falls back to on-device metadata and a uniform group policy (correct, but not tuned).
- **Concurrency:** the relay route hands a sequence's state between workgroups through a global flag array in device memory.
  Do not run two calls concurrently on different streams of the same GPU; calls on one stream are fine.
- **Tested shapes:** correctness was tested on the 10 workloads above, on held-out batch shapes and under slow-decay gates.
  The dispatch between kernel instances follows a cost model fitted on MI350X, not on the eval shapes. Other GPUs, clocks or
  co-tenant load can shift it.

## Provenance

The starting point was a faithful HIP port of vLLM's ROCm Kimi-K3 KDA prefill kernels (vLLM commit 9521c60, Apache-2.0;
copied in `sota/` as the baseline). An EAT supervisor-evolved agent team rewrote it over 73 iterations
(Sep 29 to Oct 7, 2026; $3,018 of Claude usage). The complete traces are in [eva98/eat-evolution-traces](https://huggingface.co/datasets/eva98/eat-evolution-traces) (`amd_kda_mi350x/`):
- every supervisor and agent transcript;
- the team configurations;
- the notes, and the kernel at every iteration.

This file corresponds to commit `ed8e7043c0` (iteration 73) of that run.
