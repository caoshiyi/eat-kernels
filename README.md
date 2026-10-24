# EAT Kernels

Fast, verified GPU kernels for LLM inference, for every major accelerator, packaged so engines can call them
directly. Each kernel is discovered and optimized by agent teams (EAT), and ships with:
- its reference implementation and the benchmark it was measured with;
- the recorded baseline numbers and the hardware it was tuned on;
- known limitations, and a link to the full evolution traces.

## KDA (Kimi Delta Attention), head dim 128, varlen batches

| directory | hardware | interface | ops | vs best prior kernel |
|---|---|---|---|---|
| [`kda/mi350x`](kda/mi350x) | AMD MI350X (gfx950) | **serving**: vLLM's ROCm Kimi-K3 calling convention (paged state pool, prefix-cache checkpoints, HIP graphs) | prefill + fused decode / speculative verify | prefill 3.04x vs vLLM ROCm; decode 1.69x vs the faster of AITER / vLLM HIP |
| [`kda/b200`](kda/b200) | NVIDIA B200 (sm_100a) | **serving**: paged state pool, prefix-cache checkpoints, one CUDA graph across batch layouts; plain calls too | prefill | 2.57x vs FlashKDA (serving set, bf16 checkpoints; 2.40x fp32), 3.39x (Int21, H=64/96) |
| [`kda/mi350x_plain`](kda/mi350x_plain) | AMD MI350X (gfx950) | plain calls (fla `chunk_kda` convention, contiguous state) | prefill | 3.0x vs vLLM ROCm, 7.3x vs fla Triton (H=12) |
| [`kda/b200_plain`](kda/b200_plain) | NVIDIA B200 (sm_100a) | plain calls (benchmark track) | forward | 3.44x vs FlashKDA (Int21, H=64/96) |

For an inference engine, use the **serving** kernel of your hardware. The `_plain` kernels are the earlier or
benchmark-track versions on simpler interfaces, kept with their own results.

Layout: `<op>/<hardware>[_<variant>]/`. Each directory holds the sources, a Python entry point, `reference.py`, a
benchmark, tests and a `README.md` (API, results, limitations, provenance).

License: Apache-2.0 (see NOTICE for third-party code).
