<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/eat_lockup_dark.png">
    <img alt="EAT" src="assets/eat_lockup_light.png" width="300">
  </picture>
</p>

<h3 align="center">EAT Kernels</h3>
<p align="center"><em>Verified GPU kernels discovered by evolving agent teams</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.36848"><img alt="Paper" src="https://img.shields.io/badge/Paper-NeurIPS%202026-E8A12E?style=flat-square&logo=arxiv&logoColor=white&labelColor=0E1A2B"></a>
  <a href="https://github.com/caoshiyi/EAT"><img alt="EAT" src="https://img.shields.io/badge/Harness-EAT-E8A12E?style=flat-square&logo=github&logoColor=white&labelColor=0E1A2B"></a>
  <a href="https://huggingface.co/datasets/eva98/eat-evolution-traces"><img alt="Traces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Traces-agent%20transcripts-E8A12E?style=flat-square&labelColor=0E1A2B"></a>
  <a href="https://huggingface.co/datasets/eva98/eat-gym"><img alt="EAT Gym" src="https://img.shields.io/badge/%F0%9F%A4%97%20EAT%20Gym-tasks-E8A12E?style=flat-square&labelColor=0E1A2B"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-5C7FA3?style=flat-square&labelColor=0E1A2B"></a>
</p>

Fast, verified GPU kernels for LLM inference, packaged so engines can call them directly. Each kernel was discovered
and optimized by agent teams evolved by [EAT](https://github.com/caoshiyi/EAT), and ships with:
- its reference implementation and the benchmark it was measured with;
- the recorded baseline numbers and the hardware it was tuned on;
- known limitations, and a link to the full evolution traces.

## KDA (Kimi Delta Attention), head dim 128, varlen batches

| directory | hardware | interface | ops | vs best prior kernel |
|---|---|---|---|---|
| [`kda/mi350x`](kda/mi350x) | AMD MI350X (gfx950) | **serving**: vLLM's ROCm Kimi-K3 calling convention (paged state pool, prefix-cache checkpoints, HIP graphs) | prefill + fused decode / speculative verify | prefill 3.04x vs vLLM ROCm; decode 1.69x vs the faster of AITER / vLLM HIP |
| [`kda/b200`](kda/b200) | NVIDIA B200 (sm_100a) | **serving**: paged state pool, prefix-cache checkpoints, one CUDA graph across batch layouts; plain calls too | prefill | 2.57x vs FlashKDA (serving set, bf16 checkpoints; 2.40x fp32), 3.39x (Int21, H=64/96) |
| [`kda/mi350x_plain`](kda/mi350x_plain) | AMD MI350X (gfx950) | plain calls (fla `chunk_kda` convention, contiguous state) | prefill | 3.01x vs vLLM ROCm, 7.3x vs fla Triton (H=12) |
| [`kda/b200_plain`](kda/b200_plain) | NVIDIA B200 (sm_100a) | plain calls (benchmark track) | forward | 3.44x vs FlashKDA (Int21, H=64/96) |

For an inference engine, use the **serving** kernel of your hardware. The `_plain` kernels are the earlier or
benchmark-track versions on simpler interfaces, kept with their own results.

Layout: `<op>/<hardware>[_<variant>]/`. Each directory holds the sources, a Python entry point, `reference.py`, a
benchmark, tests and a `README.md` (API, results, limitations, provenance).

License: Apache-2.0 (see NOTICE for third-party code).
