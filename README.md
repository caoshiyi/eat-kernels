# EAT Kernels

Fast, verified GPU kernels for LLM inference, for every major accelerator, packaged so engines can call them
directly. Each kernel is discovered and optimized by agent teams (EAT), and ships with:
- its reference implementation and the benchmark it was measured with;
- the recorded baseline numbers and the hardware it was tuned on;
- known limitations, and a link to the full evolution traces.

| op | hardware | directory | vs best prior kernel | status |
|---|---|---|---|---|
| KDA prefill (Kimi-K3, H=12, D=128, varlen) | AMD MI350X (gfx950) | `kda_prefill/amd_mi350x` | 3.0x vs vLLM ROCm, 7.3x vs fla Triton | released |
| KDA prefill + fused decode / speculative verify (Kimi-K3 serving, vLLM ROCm calling convention: paged state pool, checkpoints, HIP graphs) | AMD MI350X (gfx950) | `kda_serving/amd_mi350x` | prefill 3.04x vs vLLM ROCm, decode 1.69x vs the faster of AITER / vLLM HIP | released |
| KDA prefill (Kimi-K3 serving, H=12/16, D=128, varlen, state pool + checkpoints) | NVIDIA B200 (sm_100a) | `kda_prefill/b200` | 2.57x vs FlashKDA (serving, bf16 checkpoints; 2.40x fp32), 3.39x (Int21, H=64/96) | released |
| KDA forward, plain calls (benchmark track, H=64/96, D=128, varlen) | NVIDIA B200 (sm_100a) | `kda_prefill/b200_int21` | 3.44x vs FlashKDA (Int21) | released |

Layout: `<op>/<hardware>/` (`kda_serving/` = prefill and decode together, as a serving engine calls them) holds `csrc/` (or a Python DSL source), a Python entry point, `reference.py`, `bench.py`,
tests, `README.md` (API, results, limitations, provenance).

License: Apache-2.0 (see NOTICE for third-party code).
