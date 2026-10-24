# Benchmark: validating `kda/` on MI350X

`eval_kda_mi350x.py` is the evaluation the package was optimised and accepted with. It calls `kda/kernel.py` exactly as
vLLM's ROCm Kimi-K3 KDA layer does (`CONTRACT.md`), checks every output and every written pool row against
`reference.py`, and times it against vLLM's fused ROCm prefill and the faster of AITER / vLLM HIP decode, run live on
the same inputs, interleaved with the candidate.

## Setup (once)

Requirements: one AMD Instinct MI350X (gfx950), ROCm 7.2.0 (`hipcc` on `PATH`), Python 3.12, the packages of
`requirements.txt` (torch 2.14.0+rocm7.2, triton-rocm 3.8.0, ninja, numpy, flash-linear-attention 0.5.2). No vLLM or
AITER install is needed: the baseline kernels are vendored under `baselines/`.

```bash
cd <repo>
pip install -r requirements.txt               # or use an environment that already has these versions
bash benchmark/build_baselines.sh             # builds the two vLLM HIP baselines (~20 s)
```

Optional: keep JIT caches inside the checkout, e.g. `export TRITON_CACHE_DIR=$PWD/.cache/triton`.
`kda/` builds its own extensions into `kda/prefill/build/` and `kda/decode/build/` at first use (the directory must
be writable): ~35 s for prefill at `import`, ~105 s for decode at the first decode call or `decode_warmup()`; later
runs load them in under a second. The first eval run also writes ~600 MB of fp32 reference outputs for the phase-1
gate to `benchmark/.cache/` (`KDA_BENCH_CACHE` moves it).

Select ONE idle GPU with `HIP_VISIBLE_DEVICES` (the eval refuses to run without it). Timings assume nothing else runs on
that GPU.

## (a) Full official eval

```bash
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py
```

~1 min warm (~4 min on a cold checkout: JIT builds + reference cache). Runs the 13 prefill + 10 decode serving
workloads (3 trials, each a median of 30 prefill / 100 decode iterations, cache flushed before every timed call)
and every hard gate (`CONTRACT.md`, "Hard gates"). Only a run without options prints an official score. The last
block reads:

```
  prefill_geomean_vs_vllm = 3.028   decode_geomean_vs_best_baseline = 1.686
  mean_vs_base = 2.259 (sqrt(prefill geomean x decode geomean) over 13 + 10 serving workloads)
```

Any gate failure prints `GATE FAILURE: ...` and `mean_vs_base = 0.000`, exit status 1. A complete example log is in
`expected_output_official.txt`.

Expected numbers (MI350X, ROCm 7.2.0):

| | final handover: 10 officials, one GPU | 18 officials over 4 GPUs | release check from a fresh clone of this repo (5 officials, one GPU, 2026-10-13) |
|---|---|---|---|
| `mean_vs_base` | 2.267 (2.263–2.271, sd 0.003) | 2.261–2.275 | 2.224–2.259 |
| prefill geomean vs vLLM fused ROCm prefill | 3.038 (3.035–3.043) | 3.035–3.055 | 3.017–3.028 |
| decode geomean vs best of AITER / vLLM HIP | 1.691 (1.684–1.699) | 1.678–1.699 | 1.639–1.686 |
| phase-1 decode shapes (gate floor 1.500) | 1.590 (1.585–1.604) | 1.559–1.604 | 1.584–1.592 |
| phase-1 prefill shapes (gate floor 0.932) | 2.81 (2.79–2.84) | 2.78–2.84 | 2.77–2.85 |
| gate failures | 0 | 0 | 0 |

Plus 40 full evals over new seeds (691–730) in the handover: 0 gate failures, `mean_vs_base` 2.258–2.276.

How to read the spread: the baselines are timed live, so their run-to-run variation moves the ratios. The package's own
latencies repeat within about ±0.5 % per workload; most of the spread is the baselines. In the release check above
the package latencies matched the handover's (for example s8_h96_b32 330–333 µs, p12_loaded16k_1 113 µs), but the
AITER baseline was faster than at handover time on some workloads (s8_h12_b128 263–268 µs instead of 280 µs;
s8_h96_b32 566 µs in two of the runs instead of ~605 µs). Running the original workspace copy of the package and eval
interleaved on the same GPU gave 2.244 and 2.252 in the same session. To compare two package versions, alternate them
on one GPU; a difference below 1–2 % between runs on different GPUs or hours apart means nothing. The narrowest gate is
the phase-1 decode floor (1.500): lowest reading 1.559 on the slowest of four GPUs, about 4 % margin. A heavily
oversubscribed host (B = 1 decode calls are latency-bound, +0.1–0.3 µs under host load) is the realistic way to fail it.

## (b) Quick correctness tests

```bash
HIP_VISIBLE_DEVICES=0 python benchmark/smoke_test.py          # no baselines needed; ~2.5 min cold (JIT builds), ~6 s warm
```
Two prefill workloads (and again after an in-place refill of every input and the pool), three decode workloads as captured
HIP graphs (and again after replay with new slot indices / data), and the integration gate (prefill, then speculative
verify, then prefill on one pool, then a prefix-cache hit). Ends with `smoke test: PASS`. Expected errors: prefill out ~4e-3,
pool ~4e-4 (tolerance 2e-2); decode out ≤ 2e-4, state ~2e-7 (tolerances 2e-2 / 1e-2).

Gates only, without timing (the `numerics` gate also runs the vLLM prefill baseline for comparison, so it needs
`build_baselines.sh`):
```bash
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --gates contract,host,decode,integration
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --gates numerics,cotenant,legacy
```
Ends with `all selected gates pass [gates only; not an official score]`. Gate names: `contract`, `host`, `numerics`,
`decode`, `integration`, `cotenant`, `legacy`.

Serving workloads without gates, fewer iterations (not an official score):
```bash
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --quick      # 4 + 4 workloads, 1 trial
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --no-gates   # all 13 + 10 workloads
```

## (c) Individual workloads

```bash
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --part prefill --workloads p12_loaded16k_1 p96_loaded8k_0
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --part decode  --workloads s8_h12_b1 s8_h96_b32 --trials 5
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --seed 123 --no-gates      # fixed data seed (default: random)
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --heldout my_batches.txt   # unscored prefill compositions
```
Workload names and definitions: `workloads/README.md`. Per-workload expected ratios (mean of ten officials): table in
`docs/HANDOVER.md`, section 9. Each line prints the baseline and candidate latency (µs), the ratio and the correctness
detail, e.g.
```
p12_loaded16k_1    12  12  15872 |     463.9     113.2   4.098x | PASS (out 3.9e-03 pool 3.6e-04, after refill ...)
s8_h12_b1          12   1  8 |     32.20     15.41   2.087x | PASS (out 4.4e-05 state 1.5e-07, replay with new indices/data: ...)
```
`--kernel <path>/kernel.py` evaluates another copy of the package (for example one built at another path).

## What is in `benchmark/`

| path | what |
|---|---|
| `eval_kda_mi350x.py` | the eval (workloads, gates, timing, score). Same logic as the original; only paths changed: `KDA_REPO_ROOT`, `KDA_BASELINES_DIR`, `KDA_WORKLOADS_DIR`, `KDA_BENCH_CACHE` override them (docstring) |
| `smoke_test.py` | quick correctness check, uses the eval's harness |
| `build_baselines.sh` | builds `baselines/prefill/libkda_sota.so` and `baselines/decode/vllm_hip/libkda_decode_sota.so` |
| `baselines/sota_prod.py` | baseline entry points with the production signature. Prefill: vLLM's fused ROCm Kimi-K3 prefill (`fused_kda_prologue` + `fused_kda_chunk`, vLLM 9521c60), Python side as in `baselines/prefill/vllm_reference/kda_chunk.py`. Decode: AITER `fused_kda_decode` (ae984f5 + the gfx950 speculative path of ROCm/aiter#5709, Triton, all shapes) and vLLM's ROCm fused decode kernel (vLLM 0f112d1, plain decode at H ∈ {12, 24, 48, 96}); per workload the faster one counts |
| `baselines/prefill/`, `baselines/decode/` | vendored kernel sources, op bindings and build scripts of those baselines |
| `vllm_reference/kda.py`, `kda_decode.py` | vLLM's ROCm Kimi-K3 KDA layer and decode entry points: the production calling convention the contract copies (reference only, not imported) |
| `expected_output_official.txt` | full log of one official run from a fresh clone |

Not standalone: `baselines/prefill/vllm_reference/*.py` import `vllm` and are kept as the source the prefill baseline
follows, not run. The phase-1 gate compares against latencies recorded on the development MI350X
(`workloads/legacy_*_baseline_sota.json`), not live baselines, so its numbers shift with the GPU.
