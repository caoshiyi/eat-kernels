# Real-workload correctness gate

Checks both operators on activations captured from Kimi-Linear-48B (the open Kimi linear-attention model) running
GSM8K / MATH-500 prefills, against an fp64 recurrence. Not timed. The official eval's numerics gates use synthetic
data; this gate uses real activations.

**Data.** For each of 6 layers (0, 5, 10, 14, 18, 25) and 5 prompt sets, the q / k / v (after the short conv and SiLU), the gate
`g` (log-decay, in [-5, 0]) and `beta` of one forward pass, 32 heads per layer:
`math500-multi-8192` / `math500-multi-4096` / `gsm8k-8shot` (one long packed request of 8183 / 4094 / 1595 tokens),
`math500-batch32` / `gsm8k-batch32` (32 requests of 64–172 tokens, `cu_seqlens`). The data (4.8 GB of safetensors) is not in this
repository; it is available on request. Put it in `benchmark/real_gate/data/<prompt set>/layerNN.safetensors` or point `KDA_REAL_DATA` to it.

**Conversion to the package's convention.** `A_log = dt_bias = 0`, `lower_bound = -5`, `raw_g = logit(-g / 5)` (so the
package's gate reproduces `g` exactly), `raw_beta = logit(beta)`. Layers are stacked along heads: three layers give
H = 96 (Kimi-K3 at TP1), two give H = 64; `_h12` cases take 4 heads from each of three layers (H = 12, Kimi-K3 at TP8).

**prefill** (19 cases): vLLM's calling convention: strided `raw_beta`, output into a slice of a larger buffer, pool rows
filled with NaN before the call, one prefix-cache checkpoint per sequence (the last 64-aligned offset), plus one case that
resumes from a carried non-zero state (`state@4096`). Checked: output, final states, checkpoint rows, untouched pool rows
and output bytes, `prefill_health()`. Pass: finite, every element within max(0.5 x RMS, 5 % relative) of fp64,
relative L2 < 1e-2, and relative L2 <= 2 x FLA `chunk_kda`'s + 1e-3 (the rule of the B200 KDA real gate).

**decode** (8 cases): the two 32-request batches, H = 96 and H = 12, plain decode (S = 1, 16 steps) and speculative
verify (S = 8, 5 steps, random accepted counts 1–8). The prefix of each request is run in fp64 and put into the pool, then
the request's next tokens go through `decode`. The captured q / k / v are post-conv-and-SiLU, so the decode input is
SiLU^-1 of them with a pass-through conv (tap 3 = 1); the fp64 gold uses exactly the post-SiLU values the kernel computes.
`out_gate` and `norm_weight` are random (not captured). Checked: output, every written state slot (after each verified
token), conv cache bit-exact against `reference.py`, `decode_health()`. Pass: finite, the elementwise rule above,
relative L2 < 1e-2.

**Result on MI350X** (`results_mi350x.txt`): 27 / 27 PASS. Prefill relative L2 vs fp64: output 0.0017–0.0028, state
0.0004–0.0016, checkpoints 0.0004–0.0013; FLA `chunk_kda` on the same inputs: output 0.0023–0.0035. Decode: output
0.0017–0.0018 (`reference.py` in fp32: 0.0016–0.0018), states ~3e-6.

```
HIP_VISIBLE_DEVICES=0 python benchmark/real_gate/amd_real_gate.py --repo . [--data <dir>] [--only prefill|decode]
```
Needs `flash-linear-attention` (FLA) for the prefill comparison (`requirements.txt`).
