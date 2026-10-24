# Workloads

The workloads are defined in code: literal tables at the top of `benchmark/eval_kda_mi350x.py` (authoritative). Tensor
data is random and generated per run from a seed (`--seed`; default random); only the batch layouts are fixed.
`workloads.json` holds the same tables as data (regenerate / check with `python workloads/export_workloads.py [--check]`).

| table | what | count |
|---|---|---|
| `PREFILL_WORKLOADS` | scored chunked-prefill steps: `name -> (H, [(new tokens, has_initial_state, checkpoint offset), ...])` | 13 |
| `DECODE_WORKLOADS` | scored decode / speculative-verify steps: `name -> (H, B, S)` | 10 |
| `QUICK_PREFILL`, `QUICK_DECODE` | `--quick` subsets | 4 + 4 |
| `LEGACY_PREFILL` | phase-1 prefill no-regression gate: sequence lengths, H = 12, contiguous state | 10 |
| `LEGACY_DECODE` | phase-1 decode no-regression gate: `(H, B, S)`, dense pools | 17 |
| `DECODE_HELDOUT` | decode contract gate: held-out `(H, B, S)` | 17 |
| `NUMERIC_PROBES` | numerics gate vs an fp64 recurrence: layouts + input statistics (slow decay, saturated gates, aligned keys, ...) | 7 |

The prefill contract gate's cases (head counts 1-128, 64 ragged sequences, 256 sequences, a 64K-token sequence,
> 2^31-element tensors, `meta=None`, contiguous pool) are built in `gate_prefill_contract()` of the eval.

**Prefill steps** (`p<H>_<loaded|light><budget>k_<i>`): produced by `gen_prefill_batches.py --seed 1` (the output equals
`PREFILL_WORKLOADS`). It replays per-request token counts of the AgentX v1.0 256k traces (`agentx_lengths.npz`: `U` =
uncached suffix tokens, `CP` = cached prefix tokens per request; from the Hugging Face dataset
`semianalysisai/cc-traces-weka-062126-256k`, infinite prefix cache, 64-token block hashes) through a chunked-prefill
scheduler: token budget 8K or 16K, a chunk that does not fit is cut at a 64-token boundary and resumed in the next step
(`has_initial_state`), new requests start from their cached prefix (or zeros), one prefix-cache checkpoint per sequence
at the last multiple of 1024 tokens inside the chunk (vLLM `mamba_cache_mode="align"`), decode tokens of the step take
part of the budget. `loaded` = full budget, `light` = 1-3 requests. H = 12 is Kimi-K3 at TP8, H = 96 at TP1 /
attention-DP. Other seeds give held-out compositions:
```bash
python workloads/gen_prefill_batches.py --seed 7 > my_batches.txt
HIP_VISIBLE_DEVICES=0 python benchmark/eval_kda_mi350x.py --heldout my_batches.txt
```

**Decode steps** (`s8_h<H>_b<B>` = speculative verify, `d_h12_b64` = plain decode): Kimi-K3 DSpark with 7 drafts
(S = 8); accepted tokens per sequence 4 or 5 with P(5) = 0.38 (mean 4.38, SPEED-Bench coding); batch sizes over the
AgentX concurrency range; H = 12 and 96. Captured as HIP graphs and replayed.

**Shared layout** (both phases): vLLM's hybrid-allocator page pool, one page per slot = conv state then SSM state,
padded (`CONTRACT.md`, "The page layout"); `beta` / `out_gate` / `mixed_qkv` are column slices of the input projection;
the prefill output is a slice of a larger buffer (256 decode tokens first).

**Recorded baselines** for the phase-1 gate: `legacy_prefill_baseline_sota.json` (vLLM prefill, ms) and
`legacy_decode_baseline_sota.json` (best of AITER / vLLM decode, µs, per baseline too), measured on the development
MI350X.
