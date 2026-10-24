# Contract: production KDA (prefill + decode) for Kimi-K3 serving on MI350X

The calling convention, shapes, dtypes, layouts and acceptance gates the package in `kda/` is built and validated
against. They mirror vLLM's ROCm Kimi-K3 KDA layer (`benchmark/vllm_reference/kda.py`): every workload is a step of a
real chunked-prefill / speculative-decode schedule (AgentX traces of Kimi-K3 agentic coding sessions; DSpark
speculative decoding), called with vLLM's exact tensor layouts, and timed against what vLLM ships today on the same
inputs. Implemented by `benchmark/eval_kda_mi350x.py`; how to run it: `benchmark/README.md`.

Earlier ("phase-1") versions of the two kernels were optimised on simplified interfaces: `amd_kda_mi350x` (chunked
KDA prefill on a contiguous per-sequence state tensor) and `amd_kda_decode_mi350x` (fused decode + speculative verify
on dense pools). Their workloads survive as the no-regression gate 7 below.

## The package: `kda/kernel.py`

```python
meta = prefill_prepare(cu_seqlens_cpu, num_heads, checkpoint_offsets=None, checkpoint_state_indices=None, device=None)
prefill(q, k, v, raw_g, raw_beta, A_log, dt_bias, scale, lower_bound, state_cache, state_indices, has_initial_state,
        cu_seqlens, out, checkpoint_offsets=None, checkpoint_state_indices=None, meta=None) -> None
decode(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
       state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out) -> None
```

Exact math: `reference.py` (`kda_prefill_reference`, `kda_decode_reference`). Head dim 128, fp32 state, bf16 activations.

### prefill_prepare
Called once per scheduler step on **host** tensors (`cu_seqlens_cpu` int32 [N+1], `checkpoint_offsets` /
`checkpoint_state_indices` int32 [N] on the CPU), its result is shared by every KDA layer of the model in that step
(69 layers for Kimi-K3). Not timed, but limited: median ≤ 1 ms + 5 µs per sequence on new layouts (gate). It may
build device tensors (one H2D copy is fine). It must derive metadata only from the layout (it never sees data).
`prefill(..., meta=None)` must also work (compute it inline).

### prefill (the chunked-prefill part of a step, one layer)
- `q, k, v, raw_g`: bf16 [1, T, H, 128] (q / k / v after the short conv; `raw_g` before the gate activation).
- `raw_beta`: bf16 [1, T, H] **strided**: a column slice of the layer's input projection (row stride Wp ≈ 4·H·128+…).
- `A_log` fp32 [H], `dt_bias` fp32 [H·128]; gate g = lower_bound · sigmoid(exp(A_log) · (raw_g + dt_bias)).
- `state_cache`: fp32 [slots, H, 128, 128] **strided pool view** (below), read and updated in place.
- `state_indices` int32 [N]: pool row of sequence n. `has_initial_state` bool [N]: start from `state_cache[row]`
  (resumed chunk / prefix-cache hit) or from zeros — then the row's content is garbage (the eval fills it with NaN)
  and must not be read. After the call `state_cache[state_indices[n]]` holds the state after the last token.
- `checkpoint_offsets[n] = o`, `checkpoint_state_indices[n] = r`: if o > 0 and r ≥ 0, also write the state after
  the first o tokens of sequence n (o is a multiple of 64, o ≤ length; o = length is common) to pool row r
  (prefix caching: vLLM's `KDA_CHECKPOINT_ALIGNMENT` = 64). Rows are distinct and never alias `state_indices`.
  Checkpoint rows are write-only (NaN before the call in the eval).
- `cu_seqlens` int32 [N+1] on the device (same values as `prepare`'s host copy).
- `out`: bf16 [1, T, H, 128] **a slice of a larger buffer** (decode tokens come first in vLLM's packed batch).
  Write only `out`; touch no pool byte other than the rows above (the conv-state part of each page and every other
  row must stay bit-identical).
- Runs **eagerly** (vLLM breaks the KDA op out of graph capture), so host launch overhead counts, and it must not
  synchronize with the device (checked with `torch.cuda.set_sync_debug_mode("error")`): no `.item()`, `.tolist()`,
  `.cpu()`, data-dependent shapes, `nonzero`, … inside `prefill`. Anything layout-dependent goes in `prefill_prepare`.

### decode (the decode / spec-verify part of a step, one layer)
Same contract as the phase-1 decode kernel (`amd_kda_decode_mi350x`; see `kda/decode/` and the reference), with production layouts:
`mixed_qkv` [T, 3·H·128], `out_gate` [T, H·128] and `beta` [T, H] are **column slices of the input projection**
(strided rows), `gate` [T, H, 128] is contiguous, `conv_state` / `ssm_state` are the strided page views below.
B sequences × S tokens; S = 8 for Kimi-K3 DSpark (7 drafts, accepted length 4.38 on average: the eval draws 4 or 5
accepted tokens per sequence with P(5) = 0.38), S = 1 for plain decode. Must be HIP-graph capturable and correct
when the captured graph is replayed with new slot indices / accepted counts / data. Also called eagerly in the gates.

### The page layout (vLLM's hybrid KV allocator)
Each pool slot is one page holding the conv state followed by the SSM state; pages are padded:
```
page bytes P = roundup(taps·3·H·128·2 + H·128·128·4 + 4096, 256)
conv_state = as_strided(raw.bf16, [slots, taps, 3·H·128], [P/2, 3·H·128, 1], 0)
ssm_state  = as_strided(raw.fp32, [slots, H, 128, 128],   [P/4, 128·128, 128, 1], conv_bytes/4)
```
So the slot stride is not H·128·128 and rows are not adjacent. Slot 0 is reserved (never used, never written).
The gates also call both entry points with plain contiguous pools.

## Workloads and score
`mean_vs_base = sqrt(prefill_geomean × decode_geomean)`.

- **prefill** (13 workloads, geomean of vLLM latency / yours): chunked-prefill steps produced by
  `workloads/gen_prefill_batches.py --seed 1`, which replays AgentX session lengths (`workloads/agentx_lengths.npz`) through a
  chunked-prefill scheduler (token budget 8K or 16K, a prefill chunk resumes at 64-token boundaries, checkpoints every
  1024 tokens, decode tokens of the same step take part of the budget). `loaded` = full budget, `light` = 1–3
  requests. H = 12 (Kimi-K3 at TP8) and H = 96 (TP1 / attention-DP). Eager call, L2 / MALL flushed, inputs fixed
  within a workload. Baseline: vLLM's fused ROCm Kimi-K3 prefill (`benchmark/baselines/prefill`, the `fused_kda_prologue` +
  `fused_kda_chunk` ops of `chunk_kda_prefill`), which natively reads the pool, the strided beta and writes
  checkpoints, on the same inputs.
- **decode** (10 workloads, geomean of best-baseline latency / yours): H = 12, S = 8, B ∈ {1, 4, 16, 32, 64, 128};
  H = 12, S = 1, B = 64; H = 96, S = 8, B ∈ {1, 8, 32}. HIP-graph replay, cache flushed. Baseline: the faster of
  AITER `fused_kda_decode` and vLLM's HIP decode (`benchmark/baselines/decode`), with the same layouts.

Baselines are timed live, interleaved with the candidate. Fresh random data every run.

## Hard gates (any failure → score 0)
1. **Correctness on every workload** — per-sequence output and every written pool row within relative L2 2e-2 of the
   fp32 reference (decode: output 2e-2, states 1e-2, conv cache bit-exact), every other byte of the page buffer and of
   the output buffer unchanged — and again after all inputs and the pool are refilled **in place** (no caching of
   anything derived from data) / after the decode graph is replayed with new indices and data.
2. **Prefill contract**: H ∈ {1, 4, 8, 16, 24, 32, 48, 128} (not only 12 / 96), ragged batches with 1-token and
   partial-chunk sequences, 256 sequences, a 64K-token sequence, tensors over 2^31 elements, `meta=None`, contiguous
   pools; host: no device sync inside `prefill`, `prefill_prepare` latency.
3. **Numerics vs an fp64 recurrence** (vLLM's own error is printed next to yours): very slow decay over 16K tokens,
   a spectrum of decay rates, saturated gates, trained-deep-layer statistics (aligned q/k/v within chunks, beta → 1,
   slow decay, large states), one-hot keys, rescaled inputs.
4. **Decode contract**: 17 held-out shapes (H 8–96, S 1–8, B 1–512), page and contiguous layouts, graph re-replay,
   eager calls.
5. **Integration**: prefill → spec verify ×3 → next-turn prefill resuming from the accepted slot → a new request
   resuming from a checkpoint row, all on one pool.
6. **Co-tenant**: prefill and decode must finish (correctly, in < 1.5 s) while a kernel on another stream occupies 3/4
   of the CUs and all their LDS for 4 s. A design that needs all its workgroups co-resident (inter-workgroup
   spin-waits, persistent relays without a fallback) can deadlock or stall in a real server (other streams, NCCL /
   RCCL kernels, a second model on the GPU) and fails here.
7. **No regression on phase 1**: the 10 `amd_kda_mi350x` prefill shapes and the 17 `amd_kda_decode_mi350x` decode
   shapes run through this package (contiguous state, their metrics and recorded baselines) must stay ≥ the floors
   printed by the eval (95 % of the starting package of the production work; recorded baselines in `workloads/legacy_*_baseline_sota.json`).

## Score and its reading
- Only a full run of the eval without options prints an official `mean_vs_base`; `--quick`, `--part`, `--workloads`,
  `--gates`, `--no-gates`, `--iters`, `--trials` print partial numbers.
- The serving workloads are a sample of a distribution; dispatch in the package follows a cost model over the layout
  (N, lengths, H, B, S, checkpoint positions), never workload names or exact shapes. Held-out prefill compositions
  can be generated with `workloads/gen_prefill_batches.py --seed <other>` and run (unscored) with `--heldout <file>`
  (a file holding a dict like `PREFILL_WORKLOADS`).
