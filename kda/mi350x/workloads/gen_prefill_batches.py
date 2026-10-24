"""Prefill batch compositions for benchmark/eval_kda_mi350x.py, drawn from the AgentX v1.0 256k traces.

agentx_lengths.npz (from semianalysisai/cc-traces-weka-062126-256k, infinite prefix cache, 64-token block
hashes, subagents start from their parent's cache): per request U = uncached suffix tokens, CP = cached prefix tokens.

Emulates vLLM's chunked-prefill scheduler on one engine: each step has a token budget (8192 or 16384); decode tokens
of the step come first and are served by the decode kernel, the rest of the budget goes to prefill. A request whose
suffix does not fit is cut at a 64-token boundary and resumed in the next step from its pool slot (has_initial_state).
A new request starts from its cached prefix state if CP > 0, else from zeros. Prefix-cache checkpoints (vLLM
mamba_cache_mode="align"): one per sequence and step, at the last absolute multiple of BLOCK tokens inside the chunk.
"Loaded" steps fill the budget; "light" steps carry only the 1-3 requests that arrived.

    python gen_prefill_batches.py [--seed S] [--count-per-kind N]   -> prints a Python literal
"""
import argparse
import os

import numpy as np

BLOCK = 1024
MAX_SEQS = 128


def sample_request(rng, U, CP):
    while True:
        i = int(rng.integers(len(U)))
        u, cp = int(U[i]), int(CP[i])
        if u > 0:
            return [u, cp, 0]  # remaining suffix, cached prefix, tokens of the suffix already computed


def chunk_entry(req, take):
    u, cp, done = req
    start = cp + done
    end = start + take
    b = (end // BLOCK) * BLOCK
    ck = b - start if b > start else 0
    has_init = (cp + done) > 0
    return (take, int(has_init), ck)


def loaded_step(rng, U, CP, budget, state):
    left = budget - int(rng.choice([0, 128, 256, 512]))
    batch = []
    while left >= 64 and len(batch) < MAX_SEQS:
        req = state.pop("inprog", None) or sample_request(rng, U, CP)
        take = req[0]
        if take > left:
            take = (left // 64) * 64
        batch.append(chunk_entry(req, take))
        left -= take
        req[0] -= take
        req[2] += take
        if req[0] > 0:
            state["inprog"] = req
            break
    return batch


def light_step_budget(rng, U, CP, budget):
    batch, left = [], budget
    for _ in range(int(rng.integers(1, 4))):
        r = sample_request(rng, U, CP)
        take = min(r[0], (left // 64) * 64)
        if take <= 0:
            break
        batch.append(chunk_entry(r, take))
        left -= take
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--npz", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentx_lengths.npz"))
    ap.add_argument("--plan", default="12:loaded:8192:3,12:loaded:16384:3,12:light:8192:2,12:light:16384:2,"
                                      "96:loaded:8192:2,96:light:8192:1")
    args = ap.parse_args()
    d = np.load(args.npz)
    U, CP = d["U"], d["CP"]
    rng = np.random.default_rng(args.seed)
    out = {}
    for item in args.plan.split(","):
        H, kind, budget, count = item.split(":")
        H, budget, count = int(H), int(budget), int(count)
        state = {}
        for _ in range(int(rng.integers(5, 40))):  # warm the in-progress request
            loaded_step(rng, U, CP, budget, state)
        for j in range(count):
            if kind == "loaded":
                for _ in range(int(rng.integers(3, 30))):
                    loaded_step(rng, U, CP, budget, state)
                b = loaded_step(rng, U, CP, budget, state)
            else:
                b = light_step_budget(rng, U, CP, budget)
            out[f"p{H}_{kind}{budget // 1024}k_{j}"] = (H, b)
    print("{")
    for name, (H, b) in out.items():
        print(f'    "{name}": ({H}, {b}),  # N={len(b)} T={sum(x[0] for x in b)}')
    print("}")


if __name__ == "__main__":
    main()
