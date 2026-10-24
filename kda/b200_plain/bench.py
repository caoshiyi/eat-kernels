"""Benchmark kda_b200_int21 against FlashKDA on the six Int21 workloads (same harness and timing as ../b200/bench.py).

    CUDA_VISIBLE_DEVICES=0 python bench.py [--quick] [--seed N] [--json out.json]
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("kda_b200_bench", HERE.parent / "b200" / "bench.py")
_b200_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b200_bench)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 trial x 10 iterations")
    ap.add_argument("--seed", type=int, default=None, help="default: fresh random seed every run")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--json", default=None, help="write per-workload results to this file")
    args = ap.parse_args(argv)
    if args.quick:
        args.iters, args.trials = 10, 1
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little")

    import kda_b200_int21

    print(f"[OK] {torch.cuda.get_device_name()} | torch {torch.__version__} | seed {seed} | {args.trials} x {args.iters} iters")
    rows = {}
    gm, ok = _b200_bench.run_set("Int21 workload", _b200_bench.int21_case, kda_b200_int21, _b200_bench.INT21,
                                 seed + 1, args.warmup, args.iters, args.trials, rows)
    print("\n" + "=" * 64)
    print(f"  int21_geomean_vs_flashkda = {gm:.3f}")
    print(f"  correctness: {'all pass' if ok else 'FAILURES'}")
    print("=" * 64)
    if args.json:
        Path(args.json).write_text(json.dumps({"device": torch.cuda.get_device_name(), "seed": seed,
                                               "summary": {"int21_geomean_vs_flashkda": gm}, "all_pass": ok,
                                               "workloads": rows}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
