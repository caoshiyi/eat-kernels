"""Quick correctness check of kda/kernel.py against reference.py (no baselines needed, ~1 min after the JIT builds).

Runs, with the eval's own harness (benchmark/eval_kda_mi350x.py: page-layout pools, strided beta, output slice, NaN-poisoned
write-only rows, untouched-byte checks): two scored prefill workloads (and again after an in-place refill), three decode
workloads as captured HIP graphs (and again after a replay with new slot indices / data), and the integration gate.

    HIP_VISIBLE_DEVICES=0 python benchmark/smoke_test.py [--kernel kda/kernel.py] [--seed 1]
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=str(HERE.parent / "kda" / "kernel.py"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if "HIP_VISIBLE_DEVICES" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("set HIP_VISIBLE_DEVICES to one idle MI350X")
    spec = importlib.util.spec_from_file_location("kda_eval", HERE / "eval_kda_mi350x.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    import torch

    t0 = time.time()
    mod = ev.load_kernel(args.kernel)
    print(f"[OK] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | import + JIT {time.time() - t0:.1f} s")
    bad = 0
    with torch.inference_mode():
        for name in ("p12_loaded16k_1", "p96_light8k_0"):
            H, seqs = ev.PREFILL_WORKLOADS[name]
            case = ev.PrefillCase(name, H, seqs, args.seed)
            for rnd in range(2):
                if rnd:
                    case.refill(args.seed + 77)
                meta = mod.prefill_prepare(case.cu_cpu, H, case.ck_off_cpu, case.ck_rows_cpu)
                case.pages.restore(case.snap)
                mod.prefill(*case.args(), **case.ck_kwargs(), meta=meta)
                torch.cuda.synchronize()
                ok, eo, es, msg = case.verify()
                bad += not ok
                print(f"  prefill {name:<16} {'refill' if rnd else 'first ':6} {'PASS' if ok else 'FAIL'} "
                      f"(out {eo:.1e} pool {es:.1e}{' ' + msg if msg else ''})")
            del case
        t1 = time.time()
        warm = mod.decode_warmup()
        print(f"  decode_warmup {time.time() - t1:.1f} s ({'ok' if warm is not None else 'done'})")
        for name in ("s8_h12_b4", "s8_h12_b64", "d_h12_b64"):
            H, B, S = ev.DECODE_WORKLOADS[name]
            case = ev.DecodeCase(name, H, B, S, args.seed)
            g = case.capture(mod.decode)
            ok, msg = case.check(g.replay)
            if ok:
                case.new_data(args.seed + 11)
                ok, msg2 = case.check(g.replay)
                msg += f"; replay with new indices/data: {msg2}"
            bad += not ok
            print(f"  decode  {name:<16} {'PASS' if ok else 'FAIL'} ({msg})")
            del case, g
        bad += ev.gate_integration(mod, args.seed, print)
    print(f"\nsmoke test: {'PASS' if not bad else f'{bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
