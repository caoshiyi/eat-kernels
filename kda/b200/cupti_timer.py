# Adapted from FlashInfer (flashinfer-python 0.6.13, flashinfer/testing/utils.py: bench_gpu_time_with_cupti),
# Copyright (c) 2023 by FlashInfer team, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Reduced to the single-process CUPTI path used by bench.py; the
# measurement protocol is unchanged.
"""Per-iteration GPU time from CUPTI activity records: cold L2 (a 2 x L2-size buffer is zeroed before every iteration),
optionally the call captured in a CUDA graph, time = first kernel start to last kernel end of the iteration."""
import bisect
from functools import partial

import torch


def bench_gpu_time_with_cupti(fn, dry_run_iters, repeat_iters, cold_l2_cache=True, use_cuda_graph=False):
    """Returns the list of per-iteration GPU times (ms). Requires cupti-python >= 13."""
    from importlib.metadata import version

    from cupti import cupti

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("cupti-python >= 13 is required (pip install -U cupti-python)")

    buffer = None
    if cold_l2_cache:
        l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        buffer = torch.empty((l2 * 2) // (1024 * 1024) * 1024 * 1024, device="cuda", dtype=torch.int8)

    runner = fn
    if use_cuda_graph:
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        runner = g.replay

    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    for _ in range(5 + dry_run_iters):   # 5 = the original's iteration-time estimate runs
        if buffer is not None:
            buffer.zero_()
        runner()
    torch.cuda.synchronize()

    launches, kernels, iters = [], [], []

    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(launches, kernels, activities):
        for a in activities:
            if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels.append((a.name, a.start, a.end, a.correlation_id, 0, 0, 0, a.kind))
            elif a.kind == cupti.ActivityKind.MEMCPY:
                kernels.append(("MEMCPY", a.start, a.end, a.correlation_id, a.copy_kind, a.bytes, 0, a.kind))
            elif a.kind == cupti.ActivityKind.MEMSET:
                kernels.append(("MEMSET", a.start, a.end, a.correlation_id, 0, a.bytes, a.value, a.kind))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id, a.cbid, a.kind))

    kinds = (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.DRIVER,
             cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET)
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(buffer_requested, partial(buffer_completed, launches, kernels))
    for _ in range(repeat_iters):
        if buffer is not None:
            buffer.zero_()
        torch.cuda.synchronize()
        t0 = cupti.get_timestamp()
        runner()
        t1 = cupti.get_timestamp()
        torch.cuda.synchronize()
        iters.append((t0, t1))
    cupti.activity_flush_all(0)
    for kind in kinds:
        cupti.activity_disable(kind)
    cupti.finalize()

    launches.sort(key=lambda r: r[0])
    starts = [r[0] for r in launches]
    by_corr = {}
    for k in kernels:
        by_corr.setdefault(k[3], []).append(k)
    times, names = [], None
    for i, (t0, t1) in enumerate(iters):
        corr = {launches[j][2] for j in range(bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1))}
        ks = [k for c in corr for k in by_corr.get(c, ())]
        if not ks:
            raise ValueError(f"no kernel activity recorded for iteration {i}")
        cur = {f"{k[0]}_{k[4]}_{k[5]}_{k[6]}_{k[7]}" for k in ks}
        if names is None:
            names = cur
        elif names != cur:
            raise ValueError(f"inconsistent kernels across iterations: {names} != {cur}")
        times.append((max(k[2] for k in ks) - min(k[1] for k in ks)) / 1e6)
    return times
