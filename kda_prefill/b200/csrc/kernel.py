"""KDA forward (chunked prefill) for B200: internal entry of the kernel tree (use ../kda_b200.py).

 * run() and prepare(dynamic_layout=True): device-planned, graph-safe (dyn/: one planner kernel, cio / plain head launch, device-planned context-parallel tail).
   dynamic_layout=True: one sequence (no checkpoints) -> the host cost-model plan (static path; layout fixed by the shape), no contract feature -> default builds,
   <= 256 sequences -> lean planner kernels (dyn/plan_cp.cu); run() keeps the device planner only (host-latency limits).
 * prepare(dynamic_layout=False): the static host-planned path; its planners run in C++ (host/hostplan.cpp, identical plans to the Python planners in w3/).
 * Device-side validation of every caller table (cu_seqlens, slot tables, checkpoint rows; policy in w3/kernel.py), full-line checkpoint stores.
 * Every inter-CTA wait, chain-kernel mbarrier wait and the per-CTA lifetime watchdog traps after KDA_BOUND_NS = 60 s (a lost hand-off becomes
   `unspecified launch failure` instead of a hang)."""
import os, sys, functools, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
_W = os.path.join(_HERE, "w3")
sys.path.insert(0, _W)
spec = importlib.util.spec_from_file_location("kda_pg_w3", os.path.join(_W, "kernel.py")); _w1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(_w1)

try:
    _hs = importlib.util.spec_from_file_location("kda_pg_hostplan_build", os.path.join(_HERE, "host", "build.py"))
    _hm = importlib.util.module_from_spec(_hs); _hs.loader.exec_module(_hm)
    _hp = _hm.build()

    @functools.lru_cache(maxsize=256)
    def _schedule(seq_lens, H, n_sms):
        return _hp.schedule(list(seq_lens), H, n_sms)

    @functools.lru_cache(maxsize=256)
    def _cp_plan(seq_lens, H, n_sms, ck_n):
        return _hp.cp_plan(list(seq_lens), H, n_sms, ck_n)

    @functools.lru_cache(maxsize=256)
    def _plain_pieces(seq_lens, H, task_list, cta_ptr):
        return _hp.plain_pieces(list(seq_lens), H, list(task_list), list(cta_ptr))

    _b = _w1._base
    _py_schedule, _py_plain_pieces, _py_cp_plan = _b.schedule, _b.plain_pieces, _w1.cp_plan
    _b.schedule = lambda seq_lens, H, n_sms, feat=False: _schedule(tuple(int(x) for x in seq_lens), int(H), int(n_sms))
    _b.plain_pieces = lambda seq_lens, H, sched: _plain_pieces(tuple(int(x) for x in seq_lens), int(H), tuple(sched[0]), tuple(sched[1]))
    _w1.cp_plan = lambda seq_lens, H, n_sms, ck_n=0: _cp_plan(tuple(int(x) for x in seq_lens), int(H), int(n_sms), int(ck_n))
except Exception as e:   # no compiler / build failure: the Python planners stay
    print("kda: C++ host planner unavailable (%s: %s); using the Python planners" % (type(e).__name__, str(e)[:120]))

prepare = _w1.prepare
cp_plan = _w1.cp_plan
_base = _w1._base


def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None, **prod):
    prod["dynamic_layout"] = True
    prod.setdefault("dynamic_fallback", True)   # [contract2] more sequences than the device planner holds: host-planned path
    prod["_host_plan_ok"] = False     # run() is held to host-latency limits on a never-seen layout: device planner only (no host cost-model plan)
    return _w1.prepare(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, **prod)()
