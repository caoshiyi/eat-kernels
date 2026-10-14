import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False, feat=False, ck=False, bd=False, rw=False):
    # [ckrow] checkpoint row 0 (= the initial state) is copied by the idle warps 8 / 11 instead of the compute warps (non-RWAIT checkpoint builds)
    r0 = bool(feat and ck and not rw)
    flags = (["-DKDA_QEARLY"] if feat else []) + ["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"] + (["-DKDA_FEAT"] if feat else []) + (["-DKDA_CK"] if ck else []) + (["-DKDA_BETA_DIRECT"] if bd else []) + (["-DKDA_RWAIT"] if rw else []) + (["-DKDA_ROW0"] if r0 else [])
    tag = "%d%d%d%d" % (feat, ck, bd, rw) + ("q" if feat else "")
    d = os.path.join(_here, "build_q" + tag) if feat else os.path.join(_here, "build_bd" if bd else "build")
    os.makedirs(d, exist_ok=True)
    return load(name=("kda_q%s_cake" % tag) if feat else ("kda_v45b_cake" if bd else "kda_v45_cake"), sources=[os.path.join(_here, f) for f in ("binding.cu", "launch_m128.cu", "launch_m64.cu")],
                extra_cuda_cflags=flags, extra_ldflags=["-lcuda"], build_directory=d, verbose=verbose)
if __name__ == "__main__":
    import sys
    build(False, feat="feat" in sys.argv or "featck" in sys.argv, ck="featck" in sys.argv)
