import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False, feat=False, ck=False, dyn_exit=False, ck_dev=False, phl=False, noout=False):
    xf = ["-DKDA_NOOUT1", "-DKDA_NOOUT2", "-DKDA_NOOUT3", "-DKDA_NOOUT4"] if noout else []
    xt = ("_x" + "".join(c if c.isalnum() else "_" for c in "_".join(xf))) if xf else ""
    flags = xf + ["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"] + (["-DKDA_CK"] if (ck or ck_dev) else []) + (["-DKDA_CK_DEV"] if ck_dev else []) + (["-DKDA_DYN_EXIT"] if (dyn_exit or ck_dev) else []) + (["-DKDA_P_HILO"] if phl else [])
    d = os.path.join(_here, ("build_phl" if phl else ("build_ck" if ck else ("build_ckdev" if ck_dev else ("build_dynexit" if dyn_exit else "build")))) + xt)
    os.makedirs(d, exist_ok=True)
    return load(name=("kda_v50p_cake_dual" if phl else "kda_v45c_cake_dual" if ck else ("kda_v46k_cake_dual" if ck_dev else ("kda_v46x_cake_dual" if dyn_exit else "kda_v41_cake_dual"))) + xt, sources=[os.path.join(_here, f) for f in ("binding.cu", "launch_dual.cu")],
                extra_cuda_cflags=flags, extra_ldflags=["-lcuda"], build_directory=d, verbose=verbose)
if __name__ == "__main__":
    import sys
    build(False, ck="ck" in sys.argv)
