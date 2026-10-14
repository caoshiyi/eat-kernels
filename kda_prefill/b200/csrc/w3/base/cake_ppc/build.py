import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False, feat=False, ck=False):
    flags = ["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"] + (["-DKDA_FEAT"] if feat else []) + (["-DKDA_CK"] if ck else [])
    d = os.path.join(_here, ("build_featck" if ck else "build_feat") if feat else "build")
    os.makedirs(d, exist_ok=True)
    srcs = ("binding.cu", "launch_m128.cu") if feat else ("binding.cu", "launch_m128.cu", "launch_m64.cu")
    return load(name=("kda_v45fc_cake_ppc" if ck else "kda_v45f_cake_ppc") if feat else "kda_v45_cake_ppc", sources=[os.path.join(_here, f) for f in srcs],
                extra_cuda_cflags=flags, extra_ldflags=["-lcuda"], build_directory=d, verbose=verbose)
if __name__ == "__main__":
    import sys
    build(False, feat="feat" in sys.argv or "featck" in sys.argv, ck="featck" in sys.argv)
