import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False):
    return load(name="kda_v41_cake_dual", sources=[os.path.join(_here, f) for f in ("binding.cu", "launch_dual.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"],
                extra_ldflags=["-lcuda"], build_directory=_ensure(os.path.join(_here, "build")), verbose=verbose)
def _ensure(d):
    os.makedirs(d, exist_ok=True); return d
if __name__ == "__main__":
    build(True)
