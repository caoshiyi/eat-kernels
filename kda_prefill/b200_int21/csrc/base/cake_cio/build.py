import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False):
    os.makedirs(os.path.join(_here, "build"), exist_ok=True)
    return load(name="kda_v41_cake_cio", sources=[os.path.join(_here, f) for f in ("binding.cu", "launch_m128.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"],
                extra_ldflags=["-lcuda"], build_directory=os.path.join(_here, "build"), verbose=verbose)


if __name__ == "__main__":
    build(True)
