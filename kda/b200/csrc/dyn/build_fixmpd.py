import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False):
    d = os.path.join(_here, "build_fixmpd"); os.makedirs(d, exist_ok=True)
    return load(name="kda_pg_fixmpd", sources=[os.path.join(_here, "cp_fix_mpd.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-lineinfo"], extra_ldflags=["-lcuda"],
                build_directory=d, verbose=verbose)


if __name__ == "__main__":
    build(True)
