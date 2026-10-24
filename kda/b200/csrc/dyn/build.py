import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False, src="plan_cp.cu", name="kda_pg_plan_cp"):
    d = os.path.join(_here, "build_" + name); os.makedirs(d, exist_ok=True)
    return load(name=name, sources=[os.path.join(_here, src)],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-lineinfo"],
                build_directory=d, verbose=verbose)


if __name__ == "__main__":
    build(True)
