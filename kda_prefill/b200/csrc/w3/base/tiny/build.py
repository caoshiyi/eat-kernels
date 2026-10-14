import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False):
    flags = ["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"]
    d = os.path.join(_here, "build"); os.makedirs(d, exist_ok=True)
    return load(name="kda_v49_tiny", sources=[os.path.join(_here, "kda_tiny.cu")], extra_cuda_cflags=flags, build_directory=d, verbose=verbose)
if __name__ == "__main__":
    build(True)
