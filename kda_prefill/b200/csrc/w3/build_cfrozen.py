import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False):
    """contract_frozen.cu: table validation of the frozen one-sequence plans (validate_tables_frozen)."""
    d = os.path.join(_here, "build_cfrozen"); os.makedirs(d, exist_ok=True)
    return load(name="kda_v60_cfrozen", sources=[os.path.join(_here, "contract_frozen.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17"], build_directory=d, verbose=verbose)


if __name__ == "__main__":
    build(True)
