import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False):
    d = os.path.join(_here, "build_hostplan"); os.makedirs(d, exist_ok=True)
    return load(name="kda_hostplan", sources=[os.path.join(_here, "hostplan.cpp")], extra_cflags=["-O2", "-std=c++17"], build_directory=d, verbose=verbose)


if __name__ == "__main__":
    build(True)
