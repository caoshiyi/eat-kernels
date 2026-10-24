import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
def build(verbose=False, ck=False, bf=False):
    """ck: cp_fix_ck.cu = cp_fix.cu + checkpoint items (tools/patch_fix.py); the default fixup (cp_fix.cu) is the v44 kernel, unchanged."""
    d = os.path.join(_here, ("build_fix_ck" + ("_bf" if bf else "")) if ck else "build_fix"); os.makedirs(d, exist_ok=True)
    return load(name=("kda_v45c_fix" + ("_bf" if bf else "")) if ck else "kda_v41_fix", sources=[os.path.join(_here, "cp_fix_ck.cu" if ck else "cp_fix.cu")],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-lineinfo"] + (["-DKDA_CKBF"] if (ck and bf) else []),
                build_directory=d, verbose=verbose)
if __name__ == "__main__":
    import sys
    build(True, ck="ck" in sys.argv)
