import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False, feat=False, ck=False, bf=False, nopart=False):
    flags = (["-DKDA_NOPART"] if nopart else []) + ["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--use_fast_math", "-lineinfo"] + (["-DKDA_FEAT"] if feat else []) + (["-DKDA_CK"] if ck else []) + (["-DKDA_CKBF"] if bf else [])
    xt = ("_bf" if bf else "") + ("_np" if nopart else "")
    d = os.path.join(_here, ("build_featck" if ck else "build_feat") if feat else "build") + xt
    os.makedirs(d, exist_ok=True)
    return load(name=xt + (("kda_v45fc_cake_cio" if ck else "kda_v45f_cake_cio") if feat else "kda_v45_cake_cio"), sources=[os.path.join(_here, f) for f in ("binding.cu", "launch_m128.cu")],
                extra_cuda_cflags=flags, extra_ldflags=["-lcuda"], build_directory=d, verbose=verbose)


if __name__ == "__main__":
    import sys
    build(False, feat="feat" in sys.argv or "featck" in sys.argv, ck="featck" in sys.argv)
