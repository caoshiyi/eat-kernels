// [bf16ckpt] element type of the checkpoint buffer: fp32 (default builds, unchanged) or bf16 (-DKDA_CKBF: every row is the fp32 row rounded to nearest even).
#pragma once
#include <cuda_bf16.h>
#ifdef KDA_CKBF
typedef __nv_bfloat16 ck_t;
// [bf16ckpt] two fp32 bit patterns -> bf16x2 (round to nearest even; lo = first element)
__device__ __forceinline__ unsigned int ck_pack_bf16(unsigned int lo, unsigned int hi) {
    unsigned int d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(__uint_as_float(hi)), "f"(__uint_as_float(lo)));
    return d;
}
#else
typedef float ck_t;
#endif
