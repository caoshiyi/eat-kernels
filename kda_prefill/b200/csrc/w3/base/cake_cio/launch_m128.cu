#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#include <cuda_bf16.h>
#include "../ck_type.cuh"
#include <cuda_runtime.h>
#define uint8_t g_uint8_t
#define uint16_t g_uint16_t
#define uint32_t g_uint32_t
#define uint64_t g_uint64_t
#define int32_t g_int32_t
#define int16_t g_int16_t
#ifdef KDA_FEAT
#define FEAT_DECL , const int* src_idx, long long src_stride, const int* dst_idx, long long dst_stride, ck_t* ck_base, const long long* ck_cu, int ck_n
#define FEAT_ARGS , src_idx, src_stride, dst_idx, dst_stride, ck_base, ck_cu, ck_n
#else
#define FEAT_DECL
#define FEAT_ARGS
#endif
#include "kda_m128.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
void launch_m128_cio(int grid, cudaStream_t stream, void* q, void* k, void* v, void* g, void* beta, const unsigned char* mp,
               float* A_log, float* dt_bias, long long* cu, int* order, const int* task_list, const int* cta_ptr, float* scratch, int* flags, const float* s0, void* out, float* s1,
               int num_heads, float scale, float lower_bound FEAT_DECL) {
  static bool init[64] = {};   // per device: the attribute is a per-device property (audit F3)
  int dev_ = 0; cudaGetDevice(&dev_);
  if (!init[dev_ & 63]) {
    cudaFuncSetAttribute(kernel_kda_m128_cio, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    init[dev_ & 63] = true;
  }
  kernel_kda_m128_cio<<<grid, THREADS, SMEM_TOTAL, stream>>>(
      (__nv_bfloat16*)q, mp, (__nv_bfloat16*)k, mp + 128, (__nv_bfloat16*)v, mp + 256, (__nv_bfloat16*)g, mp + 384,
      (__nv_bfloat16*)beta, mp + 512, A_log, dt_bias, cu, order, task_list, cta_ptr, scratch, flags, s0, (__nv_bfloat16*)out, mp + 640, s1, num_heads, 1,
      1, scale, lower_bound FEAT_ARGS);
}
