#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define uint8_t g_uint8_t
#define uint16_t g_uint16_t
#define uint32_t g_uint32_t
#define uint64_t g_uint64_t
#define int32_t g_int32_t
#define int16_t g_int16_t
#include "kda_m128.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
void launch_m128(int grid, cudaStream_t stream, void* q, void* k, void* v, void* g, void* beta, const unsigned char* mp,
               float* A_log, float* dt_bias, long long* cu, int* order, const int* task_list, const int* cta_ptr, float* scratch, int* flags, const float* s0, void* out, float* s1,
               int num_heads, float scale, float lower_bound) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    init = true;
  }
  kernel_flashkda_bf16_fused_m128<<<grid, THREADS, SMEM_TOTAL, stream>>>(
      (__nv_bfloat16*)q, mp, (__nv_bfloat16*)k, mp + 128, (__nv_bfloat16*)v, mp + 256, (__nv_bfloat16*)g, mp + 384,
      (__nv_bfloat16*)beta, mp + 512, A_log, dt_bias, cu, order, task_list, cta_ptr, scratch, flags, s0, (__nv_bfloat16*)out, mp + 640, s1, num_heads, 1,
      1, scale, lower_bound);
}
