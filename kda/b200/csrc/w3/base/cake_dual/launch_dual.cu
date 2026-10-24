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
#ifdef KDA_CK_DEV
#define CK_DECL , float* snapL, float* snapP, int ck_n, const int* ck_geom
#define CK_ARGS , snapL, snapP, ck_n, ck_geom
#elif defined(KDA_CK)
#define CK_DECL , float* snapL, float* snapP, int ck_n, int ck_base_chunk, int ck_k0
#define CK_ARGS , snapL, snapP, ck_n, ck_base_chunk, ck_k0
#else
#define CK_DECL
#define CK_ARGS
#endif
#ifdef KDA_P_HILO
#define PHL_DECL , long long p_plane
#define PHL_ARGS , p_plane
#else
#define PHL_DECL
#define PHL_ARGS
#endif
#include "kda_m128.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
void launch_dual(int grid, cudaStream_t stream, void* q, void* k, void* v, void* g, void* beta, const unsigned char* mp,
               float* A_log, float* dt_bias, long long* cu, int* order, const int* task_list, const int* cta_ptr, const float* s0, void* out, float* s1,
               const float* s0T, float* s1T, void* outT, int l_init,
               int num_heads, float scale, float lower_bound CK_DECL PHL_DECL) {
  static bool init[64] = {};   // per device: the attribute is a per-device property (audit F3)
  int dev_ = 0; cudaGetDevice(&dev_);
  if (!init[dev_ & 63]) {
    cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    init[dev_ & 63] = true;
  }
  kernel_flashkda_bf16_fused_m128<<<grid, THREADS, SMEM_TOTAL, stream>>>(
      (__nv_bfloat16*)q, mp, (__nv_bfloat16*)k, mp + 128, (__nv_bfloat16*)v, mp + 256, (__nv_bfloat16*)g, mp + 384,
      (__nv_bfloat16*)beta, mp + 512, A_log, dt_bias, cu, order, task_list, cta_ptr, s0, (__nv_bfloat16*)out, mp + 640, s1, s0T, s1T, (__nv_bfloat16*)outT, mp + 768, num_heads, l_init,
      1, scale, lower_bound CK_ARGS PHL_ARGS);
}

