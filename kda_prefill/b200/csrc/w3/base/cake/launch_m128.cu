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
#ifdef KDA_FEAT
#define FEAT_DECL , const int* src_idx, long long src_stride, const int* dst_idx, long long dst_stride, float* ck_base, const long long* ck_cu, int ck_n
#define FEAT_ARGS , src_idx, src_stride, dst_idx, dst_stride, ck_base, ck_cu, ck_n
#else
#define FEAT_DECL
#define FEAT_ARGS
#endif
#ifdef KDA_RWAIT
#define RW_DECL , int* rwait
#define RW_ARGS , rwait
#else
#define RW_DECL
#define RW_ARGS
#endif
#include "kda_m128.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

thread_local int g_pdl_launch = 0; thread_local bool g_ee_launch = false;   // [threads] thread_local: a concurrent launcher thread must not flip another thread's mode between its set_* and its launch
//    // g_pdl_launch: 0 = none, 1 = PDL behind a kernel that does not write the task tables (identity plan), 2 = PDL behind the planner (tables written by the predecessor); both defer the first-state prefetch to after griddepcontrol.wait (the sanitized slot table is written by the predecessor: a pre-wait read is the previous launch's copy, a benchmark artifact)
void set_pdl_launch(int mode) { g_pdl_launch = mode; }
void set_ee_launch(bool on) { g_ee_launch = on; }
void launch_m128(int grid, cudaStream_t stream, void* q, void* k, void* v, void* g, void* beta, const unsigned char* mp,
               float* A_log, float* dt_bias, long long* cu, int* order, const int* task_list, const int* cta_ptr, const float* s0, void* out, float* s1,
               int num_heads, float scale, float lower_bound FEAT_DECL RW_DECL) {
  static bool init[64] = {};   // per device: the attribute is a per-device property (audit F3)
  int dev_ = 0; cudaGetDevice(&dev_);
  if (!init[dev_ & 63]) {
    cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    init[dev_ & 63] = true;
  }
  const int pdl_mode_ = g_pdl_launch; const bool ee_ = g_ee_launch;   // one read each: the launch config and the kernel flag must agree
  const bool pdl = pdl_mode_ != 0;   // [pdl] programmatic dependent launch behind the validation kernel (identity plans only: the task tables are static, see base/kernel.py)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM_TOTAL; cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kernel_flashkda_bf16_fused_m128,
      (__nv_bfloat16*)q, (const void*)mp, (__nv_bfloat16*)k, (const void*)(mp + 128), (__nv_bfloat16*)v, (const void*)(mp + 256), (__nv_bfloat16*)g, (const void*)(mp + 384),
      (__nv_bfloat16*)beta, (const void*)(mp + 512), A_log, dt_bias, cu, order, task_list, cta_ptr, s0, (__nv_bfloat16*)out, (const void*)(mp + 640), s1, num_heads, 1,
      (ee_ ? 2 : 0) | (pdl ? 4 : 0) | 1, scale, lower_bound FEAT_ARGS RW_ARGS);   // store_final_state: bit 1 = a CTA without tasks exits at once
}


