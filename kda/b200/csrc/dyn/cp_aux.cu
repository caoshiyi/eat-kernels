// Small kernels of the graph-safe context-parallel path (geometry read on the device).
// geom int32[8]: [0] x = absolute row of the tail start, [1] m = number of tail pieces (0: no CP in this replay), [2] e = absolute end row of the tail, [3] j = cut sequence.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};
#include <cuda_bf16.h>

namespace {
// zero rows [e, min(e + 128, T)) of qt [T, H, 128] bf16: the fixup's last (partial) tile reads Qt rows beyond the tail; they must contribute exactly 0
__global__ void zero_qt_tail_kernel(__nv_bfloat16* __restrict__ qt, const int* __restrict__ geom, long long T, int H) {
  if (geom[1] == 0) return;
  const long long e = geom[2];
  const long long rows = min(128LL, T - e);
  const long long per_row = (long long)H * 128;   // bf16 elements per row
  uint4* base = (uint4*)(qt + e * per_row);
  const long long n16 = rows * per_row / 8;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) base[i] = make_uint4(0, 0, 0, 0);
}

// final_state[slot(j)] = sfix [H, 128, 128] fp32 (slot = dst_idx[j] or j; stride in floats)
__global__ void scatter_state_kernel(const float4* __restrict__ sfix, float4* __restrict__ final_state, const int* __restrict__ dst_idx, long long dst_stride,
                                     const int* __restrict__ geom, int H) {
  if (geom[1] == 0) return;
  const int j = geom[3];
  const long long slot = dst_idx ? (long long)dst_idx[j] : (long long)j;
  float4* dst = final_state + (slot * dst_stride) / 4;
  const long long n4 = (long long)H * 16384 / 4;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) dst[i] = sfix[i];
}

// [dynpath2] checkpoint rows that are plain copies of S_A (kind < 0) or a chained piece state S_in(kind) in a device-planned CP replay: row ck_cu[j] + rel of the checkpoint buffer.
// geom[10] = number of items (<= 2), items int2 (rel row, kind).  S_A = sa_base[idx[j]] (pool slot of the cut sequence: single-sequence head launch) or the scratch state sa.
// [bf16ckpt] CK = float4 (fp32 rows) or __nv_bfloat16 (rows rounded to nearest even)
__device__ __forceinline__ void ck_copy_row(float4* __restrict__ d, const float4* __restrict__ s) {
  for (int k = threadIdx.x; k < 4096; k += blockDim.x) d[k] = s[k];
}
__device__ __forceinline__ void ck_copy_row(__nv_bfloat16* __restrict__ d, const float4* __restrict__ s) {
  for (int k = threadIdx.x; k < 4096; k += blockDim.x) {
    const float4 f = s[k];
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(f.x, f.y), b1 = __floats2bfloat162_rn(f.z, f.w);
    reinterpret_cast<uint2*>(d)[k] = make_uint2(*(const uint32_t*)&b0, *(const uint32_t*)&b1);
  }
}
template <typename CK>
__global__ void copy_ck_dyn_kernel(const float4* __restrict__ sa, const float4* __restrict__ sa_base, const int* __restrict__ d_idx, long long d_stride, const float4* __restrict__ sin_,
                                   CK* __restrict__ ck, const long long* __restrict__ ck_cu, const int2* __restrict__ items, const int* __restrict__ geom, int H) {
  if (geom[1] == 0) return;
  const int i = blockIdx.x / H, h = blockIdx.x % H;
  if (i >= geom[10]) return;
  const int j = geom[3];
  const long long rb = ck_cu[j];
  if (rb < 0) return;
  const int2 it = items[i];
  const float4* s;
  if (it.y < 0) s = (sa_base ? sa_base + ((d_idx ? (long long)d_idx[j] : (long long)j) * d_stride) / 4 : sa) + (long long)h * 4096;
  else s = sin_ + (long long)it.y * H * 4096 + (long long)h * 4096;
  ck_copy_row(ck + ((rb + it.x) * H + h) * (sizeof(CK) == 16 ? 4096 : 16384), s);
}
}  // namespace

void zero_qt_tail(torch::Tensor qt, torch::Tensor geom, int64_t T, int64_t H) {
  KdaDevGuard dev_guard_(qt);   // [multi-GPU] launch on the tensors' device
  zero_qt_tail_kernel<<<(unsigned)(2 * 148), 256, 0, c10::cuda::getCurrentCUDAStream()>>>((__nv_bfloat16*)qt.data_ptr(), geom.data_ptr<int>(), T, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_state(torch::Tensor sfix, torch::Tensor final_state, c10::optional<torch::Tensor> dst_idx, int64_t dst_stride, torch::Tensor geom, int64_t H) {
  KdaDevGuard dev_guard_(sfix);   // [multi-GPU] launch on the tensors' device
  scatter_state_kernel<<<(unsigned)(2 * 148), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const float4*)sfix.data_ptr(), (float4*)final_state.data_ptr(), dst_idx ? dst_idx->data_ptr<int>() : nullptr, dst_stride, geom.data_ptr<int>(), (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the neighbouring kernels (planner, head / tail launches, fixup) run with the maximum shared-memory carveout; make these tiny kernels use the same split so the SMs are not
// reconfigured around them
void copy_ck_dyn(torch::Tensor sa, c10::optional<torch::Tensor> sa_base, c10::optional<torch::Tensor> d_idx, int64_t d_stride, torch::Tensor sin_, torch::Tensor ck, torch::Tensor ck_cu,
                 torch::Tensor items, torch::Tensor geom, int64_t H) {
  KdaDevGuard dev_guard_(sa);   // [multi-GPU] launch on the tensors' device
  if (ck.scalar_type() == at::kBFloat16)   // [bf16ckpt]
    copy_ck_dyn_kernel<__nv_bfloat16><<<(unsigned)(2 * H), 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const float4*)sa.data_ptr(), sa_base ? (const float4*)sa_base->data_ptr() : nullptr,
        d_idx ? d_idx->data_ptr<int>() : nullptr, d_stride, sin_.numel() ? (const float4*)sin_.data_ptr() : nullptr, (__nv_bfloat16*)ck.data_ptr(), (const long long*)ck_cu.data_ptr(),
        (const int2*)items.data_ptr(), geom.data_ptr<int>(), (int)H);
  else
  copy_ck_dyn_kernel<float4><<<(unsigned)(2 * H), 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const float4*)sa.data_ptr(), sa_base ? (const float4*)sa_base->data_ptr() : nullptr,
      d_idx ? d_idx->data_ptr<int>() : nullptr, d_stride, sin_.numel() ? (const float4*)sin_.data_ptr() : nullptr, (float4*)ck.data_ptr(), (const long long*)ck_cu.data_ptr(),
      (const int2*)items.data_ptr(), geom.data_ptr<int>(), (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void aux_init() {
  cudaFree(0);
  cudaFuncSetAttribute(zero_qt_tail_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(scatter_state_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(copy_ck_dyn_kernel<float4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(copy_ck_dyn_kernel<__nv_bfloat16>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &aux_init);
  m.def("zero_qt_tail", &zero_qt_tail);
  m.def("scatter_state", &scatter_state);
  m.def("copy_ck_dyn", &copy_ck_dyn);
}
