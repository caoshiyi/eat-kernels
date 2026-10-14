// [contract2] Device-side table validation and slot-indexed state copies for the production-contract path.
//
// validate_tables: ONE single-block launch at the head of every contract launch (graph-safe: reads only device memory, no host sync).  It reads the caller's
// cu_seqlens / state-slot tables / checkpoint_cu_starts (int32 or int64) and writes sanitized private copies that every later kernel uses instead:
//   cu_seqlens           -> clamped to a non-decreasing sequence inside [0, T] (cu[0] = 0): the kernels can never index outside q/k/v/g/out
//   dst slot (pool)      -> out of range (< 0 or >= P) or a repeated slot (2nd and later occurrence) is redirected to a private "trash" slot (relocatable: the
//                           trash window sits at base + slot * stride for an integer slot, see _trash_slot in kernel.py), so the pool is never touched
//   src slot (source)    -> out of range -> trash slot (read side; repeated source slots are legal: committed states are shared read-only)
//   checkpoint_cu_starts -> a sequence whose rows [ck_cu, ck_cu + ceil(len / n)) do not fit in [0, C) gets -1; the KDA_CK kernels skip its checkpoints
// and ORs a bit mask into `status` (stored, not accumulated: it describes the last launch): 1 bad cu_seqlens, 2 bad / repeated dst slot, 4 bad src slot, 8 bad checkpoint rows.
// A sequence with a rejected table entry gets undefined (finite) outputs and states; every other sequence is bit-identical to a launch without the bad entry.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};

#include "validate_dev.cuh"
using kdaval::VArgs;
namespace {
__global__ void __launch_bounds__(kdaval::kThreads) validate_kernel(VArgs a, int mode) {   // mode bit 0: latency-optimised path, bit 1: a rejected cu_seqlens = no-op launch (all sequences empty, output zero-filled)
  extern __shared__ int hs[];
  asm volatile("griddepcontrol.launch_dependents;");   // [pdl] a PDL-launched kernel behind this one may start its prologue now
  if (blockIdx.x >= 1) { kdaval::pad_block(a, blockIdx.x - 1); return; }   // [long] fused beta padding copy
  const bool ok = (mode & 1) ? kdaval::validate_block_fast(a, hs, (mode & 2) != 0) : (kdaval::validate_block(a, hs, (mode & 2) != 0), false);
  if ((mode & 2) && !ok) kdaval::zero_out_if_bad(a);
}

// dst[dst_slot(i)] = src[src_slot(i)] for i in [0, n), H states of 128x128 fp32 each (slot = idx[i] or row0 + i; strides in floats)
__global__ void copy_rows_kernel(const float4* __restrict__ src, long long src_stride, const int* __restrict__ src_idx, long long src_row0,
                                 float4* __restrict__ dst, long long dst_stride, const int* __restrict__ dst_idx, long long dst_row0, int H) {
  const int i = blockIdx.x / H, h = blockIdx.x % H;
  const long long ss = src_idx ? (long long)src_idx[i] : src_row0 + i;
  const long long ds = dst_idx ? (long long)dst_idx[i] : dst_row0 + i;
  const float4* s = src + (ss * src_stride + (long long)h * 16384) / 4;
  float4* d = dst + (ds * dst_stride + (long long)h * 16384) / 4;
  for (int k = threadIdx.x; k < 4096; k += blockDim.x) d[k] = s[k];
}

// [dynpath2] checkpoint rows that are plain copies of a scratch state (the head state S_A / a chained piece state S_in): row ck_base[0] + rel[i] of the checkpoint buffer
// (device-resident row: replays may change checkpoint_cu_starts; a rejected sequence has ck_base[0] < 0 and gets no copy).  item i: kind[i] < 0 -> S_A, else S_in(kind[i]).
__global__ void copy_ck_kernel(const float4* __restrict__ sa, const float4* __restrict__ sin_, float4* __restrict__ ck, const long long* __restrict__ ck_base,
                               const int2* __restrict__ items, int H) {
  const int i = blockIdx.x / H, h = blockIdx.x % H;
  const long long rb = ck_base[0];
  if (rb < 0) return;
  const int2 it = items[i];
  const float4* s = (it.y < 0 ? sa : sin_ + (long long)it.y * H * 4096) + (long long)h * 4096;
  float4* d = ck + ((rb + it.x) * H + h) * 4096;
  for (int k = threadIdx.x; k < 4096; k += blockDim.x) d[k] = s[k];
}
// [portfolio] dst[t * Hp + h] = src[t * H + h] (bf16 beta -> its TMA-aligned padded copy): one trivial launch instead of torch's strided-copy kernel (cold-L2 graph node ~2 us vs ~3.9 us)
__global__ void pad_copy_kernel64(const unsigned short* __restrict__ src, unsigned short* __restrict__ dst, long long n, int H, int Hp) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const long long t = i / H;
    dst[t * Hp + (i - t * H)] = src[i];
  }
}
__global__ void pad_copy_kernel(const unsigned short* __restrict__ src, unsigned short* __restrict__ dst, long long n, int H, int Hp) {
  // [shortmid] 32-bit index math (float reciprocal + exact fix-up: n < 2^31) and 4 independent elements per thread and step; the 64-bit division version needed 8.5 us for T = 50000
  const float inv = 1.0f / (float)H;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < (unsigned)n; i0 += 4 * stride) {
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const unsigned i = i0 + u * stride;
      if (i < (unsigned)n) {
        int t = (int)((float)i * inv);
        if ((unsigned)(t * H) > i) t--; else if ((unsigned)((t + 1) * H) <= i) t++;
        dst[(size_t)t * Hp + (i - (unsigned)t * H)] = src[i];
      }
    }
  }
}
}  // namespace

void pad_copy(torch::Tensor src, torch::Tensor dst) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.element_size() == 2 && dst.element_size() == 2 && src.dim() == 2 && dst.dim() == 2, "pad_copy: contiguous 2-byte [T, H] -> [Tp, Hp]");
  KdaDevGuard dev_guard_(src);   // [multi-GPU] launch on the tensors' device
  const long long n = src.numel();
  if (n == 0) return;
  if (n < (1ll << 27) && src.size(0) < (1 << 21)) {   // float-reciprocal index math is exact up to this size; larger tensors take the 64-bit kernel
    const long long blocks = std::min<long long>((n + 255) / 256, 592);
    pad_copy_kernel<<<(unsigned)blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const unsigned short*)src.data_ptr(), (unsigned short*)dst.data_ptr(), n, (int)src.size(1), (int)dst.size(1));
  } else {
    const long long blocks = std::min<long long>((n + 255) / 256, 296);
    pad_copy_kernel64<<<(unsigned)blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const unsigned short*)src.data_ptr(), (unsigned short*)dst.data_ptr(), n, (int)src.size(1), (int)dst.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_tables(pybind11::dict v) {
  KdaDevGuard dev_guard_(v["status"].cast<torch::Tensor>());   // [multi-GPU] launch on the tensors' device
  const VArgs a = kdaval::args_from_dict(v);
  const int mode = (v.contains("fast") ? (v["fast"].cast<int>() ? 1 : 0) : 1) | (v.contains("noop") && v["noop"].cast<int>() ? 2 : 0);
  // [resid] the launch always has at least two blocks of >= 256 threads (the second one returns at once when there is no padding copy): a ONE-block grid made this kernel and the PDL-launched kernel behind it
  // 1.5-1.9 us slower (identity launches with H % 8 == 0: 5.4 -> 3.9 us validation, 4x64 H16 -6.8% same-process A/B; the grids with padding blocks, H % 8 != 0, were already two or more blocks of 256)
  const int bd = std::min(kdaval::kThreads, std::max(256, (int)((a.N + 1 + 31) / 32 * 32)));
  validate_kernel<<<1 + std::max(a.pad_blocks, 1), bd, ((a.d != nullptr || a.ck != nullptr) && a.N <= kdaval::kHashMaxN ? 2 * kdaval::hash_slots(a.N) : 0) * sizeof(int), c10::cuda::getCurrentCUDAStream()>>>(a, mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_rows(torch::Tensor src, int64_t src_stride, c10::optional<torch::Tensor> src_idx, int64_t src_row0,
               torch::Tensor dst, int64_t dst_stride, c10::optional<torch::Tensor> dst_idx, int64_t dst_row0, int64_t n, int64_t H) {
  if (n <= 0) return;
  KdaDevGuard dev_guard_(dst);   // [multi-GPU] launch on the tensors' device
  copy_rows_kernel<<<(unsigned)(n * H), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const float4*)src.data_ptr(), src_stride, src_idx ? src_idx->data_ptr<int>() : nullptr, src_row0,
      (float4*)dst.data_ptr(), dst_stride, dst_idx ? dst_idx->data_ptr<int>() : nullptr, dst_row0, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_ck_rows(torch::Tensor sa, torch::Tensor sin_, torch::Tensor ck, torch::Tensor ck_base, torch::Tensor items, int64_t H) {
  const int n = (int)items.size(0);
  if (n <= 0) return;
  KdaDevGuard dev_guard_(sa);   // [multi-GPU] launch on the tensors' device
  copy_ck_kernel<<<(unsigned)(n * H), 256, 0, c10::cuda::getCurrentCUDAStream()>>>((const float4*)sa.data_ptr(), sin_.numel() ? (const float4*)sin_.data_ptr() : nullptr,
      (float4*)ck.data_ptr(), (const long long*)ck_base.data_ptr(), (const int2*)items.data_ptr(), (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int64_t new_stream() {
  cudaStream_t st;
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  return (int64_t)(uintptr_t)st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("new_stream", &new_stream);
  m.def("validate_tables", &validate_tables);
  m.def("copy_rows", &copy_rows);
  m.def("copy_ck_rows", &copy_ck_rows);
  m.def("pad_copy", &pad_copy);
}
