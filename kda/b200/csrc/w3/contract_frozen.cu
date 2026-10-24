// [final6] D2: validation launch of the FROZEN one-sequence plans (planned at prepare time from the shape alone, [0, T]; they never read cu_seqlens).  Same work as contract_aux.cu's validate_kernel plus: a cu_seqlens != [0, T]
// (a padded bucket: fewer real tokens than the captured total) must not run the frozen plan on the real slot / checkpoint rows: status bit 1, destination slot -> trash window, no checkpoint rows (the redirect of a repeated /
// out-of-range slot).  N == 1, so thread 0 owns every table entry written by the validation.  Own extension module: contract_aux.cu stays byte-identical (its validation kernel is on the critical path of every identity-plan launch).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};
#include "validate_dev.cuh"
using kdaval::VArgs;
namespace {
__global__ void __launch_bounds__(kdaval::kThreads) validate_frozen_kernel(VArgs a, int mode) {
  extern __shared__ int hs[];
  asm volatile("griddepcontrol.launch_dependents;");
  if (blockIdx.x >= 1) { kdaval::pad_block(a, blockIdx.x - 1); return; }
  const bool ok = (mode & 1) ? kdaval::validate_block_fast(a, hs, (mode & 2) != 0) : (kdaval::validate_block(a, hs, (mode & 2) != 0), false);
  if ((mode & 2) && !ok) kdaval::zero_out_if_bad(a);
  if (a.cu != nullptr && threadIdx.x == 0) {
    if (kdaval::rd(a.cu, a.cu_is64, 0) != 0 || kdaval::rd(a.cu, a.cu_is64, a.N) != a.T) {
      *a.status |= 1;
      if (a.d != nullptr) a.d_out[0] = a.d_trash;
      if (a.ck != nullptr) a.ck_out[0] = -1;
    }
  }
}
}  // namespace

void validate_tables_frozen(pybind11::dict v) {
  KdaDevGuard dev_guard_(v["status"].cast<torch::Tensor>());
  const VArgs a = kdaval::args_from_dict(v);
  TORCH_CHECK(a.N == 1, "validate_tables_frozen: one sequence only");
  const int mode = (v.contains("fast") ? (v["fast"].cast<int>() ? 1 : 0) : 1) | (v.contains("noop") && v["noop"].cast<int>() ? 2 : 0);
  const int bd = std::min(kdaval::kThreads, std::max(256, (int)((a.N + 1 + 31) / 32 * 32)));
  validate_frozen_kernel<<<1 + std::max(a.pad_blocks, 1), bd, ((a.d != nullptr || a.ck != nullptr) && a.N <= kdaval::kHashMaxN ? 2 * kdaval::hash_slots(a.N) : 0) * sizeof(int), c10::cuda::getCurrentCUDAStream()>>>(a, mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("validate_tables_frozen", &validate_tables_frozen);
}
