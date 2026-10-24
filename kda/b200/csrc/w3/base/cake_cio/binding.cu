// Torch binding for the persistent M128 kernel with overlapped state I/O (double-buffered TMEM state).
#include <cuda_runtime.h>
#include "../ck_type.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};

#ifdef KDA_FEAT
#define FEAT_DECL , const int*, long long, const int*, long long, ck_t*, const long long*, int
#else
#define FEAT_DECL
#endif
void launch_m128_cio(int, cudaStream_t, void*, void*, void*, void*, void*, const unsigned char*, float*, float*,
                     long long*, int*, const int*, const int*, float*, int*, const float*, void*, float*, int, float, float FEAT_DECL);

void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor A_log,
            torch::Tensor dt_bias, torch::Tensor cu_seqlens, torch::Tensor seq_order, torch::Tensor initial_state,
            torch::Tensor out, torch::Tensor final_state, torch::Tensor maps, torch::Tensor task_desc, torch::Tensor cta_ptr,
            torch::Tensor scratch, torch::Tensor flags, int64_t num_heads, double scale, double lower_bound
#ifdef KDA_FEAT
            , c10::optional<torch::Tensor> src_idx, int64_t src_stride, c10::optional<torch::Tensor> dst_idx, int64_t dst_stride,
            c10::optional<torch::Tensor> ck_base, c10::optional<torch::Tensor> ck_cu, int64_t ck_n
#endif
            ) {
  KdaDevGuard dev_guard_(q);   // [multi-GPU] stream, cudaGetDevice() (per-device attribute flags) and the launch all belong to the tensors' device
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_m128_cio((int)(cta_ptr.numel() - 1), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                  (const unsigned char*)maps.data_ptr(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
                  (long long*)cu_seqlens.data_ptr<int64_t>(), seq_order.data_ptr<int>(), task_desc.data_ptr<int>(),
                  cta_ptr.data_ptr<int>(), scratch.data_ptr<float>(), flags.data_ptr<int>(), initial_state.data_ptr<float>(),
                  out.data_ptr(), final_state.data_ptr<float>(), (int)num_heads, (float)scale, (float)lower_bound
#ifdef KDA_FEAT
                  , src_idx ? src_idx->data_ptr<int>() : nullptr, src_stride, dst_idx ? dst_idx->data_ptr<int>() : nullptr, dst_stride,
                  ck_base ? (ck_t*)ck_base->data_ptr() : nullptr, ck_cu ? (const long long*)ck_cu->data_ptr<int64_t>() : nullptr, (int)ck_n
#endif
                  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
