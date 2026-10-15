// Torch binding for the persistent M128 kernel with overlapped state I/O (double-buffered TMEM state).
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_m128_cio(int, cudaStream_t, void*, void*, void*, void*, void*, const unsigned char*, float*, float*,
                     long long*, int*, const int*, const int*, float*, int*, const float*, void*, float*, int, float, float);

void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor A_log,
            torch::Tensor dt_bias, torch::Tensor cu_seqlens, torch::Tensor seq_order, torch::Tensor initial_state,
            torch::Tensor out, torch::Tensor final_state, torch::Tensor maps, torch::Tensor task_desc, torch::Tensor cta_ptr,
            torch::Tensor scratch, torch::Tensor flags, int64_t num_heads, double scale, double lower_bound) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_m128_cio((int)(cta_ptr.numel() - 1), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                  (const unsigned char*)maps.data_ptr(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
                  (long long*)cu_seqlens.data_ptr<int64_t>(), seq_order.data_ptr<int>(), task_desc.data_ptr<int>(),
                  cta_ptr.data_ptr<int>(), scratch.data_ptr<float>(), flags.data_ptr<int>(), initial_state.data_ptr<float>(),
                  out.data_ptr(), final_state.data_ptr<float>(), (int)num_heads, (float)scale, (float)lower_bound);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
