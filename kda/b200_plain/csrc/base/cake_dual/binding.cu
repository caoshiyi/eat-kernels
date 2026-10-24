// Torch binding for the fp32-state CAKE KDA prefill kernels (M64 / M128).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void launch_dual(int, cudaStream_t, void*, void*, void*, void*, void*, const unsigned char*, float*, float*,
                 long long*, int*, const int*, const int*, const float*, void*, float*, const float*, float*, void*, int, int, float, float);

static CUtensorMap enc(CUtensorMapDataType dt, int rank, void* ptr, const uint64_t* gdim,
                       const uint64_t* gstride, const uint32_t* box, CUtensorMapSwizzle sw) {
  uint32_t es[5] = {1, 1, 1, 1, 1};
  CUtensorMap m{};
  CUresult r = cuTensorMapEncodeTiled(&m, dt, rank, ptr, gdim, gstride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      sw, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", int(r));
  return m;
}

// q,k,v,g,out: [T, H, 128] bf16 ; beta_tma: [T', Hp] bf16. Returns 6*128 bytes of tensor maps (host).
torch::Tensor encode_maps(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta_tma,
                          torch::Tensor out, torch::Tensor out_T, int64_t value_rows, int64_t flags) {
  // flags: 1 = v as MN-major SW128 operand (patch_vmma), 2 = per-half k box + 16-row g box (patch_m1n64); M128 only
  if (value_rows != 128) flags = 0;
  const uint64_t T = q.numel() / (q.size(-1) * q.size(-2)), H = q.size(-2), D = q.size(-1);
  auto qk = [&](torch::Tensor t) {
    uint64_t gd[4] = {64, T, H, D / 64};
    uint64_t gs[3] = {H * D * 2, D * 2, 64 * 2};
    uint32_t box[4] = {64, 32, 1, 2};
    return enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  CUtensorMap maps[7];
  maps[0] = qk(q);
  if (flags & 2) {  // M128: raw k is loaded per 64-key half (qd/kd interleaved halves, see patch_m1n64.py)
    uint64_t gd[4] = {64, T, H, D / 64};
    uint64_t gs[3] = {H * D * 2, D * 2, 64 * 2};
    uint32_t box[4] = {64, 32, 1, 1};
    maps[1] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, k.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_128B);
  } else {
    maps[1] = qk(k);
  }
  if (flags & 1) {
    maps[2] = qk(v);  // M128: v is an MN-major SW128 MMA operand (residual via the tensor core), same layout as q/k
  } else {
    uint64_t gd[3] = {D, H, T};
    uint64_t gs[2] = {D * 2, H * D * 2};
    uint32_t box[3] = {(uint32_t)value_rows, 1, 32};
    maps[2] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, v.data_ptr(), gd, gs, box,
                  value_rows == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  {
    uint64_t gd[3] = {D, H, T};
    uint64_t gs[2] = {D * 2, H * D * 2};
    uint32_t box[3] = {128, 1, (uint32_t)((flags & 2) ? 16 : 32)};  // M128: raw gate rows 0-15 / 16-31 separately
    maps[3] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, g.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  {
    uint64_t gd[2] = {(uint64_t)beta_tma.size(1), (uint64_t)beta_tma.size(0)};
    uint64_t gs[1] = {(uint64_t)beta_tma.size(1) * 2};
    uint32_t box[2] = {8, 32};
    maps[4] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, beta_tma.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  {
    uint64_t gd[4] = {64, T, H, D / 64};
    uint64_t gs[3] = {H * D * 2, D * 2, 64 * 2};
    uint32_t box[4] = {64, 32, 1, (uint32_t)(value_rows / 64)};
    maps[5] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, out.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  {
    uint64_t gd[4] = {64, T, H, D / 64};
    uint64_t gs[3] = {H * D * 2, D * 2, 64 * 2};
    uint32_t box[4] = {64, 32, 1, (uint32_t)(value_rows / 64)};
    maps[6] = enc(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, out_T.data_ptr(), gd, gs, box, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  auto host = torch::empty({7 * 128}, torch::dtype(torch::kUInt8));
  memcpy(host.data_ptr(), maps, sizeof(maps));
  return host;
}

void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
            torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor cu_seqlens, torch::Tensor seq_order,
            torch::Tensor s0, torch::Tensor out, torch::Tensor s1, torch::Tensor s0T, torch::Tensor outT, torch::Tensor s1T,
            torch::Tensor maps, torch::Tensor task_list, torch::Tensor cta_ptr, int64_t num_heads, double scale, double lower_bound,
            int64_t l_init, int64_t t_init) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  unsigned char* mp = (unsigned char*)maps.data_ptr();
  launch_dual((int)(cta_ptr.numel() - 1), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(), mp,
              A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), (long long*)cu_seqlens.data_ptr<int64_t>(),
              seq_order.data_ptr<int>(), task_list.data_ptr<int>(), cta_ptr.data_ptr<int>(), s0.data_ptr<float>(),
              out.data_ptr(), s1.data_ptr<float>(), t_init ? s0T.data_ptr<float>() : nullptr, s1T.data_ptr<float>(), outT.data_ptr(),
              (int)l_init, (int)num_heads, (float)scale, (float)lower_bound);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encode_maps", &encode_maps);
  m.def("launch", &launch);
}
