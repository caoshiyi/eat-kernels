// v17_a: the 2-way V-split (64 state rows per block) of the staged spec kernel kda_vec, in its own code object.
//
// Why. kernel_v13_a.choose() had two V-split forms, 4-way (32 rows) and 8-way (16 rows), both limited to B*H <= 64 pairs (the split
// grid must fit the CUs). A spec grid of 64 < B*H <= 128 pairs ran one 1024-thread block per pair: 16 waves on one CU, VALU-issue
// bound at ~0.9 us per token, with half of the CUs idle. The 2-way split (2 x B*H <= 256 blocks of 512 + 64 threads) halves the
// per-CU token work for the price of the mailbox exchange; see choose() for where it pays.
// The kernel is kda_v14_b.cu's kda_vec<64, S, true, 512, true>, the template unchanged (same max-ILP build flags as every spec
// kernel). It lives in a third code object so that the two objects of kda_v15_a.cu (every other kernel, descriptor windows,
// placement probe) stay byte-identical: where a kernel's code and descriptor sit moves latency-bound shapes by 0.1-0.2 us.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#define KDA_V17_SRC "kda_v14_b.cu"  // through a macro: torch's hipify pass rewrites literal .cu includes
#define KDA_DEVICE_ONLY
namespace kv2 {
#include KDA_V17_SRC
}
#undef KDA_DEVICE_ONLY

namespace {
template <int S>
void go(const kv2::Params& p, int B, hipStream_t st) {
  hipLaunchKernelGGL((kv2::kda_vec<64, S, true, 512, true>), dim3(p.H * 2, B), dim3(512 + 64), 0, st, p.idx, p.acc, p.state, p.conv, p.x, p.H,
                     (int)p.state_slot_stride, (int)p.conv_slot_stride, p.nslots, p);
}
}  // namespace

// The 2-way split launch: arguments as kda_v15_a.cu's run() for a staged (vec) spec launch. mb: B*H*32 mailbox words at the sentinel.
void run2(torch::Tensor x, torch::Tensor conv, torch::Tensor cw, torch::Tensor gate, torch::Tensor beta, torch::Tensor og,
          torch::Tensor A_log, torch::Tensor dt, torch::Tensor nw, torch::Tensor state, torch::Tensor idx, torch::Tensor acc,
          torch::Tensor out, torch::Tensor mb, double eps, double lb) {
  const int B = idx.size(0);
  const int S = (int)(x.size(0) / B);
  kv2::Params p;
  p.x = reinterpret_cast<decltype(p.x)>(x.data_ptr());
  p.conv = reinterpret_cast<decltype(p.conv)>(conv.data_ptr());
  p.cw = cw.data_ptr<float>();
  p.gate = reinterpret_cast<decltype(p.gate)>(gate.data_ptr());
  p.beta = reinterpret_cast<decltype(p.beta)>(beta.data_ptr());
  p.og = reinterpret_cast<decltype(p.og)>(og.data_ptr());
  p.A_log = A_log.data_ptr<float>();
  p.dt = dt.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.state = state.data_ptr<float>();
  p.idx = idx.data_ptr<int>();
  p.acc = acc.data_ptr<int>();
  p.out = reinterpret_cast<decltype(p.out)>(out.data_ptr());
  p.cnt = nullptr;
  p.ow = nullptr;
  p.mb = reinterpret_cast<unsigned*>(mb.data_ptr());
  p.x_stride = x.stride(0);
  p.bstr = beta.stride(0);
  p.ogstr = og.stride(0);
  p.conv_slot_stride = conv.stride(0);
  p.state_slot_stride = state.stride(0);
  p.nslots = (int)std::min(state.size(0), conv.size(0));
  p.vec = 1;
  p.pf = 0;
  p.hold = 0;
  p.nxt = 0;
  p.H = (int)A_log.numel();
  p.eps = (float)eps;
  p.lb = (float)lb;
  TORCH_CHECK(mb.numel() >= (int64_t)B * p.H * 4 * S, "kda decode: mailbox too small for the 2-way split");
  hipStream_t st = c10::hip::getCurrentHIPStream();
  switch (S) {
    case 2: go<2>(p, B, st); break;
    case 3: go<3>(p, B, st); break;
    case 4: go<4>(p, B, st); break;
    case 5: go<5>(p, B, st); break;
    case 6: go<6>(p, B, st); break;
    case 7: go<7>(p, B, st); break;
    case 8: go<8>(p, B, st); break;
    default: TORCH_CHECK(false, "kda decode: unsupported configuration (2-way split, S ", S, ")");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run2", &run2); }
