// v18_a: the claim grid of the spec multi-round whole-state kernel (kda_v14_b.cu's kda_main<128, S, true, 1024, true, true> compiled
// with CLM = 1: blocks of the last generations take their item from a self-wrapping atomic counter), in its own code object.
//
// Why its own object: as for kda_v17_a.cu. The two objects of kda_v15_a.cu (every other kernel, descriptor windows, placement
// probe) stay byte-identical to the build without this form: where a kernel's code and descriptor sit moves latency-bound shapes by
// 0.1-0.2 us, and the grids that do not take the claim form run exactly the kernels they ran before.
// Why the form: see "Claim grid" in kda_main and kernel_v15_a.claim_grid().
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#define KDA_V18_SRC "kda_v14_b.cu"  // through a macro: torch's hipify pass rewrites literal .cu includes
#define KDA_DEVICE_ONLY
#define CLM 1
namespace kv3 {
#include KDA_V18_SRC
}
#undef KDA_DEVICE_ONLY

namespace {
template <int S>
void go(const kv3::Params& p, int rows, hipStream_t st) {
  hipLaunchKernelGGL((kv3::kda_main<128, S, true, 1024, true, true>), dim3(p.H, rows), dim3(1024), 0, st, p.idx, p.acc, p.state, p.conv, p.x, p.H,
                     (int)p.state_slot_stride, (int)p.conv_slot_stride, p.nslots, p);
}
}  // namespace

// The claim-grid launch: arguments as kda_v15_a.cu's run() for a spec multi-round whole-state launch (hold, nxt as there), plus
// cw: the counter word (a view whose first element is the word), extra: rows of blocks behind the grid, c0: first claiming block.
void run3(torch::Tensor x, torch::Tensor conv, torch::Tensor cw, torch::Tensor gate, torch::Tensor beta, torch::Tensor og,
          torch::Tensor A_log, torch::Tensor dt, torch::Tensor nw, torch::Tensor state, torch::Tensor idx, torch::Tensor acc,
          torch::Tensor out, torch::Tensor word, double eps, double lb, int64_t hold, int64_t nxt, int64_t extra, int64_t c0) {
  const int B = idx.size(0);
  const int S = (int)(x.size(0) / B);
  kv3::Params p;
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
  p.mb = reinterpret_cast<unsigned*>(word.data_ptr());
  p.x_stride = x.stride(0);
  p.bstr = beta.stride(0);
  p.ogstr = og.stride(0);
  p.conv_slot_stride = conv.stride(0);
  p.state_slot_stride = state.stride(0);
  p.nslots = (int)std::min(state.size(0), conv.size(0));
  p.vec = 0;
  p.late = 0;
  p.pf = 1;
  p.hold = (int)hold;
  p.nxt = (int)nxt;
  p.H = (int)A_log.numel();
  p.eps = (float)eps;
  p.lb = (float)lb;
  const int64_t items = (int64_t)B * p.H;
  TORCH_CHECK(word.numel() >= 1 && word.dtype() == torch::kInt32, "kda decode: claim grid without a counter word");
  TORCH_CHECK(extra >= 1 && extra <= 4095 && c0 >= 0 && c0 <= items && c0 < (1 << 19) && nxt > 0 && items + extra * p.H < (1 << 30),
              "kda decode: claim grid out of range (B ", B, ", H ", p.H, ", extra rows ", extra, ", c0 ", c0, ")");
  p.dbg = (int)(extra | c0 << 12);
  hipStream_t st = c10::hip::getCurrentHIPStream();
  const int rows = B + (int)extra;
  switch (S) {
    case 5: go<5>(p, rows, st); break;
    case 6: go<6>(p, rows, st); break;
    case 7: go<7>(p, rows, st); break;
    case 8: go<8>(p, rows, st); break;
    default: TORCH_CHECK(false, "kda decode: unsupported configuration (claim grid, S ", S, ")");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run3", &run3); }
