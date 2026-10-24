// v15_a: every kernel of kda_v14_b.cu (v74's kernels + the per-XCC kernels of v75) TWICE in one code object, the two copies' kernel
// descriptors 8 KiB out of phase, plus a placement probe. See kernel_v15_a.py / NOTES_v15.md.
//
// Why. Before the first instruction of a dispatch the command processor of each XCC reads the kernel descriptor (64 B in .rodata).
// MI350X interleaves 4 KiB pages over its two I/O dies by a parity of physical address bits (13, 21, 28: scratch_v15/e1.py); a
// line on the other die costs an XCC 0.08-0.12 us more, and a dispatch ends on the XCCs that start last (4..7, all on one die:
// ASYMMETRY.md A.1). Measured on v75 (scratch_v15/draw.py, 600 fresh processes, descriptor page probed in each): descriptor on the
// die of XCCs 4-7 vs the other: B = 1 spec -0.15..-0.19 us, mid-B -0.04..-0.13, plain B = 1 -0.03..-0.04. Which die an object lands on
// is decided by the driver's buddy allocator per process (v75: the fast one in 28 % / 54 % of the eval's processes for its two
// main objects). Two descriptors 8 KiB apart inside one buddy piece are always on different dies, so with two copies one of them
// is on the fast die in every process; kda_pp() measures which (one cold load per launch from XCC 1 / XCC 5 at lines of the two
// descriptor pages), kernel_v15_a.py chooses once per device, outside any capture.
// The copies are the same source in two namespaces: identical instruction streams (scratch_v15/isa15.py).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#define KDA_V15_SRC "kda_v14_b.cu"  // through a macro: torch's hipify pass rewrites literal .cu includes
#ifndef KDA_V15_PAD0
#define KDA_V15_PAD0 128  // filler descriptors (64 B each) in front of copy 0's descriptor table: one 8 KiB window of them
#endif
#ifndef KDA_V15_PAD1
#define KDA_V15_PAD1 120  // between the two tables: 128 - (kernels per copy), so that copy 1's table starts exactly 8 KiB behind copy 0's
#endif
#ifndef KDA_V15_PAD2
#define KDA_V15_PAD2 128  // behind copy 1's table (never-dispatched lines of its window, for the probe)
#endif
#ifndef KDA_V15_BSS
#define KDA_V15_BSS 0  // experiment: bytes of .bss filler (memory size only)
#endif
#define KDA_DEVICE_ONLY

// Filler kernels: one descriptor (and 256 B of code) each. Reached through two template levels, as the real kernels are
// (dispatch -> launch_s -> launch -> kernel), so that the compiler emits them between the two copies.
namespace kpad {
template <int G, int I>
__global__ __launch_bounds__(64) void kda_pad(int* o) {
  if (o) o[0] = G * 4096 + I;
}
template <int G, int I>
const void* pad2() {
  if constexpr (I > 0) (void)pad2<G, I - 1>();
  return reinterpret_cast<const void*>(&kda_pad<G, I>);
}
template <int G, int I>
const void* pad1() { return pad2<G, I>(); }
}  // namespace kpad

#ifndef KDA_V15_KALIGN
#define KDA_V15_KALIGN 0  // alignment (bytes) of every kernel of the two copies; 0 = the toolchain's 256
#endif
#if KDA_V15_KALIGN > 0
#pragma push_macro("__global__")
#undef __global__
#define __global__ __attribute__((global)) __attribute__((aligned(KDA_V15_KALIGN)))
#endif

#if KDA_V15_PAD0 > 0
const void* kda_v15_pad0() { return kpad::pad1<0, KDA_V15_PAD0 - 1>(); }
#endif
namespace kc0 {
#include KDA_V15_SRC
}
#if KDA_V15_PAD1 > 0
const void* kda_v15_pad1() { return kpad::pad1<1, KDA_V15_PAD1 - 1>(); }
#endif
namespace kc1 {
#include KDA_V15_SRC
}
#if KDA_V15_PAD2 > 0
const void* kda_v15_pad2() { return kpad::pad1<2, KDA_V15_PAD2 - 1>(); }
#endif
#undef KDA_DEVICE_ONLY
#if KDA_V15_KALIGN > 0
#pragma pop_macro("__global__")
#endif

#if KDA_V15_BSS > 0
// Experiment only (kernel_v15_a.py _PADS; NOTES_v15 2.4, not used): a loaded object of exactly 2 MiB is served from the driver's
// 2 MiB blocks instead of its 4 KiB-granular pieces.
__device__ __attribute__((used)) char kda_v15_bss[KDA_V15_BSS];
#endif

// Placement probe. One wave on XCC `xcc` issues one cold nontemporal load at `a` and reports its latency in s_memrealtime ticks
// (10 ns). The other seven workgroups return at entry.
__global__ __launch_bounds__(64) void kda_pp(const char* a, long long* out, int xcc) {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(x));
  if ((x & 15) != xcc || threadIdx.x != 0) return;
  unsigned v;
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  asm volatile("global_load_dword %0, %1, off nt\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(a));
  const unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  out[0] = (long long)(t1 - t0);
  out[1] = v;
}
// Where the code object is: the address of an instruction of this kernel (the loader places the object at offset = ELF address).
__global__ __launch_bounds__(64) void kda_where(long long* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  unsigned long long pc;
  asm volatile("s_getpc_b64 %0" : "=s"(pc));
  out[0] = (long long)pc;
}

namespace {
template <class P>
void fill(P& p, torch::Tensor& x, torch::Tensor& conv, torch::Tensor& cw, torch::Tensor& gate, torch::Tensor& beta, torch::Tensor& og,
          torch::Tensor& A_log, torch::Tensor& dt, torch::Tensor& nw, torch::Tensor& state, torch::Tensor& idx, std::optional<torch::Tensor>& acc,
          torch::Tensor& out, torch::Tensor& mb, double eps, double lb, int64_t vec, int64_t pf, int64_t hold, int64_t nxt) {
  using bf = decltype(*p.x);
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
  p.acc = acc.has_value() ? acc->data_ptr<int>() : nullptr;
  p.out = reinterpret_cast<decltype(p.out)>(out.data_ptr());
  p.cnt = nullptr;  // as kda_v14_b.cu's run(): the fields stay, no instantiated kernel reads them
  p.ow = nullptr;
  p.mb = reinterpret_cast<unsigned*>(mb.data_ptr());
  p.x_stride = x.stride(0);
  p.bstr = beta.stride(0);
  p.ogstr = og.stride(0);
  p.conv_slot_stride = conv.stride(0);
  p.state_slot_stride = state.stride(0);
  p.nslots = (int)std::min(state.size(0), conv.size(0));
  p.vec = (int)vec;
  p.pf = (int)pf;
  p.hold = (int)hold;
  p.nxt = (int)nxt;
  p.H = (int)A_log.numel();
  p.eps = (float)eps;
  p.lb = (float)lb;
}
}  // namespace

// As kda_v14_b.cu's run(), plus `copy`: which of the two copies of the kernel is launched (same instructions; the descriptors
// are 8 KiB apart).
void run(torch::Tensor x, torch::Tensor conv, torch::Tensor cw, torch::Tensor gate, torch::Tensor beta,
         torch::Tensor og, torch::Tensor A_log, torch::Tensor dt, torch::Tensor nw, torch::Tensor state,
         torch::Tensor idx, std::optional<torch::Tensor> acc, torch::Tensor out, torch::Tensor mb,
         double eps, double lb, int64_t nsplit, int64_t nth, int64_t vec, int64_t pf, int64_t hold, int64_t nxt, int64_t xv, int64_t copy) {
  const int B = idx.size(0);
  const int T = x.size(0);
  bool ok;
  if (copy) {
    kc1::Params p;
    kc1::g_xv = (int)xv;
    fill(p, x, conv, cw, gate, beta, og, A_log, dt, nw, state, idx, acc, out, mb, eps, lb, vec, pf, hold, nxt);
    ok = kc1::dispatch(p, B, T / B, acc.has_value(), (int)nsplit, (int)nth, c10::hip::getCurrentHIPStream());
  } else {
    kc0::Params p;
    kc0::g_xv = (int)xv;
    fill(p, x, conv, cw, gate, beta, og, A_log, dt, nw, state, idx, acc, out, mb, eps, lb, vec, pf, hold, nxt);
    ok = kc0::dispatch(p, B, T / B, acc.has_value(), (int)nsplit, (int)nth, c10::hip::getCurrentHIPStream());
  }
  TORCH_CHECK(ok, "kda decode: unsupported configuration (S ", T / B, ", spec ", acc.has_value(), ", nsplit ", nsplit, ", threads ", nth, ", vec ", vec, ", pf ", pf, ")");
}

// VGPRs per lane of the kernel run() would launch for (S, spec, nsplit, nth); -1 if unsupported. (Copy 0: the copies are identical.)
int64_t regs(int64_t S, bool spec, int64_t nsplit, int64_t nth, int64_t nx) {
  kc0::Params p{};
  p.pf = 1;
  p.nxt = (int)nx;
  kc0::g_xv = 0;
  kc0::g_query = 1;
  kc0::g_regs = -1;
  const bool ok = kc0::dispatch(p, 1, (int)S, spec, (int)nsplit, (int)nth, nullptr);
  kc0::g_query = 0;
  return ok ? kc0::g_regs : -1;
}

int64_t num_xcc() {
  int dev = 0, v = 0;
  if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&v, hipDeviceAttributeNumberOfXccs, dev) != hipSuccess) return 0;
  return v;
}

bool stream_unmasked() {
  int dev = 0;
  hipDeviceProp_t pr;
  if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&pr, dev) != hipSuccess) return false;
  const int ncu = pr.multiProcessorCount;
  if (ncu <= 0 || ncu > 4096) return false;
  std::vector<uint32_t> m((ncu + 31) / 32, 0u);
  if (hipExtStreamGetCUMask(c10::hip::getCurrentHIPStream().stream(), (uint32_t)m.size(), m.data()) != hipSuccess) return false;
  int on = 0;
  for (int i = 0; i < ncu; ++i) on += (m[i / 32] >> (i % 32)) & 1;
  return on == ncu;
}

void xcc_probe(torch::Tensor buf, int64_t gx, int64_t gy) {
  TORCH_CHECK(buf.numel() >= gx * gy, "kda decode: probe buffer too small");
  hipLaunchKernelGGL(kc0::kda_xcc_probe, dim3((unsigned)gx, (unsigned)gy), dim3(1024), 0, c10::hip::getCurrentHIPStream(), buf.data_ptr<int>());
}

void pp(int64_t addr, torch::Tensor out, int64_t xcc) {
  hipLaunchKernelGGL(kda_pp, dim3(8), dim3(64), 0, c10::hip::getCurrentHIPStream(), reinterpret_cast<const char*>(addr), reinterpret_cast<long long*>(out.data_ptr()), (int)xcc);
}

// The probe needs its line cold in every cache level: a 512 MB fill evicts the 256 MB memory-side cache, as the eval's flush does.
// The buffer is the extension's own (hipMalloc, above ROCr's 2 MiB sub-allocator: one driver allocation, returned to the driver
// by hipFree), not a torch tensor: nothing stays in the caller's allocator after the probe.
namespace {
void* g_flush = nullptr;
size_t g_flush_n = 0;
}  // namespace
bool flush_open(int64_t bytes) {
  if (g_flush) return true;
  void* p = nullptr;
  if (bytes <= 0 || hipMalloc(&p, (size_t)bytes) != hipSuccess || !p) {
    (void)hipGetLastError();
    return false;
  }
  g_flush = p;
  g_flush_n = (size_t)bytes;
  return true;
}
void flush_fill() {
  if (g_flush) TORCH_CHECK(hipMemsetAsync(g_flush, 0, g_flush_n, c10::hip::getCurrentHIPStream().stream()) == hipSuccess, "kda decode: probe flush failed");
}
void flush_close() {
  if (g_flush) (void)hipFree(g_flush);
  g_flush = nullptr;
  g_flush_n = 0;
}

void where(torch::Tensor out) {
  hipLaunchKernelGGL(kda_where, dim3(1), dim3(64), 0, c10::hip::getCurrentHIPStream(), reinterpret_cast<long long*>(out.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("xcc_probe", &xcc_probe);
  m.def("x1_xcc", []() { return (int64_t)X1_XCC; });
  m.def("num_xcc", &num_xcc);
  m.def("stream_unmasked", &stream_unmasked);
  m.def("xcc_consts", []() {
    std::vector<int64_t> v{X1_XCC};
#define X(m) v.push_back(m);
    KDA_XF_MASKS
#undef X
    v.push_back(UQ_XCC);
    v.push_back(UQ_RM);
    return v;
  });
  m.def("regs", &regs);
  m.def("pp", &pp);
  m.def("where", &where);
  m.def("flush_open", &flush_open);
  m.def("flush_fill", &flush_fill);
  m.def("flush_close", &flush_close);
}
