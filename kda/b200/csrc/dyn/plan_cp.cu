// Device-side planner for the CAKE KDA kernels' task tables, graph-safe: reads cu_seqlens on the GPU, writes
//  (1) the cio task table of the main launch (A group),
//  (2) optionally (cp_enabled) the context-parallel plan of the SAME cost model as the host planner (kernel.py cp_plan): which sequence to cut (one of the two
//      longest), where (xc), the SM budget n_A of the head group, one or two tail pieces; the dual-chain tail table, the fixup segment table and the tail geometry.
// See REPORT.md for the model; plan quality vs the host planners is checked by test_plan.py / test_cp_plan.py.
//
// Wrap plan (head group): tasks (sequence, head) form a stream (sequences by decreasing chunk count, heads in order); CTA j gets a contiguous range of it, filled
// from the highest CTA down.  If every boundary cuts a task, equal list loads Lambda put boundary i (between top-down CTAs i-1 and i) at raw stream cost
// p_i = (Lambda - 8) + (i - 1)(Lambda - 333), Lambda = (W + 333 (ncta - 1)) / ncta (333 = entry + front + back overhead of a cut: 277 + 48 + 8; units 1/100 chunk-step).
// Each thread locates its boundary by a binary search over the per-sequence cost prefix; a boundary becomes a cut (back piece [n-a, n) at the end of the upper CTA,
// front piece [0, n-a) at the start of the lower CTA, exact fp32 state hand-off through a scratch slot) if the task has >= 17 chunks, both pieces are non-empty, no other
// boundary falls in the same task and the back piece starts >= 24 chunk-steps after the front piece ends (the host's rule); otherwise it snaps to the nearer task boundary.
// 6 capacity variants Lambda * (1 + 0.5% m), m = 0, 1, 2, 4, 8, 16 are evaluated in parallel; if the best one is > 1.5% above max(Lambda, longest task) a snake deal of the
// sorted tasks (LPT-like whole-task plan) is also modelled and used when better.
// [dynpath iter 32] Lean planner kernels (plan_lean_plain_kernel / plan_lean_cio_kernel / lean_cio_body, used by plan_lean() and, for CP-eligible shapes, as the early-out of plan_kernel):
// the general kernel above is cold-instruction-fetch and barrier-latency bound (~22 us in situ, judge protocol: the big main kernel evicts its code between calls); for <= 256
// sequences without a context-parallel cut the lean kernels produce the SAME tables in ~10 us (plain / snake: 3.4 us): 32-bit math, every quantity computed once, 4 passes.  Decisions
// are bit-identical to the general kernel (tools/leancmp.py) except the whole-task rule: a wrap plan must model >= 5% better than the whole-task snake deal (measured: the wrap plan
// with many cuts runs ~7% slower than a whole-task deal that models 3.7% worse).  CP-eligible shapes: plan_kernel (plan mode bit 2) first applies the host cp_plan early exit and an
// exact feasibility test (the longest remaining whole task of every cuttable sequence already exceeds the bar a cut must beat) and then runs lean_cio_body.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};
#include <cstdint>
#include "../w3/validate_dev.cuh"   // [contract2] table validation, run by a second block of this launch concurrently with the planner (block 0)

namespace {
constexpr int MAXC = 256;
constexpr int NTHR = 1024;
constexpr int NVAR = 6;
constexpr int PRELOAD_MIN = 17;
// cost constants of base/kernel.py (_CIO_*), in units of 1/100 chunk-step
constexpr long long U = 100, C_ENTRY = 277, C_FRONT = 48, C_BACK = 8, C_SHORT = 70;
constexpr long long C_CUT = C_ENTRY + C_FRONT + C_BACK;   // 333
constexpr int NB = MAXC + 2;
// kernel.py CP cost model (chunk-steps unless noted)
constexpr double STEP_US = 1.70, CONTENTION = 1.0, FIX_BW = 4.5e6, FIX_STATE_US = 0.08, FIX_FIXED_US = 6.0, MIN_GAIN = 0.04, DUAL_F = 1.40, DUAL_TASK = 6.4;
constexpr double MULTI_MARGIN = 0.05, FIX_CHAIN_US = 0.07, FIX_STATE_TILES = 6.0, FIX_OWNER_TILES = 8.0, FIX_CHAIN_TILES = 4.0;
constexpr int PIECE_TASKS_PER_CTA = 3, PIECE_MIN = 8;
constexpr double SNAP_STEPS = 3.0, FIX_ITEM_US = 6.0;   // [dynpath2] checkpoints: chunk-steps per tail snapshot of a dual task, fixup us per checkpoint item (kernel.py _SNAP_STEPS / _FIX_ITEM_US)

__device__ __forceinline__ long long wcost(int n) { return (long long)n * U + C_ENTRY + (n < PRELOAD_MIN ? C_SHORT : 0); }
__device__ __forceinline__ long long front_cost(int b) { return (long long)b * U + C_ENTRY + C_FRONT + (b < PRELOAD_MIN ? C_SHORT : 0); }
__device__ __forceinline__ double wc_d(int n) { return (double)wcost(n) / 100.0; }   // chunk-steps

// x / H and x % H for 0 <= x < 2^31 by one mulhi (magic = ceil(2^32 / H): the estimate is the quotient or one above)
struct FastDiv {
  unsigned H, magic;
  __device__ __forceinline__ FastDiv(int h) : H((unsigned)h), magic((unsigned)((4294967296ull + h - 1) / (unsigned)h)) {}
  __device__ __forceinline__ int div(int x) const { if (H == 1) return x; unsigned q = __umulhi((unsigned)x, magic); q -= (q * H > (unsigned)x); return (int)q; }
};

// exclusive scan of x[0..n) by one warp (lane = threadIdx.x < 32); returns the total
__device__ __forceinline__ int warp_scan_excl(int* x, int n) {
  const int lane = threadIdx.x;
  const int per = (n + 31) / 32, b0 = lane * per, b1 = min(n, b0 + per);
  int s = 0;
  for (int i = b0; i < b1; i++) s += x[i];
  int inc = s;
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) { const int o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += o; }
  int run = inc - s;
  for (int i = b0; i < b1; i++) { const int v = x[i]; x[i] = run; run += v; }
  return __shfl_sync(0xffffffffu, inc, 31);
}

// ---------------------------------------------------------------- CP cost model (device)
__device__ double fix_us(long long tail, int H) { return FIX_FIXED_US + (double)(tail * H * 128 * 2 * 3) / FIX_BW + FIX_STATE_US * H; }

// exact LPT makespan of H tasks of cost t1 followed by H tasks of cost t2 <= t1 on n bins: smalls go to the least loaded bin = the H smallest completion values
__device__ double lpt2(int n, int H, double t1, double t2) {
  const int q = H / n, r = H % n;
  const double Lhi = (q + 1) * t1, Llo = q * t1;
  int remaining = H, clo = n - r, chi = r;
  long long klo = 1, khi = 1;
  double last = 0.0;
  while (remaining > 0) {
    const double vlo = clo > 0 ? Llo + klo * t2 : 1e300, vhi = chi > 0 ? Lhi + khi * t2 : 1e300;
    if (vlo <= vhi) { remaining -= min(remaining, clo); last = vlo; klo++; }
    else { remaining -= min(remaining, chi); last = vhi; khi++; }
  }
  return fmax(r > 0 ? Lhi : Llo, last);
}

// [dynpath2] necessary condition for ANY cut of a sequence of tc chunks to cost less than `bcost` (all in chunk-steps): the head piece (xc chunks) costs >= xc + 2.77 on one CTA, so xc <= bcost - fix_min - 2.77;
// the head + tail work (other sequences wexcl, H head tasks, H dual tasks of DUAL_F per chunk + DUAL_TASK) shrinks with xc, so its value at the largest allowed xc divided by the SM count (max(a, b_d) >= work / n_sms)
// plus the cheapest fixup bounds the cost from below.  When it exceeds bcost the whole candidate search (cp_single over all budgets / cut points) can only reject: skipped (identical decisions).
__device__ __forceinline__ bool cp_lb_ok(int H, int n_sms, double wexcl, int tc, double bcost) {
  const double fixmin = (FIX_FIXED_US + FIX_STATE_US * H) / STEP_US;
  const double xmax = fmin((double)(tc - 4), bcost - fixmin - 2.77);
  if (xmax < 4.0) return false;
  const double W = wexcl + H * (xmax + 2.77) + H * (DUAL_F * (tc - xmax) + DUAL_TASK);
  return W / n_sms + fixmin <= bcost;
}

struct CpSingle { double a, fix, cost, snap; };   // head-group makespan, fixup, whole-tail-per-head cost (dual launch, one piece), snapshot chunk-steps of the tail (checkpoints)

// cost of cutting a sequence (tokens Tj, chunks tcj) at xc chunks with n_A head-group CTAs (host cost_at, dual-chain tail only; the head group is priced by its fluid bound)
__device__ bool cp_single(int H, int n_sms, long long Wexcl, double wmax_excl, long long Tj, int tcj, int xc, int n_A, CpSingle* out, int ck_n = 0, int rows_cap = 0) {
  const int n_B = (n_sms - n_A) / 2;
  if (n_A < 1 || n_B < 1) return false;
  const int n_D = n_sms - n_A;
  const int tcx = tcj - xc;
  const double wa = (double)Wexcl / 100.0 + (double)H * wc_d(xc);
  CpSingle c;
  c.a = fmax((wa + 3.33 * (n_A - 1)) / n_A, fmax(wmax_excl, wc_d(xc)));
  // [dynpath2] tail checkpoint boundaries beyond the cut (host cost_at: rows, _SNAP_STEPS per snapshot in every dual task, _FIX_ITEM_US per fixup item on its owner CTA)
  const int rows = ck_n > 0 ? max(0, (tcj - 1) / ck_n - xc / ck_n) : 0;
  if (rows > rows_cap && ck_n > 0) return false;
  c.snap = rows * SNAP_STEPS;
  const double b_d = ceil((double)H / n_D) * (DUAL_F * tcx + DUAL_TASK + c.snap);
  c.fix = (fix_us(Tj - 32LL * xc, H) + ((H + n_sms - 1) / n_sms) * rows * FIX_ITEM_US) / STEP_US;
  c.cost = CONTENTION * fmax(c.a, b_d) + c.fix;
  *out = c;
  return true;
}

// two-piece tail: piece candidate pi (0 .. NPC-1) -> (s1, s2), s1 a multiple of 4 chunks, s2 <= s1, s2 >= 8; mk < 0 when the candidate does not exist
constexpr int NPC = 24;
__device__ double piece_mk(int pi, int tcx, int H, int n_D, int* s1o, int* s2o) {
  if (n_D < 2 || tcx < 2 * PIECE_MIN || 2 * H > PIECE_TASKS_PER_CTA * n_D) return -1.0;
  const int s1min = (((tcx + 1) / 2 + 3) / 4) * 4, s1max = ((tcx - PIECE_MIN) / 4) * 4;   // s1 a multiple of 4 chunks, s1 >= tcx / 2 (s2 <= s1), s2 >= 8
  if (s1min > s1max) return -1.0;
  const int s1 = s1min + (int)(((long long)pi * ((s1max - s1min) / 4)) / (NPC - 1)) * 4, s2 = tcx - s1;
  *s1o = s1; *s2o = s2;
  return lpt2(n_D, H, DUAL_F * s1 + DUAL_TASK, DUAL_F * s2 + DUAL_TASK);
}

struct CpOut {          // context-parallel outputs (device); null pointers when the graph has no CP branch
  int n_sms, slot_off;
  int* dtask; int* dptr;            // dual-chain tail launch: task ids (piece * H + head), per-CTA pointers [n_sms + 1]
  long long* cu_tail;               // int64 [3]: absolute tail piece offsets
  int* seg; int* seg_ptr;           // fixup segments (head, piece, first tile, tile count), per-CTA pointers [n_sms + 1]
  int* geom;                        // int32 [8]: x, m (0 = no CP), e, j, n_A, n_D, xc, 1
  long long* cu_a;                  // optional int64 [2] (single-sequence shapes, plain head launch): [cu[0], x] with CP, else [cu[0], cu[1]]
  float ovh;                        // [dynpath2] chunk-steps a cut must save beyond the host rule (device-planned replays pay the planner + the extra launches of the CP path)
  int ck_n, ck_rows_cap;            // [dynpath2] checkpoints (chunks per row, 0 = off) and the capacity of the snapshot buffers in rows
  int* cki;                         // int4 [ck_rows_cap]: checkpoint items of the fixup (snapshot row, piece, row relative to ck_cu[j], 0)
  int* ccopy;                       // int2 [2]: plain-copy checkpoint rows (relative row, kind < 0: S_A, else S_in(kind))
};

// [dynpath2] number of fixup checkpoint items of a cut (tail boundaries that are neither S_A nor the piece-1 start); 0 without checkpoints
__device__ __forceinline__ int ck_items(int ckn, int tcj, int xc, int m, int s1) {
  if (ckn <= 0) return 0;
  const int k0 = (xc + ckn - 1) / ckn, kmax = (tcj - 1) / ckn, pb1 = m == 2 ? xc + s1 : -1;
  const bool saSp = xc % ckn == 0, psSp = m == 2 && pb1 % ckn == 0 && pb1 / ckn <= kmax;
  return max(0, kmax - k0 + 1) - (saSp ? 1 : 0) - (psSp ? 1 : 0);
}

// ---------------------------------------------------------------- lean planner (N <= 32 sequences, no context-parallel branch)
// The general kernel above is cold-instruction-fetch / barrier-latency bound (~20 us in situ); the lean kernels touch a few hundred instructions and 1-4 barriers.
constexpr int LN_MAX = 256;       // sequences handled by the lean planners (warp-shuffle prologue up to 32, block-parallel above)

// block-parallel prologue for 32 < N <= LN_MAX: rank by chunk count (descending, ties by index) -> s_sidx / s_sc, exclusive group-cost prefix s_Pg (optional), total, longest and shortest.
// `parts` consecutive lanes split the comparison loop of one sequence (parts * N <= NTHR).  Ends with a barrier.
__device__ __forceinline__ int wcost32_(int n) { return n * 100 + (int)C_ENTRY + (n < PRELOAD_MIN ? (int)C_SHORT : 0); }
__device__ __noinline__ void lean_prologue_big(const long long* __restrict__ cu, int bad, int N, int H, int* s_c, int* s_sidx, int* s_sc, int* s_Pg, int* s_W, int* s_wmax, int* s_cmin) {
  const int tid = threadIdx.x;
  if (tid < N) { const long long len = bad ? 0 : cu[tid + 1] - cu[tid]; s_c[tid] = len > 0 ? (int)((len + 31) >> 5) : 0; }
  __syncthreads();
  int parts = 32;
  while (parts * N > (int)blockDim.x) parts >>= 1;
  const int i = tid / parts, part = tid & (parts - 1);
  int r = 0, pre = 0, tot = 0, cmax = 0, cmin = 1 << 30, ci = 0;
  if (i < N) {
    ci = s_c[i];
    for (int j = part; j < N; j += parts) {
      const int cj = s_c[j];
      const bool before = (cj > ci) || (cj == ci && j < i);
      const int gc = H * wcost32_(cj);
      r += before; pre += before ? gc : 0; tot += gc; cmax = max(cmax, cj); cmin = min(cmin, cj);
    }
  }
  for (int off = parts >> 1; off > 0; off >>= 1) {
    r += __shfl_xor_sync(0xffffffffu, r, off); pre += __shfl_xor_sync(0xffffffffu, pre, off); tot += __shfl_xor_sync(0xffffffffu, tot, off);
    cmax = max(cmax, __shfl_xor_sync(0xffffffffu, cmax, off)); cmin = min(cmin, __shfl_xor_sync(0xffffffffu, cmin, off));
  }
  if (i < N && part == 0) {
    s_sidx[r] = i; s_sc[r] = ci; if (s_Pg) s_Pg[r] = pre;
    if (i == 0) { if (s_W) *s_W = tot; if (s_wmax) *s_wmax = wcost32_(cmax); if (s_cmin) *s_cmin = cmin; if (s_Pg) s_Pg[N] = tot; }
  }
  __syncthreads();
}
// rank of sequence `tid` (warp 0) by chunk count descending, ties by index; s_sidx[rank] = sequence, s_sc[rank] = chunks
__device__ __forceinline__ void lean_rank(const long long* __restrict__ cu, int bad, int N, int* s_sidx, int* s_sc, int* c0o) {
  const int tid = threadIdx.x;
  if (tid < 32) {
    int c = 0;
    if (tid < N) { const long long len = bad ? 0 : cu[tid + 1] - cu[tid]; c = len > 0 ? (int)((len + 31) >> 5) : 0; }
    int r = 0;
    for (int j = 0; j < N; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); r += (cj > c) || (cj == c && j < tid); }
    if (tid < N) { s_sidx[r] = tid; s_sc[r] = c; if (c0o) c0o[tid] = c; }
  }
}

// [contract2] sanitize cu_seqlens in a lean kernel: a private copy goes to cu_out (read by the kernels of the launch); a cu that is not non-decreasing inside [0, Ttok] (starting at 0)
// is replaced by zeros (every sequence empty, pstat = 1).  Returns the block-uniform `bad` flag; contains barriers.
__device__ __forceinline__ int lean_cu_check(const long long* __restrict__ cu, int N, long long* cu_out, long long Ttok, int* pstat) {
  if (cu_out == nullptr) return 0;
  const int tid = threadIdx.x;
  int bad = 0;
  for (int i = tid; i <= N; i += blockDim.x) {
    const long long r0 = cu[i], r1 = i < N ? cu[i + 1] : r0;
    if (r0 < 0 || r0 > Ttok || r1 < r0 || (i == 0 && r0 != 0)) bad = 1;
    cu_out[i] = r0;
  }
  bad = __syncthreads_or(bad);
  if (bad) for (int i = tid; i <= N; i += blockDim.x) cu_out[i] = 0;
  if (pstat != nullptr && tid == 0) *pstat = bad;
  return bad;
}

__global__ void __launch_bounds__(NTHR) plan_lean_plain_kernel(const long long* __restrict__ cu, int N, int H, int ncta, int* __restrict__ task_list, int* __restrict__ cta_ptr,
                                                               long long* dbg, long long* cu_out, long long Ttok, int* pstat, kdaval::VArgs vv, int nlate) {
  extern __shared__ __align__(16) unsigned char dsm_v[];
  __shared__ int s_sidx[LN_MAX], s_sc[LN_MAX], s_c[LN_MAX];
  asm volatile("griddepcontrol.launch_dependents;");   // [pdlp] a PDL-launched plain kernel behind this planner may start its prologue now
  const long long t0 = clock64();
  const int tid = threadIdx.x;
  if (blockIdx.x >= 2) { kdaval::pad_block(vv, blockIdx.x - 2); return; }
  if (blockIdx.x == 1) { const bool okf = kdaval::validate_block_fast_ni(vv, (int*)dsm_v, false); if (!okf) kdaval::zero_out_if_bad(vv); return; }   // [contract2] table validation block
  if (N <= 31 && cu_out != nullptr) {
    // [chain] cu check + rank fused in warp 0: the N + 1 <= 32 entries live in one warp, so the block-wide OR (barrier), the second cu load of the rank and the second barrier are not needed
    if (tid < 32) {
      long long r0 = 0, r1 = 0; int b = 0;
      if (tid <= N) {
        r0 = cu[tid]; r1 = tid < N ? cu[tid + 1] : r0;
        if (r0 < 0 || r0 > Ttok || r1 < r0 || (tid == 0 && r0 != 0)) b = 1;
        cu_out[tid] = r0;
      }
      b = __any_sync(0xffffffffu, b);
      if (b && tid <= N) cu_out[tid] = 0;
      if (pstat != nullptr && tid == 0) *pstat = b;
      int c = 0;
      if (tid < N) { const long long len = b ? 0 : r1 - r0; c = len > 0 ? (int)((len + 31) >> 5) : 0; }
      int r = 0;
      #pragma unroll 1
      for (int j = 0; j < N; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); r += (cj > c) || (cj == c && j < tid); }
      if (tid < N) { s_sidx[r] = tid; s_sc[r] = c; }
    }
    __syncthreads();
  } else {
  const int bad = lean_cu_check(cu, N, cu_out, Ttok, pstat);
  if (N <= 32) { lean_rank(cu, bad, N, s_sidx, s_sc, nullptr); __syncthreads(); }
  else lean_prologue_big(cu, bad, N, H, s_c, s_sidx, s_sc, nullptr, nullptr, nullptr, nullptr);
  }
  // [pdlp] the last nlate CTAs of the grid start late (PDL: their SMs hold this planner's blocks): the snake deal covers the first nm = ncta - nlate CTAs and the smallest tasks (the end of the sorted
  // order) that do not fill a whole round go one each to the late CTAs
  const int T = N * H, nm = ncta - nlate;
  // [chain] exact divisions through a float reciprocal (T < 2^22: N <= 256 sequences) instead of the integer division sequences and the 64-bit division of the FastDiv constructor: ~1 k cycles of the planner tail
  const bool f22 = T < (1 << 22);
  const float inm = __frcp_rn((float)nm), iH = __frcp_rn((float)H);
  auto dv = [&](int a, float inv, int b) { if (!f22) return a / b; const int q = (int)((float)a * inv); const int r = a - q * b; return q + (r >= b) - (r < 0); };
  const int e = nlate > 0 ? min(nlate, T - dv(T, inm, nm) * nm) : 0, Tm = T - e, full = dv(Tm, inm, nm), rem = Tm - full * nm;
  const bool odd = (full & 1) != 0;
  for (int r = tid; r < T; r += blockDim.x) {     // snake deal of the tasks sorted by chunks: rank r is round q's pos-th deal
    const int g = dv(r, iH, H);
    if (r >= Tm) { task_list[r] = s_sidx[g] * H + (r - g * H); continue; }
    const int q = dv(r, inm, nm), pos = r - q * nm, k = (q & 1) ? nm - 1 - pos : pos;
    const int start = k * full + (odd ? max(0, k - (nm - rem)) : min(k, rem));
    task_list[start + q] = s_sidx[g] * H + (r - g * H);
  }
  for (int k = tid; k <= ncta; k += blockDim.x) cta_ptr[k] = k <= nm ? k * full + (odd ? max(0, k - (nm - rem)) : min(k, rem)) : min(T, Tm + (k - nm));
  if (dbg && tid == 0) dbg[3] = clock64() - t0;
}

// cio wrap plan, N <= 32, no CP: same decisions as plan_stage / expand_all above (tables are bit-identical, tools/leancmp.py).
// Passes: A1 stream position of every (variant, boundary); A2 boundary data + modelled CTA load of the pair (k, k + 1) -> variant maxima, E / a of every (variant, CTA);
// B counts + scans of the chosen variant; C entries.  Every quantity is computed once (the first version recomputed neighbours per thread: instruction-issue bound).
struct LeanBd { int E, a; int CE, w1, fc; };
__device__ __forceinline__ int wcost32(int n) { return n * 100 + (int)C_ENTRY + (n < PRELOAD_MIN ? (int)C_SHORT : 0); }
__device__ __forceinline__ int front_cost32(int b) { return b * 100 + (int)C_ENTRY + (int)C_FRONT + (b < PRELOAD_MIN ? (int)C_SHORT : 0); }
// [dynpath2] N * H <= ncta: one whole task per CTA (task t = sequence t / H, head t % H), no cuts or ordering to plan: the cio table is the identity (the wrap planner would produce the same
// one-task lists, ~8 us later in situ)
__device__ __noinline__ void ident_body(const long long* __restrict__ cu, int N, int H, int ncta, int n_pad, int* __restrict__ task_list, int* __restrict__ cta_ptr, int bad) {
  const int tid = threadIdx.x;
  for (int t = tid; t < N * H; t += NTHR) {
    const int g = t / H;
    const long long len = bad ? 0 : cu[g + 1] - cu[g];
    int* o = task_list + 4 * t;
    o[0] = t; o[1] = 0; o[2] = len > 0 ? (int)((len + 31) / 32) : 0; o[3] = 0;
  }
  for (int k = tid; k <= n_pad; k += NTHR) cta_ptr[k] = k <= ncta ? k : ncta;
}

__device__ __noinline__ void lean_cio_body(const long long* __restrict__ cu, int N, int H, int ncta, int slot_off, int n_pad, int* __restrict__ task_list, int* __restrict__ cta_ptr,
                                           long long* dbg, unsigned char* lm, int bad) {
  __shared__ int s_sidx[LN_MAX], s_sc[LN_MAX], s_c[LN_MAX];
  __shared__ int s_Pg[LN_MAX + 1], s_W, s_wmax;
  __shared__ int s_cmin;
  __shared__ unsigned int s_vM[NVAR];
  __shared__ int s_wsum[2][8];
  int (*s_t)[NB] = (int (*)[NB])lm; int (*s_dl)[NB] = s_t + NVAR; int (*s_Ev)[NB] = s_dl + NVAR; int (*s_av)[NB] = s_Ev + NVAR;
  int* s_ptr = (int*)(s_av + NVAR); int* s_slot = s_ptr + (MAXC + 2);
  const long long t0 = clock64();
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int T = N * H;
  const FastDiv fd(H);
  if (tid < NVAR) s_vM[tid] = 0;
  if (N > 32) lean_prologue_big(cu, bad, N, H, s_c, s_sidx, s_sc, s_Pg, &s_W, &s_wmax, &s_cmin);
  else if (tid < 32) {   // rank, exclusive group-cost prefix, total and longest in one pass over the sequences
    int c = 0;
    if (tid < N) { const long long len = bad ? 0 : cu[tid + 1] - cu[tid]; c = len > 0 ? (int)((len + 31) >> 5) : 0; }
    int r = 0, cmax = 0, cmin = 1 << 30, pre = 0, tot = 0;
    for (int j = 0; j < N; j++) {
      const int cj = __shfl_sync(0xffffffffu, c, j);
      const bool before = (cj > c) || (cj == c && j < tid);
      const int gc = H * wcost32(cj);
      r += before; pre += before ? gc : 0; tot += gc; cmax = max(cmax, cj); cmin = min(cmin, cj);
    }
    if (tid < N) { s_sidx[r] = tid; s_sc[r] = c; s_Pg[r] = pre; }
    if (tid == 0) { s_W = tot; s_wmax = wcost32(cmax); s_Pg[N] = tot; s_cmin = cmin; }
  }
  __syncthreads();
  const int W = s_W;
  const int Lam0 = (W + (int)C_CUT * (ncta - 1)) / ncta;
  const int lbound = max(Lam0, s_wmax);
  const int lstep = max((int)((double)Lam0 * 0.005), 50);
  auto lam_of = [&](int v) -> int { return Lam0 + (v == 0 ? 0 : 1 << (v - 1)) * lstep; };
  const int nb1 = ncta - 1;
  // ---- A1: stream position of boundary i (1 .. ncta-1): task and offset inside it (dl < 0: outside the stream)
  for (int it = tid; it < NVAR * nb1; it += NTHR) {
    const int v = it / nb1, i = it - v * nb1 + 1;
    const int Lam = lam_of(v);
    const int p = (Lam - (int)C_BACK) + (i - 1) * (Lam - (int)C_CUT);
    int t, dl = -1;
    if (p >= W || p <= 0) t = p <= 0 ? 0 : T;
    else {
      int lo = 0, hi = N - 1;
      while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_Pg[mid] <= p) lo = mid; else hi = mid - 1; }
      const int w = wcost32(s_sc[lo]), q = p - s_Pg[lo];
      int h = (int)(__fdividef((float)q, (float)w));
      h = min(h, H - 1);
      if (h * w > q) h--; else if ((h + 1) * w <= q && h + 1 < H) h++;
      dl = q - h * w;
      t = lo * H + h;
    }
    s_t[v][i] = t; s_dl[v][i] = dl;
  }
  __syncthreads();
  // boundary data of the upper CTA j for variant v (j == 0: the end of the stream)
  auto bdata = [&](int v, int j) -> LeanBd {
    LeanBd b; b.E = T; b.a = 0; b.CE = W; b.w1 = 0; b.fc = 0;
    if (j == 0) return b;
    const int i = ncta - j;
    const int t0v = s_t[v][i], dl = s_dl[v][i];
    if (dl < 0) { b.E = t0v; b.CE = (t0v >= T) ? W : 0; return b; }
    const int Lam = lam_of(v);
    const int g = fd.div(t0v), n = s_sc[g], h = t0v - g * H;
    const int w = wcost32(n);
    const bool up = 2 * dl >= w;
    b.E = up ? t0v + 1 : t0v;
    b.CE = s_Pg[g] + (h + (up ? 1 : 0)) * w;
    if (n >= PRELOAD_MIN && dl >= (int)C_ENTRY && (i == 1 || s_t[v][i - 1] < t0v) && (i == ncta - 1 || s_t[v][i + 1] > t0v)) {
      const int aa = (dl - (int)C_ENTRY + 50) / 100;
      const int bn = n - aa;
      if (aa >= 1 && bn >= 1 && front_cost32(bn) + 2400 <= Lam - (aa * 100 + (int)C_ENTRY + (int)C_BACK)) { b.E = t0v; b.a = aa; b.CE = s_Pg[g] + h * w; b.w1 = w; b.fc = front_cost32(bn); }
    }
    return b;
  };
  // ---- A2
  for (int it = tid; it < NVAR * ncta; it += NTHR) {
    const int v = it / ncta, k = it - v * ncta;
    const LeanBd bk = bdata(v, k);
    int start_c = 0, fcost = 0;
    if (k < ncta - 1) { const LeanBd b1 = bdata(v, k + 1); start_c = b1.CE + b1.w1; fcost = b1.fc; }
    int ld = bk.CE - start_c + fcost;
    if (bk.a > 0) ld += bk.a * 100 + (int)C_ENTRY + (int)C_BACK;
    s_Ev[v][k] = bk.E; s_av[v][k] = bk.a;
    atomicMax(&s_vM[v], (unsigned int)ld);
  }
  __syncthreads();
  int bv = 0;
  for (int v = 1; v < NVAR; v++) if (s_vM[v] < s_vM[bv]) bv = v;
  const int best_M = (int)s_vM[bv];
  const int full = T / ncta, rem = T - full * ncta;
  const bool odd = (full & 1) != 0;
  // whole-task snake deal (tasks sorted by chunks, dealt boustrophedon: interleaved heads, no cuts): kept unless the wrap plan models >= 5% better.  Measured (700 x 24 tokens, H = 64):
  // the wrap plan, equal in the model, runs ~7% slower than a whole-task deal (126 cuts, contiguous head ranges per CTA); with ~3.7% modelled gain.  Layouts with few tasks per CTA gain
  // 10-15% from the wrap plan (8 x 1024, H = 64 / 96).
  bool snake = false;
  {
    __shared__ unsigned int s_Msn;
    if (tid == 0) s_Msn = 0;
    __syncthreads();
    if (tid < ncta) {
      int ld = 0;
      const int rounds = (T + ncta - 1) / ncta;
      if (rounds <= N) {
        for (int q = 0; q < rounds; q++) {
          const int r = q * ncta + ((q & 1) ? ncta - 1 - tid : tid);
          if (r < T) ld += wcost32(s_sc[fd.div(r)]);
        }
      } else {   // many tasks per CTA, few sequences: ranks of CTA k are the residues k and 2 ncta - 1 - k mod 2 ncta; count them per sequence group
        const int m2 = 2 * ncta, r1 = tid, r2 = m2 - 1 - tid;
        auto fl = [&](int x) { return x < 0 ? -1 : x / m2; };
        for (int g = 0; g < N; g++) {
          const int a = g * H, b = a + H;
          const int cnt = fl(b - 1 - r1) - fl(a - 1 - r1) + fl(b - 1 - r2) - fl(a - 1 - r2);
          ld += cnt * wcost32(s_sc[g]);
        }
      }
      atomicMax(&s_Msn, (unsigned int)ld);
    }
    __syncthreads();
    snake = !((long long)best_M * 105 < (long long)s_Msn * 100);
  }
  if (snake) {
    for (int r = tid; r < T; r += NTHR) {
      const int q = r / ncta, pos = r - q * ncta, k = (q & 1) ? ncta - 1 - pos : pos;
      const int g = fd.div(r);
      int* o = task_list + 4 * (k * full + (odd ? max(0, k - (ncta - rem)) : min(k, rem)) + q);
      o[0] = s_sidx[g] * H + (r - g * H); o[1] = 0; o[2] = s_sc[g]; o[3] = 0;
    }
    for (int k = tid; k <= n_pad; k += NTHR) cta_ptr[k] = k <= ncta ? k * full + (odd ? max(0, k - (ncta - rem)) : min(k, rem)) : T;
    if (dbg && tid == 0) dbg[3] = clock64() - t0;
    return;
  }
  // ---- B: per-CTA entry counts and cut flags of the chosen variant, block scans
  int cnt = 0, flag = 0;
  if (tid < ncta) {
    const int a_n = tid < ncta - 1 ? s_av[bv][tid + 1] : 0;
    const bool has_front = a_n > 0;
    const int start = (tid == ncta - 1) ? 0 : s_Ev[bv][tid + 1] + (has_front ? 1 : 0);
    cnt = (s_Ev[bv][tid] - start) + (s_av[bv][tid] > 0) + (has_front ? 1 : 0);
    flag = s_av[bv][tid] > 0;
  }
  int icnt = cnt, ifl = flag;    // forward inclusive scan of cnt, backward inclusive scan of the cut flags (slot = number of cuts in CTAs above)
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, icnt, off); if (lane >= off) icnt += a;
    const int b = __shfl_down_sync(0xffffffffu, ifl, off); if (lane + off < 32) ifl += b;
  }
  if (wid < 8) { if (lane == 31) s_wsum[0][wid] = icnt; if (lane == 0) s_wsum[1][wid] = ifl; }    // CTAs are indexed by tid < ncta <= 256: warps 8..31 hold none
  __syncthreads();
  if (tid < ncta) {
    int pre = 0, post = 0;
    for (int w = 0; w < wid; w++) pre += s_wsum[0][w];
    for (int w = wid + 1; w < ((ncta + 31) >> 5); w++) post += s_wsum[1][w];
    const int ptr = pre + icnt - cnt;
    s_ptr[tid] = ptr; s_slot[tid] = ifl + post - flag;
    cta_ptr[tid] = ptr;
    if (tid == ncta - 1) { s_ptr[ncta] = ptr + cnt; for (int k = ncta; k <= n_pad; k++) cta_ptr[k] = ptr + cnt; }
  }
  __syncthreads();
  const int total = s_ptr[ncta];
  const int* E = s_Ev[bv]; const int* a_ = s_av[bv];
  for (int en = tid; en < total; en += NTHR) {
    int lo = 0, hi = ncta - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_ptr[mid] <= en) lo = mid; else hi = mid - 1; }
    const int k = lo, p = en - s_ptr[k];
    const int a_k = a_[k], a_n = k < ncta - 1 ? a_[k + 1] : 0;
    const bool has_front = a_n > 0;
    const int start = (k == ncta - 1) ? 0 : E[k + 1] + (has_front ? 1 : 0);
    int* o = task_list + 4 * en;
    if (has_front && p == 0) {
      const int ti = E[k + 1], g = fd.div(ti), n = s_sc[g];
      o[0] = s_sidx[g] * H + (ti - g * H); o[1] = 0; o[2] = n - a_n; o[3] = (s_slot[k + 1] + 1 + slot_off) << 16;
    } else if (start + p - (has_front ? 1 : 0) < E[k]) {
      const int ti = start + p - (has_front ? 1 : 0), g = fd.div(ti);
      o[0] = s_sidx[g] * H + (ti - g * H); o[1] = 0; o[2] = s_sc[g]; o[3] = 0;
    } else {
      const int ti = E[k], g = fd.div(ti), n = s_sc[g];
      o[0] = s_sidx[g] * H + (ti - g * H); o[1] = n - a_k; o[2] = n; o[3] = s_slot[k] + 1 + slot_off;
    }
  }
  if (dbg && tid == 0) dbg[3] = clock64() - t0;
}


__global__ void __launch_bounds__(NTHR) plan_lean_cio_kernel(const long long* __restrict__ cu, int N, int H, int ncta, int* __restrict__ task_list, int* __restrict__ cta_ptr, long long* dbg,
                                                               long long* cu_out, long long Ttok, int* pstat, kdaval::VArgs vv) {
  extern __shared__ __align__(16) unsigned char dsm_l[];
  if (blockIdx.x >= 2) { kdaval::pad_block(vv, blockIdx.x - 2); return; }
  if (blockIdx.x == 1) { if (!kdaval::validate_block_fast_ni(vv, (int*)dsm_l, false)) kdaval::zero_out_if_bad(vv); return; }   // [contract2] table validation block
  const int bad = lean_cu_check(cu, N, cu_out, Ttok, pstat);
  lean_cio_body(cu, N, H, ncta, 0, ncta, task_list, cta_ptr, dbg, dsm_l, bad);
}

constexpr int LEAN_OFF = 98304;   // lean planner arrays in the dynamic shared memory of plan_kernel (general arrays need < 80 KB for N <= 32)

// dynamic shared memory: c0[N] c[N] sc[N] sidx[N] | Pg[N + 1] (8-byte) | per-variant arrays
__global__ void __launch_bounds__(NTHR) plan_kernel(const long long* __restrict__ cu, int N, int H, int ncta0, int* __restrict__ task_list,
                                                    int* __restrict__ cta_ptr, int plain_mode, CpOut cp, long long* dbg, long long* cu_out, long long Ttok, int* pstat, kdaval::VArgs vv) {
  const int plain = plain_mode & 1; const bool force_snake = (plain_mode & 2) != 0; const bool ident_on = (plain_mode & 8) != 0;   // [dynpath2] bit 3: identity head table when N * H <= ncta
  extern __shared__ __align__(16) unsigned char dsm[];
  __shared__ int s_G, s_snake, s_bestv, s_gsp, s_badcu;
  __shared__ int s_slot[NB], s_cnt[NB], s_cnt2[NB], s_cnt3[NB];
  __shared__ long long s_W, s_wmax, s_Msn, s_vM[NVAR], s_bestM;
  // CP search state
  __shared__ int s_nlo, s_nhi;
  __shared__ int s_cp, s_cj[2], s_ctc[2], s_cna[8], s_nna, s_cstep[2], s_cnc[2], s_cxc0[2][8], s_bj, s_bna, s_bxc, s_bs1, s_bs2;
  __shared__ long long s_cT[2];
  __shared__ double s_ccost[2][8][32];
  __shared__ double s_bcost; __shared__ unsigned long long s_bkey;
  __shared__ int s_fst[MAXC], s_fen[MAXC]; __shared__ long long s_raw[MAXC]; 
  int* c0 = (int*)dsm; int* c = c0 + N; int* sc = c + N; int* sidx = sc + N;
  long long* slen = (long long*)(dsm + (((size_t)(4 * N) * 4 + 7) & ~(size_t)7));      // sequence lengths (tokens)
  long long* Pg = slen + N;
  int* vt = (int*)(Pg + N + 1);                 // [NVAR][NB] boundary task index
  int* vdl = vt + NVAR * NB;                    // offset inside the task (cost units), -1: outside the stream
  int* vE = vdl + NVAR * NB;                    // [NVAR][NB] by upper CTA: end of the whole-task range
  int* va = vE + NVAR * NB;                     // back-piece chunks (0: no cut)
  long long* vCE = (long long*)(va + NVAR * NB + (NVAR * NB & 1));    // [NVAR][NB] stream cost prefix at E
  long long* vw1 = vCE + NVAR * NB;             // cost of the cut task (front piece: the stream continues after it), 0 without a cut
  long long* vfc = vw1 + NVAR * NB;             // front-piece cost of the cut at this CTA's lower boundary, 0 without a cut
  long long* vload = vfc + NVAR * NB;           // [NVAR][NB] modelled CTA loads (variant 0's row is reused by the snake model)
  const int tid = threadIdx.x;
  const long long t0 = clock64();
  if (blockIdx.x >= 2) { kdaval::pad_block(vv, blockIdx.x - 2); return; }
  if (blockIdx.x == 1) { if (!kdaval::validate_block_fast_ni(vv, (int*)dsm, false)) kdaval::zero_out_if_bad(vv); return; }   // [contract2] the validation block: slot / checkpoint tables (and a status word for cu_seqlens)

  const bool cp_on = cp.geom != nullptr;
  const int n_sms = cp_on ? cp.n_sms : ncta0;
  const FastDiv fd(H);

  // [contract2] cu_out != null: the kernels of this launch read a private sanitized copy of cu_seqlens.  A cu_seqlens that is not non-decreasing inside [0, Ttok] (starting at 0)
  // is replaced by all zeros: every sequence is empty (the launch only copies each state through), pstat[0] = 1.
  if (cu_out != nullptr && tid == 0) s_badcu = 0;
  if (cu_out != nullptr) __syncthreads();
  for (int i = tid; i < N; i += NTHR) {
    const long long r0 = cu[i], r1 = cu[i + 1];
    const long long len = r1 - r0;
    slen[i] = len;
    c0[i] = len > 0 ? (int)((len + 31) / 32) : 0;   // zero-length sequences stay in the stream as zero-chunk tasks (state copy-through)
    if (cu_out != nullptr) {
      if (r0 < 0 || r0 > Ttok || r1 < r0 || r1 > Ttok || (i == 0 && r0 != 0)) atomicOr(&s_badcu, 1);
      cu_out[i] = r0;
      if (i == N - 1) cu_out[N] = r1;
    }
  }
  __syncthreads();
  if (cu_out != nullptr && s_badcu) {
    for (int i = tid; i < N; i += NTHR) { slen[i] = 0; c0[i] = 0; cu_out[i] = 0; }
    if (tid == 0) cu_out[N] = 0;
    __syncthreads();
  }
  if (pstat != nullptr && tid == 0) *pstat = cu_out != nullptr ? s_badcu : 0;

  // ---- lean path (N in [2, 32], plan_mode bit 2): when the host cp_plan early exit applies (the longest task does not exceed the fluid share: no context-parallel cut can pay), the
  //      plan is the lean non-CP head plan + empty tail tables; the general search below (which would reach the same tables) is skipped.
  const int bad = (cu_out != nullptr && s_badcu) ? 1 : 0;   // [contract2] sanitized cu: all sequences empty
  if (cp_on && (plain_mode & 4)) {
    __shared__ int s_exit0;
    if (tid < 32) {
      long long len = -1; int c = 0;
      if (tid < N) { len = bad ? 0 : cu[tid + 1] - cu[tid]; c = len > 0 ? (int)((len + 31) >> 5) : 0; }
      double stc = c > 0 ? c + (c < 17 ? 5.0 : 1.0) : 0.0, sc1 = tid < N ? c + 1.0 : 0.0, wsum = tid < N ? wc_d(c) : 0.0;
      int mx = c;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) { stc += __shfl_xor_sync(0xffffffffu, stc, off); sc1 += __shfl_xor_sync(0xffffffffu, sc1, off); wsum += __shfl_xor_sync(0xffffffffu, wsum, off); mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off)); }
      // ranks by tokens (cut candidates j0 / j1: longest / second longest, ties lower index) and by chunks (i_c0: first sequence of the sorted stream)
      int rk_t = 0, rk_c = 0;
      for (int j = 0; j < N; j++) {
        const long long lj = __shfl_sync(0xffffffffu, len, j); const int cj = __shfl_sync(0xffffffffu, c, j);
        rk_t += (lj > len) || (lj == len && j < tid); rk_c += (cj > c) || (cj == c && j < tid);
      }
      const int j0 = __ffs(__ballot_sync(0xffffffffu, tid < N && rk_t == 0)) - 1, j1 = __ffs(__ballot_sync(0xffffffffu, tid < N && rk_t == 1)) - 1;
      const int i_c0 = __ffs(__ballot_sync(0xffffffffu, tid < N && rk_c == 0)) - 1, i_c1 = __ffs(__ballot_sync(0xffffffffu, tid < N && rk_c == 1)) - 1;
      const int sc_0 = __shfl_sync(0xffffffffu, c, i_c0), sc_1 = i_c1 >= 0 ? __shfl_sync(0xffffffffu, c, i_c1) : 0;
      const long long T_0 = __shfl_sync(0xffffffffu, len, j0 >= 0 ? j0 : 0), T_1 = __shfl_sync(0xffffffffu, len, j1 >= 0 ? j1 : 0);
      if (tid == 0) {
        const bool exit0 = (mx + 1.0 <= (1.0 + MIN_GAIN) * stc * H / n_sms);
        // a cut of sequence jq leaves the longest remaining whole task in the head group, so its cost is >= that task's modelled length (cp_single: a >= wmax_excl): when
        // even that exceeds the bar a cut must beat (host: direct * (1 - MIN_GAIN)) no cut can be selected
        const double bar = fmax(mx + 1.0, sc1 * H / n_sms) * (1.0 - MIN_GAIN) - cp.ovh;
        bool feas = false;
        for (int q = 0; q < 2; q++) {
          const int jq = q == 0 ? j0 : j1; const long long Tq = q == 0 ? T_0 : T_1;
          const int tc = (jq >= 0 && Tq >= 32 * 8) ? (int)((Tq + 31) / 32) : 0;
          const double wmx = (i_c0 == jq) ? (N > 1 ? wc_d(sc_1) : 0.0) : wc_d(sc_0);
          feas |= tc >= 8 && wmx <= bar && cp_lb_ok(H, n_sms, (wsum - wc_d(tc)) * H, tc, bar);
        }
        s_exit0 = (exit0 || !feas) ? 1 : 0;
      }
    }
    __syncthreads();
    if (s_exit0) {
      for (int b = tid; b <= n_sms; b += NTHR) { cp.dptr[b] = 0; cp.seg_ptr[b] = 0; }
      if (tid == 0) { cp.geom[1] = 0; cp.geom[7] = 0; }
      if (N * H <= ncta0 && ident_on) ident_body(cu, N, H, ncta0, n_sms, task_list, cta_ptr, bad);
      else lean_cio_body(cu, N, H, ncta0, cp.slot_off, n_sms, task_list, cta_ptr, dbg, dsm + LEAN_OFF, bad);
      return;
    }
  }


  // ---- stage: sort the sequences by chunk count (sequence ov_seq counted as ov_chunks), prefix costs
  auto sort_stage = [&](int ov_seq, int ov_chunks) {
    for (int i = tid; i < N; i += NTHR) c[i] = (i == ov_seq) ? ov_chunks : c0[i];
    __syncthreads();
    for (int i = tid; i < N; i += NTHR) {   // rank sort: chunks descending, ties by index
      const int ci = c[i];
      int r = 0;
      for (int jx = 0; jx < N; jx++) { const int cj = c[jx]; r += (cj > ci) || (cj == ci && jx < i); }
      sc[r] = ci; sidx[r] = i;
    }
    __syncthreads();
    if (tid < 32) {   // Pg = exclusive prefix of the group costs H * wcost; W, wmax
      const int per = (N + 31) / 32, b0 = tid * per, b1 = min(N, b0 + per);
      long long s = 0;
      for (int i = b0; i < b1; i++) s += (long long)H * wcost(sc[i]);
      long long inc = s;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) { const long long o = __shfl_up_sync(0xffffffffu, inc, off); if (tid >= off) inc += o; }
      long long run = inc - s;
      for (int i = b0; i < b1; i++) { Pg[i] = run; run += (long long)H * wcost(sc[i]); }
      if (tid == 31) { s_W = inc; s_wmax = wcost(sc[0]); Pg[N] = inc; s_gsp = -1; }
    }
    __syncthreads();
    for (int g = tid; g < N; g += NTHR) if (sidx[g] == ov_seq) s_gsp = g;
    __syncthreads();
  };

  // ---- stage: plan the main (head group) launch on ncta CTAs; sp_g = group of the special (cut) sequence, which is never cut into pieces.  Results: s_bestM, s_bestv, s_snake
  auto plan_stage = [&](int ncta, int sp_g) {
    const int G = N, T = G * H;
    const long long W = s_W;
    const long long Lam0 = (long long)((double)(W + C_CUT * (ncta - 1)) / (double)ncta);   // (double: 64-bit integer division is ~1k cycles)
    const long long lbound = max(Lam0, s_wmax);
    const long long lstep = max((long long)((double)Lam0 * 0.005), 50LL);
    const int nb1 = ncta - 1;
    if (tid == 0) s_snake = 0;
    if (!plain) {
      // all variants: locate boundaries (boundary i in 1..ncta-1 sits between top-down CTAs i-1 and i, i.e. CTA indices ncta-i and ncta-i-1)
      for (int it = tid; it < NVAR * nb1; it += NTHR) {
        const int v = it / nb1, i = it - v * nb1 + 1;
        const long long Lam = Lam0 + (long long)(v == 0 ? 0 : 1 << (v - 1)) * lstep;
        const long long p = (Lam - C_BACK) + (long long)(i - 1) * (Lam - C_CUT);
        int t; int dl = 0;
        if (p >= W || p <= 0) { t = p <= 0 ? 0 : T; dl = -1; }
        else {
          int lo = 0, hi = G - 1;   // largest g with Pg[g] <= p
          while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (Pg[mid] <= p) lo = mid; else hi = mid - 1; }
          const long long w = wcost(sc[lo]), q = p - Pg[lo];
          int h = (int)(__fdividef((float)q, (float)w));
          h = min(h, H - 1);
          if ((long long)h * w > q) h--; else if ((long long)(h + 1) * w <= q && h + 1 < H) h++;
          t = lo * H + h; dl = (int)(q - (long long)h * w);
        }
        vt[v * NB + i] = t; vdl[v * NB + i] = dl;
      }
      __syncthreads();
      for (int it = tid; it < NVAR * nb1; it += NTHR) {
        const int v = it / nb1, i = it - v * nb1 + 1;
        const long long Lam = Lam0 + (long long)(v == 0 ? 0 : 1 << (v - 1)) * lstep;
        const int* tt = vt + v * NB;
        const int t = tt[i], dl = vdl[v * NB + i];
        int E, a = 0; long long CE, w1 = 0, fc = 0;
        if (dl < 0) { E = t; CE = (t >= T) ? W : 0; }                // outside the stream
        else {
          const int g = fd.div(t), n = sc[g], h = t - g * H;
          const long long w = wcost(n);
          const bool up = 2LL * dl >= w;
          E = up ? t + 1 : t;        // snap to the nearer task boundary
          CE = Pg[g] + (long long)(h + (up ? 1 : 0)) * w;
          const bool alone = (i == 1 || tt[i - 1] < t) && (i == ncta - 1 || tt[i + 1] > t);
          if (n >= PRELOAD_MIN && alone && g != sp_g && dl >= C_ENTRY) {
            const int aa = (int)((dl - C_ENTRY + U / 2) / U);
            const int bn = n - aa;
            if (aa >= 1 && bn >= 1 && front_cost(bn) + 24 * U <= Lam - ((long long)aa * U + C_ENTRY + C_BACK)) {
              E = t; a = aa; CE = Pg[g] + (long long)h * w; w1 = w; fc = front_cost(bn);
            }
          }
        }
        vE[v * NB + (ncta - i)] = E; va[v * NB + (ncta - i)] = a;    // indexed by the upper CTA j = ncta - i
        vCE[v * NB + (ncta - i)] = CE; vw1[v * NB + (ncta - i)] = w1; vfc[v * NB + (ncta - i)] = fc;
      }
      for (int v = tid; v < NVAR; v += NTHR) { vE[v * NB] = T; va[v * NB] = 0; vCE[v * NB] = W; vw1[v * NB] = 0; vfc[v * NB] = 0; }
      for (int v = tid; v < NVAR; v += NTHR) s_vM[v] = 0;
      __syncthreads();
      // modelled CTA loads per variant (+ max by shared atomics)
      for (int it = tid; it < NVAR * ncta; it += NTHR) {
        const int v = it / ncta, k = it - v * ncta;
        const long long* CE = vCE + v * NB; const long long* w1 = vw1 + v * NB; const int* a_ = va + v * NB;
        long long start_c = 0, fcost = 0;
        if (k < ncta - 1) { start_c = CE[k + 1] + w1[k + 1]; fcost = vfc[v * NB + k + 1]; }
        long long ld = CE[k] - start_c + fcost;
        if (a_[k] > 0) ld += (long long)a_[k] * U + C_ENTRY + C_BACK;
        vload[v * NB + k] = ld;
        atomicMax((unsigned long long*)&s_vM[v], (unsigned long long)ld);
      }
      __syncthreads();
      if (tid == 0) {
        int bv = 0;
        for (int v = 1; v < NVAR; v++) if (s_vM[v] < s_vM[bv]) bv = v;
        s_bestv = bv; s_bestM = s_vM[bv];
      }
      __syncthreads();
    } else if (tid == 0) { s_bestv = 0; s_bestM = 1LL << 60; }
    __syncthreads();
    const long long best_M = s_bestM;
    // few tasks per CTA: whole-task snake deal when the wrap boundaries cannot balance
    if (plain || force_snake || best_M * 1000 > lbound * 1015) {
      for (int k = tid; k < ncta; k += NTHR) vload[k] = 0;     // variant 0's load row is free again
      __syncthreads();
      for (int r = tid; r < T; r += NTHR) {
        const int q = r / ncta, pos = r - q * ncta, k = (q & 1) ? ncta - 1 - pos : pos;
        atomicAdd((unsigned long long*)&vload[k], (unsigned long long)wcost(sc[fd.div(r)]));
      }
      __syncthreads();
      if (tid < 32) {
        long long m = 0;
        for (int k = tid; k < ncta; k += 32) m = max(m, vload[k]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (tid == 0) { s_Msn = m; s_snake = (plain || force_snake || m < best_M) ? 1 : 0; if (m < best_M) s_bestM = m; }
      }
      __syncthreads();
    }
  };

  // ---- stage: expand the plan into the cio / plain task table (cta_ptr padded to n_pad CTAs); special sequence sp_g: whole-task entries hand their final state to scratch
  //      slot `head` (the S_A of the context-parallel cut); cut-piece slots start at slot_off.  With cp_tail the dual-chain tail table, the fixup segment table and the
  //      tail geometry are built in the same phases (independent warps / loops).
  auto expand_all = [&](int ncta, int sp_g, int slot_off, int n_pad, bool cp_tail) {
    const int G = N, T = G * H;
    const bool snake = s_snake != 0;
    const int bv = s_bestv;
    const int* E = vE + bv * NB; const int* a_ = va + bv * NB;
    // CP tail parameters
    const int q = cp_tail ? s_bj : 0, jq = cp_tail ? s_cj[q] : 0, xc = cp_tail ? s_bxc : 0, n_A = cp_tail ? s_bna : 0, n_D = n_sms - n_A;
    const long long x = cp_tail ? cu[jq] + 32LL * xc : 0, e_end = cp_tail ? cu[jq + 1] : 0, tail = e_end - x;
    const int m = cp_tail ? (s_bs1 ? 2 : 1) : 0;
    const int tph = cp_tail ? (int)((tail + 127) / 128) : 0;
    const int pt1 = m == 2 ? s_bs1 / 4 : (tph > 0 ? tph : 1);
    const int Gf = n_sms, fullh = H / Gf, remh = H % Gf;
    const int qb = n_D > 0 ? H / n_D : 0, rb = n_D > 0 ? H % n_D : 0, clo = n_D - rb, chi = rb;
    const double sw = m == 1 ? FIX_STATE_TILES : FIX_OWNER_TILES;
    const double Kf = sw + (m - 1) * FIX_CHAIN_TILES + (tph - pt1) + (cp_tail ? ck_items(cp.ck_n, s_ctc[q], xc, m, s_bs1) * FIX_STATE_TILES : 0.0);   // [dynpath2] + the checkpoint items of every head (host fix_schedule extra)
    const int n0 = pt1 * H;
    const double Utot = (double)n0 + ((double)H) * Kf;
    // ---- phase 1: per-CTA counts
    if (cp_tail && tid == 0) {   // piece-1 tasks per load level of the dual launch (rounds of the two-level merge)
      int nlo = 0, nhi = 0;
      if (m == 2) {
        const double t1 = DUAL_F * s_bs1 + DUAL_TASK, t2 = DUAL_F * s_bs2 + DUAL_TASK;
        int remaining = H, kl = 1, kh = 1;
        while (remaining > 0) {
          const double vlo = clo > 0 ? qb * t1 + kl * t2 : 1e300, vhi = chi > 0 ? (qb + 1) * t1 + kh * t2 : 1e300;
          if (vhi <= vlo) { const int tk = min(remaining, chi); nhi += tk; remaining -= tk; kh++; } else { const int tk = min(remaining, clo); nlo += tk; remaining -= tk; kl++; }
        }
      }
      s_nlo = nlo; s_nhi = nhi;
      cp.cu_tail[0] = x; cp.cu_tail[1] = m == 2 ? x + 32LL * s_bs1 : e_end; cp.cu_tail[2] = e_end;
      cp.geom[0] = (int)x; cp.geom[1] = m; cp.geom[2] = (int)e_end; cp.geom[3] = jq; cp.geom[4] = n_A; cp.geom[5] = n_D; cp.geom[6] = xc; cp.geom[7] = 1;
      if (cp.ck_n > 0) {   // [dynpath2] checkpoint boundaries b = kk * ck_n of the cut sequence beyond the cut: b == xc is S_A, b == the piece-1 start is S_in(0) (plain copies), the others are fixup items
        const int ckn = cp.ck_n, tcj = s_ctc[q], k0 = (xc + ckn - 1) / ckn, kmax = (tcj - 1) / ckn;
        const int pb1 = m == 2 ? xc + s_bs1 : -1;
        const bool saSp = xc % ckn == 0, psSp = m == 2 && pb1 % ckn == 0 && pb1 / ckn <= kmax;
        const int total = max(0, kmax - k0 + 1);
        cp.geom[8] = total - (saSp ? 1 : 0) - (psSp ? 1 : 0); cp.geom[9] = k0; cp.geom[10] = (saSp ? 1 : 0) + (psSp ? 1 : 0);
        cp.geom[12] = xc; cp.geom[13] = k0; cp.geom[14] = (int)x;                       // the dual kernel's ck_geom {xc, k0, x}
        int nc = 0;
        if (saSp) { cp.ccopy[2 * nc] = xc / ckn; cp.ccopy[2 * nc + 1] = -1; nc++; }
        if (psSp) { cp.ccopy[2 * nc] = pb1 / ckn; cp.ccopy[2 * nc + 1] = 0; nc++; }
      }
    } else if (cp.geom && tid == 0) { cp.geom[1] = 0; cp.geom[7] = 0; }
    if (cp.cu_a && tid == 0) { cp.cu_a[0] = cu_out != nullptr && s_badcu ? 0 : cu[0]; cp.cu_a[1] = cp_tail ? x : (cu_out != nullptr && s_badcu ? 0 : cu[1]); }
    if (snake) {
      const int full = T / ncta, rem = T - full * ncta;   // rounds 0..full-1 are complete; round `full` deals the first `rem` ranks
      for (int k = tid; k < ncta; k += NTHR) s_cnt[k] = full + (((full & 1) ? ncta - 1 - k : k) < rem ? 1 : 0);
    } else {
      for (int k = tid; k < ncta; k += NTHR) {
        const bool has_front = k < ncta - 1 && a_[k + 1] > 0;
        const int start = (k == ncta - 1) ? 0 : E[k + 1] + (has_front ? 1 : 0);
        s_cnt[k] = (E[k] - start) + (a_[k] > 0) + has_front;
        s_slot[ncta - 1 - k] = a_[k] > 0 ? 1 : 0;   // reversed for the slot scan
      }
    }
    if (cp.geom) {
      for (int b = tid; b < Gf; b += NTHR) {
        const int Cobb = (b + 1) * fullh + min(b + 1, remh);     // heads owned by CTAs 0..b
        s_raw[b] = cp_tail ? (long long)nearbyint((double)(b + 1) * Utot / Gf - (double)Cobb * Kf) : 0;
      }
    }
    __syncthreads();
    // dual per-CTA counts (need nlo / nhi)
    if (cp_tail) {
      const int nlo = s_nlo, nhi = s_nhi;
      for (int b = tid; b < Gf; b += NTHR) {
        int cnt = 0;
        if (b < n_D) {
          cnt = (b < rb) ? qb + 1 : qb;                                   // piece-0 tasks b, b + n_D, ...
          if (m == 2) cnt += (b < rb) ? (nhi > b ? (nhi - 1 - b) / chi + 1 : 0) : (nlo > b - rb ? (nlo - 1 - (b - rb)) / clo + 1 : 0);
        }
        s_cnt2[b] = cnt;
      }
    } else if (cp.geom) {
      for (int b = tid; b < Gf; b += NTHR) s_cnt2[b] = 0;
    }
    __syncthreads();
    // ---- phase 2: scans, one warp each
    if (tid < 32) {
      if (!snake) warp_scan_excl(s_slot, ncta);
      const int tot = warp_scan_excl(s_cnt, ncta);
      if (tid == 0) s_cnt[ncta] = tot;
      __syncwarp();
      for (int k = tid; k <= n_pad; k += 32) cta_ptr[k] = k <= ncta ? s_cnt[k] : tot;
    } else if (cp.geom && tid < 64) {
      const int lane = tid - 32;
      const int per = (Gf + 31) / 32, b0 = lane * per, b1 = min(Gf, b0 + per);
      int sm = 0;
      for (int i = b0; i < b1; i++) sm += s_cnt2[i];
      int inc = sm;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) { const int o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += o; }
      int run = inc - sm;
      for (int i = b0; i < b1; i++) { const int v = s_cnt2[i]; s_cnt2[i] = run; run += v; }
      const int tot = __shfl_sync(0xffffffffu, inc, 31);
      if (lane == 0) s_cnt2[Gf] = tot;
      __syncwarp();
      for (int b = lane; b <= Gf; b += 32) cp.dptr[b] = s_cnt2[b];
    } else if (cp.geom && tid < 96) {
      const int lane = tid - 64;
      const int per = (Gf + 31) / 32, b0 = lane * per, b1 = min(Gf, b0 + per);
      if (cp_tail) {   // fixup segments: running max of raw (the monotone clamp of fix_schedule) -> per-CTA tile range and segment count
        long long mx = 0;
        for (int i = b0; i < b1; i++) mx = max(mx, s_raw[i]);
        long long inc = mx;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) { const long long o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc = max(inc, o); }
        long long run = __shfl_up_sync(0xffffffffu, inc, 1); if (lane == 0) run = 0;      // max over the CTAs of earlier lanes
        const FastDiv fdp(pt1);
        for (int b = b0; b < b1; b++) {
          const long long start_b = min((long long)n0, run);
          run = max(run, s_raw[b]);
          long long end_b = b == Gf - 1 ? (long long)n0 : min((long long)n0, run);
          if (end_b < start_b) end_b = start_b;
          const int owned = fullh + (b < remh ? 1 : 0);
          const int own_segs = owned * (m - 1);
          const int p0_segs = end_b > start_b ? fdp.div((int)end_b - 1) - fdp.div((int)start_b) + 1 : 0;
          s_cnt3[b] = own_segs + p0_segs;
          s_fst[b] = (int)start_b; s_fen[b] = (int)end_b;
        }
      } else {
        for (int b = b0; b < b1; b++) s_cnt3[b] = 0;
      }
      __syncwarp();
      int sm = 0;
      for (int i = b0; i < b1; i++) sm += s_cnt3[i];
      int inc = sm;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) { const int o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += o; }
      int run2 = inc - sm;
      for (int i = b0; i < b1; i++) { const int v = s_cnt3[i]; s_cnt3[i] = run2; run2 += v; }
      const int tot = __shfl_sync(0xffffffffu, inc, 31);
      if (lane == 0) s_cnt3[Gf] = tot;
      __syncwarp();
      for (int b = lane; b <= Gf; b += 32) cp.seg_ptr[b] = s_cnt3[b];
    }
    __syncthreads();
    // ---- phase 3: entries
    if (snake) {
      for (int r = tid; r < T; r += NTHR) {     // entry-parallel: rank r is round qq's pos-th deal
        const int qq = r / ncta, pos = r - qq * ncta, k = (qq & 1) ? ncta - 1 - pos : pos;
        const int g = fd.div(r);
        if (plain) { task_list[s_cnt[k] + qq] = sidx[g] * H + (r - g * H); continue; }      // plain kernel: task ids only
        int* o = task_list + 4 * (s_cnt[k] + qq);
        o[0] = sidx[g] * H + (r - g * H); o[1] = 0; o[2] = sc[g]; o[3] = g == sp_g ? ((r - g * H + 1) << 16) : 0;
      }
    } else {
      const int total = s_cnt[ncta];
      for (int en = tid; en < total; en += NTHR) {     // entry-parallel: entry en of the flat list -> its CTA (binary search over cta_ptr) and place
        int lo = 0, hi = ncta - 1;
        while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_cnt[mid] <= en) lo = mid; else hi = mid - 1; }
        const int k = lo, p = en - s_cnt[k];
        const bool has_front = k < ncta - 1 && a_[k + 1] > 0;
        const int start = (k == ncta - 1) ? 0 : E[k + 1] + (has_front ? 1 : 0);
        int* o = task_list + 4 * en;
        if (has_front && p == 0) {      // front piece: chunks [0, n - a) of the task cut at the CTA above
          const int ti = E[k + 1], g = fd.div(ti), n = sc[g];
          o[0] = sidx[g] * H + (ti - g * H); o[1] = 0; o[2] = n - a_[k + 1]; o[3] = (s_slot[ncta - 2 - k] + 1 + slot_off) << 16;
        } else if (start + p - (has_front ? 1 : 0) < E[k]) {   // whole task
          const int ti = start + p - (has_front ? 1 : 0), g = fd.div(ti);
          o[0] = sidx[g] * H + (ti - g * H); o[1] = 0; o[2] = sc[g]; o[3] = g == sp_g ? ((ti - g * H + 1) << 16) : 0;
        } else {                        // back piece: chunks [n - a, n)
          const int ti = E[k], g = fd.div(ti), n = sc[g];
          o[0] = sidx[g] * H + (ti - g * H); o[1] = n - a_[k]; o[2] = n; o[3] = s_slot[ncta - 1 - k] + 1 + slot_off;
        }
      }
    }
    if (cp_tail) {
      const int nlo = s_nlo, nhi = s_nhi;
      for (int b = tid; b < n_D; b += NTHR) {   // dual lists: piece-0 tasks, then the piece-1 tasks of the CTA's level (identical tasks: level slots in index order)
        int* o = cp.dtask + s_cnt2[b];
        for (int t = b; t < H; t += n_D) *o++ = t;
        if (m == 2) {
          if (b < rb) { for (int j = b; j < nhi; j += chi) *o++ = H + j; }
          else { for (int j = b - rb; j < nlo; j += clo) *o++ = H + nhi + j; }
        }
      }
      if (cp.ck_n > 0) {   // [dynpath2] checkpoint items of the fixup (the plain-copy rows were written with the geometry)
        const int ckn = cp.ck_n, tcj = s_ctc[q], k0 = (xc + ckn - 1) / ckn, kmax = (tcj - 1) / ckn, pb1 = m == 2 ? xc + s_bs1 : -1;
        const bool saSp = xc % ckn == 0, psSp = m == 2 && pb1 % ckn == 0 && pb1 / ckn <= kmax;
        const int kk1 = psSp ? pb1 / ckn : -1;
        for (int kk = k0 + tid; kk <= kmax; kk += NTHR) {
          const int b = kk * ckn;
          if ((saSp && b == xc) || (psSp && kk == kk1)) continue;
          const int p = (m == 2 && b >= pb1) ? 1 : 0;
          const int idx = (kk - k0) - ((saSp && kk > k0) ? 1 : 0) - ((psSp && kk > kk1) ? 1 : 0);
          int* o = cp.cki + 4 * idx;
          o[0] = kk - k0; o[1] = p; o[2] = kk; o[3] = 0;
        }
      }
      const FastDiv fdp(pt1);
      for (int b = tid; b < Gf; b += NTHR) {    // fixup segments of CTA b: its heads' later pieces, then its range of piece-0 tiles cut at head boundaries
        int* o = cp.seg + 4 * s_cnt3[b];
        for (int h = b; h < H; h += Gf)
          for (int k = 1; k < m; k++) { o[0] = h; o[1] = k; o[2] = pt1; o[3] = tph - pt1; o += 4; }      // m == 2: piece 1 = tiles [pt1, tph)
        int u = s_fst[b];
        const int end_b = s_fen[b];
        while (u < end_b) {
          const int h = fdp.div(u);
          const int ee = min(end_b, (h + 1) * pt1);
          o[0] = h; o[1] = 0; o[2] = u - h * pt1; o[3] = ee - u; o += 4;
          u = ee;
        }
      }
    }
    __syncthreads();
  };

  // ================================================================ main plan
  sort_stage(-1, 0);
  if (dbg && tid == 0) dbg[1] = clock64() - t0;
  if (tid == 0) s_cp = 0;
  __syncthreads();
  if (cp_on) {
    // ---- context-parallel decision (kernel.py cp_plan with the dual-chain tail only; the head group P + S is one cio launch)
    if (tid == 0) {
      s_cp = 0;
      // the two longest sequences by tokens (ties: lower index)
      int j0 = -1, j1 = -1; long long l0 = -1, l1 = -1;
      for (int i = 0; i < N; i++) {
        const long long len = slen[i];
        if (len > l0) { j1 = j0; l1 = l0; j0 = i; l0 = len; } else if (len > l1) { j1 = i; l1 = len; }
      }
      s_cj[0] = j0; s_cT[0] = l0; s_cj[1] = j1; s_cT[1] = l1;
      // early exit: CP only pays when the longest task leaves SMs idle (host: max(chunks) + 1 <= 1.04 * sum(tcost) * H / n_sms)
      double stc = 0.0, sc1 = 0.0; int mx = 0;
      double wsum_all = 0.0;
      for (int i = 0; i < N; i++) { const int ch = c0[i]; stc += ch > 0 ? ch + (ch < 17 ? 5.0 : 1.0) : 0.0; sc1 += ch + 1.0; wsum_all += wc_d(ch); mx = max(mx, ch); }
      const bool exit0 = (mx + 1.0 <= (1.0 + MIN_GAIN) * stc * H / n_sms);
      const double direct = fmax(mx + 1.0, sc1 * H / n_sms);   // host: min(planned makespan, fluid / longest-task bound) = the bound whenever CP can pay
      s_bcost = direct * (1.0 - MIN_GAIN) - cp.ovh;       // CP must beat this
      s_bkey = ~0ull;
      // candidate head-group budgets n_A (sorted, even-adjusted), host list
      int cand[7] = {H, n_sms / 2, (2 * n_sms) / 3, H + (n_sms - H) / 4, H + (n_sms - H) / 3, n_sms - H, n_sms - (H + 1) / 2};
      for (int a = 0; a < 7; a++) for (int b = a + 1; b < 7; b++) if (cand[b] < cand[a]) { const int t = cand[a]; cand[a] = cand[b]; cand[b] = t; }
      int nn = 0;
      for (int a = 0; a < 7; a++) {
        int n_A = cand[a];
        if (n_A > 2) n_A -= n_A % 2;
        if (n_A < 1 || (n_sms - n_A) / 2 < 1) continue;
        if (nn == 0 || s_cna[nn - 1] != n_A) s_cna[nn++] = n_A;     // dedupe (the host evaluates equal budgets twice with identical results)
      }
      bool lb_any = false;   // [dynpath2] some cut can beat the bar at all (cp_lb_ok)
      for (int q = 0; q < 2; q++) {
        const int jq = s_cj[q]; const long long Tq = s_cT[q];
        const int tcq = (jq >= 0 && Tq >= 32 * 8) ? (int)((Tq + 31) / 32) : 0;
        if (tcq >= 8) lb_any |= cp_lb_ok(H, n_sms, (wsum_all - wc_d(c0[jq])) * H, tcq, s_bcost);
      }
      s_nna = (exit0 || !lb_any) ? 0 : nn;
      for (int q = 0; q < 2; q++) {
        const int jq = s_cj[q];
        const long long Tq = s_cT[q];
        s_ctc[q] = (jq >= 0 && Tq >= 32 * 8) ? (int)((Tq + 31) / 32) : 0;
        const int tc = s_ctc[q];
        s_cstep[q] = tc >= 8 ? max(4, (tc - 7) / 24) : 4;
        int cnt = 0;
        if (tc >= 8) for (int xc = 4; xc < tc - 3; xc += s_cstep[q]) cnt++;
        s_cnc[q] = min(cnt, 32);
      }
    }
    __syncthreads();
    if (dbg && tid == 0) dbg[21] = clock64() - t0;
    const int nna = s_nna;
    auto wexcl_of = [&](int jq) -> long long { return s_W - (long long)H * wcost(c0[jq]); };
    // longest task excluding the cut sequence (the stream is sorted by chunks: sc[0] unless the cut sequence is that one)
    auto wmx_of = [&](int jq) -> double { return (sidx[0] == jq) ? (N > 1 ? wc_d(sc[1]) : 0.0) : wc_d(sc[0]); };
    // ---- stage 1: single-piece cost (closed form) for every (sequence q, budget n_A) on a flat grid of cut points xc in [4, tc - 3) with stride ceil(range / 160)
    __shared__ double s_wc[32]; __shared__ unsigned long long s_wk[32];
    auto block_min = [&](double mycost, unsigned long long mykey, double* outc, unsigned long long* outk) {
      for (int off = 16; off > 0; off >>= 1) {
        const double oc = __shfl_xor_sync(0xffffffffu, mycost, off); const unsigned long long ok = __shfl_xor_sync(0xffffffffu, mykey, off);
        if (oc < mycost || (oc == mycost && ok < mykey)) { mycost = oc; mykey = ok; }
      }
      if ((tid & 31) == 0) { s_wc[tid >> 5] = mycost; s_wk[tid >> 5] = mykey; }
      __syncthreads();
      if (tid == 0) {
        double bc = 1e300; unsigned long long bk = ~0ull;
        for (int w = 0; w < NTHR / 32; w++) if (s_wc[w] < bc || (s_wc[w] == bc && s_wk[w] < bk)) { bc = s_wc[w]; bk = s_wk[w]; }
        *outc = bc; *outk = bk;
      }
      __syncthreads();
    };
    __shared__ double s_r1c; __shared__ unsigned long long s_r1k;
    {
      unsigned long long mykey = ~0ull; double mycost = 1e300;
      for (int it = tid; it < 2 * 8 * 160; it += NTHR) {
        const int q = it / (8 * 160), ni = (it / 160) & 7, r = it % 160;
        if (ni >= nna) continue;
        const int tc = s_ctc[q];
        if (tc < 8) continue;
        const int range = tc - 7, stride = (range + 159) / 160;
        const int xc = 4 + r * stride;
        if (xc >= tc - 3) continue;
        CpSingle cs;
        if (!cp_single(H, n_sms, wexcl_of(s_cj[q]), wmx_of(s_cj[q]), s_cT[q], tc, xc, s_cna[ni], &cs, cp.ck_n, cp.ck_rows_cap)) continue;
        const unsigned long long key = ((unsigned long long)q << 40) | ((unsigned long long)ni << 32) | (unsigned long long)xc;
        if (cs.cost < mycost || (cs.cost == mycost && key < mykey)) { mycost = cs.cost; mykey = key; }
      }
      block_min(mycost, mykey, &s_r1c, &s_r1k);
    }
    if (dbg && tid == 0) dbg[23] = clock64() - t0;
    // ---- stage 2: two-piece tails.  Pieces move the optimal cut below the single-piece one (about x0.7 ... x1) and can prefer a smaller head budget: for the stage-1 winner's
    //      sequence, its budget and the two neighbouring budgets, 16 cut points in [0.6 xc1, xc1] x NPC piece splits are evaluated in parallel (exact LPT makespan of the 2H tail
    //      tasks); a candidate uses pieces when that beats its own single-piece cost by MULTI_MARGIN (host rule).
    __shared__ double s_mk[3][16][NPC]; __shared__ int s_wx[16];
    __shared__ double s_r2c; __shared__ unsigned long long s_r2k;
    if (tid == 0) s_cp = 0;
    __syncthreads();
    if (s_r1k != ~0ull) {
      const int q = (int)(s_r1k >> 40), ni0 = (int)((s_r1k >> 32) & 255), tc = s_ctc[q], xc1 = (int)(s_r1k & 0xffffffffu);
      const int xlo = max(4, (xc1 * 6) / 10), xhi = xc1 + 1, len = xhi - xlo, R = min(max(len, 1), 16);
      if (tid < 16) s_wx[tid] = (tid < R && len > 0) ? xlo + (int)(((long long)tid * len) / R) : -1;
      __syncthreads();
      for (int it = tid; it < 3 * 16 * NPC; it += NTHR) {
        const int nj = it / (16 * NPC), w = (it / NPC) & 15, pi = it % NPC;
        const int ni = ni0 + nj - 1;
        double mk = -1.0;
        if (ni >= 0 && ni < nna && s_wx[w] >= 0) { int s1, s2; mk = piece_mk(pi, tc - s_wx[w], H, n_sms - s_cna[ni], &s1, &s2); }
        s_mk[nj][w][pi] = mk;
      }
      __syncthreads();
      unsigned long long mykey = ~0ull; double mycost = 1e300;
      if (tid < 3 * 16) {
        const int nj = tid >> 4, w = tid & 15, ni = ni0 + nj - 1;
        if (ni >= 0 && ni < nna && s_wx[w] >= 0) {
          const int xc = s_wx[w];
          CpSingle cs;
          if (cp_single(H, n_sms, wexcl_of(s_cj[q]), wmx_of(s_cj[q]), s_cT[q], tc, xc, s_cna[ni], &cs, cp.ck_n, cp.ck_rows_cap)) {
            double cost = cs.cost; int bp = -1; double bmk = 1e300;
            for (int pi = 0; pi < NPC; pi++) { const double mk = s_mk[nj][w][pi]; if (mk >= 0.0 && mk < bmk) { bmk = mk; bp = pi; } }
            if (bp >= 0) {
              const double cm = fmax(cs.a, CONTENTION * (bmk + cs.snap)) + cs.fix + FIX_CHAIN_US * H / STEP_US;
              if (cm < cost * (1.0 - MULTI_MARGIN)) cost = cm; else bp = -1;
            }
            mycost = cost; mykey = ((unsigned long long)q << 56) | ((unsigned long long)ni << 48) | ((unsigned long long)xc << 16) | (unsigned long long)(bp + 1);
          }
        }
      }
      block_min(mycost, mykey, &s_r2c, &s_r2k);
    }
    if (dbg && tid == 0) dbg[14] = clock64() - t0;
    if (dbg && tid == 0) { dbg[8] = (long long)(s_r1c * 1000); dbg[9] = (long long)(s_r1k & 0xffffffffu); dbg[10] = (long long)(s_r2c * 1000); dbg[11] = (long long)((s_r2k >> 16) & 0xffffffffu); dbg[12] = (long long)(s_bcost * 1000); }
    if (tid == 0) {
      // final: the stage-2 window contains the stage-1 winner, so its best effective cost is <= the stage-1 cost
      s_cp = 0;
      if (s_r1k != ~0ull) {
        double bc = s_r1c; unsigned long long bk = ((s_r1k >> 40) << 56) | (((s_r1k >> 32) & 255) << 48) | ((s_r1k & 0xffffffffu) << 16);
        if (s_r2k != ~0ull && s_r2c < bc) { bc = s_r2c; bk = s_r2k; }
        if (bc <= s_bcost) {
          s_cp = 1; s_bj = (int)(bk >> 56); s_bna = s_cna[(int)((bk >> 48) & 255)]; s_bxc = (int)((bk >> 16) & 0xffffffffu);
          const int pi = (int)(bk & 0xffff) - 1;
          s_bs1 = 0; s_bs2 = 0;
          if (pi >= 0) { int s1, s2; piece_mk(pi, s_ctc[s_bj] - s_bxc, H, n_sms - s_bna, &s1, &s2); s_bs1 = s1; s_bs2 = s2; }

        }
      }
    }
    __syncthreads();
  }
  if (cp_on && (plain_mode & 4) && !s_cp) {     // the search rejected every cut: the non-CP head plan (lean) + empty tail tables
    for (int b = tid; b <= n_sms; b += NTHR) { cp.dptr[b] = 0; cp.seg_ptr[b] = 0; }
    if (tid == 0) { cp.geom[1] = 0; cp.geom[7] = 0; }
    if (N * H <= ncta0 && ident_on) ident_body(cu, N, H, ncta0, n_sms, task_list, cta_ptr, cu_out != nullptr && s_badcu);
    else lean_cio_body(cu, N, H, ncta0, cp.slot_off, n_sms, task_list, cta_ptr, dbg, dsm + LEAN_OFF, cu_out != nullptr && s_badcu);
    return;
  }
  int ncta_p = ncta0, sp_g = -1;
  if (cp_on && s_cp) {
    sort_stage(s_cj[s_bj], s_bxc);                    // the head group: the cut sequence counts xc chunks
    ncta_p = s_bna; sp_g = s_gsp;
  }
  plan_stage(ncta_p, sp_g);
  if (dbg && tid == 0) { dbg[2] = clock64() - t0; dbg[4] = s_bestM; dbg[5] = s_snake; dbg[6] = s_bestv; }
  expand_all(ncta_p, sp_g, cp_on ? cp.slot_off : 0, cp_on ? cp.n_sms : ncta0, cp_on && s_cp != 0);
  if (dbg && tid == 0) dbg[3] = clock64() - t0;
}



// ================================================================ [dynpath2] single-sequence planner (N == 1, plain head launch)
// The general plan_kernel spends ~50 us (cold, judge protocol) on a one-sequence CP replay: its search stages, wrap / snake planning and table expansion are written for N sequences.  For N == 1 the decision
// is the same arithmetic on one candidate sequence (q = 0), the head is the plain kernel's snake deal of H whole tasks (closed form) and the tail tables are the same expansion; this kernel computes exactly
// the tables of plan_kernel with plan_mode bit 0 (tools/n1cmp.py compares them bitwise) with a minimal number of barriers.
__device__ __forceinline__ int snake_start(int k, int ncta, int T) {
  const int full = T / ncta, rem = T - full * ncta;
  return k * full + ((full & 1) ? max(0, k - (ncta - rem)) : min(k, rem));
}

// candidate head-group budgets (host list: sorted, even-adjusted, deduplicated); returns their number
__device__ __forceinline__ int cp_budgets(int H, int n_sms, int* out) {
  int cand[7] = {H, n_sms / 2, (2 * n_sms) / 3, H + (n_sms - H) / 4, H + (n_sms - H) / 3, n_sms - H, n_sms - (H + 1) / 2};
  for (int a = 0; a < 7; a++) for (int b = a + 1; b < 7; b++) if (cand[b] < cand[a]) { const int t = cand[a]; cand[a] = cand[b]; cand[b] = t; }
  int nn = 0;
  for (int a = 0; a < 7; a++) {
    int n_A = cand[a];
    if (n_A > 2) n_A -= n_A % 2;
    if (n_A < 1 || (n_sms - n_A) / 2 < 1) continue;
    if (nn == 0 || out[nn - 1] != n_A) out[nn++] = n_A;
  }
  return nn;
}

__device__ __forceinline__ void n1_block_min(double mycost, unsigned long long mykey, double* sc, unsigned long long* sk, double* outc, unsigned long long* outk) {
  for (int off = 16; off > 0; off >>= 1) {
    const double oc = __shfl_xor_sync(0xffffffffu, mycost, off); const unsigned long long ok = __shfl_xor_sync(0xffffffffu, mykey, off);
    if (oc < mycost || (oc == mycost && ok < mykey)) { mycost = oc; mykey = ok; }
  }
  if ((threadIdx.x & 31) == 0) { sc[threadIdx.x >> 5] = mycost; sk[threadIdx.x >> 5] = mykey; }
  __syncthreads();
  double bc = 1e300; unsigned long long bk = ~0ull;
  for (int w = 0; w < (int)(blockDim.x >> 5); w++) if (sc[w] < bc || (sc[w] == bc && sk[w] < bk)) { bc = sc[w]; bk = sk[w]; }
  *outc = bc; *outk = bk;
  __syncthreads();
}

constexpr int N1_THR = 1024;
__global__ void __launch_bounds__(N1_THR) plan_n1_kernel(const long long* __restrict__ cu, int H, int ncta0, int* __restrict__ task_list, int* __restrict__ cta_ptr, CpOut cp,
                                                         long long* dbg, long long* cu_out, long long Ttok, int* pstat, kdaval::VArgs vv) {
  extern __shared__ __align__(16) unsigned char dsm_n[];
  if (blockIdx.x >= 2) { kdaval::pad_block(vv, blockIdx.x - 2); return; }
  if (blockIdx.x == 1) { if (!kdaval::validate_block_fast_ni(vv, (int*)dsm_n, false)) kdaval::zero_out_if_bad(vv); return; }   // table validation block (+ zero-fill of the output for a rejected cu_seqlens)
  __shared__ double s_wc[N1_THR / 32]; __shared__ unsigned long long s_wk[N1_THR / 32];
  __shared__ double s_mk[3][16][NPC];
  __shared__ int s_cnt2[NB], s_cnt3[NB], s_fst[MAXC], s_fen[MAXC]; __shared__ long long s_raw[MAXC];
  __shared__ int s_nlo, s_nhi, s_bad;
  const int tid = threadIdx.x, n_sms = cp.n_sms;
  const long long t0 = clock64();
  // ---- sequence: cu = [r0, r1]; a rejected table (outside [0, T], decreasing, cu[0] != 0) makes the launch a no-op (all sequences empty, as plan_kernel)
  const long long r0 = cu[0], r1 = cu[1];
  const int badc = (cu_out != nullptr && (r0 < 0 || r0 > Ttok || r1 < r0 || r1 > Ttok || r0 != 0)) ? 1 : 0;
  if (tid == 0 && cu_out != nullptr) { cu_out[0] = badc ? 0 : r0; cu_out[1] = badc ? 0 : r1; }
  if (tid == 0 && pstat != nullptr) *pstat = cu_out != nullptr ? badc : 0;
  const long long R0 = badc ? 0 : r0, R1 = badc ? 0 : r1;
  const long long len = R1 - R0;
  const int c = len > 0 ? (int)((len + 31) / 32) : 0;
  const double stc = c > 0 ? c + (c < 17 ? 5.0 : 1.0) : 0.0, sc1 = c + 1.0;
  const bool exit0 = (c + 1.0 <= (1.0 + MIN_GAIN) * stc * H / n_sms);
  const double bcost = fmax(c + 1.0, sc1 * H / n_sms) * (1.0 - MIN_GAIN) - cp.ovh;
  int cna[8];
  const int tc = (len >= 32 * 8) ? c : 0;
  const int nna = (exit0 || !(tc >= 8 && cp_lb_ok(H, n_sms, 0.0, tc, bcost))) ? 0 : cp_budgets(H, n_sms, cna);
  const long long Wexcl = 0; const double wmx = 0.0;   // N == 1: no other sequence
  bool cp_on = false; int bxc = 0, bna = 0, bs1 = 0, bs2 = 0;
  if (dbg && tid == 0) dbg[40] = clock64() - t0;
  if (nna > 0 && tc >= 8) {
    // ---- stage 1: single-piece cost on the flat grid (cut points xc in [4, tc - 3), stride ceil(range / 160)) x budgets
    const int range = tc - 7, stride = (range + 159) / 160;
    unsigned long long mykey = ~0ull; double mycost = 1e300;
    for (int it = tid; it < 8 * 160; it += N1_THR) {
      const int ni = it / 160, r = it % 160;
      if (ni >= nna) continue;
      const int xc = 4 + r * stride;
      if (xc >= tc - 3) continue;
      CpSingle cs;
      if (!cp_single(H, n_sms, Wexcl, wmx, len, tc, xc, cna[ni], &cs, cp.ck_n, cp.ck_rows_cap)) continue;
      const unsigned long long key = ((unsigned long long)ni << 32) | (unsigned long long)xc;
      if (cs.cost < mycost || (cs.cost == mycost && key < mykey)) { mycost = cs.cost; mykey = key; }
    }
    double r1c; unsigned long long r1k;
    n1_block_min(mycost, mykey, s_wc, s_wk, &r1c, &r1k);
    if (dbg && tid == 0) dbg[41] = clock64() - t0;
    if (r1k != ~0ull) {
      // ---- stage 2: two-piece tails around the stage-1 winner (budget and its neighbours, 16 cut points in [0.6 xc1, xc1], NPC piece splits)
      const int ni0 = (int)((r1k >> 32) & 255), xc1 = (int)(r1k & 0xffffffffu);
      const int xlo = max(4, (xc1 * 6) / 10), xhi = xc1 + 1, len2 = xhi - xlo, R = min(max(len2, 1), 16);
      for (int it = tid; it < 3 * 16 * NPC; it += N1_THR) {
        const int nj = it / (16 * NPC), w = (it / NPC) & 15, pi = it % NPC;
        const int ni = ni0 + nj - 1;
        const int wx = (w < R && len2 > 0) ? xlo + (int)(((long long)w * len2) / R) : -1;
        double mk = -1.0;
        if (ni >= 0 && ni < nna && wx >= 0) { int s1, s2; mk = piece_mk(pi, tc - wx, H, n_sms - cna[ni], &s1, &s2); }
        s_mk[nj][w][pi] = mk;
      }
      __syncthreads();
      unsigned long long mykey2 = ~0ull; double mycost2 = 1e300;
      if (tid < 3 * 16) {
        const int nj = tid >> 4, w = tid & 15, ni = ni0 + nj - 1;
        const int xc = (w < R && len2 > 0) ? xlo + (int)(((long long)w * len2) / R) : -1;
        if (ni >= 0 && ni < nna && xc >= 0) {
          CpSingle cs;
          if (cp_single(H, n_sms, Wexcl, wmx, len, tc, xc, cna[ni], &cs, cp.ck_n, cp.ck_rows_cap)) {
            double cost = cs.cost; int bp = -1; double bmk = 1e300;
            for (int pi = 0; pi < NPC; pi++) { const double mk = s_mk[nj][w][pi]; if (mk >= 0.0 && mk < bmk) { bmk = mk; bp = pi; } }
            if (bp >= 0) {
              const double cm = fmax(cs.a, CONTENTION * (bmk + cs.snap)) + cs.fix + FIX_CHAIN_US * H / STEP_US;
              if (cm < cost * (1.0 - MULTI_MARGIN)) cost = cm; else bp = -1;
            }
            mycost2 = cost; mykey2 = ((unsigned long long)ni << 48) | ((unsigned long long)xc << 16) | (unsigned long long)(bp + 1);
          }
        }
      }
      double r2c; unsigned long long r2k;
      n1_block_min(mycost2, mykey2, s_wc, s_wk, &r2c, &r2k);
      double bc = r1c; unsigned long long bk = (((r1k >> 32) & 255) << 48) | ((r1k & 0xffffffffu) << 16);
      if (r2k != ~0ull && r2c < bc) { bc = r2c; bk = r2k; }
      if (bc <= bcost) {
        cp_on = true; bna = cna[(int)((bk >> 48) & 255)]; bxc = (int)((bk >> 16) & 0xffffffffu);
        const int pi = (int)(bk & 0xffff) - 1;
        if (pi >= 0) { int s1, s2; piece_mk(pi, tc - bxc, H, n_sms - bna, &s1, &s2); bs1 = s1; bs2 = s2; }
      }
    }
  }
  (void)bs2;
  if (dbg && tid == 0) dbg[42] = clock64() - t0;
  // ---- head table: the plain kernel's snake deal of the H whole tasks (task id = head) over ncta CTAs; cta_ptr padded to n_sms + 1
  const int ncta = cp_on ? bna : ncta0;
  for (int r = tid; r < H; r += N1_THR) {
    const int qq = r / ncta, pos = r - qq * ncta, k = (qq & 1) ? ncta - 1 - pos : pos;
    task_list[snake_start(k, ncta, H) + qq] = r;
  }
  for (int k = tid; k <= n_sms; k += N1_THR) cta_ptr[k] = k <= ncta ? snake_start(k, ncta, H) : H;
  if (tid == 0 && cp.cu_a != nullptr) { cp.cu_a[0] = R0; cp.cu_a[1] = cp_on ? R0 + 32LL * bxc : R1; }
  if (dbg && tid == 0) dbg[43] = clock64() - t0;
  if (!cp_on) {
    for (int b = tid; b <= n_sms; b += N1_THR) { cp.dptr[b] = 0; cp.seg_ptr[b] = 0; }
    if (tid == 0) { cp.geom[1] = 0; cp.geom[7] = 0; }
    if (dbg && tid == 0) dbg[3] = clock64() - t0;
    return;
  }
  // ---- context-parallel tail tables (plan_kernel expand_all, cp_tail, N == 1)
  const int xc = bxc, n_A = bna, n_D = n_sms - n_A;
  const long long x = R0 + 32LL * xc, e_end = R1, tail = e_end - x;
  const int m = bs1 ? 2 : 1;
  const int tph = (int)((tail + 127) / 128);
  const int pt1 = m == 2 ? bs1 / 4 : (tph > 0 ? tph : 1);
  const int Gf = n_sms, fullh = H / Gf, remh = H % Gf;
  const int qb = n_D > 0 ? H / n_D : 0, rb = n_D > 0 ? H % n_D : 0, clo = n_D - rb, chi = rb;
  const double sw = m == 1 ? FIX_STATE_TILES : FIX_OWNER_TILES;
  const double Kf = sw + (m - 1) * FIX_CHAIN_TILES + (tph - pt1) + ck_items(cp.ck_n, c, xc, m, bs1) * FIX_STATE_TILES;   // + the checkpoint items of every head (host fix_schedule extra)
  const int n0 = pt1 * H;
  const double Utot = (double)n0 + ((double)H) * Kf;
  if (tid == 0) {   // piece-1 tasks per load level of the dual launch (rounds of the two-level merge)
    int nlo = 0, nhi = 0;
    if (m == 2) {
      const double t1 = DUAL_F * bs1 + DUAL_TASK, t2 = DUAL_F * bs2 + DUAL_TASK;
      int remaining = H, kl = 1, kh = 1;
      while (remaining > 0) {
        const double vlo = clo > 0 ? qb * t1 + kl * t2 : 1e300, vhi = chi > 0 ? (qb + 1) * t1 + kh * t2 : 1e300;
        if (vhi <= vlo) { const int tk = min(remaining, chi); nhi += tk; remaining -= tk; kh++; } else { const int tk = min(remaining, clo); nlo += tk; remaining -= tk; kl++; }
      }
    }
    s_nlo = nlo; s_nhi = nhi;
    cp.cu_tail[0] = x; cp.cu_tail[1] = m == 2 ? x + 32LL * bs1 : e_end; cp.cu_tail[2] = e_end;
    cp.geom[0] = (int)x; cp.geom[1] = m; cp.geom[2] = (int)e_end; cp.geom[3] = 0; cp.geom[4] = n_A; cp.geom[5] = n_D; cp.geom[6] = xc; cp.geom[7] = 1;
    if (cp.ck_n > 0) {   // checkpoint boundaries b = kk * ck_n of the sequence beyond the cut: b == xc is S_A, b == the piece-1 start is S_in(0) (plain copies), the others are fixup items
      const int ckn = cp.ck_n, k0 = (xc + ckn - 1) / ckn, kmax = (c - 1) / ckn;
      const int pb1 = m == 2 ? xc + bs1 : -1;
      const bool saSp = xc % ckn == 0, psSp = m == 2 && pb1 % ckn == 0 && pb1 / ckn <= kmax;
      const int total = max(0, kmax - k0 + 1);
      cp.geom[8] = total - (saSp ? 1 : 0) - (psSp ? 1 : 0); cp.geom[9] = k0; cp.geom[10] = (saSp ? 1 : 0) + (psSp ? 1 : 0);
      cp.geom[12] = xc; cp.geom[13] = k0; cp.geom[14] = (int)x;
      int nc = 0;
      if (saSp) { cp.ccopy[2 * nc] = xc / ckn; cp.ccopy[2 * nc + 1] = -1; nc++; }
      if (psSp) { cp.ccopy[2 * nc] = pb1 / ckn; cp.ccopy[2 * nc + 1] = 0; nc++; }
    }
  }
  if (dbg && tid == 0) dbg[44] = clock64() - t0;
  for (int b = tid; b < Gf; b += N1_THR) {
    const int Cobb = (b + 1) * fullh + min(b + 1, remh);     // heads owned by CTAs 0..b
    s_raw[b] = (long long)nearbyint((double)(b + 1) * Utot / Gf - (double)Cobb * Kf);
  }
  __syncthreads();
  if (dbg && tid == 0) dbg[45] = clock64() - t0;
  {
    const int nlo = s_nlo, nhi = s_nhi;
    for (int b = tid; b < Gf; b += N1_THR) {
      int cnt = 0;
      if (b < n_D) {
        cnt = (b < rb) ? qb + 1 : qb;                                   // piece-0 tasks b, b + n_D, ...
        if (m == 2) cnt += (b < rb) ? (nhi > b ? (nhi - 1 - b) / chi + 1 : 0) : (nlo > b - rb ? (nlo - 1 - (b - rb)) / clo + 1 : 0);
      }
      s_cnt2[b] = cnt;
    }
  }
  __syncthreads();
  if (dbg && tid == 0) dbg[46] = clock64() - t0;
  if (tid < 32) {          // dual per-CTA task pointers
    const int lane = tid;
    const int per = (Gf + 31) / 32, b0 = lane * per, b1 = min(Gf, b0 + per);
    int sm = 0;
    for (int i = b0; i < b1; i++) sm += s_cnt2[i];
    int inc = sm;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += o; }
    int run = inc - sm;
    for (int i = b0; i < b1; i++) { const int v = s_cnt2[i]; s_cnt2[i] = run; run += v; }
    const int tot = __shfl_sync(0xffffffffu, inc, 31);
    if (lane == 0) s_cnt2[Gf] = tot;
    __syncwarp();
    for (int b = lane; b <= Gf; b += 32) cp.dptr[b] = s_cnt2[b];
  } else if (tid < 64) {   // fixup segments: running max of raw -> per-CTA tile range and segment count
    const int lane = tid - 32;
    const int per = (Gf + 31) / 32, b0 = lane * per, b1 = min(Gf, b0 + per);
    long long mx = 0;
    for (int i = b0; i < b1; i++) mx = max(mx, s_raw[i]);
    long long inc = mx;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const long long o = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc = max(inc, o); }
    long long run = __shfl_up_sync(0xffffffffu, inc, 1); if (lane == 0) run = 0;      // max over the CTAs of earlier lanes
    const FastDiv fdp(pt1);
    for (int b = b0; b < b1; b++) {
      const long long start_b = min((long long)n0, run);
      run = max(run, s_raw[b]);
      long long end_b = b == Gf - 1 ? (long long)n0 : min((long long)n0, run);
      if (end_b < start_b) end_b = start_b;
      const int owned = fullh + (b < remh ? 1 : 0);
      const int own_segs = owned * (m - 1);
      const int p0_segs = end_b > start_b ? fdp.div((int)end_b - 1) - fdp.div((int)start_b) + 1 : 0;
      s_cnt3[b] = own_segs + p0_segs;
      s_fst[b] = (int)start_b; s_fen[b] = (int)end_b;
    }
    __syncwarp();
    int sm = 0;
    for (int i = b0; i < b1; i++) sm += s_cnt3[i];
    int inc2 = sm;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) { const int o = __shfl_up_sync(0xffffffffu, inc2, off); if (lane >= off) inc2 += o; }
    int run2 = inc2 - sm;
    for (int i = b0; i < b1; i++) { const int v = s_cnt3[i]; s_cnt3[i] = run2; run2 += v; }
    const int tot = __shfl_sync(0xffffffffu, inc2, 31);
    if (lane == 0) s_cnt3[Gf] = tot;
    __syncwarp();
    for (int b = lane; b <= Gf; b += 32) cp.seg_ptr[b] = s_cnt3[b];
  }
  __syncthreads();
  if (dbg && tid == 0) dbg[47] = clock64() - t0;
  {
    const int nlo = s_nlo, nhi = s_nhi;
    for (int b = tid; b < n_D; b += N1_THR) {   // dual lists: piece-0 tasks, then the piece-1 tasks of the CTA's level (identical tasks: level slots in index order)
      int* o = cp.dtask + s_cnt2[b];
      for (int t = b; t < H; t += n_D) *o++ = t;
      if (m == 2) {
        if (b < rb) { for (int j = b; j < nhi; j += chi) *o++ = H + j; }
        else { for (int j = b - rb; j < nlo; j += clo) *o++ = H + nhi + j; }
      }
    }
    const FastDiv fdp(pt1);
    for (int b = tid; b < Gf; b += N1_THR) {    // fixup segments of CTA b: its heads' later pieces, then its range of piece-0 tiles cut at head boundaries
      int* o = cp.seg + 4 * s_cnt3[b];
      for (int h = b; h < H; h += Gf)
        for (int k = 1; k < m; k++) { o[0] = h; o[1] = k; o[2] = pt1; o[3] = tph - pt1; o += 4; }      // m == 2: piece 1 = tiles [pt1, tph)
      int u = s_fst[b];
      const int end_b = s_fen[b];
      while (u < end_b) {
        const int h = fdp.div(u);
        const int ee = min(end_b, (h + 1) * pt1);
        o[0] = h; o[1] = 0; o[2] = u - h * pt1; o[3] = ee - u; o += 4;
        u = ee;
      }
    }
    if (cp.ck_n > 0) {   // checkpoint items of the fixup
      const int ckn = cp.ck_n, k0 = (xc + ckn - 1) / ckn, kmax = (c - 1) / ckn, pb1 = m == 2 ? xc + bs1 : -1;
      const bool saSp = xc % ckn == 0, psSp = m == 2 && pb1 % ckn == 0 && pb1 / ckn <= kmax;
      const int kk1 = psSp ? pb1 / ckn : -1;
      for (int kk = k0 + tid; kk <= kmax; kk += N1_THR) {
        const int b = kk * ckn;
        if ((saSp && b == xc) || (psSp && kk == kk1)) continue;
        const int p = (m == 2 && b >= pb1) ? 1 : 0;
        const int idx = (kk - k0) - ((saSp && kk > k0) ? 1 : 0) - ((psSp && kk > kk1) ? 1 : 0);
        int* o = cp.cki + 4 * idx;
        o[0] = kk - k0; o[1] = p; o[2] = kk; o[3] = 0;
      }
    }
  }
  if (dbg && tid == 0) dbg[3] = clock64() - t0;
}

// [multi-GPU] opt-in shared-memory sizes are per-device (plan_init() runs once per device under that device): indexed by the current device
struct DevCfg { int max_dyn = 0, lp = 0, lc = 0, n1 = 0; };
static DevCfg g_cfg[64];
static DevCfg& dcfg() { int d = 0; cudaGetDevice(&d); return g_cfg[d & 63]; }
#define g_max_dyn (dcfg().max_dyn)
#define g_dyn_lp (dcfg().lp)
#define g_dyn_lc (dcfg().lc)
#define g_dyn_n1 (dcfg().n1)
size_t smem_bytes(int N) {
  return ((size_t)(4 * N) * 4 + 7) / 8 * 8 + (size_t)(2 * N + 1) * 8 + (size_t)NVAR * NB * (4 + 4 + 4 + 4 + 8 * 4) + 16;
}
}  // namespace

// task_list: int32 [4 * (N*H + 2*n_pad)] (plain: [N*H + 2*n_pad]); cta_ptr: int32 [n_pad + 1]; CP outputs (all optional, together):
//   dtask int32 [2H + 8 + 2H] (tasks then scratch), dptr [n_sms + 1], cu_tail int64 [3], seg int32 [4 * (2H + n_sms + 16)], seg_ptr int32 [n_sms + 1 + 2 n_sms], geom int32 [8]
void plan_wrap(torch::Tensor cu, int64_t H, int64_t ncta, torch::Tensor task_list, torch::Tensor cta_ptr, int64_t plain, c10::optional<torch::Tensor> dbg,
               c10::optional<torch::Tensor> dtask, c10::optional<torch::Tensor> dptr, c10::optional<torch::Tensor> cu_tail, c10::optional<torch::Tensor> seg,
               c10::optional<torch::Tensor> seg_ptr, c10::optional<torch::Tensor> geom, int64_t n_sms, int64_t slot_off, c10::optional<torch::Tensor> cu_a,
               c10::optional<pybind11::dict> val, int64_t ck_n, int64_t ck_rows_cap, c10::optional<torch::Tensor> cki, c10::optional<torch::Tensor> ccopy, double cp_ovh) {
  KdaDevGuard dev_guard_(cu);   // [multi-GPU] launch + per-device config on the tensors' device
  const int N = (int)(cu.numel() - 1);
  TORCH_CHECK(ncta >= 1 && ncta <= MAXC, "ncta out of range");
  TORCH_CHECK(task_list.numel() >= ((plain & 1) ? 1 : 4) * (N * H + 2 * (n_sms > 0 ? n_sms : ncta)), "task_list too small");
  const size_t sm = smem_bytes(N);
  TORCH_CHECK(g_max_dyn > 0, "plan_init() not called");
  int nmax = 1; while (smem_bytes(nmax + 1) <= (size_t)g_max_dyn) nmax++;
  TORCH_CHECK((int)sm <= g_max_dyn, "too many sequences for the device planner (", N, "; limit ", nmax, ")");
  CpOut cp{};
  if (geom) {
    TORCH_CHECK(dtask && dptr && cu_tail && seg && seg_ptr && n_sms >= 8 && n_sms <= MAXC, "cp outputs incomplete");
    cp.n_sms = (int)n_sms; cp.slot_off = (int)slot_off; cp.dtask = dtask->data_ptr<int>(); cp.dptr = dptr->data_ptr<int>();
    cp.cu_tail = (long long*)cu_tail->data_ptr<int64_t>(); cp.seg = seg->data_ptr<int>(); cp.seg_ptr = seg_ptr->data_ptr<int>(); cp.geom = geom->data_ptr<int>();
    cp.cu_a = cu_a ? (long long*)cu_a->data_ptr<int64_t>() : nullptr;
    cp.ck_n = (int)ck_n; cp.ck_rows_cap = (int)ck_rows_cap; cp.cki = nullptr; cp.ccopy = nullptr; cp.ovh = (float)cp_ovh;
    if (ck_n > 0) {
      TORCH_CHECK(cki && ccopy && geom->numel() >= 16 && cki->numel() >= 4 * (ck_rows_cap + 2), "checkpoint outputs incomplete");
      cp.cki = cki->data_ptr<int>(); cp.ccopy = ccopy->data_ptr<int>();
    }
  }
  kdaval::VArgs va{};
  long long* cu_out = nullptr; long long Ttok = 0; int* pstat = nullptr;
  if (val) {   // [contract2] sanitized cu_seqlens written by the planner (val['cu_out']) + the validation block (grid 2)
    va = kdaval::args_from_dict(*val);
    cu_out = va.cu_out; Ttok = va.T; va.cu_out = nullptr;
    pstat = val->contains("pstat") && !(*val)["pstat"].is_none() ? (*val)["pstat"].cast<torch::Tensor>().data_ptr<int>() : nullptr;
    TORCH_CHECK(8 * kdaval::hash_slots(N) <= g_max_dyn, "too many sequences for the table validation");
  }
  plan_kernel<<<val ? 2 + va.pad_blocks : 1, NTHR, (size_t)g_max_dyn, c10::cuda::getCurrentCUDAStream().stream()>>>(   // the full opt-in size (not just sm): same shared-memory carveout as the neighbouring persistent kernels, no SM reconfiguration
      (const long long*)cu.data_ptr<int64_t>(), N, (int)H, (int)ncta, task_list.data_ptr<int>(), cta_ptr.data_ptr<int>(), (int)plain, cp,
      dbg ? (long long*)dbg->data_ptr<int64_t>() : nullptr, cu_out, Ttok, pstat, va);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// [dynpath2] single-sequence planner (plain head launch): the tables of plan_wrap(plain_mode = 1) for N == 1
void plan_n1(torch::Tensor cu, int64_t H, int64_t ncta, torch::Tensor task_list, torch::Tensor cta_ptr, c10::optional<torch::Tensor> dbg, torch::Tensor dtask, torch::Tensor dptr,
             torch::Tensor cu_tail, torch::Tensor seg, torch::Tensor seg_ptr, torch::Tensor geom, int64_t n_sms, torch::Tensor cu_a, c10::optional<pybind11::dict> val,
             int64_t ck_n, int64_t ck_rows_cap, c10::optional<torch::Tensor> cki, c10::optional<torch::Tensor> ccopy, double cp_ovh) {
  KdaDevGuard dev_guard_(cu);   // [multi-GPU] launch + per-device config on the tensors' device
  TORCH_CHECK(cu.numel() == 2, "plan_n1: one sequence");
  TORCH_CHECK(ncta >= 1 && ncta <= MAXC && n_sms >= 8 && n_sms <= MAXC && g_dyn_n1 > 0, "plan_n1: bad arguments / plan_init() not called");
  TORCH_CHECK(task_list.numel() >= H + 2 * n_sms, "task_list too small");
  CpOut cp{};
  cp.n_sms = (int)n_sms; cp.slot_off = (int)H; cp.dtask = dtask.data_ptr<int>(); cp.dptr = dptr.data_ptr<int>(); cp.cu_tail = (long long*)cu_tail.data_ptr<int64_t>();
  cp.seg = seg.data_ptr<int>(); cp.seg_ptr = seg_ptr.data_ptr<int>(); cp.geom = geom.data_ptr<int>(); cp.cu_a = (long long*)cu_a.data_ptr<int64_t>();
  cp.ck_n = (int)ck_n; cp.ck_rows_cap = (int)ck_rows_cap; cp.cki = nullptr; cp.ccopy = nullptr; cp.ovh = (float)cp_ovh;
  if (ck_n > 0) {
    TORCH_CHECK(cki && ccopy && geom.numel() >= 16 && cki->numel() >= 4 * (ck_rows_cap + 2), "checkpoint outputs incomplete");
    cp.cki = cki->data_ptr<int>(); cp.ccopy = ccopy->data_ptr<int>();
  }
  kdaval::VArgs va{};
  long long* cu_out = nullptr; long long Ttok = 0; int* pstat = nullptr;
  if (val) {
    va = kdaval::args_from_dict(*val);
    cu_out = va.cu_out; Ttok = va.T; va.cu_out = nullptr;
    pstat = val->contains("pstat") && !(*val)["pstat"].is_none() ? (*val)["pstat"].cast<torch::Tensor>().data_ptr<int>() : nullptr;
    TORCH_CHECK(8 * kdaval::hash_slots(1) <= g_dyn_n1, "validation smem");
  }
  plan_n1_kernel<<<val ? 2 + va.pad_blocks : 1, N1_THR, (size_t)g_dyn_n1, c10::cuda::getCurrentCUDAStream().stream()>>>(
      (const long long*)cu.data_ptr<int64_t>(), (int)H, (int)ncta, task_list.data_ptr<int>(), cta_ptr.data_ptr<int>(), cp, dbg ? (long long*)dbg->data_ptr<int64_t>() : nullptr, cu_out, Ttok, pstat, va);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// largest number of sequences the planner kernel supports on this device (shared memory); plan_init() must have run
int64_t plan_max_seq() {
  int nmax = 1; while (smem_bytes(nmax + 1) <= (size_t)g_max_dyn && 8 * kdaval::hash_slots(nmax + 1) <= g_max_dyn) nmax++;
  return nmax;
}

// [contract2] shared by the lean launches: sanitized cu_seqlens output, status word, validation block (grid 2)
static kdaval::VArgs lean_val(c10::optional<pybind11::dict>& val, int N, long long** cu_out, long long* Ttok, int** pstat) {
  kdaval::VArgs va{};
  *cu_out = nullptr; *Ttok = 0; *pstat = nullptr;
  if (val) {
    va = kdaval::args_from_dict(*val);
    *cu_out = va.cu_out; *Ttok = va.T; va.cu_out = nullptr;
    *pstat = val->contains("pstat") && !(*val)["pstat"].is_none() ? (*val)["pstat"].cast<torch::Tensor>().data_ptr<int>() : nullptr;
    TORCH_CHECK(8 * kdaval::hash_slots(N) <= g_max_dyn, "too many sequences for the table validation");
  }
  return va;
}

void plan_lean(torch::Tensor cu, int64_t H, int64_t ncta, torch::Tensor task_list, torch::Tensor cta_ptr, int64_t plain, c10::optional<torch::Tensor> dbg, c10::optional<pybind11::dict> val) {
  KdaDevGuard dev_guard_(cu);   // [multi-GPU] launch + per-device config on the tensors' device
  const int N = (int)(cu.numel() - 1);
  TORCH_CHECK(N >= 1 && N <= LN_MAX && ncta >= 1 && ncta <= MAXC, "lean planner: N or ncta out of range");
  TORCH_CHECK(g_max_dyn > 0, "plan_init() not called");
  TORCH_CHECK(task_list.numel() >= ((plain & 1) ? 1 : 4) * (N * H + 2 * ncta), "task_list too small");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  long long* cu_out; long long Ttok; int* pstat;
  kdaval::VArgs va = lean_val(val, N, &cu_out, &Ttok, &pstat);
  const int grid = val ? 2 + va.pad_blocks : 1;
  if (!(plain & 1)) {
    plan_lean_cio_kernel<<<grid, NTHR, (size_t)g_dyn_lc, stream>>>((const long long*)cu.data_ptr<int64_t>(), N, (int)H, (int)ncta, task_list.data_ptr<int>(), cta_ptr.data_ptr<int>(),
                                                               dbg ? (long long*)dbg->data_ptr<int64_t>() : nullptr, cu_out, Ttok, pstat, va);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // [resid] <= 64 sequences: a 256-thread planner / validation block (the loops run on blockDim.x): the 1024-thread launch + 32-warp barriers cost ~0.5-1 us of the planner kernel (graph timeline 6.1 -> 5.0 / 6.2 -> 5.7 us)
  const int pthr = N <= 64 ? 256 : NTHR;
  plan_lean_plain_kernel<<<grid, pthr, (size_t)g_dyn_lp, stream>>>((const long long*)cu.data_ptr<int64_t>(), N, (int)H, (int)ncta, task_list.data_ptr<int>(), cta_ptr.data_ptr<int>(),
                                                                 dbg ? (long long*)dbg->data_ptr<int64_t>() : nullptr, cu_out, Ttok, pstat, va, (int)((plain >> 8) & 0xff));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// one-time per-device setup (opt-in dynamic shared memory); call outside of stream capture, e.g. from prepare()
void plan_init() {
  cudaFree(0);   // make sure the context of the current device exists
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaFuncAttributes fa;
  TORCH_CHECK(cudaFuncGetAttributes(&fa, plan_kernel) == cudaSuccess, "cudaFuncGetAttributes failed");
  g_max_dyn = optin - (int)fa.sharedSizeBytes - 1024;
  TORCH_CHECK(cudaFuncSetAttribute(plan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_max_dyn) == cudaSuccess, "cudaFuncSetAttribute failed");
  {   // lean kernels: same total shared memory (static + dynamic = opt-in - 1 KB) as the general kernel, i.e. the same carveout as the neighbouring persistent kernels
    cudaFuncAttributes fl;
    cudaFuncGetAttributes(&fl, plan_lean_plain_kernel); g_dyn_lp = optin - 1024 - (int)fl.sharedSizeBytes;
    cudaFuncSetAttribute(plan_lean_plain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_dyn_lp);
    cudaFuncGetAttributes(&fl, plan_lean_cio_kernel); g_dyn_lc = optin - 1024 - (int)fl.sharedSizeBytes;
    cudaFuncSetAttribute(plan_lean_cio_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_dyn_lc);
    cudaFuncGetAttributes(&fl, plan_n1_kernel); g_dyn_n1 = optin - 1024 - (int)fl.sharedSizeBytes;
    cudaFuncSetAttribute(plan_n1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_dyn_n1);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("init", &plan_init);
  m.def("max_seq", &plan_max_seq);
  m.def("plan_n1", &plan_n1, py::arg("cu"), py::arg("H"), py::arg("ncta"), py::arg("task_list"), py::arg("cta_ptr"), py::arg("dbg"), py::arg("dtask"), py::arg("dptr"), py::arg("cu_tail"), py::arg("seg"), py::arg("seg_ptr"), py::arg("geom"), py::arg("n_sms"), py::arg("cu_a"), py::arg("val"), py::arg("ck_n"), py::arg("ck_rows_cap"), py::arg("cki"), py::arg("ccopy"), py::arg("cp_ovh") = 0.0);
  m.def("plan_lean", &plan_lean, py::arg("cu"), py::arg("H"), py::arg("ncta"), py::arg("task_list"), py::arg("cta_ptr"), py::arg("plain") = 1, py::arg("dbg") = py::none(), py::arg("val") = py::none());
  m.def("plan_wrap", &plan_wrap, py::arg("cu"), py::arg("H"), py::arg("ncta"), py::arg("task_list"), py::arg("cta_ptr"), py::arg("plain") = 0, py::arg("dbg") = py::none(),
        py::arg("dtask") = py::none(), py::arg("dptr") = py::none(), py::arg("cu_tail") = py::none(), py::arg("seg") = py::none(), py::arg("seg_ptr") = py::none(),
        py::arg("geom") = py::none(), py::arg("n_sms") = 0, py::arg("slot_off") = 0, py::arg("cu_a") = py::none(), py::arg("val") = py::none(),
        py::arg("ck_n") = 0, py::arg("ck_rows_cap") = 0, py::arg("cki") = py::none(), py::arg("ccopy") = py::none(), py::arg("cp_ovh") = 0.0);
}
