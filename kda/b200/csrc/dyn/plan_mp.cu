// [dynfew] Device-side MANY-PIECE context-parallel planner for few-head launches with several sequences (graph-safe: reads cu_seqlens on the GPU).
// The static host planner (w3/kernel.py mp_plan / _multi_plan) cuts the long sequences of a few-head shape into many dual-chain pieces whose states are chained by a fixup
// (portfolio iter 33).  This kernel makes the same kind of plan from cu_seqlens on the device, for ONE launch that serves every cut sequence:
//   * head group (cio launch): the first x chunks of every cut sequence (their final state S_A goes to scratch slot s * H + h, one CTA per task) and the whole tasks of the uncut
//     sequences (snake-dealt over the remaining head CTAs);
//   * dual-chain tail launch: every cut sequence's tail is cut into m_s pieces of ~equal length; the pieces of all cut sequences form ONE cu_seqlens-style piece list
//     (cu_tail: piece p of slot s has id gp0_s + p, a dummy "gap piece" id gp0_s + m_s spans the tokens between two cut tails and is never scheduled);
//   * fixup (cp_fix_mpd.cu): a state chain per (cut sequence, head) - flat (mode 1) or hierarchical (fixup_hd, groups of sz pieces) - then the output tiles of every piece (mode 2,
//     segment table built here).
// Closed-form cost model (us; constants of the static planner, w3/kernel.py _MP_*): candidates = (head length x, piece length cp) common to all cut sequences (a sequence with
// >= x + 8 chunks is cut), three zooming 16 x 16 grids, one candidate per thread; the best must beat the no-cut plan by `margin`.
// Latency design: the kernel is one block, so its time is the dependent chain of phases: cu load -> 3 search stages (~256 candidates each) -> per-sequence decision (warp 0, scans
// by shuffles) -> three independent table builders on separate warp groups (named barriers): dual lists | tile segments | head lists + slot geometry + piece list.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};
#include <cstdint>
#include "../w3/validate_dev.cuh"

namespace {
constexpr int NTHR = 1024;
constexpr int NMAX = 32;               // sequences
constexpr int MAXS = 256;              // SMs
constexpr int DUAL_CAP = 3 * MAXS;     // dual tasks (piece x head): at most PIECE_WAVES per SM
constexpr int NH_CAP = 1280;           // head tasks (sequence x head)
constexpr int PIECE_WAVES = 3;
constexpr int GS = 16;
constexpr int NVAR = 3;                // wrap-plan variants: bin spacing + {0, 2, 8} * 0.5% (work_dynskew tools/wrapsim2.py: the same maximum load as lean's six variants on 32 layouts)
__device__ __forceinline__ int woff(int v) { return v == 0 ? 0 : (v == 1 ? 2 : 8); }
constexpr int MC_ENTRY = 277, MC_FRONT = 48, MC_BACK = 8, MC_CUT = MC_ENTRY + MC_FRONT + MC_BACK, MC_PRELOAD = 17, MC_SHORT = 70;   // wrap-plan costs of lean_cio_body (plan_cp.cu), centi-chunks
__device__ __forceinline__ int wcost_m(int n) { return n * 100 + MC_ENTRY + (n < MC_PRELOAD ? MC_SHORT : 0); }                 // ints per slot row of the geometry table

struct MpPar {   // cost model (us); every field has the default of the static planner (kernel.py _MP_*), measured on B200
  float a_head, f_task, f_headl, a_dual, a_dual_busy, f_dual, ch_fixed, ch_item, tile_bw, tile_fixed, gaps, ovh, margin, whole_pack, hier_item, dual_item, hier_gap, ck_row_us;
  int dual_busy, hier_min, hier_on, force_x, force_cp;
  int wrap;                                            // 1: the whole tasks of the head group are cut at CTA boundaries (wrap plan, state hand-off through slots slot0 + CTA index)
  float wrap_pen, wrap_tau; int stages;
  float split;                                         // 1: the whole tasks of the uncut sequences run in their own launch (finish anywhere up to the end of the chain-critical path), decoupled from the cut heads
                                                  // wrap plan only when the snake deal outlasts the head by this factor (K > 0)     // force_x > 0: tools / calibration: use this (head length, piece length) instead of the search result
};

// [replay] the kernel's by-value parameter struct must NOT grow (+28 B of parameters made every launch of this planner ~+5% slower end to end, measured with plumbing-only changes): the checkpoint-mode
// chain constants ride in fields that are unused in checkpoint mode (tile_bw -> ch_item_ck, tile_fixed -> ch_fixed_ck) and are widened into this kernel-local struct.
struct MpParX : MpPar { float ck_flat, ch_item_ck, ch_fixed_ck; };

// Outputs (all device pointers).
struct MpOut {
  int* tlb;                            // [split] the whole tasks as a plain task list (snake order) of launch B2 (plain kernel, used when nothing is cut): int32 [N * H]
  int* desc; int* cta_ptr;             // head launch (cio): int4 entries (task = seq * H + head, c0, c1, slots), per-CTA pointers [ncta0 + 1]
  int* dtask; int* dptr;               // dual launch: task ids (piece id * H + head), per-CTA pointers [n_sms + 1]
  long long* cu_tail;                  // piece list (absolute token offsets), capacity P_cap + 1
  int* gs; int* geom;                  // slot geometry int32 [KC][16] {x, e, m, first piece id, sequence, tiles per head, group size (0: flat chain), first job, groups, first group id},
                                       // geom[0] = K (0: no cut in this replay), [1] = M pieces, [2] = n_A, [3] = n_D, [4] = jobs of one hierarchical stage
  int* seg; int* seg_ptr;              // fixup tile segments (head, piece code, row of the first tile, tiles | rows of the last tile << 16), per-CTA pointers [n_sms + 1]
  int P_cap;
};

struct Cand { float cost; int K, M, nA; };

// ceil(a / b) of small positive integers, approximate (cost model only): one fast float division
__device__ __forceinline__ int cdiv_f(int a, int b) { return (int)ceilf(__fdividef((float)a, (float)b) * 0.9999f); }
// exact a / b for 0 <= a < 2^22 through a precomputed float reciprocal of b (cheaper than the integer division sequence)
__device__ __forceinline__ int fdiv_i(int a, float inv, int b) { int q = (int)((float)a * inv); const int r = a - q * b; return q + (r >= b) - (r < 0); }
// exact a / b for 0 <= a < 2^22, b >= 1 (one reciprocal + a correction step instead of the integer division sequence)
__device__ __forceinline__ int idiv_f(int a, int b) { return fdiv_i(a, __frcp_rn((float)b), b); }
// [replay] checkpoint mode: the pieces of a tail are NOT equal.  The chain releases piece p's entering state at ~ R0 + ch_item p, so the replay of late pieces starts late: pieces keep a plateau size
// for the first m - k pieces (they are capped by the dual launch, whose duration is set by the longest piece) and then shrink by one quantum every two pieces (slope ch_item / (rate * quantum)):
// finish times of the tapering pieces are equal and the last one is ~k * ch_item earlier than with equal pieces (E10: 1x8192 H12 -3.8%).  Sizes in quanta: b (+1 for the first rem pieces) - floor((p - p0)^2 / 4).
#define KDA_TAPER_S 2      // piece sizes drop by one quantum every KDA_TAPER_S pieces
#define KDA_TAPER_KN 1     // number of tapering pieces k = m * KN / KD
#define KDA_TAPER_KD 2
// sum over the tapering pieces j = 0 .. i - 1 of floor((j + 1) / S) (closed form: t = i / S, r = i % S: S t (t - 1) / 2 + t (r + 1))
__device__ __forceinline__ int taper_c(int i) { const int t = i / KDA_TAPER_S, r = i - t * KDA_TAPER_S; return KDA_TAPER_S * (t * (t - 1) >> 1) + t * (r + 1); }
__device__ __forceinline__ int taper_k(int m, int Q, int ck_n) {
  if (!ck_n || m < 4) return 0;
  return min((m * KDA_TAPER_KN) / KDA_TAPER_KD, max(0, KDA_TAPER_S * (Q / m - 2)));
}
__device__ __forceinline__ int pq(int p, int b, int rem, int p0) { const int d = max(0, p - p0); return p * b + min(p, rem) - taper_c(d); }
// piece count of a tail of t chunks (Q = t / 4 quanta) for piece length cp: EXACT, shared by the search and the table builders (they must agree)
__device__ __forceinline__ int pcount(int t, int Q, int cp) { return max(1, min((t + cp - 1) / cp, Q)); }

// state chain of m pieces per head over KH (sequence, head) jobs on S CTAs: flat (one chain of m items per job, sz == 0) or hierarchical (groups of sz pieces: stage 1 + stage 3,
// kernel.py _chain_us); returns the modelled us and the group size
template <bool EXACT> __device__ __forceinline__ float chain_choice(const MpParX& pr, int m, int KH, int S, int* sz_out) {
  if (pr.ck_flat > 0.5f && KH + 8 <= S) {   // [replay] checkpoint mode: flat chain, one CTA per (sequence, head), item period ch_item_ck (state in TMEM, S_loc seeded into the accumulator, stores via TMA)
    if (sz_out) *sz_out = 0;
    return pr.ch_fixed_ck + pr.ch_item_ck * (float)(m - 1);
  }
  const int waves = EXACT ? idiv_f(KH + S - 1, S) : cdiv_f(KH, S);
  float best = waves * (pr.ch_fixed + pr.ch_item * m);
  int bsz = 0;
  if (pr.hier_on && m >= pr.hier_min) {
    // t(sz) = w1 (ch_fixed + dual_item sz) + gap + w1 (ch_fixed - 2 + hier_item (G - 2 + max(sz, last))) is minimal near sz = sqrt(hier_item m / (dual_item + hier_item)): four sizes around it
    const int s0 = max(2, (int)(sqrtf(pr.hier_item * m / (pr.dual_item + pr.hier_item)) + 0.5f));
    #pragma unroll 1
    for (int d = -1; d <= 2; d++) {
      const int sz = s0 + d;
      if (sz < 2 || sz >= m) continue;
      const int G = EXACT ? idiv_f(m + sz - 1, sz) : cdiv_f(m, sz);
      if (G < 3) continue;
      if (KH * (2 * G - 1) > S - 4) continue;   // [long] stage 1, stage 3 and the flat chain launch (K * H CTAs) run concurrently (flag-synchronised) with the replay gate: all their working CTAs must be resident at once
      const int last = m - (G - 1) * sz;
      const int w1 = EXACT ? idiv_f(KH * (G - 1) + S - 1, S) : cdiv_f(KH * (G - 1), S);
      const float t = w1 * (pr.ch_fixed + pr.dual_item * sz) + pr.hier_gap + w1 * (pr.ch_fixed - 2.f + pr.hier_item * (G - 2 + max(sz, last)));
      if (t < best * 0.95f) { best = t; bsz = sz; }
    }
  }
  if (sz_out) *sz_out = bsz;
  return best;
}

// head CTAs the whole tasks of the uncut sequences need to finish within tau: H tasks per sequence, total cost sumw per head; a snake-dealt bin of tasks of size <= wmax overshoots the average by
// ~ wmax / 2 (granularity: 40 equal tasks on 19 bins = 3 on some)
__device__ __forceinline__ int whole_bins(const MpParX& pr, int H, float sumw, float wmax, float tau) {
  if (!(sumw > 0.f)) return 0;
  const float room = fmaxf(tau - (pr.wrap ? fminf(0.5f * wmax, pr.wrap_pen + 0.03f * tau) : 0.5f * wmax), 0.25f * tau);
  return (int)ceilf(__fdividef(pr.whole_pack * H * sumw, room));
}

// cost of a cut plan (K cut sequences, M pieces; the fields of eval_cand) as a function of the head-group CTAs the whole tasks get.  Without `split` the whole tasks must finish with the cut heads (tau) and the
// head group is sized for that; with `split` they only have to finish by the end of the chain-critical path (head / dual -> chain -> replay): fewer CTAs are enough and the dual / replay launches get the rest.
// The CTA count of the whole tasks is scanned (geometric steps below the tau-sized count) for the minimum of max(critical path, whole-task end).  Returns the modelled us and the whole-task CTA count.
__device__ __forceinline__ float split_scan(const MpParX& pr, int H, int S, int ck_n, int K, int M, int mmax, int cpmax, int tails, float sumw, float wmax, float headx, int* nw_out, const float* __restrict__ ctab) {
  const float tau = fmaxf(headx, wmax);
  const int nw0 = whole_bins(pr, H, sumw, wmax, tau);
  const int mh = M * H;
  const int ncand = (pr.split > 0.5f && nw0 > 1) ? 4 : 1;
  const float chain = (mmax < 64 && ctab != nullptr) ? ctab[K * 64 + mmax] : chain_choice<false>(pr, mmax, K * H, S, nullptr);   // table over (cut sequences, pieces): built once per launch
  float best = 1e30f; int bnw = nw0;
  float fac = 1.f;
  #pragma unroll 1
  for (int j = 0; j < ncand; j++, fac *= 0.6f) {
    const int nw = j == 0 ? nw0 : max(1, (int)((float)nw0 * fac + 0.5f));
    const int nA = K * H + nw;
    if (nA + (mh + PIECE_WAVES - 1) / PIECE_WAVES > S) continue;
    const int nD = min(mh, S - nA);
    const int waves = cdiv_f(mh, nD);
    const float ad = nD > pr.dual_busy ? pr.a_dual_busy : pr.a_dual;
    const float dual = waves * (ad * cpmax + pr.f_dual) + pr.f_headl;
    // output tiles (HBM-bound passes over the tail) - or, with checkpoints, the replay of every piece as a plain launch from its true entering state (one chain per piece, ~head rate)
    const float tiles = ck_n ? waves * (pr.a_head * cpmax + pr.f_headl) + pr.f_headl : pr.tile_fixed + __fdividef(32.f * tails * H * 768.f, pr.tile_bw);
    float crit = fmaxf(tau + pr.f_headl, dual) + chain + tiles;
    if (pr.split > 0.5f && sumw > 0.f) {   // whole-task end: the average bin load + the granularity overshoot of the deal
      const float room = __fdividef(pr.whole_pack * H * sumw, (float)nw);
      const float wend = fmaxf(room + (pr.wrap ? fminf(0.5f * wmax, pr.wrap_pen + 0.03f * room) : 0.5f * wmax), wmax) + pr.f_headl;
      crit = fmaxf(crit, wend);
    }
    const float cost = crit + pr.gaps + pr.ovh;
    if (cost < best) { best = cost; bnw = nw; }
  }
  *nw_out = bnw;
  return best;
}

__device__ __forceinline__ Cand eval_cand(const MpParX& pr, const int* c, int N, int H, int S, int x, int cp, int qn, int ck_n, const float* __restrict__ ctab, float inv_qn) {
  int K = 0, M = 0, mmax = 1, tails = 0, cpmax = 0;
  if (ck_n) x = x - x % qn;     // checkpoint mode: head and pieces are whole numbers of checkpoint intervals (a piece replays as its own sequence: its local chunk index % ck_n is the global one)
  const int qmin = max(4, qn), tmin = max(8, qn);
  float sumw = 0.f, wmax = 0.f;
  const float inv_cp = __frcp_rn((float)cp);   // exact integer divisions by float reciprocal (fdiv_i), one reciprocal per candidate instead of two integer divisions per sequence
  for (int i = 0; i < N; i++) {
    const int ci = c[i];
    if (x >= qmin && ci - x >= tmin) {
      const int t = ci - x, Q = fdiv_i(t, inv_qn, qn);
      const int m = max(1, min(fdiv_i(t + cp - 1, inv_cp, cp), Q));
      const int kt = 0;
      const int pl = qn * cdiv_f(Q + taper_c(kt), m) + (t - Q * qn);
      K++; M += m; tails += t; mmax = max(mmax, m); cpmax = max(cpmax, pl);
    } else {
      const float w = pr.a_head * ci + pr.f_task;
      sumw += w; wmax = fmaxf(wmax, w);
    }
  }
  Cand r; r.K = K; r.M = M; r.nA = 0;
  if (K == 0) {   // no cut: whole tasks LPT-packed on min(S, N * H) CTAs
    r.cost = fmaxf(wmax, pr.whole_pack * H * sumw * __fdividef(1.f, (float)S)) + pr.f_headl;
    r.nA = min(S, N * H);
    return r;
  }
  int bestnw = 0;
  r.cost = split_scan(pr, H, S, ck_n, K, M, mmax, cpmax, tails, sumw, wmax, pr.a_head * (float)x + pr.f_task, &bestnw, ctab);
  r.nA = K * H + bestnw;
  return r;
}

// geometric grid value j of n between lo and hi (inclusive)
__device__ __forceinline__ int gval(float lo, float hi, int j, int n) { return n > 1 ? (int)(lo * __expf(__logf(__fdividef(hi, lo)) * ((float)j * __fdividef(1.f, (float)(n - 1)))) + 0.5f) : (int)(lo + 0.5f); }

// snake deal of n sorted items over ncta bins (start offset of every bin), same arithmetic as the lean plain planner
__device__ __noinline__ int snake_start(int k, int ncta, int T) {   // [chain] one shared copy (five call sites: cold-fetched once); exact reciprocal division instead of the integer division sequence (T < 2^22)
  const int full = idiv_f(T, ncta), rem = T - full * ncta;
  return k * full + ((full & 1) ? max(0, k - (ncta - rem)) : min(k, rem));
}

__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

__global__ void __launch_bounds__(NTHR) plan_mp_kernel(const long long* __restrict__ cu, int N, int H, int S, int ncta0, int ck_n, MpPar pr0, MpOut o, long long* dbg,
                                                       long long* cu_out, long long Ttok, int* pstat, int slot0, kdaval::VArgs vv) {
  extern __shared__ __align__(16) unsigned char dsm[];
  const int tid = threadIdx.x;
  if (blockIdx.x >= 2) { kdaval::pad_block(vv, blockIdx.x - 2); return; }
  if (blockIdx.x == 1) { kdaval::validate_block_ni(vv, (int*)dsm, false); kdaval::zero_out_if_bad(vv); return; }
  __shared__ int s_c[NMAX], s_len[NMAX];
  __shared__ int s_bad;
  __shared__ float s_ctab[(NMAX + 1) * 64];
  __shared__ int s_cna[256], s_bna;   // [chain] head-group CTAs (K H + whole-task CTAs) of every candidate of the current search stage / of the best so far
  __shared__ unsigned long long s_best;
  __shared__ int s_x, s_cp;
  __shared__ float s_lim[4];          // current x / cp ranges of the search
  __shared__ int s_bx, s_bcp;
  __shared__ float s_bcost;
  __shared__ int s_stop;
  __shared__ int s_K, s_M, s_nA, s_nD;
  __shared__ int s_cutf[NMAX], s_slot[NMAX], s_gp0g[NMAX];                                             // per sequence
  __shared__ int s_seq[NMAX], s_xabs[NMAX], s_eabs[NMAX], s_q4d[NMAX], s_r4[NMAX], s_p0[NMAX], s_tph[NMAX], s_base[NMAX + 1], s_ms[NMAX], s_gp0s[NMAX], s_pms[NMAX], s_sz[NMAX], s_jb[NMAX + 1],
      s_G[NMAX], s_qg[NMAX + 1];                                                                       // per slot
  __shared__ int s_fb[MAXS], s_cnt[MAXS + 1], s_wt[8];
  __shared__ float s_itph[NMAX], s_iq1[NMAX], s_iq[NMAX], s_im[NMAX];   // reciprocals per slot
  __shared__ float s_iH, s_iS;
  __shared__ int s_wsq[NMAX], s_wn[NMAX], s_wc[NMAX], s_wPg[NMAX + 1];             // wrap plan: uncut sequences sorted by chunks (descending): sequence, chunks, task cost, prefix cost
  __shared__ int s_wld[MAXS], s_wsn[MAXS], s_wptr[MAXS + 1], s_wt2[8];   // per bin: entries, snake load, entry offset; warp sums
  __shared__ unsigned int s_wM, s_sM, s_vM[NVAR];
  __shared__ int s_snake, s_bv;
  int* s_dlen = (int*)dsm;                  // [DUAL_CAP]
  int* s_dsort = s_dlen + DUAL_CAP;         // [DUAL_CAP]
  int* s_var = s_dsort + DUAL_CAP;          // [NVAR * (MAXS + 2) * 5] wrap-plan variants: raw task, offset, boundary task, cut flag, producer chunks
  const long long t0 = clock64();
  MpParX pr; static_cast<MpPar&>(pr) = pr0;
  pr.ck_flat = ck_n > 0 ? 1.f : 0.f; pr.ch_item_ck = pr0.tile_bw; pr.ch_fixed_ck = pr0.tile_fixed;   // [replay] checkpoint-mode chain constants (see MpParX)
  pr.wrap = slot0 > 0 ? pr0.wrap : 0;   // [short] wrap plan also with checkpoints (cio entries carry global chunk indices; hand-off slots live behind the piece slots)
  const int wrap_k0 = pr.wrap;           // [sched] checkpoint mode: the wrap plan only serves layouts without a cut (K == 0, N * H > SMs); with a cut the snake deal of the whole tasks is as good (they are not on the critical path of the chain) and the wrap builder costs ~3 us of planner latency + hand-off spin in the whole-task launch
  if (ck_n > 0) pr.wrap = 0;
  if (ck_n > 0) pr.a_head += pr0.ck_row_us / (float)ck_n;      // a checkpoint row (64 KB per head task) costs the writing CTA ~ck_row_us: head and replay chunks are slower with checkpoints
  const int qn = ck_n > 0 ? ck_n : 4;      // piece / head quantum in chunks (4: one output tile of 128 tokens)
  const float inv_qn = __frcp_rn((float)qn);
  const int qtok = 32 * qn;

  // ---- cu_seqlens loads first (one entry per thread: N + 1 <= 33 <= NTHR), then the chain model table (independent of cu) while they are in flight
  long long pre0 = 0, pre1 = 0;
  if (cu_out != nullptr && tid <= N) { pre0 = cu[tid]; pre1 = tid < N ? cu[tid + 1] : pre0; }
  // ---- chain model table: chain_choice<false>(m pieces, K cut sequences) for every (K <= N, m < 64): evaluated once here (parallel over all threads) instead of per candidate
  for (int idx = tid; idx < (N + 1) * 64; idx += NTHR) {
    const int Kt = idx >> 6, mt = idx & 63;
    s_ctab[idx] = (Kt >= 1 && mt >= 1) ? chain_choice<false>(pr, mt, Kt * H, S, nullptr) : 0.f;
  }
  // ---- cu_seqlens (sanitised copy for the kernels of this launch, as the lean planners)
  if (tid == 0) s_bad = 0;
  __syncthreads();
  if (cu_out != nullptr) {
    int bad = 0;
    if (tid <= N) {
      if (pre0 < 0 || pre0 > Ttok || pre1 < pre0 || (tid == 0 && pre0 != 0)) bad = 1;
      cu_out[tid] = pre0;
    }
    bad = __syncthreads_or(bad);
    if (bad) { for (int i = tid; i <= N; i += NTHR) cu_out[i] = 0; }
    if (tid == 0) { s_bad = bad; if (pstat) *pstat = bad; }
    __syncthreads();
  }
  const int bad = s_bad;
  if (tid < N) {
    const long long len = bad ? 0 : cu[tid + 1] - cu[tid];
    s_len[tid] = len > 0 ? (int)len : 0;
    s_c[tid] = len > 0 ? (int)((len + 31) >> 5) : 0;
  }
  __syncthreads();
  const int myc_ = (tid & 31) < N ? s_c[tid & 31] : 0;           // N <= 32: one warp-wide reduction instead of a loop over the sequences
  const int cmax = __reduce_max_sync(0xffffffffu, myc_);
  if (dbg && tid == 0) dbg[4] = clock64() - t0;

  // ---- candidate search: (x, cp) on geometric grids, one candidate per thread; stage 0 spans the whole range, every next stage zooms to +- one step around the best so far
  const float xmax = (float)max(5, cmax - 8), cpmax_ = (float)max(cmax, 16);
  if (tid == 0) { s_lim[0] = 4.f; s_lim[1] = xmax; s_lim[2] = 8.f; s_lim[3] = cpmax_; s_bx = 4; s_bcp = 8; s_bcost = 1e30f; s_stop = 0; }
  __syncthreads();
  const float nc_cost = eval_cand(pr, s_c, N, H, S, 1 << 30, 8, qn, ck_n, s_ctab, inv_qn).cost;
  // lower bound of every cut plan (cf. work_planref NOTES 1.3): all chunks at the head rate (the fastest) over S CTAs + the fixed terms; when even that cannot beat the no-cut plan by the margin the search is skipped.
  // Exact for the model (max(tau, dual) >= fluid share of the work; checked on 142k random triples there); checkpoint mode has other fixed terms: no prune.
  bool prune = false;
  if (ck_n == 0 && !bad) {
    float sc_ = 0.f;
    for (int i = 0; i < N; i++) sc_ += (float)s_c[i];
    const float lb = __fdividef(pr.a_head * sc_ * (float)H, (float)S) + pr.ovh + pr.gaps + pr.tile_fixed + pr.ch_fixed;
    prune = lb >= nc_cost * (1.f - pr.margin) && pr.force_x <= 0;
  } else if (!bad && pr.force_x <= 0) {
    // [ckprune] checkpoint mode: every cut plan runs the tails of the longest sequences (c = cmax chunks, n of them; they must all be cut or the uncut one is already the no-cut critical path) through the dual
    // pass and the replay pass, and the CTAs those passes can use are at most S - n H (the cut heads take n H): waves * piece length >= n H u / (S - n H) for a tail of u = cmax - x chunks, whichever way
    // the pieces are cut.  crit >= max(head a x + f_task + f_headl, dual) + ch_fixed + replay, minimised over u (piecewise linear: the two ends and the crossing).  When even that cannot beat the
    // no-cut plan by the margin the search is skipped (the decision is the one the full search makes, tools/plan_eq.py).
    const int nmax_ = __popc(__ballot_sync(0xffffffffu, (tid & 31) < N && myc_ == cmax));
    const float k0h = (float)(nmax_ * H);
    if (k0h >= (float)S || cmax < 12) prune = true;
    else {
      const float wk = k0h / ((float)S - k0h), ad = fminf(pr.a_dual, pr.a_dual_busy), a = pr.a_head, cm = (float)cmax;
      const float us = (a * cm + pr.f_task - pr.f_dual) / (a + ad * wk);
      float lb = 1e30f;
      #pragma unroll 1
      for (int k = 0; k < 3; k++) {   // u = 8, u = cmax - 4, the crossing of the head and dual terms (when inside): one copy of the formula
        const float u = k == 0 ? 8.f : (k == 1 ? cm - 4.f : us);
        if (k == 2 && !(us > 8.f && us < cm - 4.f)) break;
        lb = fminf(lb, fmaxf(a * (cm - u) + pr.f_task + pr.f_headl, ad * wk * u + pr.f_dual + pr.f_headl) + pr.ch_fixed + a * wk * u + 2.f * pr.f_headl + pr.gaps + pr.ovh);
      }
      prune = lb >= nc_cost * (1.f - pr.margin);
    }
  }
  if (cmax >= 12 && !bad && !prune) {
    #pragma unroll 1
    for (int st = 0; st < pr.stages && !s_stop; st++) {
      const float xlo = s_lim[0], xhi = s_lim[1], clo = s_lim[2], chi = s_lim[3];
      if (tid == 0) s_best = ~0ull;
      __syncthreads();
      if (tid < 256) {
        const int j = tid >> 4, k = tid & 15;
        const int gx_ = gval(xlo, xhi, j, 16), gc_ = gval(clo, chi, k, 16);
        const Cand r = eval_cand(pr, s_c, N, H, S, gx_, gc_, qn, ck_n, s_ctab, inv_qn);
        s_cna[tid] = r.nA;
        unsigned long long key = ((unsigned long long)__float_as_uint(r.cost) << 32) | (unsigned)tid;
        #pragma unroll 1
        for (int off = 16; off > 0; off >>= 1) key = min(key, __shfl_xor_sync(0xffffffffu, key, off));
        if ((tid & 31) == 0) atomicMin(&s_best, key);
        if (dbg && tid == 0) dbg[st == 0 ? 12 : 16] = clock64() - t0;
      }
      __syncthreads();
      if (dbg && tid == 0) dbg[st == 0 ? 13 : 17] = clock64() - t0;
      if (tid == 0) {
        const int id = (int)(s_best & 0xffffffffu);
        const float bc = __uint_as_float((unsigned)(s_best >> 32));
        const int bx = gval(xlo, xhi, id >> 4, 16), bcp = gval(clo, chi, id & 15, 16);
        if (bc < s_bcost) { s_bcost = bc; s_bx = bx; s_bcp = bcp; s_bna = s_cna[id]; }
        if (st == 0 && s_bcost > nc_cost * (1.f - pr.margin) * 1.2f) s_stop = 1;     // the coarse optimum is far from profitable: no cut (saves two stages)
        const float rx = __expf(__logf(__fdividef(xhi, xlo)) * (1.f / 15.f)), rc = __expf(__logf(__fdividef(chi, clo)) * (1.f / 15.f));
        s_lim[0] = fmaxf(4.f, bx / rx); s_lim[1] = fminf(xmax, fmaxf(s_lim[0], bx * rx));
        s_lim[2] = fmaxf(8.f, bcp / rc); s_lim[3] = fminf(cpmax_, fmaxf(s_lim[2], bcp * rc));
        if (dbg) dbg[st == 0 ? 14 : 18] = clock64() - t0;
      }
      __syncthreads();
      if (dbg && tid == 0) dbg[st == 0 ? 15 : 19] = clock64() - t0;
    }
  }
  if (tid == 0) {
    const float bc = s_bcost;
    bool cut = cmax >= 12 && !bad && bc < nc_cost * (1.f - pr.margin);
    if (pr.force_x > 0 && cmax >= 12 && !bad) { const Cand rf = eval_cand(pr, s_c, N, H, S, pr.force_x, max(1, pr.force_cp), qn, ck_n, s_ctab, inv_qn); if (rf.cost < 1e29f) { cut = true; s_bx = pr.force_x; s_bcp = max(1, pr.force_cp); s_bna = rf.nA; } }   // an infeasible forced pair (capacity) is ignored
    s_x = cut ? (ck_n ? s_bx - s_bx % qn : s_bx) : (1 << 30);
    s_cp = cut ? s_bcp : 8;
    if (dbg) { dbg[0] = (long long)(bc * 100); dbg[1] = (long long)(nc_cost * 100); dbg[2] = (long long)s_bx * 100000 + s_bcp; }
  }
  __syncthreads();
  const int x = s_x, cp = s_cp;
  if (dbg && tid == 0) dbg[5] = clock64() - t0;

  // ---- per-sequence decisions: warp 0, lane = sequence; scans over the sequences by shuffles
  if (tid < 32 && x == (1 << 30)) {   // [k0] no cut: the generic code below yields K = M = 0, one-task-per-CTA head sizing and empty slot / piece tables (checked equal by tools/plan_eq.py)
    if (tid < N) { s_cutf[tid] = 0; s_slot[tid] = 0; s_gp0g[tid] = 0; }
    if (tid == 0) {
      const int nA = max(1, min(min(S, N * H), ncta0));
      s_iH = __frcp_rn((float)H); s_iS = __frcp_rn((float)S);
      s_K = 0; s_M = 0; s_nA = nA; s_nD = 1; s_jb[0] = 0; s_qg[0] = 0; s_base[0] = 0;
      o.geom[0] = 0; o.geom[1] = 0; o.geom[2] = nA; o.geom[3] = 1; o.geom[4] = 0;
    }
  } else if (tid < 32) {
    const int lane = tid;
    int cf = 0, m = 0, ci = 0, tph = 0, Q = 0;
    if (lane < N) {
      ci = s_c[lane];
      cf = (x >= max(4, qn) && ci - x >= max(8, qn)) ? 1 : 0;
      if (cf) { const int t = ci - x; Q = fdiv_i(t, inv_qn, qn); m = max(1, min(idiv_f(t + cp - 1, cp), Q)); tph = (t + 3) >> 2; }      // tph: output tiles of 128 tokens per head (the chunk count t rounds up to whole tiles)
    }
    const unsigned bal = __ballot_sync(0xffffffffu, cf);
    const int K = __popc(bal);
    const int slot = __popc(bal & ((1u << lane) - 1));
    int sz = 0;
    if (cf) chain_choice<true>(pr, m, K * H, S, &sz);
    const int G = sz ? (m + sz - 1) / sz : 0;
    const int jobs = sz ? H * (G - 1) : 0;
    // inclusive scans (over the cut sequences, in sequence order = slot order) of m, G, jobs, tile count
    int im = m, ig = G, ij = jobs, it = cf ? H * tph : 0;
    #pragma unroll 1
    for (int off = 1; off < 32; off <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, im, off), b = __shfl_up_sync(0xffffffffu, ig, off), c2 = __shfl_up_sync(0xffffffffu, ij, off), d = __shfl_up_sync(0xffffffffu, it, off);
      if (lane >= off) { im += a; ig += b; ij += c2; it += d; }
    }
    const int pm = im - m, qg = ig - G, jb = ij - jobs, base = it - (cf ? H * tph : 0);
    const int Mtot = __shfl_sync(0xffffffffu, im, 31), Gtot = __shfl_sync(0xffffffffu, ig, 31), Jtot = __shfl_sync(0xffffffffu, ij, 31), Utot = __shfl_sync(0xffffffffu, it, 31);
    // head-side sizing: whole tasks of the uncut sequences
    float sw = 0.f, wm = 0.f;
    if (lane < N && !cf) { sw = pr.a_head * ci + pr.f_task; wm = sw; }
    int tl = cf ? ci - x : 0, mm = cf ? m : 1, plx = cf ? qn * cdiv_f(Q, m) + (tl - Q * qn) : 0;   // tail chunks, pieces per head, longest piece (split_scan inputs)
    #pragma unroll 1
    for (int off = 16; off > 0; off >>= 1) {
      sw += __shfl_xor_sync(0xffffffffu, sw, off); wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, off));
      tl += __shfl_xor_sync(0xffffffffu, tl, off); mm = max(mm, __shfl_xor_sync(0xffffffffu, mm, off)); plx = max(plx, __shfl_xor_sync(0xffffffffu, plx, off));
    }
    if (lane < N) { s_cutf[lane] = cf; s_slot[lane] = slot; s_gp0g[lane] = pm + slot; }
    if (cf) {
      const long long bos = cu[lane];
      const int sl = slot;
      s_seq[sl] = lane; s_ms[sl] = m; s_gp0s[sl] = pm + slot; s_pms[sl] = pm;
      s_xabs[sl] = (int)(bos + 32ll * x); s_eabs[sl] = (int)(bos + s_len[lane]);
      const float im_ = __frcp_rn((float)m); const int kt_ = taper_k(m, Q, ck_n); const int Qe_ = Q + taper_c(kt_); const int q4_ = fdiv_i(Qe_, im_, m);
      s_q4d[sl] = q4_; s_r4[sl] = Qe_ - q4_ * m; s_p0[sl] = m - kt_;
      s_itph[sl] = __frcp_rn((float)tph); s_iq1[sl] = __frcp_rn((float)(q4_ + 1)); s_iq[sl] = __frcp_rn((float)max(1, q4_)); s_im[sl] = im_; s_tph[sl] = tph; s_sz[sl] = sz; s_G[sl] = G; s_jb[sl] = jb; s_qg[sl] = qg; s_base[sl] = base;
    }
    if (lane == 0) {
      int nA;
      if (K == 0) nA = min(S, N * H);
      else nA = s_bna;   // [chain] the head-group CTA count of the winning candidate (evaluated by the search with the same split_scan): no second serial evaluation here
      nA = max(1, min(nA, ncta0));
      const int nD = K ? max(1, min(Mtot * H, S - nA)) : 1;
      s_iH = __frcp_rn((float)H); s_iS = __frcp_rn((float)S);
      s_K = K; s_M = Mtot; s_nA = nA; s_nD = nD; s_jb[K] = Jtot; s_qg[K] = Gtot; s_base[K] = Utot;
      o.geom[0] = K; o.geom[1] = Mtot; o.geom[2] = nA; o.geom[3] = nD; o.geom[4] = Jtot;
    }
  }
  __syncthreads();
  if (dbg && tid == 0) dbg[6] = clock64() - t0;
  const int K = s_K, M = s_M, nA = s_nA, nD = s_nD;
  if (K == 0 && N * H <= nA) {
    // [k0] nothing cut and one whole task per CTA (N * H <= the head grid): every launch of the structure except B2 is empty, so only the tables those launches read are written (B2's task list in
    // sequence order: all its CTAs start together, the order only decides which SM each task lands on).  The generic builders below would write the same empty tables, a sorted B2 list and unused
    // descriptors, ~10 k cycles later (tools/plan_eq.py compares the tables the launches read).
    const int NH = N * H;
    for (int k = tid; k <= ncta0 + 1; k += NTHR) {
      const int kk = min(k, ncta0);
      o.cta_ptr[k] = kk <= nA ? kk : NH;
      o.cta_ptr[ncta0 + 2 + k] = 0;
      o.cta_ptr[2 * (ncta0 + 2) + k] = 0;
      o.cta_ptr[3 * (ncta0 + 2) + k] = kk <= nA ? kk : NH;
    }
    for (int r = tid; r < NH; r += NTHR) o.tlb[r] = r;
    for (int k = tid; k <= S; k += NTHR) { o.dptr[k] = 0; o.seg_ptr[k] = 0; }
    for (int i = tid; i <= o.P_cap; i += NTHR) o.cu_tail[i] = 0;
    return;
  }

  // ---- three independent builders on separate warp groups
  if (tid < 512) {   // ---------------- dual tasks: (real piece, head) sorted by length (descending), snake deal over nD CTAs
    const int Mh = M * H;
    const float iH = s_iH;
    for (int u = tid; u < Mh; u += 512) {
      const int pp = fdiv_i(u, iH, H);
      int s = 0;
      while (s + 1 < K && s_pms[s + 1] <= pp) s++;
      const int p = pp - s_pms[s], m = s_ms[s];
      const int q4d = s_q4d[s], r4 = s_r4[s], p0 = s_p0[s];
      const int st = qtok * pq(p, q4d, r4, p0);
      const int en = p < m - 1 ? qtok * pq(p + 1, q4d, r4, p0) : s_eabs[s] - s_xabs[s];
      s_dlen[u] = en - st;                       // tokens (the chunk count is monotone in it)
    }
    named_sync(1, 512);
    for (int u = tid; u < Mh; u += 512) {
      const int lu = s_dlen[u];
      int r = 0;
      for (int v = 0; v < Mh; v++) { const int lv = s_dlen[v]; r += (lv > lu) || (lv == lu && v < u); }
      const int pp = fdiv_i(u, iH, H), h = u - pp * H;
      int s = 0;
      while (s + 1 < K && s_pms[s + 1] <= pp) s++;
      s_dsort[r] = (s_gp0s[s] + (pp - s_pms[s])) * H + h;
    }
    named_sync(1, 512);
    for (int r = tid; r < Mh; r += 512) {
      const int q = r / nD, pos = r - q * nD, k = (q & 1) ? nD - 1 - pos : pos;
      o.dtask[snake_start(k, nD, Mh) + q] = s_dsort[r];
    }
    for (int k = tid; k <= S; k += 512) o.dptr[k] = (Mh == 0 || k > nD) ? Mh : snake_start(k, nD, Mh);
    if (dbg && tid == 0) dbg[8] = clock64() - t0;
  } else if (tid < 768) {   // ---------------- fixup tile segments: all H * tph tiles of every cut sequence in (slot, head, tile) order, cut into S equal contiguous ranges, each cut at (head, piece) boundaries
    const int t = tid - 512;     // one thread per CTA b < S
    const int U = ck_n ? 0 : s_base[K];      // checkpoint mode: no output tiles (the replay launch computes the tail outputs)
    const int b = t;
    int cnt = 0, fb = 0;
    if (b < S) {
      const int lo = fdiv_i(U * b, s_iS, S), hi = fdiv_i(U * (b + 1), s_iS, S);   // U * S < 2^31: tiles <= 2^23 per head set
      if (hi > lo) {
        auto unit_of = [&](int g) {   // unit index (slot-major, head, piece) of global tile g
          int s = 0;
          while (s + 1 < K && s_base[s + 1] <= g) s++;
          const int tph = s_tph[s], lt = g - s_base[s], h = fdiv_i(lt, s_itph[s], tph), tt = lt - h * tph;
          const int q4d = s_q4d[s], r4 = s_r4[s], m = s_ms[s];
          const int split = r4 * (q4d + 1);
          int p = tt < split ? fdiv_i(tt, s_iq1[s], q4d + 1) : r4 + fdiv_i(tt - split, s_iq[s], max(1, q4d));
          p = min(p, m - 1);
          return s_pms[s] * H + h * m + p;
        };
        fb = unit_of(lo);
        cnt = unit_of(hi - 1) - fb + 1;
      }
      s_fb[b] = fb;
    }
    if (dbg && t == 0) dbg[11] = clock64() - t0;
    {   // exclusive prefix of the counts over the 8 warps of this group
      const int lane = t & 31, w = t >> 5;
      int inc = cnt;
      #pragma unroll 1
      for (int off = 1; off < 32; off <<= 1) { const int u = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += u; }
      if (lane == 31) s_wt[w] = inc;
      named_sync(2, 256);
      int base = 0;
      for (int i = 0; i < w; i++) base += s_wt[i];
      const int excl = base + inc - cnt;
      if (b < S) s_cnt[b] = excl;
      if (b == S - 1) s_cnt[S] = base + inc;
      named_sync(2, 256);
    }
    for (int bb = t; bb <= S; bb += 256) o.seg_ptr[bb] = s_cnt[bb];
    const int nseg = s_cnt[S];
    for (int z = t; z < nseg; z += 256) {
      int lo_b = 0, hi_b = S - 1;                      // CTA b with s_cnt[b] <= z < s_cnt[b + 1] (empty CTAs share their prefix)
      while (lo_b < hi_b) { const int mid = (lo_b + hi_b + 1) >> 1; if (s_cnt[mid] <= z) lo_b = mid; else hi_b = mid - 1; }
      const int bq = lo_b, u = s_fb[bq] + (z - s_cnt[bq]);
      int s = 0;
      while (s + 1 < K && s_pms[s + 1] * H <= u) s++;
      const int m = s_ms[s], rem = u - s_pms[s] * H, h = fdiv_i(rem, s_im[s], m), p = rem - h * m;
      const int q4d = s_q4d[s], r4 = s_r4[s], tph = s_tph[s];
      const int ps0 = p * q4d + min(p, r4), ps1 = p + 1 < m ? (p + 1) * q4d + min(p + 1, r4) : tph;
      const int jb = s_base[s] + h * tph;
      const int lo = fdiv_i(U * bq, s_iS, S), hi = fdiv_i(U * (bq + 1), s_iS, S);
      const int a0 = max(jb + ps0, lo), a1 = min(jb + ps1, hi);
      const int tail_tok = s_eabs[s] - s_xabs[s];
      const int vlast = (a1 == jb + tph && tail_tok < 128 * tph) ? tail_tok - 128 * (tph - 1) : 128;   // rows of the last tile inside the tail
      int4 sg;
      // piece 0: S_A, stored by the head at scratch slot (first piece id) * H + head
      sg.x = h; sg.y = p == 0 ? -1 - s_gp0s[s] : s_gp0s[s] + p; sg.z = s_xabs[s] + 128 * (a0 - jb); sg.w = (a1 - a0) | (vlast << 16);
      ((int4*)o.seg)[z] = sg;
    }
    if (dbg && t == 0) dbg[9] = clock64() - t0;
  } else {   // ---------------- head tasks + slot geometry + piece list
    const int t = tid - 768;
    // slot rows
    for (int s = t; s < K; s += 256) {
      int* g = o.gs + GS * s;
      g[0] = s_xabs[s]; g[1] = s_eabs[s]; g[2] = s_ms[s]; g[3] = s_gp0s[s]; g[4] = s_seq[s]; g[5] = s_tph[s]; g[6] = s_sz[s]; g[7] = s_jb[s]; g[8] = s_G[s]; g[9] = s_qg[s];
      g[10] = 0; g[11] = 0; g[12] = 0; g[13] = 0; g[14] = 0; g[15] = 0;
    }
    {   // piece list: entries 0 .. Pt - 1 are piece starts and the gap / end entries of every slot; the rest repeats the last end
      const int Pt = K ? s_gp0s[K - 1] + s_ms[K - 1] + 1 : 0;
      const long long elast = K ? (long long)s_eabs[K - 1] : 0ll;
      for (int s = 0; s < K; s++) {
        const int m = s_ms[s], gp0 = s_gp0s[s], xa = s_xabs[s], q4d = s_q4d[s], r4 = s_r4[s], p0 = s_p0[s];
        for (int p = t; p <= m; p += 256) o.cu_tail[gp0 + p] = p < m ? (long long)xa + (long long)qtok * pq(p, q4d, r4, p0) : (long long)s_eabs[s];
      }
      for (int i = Pt + t; i <= o.P_cap; i += 256) o.cu_tail[i] = elast;
    }
    // head list: every cut sequence contributes one task per head (its first x chunks, hand-off slot s * H + h) on a CTA of its own; the whole tasks of the uncut sequences are sorted by
    // length (descending) and snake-dealt over the remaining nA - K * H CTAs (they never share a CTA with a cut head: that one is the longest chain already)
    const int NH = N * H, NHc = K * H, NHw = NH - NHc;
    const int nW = max(1, nA - NHc);
    const bool wrap_ok = (K == 0 ? wrap_k0 : pr.wrap) && nW > 1 && NHw > nW;
    for (int u = t; u < NHc; u += 256) {   // cut heads: entry u = (slot, head)
      const int sl = fdiv_i(u, s_iH, H), h = u - sl * H, sq = s_seq[sl];
      int* e = o.desc + 4 * u;
      *(int4*)e = make_int4(sq * H + h, 0, x, (s_gp0g[sq] * H + h + 1) << 16);
    }
    if (t < 32) {   // uncut sequences sorted by chunks (descending, ties by index): the whole tasks in this order are sequence rank r / H, head r % H (warp 0, lane = sequence; cut ones: key -1)
      const int ct = (t < N && !s_cutf[t]) ? s_c[t] : -1;
      int r = 0;
      #pragma unroll 1
      for (int j = 0; j < N; j++) { const int o = __shfl_sync(0xffffffffu, ct, j); r += (o > ct) || (o == ct && j < t); }
      if (ct >= 0) { s_wsq[r] = t; s_wn[r] = ct; s_wc[r] = wcost_m(ct); }
    }
    named_sync(3, 256);
    bool wrap_try = false;
    if (wrap_ok) {   // wrap plan of the whole tasks (cf. lean_cio_body): stream of the uncut tasks sorted by length (descending), cut into nW equal-cost ranges; a task that straddles a boundary is split into a
                     // producer part (its first d chunks, FIRST entry of the lower CTA, state to slot0 + upper CTA) and a consumer part (the rest, LAST entry of the upper CTA).  Kept unless it models < 5% better than the snake deal.
      const int U = N - K;
      if (t == 0) { int a = 0; s_wPg[0] = 0; for (int r = 0; r < U; r++) { a += H * s_wc[r]; s_wPg[r + 1] = a; } s_wM = 0; s_sM = 0; }
      if (t < nW) s_wsn[t] = 0;
      named_sync(3, 256);
      const int W = s_wPg[U];
      const int Lp = (W + (int)MC_CUT * (nW - 1)) / nW;
      for (int w = t; w < NHw; w += 256) {   // snake deal loads
        const int q = w / nW, pos = w - q * nW, k = (q & 1) ? nW - 1 - pos : pos;
        const int wc = s_wc[w / H];
        atomicMax(&s_sM, (unsigned int)(atomicAdd(&s_wsn[k], wc) + wc));
      }
      named_sync(3, 256);
      // the wrap plan cannot beat the fluid bound W / nW: if the snake deal is within 5% of it, keep the snake deal; and with cut heads (K > 0) the whole bins only matter when the snake deal
      // outlasts the head (x chunks): bins that finish under the head time are not on the critical path (the planner latency of the wrap search is ~10 us)
      wrap_try = (long long)s_sM * 100 > (long long)105 * (W / nW) && (K == 0 || (float)s_sM * 0.01f * pr.a_head > pr.wrap_tau * (pr.a_head * (float)x + pr.f_task));
      if (wrap_try) {
        // NVAR variants of the bin spacing (lean_cio_body uses six: Lp_v = Lp + {0, 1, 2, 4, 8, 16} * 0.5%), each cutting the stream at j * (Lp_v - C_CUT): a task at a boundary is cut into a producer part (its
        // first d chunks: FIRST entry of bin j - 1, state to slot0 + j) and a consumer part (the rest: LAST entry of bin j) when it is shorter than a bin by 24 chunks (the producer must be done
        // before the consumer starts), else the boundary is rounded to the nearer task edge.  The variant with the smallest modelled maximum load wins (it moves the rounding positions around).
        const int ST = nW + 2;
        int* v_tb = s_var; int* v_dl = v_tb + NVAR * ST; int* v_E = v_dl + NVAR * ST; int* v_cut = v_E + NVAR * ST; int* v_d = v_cut + NVAR * ST;
        const int lstep = max((int)((long long)Lp * 5 / 1000), 50);
        for (int it = t; it < NVAR * (nW + 1); it += 256) {   // raw boundary (v, j): stream position -> (task, offset)
          const int v = it / (nW + 1), j = it - v * (nW + 1);
          int tb = NHw, dl = 0;
          if (j == 0) tb = 0;
          else if (j < nW) {
            const long long q = (long long)j * (Lp + woff(v) * lstep - (int)MC_CUT);
            if (q < W) {
              int lo = 0, hi = U - 1;
              while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (s_wPg[mid] <= q) lo = mid; else hi = mid - 1; }
              const int w = s_wc[lo], rem = (int)q - s_wPg[lo];
              const int h = min(rem / w, H - 1);
              tb = lo * H + h; dl = rem - h * w;
            }
          }
          v_tb[v * ST + j] = tb; v_dl[v * ST + j] = dl;
        }
        if (t < NVAR) s_vM[t] = 0;
        named_sync(3, 256);
        for (int it = t; it < NVAR * (nW + 1); it += 256) {   // cut decisions
          const int v = it / (nW + 1), j = it - v * (nW + 1);
          const int* tbv = v_tb + v * ST;
          const int Lpv = Lp + woff(v) * lstep;
          int E = tbv[j], cut = 0, d = 0;
          if (j > 0 && j < nW && E < NHw) {
            const int n = s_wn[E / H], w = s_wc[E / H], dl = v_dl[v * ST + j];
            d = dl >= (int)MC_ENTRY ? (dl - (int)MC_ENTRY + 50) / 100 : 0;
            const int tprev = j > 1 ? tbv[j - 1] : -1, tnext = j + 1 < nW ? tbv[j + 1] : NHw;
            if (n >= MC_PRELOAD && d >= 2 && n - d >= 2 && tprev < E && tnext > E && n * 100 + 2400 + (int)MC_CUT <= Lpv) cut = 1;
            else { d = 0; if (2 * dl >= w) E++; }
          }
          v_E[v * ST + j] = E; v_cut[v * ST + j] = cut; v_d[v * ST + j] = d;
        }
        named_sync(3, 256);
        auto pre = [&](int e) { const int r = e / H; return r >= U ? s_wPg[U] : s_wPg[r] + (e - r * H) * s_wc[r]; };
        for (int it = t; it < NVAR * nW; it += 256) {   // modelled load of every bin of every variant, maximum per variant
          const int v = it / nW, b = it - v * nW;
          const int* Ev = v_E + v * ST; const int* cv = v_cut + v * ST; const int* dv = v_d + v * ST;
          const int Ej = Ev[b], cj = cv[b], Ej1 = Ev[b + 1], cj1 = cv[b + 1];
          int ld = pre(Ej1) - pre(Ej + cj);
          if (cj1) ld += dv[b + 1] * 100 + (int)MC_ENTRY + (int)MC_FRONT;
          if (cj) ld += (s_wn[Ej / H] - dv[b]) * 100 + (int)MC_ENTRY + (int)MC_BACK;
          atomicMax(&s_vM[v], (unsigned int)ld);
        }
        named_sync(3, 256);
        int bv = 0;
        for (int v = 1; v < NVAR; v++) if (s_vM[v] < s_vM[bv]) bv = v;
        if (t == 0) { s_wM = s_vM[bv]; s_bv = bv; s_snake = !((long long)s_wM * 105 < (long long)s_sM * 100); if (dbg) dbg[7] = ((long long)s_wM << 32) | s_sM; }
        if (t < nW) {   // entry counts of the winning variant
          const int* Ev = v_E + bv * ST; const int* cv = v_cut + bv * ST;
          s_wld[t] = max(0, Ev[t + 1] - (Ev[t] + cv[t])) + cv[t + 1] + cv[t];
        }
        named_sync(3, 256);
      }
    }
    if (wrap_try && !s_snake) {
      {   // exclusive prefix of the entry counts of the bins (CTA = thread)
        const int lane = t & 31, w = t >> 5;
        const int cnt = t < nW ? s_wld[t] : 0;
        int inc = cnt;
        #pragma unroll 1
        for (int off = 1; off < 32; off <<= 1) { const int u = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += u; }
        if (lane == 31) s_wt2[w] = inc;
        named_sync(3, 256);
        int base = 0;
        for (int i = 0; i < w; i++) base += s_wt2[i];
        if (t < nW) s_wptr[t] = base + inc - cnt;
        if (t == nW - 1) s_wptr[nW] = base + inc;
        named_sync(3, 256);
      }
      if (t < nW) {
        int* e = o.desc + 4 * (NHc + s_wptr[t]);
        auto task = [&](int g) { return s_wsq[g / H] * H + (g % H); };
        const int ST = nW + 2;
        const int* Ev = s_var + (2 * NVAR + s_bv) * ST; const int* cv = s_var + (3 * NVAR + s_bv) * ST; const int* dv = s_var + (4 * NVAR + s_bv) * ST;   // v_E, v_cut, v_d of the winning variant
        const int Ej = Ev[t], cj = cv[t], Ej1 = Ev[t + 1], cj1 = cv[t + 1];
        if (cj1) { *(int4*)e = make_int4(task(Ej1), 0, dv[t + 1], (slot0 + t + 2) << 16); e += 4; }
        for (int g = Ej + cj; g < Ej1; g++, e += 4) *(int4*)e = make_int4(task(g), 0, s_wn[g / H], 0);
        if (cj) *(int4*)e = make_int4(task(Ej), dv[t], s_wn[Ej / H], slot0 + t + 1);
      }
      for (int k = t; k <= ncta0 + 1; k += 256) {   // entry ncta0 + 1: a duplicate of the last (even-grid launches pad one empty CTA)
        const int kk = min(k, ncta0);
        o.cta_ptr[k] = kk <= NHc ? kk : (kk > nA ? NHc + s_wptr[nW] : NHc + s_wptr[kk - NHc]);
        o.cta_ptr[ncta0 + 2 + k] = min(kk, NHc);                                          // [split] launch A: the cut heads only
        o.cta_ptr[2 * (ncta0 + 2) + k] = NHc + s_wptr[min(kk, nW)];                       // [split] launch B (cio): the whole tasks (wrap plan: hand-off cuts need the cio kernel), bin k = CTA k
        o.cta_ptr[3 * (ncta0 + 2) + k] = 0;                                                 // [split] launch B2 (plain): only when nothing is cut
      }
      if (dbg && t == 0) dbg[10] = clock64() - t0;
    } else {
    {   // snake deal of the sorted whole tasks over the nW bins
      const float iW = __frcp_rn((float)nW);
      const int full = NHw / nW, rem = NHw - full * nW;
      for (int r = t; r < NHw; r += 256) {
        const int q = fdiv_i(r, iW, nW), pos = r - q * nW, k = (q & 1) ? nW - 1 - pos : pos;
        const int g = fdiv_i(r, s_iH, H);
        int* e = o.desc + 4 * (NHc + k * full + ((full & 1) ? max(0, k - (nW - rem)) : min(k, rem)) + q);
        *(int4*)e = make_int4(s_wsq[g] * H + (r - g * H), 0, s_wn[g], 0);
        o.tlb[k * full + ((full & 1) ? max(0, k - (nW - rem)) : min(k, rem)) + q] = s_wsq[g] * H + (r - g * H);
      }
    }
    for (int k = t; k <= ncta0 + 1; k += 256) {
      const int kk = min(k, ncta0);
      o.cta_ptr[k] = kk <= NHc ? kk : (kk > nA ? NH : NHc + snake_start(kk - NHc, nW, NHw));
      o.cta_ptr[ncta0 + 2 + k] = min(kk, NHc);
      o.cta_ptr[2 * (ncta0 + 2) + k] = K == 0 ? 0 : (kk <= nW ? NHc + snake_start(kk, nW, NHw) : NH);
      o.cta_ptr[3 * (ncta0 + 2) + k] = K == 0 ? (kk <= nW ? snake_start(kk, nW, NHw) : NHw) : 0;
    }
    if (dbg && t == 0) dbg[10] = clock64() - t0;
    }
  }
  __syncthreads();
  if (dbg && tid == 0) dbg[3] = clock64() - t0;
}

static int g_max_dyn_a[64];   // [multi-GPU] per-device opt-in shared-memory size (plan_mp_init() runs once per device under that device)
static int& g_max_dyn_ref() { int d = 0; cudaGetDevice(&d); return g_max_dyn_a[d & 63]; }
#define g_max_dyn (g_max_dyn_ref())
size_t smem_bytes() { return (size_t)(2 * DUAL_CAP + NVAR * (MAXS + 2) * 5) * 4; }
}  // namespace

// Inputs: cu int64 [N + 1]; H; ncta0 = head launch grid (<= N * H); n_sms.  Outputs:
//   desc int32 [4 * N * H], cta_ptr int32 [ncta0 + 1]; dtask int32 [>= 3 * n_sms + 8], dptr int32 [n_sms + 1]; cu_tail int64 [P_cap + 1]; gs int32 [16 * N]; geom int32 [16];
//   seg int32 [4 * (4 * n_sms + 16)], seg_ptr int32 [n_sms + 1]
void plan_mp(torch::Tensor cu, int64_t H, int64_t ncta0, int64_t n_sms, int64_t ck_n, torch::Tensor desc, torch::Tensor cta_ptr, torch::Tensor dtask, torch::Tensor dptr, torch::Tensor cu_tail,
             torch::Tensor gs, torch::Tensor geom, torch::Tensor seg, torch::Tensor seg_ptr, std::vector<double> par, c10::optional<torch::Tensor> dbg, c10::optional<pybind11::dict> val, int64_t slot0) {
  KdaDevGuard dev_guard_(cu);   // [multi-GPU] launch + per-device config on the tensors' device
  const int N = (int)(cu.numel() - 1);
  TORCH_CHECK(N >= 1 && N <= NMAX && n_sms >= 8 && n_sms <= MAXS && ncta0 >= 1 && ncta0 <= n_sms && H >= 1, "plan_mp: arguments out of range");
  TORCH_CHECK(N * H <= NH_CAP, "plan_mp: too many tasks");
  TORCH_CHECK(g_max_dyn > 0, "plan_mp_init() not called");
  TORCH_CHECK(desc.numel() >= 4 * (N * H + ncta0) + 16 + N * H && cta_ptr.numel() >= 4 * (ncta0 + 2) && dtask.numel() >= 3 * n_sms + 8 && dptr.numel() >= n_sms + 1 && gs.numel() >= GS * N && geom.numel() >= 16 &&
              seg.numel() >= 4 * (4 * n_sms + 16) && seg_ptr.numel() >= n_sms + 1, "plan_mp: output tensors too small");
  TORCH_CHECK(par.size() == 28, "plan_mp: 28 model parameters");
  MpPar pr{(float)par[0], (float)par[1], (float)par[2], (float)par[3], (float)par[4], (float)par[5], (float)par[6], (float)par[7], (float)par[8], (float)par[9], (float)par[10],
           (float)par[11], (float)par[12], (float)par[13], (float)par[14], (float)par[15], (float)par[16], (float)par[17], (int)par[18], (int)par[19], (int)par[20], (int)par[21], (int)par[22], (int)par[23], (float)par[24], (float)par[25], (int)par[26], (float)par[27]};
  MpOut o{};
  o.desc = desc.data_ptr<int>(); o.tlb = desc.data_ptr<int>() + 4 * (N * H + ncta0) + 16; o.cta_ptr = cta_ptr.data_ptr<int>(); o.dtask = dtask.data_ptr<int>(); o.dptr = dptr.data_ptr<int>(); o.cu_tail = (long long*)cu_tail.data_ptr<int64_t>();
  o.gs = gs.data_ptr<int>(); o.geom = geom.data_ptr<int>(); o.seg = seg.data_ptr<int>(); o.seg_ptr = seg_ptr.data_ptr<int>(); o.P_cap = (int)cu_tail.numel() - 1;
  kdaval::VArgs va{};
  long long* cu_out = nullptr; long long Ttok = 0; int* pstat = nullptr;
  if (val) {
    va = kdaval::args_from_dict(*val);
    cu_out = va.cu_out; Ttok = va.T; va.cu_out = nullptr;
    pstat = val->contains("pstat") && !(*val)["pstat"].is_none() ? (*val)["pstat"].cast<torch::Tensor>().data_ptr<int>() : nullptr;
    TORCH_CHECK(8 * kdaval::hash_slots(N) <= g_max_dyn, "too many sequences for the table validation");
  }
  plan_mp_kernel<<<val ? 2 + va.pad_blocks : 1, NTHR, (size_t)g_max_dyn, c10::cuda::getCurrentCUDAStream().stream()>>>(
      (const long long*)cu.data_ptr<int64_t>(), N, (int)H, (int)n_sms, (int)ncta0, (int)ck_n, pr, o, dbg ? (long long*)dbg->data_ptr<int64_t>() : nullptr, cu_out, Ttok, pstat, (int)slot0, va);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void plan_mp_init() {
  cudaFree(0);
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaFuncAttributes fa;
  TORCH_CHECK(cudaFuncGetAttributes(&fa, plan_mp_kernel) == cudaSuccess, "cudaFuncGetAttributes failed");
  g_max_dyn = optin - (int)fa.sharedSizeBytes - 1024;   // the same total shared memory as the neighbouring persistent kernels (same carveout)
  TORCH_CHECK((size_t)g_max_dyn >= smem_bytes(), "plan_mp: shared memory");
  TORCH_CHECK(cudaFuncSetAttribute(plan_mp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_max_dyn) == cudaSuccess, "cudaFuncSetAttribute failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("init", &plan_mp_init);
  m.def("plan_mp", &plan_mp, py::arg("cu"), py::arg("H"), py::arg("ncta0"), py::arg("n_sms"), py::arg("ck_n"), py::arg("desc"), py::arg("cta_ptr"), py::arg("dtask"), py::arg("dptr"), py::arg("cu_tail"),
        py::arg("gs"), py::arg("geom"), py::arg("seg"), py::arg("seg_ptr"), py::arg("par"), py::arg("dbg") = py::none(), py::arg("val") = py::none(), py::arg("slot0") = 0);
}
