// [race] one bound for every wait / lifetime trap of this file: KDA_BOUND_NS (default 60 s; production waits are microseconds, CTAs live < 100 ms).  A bound that is too small false-traps on a
// time-sliced or debugged GPU (the globaltimer keeps running while the CTA is descheduled).
#define KDA_BOUND_NS 60000000000ull
// Exact context-parallel fixup for the KDA forward (the recurrence is affine in the state).
// A sequence is split at token x: the head part [0, x) runs from S_init (-> o_A in place, S_A); the tail [x, T) is cut into m consecutive
// pieces (m = 1 or 2, planner), each run twice through the dual-chain kernel: from S = 0 (-> o_loc written in place into the final output,
// S_loc) and with v = 0, S0 = I (-> Qt_t = scale P_t q_t, final P).  Per head (value-first states), piece p with entering state S_in(p)
// (S_in(0) = S_A, S_in(p + 1) = S_fin of piece p):
//   out[tokens of p][v] = o_loc[t][v] + sum_k Qt[t][k] S_in(p)[v][k]    (memory bound: read o_loc + Qt, write out)
//   S_in(p + 1)[v][k]   = S_loc_p[v][k] + sum_m S_in(p)[v][m] P_p[m][k]
// Output (tcgen05, bf16 operands, fp32 accumulation in TMEM): Qt . bf16(S_in) - the operand precision the base kernel itself uses for
// its inter-chunk output term (q . bf16(S)) - plus o_loc, one bf16 rounding of the result.
// State (tcgen05): fp32 operands split into bf16 hi + lo (hi = bf16(x), lo = bf16(x - hi)); A_hi B_hi + A_hi B_lo + A_lo B_hi (the dropped
// lo x lo term is < 2^-16 relative): an fp32-class carry.  A = S_in (K-major), B = P as stored ([m][k] = MN-major operand).  S_loc is staged
// in smem (SW128 fp32 tiles), the result is added in place and written by TMA store; for a chained item the result is also handed to the
// next item as its A operand (hi / lo) straight from registers.
//
// Persistent grid: the host planner (fix_sched.py) cuts the work into per-CTA segments (head, piece, first tile, tile count).  CTA b OWNS the
// heads h = b mod G: it first runs their state chains (m items per head, S_in(1..m-1) -> global scratch sin_, S_fin -> sfin), then its
// segments: the output tiles of its heads' pieces 1..m-1 (their entering state was computed by this very CTA: no inter-CTA dependency) and a
// contiguous range of piece-0 tiles (state S_A) in head-major order.  The B operand bf16(S_in) is rebuilt in smem per segment; tiles stream
// through a 3-stage TMA ring.  sa must not alias sfin / sin_.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cuda.h>

namespace cpfix {
#define HD_STAMP(slot) do {} while (0)
#define HD_STAMPI(slot) do {} while (0)
#define FLS(tid_, slot) do {} while (0)

constexpr int NT = 256;        // 8 epilogue warps: warp w -> TMEM lane quarter w%4 (32 tokens), value half w/4; + warp 8 = TMA / MMA issuer
constexpr int NTH = 288;
constexpr int NTH2 = 384;       // fixup_mpd: + a flag warp (warp 9) and a TMA-load warp (warp 10) for the checkpoint-mode flat chain
constexpr int TM = 128;        // tokens per tile (= MMA M)
constexpr int NS = 3;          // pipeline stages (Qt tile + o tile each)
constexpr int TB = 128 * 256;  // bytes of one 128 x 128 bf16 tile
constexpr int SMEM = 1024 + TB + NS * 2 * TB;  // align slack + bf16(S_A) + stages = 230,400 B

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ uint32_t pack(__nv_bfloat16 a, __nv_bfloat16 b) {
  return (uint32_t)__bfloat16_as_ushort(a) | ((uint32_t)__bfloat16_as_ushort(b) << 16);
}
__device__ __forceinline__ void split2(float2 x, uint32_t& hi, uint32_t& lo) {   // hi = bf16(x), lo = bf16(x - hi), two elements per word
  uint32_t h, l;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(x.y), "f"(x.x));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(l) : "f"(x.y - __uint_as_float(h & 0xffff0000u)), "f"(x.x - __uint_as_float(h << 16)));
  hi = h; lo = l;
}

// ---- tcgen05 helpers ----
// MN-major SW128 operand (rows = K index, 128 B = 64 MN elements, 8-row atoms: SBO 1024 B; 64-wide MN atom columns 16 KB apart: LBO)
__device__ __forceinline__ uint64_t sdesc_mn(uint32_t a, int kk) {
  const uint32_t lo = ((a >> 4) & 0x3FFF) + kk * 128;
  return ((uint64_t)0x40004040u << 32) | (uint64_t)(lo | (1024u << 16));
}
__device__ __forceinline__ void mbar_init(uint32_t a, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c)); }
__device__ unsigned long long kda_wait_fail_rec[2];   // [latent] written right before a bounded mbarrier wait traps
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph) { 
  // [race] t0 is an early-clobber output (+&r): the asm arms it before the phase operand is read again (a constant-0 phase could share its register)
  // [latent] bounded (KDA_BOUND_NS = 60 s, 32-bit window in units of 1024 ns of the 64-bit globaltimer, armed by the first failed try_wait): a lost arrival traps (status in kda_wait_fail_rec) instead of wedging the GPU
  uint32_t t0 = 0;
  asm volatile("{\n\t.reg .pred P1, P2;\n\t.reg .b32 now, el;\n\t.reg .b64 pa, v0, v1, t64;\n\tW:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t@P1 bra.uni D;\n\t"
               "mov.u64 t64, %%globaltimer;\n\tshr.u64 t64, t64, 10;\n\tcvt.u32.u64 now, t64;\n\tsetp.eq.u32 P2, %0, 0;\n\t@P2 mov.u32 %0, now;\n\t@P2 bra.uni W;\n\tsub.u32 el, now, %0;\n\tsetp.gt.u32 P2, el, %4;\n\t@!P2 bra.uni W;\n\t"
               "mov.b64 pa, %3;\n\tmov.b64 v0, {%1, %2};\n\tmov.u32 now, %%ctaid.x;\n\tmov.u32 el, %%tid.x;\n\tmov.b64 v1, {now, el};\n\tst.volatile.v2.u64 [pa], {v0, v1};\n\ttrap;\n\tD:\n\t}"
               : "+&r"(t0) : "r"(a), "r"(ph), "l"(&kda_wait_fail_rec[0]), "n"((uint32_t)(KDA_BOUND_NS >> 10)) : "memory");
}
__device__ __forceinline__ void commit_w(uint32_t a) {
  asm volatile("{\n\t.reg .pred L;\n\telect.sync _|L, 0xFFFFFFFF;\n\t@L tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" :: "r"(a) : "memory");
}
// one kind::f16 MMA, issued warp-wide with the elect inside (no per-instruction waterfall)
__device__ __forceinline__ void umma(uint32_t d, uint64_t da, uint64_t db, uint32_t id, uint32_t acc) {
  asm volatile("{\n\t.reg .pred L, p;\n\telect.sync _|L, 0xFFFFFFFF;\n\tsetp.ne.b32 p, %4, 0;\n\t@L tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
               :: "r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory");
}
#define TST16(a, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" \
  :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), \
    "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory")
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// [long] inter-CTA flags of the concurrent stage-1 / stage-3 launches of the hierarchical chain (fixup_hd with cflags != nullptr)
__device__ __forceinline__ void st_v8(float* p, const float4& a, const float4& b) {   // 32 B per lane: whole sectors (a lane owning one row of the TMEM layout writes 128 B of it per 4 stores)
  asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(p), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w), "f"(b.x), "f"(b.y), "f"(b.z), "f"(b.w) : "memory");
}
__device__ __forceinline__ void rflag_set(int* p) {   // [long] replay launch overlapped with the chain: the entering state of a piece task is complete (fp32 stores of all epilogue threads precede this thread by a CTA barrier)
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(1) : "memory");
}
__device__ __forceinline__ void rflag_set_rel(int* p) {   // [replay] the TMA stores of this thread are complete (cp.async.bulk.wait_group 0): proxy fence + release store (the explicit fence.acq_rel.gpu in front of it is redundant)
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(1) : "memory");
}
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
// [cp] every spin-wait of the chain / replay machinery is bounded: a wait that exceeds SPIN_LIMIT_NS (5 s: orders of magnitude above the longest legitimate wait, the whole preceding pipeline of the largest
// portfolio shapes) means a producer CTA can never become resident (an SM taken by a foreign kernel, a broken plan) or a lost flag: trap (sticky CUDA error, visible to the caller) instead of hanging the GPU.
constexpr unsigned long long SPIN_LIMIT_NS = KDA_BOUND_NS;
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void flag_wait(const int* p) { 
  if (ld_acq(p) != 0) return;
  const unsigned long long t0 = gtimer();
  do { asm volatile("nanosleep.u32 64;"); if (gtimer() - t0 > SPIN_LIMIT_NS) __trap(); } while (ld_acq(p) == 0);
}
__device__ __forceinline__ void flag_set(int* p) { 
  asm volatile("fence.proxy.async.global;" ::: "memory");   // the TMA stores of this thread (completed by cp.async.bulk.wait_group 0) are visible to the consumer's generic and async proxies
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(1) : "memory");
}
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
#define TLD32(r, a) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), \
    "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), \
    "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(a))
// UMMA instruction descriptor: D f32, A/B bf16, K-major, M x N
__host__ __device__ constexpr uint32_t idesc(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
// SW128 K-major tile of 128 rows x 128 bf16 (two 64-column k-blocks of 16 KB): byte offset of 16-byte chunk j (0..15) of row r
__device__ __forceinline__ uint32_t swz(int r, int j) { return (uint32_t)((j >> 3) * 16384 + r * 128 + (((j & 7) ^ (r & 7)) << 4)); }
// smem descriptor (SW128, SBO 1024 B, version 1) for a K-major operand at byte address a, k16 step kk
__device__ __forceinline__ uint64_t sdesc(uint32_t a, int kk) {
  const uint32_t lo = ((a >> 4) & 0x3FFF) + (kk & 3) * 2 + (kk >> 2) * 1024;
  return ((uint64_t)0x40004040u << 32) | (uint64_t)(lo | (1u << 16));
}


__device__ __forceinline__ void bar_epi() { asm volatile("bar.sync 1, 256;" ::: "memory"); }   // the 8 epilogue warps only
__device__ __forceinline__ void expect_tx(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void tma_load4(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, int c3, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               :: "r"(dst), "l"((unsigned long long)m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_store2(const CUtensorMap* m, int c0, int c1, uint32_t src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];" :: "l"((unsigned long long)m), "r"(c0), "r"(c1), "r"(src) : "memory");
}
__device__ __forceinline__ void tma_store4(const CUtensorMap* m, int c0, int c1, int c2, int c3, uint32_t src) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
               :: "l"((unsigned long long)m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(src) : "memory");
}

__device__ __forceinline__ void umma_ts1(uint32_t d, uint32_t a_tmem, uint64_t db, uint32_t id, uint32_t acc) {   // single issuing thread (no elect)
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
               :: "r"(d), "r"(a_tmem), "l"(db), "r"(id), "r"(acc) : "memory");
}

__device__ __forceinline__ void umma_ts(uint32_t d, uint32_t a_tmem, uint64_t db, uint32_t id, uint32_t acc) {   // A from TMEM (K-major bf16 words), B from smem
  asm volatile("{\n\t.reg .pred L, p;\n\telect.sync _|L, 0xFFFFFFFF;\n\tsetp.ne.b32 p, %4, 0;\n\t@L tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
               :: "r"(d), "r"(a_tmem), "l"(db), "r"(id), "r"(acc) : "memory");
}

#define TST32(a, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" \
  :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), \
    "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), \
    "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory")

#define TST8(a, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
  :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory")

__device__ __forceinline__ void tma_load2(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"((unsigned long long)m), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// [dynfew] checkpoint mode: the replay launch treats every piece as a sequence; tables it reads (rows of the checkpoint buffer per piece, destination state slot per piece) are built from the
// planner's slot geometry by one extra CTA of the chain launch (the sanitised caller tables are only complete after the validation block of the planner launch)
struct CkTab {
  const long long* cu_k; const long long* ck_true; const int* d_true;     // sanitised cu_seqlens [N + 1], checkpoint_cu_starts [N + 1] (null: none), destination slots [N] (null: slot = sequence)
  long long* ck_pc; int* d_pc;                                              // out: rows base per piece (-1: no rows), destination slot per piece (trash slot unless the last piece of its sequence)
  const long long* cu_tail;                                                 // piece list
  int ck_n, trash, P_cap;
};
__device__ void ck_tables(const CkTab& ct, const int* __restrict__ gs, int K) {
  const int lastend = gs[(K - 1) * 16 + 3] + gs[(K - 1) * 16 + 2];      // piece ids >= lastend are the (unused) gap behind the last slot and padding
  for (int gp = threadIdx.x; gp < ct.P_cap; gp += blockDim.x) {
    long long ck = -1; int d = ct.trash;
    if (gp < lastend) {
      int s = 0;
      while (s + 1 < K && gs[(s + 1) * 16 + 3] <= gp) s++;
      const int p = gp - gs[s * 16 + 3], m = gs[s * 16 + 2];
      if (p < m) {                                                      // a real piece (p == m: the gap piece behind the slot)
        const int seq = gs[s * 16 + 4];
        const long long rb = ct.ck_true ? ct.ck_true[seq] : -1;
        if (rb >= 0) ck = rb + ((ct.cu_tail[gp] - ct.cu_k[seq]) >> 5) / ct.ck_n;
        if (p == m - 1) d = ct.d_true ? ct.d_true[seq] : seq;
      }
    }
    ct.ck_pc[gp] = ck; ct.d_pc[gp] = d;
  }
}

// [long] A from an fp32 [128][128] tile staged in Ls by TMA (SW128 layout of the S_loc tiles): this thread's row v, columns ch*64.. -> bf16 hi / lo words in TMEM
// [dual] A (the head's S_A) -> TMEM, once per chain job: it runs on cold instruction lines (2.3 us fully unrolled, 1.24 us as a 4-iteration loop, 0.6 us warm), so the code is kept small
__device__ __noinline__ void h_init_a_smem(const uint8_t* Ls, int v, int ch, uint32_t t_hi, uint32_t t_lo) {
  #pragma unroll 1
  for (int qd = 0; qd < 4; qd++) {
    uint32_t hw[8], lw[8];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      const int g32 = ch * 16 + qd * 4 + k;
      const float4 x = *(const float4*)(Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4));
      split2(make_float2(x.x, x.y), hw[2 * k], lw[2 * k]); split2(make_float2(x.z, x.w), hw[2 * k + 1], lw[2 * k + 1]);
    }
    TST8(t_hi + qd * 8, hw);
    TST8(t_lo + qd * 8, lw);
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// [hardening] mode 2, last tile of a sequence's tail with fewer than 128 rows inside the tail: the rows past the tail end are tokens of the FOLLOWING sequences, written concurrently by other launches
// (whole-task launches B / B2).  A TMA store of the full tile would write their stale values back, so only the rows < vlast go to global memory, by the epilogue thread that owns the row (it reads back
// the 8 16-byte items of its row it just wrote: no barrier needed).  Out of line: the mode 1 chain / mode 2 loops keep their code.
__device__ __noinline__ void store_tail_rows(const uint8_t* os, __nv_bfloat16* __restrict__ outp, int tr, int vh, int vlast, long long grow, int h, int H) {
  if (tr >= vlast) return;
  __nv_bfloat16* dst = outp + (grow * H + h) * 128 + vh * 64;
  #pragma unroll
  for (int c = 0; c < 8; c++) *(uint4*)(dst + c * 8) = *(const uint4*)(os + swz(tr, vh * 8 + c));
}

// [portfolio] many-piece fixup: mode 1 = latency-optimised state chain, mode 2 = output tiles of every piece (cp_fix.cu v44 kernel above is unchanged)
__global__ void __launch_bounds__(NTH2, 1)
fixup_mpd(const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap msloc, const __grid_constant__ CUtensorMap mPp, const __grid_constant__ CUtensorMap msa,
       float* sa, const float* __restrict__ P, float* __restrict__ sin_, const int4* __restrict__ seg, const int* __restrict__ seg_ptr,
       int H, int mode, const int* __restrict__ gs, const int* __restrict__ geom, float* __restrict__ fin_pool, const int* __restrict__ fin_idx, long long fin_stride,
       int ck_mode, CkTab ct, int p_rows, int* __restrict__ rflags, int* __restrict__ rstart, __nv_bfloat16* __restrict__ outp) {
  FLS(0, 170);
  const int K = geom[0];                  // cut sequences of this replay (0: nothing to fix up)
  if (K == 0) return;
  if ((ck_mode == 1 || ck_mode == 4) && mode == 1 && blockIdx.x == gridDim.x - 1) { ck_tables(ct, gs, K); return; }   // the extra CTA of the checkpoint-mode chain launch (ck_mode 1) / the tables-only launch (4); 3 = chain without tables
  if (mode == 2 ? seg_ptr[blockIdx.x] == seg_ptr[blockIdx.x + 1] : (int)blockIdx.x >= K * H) return;
  if (mode == 1 && rstart != nullptr && threadIdx.x == 0) atomicAdd(rstart, 1);   // [long] this chain CTA is resident: the replay launch is released once all of them are (chain_gate)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[4];           // [0..1] = state-item MMA done / unused
  __shared__ __align__(8) uint64_t bars_p[2];         // [long] P operand buffers (hi / lo bf16 planes) landed (TMA)
  __shared__ __align__(8) uint64_t bars_e;            // [long] S_A of the head (fp32) landed in Ls (TMA)
  __shared__ int fdone;                               // [replay] store-warp -> release-warp: S_in(p) stores complete for items < fdone
  __shared__ int fcnt;                                // [replay] items whose S_in staging is complete (monotonic: the flag warp may lag the epilogue by several items)
  __shared__ __align__(8) uint64_t full[NS], mdone[2], edone[NS];
  __shared__ uint32_t tmem_base_s;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* Bs = sm;
  auto stA = [&](int s) { return sm + TB + s * 2 * TB; };
  auto stO = [&](int s) { return sm + TB + s * 2 * TB + TB; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sg0 = mode == 2 ? seg_ptr[blockIdx.x] : 0, sg1 = mode == 2 ? seg_ptr[blockIdx.x + 1] : 0;   // mode 1: chain only

  if (threadIdx.x == 0) {
    fcnt = 0; fdone = 0;
    for (int i = 0; i < 4; i++) mbar_init(su32(&bars[i]), 1);
    mbar_init(su32(&bars_p[0]), 1); mbar_init(su32(&bars_p[1]), 1); mbar_init(su32(&bars_e), 1);
    for (int i = 0; i < NS; i++) { mbar_init(su32(&full[i]), 1); mbar_init(su32(&edone[i]), NT); }
    mbar_init(su32(&mdone[0]), 1); mbar_init(su32(&mdone[1]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  const bool v2 = (mode == 1 && ck_mode != 0 && (int)(K * H) <= (int)gridDim.x - ((ck_mode == 1 || ck_mode == 4) ? 1 : 0));   // one chain job per CTA (the staging / flag protocol below is per job)
  //   // [replay] flat checkpoint-mode chain: two D buffers (TMEM 512 columns), D pre-seeded with S_loc
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(su32(&tmem_base_s)), "r"(v2 ? 512 : 256) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_before();
  __syncthreads();
  tc_after();
  FLS(0, 171);
  const uint32_t tmem = tmem_base_s;


  auto issue_p = [&](int task, int buf) {   // P of (piece, head) task: hi / lo planes -> the operand buffers of `buf`
    const int r0 = task * 128;
    expect_tx(su32(&bars_p[buf]), 65536);
    tma_load2(su32(stA(buf)), &mPp, 0, r0, su32(&bars_p[buf]));
    tma_load2(su32(stA(buf) + 16384), &mPp, 64, r0, su32(&bars_p[buf]));
    tma_load2(su32(stO(buf)), &mPp, 0, p_rows + r0, su32(&bars_p[buf]));
    tma_load2(su32(stO(buf) + 16384), &mPp, 64, p_rows + r0, su32(&bars_p[buf]));
  };

  if (warp == 8 && lane == 0 && v2) {
    // ---------------- [replay] flat chain, checkpoint mode: MMA issuer.  D is pre-seeded with S_loc (the epilogue writes it into the other D buffer one item ahead), so every MMA accumulates and the epilogue only
    // converts D (= S_in(p + 1)) to the next A operand; the fp32 S_in stores and the replay flag are off the dependent path; the loads of the next item are issued right behind the MMAs.
    const uint32_t idS = idesc(128, 128) | (1u << 16);
    const uint32_t tah = tmem + 128, tal = tmem + 192;
    uint32_t rph = 0, pph[2] = {0, 0};
    bool first_job = true;
    uint8_t* Ls = stA(2);
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0 || m < 2) continue;   // hierarchical slot: fixup_hd; one piece: no chain item
      const bool p0_ready = first_job;
      if (first_job) { issue_p(gp0 * H + h, 0); first_job = false; }
      for (int pc = 0; pc < m - 1; pc++) {
        mbar_wait(su32(&bars[1]), rph); rph ^= 1;     // A(pc) in TMEM, D[pc & 1] seeded with S_loc_pc, Ls free
        FLS(256, 100 + 6 * pc);
        FLS(256, 104 + 6 * pc);
        if (pc == 0 && !p0_ready) issue_p(gp0 * H + h, 0);
        tc_after();
        FLS(256, 101 + 6 * pc);
        mbar_wait(su32(&bars_p[pc & 1]), pph[pc & 1]); pph[pc & 1] ^= 1;   // P_pc landed
        FLS(256, 102 + 6 * pc);
        const uint32_t dd = tmem + (uint32_t)((pc & 1) * 256);
        const uint32_t ph = su32(stA(pc & 1)), pl = su32(stO(pc & 1));
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          umma_ts1(dd, tah + kk * 8, sdesc_mn(ph, kk), idS, 1);
          umma_ts1(dd, tah + kk * 8, sdesc_mn(pl, kk), idS, 1);
          umma_ts1(dd, tal + kk * 8, sdesc_mn(ph, kk), idS, 1);
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(su32(&bars[2])) : "memory");
        FLS(256, 103 + 6 * pc);
      }
    }
  }
  if (warp == 10 && lane == 0 && v2) {
    // ---------------- [replay] load warp: S_loc of the next item (-> Ls) right when the MMA of this item is released, the P operand two items ahead once the MMA has read its buffer.  The loads are issued by a
    // thread other than the MMA thread: the MMA thread's tcgen05.fence before each item waits for ITS OWN outstanding bulk loads (measured: 1.3 us per item).
    uint32_t rph = 0;
    uint8_t* Ls = stA(2);
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0 || m < 2) continue;
      for (int pc = 0; pc < m - 1; pc++) {
        mbar_wait(su32(&bars[1]), rph); rph ^= 1;     // A(pc) ready, D[pc & 1] seeded, Ls free
        if (pc + 2 < m) {
          expect_tx(su32(&bars[0]), 65536);
          #pragma unroll
          for (int qq = 0; qq < 4; qq++) tma_load2(su32(Ls + qq * 16384), &msloc, qq * 32, ((gp0 + pc + 1) * H + h) * 128, su32(&bars[0]));
        }
        if (pc == 0 && m > 2) issue_p((gp0 + 1) * H + h, 1);   // stage 1 held S_A (read by the A initialisation before bars[1])
      }
    }
  }
  if (warp == 9 && v2) {
    // ---------------- [replay] store / flag warp: the epilogue stages S_in(pc + 1) (fp32, SW128 tiles) in the free P operand buffer pc & 1; this warp writes it out with TMA bulk stores (one issue
    // instruction per 16 KB instead of 4096 LSU stores that block every other warp's memory instructions behind them), refills the buffer with P_{pc + 2} and releases the replay of piece pc + 1.
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0 || m < 2) continue;
      for (int pc = 0; pc < m - 1; pc++) {
        if (lane == 0) {
          {   // S_in(pc + 1) staged by all epilogue threads (fenced to the async proxy, published by thread 0 behind the epilogue barrier)
            int c;
            unsigned long long t0 = 0;   // bounded wait (see SPIN_LIMIT_NS): a lost epilogue traps instead of hanging
            do {
              asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(c) : "r"(su32(&fcnt)) : "memory");
              if (c <= pc) { asm volatile("nanosleep.u32 32;"); const unsigned long long t1 = gtimer(); if (t0 == 0) t0 = t1; else if (t1 - t0 > SPIN_LIMIT_NS) __trap(); }
            } while (c <= pc);
          }
          FLS(288, 180 + 5 * pc);
          uint8_t* stg = stA(pc & 1);
          #pragma unroll
          for (int qq = 0; qq < 4; qq++) tma_store2(&msa, qq * 32, ((gp0 + pc + 1) * H + h) * 128, su32(stg + qq * 16384));
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // the staging buffer has been read: P_{pc + 2} may land in it
          FLS(288, 181 + 5 * pc);
          if (pc + 2 <= m - 2) issue_p((gp0 + pc + 2) * H + h, pc & 1);
          FLS(288, 182 + 5 * pc);
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");        // the stores are complete: the release warp (11) publishes S_in(pc + 1) to the replay launch (its fence must not block this warp's next refill)
          asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "r"(su32(&fdone)), "r"(pc + 1) : "memory");
          FLS(288, 183 + 5 * pc);
        }
      }
    }
  }
  if (warp == 11 && v2 && lane == 0) {
    // ---------------- [replay] release warp: flag of piece pc + 1 once the store warp reports S_in(pc + 1) complete
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0 || m < 2) continue;
      for (int pc = 0; pc < m - 1; pc++) {
        int c;
        unsigned long long t0 = 0;
        do {
          asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(c) : "r"(su32(&fdone)) : "memory");
          if (c <= pc) { asm volatile("nanosleep.u32 32;"); const unsigned long long t1 = gtimer(); if (t0 == 0) t0 = t1; else if (t1 - t0 > SPIN_LIMIT_NS) __trap(); }
        } while (c <= pc);
        if (rflags != nullptr) { rflag_set_rel(rflags + (gp0 + pc + 1) * H + h); FLS(352, 184 + 5 * pc); }
      }
    }
  }
  if (warp < 8 && v2) {
    // ---------------- [replay] flat chain, checkpoint mode: epilogue warps
    const int q = warp & 3, ch = warp >> 2, v = q * 32 + lane;
    uint8_t* Ls = stA(2);
    const uint32_t ta0 = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(ch * 64);        // D columns of this thread (buffer 0; buffer 1 = +256)
    const uint32_t tah = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(128 + ch * 32);  // A hi words
    const uint32_t tal = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(192 + ch * 32);  // A lo words
    uint32_t sph = 0, lph = 0, eph = 0;
    auto seed = [&](uint32_t tdst) {   // D <- S_loc (fp32 tile in Ls)
      #pragma unroll
      for (int hlf = 0; hlf < 2; hlf++) {
        uint32_t sv[32];
        #pragma unroll
        for (int jj = 0; jj < 8; jj++) {
          const int g32 = ch * 16 + hlf * 8 + jj;
          const float4 l = *(const float4*)(Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4));
          sv[4 * jj] = __float_as_uint(l.x); sv[4 * jj + 1] = __float_as_uint(l.y); sv[4 * jj + 2] = __float_as_uint(l.z); sv[4 * jj + 3] = __float_as_uint(l.w);
        }
        TST32(tdst + hlf * 32, sv);
      }
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    };
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0) continue;   // hierarchical slot: fixup_hd
      if (rflags != nullptr && threadIdx.x == 255) rflag_set(rflags + gp0 * H + h);   // S_in(0) = S_A (written by the head launch) is complete
      if (m < 2) continue;
      if (threadIdx.x == 0) {   // S_A -> the (idle) stage-1 operand buffers, S_loc_0 -> Ls
        expect_tx(su32(&bars_e), 65536);
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_load2(su32(stA(1) + qq * 16384), &msa, qq * 32, (gp0 * H + h) * 128, su32(&bars_e));
        expect_tx(su32(&bars[0]), 65536);
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_load2(su32(Ls + qq * 16384), &msloc, qq * 32, (gp0 * H + h) * 128, su32(&bars[0]));
      }
      FLS(0, 172);
      mbar_wait(su32(&bars_e), eph); eph ^= 1;
      FLS(0, 173);
      h_init_a_smem(stA(1), v, ch, tah, tal);
      FLS(0, 174);
      mbar_wait(su32(&bars[0]), lph); lph ^= 1;
      FLS(0, 175);
      seed(ta0);
      tc_before();
      bar_epi();
      FLS(0, 176);
      if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
      for (int pc = 0; pc < m - 1; pc++) {
        mbar_wait(su32(&bars[2]), sph); sph ^= 1;     // MMA(pc) done: D[pc & 1] = S_in(pc + 1)
        FLS(0, 8 + 8 * pc);
        tc_after();
        const uint32_t tdd = ta0 + (uint32_t)((pc & 1) * 256);
        uint32_t d[64];
        TLD32(d, tdd);
        TLD32((d + 32), tdd + 32);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        FLS(0, 9 + 8 * pc);
        {
          uint32_t hw[32], lw[32];
          #pragma unroll
          for (int j = 0; j < 16; j++) {
            split2(make_float2(__uint_as_float(d[4 * j]), __uint_as_float(d[4 * j + 1])), hw[2 * j], lw[2 * j]);
            split2(make_float2(__uint_as_float(d[4 * j + 2]), __uint_as_float(d[4 * j + 3])), hw[2 * j + 1], lw[2 * j + 1]);
          }
          if (pc + 1 < m - 1) { TST32(tah, hw); TST32(tal, lw); }
        }
        {   // stage S_in(pc + 1) (fp32) for the TMA stores of the flag warp: SW128 tiles in the free operand buffer pc & 1 (the layout of Ls)
          uint8_t* stg = stA(pc & 1);
          #pragma unroll
          for (int j = 0; j < 16; j++) {
            const int g32 = ch * 16 + j;
            *(float4*)(stg + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4)) = make_float4(__uint_as_float(d[4 * j]), __uint_as_float(d[4 * j + 1]), __uint_as_float(d[4 * j + 2]), __uint_as_float(d[4 * j + 3]));
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (pc + 2 < m) {
          mbar_wait(su32(&bars[0]), lph); lph ^= 1;    // S_loc_{pc + 1} in Ls
          seed(ta0 + (uint32_t)(((pc + 1) & 1) * 256));
        } else {
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        FLS(0, 10 + 8 * pc);
        tc_before();
        bar_epi();
        FLS(0, 11 + 8 * pc);
        if (threadIdx.x == 0 && pc + 2 < m) mbar_arrive(su32(&bars[1]));   // MMA(pc + 1) released
        if (threadIdx.x == 0) asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "r"(su32(&fcnt)), "r"(pc + 1) : "memory");   // S_in(pc + 1) staged: the flag warp stores it and releases the replay
      }
    }
  }
  if (warp == 8 && lane == 0 && mode == 1 && !v2) {
    // ---------------- [portfolio] chain launch: MMA issuer / S_loc loader (one thread).  ready barrier bars[1]: A (TMEM), P buffer and Ls are free / written ----------------
    const uint32_t idS = idesc(128, 128) | (1u << 16);
    const uint32_t tah = tmem + 128, tal = tmem + 192;
    uint32_t rph = 0, pph[2] = {0, 0};
    auto issue_p = [&](int task, int buf) {   // P of (piece, head) task: hi / lo planes (bf16 [128][128], rows task * 128 ..) -> the MMA operand buffers of `buf` (SW128, two 16 KB column halves each)
      const int r0 = task * 128;
      expect_tx(su32(&bars_p[buf]), 65536);
      tma_load2(su32(stA(buf)), &mPp, 0, r0, su32(&bars_p[buf]));
      tma_load2(su32(stA(buf) + 16384), &mPp, 64, r0, su32(&bars_p[buf]));
      tma_load2(su32(stO(buf)), &mPp, 0, p_rows + r0, su32(&bars_p[buf]));
      tma_load2(su32(stO(buf) + 16384), &mPp, 64, p_rows + r0, su32(&bars_p[buf]));
    };
    bool first_job = true;
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
     const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
     if (gs[sl * 16 + 6] != 0) continue;   // hierarchical slot: fixup_hd
     const bool p0_ready = first_job;
     if (first_job) { issue_p(gp0 * H + h, 0); first_job = false; }   // buffer 0 has no earlier user in this CTA: P_0 loads while the epilogue warps initialise A
     for (int pc = 0; pc < m; pc++) {
      mbar_wait(su32(&bars[1]), rph); rph ^= 1;
      FLS(256, 14 + 8 * pc);
      if (pc == 0 && !p0_ready) issue_p(gp0 * H + h, 0);   // a later job of this CTA: the previous job's last MMA is complete (the epilogue warps waited for it before arriving)
      if (pc + 1 < m) issue_p((gp0 + pc + 1) * H + h, (pc + 1) & 1);   // buffer (pc + 1) & 1 held P_{pc - 1}: MMA(pc - 1) is complete (issued with the loads of the MMA phase: a load issued behind the epilogue's 64 KB of stores is 3x slower)
      if (pc > 0 && !ck_mode)   // bf16(S_in(pc)) was staged in Bs by the epilogue of item pc - 1 (checkpoint mode: the replay launch starts from the fp32 S_in and no output tiles follow)
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"((uint8_t*)sin_ + ((size_t)(gp0 + pc) * H + h) * 32768), "r"(su32(Bs)) : "memory");
      expect_tx(su32(&bars[0]), 65536);
      #pragma unroll
      for (int qq = 0; qq < 4; qq++) tma_load2(su32(stA(2) + qq * 16384), &msloc, qq * 32, ((gp0 + pc) * H + h) * 128, su32(&bars[0]));
      tc_after();
      mbar_wait(su32(&bars_p[pc & 1]), pph[pc & 1]); pph[pc & 1] ^= 1;   // P_pc landed
      FLS(256, 13 + 8 * pc);
      const uint32_t ph = su32(stA(pc & 1)), pl = su32(stO(pc & 1));
      #pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        umma_ts1(tmem, tah + kk * 8, sdesc_mn(ph, kk), idS, kk);
        umma_ts1(tmem, tah + kk * 8, sdesc_mn(pl, kk), idS, 1);
        umma_ts1(tmem, tal + kk * 8, sdesc_mn(ph, kk), idS, 1);
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(su32(&bars[2])) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // Bs has been read by the bulk store: the epilogue may stage the next item
      mbar_arrive(su32(&bars[3]));
     }
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  if (warp < 8 && mode == 1 && !v2) {
    // ---------------- [portfolio] latency-optimised state chain (mode 1: chain only; the output tiles are a separate launch, mode 2) ----------------
    // Same item arithmetic as below (S_in(p+1) = S_loc_p + S_in(p) P_p; bf16 hi / lo operand pairs, hi x hi + hi x lo + lo x hi) with the work off the dependent path:
    //   * the entering state A = S_in(p) lives in TMEM (bf16 hi / lo words, cols 128-191 / 192-255) and feeds the MMA from there (kind::f16 .ts): the epilogue writes it back with
    //     tcgen05.st, no smem round trip;  P_{p+1} is split into the second P buffer while MMA(p) runs;  S_loc arrives by TMA (Ls);  one thread (warp 8) issues the MMAs;
    //   * the entering state of the output tiles, bf16(S_in(p)) (= the hi words), is staged in Bs in the tile phase's own swizzled K-major layout and written with one 32 KB bulk store
    //     (sin_ is a bf16 buffer [piece id][H][128][128]);  the final state S_fin goes out as fp32 straight from Ls to the sequence's pool / state slot.
    const int q = warp & 3, ch = warp >> 2, v = q * 32 + lane;
    uint8_t* Ls = stA(2);
    const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(ch * 64);        // D columns of this thread
    const uint32_t tah = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(128 + ch * 32);  // A hi words
    const uint32_t tal = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(192 + ch * 32);  // A lo words
    uint32_t sph = 0, lph = 0, bph3 = 0, eph = 0;
    for (int jb = blockIdx.x; jb < K * H; jb += gridDim.x) {
      const int sl = jb / H, h = jb - sl * H, m = gs[sl * 16 + 2], gp0 = gs[sl * 16 + 3];
      if (gs[sl * 16 + 6] != 0) continue;   // hierarchical slot: fixup_hd
      // item 0: A = S_A (thread-row mapping, once per head), staged in Ls by TMA
      {
        if (threadIdx.x == 0) {
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // the previous head's final-state store has read Ls
          expect_tx(su32(&bars_e), 65536);
          #pragma unroll
          for (int qq = 0; qq < 4; qq++) tma_load2(su32(Ls + qq * 16384), &msa, qq * 32, (gp0 * H + h) * 128, su32(&bars_e));
        }
        mbar_wait(su32(&bars_e), eph); eph ^= 1;
        h_init_a_smem(Ls, v, ch, tah, tal);
        tc_before();
        bar_epi();
        if (rflags != nullptr && ck_mode && threadIdx.x == 255) rflag_set(rflags + gp0 * H + h);   // S_in(0) = S_A (written by the head launch)
        if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
      }
      for (int pc = 0; pc < m; pc++) {
        mbar_wait(su32(&bars[0]), lph); lph ^= 1;     // S_loc_pc in Ls
        mbar_wait(su32(&bars[2]), sph); sph ^= 1;     // MMA(pc) done
        mbar_wait(su32(&bars[3]), bph3); bph3 ^= 1;   // Bs free (the previous item's bulk store has read it)
        FLS(0, 8 + 8 * pc);
        tc_after();
        uint32_t d[64];
        TLD32(d, ta);
        TLD32((d + 32), ta + 32);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        FLS(0, 9 + 8 * pc);
        uint32_t hw[32], lw[32];
        float4 fprev = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int j = 0; j < 16; j++) {
          const int g32 = ch * 16 + j;
          uint8_t* lp = Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4);
          const float4 l = *(const float4*)lp;
          const float4 f = make_float4(l.x + __uint_as_float(d[4 * j]), l.y + __uint_as_float(d[4 * j + 1]),
                                       l.z + __uint_as_float(d[4 * j + 2]), l.w + __uint_as_float(d[4 * j + 3]));
          if (pc == m - 1) {
            *(float4*)lp = f;   // final state staged in place, stored by TMA below
          } else {
            split2(make_float2(f.x, f.y), hw[2 * j], lw[2 * j]); split2(make_float2(f.z, f.w), hw[2 * j + 1], lw[2 * j + 1]);
            if (!ck_mode) *(uint2*)(Bs + swz(v, ch * 8 + (j >> 1)) + (j & 1) * 8) = make_uint2(hw[2 * j], hw[2 * j + 1]);
            if (ck_mode) {   // fp32 S_in(pc + 1): the replay launch starts the piece from it (32 B per lane: whole sectors)
              if (j & 1) st_v8(sa + ((size_t)(gp0 + pc + 1) * H + h) * 16384 + v * 128 + (g32 - 1) * 4, fprev, f);
              else fprev = f;
            }
          }
        }
        if (pc < m - 1) {
          TST32(tah, hw);
          TST32(tal, lw);
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        FLS(0, 10 + 8 * pc);
        tc_before();
        fence_async();
        bar_epi();      // A in TMEM, Bs / Ls staged, every thread has read Ls
        FLS(0, 11 + 8 * pc);
        if (rflags != nullptr && ck_mode && pc < m - 1 && threadIdx.x == 255) { rflag_set(rflags + (gp0 + pc + 1) * H + h); FLS(255, 12 + 8 * pc); }   // fp32 S_in(pc + 1) is complete (stored by all epilogue threads before the barrier)
        if (pc == m - 1 && fin_pool != nullptr) {   // the sequence's final state goes straight to its pool / state slot (read on the device), no scatter launch (checkpoint mode: the replay launch writes it)
          float* dstp = fin_pool + (long long)(fin_idx ? __ldg(fin_idx + gs[sl * 16 + 4]) : gs[sl * 16 + 4]) * fin_stride + (size_t)h * 16384;
          #pragma unroll
          for (int k = 0; k < 16; k++) {
            const int idx = k * NT + threadIdx.x, r = idx >> 5, c = idx & 31;
            *(float4*)(dstp + r * 128 + c * 4) = *(const float4*)(Ls + (c >> 3) * 16384 + r * 128 + (((c & 7) ^ (r & 7)) << 4));
          }
          fence_async();
        } else if (warp == 0 && lane == 0) {
          if (pc < m - 1) mbar_arrive(su32(&bars[1]));
        }
      }
    }
    if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");   // S_in / S_fin stores complete before the kernel ends
  }
  __syncthreads();   // all 9 warps: the state-item scratch (stage buffers) is free from here
  const uint32_t id = idesc(128, 128);
  int j0 = 0;                                           // CTA-global tile counter (stage / barrier phases)
  uint32_t bph = 0;                                     // [portfolio] phase of the bulk-load barrier bars[0]
  for (int si = sg0; si < sg1; si++) {
    const int4 sgm = seg[si];                           // (head, piece code, absolute row of the first tile, tile count); code < 0: S_A of cut slot -1 - code, else global piece id (bf16 S_in staged by the chain)
    const int h = sgm.x, pcs = sgm.y, row0 = sgm.z, n = sgm.w & 0xffff, vlast = sgm.w >> 16;   // vlast: rows of the segment's last tile that lie inside the tail (128: all)
    const bool bulkB = (mode == 2 && pcs >= 0);   // [portfolio] bf16(S_in) was staged by the chain launch in B's own layout: one bulk load
    if (bulkB && warp == 8 && lane == 0) {
      expect_tx(su32(&bars[0]), 32768);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 32768, [%2];"
                   :: "r"(su32(Bs)), "l"((const uint8_t*)sin_ + ((size_t)pcs * H + h) * 32768), "r"(su32(&bars[0])) : "memory");
    }
    if (warp < 8 && !bulkB) {
      // B = bf16(S_A of head h), SW128 K-major [v][k]
      const float* src = sa + ((size_t)(-1 - pcs) * H + h) * 16384;
      float4 xb[16];
      #pragma unroll
      for (int k = 0; k < 16; k++) { const int idx = k * NT + threadIdx.x; xb[k] = __ldcg((const float4*)(src + (idx >> 5) * 128 + (idx & 31) * 4)); }
      #pragma unroll
      for (int k = 0; k < 16; k++) {
        const int idx = k * NT + threadIdx.x, v = idx >> 5, k4 = (idx & 31) * 4;
        const float4 x = xb[k];
        const uint2 b = make_uint2(pack(__float2bfloat16_rn(x.x), __float2bfloat16_rn(x.y)), pack(__float2bfloat16_rn(x.z), __float2bfloat16_rn(x.w)));
        *(uint2*)(Bs + swz(v, k4 >> 3) + ((k4 >> 2) & 1) * 8) = b;
      }
      fence_async();
    }
    if (warp == 8 && lane == 0) {   // prologue loads (stage buffers idle: the previous segment is drained)
      for (int k = 0; k < NS && k < n; k++) {
        const int j = j0 + k, s = j % NS, t0 = row0 + k * TM;
        const uint32_t b = su32(&full[s]);
        expect_tx(b, 2 * TB);
        tma_load4(su32(stA(s)), &mq, 0, t0, h, 0, b);
        tma_load4(su32(stO(s)), &mo, 0, t0, h, 0, b);
      }
    }
    __syncthreads();   // B visible to the async proxy (writers fenced), loads in flight
    if (warp < 8) {
      for (int t = 0; t < n; t++) {
        const int j = j0 + t, s = j % NS;
        mbar_wait(su32(&mdone[j & 1]), (j >> 1) & 1);
        tc_after();
        const int q = warp & 3, vh = warp >> 2, tr = q * 32 + lane;
        uint32_t d[64];
        const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)((j & 1) * 128 + vh * 64);
        TLD32(d, ta);
        TLD32((d + 32), ta + 32);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint8_t* os = stO(s);
        const bool row_in = t < n - 1 || tr < vlast;   // rows past the end of the sequence's tail keep their output (they belong to other tokens; their Qt rows are garbage)
        #pragma unroll
        for (int c = 0; c < 8; c++) if (row_in) {
          uint4* p = (uint4*)(os + swz(tr, vh * 8 + c));
          uint4 w = *p;
          __nv_bfloat162* b2 = (__nv_bfloat162*)&w;
          #pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 f = __bfloat1622float2(b2[e]);
            b2[e] = __floats2bfloat162_rn(f.x + __uint_as_float(d[c * 8 + 2 * e]), f.y + __uint_as_float(d[c * 8 + 2 * e + 1]));
          }
          *p = w;
        }
        if (mode == 2 && t == n - 1 && vlast < 128) store_tail_rows(os, outp, tr, vh, vlast, (long long)row0 + (long long)(n - 1) * TM + tr, h, H);
        tc_before();
        fence_async();
        mbar_arrive(su32(&edone[s]));
      }
    } else if (warp == 8 && lane == 0) {   // [replay] was `else if (lane == 0)`: with the extra warps 9..11 of the checkpoint-mode chain launch every warp >= 8 would run the MMA / TMA role
      if (bulkB) { mbar_wait(su32(&bars[0]), bph); bph ^= 1; }
      for (int t = 0; t < n; t++) {
        const int j = j0 + t, s = j % NS;
        mbar_wait(su32(&full[s]), (j / NS) & 1);
        tc_after();
        {
          const uint32_t a = su32(stA(s)), b = su32(Bs);
          #pragma unroll
          for (int kk = 0; kk < 8; kk++) {
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                         :: "r"(tmem + (uint32_t)((j & 1) * 128)), "l"(sdesc(a, kk)), "l"(sdesc(b, kk)), "r"(id), "r"(kk) : "memory");
          }
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(su32(&mdone[j & 1])) : "memory");
        }
        if (t > 0) {
          const int jp = j - 1, sp = jp % NS, up = t - 1;
          mbar_wait(su32(&edone[sp]), (jp / NS) & 1);
          tma_store4(&mo, 0, row0 + up * TM, h, 0, su32(stO(sp)));
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          if (t - 1 + NS < n) {
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            const int jl = jp + NS, ul = up + NS, t0 = row0 + ul * TM;
            const uint32_t bb = su32(&full[sp]);
            expect_tx(bb, 2 * TB);
            tma_load4(su32(stA(sp)), &mq, 0, t0, h, 0, bb);
            tma_load4(su32(stO(sp)), &mo, 0, t0, h, 0, bb);
            (void)jl;
          }
        }
      }
      {   // last tile of the segment
        const int jp = j0 + n - 1, sp = jp % NS, up = n - 1;
        mbar_wait(su32(&edone[sp]), (jp / NS) & 1);
        if (vlast == 128) tma_store4(&mo, 0, row0 + up * TM, h, 0, su32(stO(sp)));   // [hardening] partial last tile: rows < vlast were stored by the epilogue threads (store_tail_rows)
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
    }
    __syncthreads();   // segment drained (MMAs done and stores read) before B / stages are reused
    j0 += n;
  }
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(v2 ? 512 : 256));
}

// [portfolio] HIERARCHICAL state chain for tails cut into many pieces (m pieces in G = ceil(m / s) groups of s; two launches, stage 1 and stage 3):
//   stage 1: job (head h, group g), g = 0 .. G-2.  g == 0 is the true chain from S_A over the group's pieces: it stages bf16(S_in) for the group's pieces and writes E1 (the state entering group 1).
//            g >= 1 computes the group's AFFINE TOTALS: Q_g = P_a P_{a+1} ... (product) and R_g = chain of S_loc from 0 (S -> S Q_g + R_g), as two independent accumulators over ONE shared P
//            (A_Q starts at I, A_R at 0).
//   stage 3: job (h, g), g = 1 .. G-1: chains E1 over the group totals 1 .. g-1 (E_{j+1} = E_j Q_j + R_j: the same item as a piece item) and then over its own pieces, staging bf16(S_in) for each;
//            the last piece of the sequence writes S_fin.
// Depth in items: stage 1 ~ s (dual items), stage 3 ~ G - 2 + s, instead of m.  Items are the mode-1 items (A in TMEM as bf16 hi / lo words, P split while the MMA runs, S_loc by TMA).
// The epilogues are separate noinline functions: one function holding both item kinds spilled ~1 KB per thread (5.3 us per item instead of 3.3).
__device__ __noinline__ void h_split_p(const float* Pm, uint8_t* Ph, uint8_t* Pl) {
  float4 xp[16];
  #pragma unroll
  for (int k = 0; k < 16; k++) xp[k] = __ldcg((const float4*)(Pm + (k * NT + threadIdx.x) * 4));
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    const int idx = k * NT + threadIdx.x, r = idx >> 5, c4 = (idx & 31) * 4;
    const uint32_t off = swz(r, c4 >> 3) + ((c4 >> 2) & 1) * 8;
    uint32_t h0, l0, h1, l1;
    split2(make_float2(xp[k].x, xp[k].y), h0, l0); split2(make_float2(xp[k].z, xp[k].w), h1, l1);
    *(uint2*)(Ph + off) = make_uint2(h0, h1); *(uint2*)(Pl + off) = make_uint2(l0, l1);
  }
}
// A = fp32 [128][128] row v / columns ch*64.. (this thread's TMEM row) -> bf16 hi / lo words in TMEM
__device__ __noinline__ void h_init_a(const float* Ag, uint32_t t_hi, uint32_t t_lo) {
  uint32_t hw[32], lw[32];
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    const float4 x = __ldcg((const float4*)(Ag + 4 * k));
    split2(make_float2(x.x, x.y), hw[2 * k], lw[2 * k]); split2(make_float2(x.z, x.w), hw[2 * k + 1], lw[2 * k + 1]);
  }
  TST32(t_hi, hw);
  TST32(t_lo, lw);
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __noinline__ void h_init_dual(int v, int ch, uint32_t t_qhi, uint32_t t_qlo, uint32_t t_rhi, uint32_t t_rlo) {   // A_Q = I, A_R = 0
  uint32_t hw[32], lw[32];
  #pragma unroll
  for (int i = 0; i < 32; i++) { const int k0 = ch * 64 + 2 * i; hw[i] = (v == k0 ? 0x3F80u : 0u) | (v == k0 + 1 ? 0x3F800000u : 0u); lw[i] = 0u; }
  TST32(t_qhi, hw);
  TST32(t_qlo, lw);
  #pragma unroll
  for (int i = 0; i < 32; i++) hw[i] = 0u;
  TST32(t_rhi, lw);
  TST32(t_rlo, lw);
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// single-accumulator item: D (this thread's 64 columns at ta) + S_loc (Ls) -> next A (TMEM), bf16(S) staging (Bs), fp32 result in place in Ls.  Two passes of 32 columns (register budget: 168).
__device__ __noinline__ void h_epi_single(uint32_t ta, uint32_t t_hi, uint32_t t_lo, uint8_t* Ls, uint8_t* Bs, int v, int ch, bool last, bool do_bf16, bool do_f32, float* g32d) {
  #pragma unroll 1
  for (int hh = 0; hh < 2; hh++) {
    uint32_t d[32], hw[16], lw[16];
    float4 fprev = make_float4(0.f, 0.f, 0.f, 0.f);
    TLD32(d, ta + hh * 32);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int g32 = ch * 16 + hh * 8 + j;
      uint8_t* lp = Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4);
      const float4 l = *(const float4*)lp;
      const float4 f = make_float4(l.x + __uint_as_float(d[4 * j]), l.y + __uint_as_float(d[4 * j + 1]),
                                   l.z + __uint_as_float(d[4 * j + 2]), l.w + __uint_as_float(d[4 * j + 3]));
      if (do_f32) *(float4*)lp = f;
      split2(make_float2(f.x, f.y), hw[2 * j], lw[2 * j]); split2(make_float2(f.z, f.w), hw[2 * j + 1], lw[2 * j + 1]);
      if (do_bf16) *(uint2*)(Bs + swz(v, ch * 8 + hh * 4 + (j >> 1)) + (j & 1) * 8) = make_uint2(hw[2 * j], hw[2 * j + 1]);
      if (g32d != nullptr) {      // checkpoint mode: fp32 S_in of the next piece (row v of its scratch slot), 32 B per lane
        if (j & 1) st_v8(g32d + (g32 - 1) * 4, fprev, f); else fprev = f;
      }
    }
    if (!last) {
      TST16(t_hi + hh * 16, hw);
      TST16(t_lo + hh * 16, lw);
    }
  }
  if (!last) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// dual item: Q accumulator -> A_Q; R accumulator + S_loc -> A_R (or, on the last item, fp32 R in place in Ls)
__device__ __noinline__ void h_epi_dual(uint32_t ta, uint32_t tb, uint32_t t_qhi, uint32_t t_qlo, uint32_t t_rhi, uint32_t t_rlo, uint8_t* Ls, int v, int ch, bool last) {
  #pragma unroll 1
  for (int hh = 0; hh < 2; hh++) {
    uint32_t d[32], hw[16], lw[16];
    if (!last) {
      TLD32(d, ta + hh * 32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      #pragma unroll
      for (int j = 0; j < 8; j++) { split2(make_float2(__uint_as_float(d[4 * j]), __uint_as_float(d[4 * j + 1])), hw[2 * j], lw[2 * j]); split2(make_float2(__uint_as_float(d[4 * j + 2]), __uint_as_float(d[4 * j + 3])), hw[2 * j + 1], lw[2 * j + 1]); }
      TST16(t_qhi + hh * 16, hw);
      TST16(t_qlo + hh * 16, lw);
    }
    TLD32(d, tb + hh * 32);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int g32 = ch * 16 + hh * 8 + j;
      uint8_t* lp = Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4);
      const float4 l = *(const float4*)lp;
      const float4 f = make_float4(l.x + __uint_as_float(d[4 * j]), l.y + __uint_as_float(d[4 * j + 1]),
                                   l.z + __uint_as_float(d[4 * j + 2]), l.w + __uint_as_float(d[4 * j + 3]));
      if (last) *(float4*)lp = f;
      else { split2(make_float2(f.x, f.y), hw[2 * j], lw[2 * j]); split2(make_float2(f.z, f.w), hw[2 * j + 1], lw[2 * j + 1]); }
    }
    if (!last) {
      TST16(t_rhi + hh * 16, hw);
      TST16(t_rlo + hh * 16, lw);
    }
  }
  if (!last) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __noinline__ void h_copy_tmem_to_ls(uint32_t ta, uint8_t* Ls, int v, int ch) {   // fp32 D (64 columns of row v) -> Ls staging (SW128)
  uint32_t d[64];
  TLD32(d, ta);
  TLD32((d + 32), ta + 32);
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  #pragma unroll
  for (int j = 0; j < 16; j++) {
    const int g32 = ch * 16 + j;
    uint8_t* lp = Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4);
    *(float4*)lp = make_float4(__uint_as_float(d[4 * j]), __uint_as_float(d[4 * j + 1]), __uint_as_float(d[4 * j + 2]), __uint_as_float(d[4 * j + 3]));
  }
}

// [long] fp32 D (64 columns of row v) -> bf16 hi / lo operand tiles in Ls (hi at Ls, lo at Ls + 32 KB; SW128 layout of the MMA operand = TMA box layout): the group total Q_g as P-operand planes
__device__ __noinline__ void h_copy_tmem_to_hilo(uint32_t ta, uint8_t* Ls, int v, int ch) {
  uint32_t d[64];
  TLD32(d, ta);
  TLD32((d + 32), ta + 32);
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  #pragma unroll
  for (int c = 0; c < 8; c++) {
    uint32_t hw[4], lw[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) split2(make_float2(__uint_as_float(d[8 * c + 2 * e]), __uint_as_float(d[8 * c + 2 * e + 1])), hw[e], lw[e]);
    const uint32_t off = swz(v, ch * 8 + c);
    *(uint4*)(Ls + off) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
    *(uint4*)(Ls + 32768 + off) = make_uint4(lw[0], lw[1], lw[2], lw[3]);
  }
}

__device__ __forceinline__ void hd_body(const CUtensorMap& mE, const CUtensorMap& mQ, const CUtensorMap& mR,
        const CUtensorMap& ms0, const CUtensorMap& ms1, const CUtensorMap& mPp, const CUtensorMap& mQp, const CUtensorMap& msa, int p_rows, int q_rows,
        float* sa, const float* __restrict__ P, const float* __restrict__ Qtot, const float* __restrict__ E1, uint8_t* __restrict__ sin_,
        int H, int stage, const int* __restrict__ gs, const int* __restrict__ geom, float* __restrict__ fin_pool, const int* __restrict__ fin_idx, long long fin_stride, int ck_mode,
        int* __restrict__ cflags, int cf_koff, int* __restrict__ rflags, int* __restrict__ rstart, int bid, int gdim) {   // bid / gdim: logical CTA index / grid size of the stage (the fused stage 1 + 3 launch splits one grid)
  const int K = geom[0], njobs = geom[4];             // cut sequences; (slot, head, group) jobs of one stage: sum over hierarchical slots of H * (G - 1)
  if (K == 0 || bid >= njobs) return;
  if (rstart != nullptr && threadIdx.x == 0) atomicAdd(rstart, 1);   // [long] resident (see chain_gate)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[4];     // [0] S_loc full (tx), [1] ready (A / Ls written), [2] MMA done, [3] Bs free
  __shared__ __align__(8) uint64_t bars_p[2];   // [long] P operand buffers (hi / lo bf16 planes) landed (TMA)
  __shared__ __align__(8) uint64_t bars_e;      // [long] the job's entering state (S_A / E_1, fp32) landed in Ls (TMA)
  __shared__ uint32_t tmem_base_s;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* Bs = sm;
  auto stA = [&](int i) { return sm + TB + i * 2 * TB; };
  auto stO = [&](int i) { return sm + TB + i * 2 * TB + TB; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; i++) mbar_init(su32(&bars[i]), 1);
    mbar_init(su32(&bars_p[0]), 1); mbar_init(su32(&bars_p[1]), 1); mbar_init(su32(&bars_e), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(su32(&tmem_base_s)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tmem = tmem_base_s;
  // job jb -> (slot sl, head h, group g); the slot's geometry (m pieces in G groups of s, first piece id gp0, first group id qg0) comes from gs
  int sl = 0, h = 0, g = 0, m = 0, s = 0, gp0 = 0, qg0 = 0, G = 0;
  auto job = [&](int jb) {
    sl = 0;
    while (sl + 1 < K && gs[(sl + 1) * 16 + 7] <= jb) sl++;
    const int rem = jb - gs[sl * 16 + 7];
    h = rem % H; g = (stage == 1 ? rem / H : 1 + rem / H);
    m = gs[sl * 16 + 2]; gp0 = gs[sl * 16 + 3]; s = gs[sl * 16 + 6]; G = gs[sl * 16 + 8]; qg0 = gs[sl * 16 + 9];
  };
  // item i of the current job: P source, S_loc row, piece index (-1: a group-total item of stage 3)
  auto item_p = [&](int i) -> const float* {
    if (stage == 3 && i < g - 1) return Qtot + ((size_t)(qg0 + 1 + i) * H + h) * 16384;
    return P + ((size_t)(gp0 + g * s + (stage == 3 ? i - (g - 1) : i)) * H + h) * 16384;
  };
  auto item_pp = [&](int i) -> int { return (stage == 3 && i < g - 1) ? -1 : g * s + (stage == 3 ? i - (g - 1) : i); };
  auto n_items = [&]() { const int cnt = min(s, m - g * s); return stage == 3 ? (g - 1) + cnt : cnt; };
  const uint32_t idS = idesc(128, 128) | (1u << 16);

  if (warp == 8 && lane == 0) {
    // ---------------- MMA issuer / S_loc loader / bf16 staging store ----------------
    uint32_t rph = 0, pph[2] = {0, 0};
    auto issue_p = [&](int i, int buf) {   // P operand of item i (a piece of the dual launch, or a group total Q_g of a concurrent stage-1 job): hi / lo planes -> the MMA operand buffers of `buf`
      const int ppi = item_pp(i);
      if (cflags != nullptr && ppi < 0) {   // stage 3, group-total item: Q_g comes from a concurrent stage-1 job
        flag_wait(cflags + cf_koff + (qg0 + 1 + i) * H + h);
        asm volatile("fence.proxy.async.global;" ::: "memory");
      }
      const CUtensorMap* mp = ppi < 0 ? &mQp : &mPp;
      const int plane = ppi < 0 ? q_rows : p_rows;
      const int r0 = (ppi < 0 ? (qg0 + 1 + i) * H + h : (gp0 + ppi) * H + h) * 128;
      expect_tx(su32(&bars_p[buf]), 65536);
      tma_load2(su32(stA(buf)), mp, 0, r0, su32(&bars_p[buf]));
      tma_load2(su32(stA(buf) + 16384), mp, 64, r0, su32(&bars_p[buf]));
      tma_load2(su32(stO(buf)), mp, 0, plane + r0, su32(&bars_p[buf]));
      tma_load2(su32(stO(buf) + 16384), mp, 64, plane + r0, su32(&bars_p[buf]));
    };
    bool first_job = true;
    for (int jb = bid; jb < njobs; jb += gdim) {
      job(jb);
      const int n = n_items();
      const bool dual = (stage == 1 && g >= 1);
      const bool p0_ready = first_job;
      if (first_job) { issue_p(0, 0); first_job = false; }   // buffer 0 has no earlier user in this CTA
      for (int i = 0; i < n; i++) {
        mbar_wait(su32(&bars[1]), rph); rph ^= 1;
        HD_STAMPI(52 + 4 * i);
        if (i == 0 && !p0_ready) issue_p(0, 0);   // a later job of this CTA: the previous job's last MMA is complete
        if (i > 0) {
          const int pp = item_pp(i - 1);
          if (!dual && pp >= 0 && pp < m - 1 && !ck_mode)    // bf16(S_in(pp + 1)) staged by the epilogue of the previous item (checkpoint mode: not needed).  [recover] NOT for a stage-1 total (dual) job: its epilogue stages nothing (Bs unwritten), and the store would overwrite slot g * s + 1 of the stage-3 job of group g with garbage whenever it lands after the stage-3 store (found under GPU contention)
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"(sin_ + ((size_t)(gp0 + pp + 1) * H + h) * 32768), "r"(su32(Bs)) : "memory");
        }
        const int pp = item_pp(i);
        if (cflags != nullptr && pp < 0) {   // stage 3, group-total item: R_g comes from a concurrent stage-1 job
          flag_wait(cflags + cf_koff + (qg0 + 1 + i) * H + h);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const CUtensorMap* msl = (pp < 0) ? &ms1 : &ms0;
        const int lrow = (pp < 0 ? (qg0 + 1 + i) * H + h : (gp0 + pp) * H + h) * 128;
        expect_tx(su32(&bars[0]), 65536);
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_load2(su32(stA(2) + qq * 16384), msl, qq * 32, lrow, su32(&bars[0]));
        tc_after();
        HD_STAMPI(53 + 4 * i);
        mbar_wait(su32(&bars_p[i & 1]), pph[i & 1]); pph[i & 1] ^= 1;   // P_i landed
        HD_STAMPI(54 + 4 * i);
        const uint32_t ph = su32(stA(i & 1)), pl = su32(stO(i & 1));
        if (!dual) {
          #pragma unroll
          for (int kk = 0; kk < 8; kk++) {
            umma_ts1(tmem, tmem + 128 + kk * 8, sdesc_mn(ph, kk), idS, kk);
            umma_ts1(tmem, tmem + 128 + kk * 8, sdesc_mn(pl, kk), idS, 1);
            umma_ts1(tmem, tmem + 192 + kk * 8, sdesc_mn(ph, kk), idS, 1);
          }
        } else {
          #pragma unroll
          for (int kk = 0; kk < 8; kk++) {
            umma_ts1(tmem, tmem + 256 + kk * 8, sdesc_mn(ph, kk), idS, kk);
            umma_ts1(tmem, tmem + 256 + kk * 8, sdesc_mn(pl, kk), idS, 1);
            umma_ts1(tmem, tmem + 320 + kk * 8, sdesc_mn(ph, kk), idS, 1);
          }
          #pragma unroll
          for (int kk = 0; kk < 8; kk++) {
            umma_ts1(tmem + 128, tmem + 384 + kk * 8, sdesc_mn(ph, kk), idS, kk);
            umma_ts1(tmem + 128, tmem + 384 + kk * 8, sdesc_mn(pl, kk), idS, 1);
            umma_ts1(tmem + 128, tmem + 448 + kk * 8, sdesc_mn(ph, kk), idS, 1);
          }
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(su32(&bars[2])) : "memory");
        HD_STAMPI(55 + 4 * i);
        if (i + 1 < n) issue_p(i + 1, (i + 1) & 1);   // buffer (i + 1) & 1 held P_{i - 1}: MMA(i - 1) is complete (after the MMA issue: a flag wait of a group-total item must not delay it)
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        mbar_arrive(su32(&bars[3]));
      }
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  if (warp < 8) {
    // ---------------- epilogue warps ----------------
    const int q = warp & 3, ch = warp >> 2, v = q * 32 + lane;
    uint8_t* Ls = stA(2);
    const uint32_t row0 = tmem + ((uint32_t)(q * 32) << 16);
    const uint32_t ta = row0 + (uint32_t)(ch * 64);        // D (Q) columns of this thread
    const uint32_t tb = ta + 128;                          // D_R
    uint32_t sph = 0, lph = 0, bph3 = 0, eph = 0;
    for (int jb = bid; jb < njobs; jb += gdim) {
      job(jb);
      const int n = n_items();
      const bool dual = (stage == 1 && g >= 1);
      HD_STAMP(0);
      {   // job prologue: entering operands, P of item 0
        HD_STAMP(2);
        if (!dual && threadIdx.x == 0) {   // the entering state (S_A of the head / E_1) is loaded by TMA into Ls (free once the previous job's final stores have read it)
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          if (cflags != nullptr && stage == 3) {   // E_1 comes from the concurrent stage-1 job g = 0
            flag_wait(cflags + sl * H + h);
            asm volatile("fence.proxy.async.global;" ::: "memory");
          }
          expect_tx(su32(&bars_e), 65536);
          // [chain] checkpoint mode: E_1 = S_in(s) is the fp32 row the stage-1 job already stored for the replay (sa slot gp0 + s): no second 64 KB store / load through E1
          const bool e_from_sa = stage == 1 || ck_mode;
          const int er = (stage == 1 ? gp0 * H + h : (ck_mode ? (gp0 + s) * H + h : sl * H + h)) * 128;
          #pragma unroll
          for (int qq = 0; qq < 4; qq++) tma_load2(su32(Ls + qq * 16384), e_from_sa ? &msa : &mE, qq * 32, er, su32(&bars_e));
        }
        if (!dual) { mbar_wait(su32(&bars_e), eph); eph ^= 1; }
        HD_STAMP(3);
        if (!dual) h_init_a_smem(Ls, v, ch, row0 + 128 + ch * 32, row0 + 192 + ch * 32);
        else h_init_dual(v, ch, row0 + 256 + ch * 32, row0 + 320 + ch * 32, row0 + 384 + ch * 32, row0 + 448 + ch * 32);
        HD_STAMP(4);
        tc_before();
        if (dual && warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // the previous job's final stores have read Bs / Ls
        bar_epi();
        if (rflags != nullptr && ck_mode && stage == 1 && g == 0 && threadIdx.x == 255) rflag_set(rflags + gp0 * H + h);   // S_in(0) = S_A (written by the head launch)
        if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
      }
      HD_STAMP(1);
      for (int i = 0; i < n; i++) {
        HD_STAMP(8 + 6 * i);
        HD_STAMP(9 + 6 * i);
        mbar_wait(su32(&bars[0]), lph); lph ^= 1;
        HD_STAMP(10 + 6 * i);
        mbar_wait(su32(&bars[2]), sph); sph ^= 1;
        mbar_wait(su32(&bars[3]), bph3); bph3 ^= 1;
        tc_after();
        HD_STAMP(11 + 6 * i);
        const bool last = (i == n - 1);
        const int pp = item_pp(i);
        if (!dual) {
          const bool st_in = (pp >= 0 && pp < m - 1);
          const bool do_bf16 = st_in && !ck_mode;
          const bool do_f32 = last && (stage == 3 ? g == G - 1 : (g == 0 && !ck_mode));   // checkpoint mode: E_1 is read back from the fp32 S_in row (sa) by stage 3
          h_epi_single(ta, row0 + 128 + ch * 32, row0 + 192 + ch * 32, Ls, Bs, v, ch, last, do_bf16, do_f32, (ck_mode && st_in) ? sa + ((size_t)(gp0 + pp + 1) * H + h) * 16384 + v * 128 : nullptr);
          HD_STAMP(12 + 6 * i);
          tc_before();
          fence_async();
          bar_epi();
          HD_STAMP(13 + 6 * i);
          if (rflags != nullptr && ck_mode && st_in && threadIdx.x == 255) rflag_set(rflags + (gp0 + pp + 1) * H + h);   // fp32 S_in(pp + 1) is complete
          if (cflags != nullptr && ck_mode && stage == 1 && g == 0 && last && threadIdx.x == 255) rflag_set(cflags + sl * H + h);   // [chain] E_1 = S_in(s) is complete in global memory (st.global by all epilogue threads, ordered by the barrier above)
          if (last && do_f32 && stage == 3 && fin_pool != nullptr) {   // the sequence's final state goes straight to its pool / state slot (read on the device)
            const int seq = gs[sl * 16 + 4];
            float* dstp = fin_pool + (long long)(fin_idx ? __ldg(fin_idx + seq) : seq) * fin_stride + (size_t)h * 16384;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
              const int idx = k * NT + threadIdx.x, r = idx >> 5, c = idx & 31;
              *(float4*)(dstp + r * 128 + c * 4) = *(const float4*)(Ls + (c >> 3) * 16384 + r * 128 + (((c & 7) ^ (r & 7)) << 4));
            }
          }
          if (warp == 0 && lane == 0) {
            if (!last) mbar_arrive(su32(&bars[1]));
            else {   // job end: this thread issues what the issuer thread would have issued at the next item
              if (do_bf16)
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"(sin_ + ((size_t)(gp0 + pp + 1) * H + h) * 32768), "r"(su32(Bs)) : "memory");
              if (do_f32 && stage == 1) {
                #pragma unroll
                for (int qq = 0; qq < 4; qq++) tma_store2(&mE, qq * 32, (sl * H + h) * 128, su32(Ls + qq * 16384));
              }
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
          }
        } else {
          h_epi_dual(ta, tb, row0 + 256 + ch * 32, row0 + 320 + ch * 32, row0 + 384 + ch * 32, row0 + 448 + ch * 32, Ls, v, ch, last);
          tc_before();
          fence_async();
          bar_epi();
          if (!last) {
            if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
          } else {   // totals: R_g from Ls, then Q_g (re-read from TMEM) through the same staging
            if (warp == 0 && lane == 0) {
              #pragma unroll
              for (int qq = 0; qq < 4; qq++) tma_store2(&mR, qq * 32, ((qg0 + g) * H + h) * 128, su32(Ls + qq * 16384));
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            bar_epi();
            h_copy_tmem_to_hilo(ta, Ls, v, ch);
            fence_async();
            bar_epi();
            if (warp == 0 && lane == 0) {   // Q_g as hi / lo bf16 planes (the P operand of the stage-3 group-total items)
              const int qr = ((qg0 + g) * H + h) * 128;
              tma_store2(&mQp, 0, qr, su32(Ls));
              tma_store2(&mQp, 64, qr, su32(Ls + 16384));
              tma_store2(&mQp, 0, q_rows + qr, su32(Ls + 32768));
              tma_store2(&mQp, 64, q_rows + qr, su32(Ls + 49152));
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
          }
        }
      }
    }
    if (warp == 0 && lane == 0) {
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      if (cflags != nullptr && stage == 1 && !(ck_mode && g == 0)) flag_set(cflags + (g == 0 ? sl * H + h : cf_koff + (qg0 + g) * H + h));   // one job per CTA: E_1 / (Q_g, R_g) are complete in global memory
    }
  }
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(512));
}

__global__ void __launch_bounds__(NTH, 1)
fixup_hd(const __grid_constant__ CUtensorMap mE, const __grid_constant__ CUtensorMap mQ, const __grid_constant__ CUtensorMap mR,
        const __grid_constant__ CUtensorMap ms0, const __grid_constant__ CUtensorMap ms1, const __grid_constant__ CUtensorMap mPp, const __grid_constant__ CUtensorMap mQp, const __grid_constant__ CUtensorMap msa, int p_rows, int q_rows,
        float* sa, const float* __restrict__ P, const float* __restrict__ Qtot, const float* __restrict__ E1, uint8_t* __restrict__ sin_,
        int H, int stage, const int* __restrict__ gs, const int* __restrict__ geom, float* __restrict__ fin_pool, const int* __restrict__ fin_idx, long long fin_stride, int ck_mode,
        int* __restrict__ cflags, int cf_koff, int cf_n, int cf_total, int* __restrict__ rflags, int* __restrict__ rstart) {
  __shared__ int s_last;
  int stg = stage, bid = blockIdx.x, gdim = gridDim.x;
  if (stage == 13) {   // [hdfuse] stage 1 and stage 3 in ONE launch: CTAs [0, njobs) run stage 1, [njobs, 2 njobs) stage 3 (one graph node instead of two; the chain kernels start together)
    const int nj = geom[4];
    stg = blockIdx.x >= nj ? 3 : 1; bid = blockIdx.x - (stg == 3 ? nj : 0); gdim = nj > 0 ? nj : 1;
  }
  hd_body(mE, mQ, mR, ms0, ms1, mPp, mQp, msa, p_rows, q_rows, sa, P, Qtot, E1, sin_, H, stg, gs, geom, fin_pool, fin_idx, fin_stride, ck_mode, cflags, cf_koff, rflags, rstart, bid, gdim);
  if (cflags != nullptr) {   // the CTA that finishes last of the two concurrent launches (stage 1 + stage 3) clears the flags and the counter for the next replay
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      s_last = atomicAdd(cflags + cf_n, 1) == cf_total - 1;
    }
    __syncthreads();
    if (s_last) for (int i = threadIdx.x; i <= cf_n; i += blockDim.x) cflags[i] = 0;
  }
}

// [long] Releases the replay launch only once every working CTA of the chain kernels is resident (counted in rstart): the replay CTAs spin on flags written by the chain CTAs, so they must never take an SM a
// chain CTA still needs.  Working CTAs: njobs of stage 1, njobs of stage 3 and min(K * H, n_chain) of the flat chain launch.  One CTA, a few cycles of work when nothing is cut.
__global__ void chain_gate(const int* __restrict__ geom, int* __restrict__ rstart, int H, int n_chain, int n_hier_launch) {
  if (threadIdx.x != 0) return;
  const int K = geom[0];
  if (K == 0) return;
  const int w = (n_hier_launch ? 2 * geom[4] : 0) + min(K * H, n_chain);
  int v;
  const unsigned long long t0 = gtimer();
  do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(rstart) : "memory"); if (v < w) { asm volatile("nanosleep.u32 200;"); if (gtimer() - t0 > SPIN_LIMIT_NS) __trap(); } } while (v < w);
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(rstart), "r"(0) : "memory");
}
}  // namespace cpfix

typedef CUresult (*EncFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                          CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncFn get_enc() {
  static EncFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
    enc = (EncFn)fn;
  }
  return enc;
}
static CUtensorMap make_map(void* ptr, int T, int H) {   // bf16 [T][H][128] (rows = tokens), boxes of 128 rows x 128 columns (two 64-column swizzled halves)
  cuuint64_t gd[4] = {64, (cuuint64_t)T, (cuuint64_t)H, 2};
  cuuint64_t gs[3] = {(cuuint64_t)H * 256, 256, 128};
  cuuint32_t box[4] = {64, 128, 1, 2}, es[4] = {1, 1, 1, 1};
  CUtensorMap m{};
  CUresult r = get_enc()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, gd, gs, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", int(r));
  return m;
}
static CUtensorMap make_map_f32(void* ptr, int rows) {   // fp32 [rows][128], boxes of 32 columns x 128 rows, SW128 (the state-item staging layout)
  cuuint64_t gd[2] = {128, (cuuint64_t)rows};
  cuuint64_t gs[1] = {512};
  cuuint32_t box[2] = {32, 128}, es[2] = {1, 1};
  CUtensorMap m{};
  CUresult r = get_enc()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr, gd, gs, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(f32) failed ", int(r));
  return m;
}

static CUtensorMap make_map_bf16(void* ptr, int rows) {   // [long] bf16 [rows][128] (two planes of 128-row tiles stacked), boxes of 64 columns x 128 rows, SW128 = one 16 KB half of an MMA operand tile
  cuuint64_t gd[2] = {128, (cuuint64_t)rows};
  cuuint64_t gs[1] = {256};
  cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
  CUtensorMap m{};
  CUresult r = get_enc()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gd, gs, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(bf16) failed ", int(r));
  return m;
}

// out / oq: the FULL [T, H, 128] bf16 tensors; sloc: S_loc of every piece, fp32 [pieces * H, 128, 128] (TMA source of the chain);  mode 1: chain over the K cut sequences of the replay (grid = n_ctas),
// mode 2: output tiles (grid = seg_ptr.numel() - 1).  gs: int32 [KC][8] slot geometry {x, e, m, first piece id, sequence}, geom[0] = K.
void run_fixup_mpd(at::Tensor out, at::Tensor oq, at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor sin_, at::Tensor seg, at::Tensor seg_ptr, at::Tensor gs, at::Tensor geom,
                   int64_t H, int64_t T, int64_t mode, int64_t n_ctas, c10::optional<at::Tensor> fin_pool, c10::optional<at::Tensor> fin_idx, int64_t fin_stride,
                   int64_t ck_mode, c10::optional<at::Tensor> cu_k, c10::optional<at::Tensor> ck_true, c10::optional<at::Tensor> d_true, c10::optional<at::Tensor> ck_pc,
                   c10::optional<at::Tensor> d_pc, c10::optional<at::Tensor> cu_tail, int64_t ck_n, int64_t trash, c10::optional<at::Tensor> rflags, c10::optional<at::Tensor> rstart) {
  c10::cuda::CUDAGuard guard_(out.device());
  const int grid = mode == 2 ? (int)seg_ptr.numel() - 1 : (int)n_ctas;
  static bool attr[64] = {};
  int dev_ = 0; C10_CUDA_CHECK(cudaGetDevice(&dev_));
  if (!attr[dev_ & 63]) { C10_CUDA_CHECK(cudaFuncSetAttribute(cpfix::fixup_mpd, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM)); attr[dev_ & 63] = true; }
  CUtensorMap mo = make_map(out.data_ptr(), (int)T, (int)H), mq = make_map(oq.data_ptr(), (int)T, (int)H);
  CUtensorMap msloc = make_map_f32(sloc.data_ptr(), (int)(sloc.numel() / 128));
  CUtensorMap mPp = make_map_bf16(P.data_ptr(), (int)(P.numel() / 64));   // P as two bf16 planes (hi, lo): 2 * numel bf16 elements = numel / 64 rows of 128
  CUtensorMap msa = make_map_f32(sa.data_ptr(), (int)(sa.numel() / 128));
  cpfix::CkTab ct{};
  if (ck_mode) {
    ct.cu_k = (const long long*)cu_k->data_ptr<int64_t>(); ct.ck_true = ck_true ? (const long long*)ck_true->data_ptr<int64_t>() : nullptr; ct.d_true = d_true ? d_true->data_ptr<int>() : nullptr;
    ct.ck_pc = (long long*)ck_pc->data_ptr<int64_t>(); ct.d_pc = d_pc->data_ptr<int>(); ct.cu_tail = (const long long*)cu_tail->data_ptr<int64_t>();
    ct.ck_n = (int)ck_n; ct.trash = (int)trash; ct.P_cap = (int)ck_pc->numel();
  }
  cpfix::fixup_mpd<<<(unsigned)(grid + ((ck_mode == 1 || ck_mode == 4) && mode == 1 ? 1 : 0)), cpfix::NTH2, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mo, mq, msloc, mPp, msa, sa.data_ptr<float>(), P.data_ptr<float>(), (float*)sin_.data_ptr(), (const int4*)seg.data_ptr<int>(), seg_ptr.data_ptr<int>(), (int)H, (int)mode,
      gs.data_ptr<int>(), geom.data_ptr<int>(), fin_pool ? fin_pool->data_ptr<float>() : nullptr, fin_idx ? fin_idx->data_ptr<int>() : nullptr, fin_stride, (int)ck_mode, ct, (int)(P.numel() / 128), rflags ? rflags->data_ptr<int>() : nullptr, rstart ? rstart->data_ptr<int>() : nullptr, (__nv_bfloat16*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// hierarchical chain of the slots with gs[16 * s + 6] (group size) != 0: stage 1 and stage 3 (see fixup_hd); grid = n_ctas (CTAs beyond the jobs of the replay exit at once)
void run_fixup_hd(at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor Qtot, at::Tensor Rtot, at::Tensor E1, at::Tensor sin_, at::Tensor gs, at::Tensor geom, int64_t H, int64_t stage,
                  int64_t n_ctas, c10::optional<at::Tensor> fin_pool, c10::optional<at::Tensor> fin_idx, int64_t fin_stride, int64_t ck_mode,
                  c10::optional<at::Tensor> cflags, int64_t cf_koff, int64_t cf_total, c10::optional<at::Tensor> rflags, c10::optional<at::Tensor> rstart) {
  c10::cuda::CUDAGuard guard_(sa.device());
  static bool attr[64] = {};
  int dev_ = 0; C10_CUDA_CHECK(cudaGetDevice(&dev_));
  if (!attr[dev_ & 63]) { C10_CUDA_CHECK(cudaFuncSetAttribute(cpfix::fixup_hd, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM)); attr[dev_ & 63] = true; }
  CUtensorMap mE = make_map_f32(E1.data_ptr(), (int)(E1.numel() / 128)), mQ = make_map_f32(Qtot.data_ptr(), (int)(Qtot.numel() / 128)), mR = make_map_f32(Rtot.data_ptr(), (int)(Rtot.numel() / 128));
  CUtensorMap ms0 = make_map_f32(sloc.data_ptr(), (int)(sloc.numel() / 128));
  CUtensorMap mPp = make_map_bf16(P.data_ptr(), (int)(P.numel() / 64)), mQp = make_map_bf16(Qtot.data_ptr(), (int)(Qtot.numel() / 64));
  CUtensorMap msa = make_map_f32(sa.data_ptr(), (int)(sa.numel() / 128));
  cpfix::fixup_hd<<<(unsigned)n_ctas, cpfix::NTH, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mE, mQ, mR, ms0, mR, mPp, mQp, msa, (int)(P.numel() / 128), (int)(Qtot.numel() / 128), sa.data_ptr<float>(), P.data_ptr<float>(), Qtot.data_ptr<float>(), E1.data_ptr<float>(), (uint8_t*)sin_.data_ptr(), (int)H, (int)stage,
      gs.data_ptr<int>(), geom.data_ptr<int>(), fin_pool ? fin_pool->data_ptr<float>() : nullptr, fin_idx ? fin_idx->data_ptr<int>() : nullptr, fin_stride, (int)ck_mode,
      cflags ? cflags->data_ptr<int>() : nullptr, (int)cf_koff, cflags ? (int)cflags->numel() - 1 : 0, (int)cf_total, rflags ? rflags->data_ptr<int>() : nullptr, rstart ? rstart->data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_chain_gate(at::Tensor geom, at::Tensor rstart, int64_t H, int64_t n_chain, int64_t n_hier_launch) {
  c10::cuda::CUDAGuard guard_(geom.device());
  cpfix::chain_gate<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(geom.data_ptr<int>(), rstart.data_ptr<int>(), (int)H, (int)n_chain, (int)n_hier_launch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixup_mpd", &run_fixup_mpd, pybind11::arg("out"), pybind11::arg("oq"), pybind11::arg("sa"), pybind11::arg("P"), pybind11::arg("sloc"), pybind11::arg("sin_"), pybind11::arg("seg"), pybind11::arg("seg_ptr"), pybind11::arg("gs"), pybind11::arg("geom"),
        pybind11::arg("H"), pybind11::arg("T"), pybind11::arg("mode"), pybind11::arg("n_ctas"), pybind11::arg("fin_pool"), pybind11::arg("fin_idx"), pybind11::arg("fin_stride"),
        pybind11::arg("ck_mode"), pybind11::arg("cu_k"), pybind11::arg("ck_true"), pybind11::arg("d_true"), pybind11::arg("ck_pc"), pybind11::arg("d_pc"), pybind11::arg("cu_tail"), pybind11::arg("ck_n"), pybind11::arg("trash"), pybind11::arg("rflags") = pybind11::none(), pybind11::arg("rstart") = pybind11::none()); m.def("fixup_hd", &run_fixup_hd, pybind11::arg("sa"), pybind11::arg("P"), pybind11::arg("sloc"), pybind11::arg("Qtot"), pybind11::arg("Rtot"), pybind11::arg("E1"), pybind11::arg("sin_"), pybind11::arg("gs"), pybind11::arg("geom"), pybind11::arg("H"), pybind11::arg("stage"),
        pybind11::arg("n_ctas"), pybind11::arg("fin_pool"), pybind11::arg("fin_idx"), pybind11::arg("fin_stride"), pybind11::arg("ck_mode"), pybind11::arg("cflags") = pybind11::none(), pybind11::arg("cf_koff") = 0, pybind11::arg("cf_total") = 0, pybind11::arg("rflags") = pybind11::none(), pybind11::arg("rstart") = pybind11::none());
  m.def("chain_gate", &run_chain_gate); }
