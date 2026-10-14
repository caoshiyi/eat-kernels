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
// [multi-GPU] switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; a plain CUDAGuard costs two)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};
#include <cuda_bf16.h>
#include <stdint.h>
#include <cuda.h>

namespace cpfix {

#ifdef KDA_FIX_CK
#define CHAIN_ chain
#else
#define CHAIN_ true
#endif

constexpr int NT = 256;        // 8 epilogue warps: warp w -> TMEM lane quarter w%4 (32 tokens), value half w/4; + warp 8 = TMA / MMA issuer
constexpr int NTH = 288;
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
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
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

// Warp-specialized: warps 0-7 = state items + B build + TMEM epilogue; warp 8 lane 0 = TMA loads / tcgen05 MMA issue / TMA stores.
// Tiles: D[t][v] = Qt_tile[t][:] . bf16(S_A)[v][:]^T (M = 128 tokens, N = 128, K = 128); the epilogue adds D to the o tile IN SMEM and warp 8
// stores the stage with one TMA store.  Per-CTA tile range is cut into segments of equal head (bf16(S_A) is rebuilt per segment).
__global__ void __launch_bounds__(NTH, 1)
fixup2(const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap msin,
       const __grid_constant__ CUtensorMap mfin,
#ifdef KDA_FIX_CK
       const __grid_constant__ CUtensorMap mck,
#endif
       const float* __restrict__ sa, const float* __restrict__ P,
       const float* __restrict__ sloc, float* __restrict__ sin_, float* __restrict__ sfin, const int4* __restrict__ seg, const int* __restrict__ seg_ptr,
       int H, int tail, const int* __restrict__ geom, const float* __restrict__ sa_base, const int* __restrict__ d_idx, long long d_stride,
       float* __restrict__ fin_pool, const int* __restrict__ fin_idx, long long fin_stride
#ifdef KDA_FIX_CK
       , const float* __restrict__ snapP, const float* __restrict__ snapL, const int4* __restrict__ cki, const long long* __restrict__ ck_cu
#endif
       ) {
  const int xoff = geom[0], m = geom[1];   // [planner_graph] tail geometry from the device
  if (m == 0) return;                      // no CP in this replay: nothing to fix up (leave before any initialisation)
#ifdef KDA_FIX_CK
  const int ns = geom[8];                  // [dynpath2] checkpoint items of this replay (device-planned): (snapshot index, piece, row relative to ck_cu[j], 0)
#endif
  if (sa_base) sa = sa_base + (long long)(d_idx ? d_idx[geom[3]] : geom[3]) * d_stride;   // S_A = the head launch's final state of the cut sequence (its pool / state slot)
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[3];           // [0..1] = state-item MMA done / unused
  __shared__ __align__(8) uint64_t full[NS], mdone[2], edone[NS];
  __shared__ uint32_t tmem_base_s;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* Bs = sm;
  auto stA = [&](int s) { return sm + TB + s * 2 * TB; };
  auto stO = [&](int s) { return sm + TB + s * 2 * TB + TB; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sg0 = seg_ptr[blockIdx.x], sg1 = seg_ptr[blockIdx.x + 1];

  if (threadIdx.x == 0) {
    for (int i = 0; i < 3; i++) mbar_init(su32(&bars[i]), 1);
    for (int i = 0; i < NS; i++) { mbar_init(su32(&full[i]), 1); mbar_init(su32(&edone[i]), NT); }
    mbar_init(su32(&mdone[0]), 1); mbar_init(su32(&mdone[1]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(su32(&tmem_base_s)), "r"(256) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tmem = tmem_base_s;

  if (warp < 8) {
    // ---------------- state items (epilogue warps only; the stage buffers double as scratch) ----------------
    {
    uint32_t sph = 0;
    const uint32_t idS = idesc(128, 128) | (1u << 16);
#ifdef KDA_FIX_CK
    for (int h = blockIdx.x; h < H; h += gridDim.x) for (int it = 0; it < m + ns; it++) {      // state items: heads h = b mod G, chain over the tail pieces, then the checkpoint items
      const bool chain = it < m;
      const int pc = chain ? it : 0;
      const float* Am; const float* Pm; const float* Lg; const CUtensorMap* md; int row; bool a_glob, hand_out;
      if (chain) {
        Am = (pc == 0 ? sa : sin_ + (size_t)(pc - 1) * H * 16384) + (size_t)h * 16384;   // state entering piece pc
        Pm = P + ((size_t)pc * H + h) * 16384;
        Lg = sloc + ((size_t)pc * H + h) * 16384;
        md = (pc == m - 1) ? &mfin : &msin;
        row = ((pc == m - 1 ? 0 : pc) * H + h) * 128;
        a_glob = (pc == 0); hand_out = pc < m - 1;
      } else {
        const int4 ci = cki[it - m];
        if (it == m && m > 1) {   // the chain's S_in stores must be complete before they are re-read below
          if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
          bar_epi();
        }
        Am = (ci.y == 0 ? sa : sin_ + (size_t)(ci.y - 1) * H * 16384) + (size_t)h * 16384;
        Pm = snapP + ((size_t)ci.x * H + h) * 16384;
        Lg = snapL + ((size_t)ci.x * H + h) * 16384;
        md = &mck;
        // ci.z = checkpoint row relative to the cut sequence's first row ck_cu[j] (device: replays may change checkpoint_cu_starts); a rejected sequence (< 0) gets a row past the end of the map
        const long long rb = __ldg(ck_cu + geom[3]);
        row = rb < 0 ? (1 << 30) : (int)((rb + ci.z) * H + h) * 128;
        a_glob = true; hand_out = false;
      }
      uint8_t* Ah = stA(0); uint8_t* Al = stO(0); uint8_t* Ph = stA(1); uint8_t* Pl = stO(1);
      uint8_t* Ls = stA(2);                            // S_loc fp32 [128][128] as four 32-column SW128 tiles (128 B rows), also the result staging for the TMA store
      if (chain && pc == 0 && m > 1) {
#else
    for (int h = blockIdx.x; h < H; h += gridDim.x) for (int pc = 0; pc < m; pc++) {      // state items: heads h = b mod G, chain over the tail pieces
      const float* Am = (pc == 0 ? sa : sin_ + (size_t)(pc - 1) * H * 16384) + (size_t)h * 16384;   // state entering piece pc
      const float* Pm = P + ((size_t)pc * H + h) * 16384;
      uint8_t* Ah = stA(0); uint8_t* Al = stO(0); uint8_t* Ph = stA(1); uint8_t* Pl = stO(1);
      uint8_t* Ls = stA(2);                            // S_loc fp32 [128][128] as four 32-column SW128 tiles (128 B rows), also the result staging for the TMA store
      if (pc == 0 && m > 1) {
#endif   // L2 prefetch of the next item's P and S_loc (independent of this item's result)
        const char* p1 = (const char*)(P + ((size_t)H + h) * 16384);
        const char* l1 = (const char*)(sloc + ((size_t)H + h) * 16384);
        #pragma unroll
        for (int k = 0; k < 2; k++) {
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p1 + (size_t)(k * NT + threadIdx.x) * 128));
          asm volatile("prefetch.global.L2 [%0];" :: "l"(l1 + (size_t)(k * NT + threadIdx.x) * 128));
        }
      }
      {
#ifndef KDA_FIX_CK
        const float* Lg = sloc + ((size_t)pc * H + h) * 16384;
#endif
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const int idx = k * NT + threadIdx.x, r = idx >> 5, c = idx & 31;
          cp16(su32(Ls + (c >> 3) * 16384 + r * 128 + (((c & 7) ^ (r & 7)) << 4)), Lg + idx * 4, true);
        }
        cp_commit();
      }
      {
        float4 xa[16], xp[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const int idx = k * NT + threadIdx.x;
#ifdef KDA_FIX_CK
          if (a_glob) xa[k] = __ldcg((const float4*)(Am + idx * 4));   // pc > 0: A (= S_in) was handed over in smem by the previous epilogue
#else
          if (pc == 0) xa[k] = __ldcg((const float4*)(Am + idx * 4));   // pc > 0: A (= S_in) was handed over in smem by the previous epilogue
#endif
          xp[k] = __ldcg((const float4*)(Pm + idx * 4));
        }
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const int idx = k * NT + threadIdx.x, r = idx >> 5, c4 = (idx & 31) * 4;
          const uint32_t off = swz(r, c4 >> 3) + ((c4 >> 2) & 1) * 8;
          uint32_t h0, l0, h1, l1;
#ifdef KDA_FIX_CK
          if (a_glob) {
#else
          if (pc == 0) {
#endif
            split2(make_float2(xa[k].x, xa[k].y), h0, l0); split2(make_float2(xa[k].z, xa[k].w), h1, l1);
            *(uint2*)(Ah + off) = make_uint2(h0, h1); *(uint2*)(Al + off) = make_uint2(l0, l1);
          }
          split2(make_float2(xp[k].x, xp[k].y), h0, l0); split2(make_float2(xp[k].z, xp[k].w), h1, l1);
          *(uint2*)(Ph + off) = make_uint2(h0, h1); *(uint2*)(Pl + off) = make_uint2(l0, l1);
        }
      }
      fence_async();
      bar_epi();
      if (warp == 0) {
        tc_after();
        const uint32_t ah = su32(Ah), al = su32(Al), ph = su32(Ph), pl = su32(Pl);
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          umma(tmem, sdesc(ah, kk), sdesc_mn(ph, kk), idS, kk);
          umma(tmem, sdesc(ah, kk), sdesc_mn(pl, kk), idS, 1);
          umma(tmem, sdesc(al, kk), sdesc_mn(ph, kk), idS, 1);
        }
        commit_w(su32(&bars[2]));
      }
      const int q = warp & 3, ch = warp >> 2, v = q * 32 + lane;
      cp_wait<0>();
      bar_epi();   // [finalrev] S_loc rows staged by cp.async are read by OTHER epilogue threads: every thread's cp.async must have completed (racecheck: 80-100 hazards per run without this barrier)
      mbar_wait(su32(&bars[2]), sph); sph ^= 1;
      tc_after();
      uint32_t d[64];
      const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(ch * 64);
      TLD32(d, ta);
      TLD32((d + 32), ta + 32);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      #pragma unroll
      for (int j = 0; j < 16; j++)
      {
        const int g32 = ch * 16 + j;
        uint8_t* lp = Ls + (g32 >> 3) * 16384 + v * 128 + (((g32 & 7) ^ (v & 7)) << 4);
        const float4 l = *(const float4*)lp;
        const float4 f = make_float4(l.x + __uint_as_float(d[4 * j]), l.y + __uint_as_float(d[4 * j + 1]),
                                     l.z + __uint_as_float(d[4 * j + 2]), l.w + __uint_as_float(d[4 * j + 3]));
        *(float4*)lp = f;   // result staged in place (SW128), stored by TMA below
#ifdef KDA_FIX_CK
        if (hand_out) {   // hand S_in(pc + 1) to the next item as its A operand (bf16 hi + lo); the MMA that read Ah / Al is complete
#else
        if (pc < m - 1) {
#endif
          uint32_t h0, l0, h1, l1;
          split2(make_float2(f.x, f.y), h0, l0); split2(make_float2(f.z, f.w), h1, l1);
          const uint32_t off = swz(v, ch * 8 + (j >> 1)) + (j & 1) * 8;
          *(uint2*)(Ah + off) = make_uint2(h0, h1); *(uint2*)(Al + off) = make_uint2(l0, l1);
        }
      }
      tc_before();
      fence_async();
      bar_epi();
      if (CHAIN_ && pc == m - 1 && fin_pool != nullptr) {   // [dynpath2] the final state goes straight to its pool / state slot (read on the device; S_A lives in scratch: no alias), no scatter launch
        float* dstp = fin_pool + (long long)(fin_idx ? __ldg(fin_idx + geom[3]) : geom[3]) * fin_stride + (size_t)h * 16384;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const int idx = k * NT + threadIdx.x, r = idx >> 5, c = idx & 31;
          *(float4*)(dstp + r * 128 + c * 4) = *(const float4*)(Ls + (c >> 3) * 16384 + r * 128 + (((c & 7) ^ (r & 7)) << 4));
        }
      } else if (warp == 0 && lane == 0) {
#ifndef KDA_FIX_CK
        const CUtensorMap* md = (pc == m - 1) ? &mfin : &msin;
        const int row = ((pc == m - 1 ? 0 : pc) * H + h) * 128;
#endif
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_store2(md, qq * 32, row, su32(Ls + qq * 16384));
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
      bar_epi();
    }
    }
#ifdef KDA_FIX_CK
    if (m > 1 && warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");   // S_in(1..m-1) stores complete (checkpoint / final-state stores: waited at exit by the TMA engine)
#else
    if (m > 1 && warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");   // S_in(1..m-1) stores complete (the final-state store may still be in flight)
#endif
   // S_in(1..m-1) are re-read (ldcg) as tile operands by other threads of this CTA after the barrier below
  }

  __syncthreads();   // all 9 warps: the state-item scratch (stage buffers) is free from here
  const uint32_t id = idesc(128, 128);
  int j0 = 0;                                           // CTA-global tile counter (stage / barrier phases)
  for (int si = sg0; si < sg1; si++) {
    const int4 sgm = seg[si];                           // (head, piece, first tile of the head's tail, tile count)
    const int h = sgm.x, pcs = sgm.y, tile0 = sgm.z, n = sgm.w;
    if (warp < 8) {
      // B = bf16(S_A of head h), SW128 K-major [v][k]
      const float* src = (pcs == 0 ? sa : sin_ + (size_t)(pcs - 1) * H * 16384) + (size_t)h * 16384;
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
        const int j = j0 + k, s = j % NS, t0 = xoff + (tile0 + k) * TM;
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
        const bool row_in = (xoff + (tile0 + t) * TM + tr) < geom[2];   // [finalrev] rows past the tail end belong to the next sequence: keep their output (0 * Inf / NaN of an overflowed S_A must not reach them)
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
        tc_before();
        fence_async();
        mbar_arrive(su32(&edone[s]));
      }
    } else if (lane == 0) {
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
          const int jp = j - 1, sp = jp % NS, up = tile0 + t - 1;
          mbar_wait(su32(&edone[sp]), (jp / NS) & 1);
          tma_store4(&mo, 0, xoff + up * TM, h, 0, su32(stO(sp)));
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          if (t - 1 + NS < n) {
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            const int jl = jp + NS, ul = up + NS, t0 = xoff + ul * TM;
            const uint32_t bb = su32(&full[sp]);
            expect_tx(bb, 2 * TB);
            tma_load4(su32(stA(sp)), &mq, 0, t0, h, 0, bb);
            tma_load4(su32(stO(sp)), &mo, 0, t0, h, 0, bb);
            (void)jl;
          }
        }
      }
      {   // last tile of the segment
        const int jp = j0 + n - 1, sp = jp % NS, up = tile0 + n - 1;
        mbar_wait(su32(&edone[sp]), (jp / NS) & 1);
        tma_store4(&mo, 0, xoff + up * TM, h, 0, su32(stO(sp)));
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
    }
    __syncthreads();   // segment drained (MMAs done and stores read) before B / stages are reused
    j0 += n;
  }
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(256));
}

}  // namespace cpfix

typedef CUresult (*EncFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                          CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static CUtensorMap make_map(void* ptr, int tail, int H) {
  static EncFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
    enc = (EncFn)fn;
  }
  cuuint64_t gd[4] = {64, (cuuint64_t)tail, (cuuint64_t)H, 2};
  cuuint64_t gs[3] = {(cuuint64_t)H * 256, 256, 128};
  cuuint32_t box[4] = {64, 128, 1, 2}, es[4] = {1, 1, 1, 1};
  CUtensorMap m{};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, gd, gs, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", int(r));
  return m;
}

static CUtensorMap make_map_f32(void* ptr, int rows) {   // fp32 [rows][128], boxes of 32 columns x 128 rows, SW128 (the state-item staging layout)
  static EncFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr; cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
    enc = (EncFn)fn;
  }
  cuuint64_t gd[2] = {128, (cuuint64_t)rows};
  cuuint64_t gs[1] = {512};
  cuuint32_t box[2] = {32, 128}, es[2] = {1, 1};
  CUtensorMap m{};
  CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr, gd, gs, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(f32) failed ", int(r));
  return m;
}

// [planner_graph] graph-safe entry: out / oq are the FULL [T, H, 128] bf16 tensors (absolute row coordinates), geom = int32 [2] on the device
void run_fixup_dyn(at::Tensor out, at::Tensor oq, at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor sin_, at::Tensor sfin, at::Tensor seg,
                   at::Tensor seg_ptr, at::Tensor geom, int64_t H, int64_t T, c10::optional<at::Tensor> sa_base, c10::optional<at::Tensor> d_idx, int64_t d_stride,
                   c10::optional<at::Tensor> fin_pool, c10::optional<at::Tensor> fin_idx, int64_t fin_stride
#ifdef KDA_FIX_CK
                   , c10::optional<at::Tensor> ck_buf, c10::optional<at::Tensor> snap_p, c10::optional<at::Tensor> snap_l, c10::optional<at::Tensor> cki, c10::optional<at::Tensor> ck_cu
#endif
                   ) {
  KdaDevGuard dev_guard_(out);   // [multi-GPU] launch (and the per-device attribute flag below) on the tensors' device
  const int grid = (int)seg_ptr.numel() - 1;
  static bool attr[64] = {};   // per device (audit F3): the attribute is a per-device property
  int dev_ = 0; cudaGetDevice(&dev_);
  if (!attr[dev_ & 63]) { cudaFuncSetAttribute(cpfix::fixup2, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM); attr[dev_ & 63] = true; }
  CUtensorMap mo = make_map(out.data_ptr(), (int)T, (int)H), mq = make_map(oq.data_ptr(), (int)T, (int)H);
  CUtensorMap mfin = make_map_f32(sfin.data_ptr(), (int)H * 128);
  CUtensorMap msin = make_map_f32(sin_.data_ptr(), (int)H * 128);      // m <= 2: one chained state per head
#ifdef KDA_FIX_CK
  CUtensorMap mck = ck_buf ? make_map_f32(ck_buf->data_ptr(), (int)(ck_buf->size(0) * H * 128)) : mfin;
#endif
  cpfix::fixup2<<<(unsigned)grid, cpfix::NTH, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mo, mq, msin, mfin,
#ifdef KDA_FIX_CK
      mck,
#endif
      sa.data_ptr<float>(), P.data_ptr<float>(), sloc.data_ptr<float>(), sin_.data_ptr<float>(), sfin.data_ptr<float>(),
      (const int4*)seg.data_ptr<int>(), seg_ptr.data_ptr<int>(), (int)H, 0, geom.data_ptr<int>(),
      sa_base ? sa_base->data_ptr<float>() : nullptr, d_idx ? d_idx->data_ptr<int>() : nullptr, d_stride,
      fin_pool ? fin_pool->data_ptr<float>() : nullptr, fin_idx ? fin_idx->data_ptr<int>() : nullptr, fin_stride
#ifdef KDA_FIX_CK
      , snap_p ? snap_p->data_ptr<float>() : nullptr, snap_l ? snap_l->data_ptr<float>() : nullptr, cki ? (const int4*)cki->data_ptr<int>() : nullptr,
      ck_cu ? (const long long*)ck_cu->data_ptr() : nullptr
#endif
      );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t new_stream() {
  cudaStream_t st;
  int lo = 0, hi = 0; C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); C10_CUDA_CHECK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, hi));
  return (int64_t)(uintptr_t)st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fixup_dyn", &run_fixup_dyn); m.def("new_stream", &new_stream); }
