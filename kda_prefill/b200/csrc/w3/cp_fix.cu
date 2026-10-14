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
#define TST16(a, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" \
  :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), \
    "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory")
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

__device__ __forceinline__ void tma_load2(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"((unsigned long long)m), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// Warp-specialized: warps 0-7 = state items + B build + TMEM epilogue; warp 8 lane 0 = TMA loads / tcgen05 MMA issue / TMA stores.
// Tiles: D[t][v] = Qt_tile[t][:] . bf16(S_A)[v][:]^T (M = 128 tokens, N = 128, K = 128); the epilogue adds D to the o tile IN SMEM and warp 8
// stores the stage with one TMA store.  Per-CTA tile range is cut into segments of equal head (bf16(S_A) is rebuilt per segment).
__global__ void __launch_bounds__(NTH, 1)
fixup2(const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap msin,
       const __grid_constant__ CUtensorMap mfin, const float* __restrict__ sa, const float* __restrict__ P,
       const float* __restrict__ sloc, float* __restrict__ sin_, float* __restrict__ sfin, const int4* __restrict__ seg, const int* __restrict__ seg_ptr,
       int H, int tail, int m) {
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
    for (int h = blockIdx.x; h < H; h += gridDim.x) for (int pc = 0; pc < m; pc++) {      // state items: heads h = b mod G, chain over the tail pieces
      const float* Am = (pc == 0 ? sa : sin_ + (size_t)(pc - 1) * H * 16384) + (size_t)h * 16384;   // state entering piece pc
      const float* Pm = P + ((size_t)pc * H + h) * 16384;
      uint8_t* Ah = stA(0); uint8_t* Al = stO(0); uint8_t* Ph = stA(1); uint8_t* Pl = stO(1);
      uint8_t* Ls = stA(2);                            // S_loc fp32 [128][128] as four 32-column SW128 tiles (128 B rows), also the result staging for the TMA store
      if (pc == 0 && m > 1) {   // L2 prefetch of the next item's P and S_loc (independent of this item's result)
        const char* p1 = (const char*)(P + ((size_t)H + h) * 16384);
        const char* l1 = (const char*)(sloc + ((size_t)H + h) * 16384);
        #pragma unroll
        for (int k = 0; k < 2; k++) {
          asm volatile("prefetch.global.L2 [%0];" :: "l"(p1 + (size_t)(k * NT + threadIdx.x) * 128));
          asm volatile("prefetch.global.L2 [%0];" :: "l"(l1 + (size_t)(k * NT + threadIdx.x) * 128));
        }
      }
      {
        const float* Lg = sloc + ((size_t)pc * H + h) * 16384;
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
          if (pc == 0) xa[k] = __ldcg((const float4*)(Am + idx * 4));   // pc > 0: A (= S_in) was handed over in smem by the previous epilogue
          xp[k] = __ldcg((const float4*)(Pm + idx * 4));
        }
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const int idx = k * NT + threadIdx.x, r = idx >> 5, c4 = (idx & 31) * 4;
          const uint32_t off = swz(r, c4 >> 3) + ((c4 >> 2) & 1) * 8;
          uint32_t h0, l0, h1, l1;
          if (pc == 0) {
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
        if (pc < m - 1) {   // hand S_in(pc + 1) to the next item as its A operand (bf16 hi + lo); the MMA that read Ah / Al is complete
          uint32_t h0, l0, h1, l1;
          split2(make_float2(f.x, f.y), h0, l0); split2(make_float2(f.z, f.w), h1, l1);
          const uint32_t off = swz(v, ch * 8 + (j >> 1)) + (j & 1) * 8;
          *(uint2*)(Ah + off) = make_uint2(h0, h1); *(uint2*)(Al + off) = make_uint2(l0, l1);
        }
      }
      tc_before();
      fence_async();
      bar_epi();
      if (warp == 0 && lane == 0) {
        const CUtensorMap* md = (pc == m - 1) ? &mfin : &msin;
        const int row = ((pc == m - 1 ? 0 : pc) * H + h) * 128;
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_store2(md, qq * 32, row, su32(Ls + qq * 16384));
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
      bar_epi();
    }
    }
    if (m > 1 && warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");   // S_in(1..m-1) stores complete (the final-state store may still be in flight)
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
        const int j = j0 + k, s = j % NS, t0 = (tile0 + k) * TM;
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
        #pragma unroll
        for (int c = 0; c < 8; c++) {
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
          tma_store4(&mo, 0, up * TM, h, 0, su32(stO(sp)));
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          if (t - 1 + NS < n) {
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            const int jl = jp + NS, ul = up + NS, t0 = ul * TM;
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
        tma_store4(&mo, 0, up * TM, h, 0, su32(stO(sp)));
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

// [portfolio] many-piece fixup: mode 1 = latency-optimised state chain, mode 2 = output tiles of every piece (cp_fix.cu v44 kernel above is unchanged)
__global__ void __launch_bounds__(NTH, 1)
fixup_mp(const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap msin,
       const __grid_constant__ CUtensorMap mfin, const __grid_constant__ CUtensorMap msloc, const float* __restrict__ sa, const float* __restrict__ P,
       const float* __restrict__ sloc, float* __restrict__ sin_, float* __restrict__ sfin, const int4* __restrict__ seg, const int* __restrict__ seg_ptr,
       int H, int tail, int m, int mode) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[4];           // [0..1] = state-item MMA done / unused
  __shared__ __align__(8) uint64_t full[NS], mdone[2], edone[NS];
  __shared__ uint32_t tmem_base_s;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* Bs = sm;
  auto stA = [&](int s) { return sm + TB + s * 2 * TB; };
  auto stO = [&](int s) { return sm + TB + s * 2 * TB + TB; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int sg0 = seg_ptr[blockIdx.x], sg1 = seg_ptr[blockIdx.x + 1];

  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; i++) mbar_init(su32(&bars[i]), 1);
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

  if (warp == 8 && lane == 0 && mode == 1) {
    // ---------------- [portfolio] chain launch: MMA issuer / S_loc loader (one thread).  ready barrier bars[1]: A (TMEM), P buffer and Ls are free / written ----------------
    const uint32_t idS = idesc(128, 128) | (1u << 16);
    const uint32_t tah = tmem + 128, tal = tmem + 192;
    uint32_t rph = 0;
    for (int h = blockIdx.x; h < H; h += gridDim.x) for (int pc = 0; pc < m; pc++) {
      mbar_wait(su32(&bars[1]), rph); rph ^= 1;
      if (pc > 0)   // bf16(S_in(pc)) was staged in Bs by the epilogue of item pc - 1
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"((uint8_t*)sin_ + ((size_t)(pc - 1) * H + h) * 32768), "r"(su32(Bs)) : "memory");
      expect_tx(su32(&bars[0]), 65536);
      #pragma unroll
      for (int qq = 0; qq < 4; qq++) tma_load2(su32(stA(2) + qq * 16384), &msloc, qq * 32, (pc * H + h) * 128, su32(&bars[0]));
      tc_after();
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
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  if (warp < 8 && mode == 1) {
    // ---------------- [portfolio] latency-optimised state chain (mode 1: chain only; the output tiles are a separate launch, mode 2) ----------------
    // Same item arithmetic as below (S_in(p+1) = S_loc_p + S_in(p) P_p; bf16 hi / lo operand pairs, hi x hi + hi x lo + lo x hi) with the work off the dependent path:
    //   * the entering state A = S_in(p) lives in TMEM (bf16 hi / lo words, cols 128-191 / 192-255) and feeds the MMA from there (kind::f16 .ts): the epilogue writes it back with
    //     tcgen05.st, no smem round trip;  P_{p+1} is split into the second P buffer while MMA(p) runs;  S_loc arrives by TMA (Ls);  one thread (warp 8) issues the MMAs;
    //   * the entering state of the output tiles, bf16(S_in(p)) (= the hi words), is staged in Bs in the tile phase's own swizzled K-major layout and written with one 32 KB bulk store
    //     (sin_ is a bf16 buffer [m - 1][H][128][128] in mode 1 / 2);  the final state S_fin goes out as fp32 through Ls + TMA store.
    const int q = warp & 3, ch = warp >> 2, v = q * 32 + lane;
    uint8_t* Pbh[2] = {stA(0), stA(1)}; uint8_t* Pbl[2] = {stO(0), stO(1)};
    uint8_t* Ls = stA(2);
    const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(ch * 64);        // D columns of this thread
    const uint32_t tah = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(128 + ch * 32);  // A hi words
    const uint32_t tal = tmem + ((uint32_t)(q * 32) << 16) + (uint32_t)(192 + ch * 32);  // A lo words
    uint32_t sph = 0, lph = 0, bph3 = 0;
    auto load_split_p = [&](const float* Pm, uint8_t* Ph, uint8_t* Pl) {
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
    };
    for (int h = blockIdx.x; h < H; h += gridDim.x) {
      // item 0: A = S_A (thread-row mapping, once per head), P_0 -> buffer 0
      {
        uint32_t hw[32], lw[32];
        const float* Ag = sa + (size_t)h * 16384 + v * 128 + ch * 64;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
          const float4 x = __ldcg((const float4*)(Ag + 4 * k));
          split2(make_float2(x.x, x.y), hw[2 * k], lw[2 * k]); split2(make_float2(x.z, x.w), hw[2 * k + 1], lw[2 * k + 1]);
        }
        TST32(tah, hw);
        TST32(tal, lw);
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        load_split_p(P + (size_t)h * 16384, Pbh[0], Pbl[0]);
        tc_before();
        fence_async();
        if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // the previous head's final-state store has read Ls
        bar_epi();
        if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
      }
      for (int pc = 0; pc < m; pc++) {
        if (pc + 1 < m) load_split_p(P + ((size_t)(pc + 1) * H + h) * 16384, Pbh[(pc + 1) & 1], Pbl[(pc + 1) & 1]);
        fence_async();
        mbar_wait(su32(&bars[0]), lph); lph ^= 1;     // S_loc_pc in Ls
        mbar_wait(su32(&bars[2]), sph); sph ^= 1;     // MMA(pc) done
        mbar_wait(su32(&bars[3]), bph3); bph3 ^= 1;   // Bs free (the previous item's bulk store has read it)
        tc_after();
        uint32_t d[64];
        TLD32(d, ta);
        TLD32((d + 32), ta + 32);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t hw[32], lw[32];
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
            *(uint2*)(Bs + swz(v, ch * 8 + (j >> 1)) + (j & 1) * 8) = make_uint2(hw[2 * j], hw[2 * j + 1]);
          }
        }
        if (pc < m - 1) {
          TST32(tah, hw);
          TST32(tal, lw);
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        tc_before();
        fence_async();
        bar_epi();      // A in TMEM, Bs / Ls staged, every thread has read Ls
        if (warp == 0 && lane == 0) {
          if (pc < m - 1) {
            mbar_arrive(su32(&bars[1]));
          } else {
            #pragma unroll
            for (int qq = 0; qq < 4; qq++) tma_store2(&mfin, qq * 32, h * 128, su32(Ls + qq * 16384));
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          }
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
    const int4 sgm = seg[si];                           // (head, piece, first tile of the head's tail, tile count)
    const int h = sgm.x, pcs = sgm.y, tile0 = sgm.z, n = sgm.w;
    const bool bulkB = (mode == 2 && pcs >= 1);   // [portfolio] bf16(S_in) was staged by the chain launch in B's own layout: one bulk load
    if (bulkB && warp == 8 && lane == 0) {
      expect_tx(su32(&bars[0]), 32768);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 32768, [%2];"
                   :: "r"(su32(Bs)), "l"((const uint8_t*)sin_ + ((size_t)(pcs - 1) * H + h) * 32768), "r"(su32(&bars[0])) : "memory");
    }
    if (warp < 8 && !bulkB) {
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
        const int j = j0 + k, s = j % NS, t0 = (tile0 + k) * TM;
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
        #pragma unroll
        for (int c = 0; c < 8; c++) {
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
          const int jp = j - 1, sp = jp % NS, up = tile0 + t - 1;
          mbar_wait(su32(&edone[sp]), (jp / NS) & 1);
          tma_store4(&mo, 0, up * TM, h, 0, su32(stO(sp)));
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          if (t - 1 + NS < n) {
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            const int jl = jp + NS, ul = up + NS, t0 = ul * TM;
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
        tma_store4(&mo, 0, up * TM, h, 0, su32(stO(sp)));
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
__device__ __noinline__ void h_epi_single(uint32_t ta, uint32_t t_hi, uint32_t t_lo, uint8_t* Ls, uint8_t* Bs, int v, int ch, bool last, bool do_bf16, bool do_f32) {
  #pragma unroll 1
  for (int hh = 0; hh < 2; hh++) {
    uint32_t d[32], hw[16], lw[16];
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

__global__ void __launch_bounds__(NTH, 1)
fixup_h(const __grid_constant__ CUtensorMap mfin, const __grid_constant__ CUtensorMap mE, const __grid_constant__ CUtensorMap mQ, const __grid_constant__ CUtensorMap mR,
        const __grid_constant__ CUtensorMap ms0, const __grid_constant__ CUtensorMap ms1,
        const float* __restrict__ sa, const float* __restrict__ P, const float* __restrict__ Qtot, const float* __restrict__ E1, uint8_t* __restrict__ sin_,
        int H, int m, int s, int G, int stage) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t bars[4];     // [0] S_loc full (tx), [1] ready (A / P / Ls written), [2] MMA done, [3] Bs free
  __shared__ uint32_t tmem_base_s;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* Bs = sm;
  auto stA = [&](int i) { return sm + TB + i * 2 * TB; };
  auto stO = [&](int i) { return sm + TB + i * 2 * TB + TB; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; i++) mbar_init(su32(&bars[i]), 1);
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
  const int njobs = H * (G - 1);
  // item i of job (h, g): P source, S_loc map / row, piece index (-1: a group-total item of stage 3)
  auto item_p = [&](int h, int g, int i) -> const float* {
    if (stage == 3 && i < g - 1) return Qtot + ((size_t)(1 + i) * H + h) * 16384;
    return P + ((size_t)(g * s + (stage == 3 ? i - (g - 1) : i)) * H + h) * 16384;
  };
  auto item_pp = [&](int g, int i) -> int { return (stage == 3 && i < g - 1) ? -1 : g * s + (stage == 3 ? i - (g - 1) : i); };
  auto n_items = [&](int g) { const int cnt = min(s, m - g * s); return stage == 3 ? (g - 1) + cnt : cnt; };
  const uint32_t idS = idesc(128, 128) | (1u << 16);

  if (warp == 8 && lane == 0) {
    // ---------------- MMA issuer / S_loc loader / bf16 staging store ----------------
    uint32_t rph = 0;
    for (int jb = blockIdx.x; jb < njobs; jb += gridDim.x) {
      const int h = jb % H, g = (stage == 1 ? jb / H : 1 + jb / H);
      const int n = n_items(g);
      const bool dual = (stage == 1 && g >= 1);
      for (int i = 0; i < n; i++) {
        mbar_wait(su32(&bars[1]), rph); rph ^= 1;
        if (i > 0) {
          const int pp = item_pp(g, i - 1);
          if (pp >= 0 && pp < m - 1)    // bf16(S_in(pp + 1)) staged by the epilogue of the previous item
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"(sin_ + ((size_t)pp * H + h) * 32768), "r"(su32(Bs)) : "memory");
        }
        const int pp = item_pp(g, i);
        const CUtensorMap* msl = (pp < 0) ? &ms1 : &ms0;
        const int lrow = (pp < 0 ? (1 + i) * H + h : pp * H + h) * 128;
        expect_tx(su32(&bars[0]), 65536);
        #pragma unroll
        for (int qq = 0; qq < 4; qq++) tma_load2(su32(stA(2) + qq * 16384), msl, qq * 32, lrow, su32(&bars[0]));
        tc_after();
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
    uint32_t sph = 0, lph = 0, bph3 = 0;
    for (int jb = blockIdx.x; jb < njobs; jb += gridDim.x) {
      const int h = jb % H, g = (stage == 1 ? jb / H : 1 + jb / H);
      const int n = n_items(g);
      const bool dual = (stage == 1 && g >= 1);
      {   // job prologue: entering operands, P of item 0
        if (!dual) h_init_a((stage == 1 ? sa : E1) + (size_t)h * 16384 + v * 128 + ch * 64, row0 + 128 + ch * 32, row0 + 192 + ch * 32);
        else h_init_dual(v, ch, row0 + 256 + ch * 32, row0 + 320 + ch * 32, row0 + 384 + ch * 32, row0 + 448 + ch * 32);
        h_split_p(item_p(h, g, 0), stA(0), stO(0));
        tc_before();
        fence_async();
        if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // the previous job's final stores have read Bs / Ls
        bar_epi();
        if (threadIdx.x == 0) mbar_arrive(su32(&bars[1]));
      }
      for (int i = 0; i < n; i++) {
        if (i + 1 < n) h_split_p(item_p(h, g, i + 1), stA((i + 1) & 1), stO((i + 1) & 1));
        fence_async();
        mbar_wait(su32(&bars[0]), lph); lph ^= 1;
        mbar_wait(su32(&bars[2]), sph); sph ^= 1;
        mbar_wait(su32(&bars[3]), bph3); bph3 ^= 1;
        tc_after();
        const bool last = (i == n - 1);
        const int pp = item_pp(g, i);
        if (!dual) {
          const bool do_bf16 = (pp >= 0 && pp < m - 1);
          const bool do_f32 = last && (stage == 3 ? g == G - 1 : g == 0);
          h_epi_single(ta, row0 + 128 + ch * 32, row0 + 192 + ch * 32, Ls, Bs, v, ch, last, do_bf16, do_f32);
          tc_before();
          fence_async();
          bar_epi();
          if (warp == 0 && lane == 0) {
            if (!last) mbar_arrive(su32(&bars[1]));
            else {   // job end: this thread issues what the issuer thread would have issued at the next item
              if (do_bf16)
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 32768;" :: "l"(sin_ + ((size_t)pp * H + h) * 32768), "r"(su32(Bs)) : "memory");
              if (do_f32) {
                const CUtensorMap* mo = (stage == 3 ? &mfin : &mE);
                #pragma unroll
                for (int qq = 0; qq < 4; qq++) tma_store2(mo, qq * 32, h * 128, su32(Ls + qq * 16384));
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
              for (int qq = 0; qq < 4; qq++) tma_store2(&mR, qq * 32, (g * H + h) * 128, su32(Ls + qq * 16384));
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
            bar_epi();
            h_copy_tmem_to_ls(ta, Ls, v, ch);
            fence_async();
            bar_epi();
            if (warp == 0 && lane == 0) {
              #pragma unroll
              for (int qq = 0; qq < 4; qq++) tma_store2(&mQ, qq * 32, (g * H + h) * 128, su32(Ls + qq * 16384));
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
          }
        }
      }
    }
    if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
  tc_before();
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem), "r"(512));
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

void run_fixup(at::Tensor out, at::Tensor oq, at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor sin_, at::Tensor sfin, at::Tensor seg,
               at::Tensor seg_ptr, int64_t H, int64_t tail, int64_t m) {
  c10::cuda::CUDAGuard guard_(out.device());
  const int grid = (int)seg_ptr.numel() - 1;
  static bool attr[64] = {};   // per device (audit F3): the attribute is a per-device property
  int dev_ = 0; C10_CUDA_CHECK(cudaGetDevice(&dev_));
  if (!attr[dev_ & 63]) { C10_CUDA_CHECK(cudaFuncSetAttribute(cpfix::fixup2, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM)); attr[dev_ & 63] = true; }
  CUtensorMap mo = make_map(out.data_ptr(), (int)tail, (int)H), mq = make_map(oq.data_ptr(), (int)tail, (int)H);
  CUtensorMap mfin = make_map_f32(sfin.data_ptr(), (int)H * 128);
  CUtensorMap msin = sin_.numel() ? make_map_f32(sin_.data_ptr(), (int)(m - 1) * (int)H * 128) : mfin;
  cpfix::fixup2<<<(unsigned)grid, cpfix::NTH, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mo, mq, msin, mfin, sa.data_ptr<float>(), P.data_ptr<float>(), sloc.data_ptr<float>(), sin_.numel() ? sin_.data_ptr<float>() : nullptr, sfin.data_ptr<float>(),
      (const int4*)seg.data_ptr<int>(), seg_ptr.data_ptr<int>(), (int)H, (int)tail, (int)m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void run_fixup_mp(at::Tensor out, at::Tensor oq, at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor sin_, at::Tensor sfin, at::Tensor seg,
                  at::Tensor seg_ptr, int64_t H, int64_t tail, int64_t m, int64_t mode) {
  c10::cuda::CUDAGuard guard_(out.device());
  const int grid = (int)seg_ptr.numel() - 1;   // mode 1: one CTA per chain head (zero seg_ptr); mode 2: tile segments per CTA
  static bool attr[64] = {};   // per device (audit F3): the attribute is a per-device property
  int dev_ = 0; C10_CUDA_CHECK(cudaGetDevice(&dev_));
  if (!attr[dev_ & 63]) { C10_CUDA_CHECK(cudaFuncSetAttribute(cpfix::fixup_mp, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM)); attr[dev_ & 63] = true; }
  CUtensorMap mo = make_map(out.data_ptr(), (int)tail, (int)H), mq = make_map(oq.data_ptr(), (int)tail, (int)H);
  CUtensorMap mfin = make_map_f32(sfin.data_ptr(), (int)H * 128);
  CUtensorMap msloc = mode == 1 ? make_map_f32(sloc.data_ptr(), (int)(m * H) * 128) : mfin;
  cpfix::fixup_mp<<<(unsigned)grid, cpfix::NTH, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mo, mq, mfin, mfin, msloc, sa.data_ptr<float>(), P.data_ptr<float>(), sloc.data_ptr<float>(), (float*)sin_.data_ptr(), sfin.data_ptr<float>(),
      (const int4*)seg.data_ptr<int>(), seg_ptr.data_ptr<int>(), (int)H, (int)tail, (int)m, (int)mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void run_fixup_h(at::Tensor sa, at::Tensor P, at::Tensor sloc, at::Tensor Qtot, at::Tensor Rtot, at::Tensor E1, at::Tensor sin_, at::Tensor sfin,
                 int64_t H, int64_t m, int64_t s, int64_t G, int64_t stage, int64_t max_ctas) {
  c10::cuda::CUDAGuard guard_(sa.device());
  const int njobs = (int)(H * (G - 1));
  const int grid = njobs < (int)max_ctas ? njobs : (int)max_ctas;
  static bool attr[64] = {};   // per device (audit F3): the attribute is a per-device property
  int dev_ = 0; C10_CUDA_CHECK(cudaGetDevice(&dev_));
  if (!attr[dev_ & 63]) { C10_CUDA_CHECK(cudaFuncSetAttribute(cpfix::fixup_h, cudaFuncAttributeMaxDynamicSharedMemorySize, cpfix::SMEM)); attr[dev_ & 63] = true; }
  CUtensorMap mfin = make_map_f32(sfin.data_ptr(), (int)H * 128), mE = make_map_f32(E1.data_ptr(), (int)H * 128);
  CUtensorMap mQ = make_map_f32(Qtot.data_ptr(), (int)(G * H) * 128), mR = make_map_f32(Rtot.data_ptr(), (int)(G * H) * 128);
  CUtensorMap ms0 = make_map_f32(sloc.data_ptr(), (int)(m * H) * 128), ms1 = mR;
  cpfix::fixup_h<<<(unsigned)grid, cpfix::NTH, cpfix::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      mfin, mE, mQ, mR, ms0, ms1, sa.data_ptr<float>(), P.data_ptr<float>(), Qtot.data_ptr<float>(), E1.data_ptr<float>(), (uint8_t*)sin_.data_ptr(),
      (int)H, (int)m, (int)s, (int)G, (int)stage);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t new_stream() {
  cudaStream_t st;
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  return (int64_t)(uintptr_t)st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fixup", &run_fixup);
  m.def("fixup_mp", &run_fixup_mp); m.def("fixup_h", &run_fixup_h); m.def("new_stream", &new_stream); }
