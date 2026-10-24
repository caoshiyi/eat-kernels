// Bespoke short-sequence KDA forward (work_tiny2): one CTA per (sequence, head) task, every task at most ONE 32-token chunk.  mma.sync bf16 tensor-core math with fp32 state in
// registers, no TMA / TMEM / pipeline machinery: the persistent kernels pay ~9.7 us for a one-chunk launch (start-up chain + one pass through a latency-bound pipeline);
// this kernel is 6.6-7.5 us (cold L2, graph replay) for <= 16 tokens and 9.6 us for 17..32, i.e. launch + cold-HBM latency + a short dependent chain.
//
// Math (exactly the main kernels' chunked WY form).  Per task, C <= 32 tokens, D = 128; G = inclusive prefix over the chunk of the log-decay a_t = lb * sigmoid(exp(A_log)(g_t + dt_bias)),
// c = G_last / 2 per key channel (midpoint scaling keeps every bf16 factor inside e^{+-80}, the exponent range of a chunk), q, k L2-normalised (rsqrt(max(|x|^2, 1e-12)), the eps convention
// of the reference), beta = sigmoid(logit):
//     Akk[i,j] = sum_d k_i k_j e^{G_i-G_j} (j < i),   Aqk[i,j] = sum_d q_i k_j e^{G_i-G_j} (j <= i)              (bf16 MMA of k e^{G-c} and k e^{c-G}, fp32 accumulate)
//     R^T = S0 (k e^G)^T,  X = beta (v - R),  v_new = (I + diag(beta) Akk)^-1 X                                      (T = (I + L)^-1 in fp32, stored as bf16 hi + lo, see below)
//     o = scale (S0 (q e^G)^T + Aqk v_new),      S <- S e^{G_last} + v_new^T (k e^{G_last-G})                           (state stays fp32)
// Padded tokens of a partial chunk have k = q = v = 0, beta = 0 and a = 0, so they change nothing (and G_last is the last valid token's prefix).
//
// Layout.  S [V][K] is held in the MMA accumulator layout: warp w (0-7) owns value rows 16w..16w+15 and all 128 key channels (64 fp32 registers).  The key axis is permuted inside each
// 32-channel block so that a lane's state access is one 32-byte ld/st.global.v8 and the 4 lanes of a quad cover one 128-byte line (see load_state); all key-axis tiles use the same permutation.
// Warps 0-7 also own the tokens of the element-wise phase (TPW tokens each); warp 8 solves the triangular system (right-looking, lane = column of T) while the others run the S products.
// The shared-memory tiles are 128 B-swizzled [token][channel] bf16 for ldmatrix; V^T / the C->A register reuse (FA2 trick) avoid every other transpose.
//
// Precision decisions (all measured, tools/tinytest): single bf16 operands everywhere (as the main kernels: FLA and FlashKDA round the same factors) EXCEPT T, which is bf16 hi + lo:
// with a single-bf16 T the rank-1 / beta~1 case (q = k = v shared by all tokens, the worst-conditioned (I + L)) lost a factor 3-6 against the main kernel (1.8e-2 vs 3.6e-3 relL2 at 32
// tokens); the hi/lo T restores 3.1e-3 / 2.9e-3.  An hi/lo split of the Akk operands, of S or of X / v_new changed nothing there.
// Limits: tasks of at most 32 tokens (host rule), lb = -5 (the |G| <= 160 chunk range), N <= 256 sequences, 32-byte aligned state.
//
// Launch modes: static (the layout is baked into the kernel parameters: no dependent cu_seqlens load) or dynamic (cu_seqlens read and validated on the device: it must start at 0, be
// non-decreasing and end at <= T; otherwise every sequence is empty: output zero-filled, state passes through, status = 1).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

// switch to the tensors' device only when it differs from the current one (one cudaGetDevice on the hot path; same helper as the other bindings)
struct KdaDevGuard {
  c10::cuda::OptionalCUDAGuard g_;
  explicit KdaDevGuard(const at::Tensor& t) { int c = -1; cudaGetDevice(&c); if (c != (int)t.get_device()) g_.set_index((c10::DeviceIndex)t.get_device()); }
};

#ifndef SPLIT_S
#define SPLIT_S 1     // 2: S operand of the S products as bf16 hi + lo (fp32-class), 1: single bf16
#endif
#ifndef SPLIT_V
#define SPLIT_V 1     // 2: X / v_new operands as bf16 hi + lo, 1: single bf16
#endif

namespace {
constexpr int NWARP = 9, NTHR = NWARP * 32;   // warps 0-7: tokens / value rows 16w..; warp 8: triangular solve
constexpr int O_KA = 0, O_KB = 8192, O_QA = 16384, O_KG = 24576, O_QG = 32768, O_KD = 40960, O_V = 49152;
constexpr int O_AQ = 57344, O_TM = 59904, O_LM = 62464, O_WT = 67072, O_DEC = 71168, O_BETA = 71680, O_OUT = 71808;
constexpr int O_TL = O_OUT + 8 * 32 * 48;   // low half of T (bf16 hi + lo: T enters v_new = T X; a single bf16 T loses accuracy on near-singular (I + L), see NOTES)
constexpr int SMEM_BYTES = O_TL + 32 * 80;
constexpr int LP = 36;        // fp32 pitch of the L matrix (elements)
constexpr int MP = 80;        // byte pitch of the 32x32 bf16 matrices (Aqk, T): conflict-free ldmatrix

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float bf2f(uint32_t u16) { return __uint_as_float(u16 << 16); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// swizzled [32 rows][128 bf16] tile: 16-byte chunk index xor (row & 7)
__device__ __forceinline__ uint32_t tile_addr(uint32_t base, int row, int chunk) { return base + row * 256 + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ void split_pair(float x0, float x1, uint32_t& hi, uint32_t& lo) {
  hi = pack2(x0, x1);
  const float h0 = __uint_as_float(hi << 16), h1 = __uint_as_float(hi & 0xffff0000u);
  lo = pack2(x0 - h0, x1 - h1);
}
#define EXPF(x) __expf(x)
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + EXPF(-x)); }
__device__ __forceinline__ uint2 ldg_u2(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// State: 32 B per lane (ld.global.nc.v8.f32).  Lane t4 holds channels 32sb + 8t4 + 0..7 of 32-block sb; accumulator tile j = 4sb + jj <- channels
// 32sb + 8t4 + 2jj + {0,1}, i.e. MMA position k' = 16(m>>2) + 8((m>>1)&1) + 2t4 + (m&1) <-> channel 8t4 + m inside a block (the K-axis tiles are written with the
// same permutation).
__device__ __forceinline__ void ld_v8(const float* p, float (&x)[8]) {
  asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]), "=f"(x[7]) : "l"(p));
}
__device__ __forceinline__ void st_v8(float* p, const float (&x)[8]) {
  asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(p), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
}
__device__ __forceinline__ void load_state(float (&S)[16][4], const float* __restrict__ s0, const int task, const int tid) {
  const int w = tid >> 5, l = tid & 31, g8 = l >> 2, t4 = l & 3;
  if (w < 8) {
    const float* p = s0 + (size_t)task * 16384 + (16 * w + g8) * 128 + 8 * t4;
#pragma unroll
    for (int sb = 0; sb < 4; sb++) {
      float x[8], y[8];
      ld_v8(p + 32 * sb, x); ld_v8(p + 1024 + 32 * sb, y);
#pragma unroll
      for (int jj = 0; jj < 4; jj++) { S[4 * sb + jj][0] = x[2 * jj]; S[4 * sb + jj][1] = x[2 * jj + 1]; S[4 * sb + jj][2] = y[2 * jj]; S[4 * sb + jj][3] = y[2 * jj + 1]; }
    }
  }
}
__device__ __forceinline__ void store_state(float (&S)[16][4], float* __restrict__ s1, const int task, const int tid) {
  const int w = tid >> 5, l = tid & 31, g8 = l >> 2, t4 = l & 3;
  if (w < 8) {
    float* p = s1 + (size_t)task * 16384 + (16 * w + g8) * 128 + 8 * t4;
#pragma unroll
    for (int sb = 0; sb < 4; sb++) {
      float x[8], y[8];
#pragma unroll
      for (int jj = 0; jj < 4; jj++) { x[2 * jj] = S[4 * sb + jj][0]; x[2 * jj + 1] = S[4 * sb + jj][1]; y[2 * jj] = S[4 * sb + jj][2]; y[2 * jj + 1] = S[4 * sb + jj][3]; }
      st_v8(p + 32 * sb, x); st_v8(p + 1024 + 32 * sb, y);
    }
  }
}

constexpr int MAXSEQ = 256;
struct SeqTab { int bos[MAXSEQ]; int len[MAXSEQ]; };   // static launches: layout baked into the kernel parameters (no dependent cu_seqlens load)

// NTA = active token n8-tiles (2: tokens <= 16, 4: tokens <= 32)
template <int NTA>
__device__ __forceinline__ void issue_loads(unsigned char* smem, const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, const bf16* __restrict__ g,
                                            const bf16* __restrict__ beta, uint2 (&rq)[NTA], uint2 (&rk)[NTA], uint2 (&rg)[NTA], uint2 (&rv)[NTA], const long long bos,
                                            const int nv, const int head, const int H, const int tid) {
  constexpr int TPW = NTA;               // tokens per warp in the element-wise phase (8 warps)
  const int w = tid >> 5, l = tid & 31;
  if (w < 8) {
#pragma unroll
    for (int u = 0; u < TPW; u++) {
      const int tk = TPW * w + u;
      if (tk < nv) {
        const size_t row = ((size_t)(bos + tk) * H + head) * 128 + 4 * l;
        rq[u] = ldg_u2(q + row); rk[u] = ldg_u2(k + row); rg[u] = ldg_u2(g + row); rv[u] = ldg_u2(v + row);
      } else {
        rq[u] = rk[u] = rg[u] = rv[u] = make_uint2(0u, 0u);
      }
    }
    if (w == 0) {
      float b = 0.f;
      if (l < nv) b = sigmoidf_(bf2f(*reinterpret_cast<const unsigned short*>(beta + (size_t)(bos + l) * H + head)));
      reinterpret_cast<float*>(smem + O_BETA)[l] = b;
    }
  }
}

template <int NTA>
__device__ __forceinline__ void body(unsigned char* smem, const uint32_t sb, float (&S)[16][4], uint2 (&rq)[NTA], uint2 (&rk)[NTA], uint2 (&rg)[NTA], uint2 (&rv)[NTA],
                                     const float expA, const float4 dtb, const float* __restrict__ s0, float* __restrict__ s1, const int task, bf16* __restrict__ out, const long long bos,
                                     const int nv, const int head, const int H, const float scale, const float lb, const int tid,
                                     const bf16* __restrict__ pq, const bf16* __restrict__ pk, const bf16* __restrict__ pg, const bf16* __restrict__ pv, const long long pf_bos, const int pf_nv, const int pf_head, const int pf_task) {
  constexpr int TPW = NTA;
  constexpr int NKS = NTA / 2;           // token k16 steps
  constexpr int NPRODUCER = (NTA == 4) ? 6 : 2;   // warps producing Akk (the solver waits for them)
  const int w = tid >> 5, l = tid & 31, g8 = l >> 2, t4 = l & 3;
  if (w < 8) {
    { uint32_t dm = rg[TPW - 1].y ^ rk[TPW - 1].y ^ rq[TPW - 1].y ^ rv[TPW - 1].y; asm volatile("" :: "r"(dm)); }
    load_state(S, s0, task, tid);   // issued once the inputs have arrived (the state stream would otherwise delay them)
    // ---- gate prefix
    float G[TPW][4], run[4] = {0.f, 0.f, 0.f, 0.f};
    const float dtv[4] = {dtb.x, dtb.y, dtb.z, dtb.w};
#pragma unroll
    for (int u = 0; u < TPW; u++) {
      const bool valid = TPW * w + u < nv;
      const uint32_t g01 = rg[u].x, g23 = rg[u].y;
      const float gv[4] = {__uint_as_float(g01 << 16), __uint_as_float(g01 & 0xffff0000u), __uint_as_float(g23 << 16), __uint_as_float(g23 & 0xffff0000u)};
#pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        const float a = valid ? lb * sigmoidf_(expA * (gv[ch] + dtv[ch])) : 0.f;
        run[ch] += a;
        G[u][ch] = run[ch];
      }
    }
    *reinterpret_cast<float4*>(smem + O_WT + (w * 128 + 4 * l) * 4) = make_float4(run[0], run[1], run[2], run[3]);
    // ---- q / k norms
    float kinv[TPW], qinv[TPW];
#pragma unroll
    for (int u = 0; u < TPW; u++) {
      float sk = 0.f, sq = 0.f;
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        const uint32_t kw = h2 ? rk[u].y : rk[u].x, qw = h2 ? rq[u].y : rq[u].x;
        const float k0 = __uint_as_float(kw << 16), k1 = __uint_as_float(kw & 0xffff0000u);
        const float q0 = __uint_as_float(qw << 16), q1 = __uint_as_float(qw & 0xffff0000u);
        const float k0s = k0 * 2.9103830456733704e-11f, k1s = k1 * 2.9103830456733704e-11f, q0s = q0 * 2.9103830456733704e-11f, q1s = q1 * 2.9103830456733704e-11f;   // [d1] exact 2^-35 prescale of the sums
        sk += k0s * k0s + k1s * k1s; sq += q0s * q0s + q1s * q1s;
      }
      kinv[u] = sk; qinv[u] = sq;
    }
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
#pragma unroll
      for (int u = 0; u < TPW; u++) {
        kinv[u] += __shfl_xor_sync(0xffffffffu, kinv[u], off);
        qinv[u] += __shfl_xor_sync(0xffffffffu, qinv[u], off);
      }
#pragma unroll
    for (int u = 0; u < TPW; u++) {   // [d1] 1 / max(|row|, 1e-6) from the prescaled sums; NaN above the window (z = s * -0)
      float zk_, zq_; asm("mul.rn.ftz.f32 %0, %1, 0f80000000;" : "=f"(zk_) : "f"(kinv[u])); asm("mul.rn.ftz.f32 %0, %1, 0f80000000;" : "=f"(zq_) : "f"(qinv[u]));
      kinv[u] = fmaf(rsqrtf(fmaxf(kinv[u], 8.47032947e-34f)), 2.9103830456733704e-11f, zk_); qinv[u] = fmaf(rsqrtf(fmaxf(qinv[u], 8.47032947e-34f)), 2.9103830456733704e-11f, zq_); }
    bar_sync(3, 256);   // warp totals / betas visible (warps 0-7)
    float off[4] = {0.f, 0.f, 0.f, 0.f}, tot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int w2 = 0; w2 < 8; w2++) {
      const float4 x = *reinterpret_cast<const float4*>(smem + O_WT + (w2 * 128 + 4 * l) * 4);
      tot[0] += x.x; tot[1] += x.y; tot[2] += x.z; tot[3] += x.w;
      if (w2 < w) { off[0] += x.x; off[1] += x.y; off[2] += x.z; off[3] += x.w; }
    }
    float cd[4], ec[4];
#pragma unroll
    for (int ch = 0; ch < 4; ch++) { cd[ch] = 0.5f * tot[ch]; ec[ch] = EXPF(cd[ch]); }
    if (w == 0) *reinterpret_cast<float4*>(smem + O_DEC + 16 * l) = make_float4(EXPF(tot[0]), EXPF(tot[1]), EXPF(tot[2]), EXPF(tot[3]));
#pragma unroll
    for (int u = 0; u < TPW; u++) {
      const int tk = TPW * w + u;
      const float kk[4] = {__uint_as_float(rk[u].x << 16) * kinv[u], __uint_as_float(rk[u].x & 0xffff0000u) * kinv[u],
                           __uint_as_float(rk[u].y << 16) * kinv[u], __uint_as_float(rk[u].y & 0xffff0000u) * kinv[u]};
      const float qq[4] = {__uint_as_float(rq[u].x << 16) * qinv[u], __uint_as_float(rq[u].x & 0xffff0000u) * qinv[u],
                           __uint_as_float(rq[u].y << 16) * qinv[u], __uint_as_float(rq[u].y & 0xffff0000u) * qinv[u]};
      float ka[4], kb[4], qa[4], kg[4], qg[4], kd[4];
#pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        const float xp = off[ch] + G[u][ch] - cd[ch];
        const float ep = EXPF(xp), em = EXPF(-xp);
        ka[ch] = kk[ch] * ep; kb[ch] = kk[ch] * em; qa[ch] = qq[ch] * ep;
        kg[ch] = ka[ch] * ec[ch]; qg[ch] = qa[ch] * ec[ch]; kd[ch] = kb[ch] * ec[ch];
      }
      // K-axis tiles: channel 4l + i goes to column 32(l>>3) + 16(l&1) + 8(i>>1) + 2((l>>1)&3) + (i&1), see load_state
      const int c0 = 4 * (l >> 3) + 2 * (l & 1);
      const uint32_t p0 = tk * 256 + ((c0 ^ (tk & 7)) << 4) + ((l >> 1) & 3) * 4, p1 = tk * 256 + (((c0 + 1) ^ (tk & 7)) << 4) + ((l >> 1) & 3) * 4;
      *reinterpret_cast<uint32_t*>(smem + O_KA + p0) = pack2(ka[0], ka[1]); *reinterpret_cast<uint32_t*>(smem + O_KA + p1) = pack2(ka[2], ka[3]);
      *reinterpret_cast<uint32_t*>(smem + O_KB + p0) = pack2(kb[0], kb[1]); *reinterpret_cast<uint32_t*>(smem + O_KB + p1) = pack2(kb[2], kb[3]);
      *reinterpret_cast<uint32_t*>(smem + O_QA + p0) = pack2(qa[0], qa[1]); *reinterpret_cast<uint32_t*>(smem + O_QA + p1) = pack2(qa[2], qa[3]);
      *reinterpret_cast<uint32_t*>(smem + O_KG + p0) = pack2(kg[0], kg[1]); *reinterpret_cast<uint32_t*>(smem + O_KG + p1) = pack2(kg[2], kg[3]);
      *reinterpret_cast<uint32_t*>(smem + O_QG + p0) = pack2(qg[0], qg[1]); *reinterpret_cast<uint32_t*>(smem + O_QG + p1) = pack2(qg[2], qg[3]);
      *reinterpret_cast<uint32_t*>(smem + O_KD + p0) = pack2(kd[0], kd[1]); *reinterpret_cast<uint32_t*>(smem + O_KD + p1) = pack2(kd[2], kd[3]);
      *reinterpret_cast<uint2*>(smem + O_V + tk * 256 + (((l >> 1) ^ (tk & 7)) << 4) + (l & 1) * 8) = rv[u];
    }
  }
  __syncthreads();   // tiles complete (all 9 warps)
  if (w == 8) {
#ifndef NO_WAVE_PF
    if (pf_nv > 0) {   // [shortmid] next wave's task on this SM: pull its inputs and state into L2 now (this wave's inputs have arrived), so that the next CTA's cold loads are L2 hits
      for (int i = l; i < pf_nv * 8; i += 32) {
        const int tk = i >> 3, tn = (i >> 1) & 3;
        const bf16* base = tn == 0 ? pq : tn == 1 ? pk : tn == 2 ? pg : pv;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(base + ((size_t)(pf_bos + tk) * H + pf_head) * 128 + (i & 1) * 64) : "memory");
      }
      if (l == 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(s0 + (size_t)pf_task * 16384), "r"(65536) : "memory");
    }
#endif
    // ---- triangular inverse T = (I + L)^-1 (lane = column of T), right-looking: once t_j is final, every later row's accumulator is updated (independent FMAs); written as bf16 [i][j]
    bar_sync(1, (NPRODUCER + 1) * 32);
    constexpr int NR = NTA * 8;
    constexpr int PFD = (NTA == 2) ? 4 : 3;       // rows of L^T loaded ahead (shared-memory latency is far above the 4-cycle FMA dependency of the chain)
    constexpr int NG = NR / 4;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; i++) acc[i] = (i == l) ? 1.f : 0.f;
    float4 pf[PFD][NG];
#pragma unroll
    for (int jp = 0; jp < PFD; jp++)
#pragma unroll
      for (int i4 = (jp + 1) / 4; i4 < NG; i4++) pf[jp][i4] = *reinterpret_cast<const float4*>(smem + O_LM + (jp * LP + 4 * i4) * 4);
#pragma unroll
    for (int j = 0; j < NR; j++) {
      const float tj = acc[j];
      if (l < NR) reinterpret_cast<bf16*>(smem + O_TM + j * MP)[l] = __float2bfloat16_rn(tj);
      if (l < NR) reinterpret_cast<bf16*>(smem + O_TL + j * MP)[l] = __float2bfloat16_rn(tj - __bfloat162float(__float2bfloat16_rn(tj)));
#pragma unroll
      for (int i4 = (j + 1) / 4; i4 < NG; i4++) {
        const float4 lv = pf[j % PFD][i4];       // L^T[j][i] = L[i][j] (zero for i <= j)
        acc[4 * i4 + 0] = fmaf(-lv.x, tj, acc[4 * i4 + 0]); acc[4 * i4 + 1] = fmaf(-lv.y, tj, acc[4 * i4 + 1]);
        acc[4 * i4 + 2] = fmaf(-lv.z, tj, acc[4 * i4 + 2]); acc[4 * i4 + 3] = fmaf(-lv.w, tj, acc[4 * i4 + 3]);
      }
      if (j + PFD < NR) {
#pragma unroll
        for (int i4 = (j + PFD + 1) / 4; i4 < NG; i4++) pf[j % PFD][i4] = *reinterpret_cast<const float4*>(smem + O_LM + ((j + PFD) * LP + 4 * i4) * 4);
      }
    }
  }
  float O[NTA][4], X[NTA][4];   // live from the R/O products to the end
  if (w != 8) {
    float R[NTA][4];
  // ---- Akk -> L^T (producers; the solver waits for them), Aqk after that (NTA = 4: the same warps; NTA = 2: two more warps in parallel)
  {
    const float* sbeta = reinterpret_cast<const float*>(smem + O_BETA);
    auto tile = [&](const int base_a, const int mt, const int nt, float (&c)[4]) {
      float c1[4] = {0.f, 0.f, 0.f, 0.f};
      c[0] = c[1] = c[2] = c[3] = 0.f;
      uint32_t fa[8][4], fb[8][2];
      const int m = l >> 3, r = l & 7;
#pragma unroll
      for (int s = 0; s < 8; s++) {   // all fragment loads first: the shared-memory latency is paid once, not per k-step
        ldsm4(fa[s], tile_addr(sb + base_a, 16 * mt + 8 * (m & 1) + r, 2 * s + (m >> 1)));
        ldsm2(fb[s][0], fb[s][1], tile_addr(sb + O_KB, 8 * nt + r, 2 * s + ((l >> 3) & 1)));
      }
#pragma unroll
      for (int s = 0; s < 8; s++) { if (s & 1) mma(c1, fa[s], fb[s][0], fb[s][1]); else mma(c, fa[s], fb[s][0], fb[s][1]); }
#pragma unroll
      for (int e = 0; e < 4; e++) c[e] += c1[e];
    };
    if (NTA == 4 ? (w < 6) : (w < 2)) {
      const int mt = (NTA == 4) ? ((w == 0 || w == 5) ? 0 : 1) : 0;
      const int nt = (NTA == 4) ? ((w == 0) ? 0 : (w == 5) ? 1 : (w - 1)) : w;
      float ac[4];
      tile(O_KA, mt, nt, ac);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int i = 16 * mt + g8 + 8 * (e >> 1), j = 8 * nt + 2 * t4 + (e & 1);
        reinterpret_cast<float*>(smem + O_LM)[j * LP + i] = (i > j) ? sbeta[i] * ac[e] : 0.f;
      }
      bar_arrive(1, (NPRODUCER + 1) * 32);
    }
    if (NTA == 4 ? (w < 6) : (w >= 2 && w < 4)) {
      const int ww = (NTA == 4) ? w : w - 2;
      const int mt = (NTA == 4) ? ((ww == 0 || ww == 5) ? 0 : 1) : 0;
      const int nt = (NTA == 4) ? ((ww == 0) ? 0 : (ww == 5) ? 1 : (ww - 1)) : ww;
      float aq[4];
      tile(O_QA, mt, nt, aq);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int i = 16 * mt + g8 + 8 * (e >> 1), j = 8 * nt + 2 * t4 + (e & 1);
        reinterpret_cast<bf16*>(smem + O_AQ + i * MP)[j] = __float2bfloat16_rn((i >= j) ? aq[e] : 0.f);
      }
    }
  }
  { uint32_t dm = __float_as_uint(S[15][3]) ^ __float_as_uint(S[0][0]) ^ __float_as_uint(S[7][1]); asm volatile("" :: "r"(dm)); }
  // ---- R^T = S0 Kg^T, O = S0 Qg^T (A = S hi/lo from registers)
#pragma unroll
  for (int nt = 0; nt < NTA; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) { R[nt][e] = 0.f; O[nt][e] = 0.f; }
  constexpr int RB = (NTA == 2) ? 4 : 2;        // k-steps whose B fragments are loaded ahead of their MMAs
#pragma unroll
  for (int s0 = 0; s0 < 8; s0 += RB) {
    uint32_t bk[RB][NTA / 2][4], bq[RB][NTA / 2][4];
#pragma unroll
    for (int sr = 0; sr < RB; sr++)
#pragma unroll
      for (int np = 0; np < NTA / 2; np++) {
        const int m = l >> 3, r = l & 7;
        const int brow = 8 * (2 * np + (m >> 1)) + r, bchunk = 2 * (s0 + sr) + (m & 1);
        ldsm4(bk[sr][np], tile_addr(sb + O_KG, brow, bchunk));
        ldsm4(bq[sr][np], tile_addr(sb + O_QG, brow, bchunk));
      }
#pragma unroll
    for (int sr = 0; sr < RB; sr++) {
      const int s = s0 + sr;
      uint32_t ah[4], al[4];
      split_pair(S[2 * s][0], S[2 * s][1], ah[0], al[0]);
      split_pair(S[2 * s][2], S[2 * s][3], ah[1], al[1]);
      split_pair(S[2 * s + 1][0], S[2 * s + 1][1], ah[2], al[2]);
      split_pair(S[2 * s + 1][2], S[2 * s + 1][3], ah[3], al[3]);
#pragma unroll
      for (int np = 0; np < NTA / 2; np++)
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
          mma(R[2 * np + h2], ah, bk[sr][np][2 * h2], bk[sr][np][2 * h2 + 1]);
          mma(O[2 * np + h2], ah, bq[sr][np][2 * h2], bq[sr][np][2 * h2 + 1]);
#if SPLIT_S == 2
          mma(R[2 * np + h2], al, bk[sr][np][2 * h2], bk[sr][np][2 * h2 + 1]);
          mma(O[2 * np + h2], al, bq[sr][np][2 * h2], bq[sr][np][2 * h2 + 1]);
#endif
        }
    }
  }
  // ---- X = beta (V - R)  (V^T straight from the V tile with ldmatrix.trans)
  {
    const float* sbeta = reinterpret_cast<const float*>(smem + O_BETA);
#pragma unroll
    for (int np = 0; np < NTA / 2; np++) {
      const int m = l >> 3, r = l & 7;
      uint32_t vt[4];
      ldsm4t(vt, tile_addr(sb + O_V, 8 * (2 * np + (m >> 1)) + r, 2 * w + (m & 1)));
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        const int nt = 2 * np + h2;
        const float2 bt = *reinterpret_cast<const float2*>(sbeta + 8 * nt + 2 * t4);
        const uint32_t v0 = vt[2 * h2], v1 = vt[2 * h2 + 1];   // v0: rows g8 (h = 0), v1: rows g8 + 8
        X[nt][0] = bt.x * (__uint_as_float(v0 << 16) - R[nt][0]);
        X[nt][1] = bt.y * (__uint_as_float(v0 & 0xffff0000u) - R[nt][1]);
        X[nt][2] = bt.x * (__uint_as_float(v1 << 16) - R[nt][2]);
        X[nt][3] = bt.y * (__uint_as_float(v1 & 0xffff0000u) - R[nt][3]);
      }
    }
  }
  }
  bar_sync(2, NWARP * 32);   // T complete (one call site for all nine warps)
  if (w == 8) return;
  // ---- v_new^T = X^T T^T
  float Vn[NTA][4];
#pragma unroll
  for (int nt = 0; nt < NTA; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) Vn[nt][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < NKS; ks++) {
    uint32_t ah[4], al[4];
    split_pair(X[2 * ks][0], X[2 * ks][1], ah[0], al[0]);
    split_pair(X[2 * ks][2], X[2 * ks][3], ah[1], al[1]);
    split_pair(X[2 * ks + 1][0], X[2 * ks + 1][1], ah[2], al[2]);
    split_pair(X[2 * ks + 1][2], X[2 * ks + 1][3], ah[3], al[3]);
#pragma unroll
    for (int np = 0; np < NTA / 2; np++) {
      const int m = l >> 3, r = l & 7;
      uint32_t bt[4];
      ldsm4(bt, sb + O_TM + (8 * (2 * np + (m >> 1)) + r) * MP + (2 * ks + (m & 1)) * 16);
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        mma(Vn[2 * np + h2], ah, bt[2 * h2], bt[2 * h2 + 1]);
#if SPLIT_V == 2
        mma(Vn[2 * np + h2], al, bt[2 * h2], bt[2 * h2 + 1]);
#endif
      }
      uint32_t bl[4];
      ldsm4(bl, sb + O_TL + (8 * (2 * np + (m >> 1)) + r) * MP + (2 * ks + (m & 1)) * 16);
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) mma(Vn[2 * np + h2], ah, bl[2 * h2], bl[2 * h2 + 1]);
    }
  }
  // ---- O += Aqk v_new ; S update operand
  uint32_t vh[NKS][4], vl[NKS][4];
#pragma unroll
  for (int ks = 0; ks < NKS; ks++) {
    split_pair(Vn[2 * ks][0], Vn[2 * ks][1], vh[ks][0], vl[ks][0]);
    split_pair(Vn[2 * ks][2], Vn[2 * ks][3], vh[ks][1], vl[ks][1]);
    split_pair(Vn[2 * ks + 1][0], Vn[2 * ks + 1][1], vh[ks][2], vl[ks][2]);
    split_pair(Vn[2 * ks + 1][2], Vn[2 * ks + 1][3], vh[ks][3], vl[ks][3]);
#pragma unroll
    for (int np = 0; np < NTA / 2; np++) {
      const int m = l >> 3, r = l & 7;
      uint32_t bt[4];
      ldsm4(bt, sb + O_AQ + (8 * (2 * np + (m >> 1)) + r) * MP + (2 * ks + (m & 1)) * 16);
#pragma unroll
      for (int h2 = 0; h2 < 2; h2++) {
        mma(O[2 * np + h2], vh[ks], bt[2 * h2], bt[2 * h2 + 1]);
#if SPLIT_V == 2
        mma(O[2 * np + h2], vl[ks], bt[2 * h2], bt[2 * h2 + 1]);
#endif
      }
    }
  }
  // ---- output tile: bf16 through a warp-private smem tile [32 tok][16 v] (pitch 48 B), then 32-byte row segments
  {
    unsigned char* ow = smem + O_OUT + w * (32 * 48);
#pragma unroll
    for (int nt = 0; nt < NTA; nt++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int tok = 8 * nt + 2 * t4 + (e & 1), vv = g8 + 8 * (e >> 1);
        reinterpret_cast<bf16*>(ow + tok * 48)[vv] = __float2bfloat16_rn(O[nt][e] * scale);
      }
    __syncwarp();
    if (l < nv) {
      const uint4 a = *reinterpret_cast<const uint4*>(ow + l * 48), b = *reinterpret_cast<const uint4*>(ow + l * 48 + 16);
      bf16* dst = out + ((size_t)(bos + l) * H + head) * 128 + 16 * w;
      *reinterpret_cast<uint4*>(dst) = a;
      *reinterpret_cast<uint4*>(dst + 8) = b;
    }
  }
  // ---- state update S = S e^{G_C} + v_new^T kd
#pragma unroll
  for (int sup = 0; sup < 4; sup++) {
    uint32_t bd[4][4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) ldsm4t(bd[jj], tile_addr(sb + O_KD, 8 * (l >> 3) + (l & 7), 4 * sup + jj));
    float dc[8];
    *reinterpret_cast<float4*>(&dc[0]) = *reinterpret_cast<const float4*>(smem + O_DEC + (32 * sup + 8 * t4) * 4);
    *reinterpret_cast<float4*>(&dc[4]) = *reinterpret_cast<const float4*>(smem + O_DEC + (32 * sup + 8 * t4 + 4) * 4);
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      const int j = 4 * sup + jj;
      S[j][0] *= dc[2 * jj]; S[j][1] *= dc[2 * jj + 1]; S[j][2] *= dc[2 * jj]; S[j][3] *= dc[2 * jj + 1];
      mma(S[j], vh[0], bd[jj][0], bd[jj][1]);
#if SPLIT_V == 2
      mma(S[j], vl[0], bd[jj][0], bd[jj][1]);
#endif
      if (NKS == 2) {
        mma(S[j], vh[NKS - 1], bd[jj][2], bd[jj][3]);
#if SPLIT_V == 2
        mma(S[j], vl[NKS - 1], bd[jj][2], bd[jj][3]);
#endif
      }
    }
    {   // this 32-channel block of the final state is complete: store it while the next blocks are still being updated
      float* p = s1 + (size_t)task * 16384 + (16 * w + g8) * 128 + 8 * t4 + 32 * sup;
      float x[8], y[8];
#pragma unroll
      for (int jj = 0; jj < 4; jj++) { x[2 * jj] = S[4 * sup + jj][0]; x[2 * jj + 1] = S[4 * sup + jj][1]; y[2 * jj] = S[4 * sup + jj][2]; y[2 * jj + 1] = S[4 * sup + jj][3]; }
      st_v8(p, x); st_v8(p + 1024, y);
    }
  }
}

__global__ void __launch_bounds__(NTHR, 1)
kda_tiny_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, const bf16* __restrict__ g,
                const bf16* __restrict__ beta, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                const long long* __restrict__ cu, const float* __restrict__ s0, bf16* __restrict__ out, float* __restrict__ s1,
                int H, float scale, float lb, SeqTab tab, int use_tab, int nseq, int T, int* __restrict__ pstat, int wave_n) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  const int task = blockIdx.x;
  const int seq = task / H, head = task - seq * H;
  long long bos; int L;
  bool dyn_ok = true;
  if (use_tab) { bos = tab.bos[seq]; L = tab.len[seq]; }
  else {
    // dynamic layout: cu_seqlens is read on the device.  The addresses are clamped so that the speculative loads below stay inside the tensors whatever the table holds;
    // the table itself is validated by the whole CTA (entry pair (tid, tid + 1)) before anything is written: it must start at 0, be non-decreasing and end at <= T.
    const long long c0 = cu[seq], c1 = cu[seq + 1];
    const long long b0 = min(max(c0, 0ll), (long long)T);
    bos = b0;
    L = (int)min(min(max(c1 - c0, 0ll), 32ll), (long long)T - b0);
    bool ok = true;
    if (tid < nseq) {
      const long long a = cu[tid], b = cu[tid + 1];
      ok = a <= b && (tid != 0 || a == 0) && (tid != nseq - 1 || b <= (long long)T);
    }
    dyn_ok = __syncthreads_and(ok) != 0;
  }
  int nv = min(L, 32);
  float S[16][4];
  uint2 rq4[4], rk4[4], rg4[4], rv4[4], rq2[2], rk2[2], rg2[2], rv2[2];
  if (!use_tab && !dyn_ok) nv = 0;
  if (nv > 0) {
    if (nv <= 16) issue_loads<2>(smem, q, k, v, g, beta, rq2, rk2, rg2, rv2, bos, nv, head, H, tid);
    else issue_loads<4>(smem, q, k, v, g, beta, rq4, rk4, rg4, rv4, bos, nv, head, H, tid);
  } else {
    load_state(S, s0, task, tid);   // empty sequence: the state passes through
  }
  if (!use_tab && !dyn_ok) {   // rejected table: every sequence is empty, the output is zero-filled
    for (int i = tid; i < T * 16; i += NTHR) *reinterpret_cast<uint4*>(out + ((size_t)(i >> 4) * H + head) * 128 + (i & 15) * 8) = make_uint4(0u, 0u, 0u, 0u);
  }
  if (!use_tab && pstat && task == 0 && tid == 0) pstat[0] = dyn_ok ? 0 : 1;
  const float expA = __expf(A_log[head]);
  const float4 dtb = *reinterpret_cast<const float4*>(dt_bias + head * 128 + 4 * l);
  // zero the matrices whose upper triangle is never written (Aqk, T)
  for (int i = tid; i < 2 * 32 * MP / 4; i += NTHR) reinterpret_cast<uint32_t*>(smem + O_AQ)[i] = 0u;   // Aqk and T are adjacent (O_TM = O_AQ + 2560)
  long long pf_bos = 0; int pf_nv = 0, pf_head = 0, pf_task = 0;
  if (use_tab && wave_n > 0 && w == 8 && task + wave_n < (int)gridDim.x) {
    pf_task = task + wave_n;
    const int sq2 = pf_task / H;
    pf_head = pf_task - sq2 * H; pf_bos = tab.bos[sq2]; pf_nv = min(tab.len[sq2], 32);
  }
  if (nv > 0) {
    if (nv <= 16) body<2>(smem, su32(smem), S, rq2, rk2, rg2, rv2, expA, dtb, s0, s1, task, out, bos, nv, head, H, scale, lb, tid, q, k, g, v, pf_bos, pf_nv, pf_head, pf_task);
    else body<4>(smem, su32(smem), S, rq4, rk4, rg4, rv4, expA, dtb, s0, s1, task, out, bos, nv, head, H, scale, lb, tid, q, k, g, v, pf_bos, pf_nv, pf_head, pf_task);
  }
  if (nv == 0) store_state(S, s1, task, tid);   // empty sequence: the state passes through (the chunk path stores its blocks itself)
}
}  // namespace

void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor A_log, torch::Tensor dt_bias,
            torch::Tensor cu, torch::Tensor s0, torch::Tensor out, torch::Tensor s1, int64_t ntasks, int64_t H, double scale, double lb,
            std::vector<int64_t> bos, std::vector<int64_t> len, int64_t nseq, int64_t T, torch::Tensor pstat, int64_t wave_n) {
  KdaDevGuard dev_guard_(q);   // stream, cudaGetDevice() (per-device attribute flag) and the launch belong to the tensors' device
  static bool attr[64] = {};
  int dev = 0; cudaGetDevice(&dev);
  if (!attr[dev & 63]) { cudaFuncSetAttribute(kda_tiny_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); attr[dev & 63] = true; }
  SeqTab tab;
  const int use_tab = !bos.empty();
  if (use_tab) {
    TORCH_CHECK(bos.size() <= (size_t)MAXSEQ && bos.size() == len.size() && (int64_t)bos.size() * H == ntasks, "kda_tiny: bad sequence table");
    for (size_t i = 0; i < bos.size(); i++) {
      TORCH_CHECK(len[i] >= 0 && len[i] <= 32 && bos[i] >= 0 && bos[i] + len[i] <= T, "kda_tiny: sequence longer than one chunk or outside the tokens");
      tab.bos[i] = (int)bos[i]; tab.len[i] = (int)len[i];
    }
  } else {
    TORCH_CHECK(T <= 32 && nseq <= MAXSEQ && nseq * H == ntasks && cu.numel() >= nseq + 1, "kda_tiny: a dynamic launch needs T <= 32 and at most 256 sequences");
  }
  kda_tiny_kernel<<<(unsigned)ntasks, NTHR, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(), (const bf16*)g.data_ptr(), (const bf16*)beta.data_ptr(),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), (const long long*)cu.data_ptr<int64_t>(), s0.data_ptr<float>(), (bf16*)out.data_ptr(),
      s1.data_ptr<float>(), (int)H, (float)scale, (float)lb, tab, use_tab, (int)nseq, (int)T, pstat.numel() ? pstat.data_ptr<int>() : nullptr, (int)wave_n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
}
