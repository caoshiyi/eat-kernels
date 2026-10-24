// [contract2] device-side table validation (shared by contract_aux.cu's standalone kernel and the dynamic planner kernel, dyn/plan_cp.cu).  See contract_aux.cu.
#pragma once
#include <torch/extension.h>
namespace kdaval {
constexpr int kThreads = 1024;

__device__ __forceinline__ long long rd(const void* p, int is64, int i) { return is64 ? ((const long long*)p)[i] : (long long)((const int*)p)[i]; }

struct VArgs {
  int N;                  // sequences
  long long T;            // tokens
  const void* cu; int cu_is64; long long* cu_out;          // cu may be null (one sequence, [0, T])
  const void* d; int d_is64; int* d_out; long long P_d; int d_trash;   // d null: no destination table
  const void* s; int s_is64; int* s_out; long long P_s; int s_trash;   // s null: no source table;  s_out == null with d != null: source table aliases the destination (in place)
  const void* ck; int ck_is64; long long* ck_out; long long C; long long ck_tokens;   // ck null: no checkpoints
  int* status;
  void* out; unsigned long long out_bytes;   // [dynpath2] optional: output buffer zero-filled by the planner's validation block when cu_seqlens is rejected (never left uninitialised)
  const unsigned short* pad_src; unsigned short* pad_dst; long long pad_n; int pad_H, pad_Hp, pad_blocks;   // [long] optional beta padding copy (H % 8 != 0): dst[t * Hp + h] = src[t * H + h], done by extra blocks of the planner / validation launch
};

constexpr int kHashMaxN = 2048;   // repeated-slot detection by an in-shared-memory hash table up to this many sequences (larger N: O(N^2) scan of global memory)

__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

// rows a sequence claims in the checkpoint buffer: ceil(len / ck_tokens), len from the sanitized cu_seqlens when there is one
__device__ __forceinline__ long long ck_cnt(const VArgs& a, int i) {
  const long long len = a.cu == nullptr ? a.T : (a.cu_out != nullptr ? a.cu_out[i + 1] - a.cu_out[i] : max(0LL, rd(a.cu, a.cu_is64, i + 1) - rd(a.cu, a.cu_is64, i)));
  return (len + a.ck_tokens - 1) / a.ck_tokens;
}

__device__ void validate_block(const VArgs& a, int* hs, bool noop_cu = false) {   // noop_cu [pdl]: a rejected cu_seqlens makes cu_out all zero (every sequence empty, as the lean planner) instead of the clamped copy   // hs: key[HS] | first[HS] (shared memory, 8 * HS bytes, HS = pow2 >= 2N, >= 64), blockDim.x threads
  __shared__ int s_bits;
  const int N = a.N, bd = blockDim.x;
  const bool hashed = a.d != nullptr && N <= kHashMaxN;
  int HS = 64; while (HS < 2 * N) HS <<= 1;
  int* key = hs; int* first = hs + HS;
  if (threadIdx.x == 0) s_bits = 0;
  if (hashed) for (int i = threadIdx.x; i < HS; i += bd) { key[i] = -1; first[i] = 0x7fffffff; }
  __syncthreads();
  // ---- pass 1: every table entry is loaded up front (one memory latency), checked locally and written to its sanitized slot
  bool cu_ok = true;
  long long len_r = 0;
  for (int i = threadIdx.x; i < N + 1; i += bd) {
    if (a.cu != nullptr) {
      const long long c0 = rd(a.cu, a.cu_is64, i);
      const long long c1 = i < N ? rd(a.cu, a.cu_is64, i + 1) : c0;
      if (c0 < 0 || c0 > a.T || c1 < c0 || (i == 0 && c0 != 0)) cu_ok = false;
      if (a.cu_out != nullptr) a.cu_out[i] = c0;
      len_r = c1 < c0 ? 0 : c1 - c0;
    }
    if (i < N) {
      if (a.d != nullptr) {
        const long long v = rd(a.d, a.d_is64, i);
        const bool ok = v >= 0 && v < a.P_d;
        if (!ok) atomicOr(&s_bits, 2);
        a.d_out[i] = ok ? (int)v : a.d_trash;
        if (ok && hashed) {
          unsigned h = hash32((unsigned)v) & (HS - 1);
          for (;;) {
            const int old = atomicCAS(&key[h], -1, (int)v);
            if (old == -1 || old == (int)v) { atomicMin(&first[h], i); break; }
            h = (h + 1) & (HS - 1);
          }
        }
      }
      if (a.s != nullptr && a.s_out != nullptr) {
        const long long v = rd(a.s, a.s_is64, i);
        const bool ok = v >= 0 && v < a.P_s;
        if (!ok) atomicOr(&s_bits, 4);
        a.s_out[i] = ok ? (int)v : a.s_trash;
      }
    }
  }
  if (!cu_ok) atomicOr(&s_bits, 1);
  __syncthreads();
  // ---- pass 2: repeated destination slots (the 2nd and later occurrence go to the trash window)
  if (a.d != nullptr) {
    for (int i = threadIdx.x; i < N; i += bd) {
      if (a.d_out[i] == a.d_trash) continue;      // already rejected (out of range)
      const int v = a.d_out[i];
      bool dup = false;
      if (hashed) {
        unsigned h = hash32((unsigned)v) & (HS - 1);
        while (key[h] != v) h = (h + 1) & (HS - 1);
        dup = first[h] != i;
      } else {
        for (int j = 0; j < i && !dup; j++) dup = a.d_out[j] == v;     // first occurrences keep their slot, so comparing sanitized values is exact
      }
      if (dup) { a.d_out[i] = a.d_trash; atomicOr(&s_bits, 2); }
    }
  }
  // ---- rare path: a broken cu_seqlens is clamped to a non-decreasing sequence inside [0, T] (running maximum, serial)
  if ((s_bits & 1) && a.cu_out != nullptr) {
    if (threadIdx.x == 0) {
      long long run = 0;
      for (int i = 0; i < N + 1; i++) {
        long long c = rd(a.cu, a.cu_is64, i);
        c = noop_cu ? 0 : (c < 0 ? 0 : (c > a.T ? a.T : c));
        run = c > run ? c : run;
        a.cu_out[i] = run;
      }
    }
  }
  __syncthreads();
  // ---- checkpoint rows
  if (a.ck != nullptr) {
    const bool regs = (a.cu_out == nullptr || !(s_bits & 1)) && N + 1 <= bd;
    for (int i = threadIdx.x; i < N; i += bd) {
      const long long v = rd(a.ck, a.ck_is64, i);
      const long long len = a.cu == nullptr ? a.T : (regs ? len_r : (a.cu_out != nullptr ? a.cu_out[i + 1] - a.cu_out[i] : max(0LL, rd(a.cu, a.cu_is64, i + 1) - rd(a.cu, a.cu_is64, i))));
      const long long cnt = (len + a.ck_tokens - 1) / a.ck_tokens;
      const bool ok = v >= 0 && cnt <= a.C && v <= a.C - cnt;   // [portfolio audit F1] no overflow of v + cnt for a caller-supplied v near INT64_MAX
      if (!ok) atomicOr(&s_bits, 8);
      a.ck_out[i] = ok ? v : -1;
    }
    if (threadIdx.x == 0) a.ck_out[N] = 0;   // entry N is never read by a kernel
    __syncthreads();
    // ---- [tiny audit F2] row ranges of different sequences must be disjoint (two sequences writing one row = a race the contract does not define).  Policy = repeated destination slots: the first claim
    // keeps its rows, a sequence whose rows [v, v + cnt) overlap the rows of an EARLIER sequence gets none (row -1, status bit 8).  Fast path: the usual prefix-sum table (every entry valid and
    // v[i + 1] >= v[i] + cnt[i]) is disjoint by construction; the exact scan below only runs for a table that is out of order or already has a rejected entry.
    __shared__ int s_slow;
    if (threadIdx.x == 0) s_slow = (s_bits & 8) ? 1 : 0;
    __syncthreads();
    for (int i = threadIdx.x; i + 1 < N; i += bd) {
      const long long v0 = a.ck_out[i], v1 = a.ck_out[i + 1];
      if (v0 >= 0 && v1 >= 0 && v1 < v0 + ck_cnt(a, i)) s_slow = 1;
    }
    __syncthreads();
    if (s_slow) {
      for (int i = 1; i < N; i++) {   // ascending: entries j < i already hold their final value (-1 = no rows)
        const long long vi = a.ck_out[i];
        bool ov = false;
        if (vi >= 0) {
          const long long ci = ck_cnt(a, i);
          if (ci > 0) for (int j = threadIdx.x; j < i; j += bd) {
            const long long vj = a.ck_out[j];
            if (vj >= 0 && vj < vi + ci && vi < vj + ck_cnt(a, j)) ov = true;
          }
        }
        ov = __syncthreads_or(ov);
        if (ov && threadIdx.x == 0) { a.ck_out[i] = -1; atomicOr(&s_bits, 8); }
        __syncthreads();
      }
    }
  }
  if (threadIdx.x == 0) *a.status = s_bits;
}

// [pdl] latency-optimised validation for the usual case (N + 1 <= blockDim, N <= kHashMaxN, a well-formed launch): every table entry is loaded in ONE batch by the thread that owns it, the checks run on registers / a shared
// hash, and the outputs are written once.  Any anomaly (bad cu_seqlens, slot out of range or repeated, checkpoint rows that do not fit or overlap) falls back to validate_block_slow, which redoes everything
// (same outputs and status word as before).  The slow version's serial chain of dependent global round trips (cu / slots, checkpoint table, re-reads of its own outputs) cost ~5 us per launch.
__device__ bool validate_block_fast(const VArgs& a, int* hs, bool noop_cu) {   // true: the fast path accepted every table (a well-formed cu_seqlens in particular)
  const int N = a.N, bd = blockDim.x, i = threadIdx.x;
  if (!(N + 1 <= bd && N <= kHashMaxN && a.ck_tokens > 0)) { validate_block(a, hs, noop_cu); return false; }
  int HS = 64; while (HS < 2 * N) HS <<= 1;
  int* key = hs;
  const bool hashed = a.d != nullptr;
  // ---- one batch of loads (all independent)
  long long c0 = 0, c1 = 0, dv = 0, sv = 0, kv = 0;
  if (a.cu != nullptr && i <= N) { c0 = rd(a.cu, a.cu_is64, i); c1 = i < N ? rd(a.cu, a.cu_is64, i + 1) : c0; }
  if (i < N) {
    if (a.d != nullptr) dv = rd(a.d, a.d_is64, i);
    if (a.s != nullptr && a.s_out != nullptr) sv = rd(a.s, a.s_is64, i);
    if (a.ck != nullptr) kv = rd(a.ck, a.ck_is64, i);
  }
  if (hashed) for (int j = i; j < HS; j += bd) key[j] = -1;
  long long* s_ck = (long long*)(hs + HS);   // the checkpoint rows, for the neighbour (disjointness) check (the hash's second half: 4 HS bytes >= 8 N)
  __syncthreads();
  int anom = 0;
  if (a.cu != nullptr && i <= N && (c0 < 0 || c0 > a.T || c1 < c0 || (i == 0 && c0 != 0))) anom = 1;
  const long long len = a.cu == nullptr ? a.T : (c1 - c0);
  if (i < N) {
    if (a.d != nullptr) {
      if (!(dv >= 0 && dv < a.P_d)) anom = 1;
      else {
        unsigned h = hash32((unsigned)dv) & (HS - 1);
        for (;;) {
          const int old = atomicCAS(&key[h], -1, (int)dv);
          if (old == -1) break;
          if (old == (int)dv) { anom = 1; break; }
          h = (h + 1) & (HS - 1);
        }
      }
    }
    if (a.s != nullptr && a.s_out != nullptr && !(sv >= 0 && sv < a.P_s)) anom = 1;
    if (a.ck != nullptr) {
      const long long cnt = (len + a.ck_tokens - 1) / a.ck_tokens;
      if (!(kv >= 0 && cnt <= a.C && kv <= a.C - cnt)) anom = 1;
      s_ck[i] = kv;
    }
  }
  __syncthreads();
  if (a.ck != nullptr && i + 1 < N) {   // rows of consecutive sequences must be disjoint and in order (prefix-sum table); anything else takes the exact scan of the slow version
    const long long cnt = (len + a.ck_tokens - 1) / a.ck_tokens;
    if (s_ck[i + 1] < kv + cnt) anom = 1;
  }
  const int bad = __syncthreads_or(anom);
  if (bad) { validate_block(a, hs, noop_cu); return false; }
  if (a.cu != nullptr && a.cu_out != nullptr && i <= N) a.cu_out[i] = c0;
  if (i < N) {
    if (a.d != nullptr) a.d_out[i] = (int)dv;
    if (a.s != nullptr && a.s_out != nullptr) a.s_out[i] = (int)sv;
    if (a.ck != nullptr) a.ck_out[i] = kv;
  }
  if (a.ck != nullptr && i == N) a.ck_out[N] = 0;
  if (i == 0) *a.status = 0;
  return true;
}
// [chain] the same fast validation as a separate (non-inlined) function for the planner kernels: inlined into a planner kernel it raised the register allocation / changed the schedule of the planner block
// (block 0 of the same kernel: 5.0 k -> 8.2 k cycles); as its own function the planner block is unaffected and the validation block (the longest block of a lean planner launch) loses ~2 k cycles.
__device__ __noinline__ void validate_block_ni(const VArgs& a, int* hs, bool noop_cu) { validate_block(a, hs, noop_cu); }
__device__ __noinline__ bool validate_block_fast_ni(const VArgs& a, int* hs, bool noop_cu) { return validate_block_fast(a, hs, noop_cu); }
}  // namespace kdaval

// host side: kernel arguments from a Python dict {N, T, cu, cu_out, d, d_out, P_d, d_trash, s, s_out, P_s, s_trash, ck, ck_out, C, ck_tokens, status} (tensors may be None)
namespace kdaval {
inline const void* opt_table(const pybind11::dict& v, const char* key, int* is64) {
  *is64 = 0;
  if (!v.contains(key) || v[key].is_none()) return nullptr;
  torch::Tensor t = v[key].cast<torch::Tensor>();
  TORCH_CHECK(t.is_contiguous() && (t.scalar_type() == torch::kInt64 || t.scalar_type() == torch::kInt32), key, ": table must be contiguous int32 / int64");
  *is64 = t.scalar_type() == torch::kInt64;
  return t.data_ptr();
}
inline void* opt_out(const pybind11::dict& v, const char* key) {
  if (!v.contains(key) || v[key].is_none()) return nullptr;
  return v[key].cast<torch::Tensor>().data_ptr();
}
inline VArgs args_from_dict(const pybind11::dict& v) {
  VArgs a{};
  a.N = v["N"].cast<int>(); a.T = v["T"].cast<long long>();
  a.cu = opt_table(v, "cu", &a.cu_is64); a.cu_out = (long long*)opt_out(v, "cu_out");
  a.d = opt_table(v, "d", &a.d_is64); a.d_out = (int*)opt_out(v, "d_out"); a.P_d = v["P_d"].cast<long long>(); a.d_trash = v["d_trash"].cast<int>();
  a.s = opt_table(v, "s", &a.s_is64); a.s_out = (int*)opt_out(v, "s_out"); a.P_s = v["P_s"].cast<long long>(); a.s_trash = v["s_trash"].cast<int>();
  a.ck = opt_table(v, "ck", &a.ck_is64); a.ck_out = (long long*)opt_out(v, "ck_out"); a.C = v["C"].cast<long long>(); a.ck_tokens = v["ck_tokens"].cast<long long>();
  a.status = (int*)v["status"].cast<torch::Tensor>().data_ptr();
  a.out = opt_out(v, "out"); a.out_bytes = a.out ? (unsigned long long)v["out"].cast<torch::Tensor>().nbytes() : 0;
  if (v.contains("pad_src") && !v["pad_src"].is_none()) {   // [long] fused beta padding copy: pad_src [T, H] -> pad_dst [Tp, Hp] (bf16)
    const torch::Tensor ps = v["pad_src"].cast<torch::Tensor>(), pd = v["pad_dst"].cast<torch::Tensor>();
    a.pad_src = (const unsigned short*)ps.data_ptr(); a.pad_dst = (unsigned short*)pd.data_ptr(); a.pad_n = ps.numel(); a.pad_H = (int)ps.size(1); a.pad_Hp = (int)pd.size(1);
    a.pad_blocks = a.pad_n > 0 ? (int)std::min<long long>((a.pad_n + 1023) / 1024, 128) : 0;   // one pass of 4 elements per thread for blockDim >= 256 up to ~130k elements (one cold-L2 load latency)
  }
  return a;
}
// [dynpath2] a rejected cu_seqlens (not non-decreasing inside [0, T], cu[0] != 0) makes the launch a no-op: the kernels write nothing, so the output buffer is zero-filled here (one block, only
// when the table is bad; the normal path pays one cu scan inside the validation block, which runs concurrently with the planner).  Called by the validation block after validate_block.
__device__ void zero_out_if_bad(const VArgs& a) {
  if (a.out == nullptr || a.cu == nullptr) return;
  __shared__ int s_bad;
  if (threadIdx.x == 0) s_bad = 0;
  __syncthreads();
  int bad = 0;
  for (int i = threadIdx.x; i <= a.N; i += blockDim.x) {
    const long long r0 = rd(a.cu, a.cu_is64, i), r1 = i < a.N ? rd(a.cu, a.cu_is64, i + 1) : r0;
    if (r0 < 0 || r0 > a.T || r1 < r0 || (i == 0 && r0 != 0)) bad = 1;
  }
  if (bad) s_bad = 1;
  __syncthreads();
  if (!s_bad) return;
  uint4* p = (uint4*)a.out;
  const unsigned long long n16 = a.out_bytes / 16;
  for (unsigned long long i = threadIdx.x; i < n16; i += blockDim.x) p[i] = make_uint4(0, 0, 0, 0);
}
// [long] the padded-beta copy (bf16 [T, H] -> [Tp, Hp], TMA needs 16-byte rows) as extra blocks of the planner / validation kernel: it overlaps the planner instead of being a graph node of its own
// (a graph node costs ~2.5 us of launch latency on the critical path).  Block j of a.pad_blocks; any blockDim.
__device__ void pad_block(const VArgs& a, int j) {
  const long long n = a.pad_n;
  const unsigned long long stride = (unsigned long long)a.pad_blocks * blockDim.x;
  const unsigned short* __restrict__ src = a.pad_src;
  unsigned short* __restrict__ dst = a.pad_dst;
  if (n < (1ll << 27)) {
    const float inv = 1.0f / (float)a.pad_H;
    const unsigned st = (unsigned)stride;
    constexpr int U = 8;   // elements per thread and pass: all U loads are issued before the first store (src / dst do not alias)
    for (unsigned i0 = j * blockDim.x + threadIdx.x; i0 < (unsigned)n; i0 += U * st) {
      unsigned short v[U];
      #pragma unroll
      for (int u = 0; u < U; u++) { const unsigned i = i0 + u * st; v[u] = i < (unsigned)n ? src[i] : (unsigned short)0; }
      #pragma unroll
      for (int u = 0; u < U; u++) {
        const unsigned i = i0 + u * st;
        if (i < (unsigned)n) {
          int t = (int)((float)i * inv);
          if ((unsigned)(t * a.pad_H) > i) t--; else if ((unsigned)((t + 1) * a.pad_H) <= i) t++;
          dst[(size_t)t * a.pad_Hp + (i - (unsigned)t * a.pad_H)] = v[u];
        }
      }
    }
  } else {
    for (unsigned long long i = (unsigned long long)j * blockDim.x + threadIdx.x; i < (unsigned long long)n; i += stride) {
      const unsigned long long t = i / a.pad_H;
      dst[t * a.pad_Hp + (i - t * a.pad_H)] = src[i];
    }
  }
}
inline int hash_slots(int N) { int HS = 64; while (HS < 2 * N) HS <<= 1; return HS; }
}  // namespace kdaval
