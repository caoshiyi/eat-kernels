// Token loop of kda_main (section 4, recurrence): included from inside kda_main below (#include of this file with
// KDA_TOKLOOP_BODY defined), once per state-store cache policy KDA_TOKPOL. The rest of the file is the #else branch.
#ifdef KDA_TOKLOOP_BODY
#pragma unroll
  for (int t = 0; t < S; ++t) {
    float rq, rk, kq, beta;
    if constexpr (S2_) {
      const f4 nm = *reinterpret_cast<const f4*>(&s_nrm2[t][0]);
      rq = nm[0]; rk = nm[1]; kq = nm[2]; beta = nm[3];
    } else {
    const float sq = s_red[0][3 * t] + s_red[1][3 * t];
    const float sk = s_red[0][3 * t + 1] + s_red[1][3 * t + 1];
    const float sqk = s_red[0][3 * t + 2] + s_red[1][3 * t + 2];
    // q/k L2 norms are folded into the reduced scalars (q, k stay raw in LDS): saves the per-token rescale.
    rq = __builtin_amdgcn_rsqf(sq + 1e-6f) * scale;
    rk = __builtin_amdgcn_rsqf(sk + 1e-6f);
    kq = sqk * rq * rk;
    beta = s_beta[t];
    }
    f4 q4[2], k4[2], d4[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int kb = 4 * rlane + 64 * j;
      q4[j] = *reinterpret_cast<const f4*>(&s_q[t][kb]);
      k4[j] = *reinterpret_cast<const f4*>(&s_k[t][kb]);
      d4[j] = *reinterpret_cast<const f4*>(&s_d[t][kb]);
    }
    constexpr bool HN_ = PR_HNORM && !SPEC && RPT >= 4;
    if constexpr (HN_) {
#pragma unroll
      for (int j = 0; j < 2; ++j) { k4[j] = k4[j] * rk; q4[j] = q4[j] * rq; }
    }
    if constexpr (PR_KEEPW && !SPEC && PF && !NX && NT == 1024) {
#pragma unroll
      for (int j = 0; j < 4; ++j) asm volatile("" :: "v"(w0[j]), "v"(w1[j]));
    }
    float vv[RPT];
    if constexpr ((SPEC || PR_SVC) && RPT >= 2) {
#pragma unroll
      for (int i = 0; i < RPT; ++i) vv[i] = s_v[t][rgroup * RPT + i];
    }
    bool do_store = SPEC ? (wsl[t] > 0) : (t == S - 1);
    float* ubase = k_state + SLOTMUL(wsl[t], k_sss) + (int64_t)h * D * D + split * VB * D;  // uniform
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      f4 hd[2];
      float pq[4];
#pragma unroll
      for (int j = 0; j < 2; ++j) hd[j] = hr[i][j] * d4[j];
#pragma unroll
      for (int e = 0; e < 4; ++e) pq[e] = (SPEC && !AB_NOPQ) ? 0.f : hd[0][e] * q4[0][e] + hd[1][e] * q4[1][e];
      const f2 ak = hd[0].lo * k4[0].lo + hd[0].hi * k4[0].hi + hd[1].lo * k4[1].lo + hd[1].hi * k4[1].hi;
      float aks_ = ak.x + ak.y;
      if constexpr (AB_NOPK & (SPEC ? 1 : 2)) {
        float pk_[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) pk_[e] = hd[0][e] * k4[0][e] + hd[1][e] * k4[1][e];
        aks_ = (pk_[0] + pk_[1]) + (pk_[2] + pk_[3]);
      }
      const float dk = HN_ ? row_reduce(aks_) : row_reduce(aks_) * rk;
      float dqs;
      if constexpr (SPEC && !AB_NOPQ) {  // packed (2-wide) h.q partials, like h.k
        const f2 aq = hd[0].lo * q4[0].lo + hd[0].hi * q4[0].hi + hd[1].lo * q4[1].lo + hd[1].hi * q4[1].hi;
        dqs = aq.x + aq.y;
      } else {
        dqs = (pq[0] + pq[1]) + (pq[2] + pq[3]);
      }
      const float dq = HN_ ? row_reduce(dqs) : row_reduce(dqs) * rq;
      const int r = rgroup + RG * i;
      const float vn = (((SPEC || PR_SVC) && RPT >= 2 ? vv[i] : s_v[t][r]) - dk) * beta;
      const float vk = HN_ ? vn : vn * rk;
#pragma unroll
      for (int j = 0; j < 2; ++j) hr[i][j] = hd[j] + k4[j] * vk;
      // split: wave 0 defers its last-token stores until after the arrival atomic (it waits on its o stores).
      // Spec: branchless store (num_records 0 when the slot is not written: the hardware drops out-of-range buffer
      // stores), so the token loop stays one basic block and the stores issue right after the update.
      if constexpr ((SPEC && !AB_BRST) || NSPLIT > 1) {
        const bool st = do_store && !(NSPLIT > 1 && t == S - 1 && wave == 0);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          __builtin_amdgcn_raw_buffer_store_b128(
              __builtin_bit_cast(u4, hr[i][j]),
              __builtin_amdgcn_make_buffer_rsrc(ubase, 0, st ? 0x7fffffff : 0, 0x00020000),
              ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPEC ? KDA_TOKPOL : SPOL_PLAIN);
      } else if ((SPEC || !(NSPLIT == 1 && ((NX && BSTS) || HB_))) && do_store && !(NSPLIT > 1 && t == S - 1 && wave == 0)) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          __builtin_amdgcn_raw_buffer_store_b128(
              __builtin_bit_cast(u4, hr[i][j]),
              __builtin_amdgcn_make_buffer_rsrc(ubase, 0, 0x7fffffff, 0x00020000),
              ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPEC ? SPOL_SPEC : ((PF && NX) ? SPOL_PLAIN_MR : SPOL_PLAIN));
      }
      if constexpr (HB_) {  // PR_HBST: stores of the last PR_HBST rows as one burst
        constexpr int HBN_ = PR_HBST > 0 ? PR_HBST : 1;
        if ((i % HBN_ == HBN_ - 1 || i == RPT - 1) && do_store) {
#pragma unroll
          for (int i2 = 0; i2 < RPT; ++i2)
            if (i2 <= i && i2 >= i - (i % HBN_))
#pragma unroll
              for (int j = 0; j < 2; ++j)
                __builtin_amdgcn_raw_buffer_store_b128(
                    __builtin_bit_cast(u4, hr[i2][j]), __builtin_amdgcn_make_buffer_rsrc(ubase, 0, 0x7fffffff, 0x00020000),
                    ((rgroup + i2 * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPOL_PLAIN);
        }
      }
      if constexpr (SPEC || PR_SOALL) s_o[t][r] = dq + vn * kq;  // all 16 lanes write the same value: no exec branch
      else if (rlane == 0) s_o[t][r] = dq + vn * kq;
    }
    if constexpr (!SPEC && NSPLIT == 1 && NX && BSTS) {
      if (do_store) {
#pragma unroll
        for (int i = 0; i < RPT; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            __builtin_amdgcn_raw_buffer_store_b128(
                __builtin_bit_cast(u4, hr[i][j]), __builtin_amdgcn_make_buffer_rsrc(ubase, 0, 0x7fffffff, 0x00020000),
                ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPOL_PLAIN_MR);
      }
    }
    if constexpr (!SPEC && NX && NSPLIT == 1 && PR_MRLATE != 3) {
      if (t == S - 1) {
        if (wave < 2) {
#pragma unroll
          for (int u = 0; u < S + 2; ++u) {
            stbs<false, false, SCP>(crow + u * ctap + ha_off, u < 2 ? ha[u + 1] : xa[u - 2]);
            stbs<false, false, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
          }
        } else if (wave < 4 && c < VB) {
#pragma unroll
          for (int u = 0; u < S + 2; ++u) stbs<false, false, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
        }
      }
    }
    if constexpr (LBL_ && LBLP_ == 1) {
      if (t == S - 1) {
      if (wave < 2) {
#pragma unroll
        for (int u = 0; u < S + 2; ++u) {
          stbs<false, NTSP, SCP>(crow + u * ctap + ha_off, u < 2 ? ha[u + 1] : xa[u - 2]);
          stbs<false, NTSP, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
        }
      } else if (wave < 4 && c < VB) {
#pragma unroll
        for (int u = 0; u < S + 2; ++u) stbs<false, NTSP, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
      }
      }
    }
    if constexpr (SPEC && !(AB_NOSB & 2)) __builtin_amdgcn_sched_barrier(0);  // no cross-token hoisting (VGPRs)
    if (NX && SPEC && !(AB_NONX & 2) && t == (PR_NXT < S ? PR_NXT : S - 1)) {  // next-round prefetch after the first token
    __shared__ float s_pfsink[64];
    if (wave == NXW_ && b + nxd_ < NBLK_) {
      if constexpr (PR_PRIO_NX) __builtin_amdgcn_s_setprio(3);
      const unsigned pfm = (unsigned)(uintptr_t)s_pfsink;
      const int gn = b + nxd_, hn = gn % H, nn = gn / H;
      int an, rdn, csn;
      int wsln[S];
      if (slots<S, SPEC>(k_idx, k_acc, nn, an, rdn, csn, wsln)) {
        // one 128 B line per lane: conv history taps [a-1, a+2) (18 lines), x of S tokens (6S), gate (2S), og (2S)
        const bf16* crn = k_conv + SLOTMUL(csn, k_css) + hn * D;
        const int64_t tn0 = (int64_t)nn * S;
        constexpr int NL = 18 + 10 * S;
#pragma unroll
        for (int k = 0; k < (NL + 63) / 64; ++k) {
          int l = lane + 64 * k;
          l = l < NL ? l : NL - 1;
          const int half = (l & 1) * 64;
          const bf16* pa;
          const bf16* b0_ = crn + (int64_t)(an - 1) * (3 * dim);  // wave-uniform bases; lane terms in 32 bits
          const bf16* b1_ = k_x + tn0 * p.x_stride + hn * D;
          const bf16* b2_ = p.gate + (tn0 * H + hn) * D;
          const bf16* b3_ = p.og + tn0 * dim + hn * D;
          const int xs32_ = (int)p.x_stride;
          if (l < 18) {
            const int tap = l / 6, grp = (l % 6) >> 1;
            pa = b0_ + (tap * (3 * dim) + grp * dim + half);
          } else if (l < 18 + 6 * S) {
            const int m = l - 18, t = m / 6, grp = (m % 6) >> 1;
            pa = b1_ + (t * xs32_ + grp * dim + half);
          } else if (l < 18 + 8 * S) {
            const int t = (l - 18 - 6 * S) >> 1;
            pa = b2_ + (t * dim + half);
          } else {
            const int t = (l - 18 - 8 * S) >> 1;
            pa = b3_ + (t * dim + half);
          }
          asm volatile("s_mov_b32 m0, %1\n global_load_lds_dword %0, off" KDA_NXNT_ :: "v"(pa), "s"(pfm) : "memory", "m0");
        }
      }
      if constexpr (PR_PRIO_NX) __builtin_amdgcn_s_setprio(0);
    }
  }
    if constexpr (SPEC && !(AB_NOSB & 2)) __builtin_amdgcn_sched_barrier(0);
    if (SPEC && PF && NT == 1024 && S >= LCS && !PR_LCEP && t == LCTT_) {
      // conv commit after the last token (state rows consumed): the token loop's waits for the state rows no longer retire
      // the commit stores (q/k waves 0-1, v waves 2-3)
      if (wave < 2) {
        if (NSPLIT == 1) {
#pragma unroll
          for (int u = 0; u < S + 2; ++u) {
            stn<SNTK_>(crow + u * ctap + ha_off, u < 2 ? ha[u + 1] : xa[u - 2]);
            stn<SNTK_>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
          }
        }
      } else if (wave < 4 && c < VB) {
#pragma unroll
        for (int u = 0; u < S + 2; ++u) stn<SNTK_>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
      }
      if constexpr (SPEC && !(AB_NOSB & 2)) __builtin_amdgcn_sched_barrier(0);
    }
  }
#else
// Fused KDA decode / MTP verify for MI350X (gfx950). See program.md for semantics.
//
// Grid: one block (256 threads) per (sequence n, head h, V-split). A block owns VB = 128 / NSPLIT state rows.
// S (tokens per sequence) is a template parameter: every prologue load is unconditional, with role-dependent
// addresses chosen by selects, so all loads issue back to back in straight-line code (loads under divergent
// branches each got their own s_waitcnt vmcnt(0), a ~1 us round trip per token after the cache flush).
// 1. Slot-independent loads (x, gate, beta, out_gate, conv weights) first, then the slot chain
//    (num_accepted -> state index -> state tile + conv history).
// 2. Prologue: waves 0-1 conv1d + SiLU of q/k (all 128 channels) for all S tokens, q/k L2 norms and q.k
//    (DPP + readlane reductions); waves 2-3 decay, beta and the conv of this block's v rows. All to LDS.
// 3. Recurrence: 16 lanes per state row, 8 floats per lane; per token decay, h.k and h.q, rank-1 update,
//    nontemporal store of the new state.
// 4. NSPLIT == 1: gated RMSNorm epilogue + conv commit in-kernel.
//    NSPLIT  > 1: o goes to an fp32 workspace with write-through (system-scope) stores; after s_waitcnt the
//    block bumps a per-(n,h) counter (relaxed device atomic). The last block to arrive resets the counter,
//    reads o back with system-scope loads, does the RMSNorm and commits the q/k conv taps (every split reads
//    them). v taps are committed by their owning split. No fences: an agent-scope release (buffer_wbl2)
//    costs 100s of us here.
#ifndef KDA_DEVICE_ONLY
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifndef SPOL_SPEC
#define SPOL_SPEC 0
#endif
#ifndef SPOL_PLAIN_MR
// plain multi-round kernels (prologue-first + next-round prefetch): sc1 nt state stores (A/B: b256 -1.2, d_k3_b128 -0.3 us;
// the co-resident hold grids lose +1.4 us with nt stores, NOTES_bw M12)
#define SPOL_PLAIN_MR 18
#endif
#ifndef SPOL_PLAIN
#define SPOL_PLAIN 16
#endif
// Plain multi-round kda_main: state stores batched after the token's row math (BST) -> 64 VGPRs, 2 blocks per CU; the first
// co-resident generation desynchronised by a one-time delay of 0..STAG ticks in STAGL levels, by a hash of the sequence (MECHANISMS.md (a)).
#ifndef BLD
#define BLD 2
#endif
#ifndef BST
#define BST 1
#endif
#ifndef STAG
#define STAG 250
#endif
#ifndef STAGL
#define STAGL 2
#endif
#ifndef SBAR
#define SBAR 0
#endif
#ifndef SNT
#define SNT 2
#endif
#ifndef PFSTRIDE
#define PFSTRIDE (2 << 20)
#endif
// ===== PROBE SWITCHES (scratch_probes/PROBES.md); every switch 0 = v54 =====
#ifndef PR_RSQ
#define PR_RSQ 1       // epilogue rsqrtf -> raw v_rsq_f32 (no denorm scaling)
#endif
#ifndef PR_EPWS
#define PR_EPWS 0      // epilogue wave sum via DPP row_bcast (1 readlane) instead of 4 readlanes
#endif
#ifndef PR_V16DUP
#define PR_V16DUP 0    // kda_vec (TT < NT): waves >= TT/64 issue their (duplicate, unused) state-row loads with num_records 0
#endif
#ifndef PR_NRMS
#define PR_NRMS 0      // spec kda_main: per-token norm scalars via readlane into SGPRs (no second barrier, no s_nrm2)
#endif
#ifndef PR_RR2
#define PR_RR2 0       // kda_main token loop: h.k and h.q row reductions interleaved (row_reduce2)
#endif
#ifndef PR_SWALK
#define PR_SWALK 0     // kda_main row walk per token: 1 = reverse on odd tokens, 2 = always descending
#endif
#ifndef PR_LDORD
#define PR_LDORD 0     // kda_main state-row loads j-outer / i-inner
#endif
#ifndef PR_PRIO_MRPRO
#define PR_PRIO_MRPRO 0  // plain multi-round: prologue waves (0-3) s_setprio(N) from entry to the prologue barrier
#endif
#ifndef PR_PRIO_SPECW
#define PR_PRIO_SPECW 0  // spec kda_main: s_setprio(3 - wave / 4) for the whole block (wave-ordered issue priority)
#endif
#ifndef PR_PRIO_NX
#define PR_PRIO_NX 0     // multi-round: NX wave s_setprio(3) around the next-round prefetch block
#endif
#ifndef PR_NXBI
#define PR_NXBI 0      // NX LDS-DMA prefetch via __builtin_amdgcn_global_load_lds (visible to the waitcnt pass) instead of asm
#endif
#ifndef PR_WUW
#define PR_WUW 1       // kda_main translation warm-up issued by wave 4 (first non-prologue wave) instead of wave NW-1
#endif
#ifndef PR_NXW
#define PR_NXW 0       // kda_main NX prefetch issued by wave 4 instead of wave NW-1
#endif
#ifndef PR_HOLDBAR
#define PR_HOLDBAR 0   // read-first hold polled by wave 0 only, other waves wait at a block barrier
#endif
#ifndef PR_HSLEEP
#define PR_HSLEEP 2    // s_sleep argument in the read-first hold poll loop
#endif
#ifndef PR_SOALL
#define PR_SOALL 1     // plain kda_main: s_o written by all 16 lanes of a row (no exec branch), as in spec
#endif
#ifndef PR_GRID2D
#define PR_GRID2D 1    // 2-D grid (x = H * NSPLIT, y = B): no runtime integer division for (n, h, split)
#endif
#ifndef PR_EXP2
#define PR_EXP2 0      // decay exp(lb * sigm) as exp2((lb * log2e) * sigm), lb * log2e folded once per thread
#endif
#ifndef PR_V16NT
#define PR_V16NT 0     // kda_vec: read-once v-row chunks (x_v, conv-history v taps) loaded nontemporal
#endif
#ifndef PR_V16OUTNT
#define PR_V16OUTNT 0  // kda_vec exchange wave: out stores and v-tap commit stores nontemporal
#endif
#ifndef PR_PLBST
#define PR_PLBST 0     // latency-bound plain kda_main (PF, !NX, 1024 thr): small bf16 stores 1 = nt, 2 = default (v54: sc0 sc1)
#endif
#ifndef PR_SOFF
#define PR_SOFF 0      // kda_main state-row loads: row-block offset i * RG * D * 4 in soffset (SGPR) instead of the lane offset
#endif
#ifndef PR_WS3B
#define PR_WS3B 0      // kda_main prologue: the three wave sums of a token interleaved, DPP row_bcast form (wave_sum3b)
#endif
#ifndef LCS
#define LCS 7          // spec 1024-thread PF kda_main: late conv commit (after the token loop) for S >= LCS
#endif
#ifndef PR_POLLSLEEP
#define PR_POLLSLEEP 0 // kda_vec exchange wave: s_sleep(N) between mailbox polls (0 = none)
#endif
#ifndef PR_SVC
#define PR_SVC 1       // plain kda_main: s_v thread-contiguous (as spec RPT >= 2), v of a thread's rows read as one vector
#endif
#ifndef PR_ENDW
#define PR_ENDW 0      // kda_main: end-of-kernel s_waitcnt vmcnt(0) only in the warm-up / NX waves (others exit with stores in flight)
#endif
#ifndef PR_V16QKSPREAD
#define PR_V16QKSPREAD 0  // kda_vec exchange: the q/k conv-tap commit spread over all splits (chunk i -> split i % NSPLIT), not split 0 only
#endif
#ifndef PR_V16TOK
#define PR_V16TOK 0    // kda_vec prologue: token t on wave (t + PR_V16TOK) % NW
#endif
#ifndef PR_MRLATE
#define PR_MRLATE 1    // plain multi-round kda_main: conv commit (q/k/v taps) after the token loop's state stores, not in the prologue
#endif
#ifndef PR_V16BL
#define PR_V16BL 0     // kda_vec first-hop chunk addressing branchless (P61)
#endif
#ifndef PR_V16LDS
#define PR_V16LDS 0    // kda_vec conv-history staging via 16 B LDS-DMA (P62)
#endif
#ifndef PR_LBLATE
#define PR_LBLATE 0    // latency-bound plain kda_main: conv commit 1 = after the token loop, 2 = after the out stores
#endif
#ifndef PR_EPW
#define PR_EPW 0       // kda_main: epilogue token t (and its og / nw loads) on wave (t + 4) % NW instead of wave t
#endif
#ifndef PR_MRPAD
#define PR_MRPAD 0     // plain multi-round kda_main: LDS pad (floats) to force 1 block per CU (occupancy probe)
#endif
#ifndef PR_HLATE
#define PR_HLATE 0     // prologue-first plain hold kda_main (256 thr): conv commit after the token loop (taps in VGPRs)
#endif
#ifndef PR_PFROT
#define PR_PFROT 0     // translation warm-up: block b touches 2 MiB regions (32 b + lane) mod pool, i.e. the grid covers the pools
#endif
// ---- PROBES2 switches (all default 0 = v57)
// ---- probe-runner switches (iteration 9 phase 2, Y series; all default 0 = v61)
#ifndef PR_PNX
#define PR_PNX 0       // plain multi-round NX prefetch position: 0 = after the state-row loads (v61), 1 = after the prologue barrier, 2 = after the token's state stores
#endif
#ifndef PR_VNOA
#define PR_VNOA 2      // kda_vec exchange wave warm-up loads: 1 = no a1 (mailbox word), 2 = no a0 (pool translation), 3 = neither
#endif
#ifndef PR_NXT
#define PR_NXT 0       // spec multi-round: next-round prefetch after token min(PR_NXT, S - 1) (0 = after the first token, v61)
#endif
#ifndef PR_STAGPOS
#define PR_STAGPOS 1   // plain multi-round first-generation stagger: 0 = at entry (v61), 1 = before the state-row loads (after the history loads), 2 = after the first-hop loads (before the slot branch)
#endif
#ifndef PR_WUB
#define PR_WUB 64       // kda_main: pool translation warm-up a0 only in blocks b < PR_WUB (0 = every block)
#endif
#ifndef PR_NOA1
#define PR_NOA1 3      // kda_main: no a1 (out / exchange word) warm-up load: 1 = spec, 2 = plain, 3 = both
#endif
#ifndef PR_WUS
#define PR_WUS 0       // warm-up a0: pool byte bounds as opaque wave-uniform SGPR pairs (no per-lane 64-bit multiplies)
#endif
#ifndef PR_SWU0
#define PR_SWU0 0      // spec kda_main: a1 warm-up after the first-hop loads (plain position), a0 before the Params wait
#endif
#ifndef PR_V16WU0
#define PR_V16WU0 0    // kda_vec exchange wave: sched_barrier after the a0 warm-up (a0 ahead of the Params wait / og address math)
#endif
#ifndef PR_NOHOLD
#define PR_NOHOLD 0    // 1024-thread whole-state plain kda_main: read-first hold compiled out (no t_start s_memrealtime, no p.hold load / branch)
#endif
#ifndef PR_ACLAMP
#define PR_ACLAMP 1    // slots(): accepted-count clamp as smax + umin (stays on the SALU; no v_med3 + VALU multiplies ahead of the history hop)
#endif
#ifndef PR_BETA32
#define PR_BETA32 0    // spec kda_main: beta address = scalar (tok0 * H + h) + 32-bit lane term (no 64-bit vector multiply)
#endif
#ifndef LCT
#define LCT 99         // spec late conv commit (S >= LCS): issued after token min(LCT, S - 1) (99 = after the last token, v57)
#endif
#ifndef PR_LCEP
#define PR_LCEP 0      // spec late conv commit (S >= LCS) after the epilogue out stores instead of after the last token
#endif
#ifndef PR_HBST
#define PR_HBST 0      // plain 256-thread whole-state (hold) kda_main: state stores issued in bursts of PR_HBST rows (8 = all after the row math, as multi-round BST)
#endif
#ifndef PR_HOLD32
#define PR_HOLD32 0    // hold / stagger spin loops: 32-bit tick difference on the SALU (no v_cmp_*_u64 per poll)
#endif
#ifndef PR_NXND
#define PR_NXND 0      // NX target (hn, nn) from (h, n) + host-precomputed nxt / H, nxt % H: no runtime division; 2 = also B from Params (no hidden-arg load)
#endif
#ifndef PR_NX32
#define PR_NX32 1      // NX prefetch addresses: scalar 64-bit bases + 32-bit lane offsets (no 64-bit vector multiplies)
#endif
#ifndef PR_MRWU0
#define PR_MRWU0 1     // plain multi-round: a0 warm-up issued before the first-generation stagger / Params + idx wait
#endif
#ifndef PR_V16M32
#define PR_V16M32 0    // kda_vec first-hop chunk addresses: token terms as scalar base + 32-bit lane multiply
#endif
#ifndef PR_V16H32
#define PR_V16H32 0    // kda_vec history chunk addresses: scalar (crow + start * ctap) + slot-independent 32-bit lane offset
#endif
#ifndef PR_V16BETA
#define PR_V16BETA 0   // kda_vec: beta load pinned above the slot branch (issued with the first-hop chunk)
#endif
#ifndef PR_V16VCE
#define PR_V16VCE 0    // kda_vec V16 (TT < NT): v-tap commit by wave NW-1 right after the prologue barrier (not after the token loop)
#endif
#ifndef PR_V16PUB
#define PR_V16PUB 0    // kda_vec exchange: per-lane partial sums straight from s_o (no DPP chain, no ds_bpermute)
#endif
#ifndef PR_V16VALL
#define PR_V16VALL 1   // kda_vec tokpro: v-row conv and s_v write by all lanes (no exec-masked branch)
#endif
#ifndef PR_KEEPW
#define PR_KEEPW 0     // latency-bound plain kda_main: conv-weight VGPRs kept live past the first LDS operand reads (drops the false vmcnt(7))
#endif
#ifndef PR_PS2
#define PR_PS2 0       // plain kda_main: norm scalars once per block (wave 0 -> LDS, one more barrier), as spec S2
#endif
#ifndef PR_B2FO
#define PR_B2FO 0      // b2f_o: 16-bit asm operand (no v_and 0xffff in front of the shift)
#endif
#ifndef PR_OGLATE
#define PR_OGLATE 0    // PF kda_main: og / nw loads issued after the conv-history loads (not ahead of the slot chain)
#endif
#ifndef PR_HNORM
#define PR_HNORM 0     // plain kda_main with >= 4 rows per thread (hold): k, q pre-scaled by rk, rq once per thread (3 fewer multiplies per row)
#endif
#ifndef PR_NXPRE
#define PR_NXPRE 0     // spec multi-round: the NX wave issues the next block's slot s_loads at block start (consumed after token 0: no SMEM stall between tokens)
#endif
#ifndef PR_V16HDUP
#define PR_V16HDUP 0   // kda_vec: waves without a real conv-history chunk (wave * 64 >= NB) skip the duplicate history load + LDS write
#endif
#ifndef PR_OGS
#define PR_OGS 0       // kda_main (NSPLIT 1): sigm(og) pinned before the epilogue barrier (after the token loop; og landed long before)
#endif
#ifndef PR_V16RST
#define PR_V16RST 0    // kda_vec exchange wave: mailbox reset store issued after the out stores
#endif
#ifndef PR_V16XSLOT
#define PR_V16XSLOT 0  // kda_vec exchange wave: slot chain read after the prologue barrier (it reaches the staging barrier without an SMEM wait)
#endif
#ifndef PR_WS3L
#define PR_WS3L 0      // kda_main prologue: wave sums in row_bcast form, s_red written straight from lane 63 (no v_readlane / SGPR round trip)
#endif
// ---- codegen switches (iteration 10, NOTES_codegen.md)
#ifndef CG_FH
#define CG_FH 1        // kda_vec staging loads: per-wave compile-time run tables (scalar run base + lane * 16; history / beta as buffer loads with slot-independent lane offsets computed ahead of the slot wait)
#endif
#ifndef CG_SL
#define CG_SL 1        // kda_vec token waves: slot chain s_loads issued at entry, in parallel with the Params kernarg read (one merged wait); history loads then issue with the first hop
#endif
#ifndef CG_XP
#define CG_XP 0        // kda_vec exchange wave: everything of its tail that does not depend on s_o / the mailbox (store offsets, permute index, sigm(og) * nw, q/k commit chunks) is computed and pinned ahead of the barriers; mailbox / out / commit as buffer accesses with lane-masking by out-of-bounds offsets
#endif
#ifndef CG_TP
#define CG_TP 1        // kda_vec tokpro: packed (2-wide) SiLU / decay math with folded scalars, v and beta sigmoids as one pair, the three wave sums as one 18-instruction DPP asm block (row_bcast steps as masked v_add_f32_dpp), s_nrm written by all lanes
#endif
#ifndef CG_RE
#define CG_RE 0        // kda_vec 16-row mode: state rows issued before the staging barrier (behind the history loads), 1 = every wave, 2 = token waves only
#endif
#ifndef CG_ML
#define CG_ML 3        // kda_main prologue loads as (uniform base on the SALU) + (32-bit lane offset): one load instruction per value, no 64-bit per-lane address stepping. 1 = every kda_main, 2 = 1024-thread prologue-first single-round only, 3 = the same, spec only, 4 = spec only and not the SB (state rows behind the block barrier) instantiation: there the prologue loads already have the memory to themselves, with the state rows racing them their issue speed pays
#endif
#define KDA_SELF "kda_v13_a.cu"  // this file (included through a macro: torch's hipify pass rewrites literal .cu includes)
#ifndef ST_TN
#define ST_TN 1        // spec multi-round kda_main: second token-loop copy with state-store policy ST_POL for the blocks selected at run time
#endif
#ifndef ST_PM
#define ST_PM (xcc_ & 1)  // which XCDs: the odd member of a write-path pair
#endif
#ifndef ST_TNV
#define ST_TNV 4        // shortest window with a two-copy (TN) multi-round kernel. S in [ST_TNV, ST_TNS) has it as a separate instantiation, launched only when tail_nt() selects a tail: the grids that do not (<= 2 rounds) keep the single-loop kernel (the second copy costs it +0.06..+0.08 us unused, NOTES_final.md)
#endif
#ifndef ST_TNS
#define ST_TNS 5       // ... only in the instantiations the cost model can select (tail_nt() in kernel_*.py: S * TN_US_PER_TOKEN > TN_FIXED_US, i.e. S >= 5); S < ST_TNS keeps the single loop (v65 instructions: the second copy cost the S = 4 kernel +0.4 us unused)
#endif
#ifndef ST_POL
#define ST_POL 2       // cache policy of the selected blocks' state stores (2 = nt)
#endif
#ifndef ST_POLT
#define ST_POLT 0      // ... from token ST_POLT on
#endif
#ifndef MB_CLR
#define MB_CLR 5       // kda_vec: a block resets the mailbox words it will publish before it publishes (best-effort self-healing of stale mailbox words, REDTEAM R2; the hard reset after a device error is reset_workspace()). 0 = off (v69); 1 = before the staging barrier, lane -> (sibling lane / S, token lane % S) (v70-v72); 2 = same place, lane -> (sibling lane % NSPLIT, token lane / NSPLIT): any bijection resets the same words, no division by S; 3 = behind the prologue barriers (the youngest wave is starved by the token loop: a fully stale mailbox is consumed in 96 % / 0-42 % / 1-5 % of the calls at S = 2 / 3 / 4: not used); 4 = right behind the staging barrier; 5 = form 4 in the 16-row kernel for S >= MB_LATE_S, form 2 everywhere else; 7 = forms 2 + 3 (no faster than 2)
#endif
#ifndef MB_LATE_S
#define MB_LATE_S 4      // 16-row kernels (<= 128 blocks, each alone on its CU, at most half of the CUs busy): the reset may wait for the staging barrier from this window length on. The early store costs the latency-bound 16-row grids 0.06-0.17 us; 32-row grids (up to every CU busy) keep the early reset: with the late one full grids consumed a fully stale mailbox in 3-18 of 12 000 injected faults (early: 0-2 of 12 000); 16-row late: 0 of 108 000. NOTES_final.md section 10
#endif
#ifndef TP_WS1
#define TP_WS1 1       // WS also in the single-round spec kda_main (1024 threads, no NX, no state-row barrier) where the dispatcher asks for it (p.hold < 0; wave_offset() in kernel_*.py); 0 = off (v68)
#endif
#ifndef TP_WSM
#define TP_WSM 3       // WS also in the middle generations (blocks with a successor on their CU: b < blocks - p.nxt) for S >= TP_WSM; 0 = first generation only (v67)
#endif
#ifndef TP_WS
#define TP_WS 80      // spec multi-round kda_main: waves >= NW/2 start the token loop TP_WS x s_sleep(1) late (0 = off, v64)
#endif
// ===== ABLATION SWITCHES (ABLATIONS.md, NOTES_ablate.md). v67 defaults differ from v66 in: AB_NOPIN=2, PR_VNOA=2, AB_NORR2V=1, SNT=2, AB_PLD=12, PR_V16VCE=0 (and launcher flag SPOL_PLAIN=17); every other default = v66 =====
#ifndef AB_NOA0
#define AB_NOA0 0      // kda_main: no pool translation warm-up a0 at all: 1 = spec, 2 = plain, 3 = both (PR_WUB can only bound it to b < N)
#endif
#ifndef AB_NONX
#define AB_NONX 0      // multi-round kda_main: next-round prefetch loads compiled out, kernel instantiation unchanged: 1 = plain, 2 = spec
#endif
#ifndef AB_NXMUL
#define AB_NXMUL 2     // next-round prefetch distance = p.nxt * AB_NXMUL / 2 (p.nxt itself, the first-generation bound of the stagger / WS, unchanged)
#endif
#ifndef AB_NXC
#define AB_NXC 0       // next-round prefetch LDS-DMA loads cached (no nt), as before v45
#endif
#ifndef AB_PLD
#define AB_PLD 12       // kda_main prologue-load policy per class: 1 = plain hold (256 thr) cached, 2 = spec cached, 4 = plain latency-bound nt, 8 = plain multi-round nt
#endif
#ifndef AB_BETA
#define AB_BETA 0      // kda_main beta load: 1 = cached everywhere (also spec multi-round), 2 = follows the class policy of the other prologue loads (v45)
#endif
#ifndef AB_NOSCP
#define AB_NOSCP 0     // plain multi-round kda_main: small bf16 stores default policy (no sc0 sc1); the latency-bound kernel has PR_PLBST = 2
#endif
#ifndef BSTS
#define BSTS BST       // plain multi-round: state stores batched after the row math (BST also keys the stagger / early a0; BSTS = 0 only unbatches)
#endif
#ifndef BLDAUX
#define BLDAUX BLD     // aux of the state-row buffer loads (BLD = 2: nt); 0 = cached
#endif
#ifndef AB_NOS2
#define AB_NOS2 0      // spec kda_main: per-token norm scalars in every wave for every token (no s_nrm2, no second barrier), as before v40
#endif
#ifndef AB_NOPQ
#define AB_NOPQ 0      // spec kda_main token loop: h.q dot as 4 scalar partials (as plain), as before v41
#endif
#ifndef AB_NOPK
#define AB_NOPK 0      // kda_main token loop: h.k dot as 4 scalar partials, as before v26: 1 = spec, 2 = plain
#endif
#ifndef AB_BRST
#define AB_BRST 0      // spec kda_main: state stores under a wave-uniform branch (not num_records 0), as before v21
#endif
#ifndef AB_NOPIN
#define AB_NOPIN 2     // no fake-use pins of the slot-independent loads above the slot branch: 1 = kda_main (PF), 2 = kda_vec
#endif
#ifndef AB_SLOTE
#define AB_SLOTE 0     // kda_main slot chain position: 1 = plain after the slot-independent loads (as spec), 2 = spec at entry (as plain)
#endif
#ifndef AB_ALLW
#define AB_ALLW 0      // PF kda_main: every wave issues the prologue input / history loads (not only waves 0-3), as before v14
#endif
#ifndef AB_HISTLATE
#define AB_HISTLATE 0  // kda_main: conv-history loads issued after the state-row loads, as before v10
#endif
#ifndef AB_NONBF
#define AB_NONBF 0     // kda_vec 16-row, S > 4: prologue barrier instead of the LDS ready flags (v57)
#endif
#ifndef AB_NOPIPE
#define AB_NOPIPE 0    // kda_vec token loop without the one-token LDS operand lookahead: 1 = 16-row mode, 2 = RPT >= 2
#endif
#ifndef AB_NORR2V
#define AB_NORR2V 1    // kda_vec 16-row mode: h.k / h.q row reductions not interleaved (dq after the stores)
#endif
#ifndef AB_NOSB
#define AB_NOSB 0      // no per-token sched_barrier: 1 = kda_vec, 2 = spec kda_main
#endif
#ifndef AB_SIGDIV
#define AB_SIGDIV 0    // sigmoid by fp division instead of v_rcp_f32
#endif
#ifndef AB_WSALL
#define AB_WSALL 0     // WS wave offset in every block (not only the first generation)
#endif
#ifndef AB_STAGKEY
#define AB_STAGKEY 0   // plain multi-round stagger keyed by a hash of the block (not of the sequence)
#endif
#ifndef AB_NOHALF
#define AB_NOHALF 0    // kda_vec RPT >= 2 (LATE): every state row after the staging barrier (no early half)
#endif
#ifndef AB_VPAD
#define AB_VPAD 0       // P-MR: N extra VGPRs kept live from entry to exit (holds the VGPR-based occupancy without touching memory)
#endif
#define KDA_NXNT_ " nt"
#ifndef PR_USS
#define PR_USS 0       // slot * slot-stride products as unsigned 32 x 32 -> 64 (one s_mul_hi + one s_mul; no sign-extension multiply)
#endif
#define SLOTMUL(a, b) ((int64_t)(a) * (int64_t)(b))
#define STAMP(k) do {} while (0)
#ifndef TL
#define TLDECL
#define TL(k) do {} while (0)
#define TLR(k) do {} while (0)
#define TLFLUSH() do {} while (0)
#endif
#define NBLK_ ((int)(gridDim.x * gridDim.y))
#define GRIDD(B, H, ns) dim3((H) * (ns), (B))
// X1 (v74): the plain latency-bound whole-state kernel on one XCC. The dispatch hands workgroups to the 8 XCCs round-robin and one of
// them starts 0.14-0.45 us before the others (UNDERFLOOR.md E5: physical XCC 1 on MI350X); the X1 instantiation is launched on a grid of
// 8 H x B blocks and only the blocks that find themselves on XCC X1_XCC work, on head blockIdx.x / 8. The first XCC of the round-robin
// is a property of the HSA queue the replay runs on (measured: 0, 5, 6, 7 for different streams of one process), so blockIdx alone
// cannot tell; the register can. One working block per head needs the round-robin to be exact (every 8 consecutive workgroups on 8
// different XCCs): kernel_*.py verifies that on the device with kda_xcc_probe before it asks for X1.
#ifndef X1_XCC
#define X1_XCC 1
#endif
namespace {

using bf16 = __hip_bfloat16;
using f4 = float __attribute__((ext_vector_type(4)));
using u4 = unsigned __attribute__((ext_vector_type(4)));
using f2 = float __attribute__((ext_vector_type(2)));
constexpr int D = 128;

struct Params {
  const bf16* x;        // [T, 3*dim]
  bf16* conv;           // [slots, taps, 3*dim]
  const float* cw;      // [3, 4, dim]
  const bf16* gate;     // [T, H, 128]
  const bf16* beta;     // [T, H]
  const bf16* og;       // [T, dim]
  const float* A_log;   // [H]
  const float* dt;      // [dim]
  const float* nw;      // [128]
  float* state;         // [slots, H, 128, 128]
  const int* idx;       // [B] or [B, S]
  const int* acc;       // [B] or null
  bf16* out;            // [T, dim]
  int* cnt;             // [B*H] arrival counters, zero between calls
  float* ow;            // [B*H, S, 128] fp32 o exchange
  unsigned* mb;         // [B*H*NSPLIT][NSPLIT][S] partial-sum mailboxes (sentinel between calls), V-split kda_vec
  int64_t x_stride, conv_slot_stride, state_slot_stride;
  int nslots;
  int pf;
  int nxt;   // next-round prefetch offset in blocks (0 = off), plain kda_main only
  int hold;  // read-first hold in s_memrealtime ticks (100 MHz), plain kda_main only
  int vec;
  int late;
  int H;
  int dbg;
  float eps, lb;
};

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
// bf16 -> f32 that the optimizer cannot speculate into the (conditional) load block of its operand
__device__ __forceinline__ float b2f_o(bf16 v) {
  const uint32_t u = __builtin_bit_cast(unsigned short, v);
  float f;
  asm volatile("v_lshlrev_b32 %0, 16, %1" : "=v"(f) : "v"(u));
  return f;
}
// bf16 load, nontemporal when NT: the per-(n, h) prologue inputs of (a) the plain whole-state, state-first (co-resident,
// read-first hold) kernels and (b) every spec kda_main kernel (write-heavy grids: A/B all mid/large-B spec -0.1..-0.7 us).
// Plain latency-bound and multi-round (prologue-first) kernels keep cached loads (A/B: nt there +0.3 us; NOTES_bw M10-M11). After the eval's flush the MALL
// holds dirty lines, so every MALL-allocating (default-policy) load of a cold line costs a dirty write-back; for the
// ~28 scattered 128 B lines a block reads besides its state that stretches the saturated read phase ~4x more than
// their bytes (NOTES_bw M6/M7). Shared lines (conv weights, dt, norm weight) stay cached: they hit in L2.
template <bool NT>
__device__ __forceinline__ bf16 ldb(const bf16* q) {
  if constexpr (NT) return __builtin_bit_cast(bf16, __builtin_nontemporal_load(reinterpret_cast<const unsigned short*>(q)));
  else return *q;
}
// bf16 store of the small plain outputs (conv-commit taps, out) with system-scope (sc0 sc1) policy instead of the default
// write-back, in the whole-state prologue-first plain kernels (latency-bound and multi-round grids: -0.07..-0.7 us). The
// co-resident read-first-hold grids (whole-state state-first, and V-split, whose kernels are PF instantiations) keep default stores: there the commit stores land inside the
// read phase (+2.5..+3.2 us whole-state, +0.4 us V-split 16,40,1), NOTES_bw M13.
template <bool SC>
__device__ __forceinline__ void stb(bf16* q, bf16 v) {
  if constexpr (SC) __hip_atomic_store(reinterpret_cast<unsigned short*>(q), __builtin_bit_cast(unsigned short, v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  else *q = v;
}
// small bf16 stores of the spec multi-round kernels (conv commit, out): nontemporal (SNT) — like the plain multi-round
// kernel's state stores (aux 18), a per-family policy
template <bool NTS>
__device__ __forceinline__ void stn(bf16* q, bf16 v) {
  if constexpr (NTS) __builtin_nontemporal_store(__builtin_bit_cast(unsigned short, v), reinterpret_cast<unsigned short*>(q));
  else *q = v;
}
template <bool SP, bool NTS, bool SC>
__device__ __forceinline__ void stbs(bf16* q, bf16 v) {
  if constexpr (SP) stn<NTS>(q, v);
  else stb<SC>(q, v);
}
__device__ __forceinline__ float bfround(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float lo2f(uint32_t u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float hi2f(uint32_t u) { return __builtin_bit_cast(float, u & 0xffff0000u); }
__device__ __forceinline__ float sigm(float x) { return __builtin_amdgcn_rcpf(1.0f + __expf(-x)); }

// compile-time loop: f(std::integral_constant<int, I>) for I in [I0, N)
template <int I, int N, class F>
__device__ __forceinline__ void sfor(F&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    sfor<I + 1, N>(f);
  }
}

// opaque wave-uniform pointer (SGPR pair) in the global address space: a load through it with an unsigned 32-bit index
// is (scalar base) + (32-bit lane offset), i.e. global_load vdst, voff, s[base]. Without the opacity the loop-strength
// reduction rebuilds 64-bit per-lane addresses and steps those on the VALU; without the address space it is a flat_load.
#define KDA_AS1 __attribute__((address_space(1)))
template <class T>
__device__ __forceinline__ const T KDA_AS1* opq(const T* q) {
  uint64_t u = reinterpret_cast<uint64_t>(q);
  asm volatile("" : "+s"(u));
  return reinterpret_cast<const T KDA_AS1*>(u);
}
template <bool NT>
__device__ __forceinline__ __hip_bfloat16 ldb1(const __hip_bfloat16 KDA_AS1* q) {
  if constexpr (NT) return __builtin_bit_cast(__hip_bfloat16, __builtin_nontemporal_load(reinterpret_cast<const unsigned short KDA_AS1*>(q)));
  else return __builtin_bit_cast(__hip_bfloat16, *reinterpret_cast<const unsigned short KDA_AS1*>(q));
}
// wave-uniform dispatch: f(integral_constant<int, W>) for the one W == wave, as an if / else chain (mutually exclusive
// paths: no CFG path carries one wave's pending loads into another wave's code)
template <int W, int N, class F>
__device__ __forceinline__ void wsel(int wave, F&& f) {
  if constexpr (W + 1 < N) {
    if (wave == W) f(std::integral_constant<int, W>{});
    else wsel<W + 1, N>(wave, f);
  } else if constexpr (W < N) {
    if (wave == W) f(std::integral_constant<int, W>{});
  }
}

template <int kCtrl>
__device__ __forceinline__ float dpp_add(float x) {
  const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), kCtrl, 0xf, 0xf, true);
  return x + __builtin_bit_cast(float, s);
}
// Sum over aligned groups of 16 lanes; every lane gets the total.
__device__ __forceinline__ float row_reduce(float x) {
  x = dpp_add<0xb1>(x);
  x = dpp_add<0x4e>(x);
  x = dpp_add<0x141>(x);
  x = dpp_add<0x140>(x);
  return x;
}
__device__ __forceinline__ float rl(float x, int l) {
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), l));
}
// Full-wave sum, wave-uniform result.
__device__ __forceinline__ float wave_sum(float x) {
  x = row_reduce(x);
  return (rl(x, 0) + rl(x, 16)) + (rl(x, 32) + rl(x, 48));
}

// PR_WUS: a0 of lane l: conv pool (l < 32) / state pool, one word per PFSTRIDE, clamped to the pool; bounds are opaque SGPR pairs
__device__ __forceinline__ const char* wu_a0(const void* conv, const void* state, int lane, int nslots, int css, int sss) {
  uint64_t cb = (uint64_t)((int64_t)nslots * css * 2 - 4), sb = (uint64_t)((int64_t)nslots * sss * 4 - 4);
  asm volatile("" : "+s"(cb), "+s"(sb));
  const uint64_t o0 = (uint64_t)(lane & 31) * PFSTRIDE;
  return lane < 32 ? reinterpret_cast<const char*>(conv) + (o0 < cb ? o0 : cb)
                   : reinterpret_cast<const char*>(state) + (o0 < sb ? o0 : sb);
}
#define RSQRT(x) __builtin_amdgcn_rsqf(x)
#define EPWSUM(x) wave_sum(x)
// Two 16-lane row sums with the DPP steps interleaved (every lane of a row gets its row's total)
__device__ __forceinline__ void row_reduce2(float& a, float& b) {
  a = dpp_add<0xb1>(a); b = dpp_add<0xb1>(b);
  a = dpp_add<0x4e>(a); b = dpp_add<0x4e>(b);
  a = dpp_add<0x141>(a); b = dpp_add<0x141>(b);
  a = dpp_add<0x140>(a); b = dpp_add<0x140>(b);
}
// x + (row r-1's lane 15 / lanes 31) for the rows in kMask; other rows unchanged
template <int kCtrl, int kMask>
__device__ __forceinline__ float dpp_bcast_add(float x) {
  const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), kCtrl, kMask, 0xf, false);
  return x + __builtin_bit_cast(float, s);
}
// Three full-wave sums: row sums, then row_bcast:15 (rows 1, 3) and row_bcast:31 (rows 2, 3); lane 63 has the total
__device__ __forceinline__ void wave_sum3b(float& a, float& b, float& c) {
  a = dpp_add<0xb1>(a); b = dpp_add<0xb1>(b); c = dpp_add<0xb1>(c);
  a = dpp_add<0x4e>(a); b = dpp_add<0x4e>(b); c = dpp_add<0x4e>(c);
  a = dpp_add<0x141>(a); b = dpp_add<0x141>(b); c = dpp_add<0x141>(c);
  a = dpp_add<0x140>(a); b = dpp_add<0x140>(b); c = dpp_add<0x140>(c);
  a = dpp_bcast_add<0x142, 0xa>(a); b = dpp_bcast_add<0x142, 0xa>(b); c = dpp_bcast_add<0x142, 0xa>(c);
  a = dpp_bcast_add<0x143, 0xc>(a); b = dpp_bcast_add<0x143, 0xc>(b); c = dpp_bcast_add<0x143, 0xc>(c);
  a = rl(a, 63); b = rl(b, 63); c = rl(c, 63);
}


// Three full-wave sums, lane 63 of each holds the total. One asm block, chains interleaved a, b, c: every DPP read of a
// register is two instructions after its last write (DPP needs 2 wait states after a VALU write of its source; the
// leading s_nop 1 covers whatever the compiler placed right in front). The row_bcast steps are single masked adds
// (rows outside row_mask keep their value), where the compiler emits v_mov 0 / v_mov_dpp / v_add.
__device__ __forceinline__ void wave_sum3c(float& a, float& b, float& c) {
#define KDA_DPP3(ctl) "v_add_f32_dpp %0, %0, %0 " ctl "\n v_add_f32_dpp %1, %1, %1 " ctl "\n v_add_f32_dpp %2, %2, %2 " ctl "\n "
  asm volatile("s_nop 1\n "
               KDA_DPP3("quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf")
               KDA_DPP3("quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf")
               KDA_DPP3("row_half_mirror row_mask:0xf bank_mask:0xf")
               KDA_DPP3("row_mirror row_mask:0xf bank_mask:0xf")
               KDA_DPP3("row_bcast:15 row_mask:0xa bank_mask:0xf")
               KDA_DPP3("row_bcast:31 row_mask:0xc bank_mask:0xf")
               "s_nop 1"
               : "+v"(a), "+v"(b), "+v"(c));
#undef KDA_DPP3
}

template <int S, bool SPEC>
__device__ __forceinline__ bool slots(const int* __restrict__ idx, const int* __restrict__ acc, int n, int& a, int& rd,
                                      int& cs, int (&wsl)[S]) {
  if (SPEC) {
    // acc and the whole index row are independent loads; rd is selected (no dependent load).
    a = acc[n];
#pragma unroll
    for (int t = 0; t < S; ++t) wsl[t] = idx[n * S + t];
    a = a < 1 ? 1 : a;
    asm volatile("" : "+s"(a));  // opaque SGPR: keeps smax / smin from being combined into v_med3 (VALU only)
    a = a > S ? S : a;
    rd = wsl[0];
#pragma unroll
    for (int t = 1; t < S; ++t) rd = (a - 1 == t) ? wsl[t] : rd;
    cs = wsl[0];
    return rd > 0 && cs > 0;
  }
  a = 1;
  rd = idx[n];
  cs = rd;
#pragma unroll
  for (int t = 0; t < S; ++t) wsl[t] = rd;
  return rd >= 0;
}

// Entry form of slots(): the raw slot loads (accepted count, index row) and four kernarg pointers are inputs of ONE asm
// statement (input-only: no register copies ahead of it), so all their s_loads are issued before a single lgkmcnt wait (SMEM returns out of order: any earlier
// wait would cover the slot hop anyway, any later s_load would be a second serial hop). slots_sel() is slots() on the values.
template <int S, bool SPEC, class PA, class PB, class PC, class PD>
__device__ __forceinline__ void slots_load_pin(const int* __restrict__ idx, const int* __restrict__ acc, int n, int& a,
                                               int (&w)[S], PA& pa, PB& pb, PC& pc, PD& pd) {
  if (SPEC) {
    a = acc[n];
#pragma unroll
    for (int t = 0; t < S; ++t) w[t] = idx[n * S + t];
  } else {
    a = 1;
    w[0] = idx[n];
#pragma unroll
    for (int t = 1; t < S; ++t) w[t] = w[0];
  }
  if constexpr (S == 1) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]));
  else if constexpr (S == 2) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]));
  else if constexpr (S == 3) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]));
  else if constexpr (S == 4) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]), "s"(w[3]));
  else if constexpr (S == 5) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]), "s"(w[3]), "s"(w[4]));
  else if constexpr (S == 6) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]), "s"(w[3]), "s"(w[4]), "s"(w[5]));
  else if constexpr (S == 7) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]), "s"(w[3]), "s"(w[4]), "s"(w[5]), "s"(w[6]));
  else if constexpr (S == 8) asm volatile("" :: "s"(pa), "s"(pb), "s"(pc), "s"(pd), "s"(a), "s"(w[0]), "s"(w[1]), "s"(w[2]), "s"(w[3]), "s"(w[4]), "s"(w[5]), "s"(w[6]), "s"(w[7]));
}
template <int S, bool SPEC>
__device__ __forceinline__ bool slots_sel(int& a, int& rd, int& cs, int (&wsl)[S]) {
  if (SPEC) {
    a = a < 1 ? 1 : a;
    asm volatile("" : "+s"(a));
    a = a > S ? S : a;
    rd = wsl[0];
#pragma unroll
    for (int t = 1; t < S; ++t) rd = (a - 1 == t) ? wsl[t] : rd;
    cs = wsl[0];
    return rd > 0 && cs > 0;
  }
  a = 1;
  rd = wsl[0];
  cs = rd;
  return rd >= 0;
}

template <int VB, int S, bool SPEC, int NT, bool PF = !SPEC, bool NX = false, bool SB = false, bool W1 = false, bool X1 = false>
__global__ __launch_bounds__(NT) void kda_main(const int* __restrict__ k_idx, const int* __restrict__ k_acc,
                                                float* k_state, bf16* k_conv, const bf16* __restrict__ k_x, int k_H,
                                                int k_sss, int k_css, int k_nslots, Params p) {
  constexpr int NSPLIT = D / VB;
  constexpr int NW = NT / 64;      // waves per block; waves 0-3 run the prologue
  constexpr int RG = NT / 16;      // row groups (16 lanes per row)
  constexpr int RPT = VB / RG;     // state rows per thread
  constexpr int OGN = (S + NW - 1) / NW;  // epilogue tokens per wave
  static_assert(RPT >= 1, "block too large for VB");
  constexpr int WUW_ = (PR_WUW && NW > 4) ? 4 : NW - 1;  // warm-up wave
  constexpr int NXW_ = (PR_NXW && NW > 4) ? (PR_NXW == 1 ? 4 : PR_NXW) : NW - 1;  // next-round prefetch wave (PR_NXW: 1 = wave 4, n > 1 = wave n)
  // small bf16 store policy of the plain kernels: SCP = sc0 sc1 (v54: PF whole-state 1024-thread), NTSP = nt (PR_PLBST 1, latency-bound only)
  constexpr bool SCP = !SPEC && PF && NSPLIT == 1 && NT == 1024 && !(PR_PLBST == 2 && !NX) && !(AB_NOSCP && NX);
  constexpr bool NTSP = PR_PLBST == 1 && !SPEC && PF && NSPLIT == 1 && NT == 1024 && !NX;
  constexpr bool LBL_ = (PR_LBLATE && !SPEC && PF && !NX && NSPLIT == 1 && NT == 1024) || (PR_HLATE && !SPEC && PF && !NX && NSPLIT == 1 && NT == 256);  // latency-bound plain / PF hold: late conv commit
  constexpr int LBLP_ = PR_LBLATE ? PR_LBLATE : 1;
  constexpr bool HB_ = PR_HBST && !SPEC && NT == 256 && NSPLIT == 1 && !NX;       // hold kernels: batched state stores
  constexpr bool HOLDK_ = !SPEC && !(PR_NOHOLD && NT == 1024 && NSPLIT == 1);     // kernel can hold
  constexpr bool WU0E_ = PR_MRWU0 && !SPEC && NX && NT == 1024 && BST;            // a0 ahead of the stagger
  constexpr int LCTT_ = LCT < S - 1 ? LCT : S - 1;                                 // spec late commit after this token
  constexpr bool A0_ = !(AB_NOA0 & (SPEC ? 1 : 2));                                // pool translation warm-up a0
  // prologue-load policy (nt) per class: v65 = plain whole-state 256-thread (hold) and every spec kernel
  constexpr bool NTL_ = SPEC ? !(AB_PLD & 2)
                             : ((NT == 256 && NSPLIT == 1) ? !(AB_PLD & 1)
                                                           : (NSPLIT == 1 && NT == 1024 && (((AB_PLD & 4) && !NX) || ((AB_PLD & 8) && NX))));
  constexpr bool NTB_ = AB_BETA == 1 ? false : (AB_BETA == 2 ? NTL_ : (SPEC && NX));   // beta: v65 nt only in the spec multi-round kernel
  constexpr bool SNTK_ = SPEC && SNT && (NX || SNT == 2);                          // nt small stores: spec multi-round (SNT 2: every spec kda_main)
  constexpr bool S2_ = (SPEC && !AB_NOS2) || PR_PS2;                               // norm scalars once per block
  constexpr bool SLE_ = SPEC ? bool(AB_SLOTE & 2) : !(AB_SLOTE & 1);               // slot chain at entry
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);  // wave-uniform
  const int lane = tid & 63;
  const int EW_ = (PR_EPW && NW > 4) ? (wave >= 4 ? wave - 4 : wave + NW - 4) : wave;  // epilogue wave of token 0 (PR_EPW: wave 4)
  if constexpr (PR_PRIO_MRPRO && !SPEC && NX) {
    if (wave < 4) __builtin_amdgcn_s_setprio(PR_PRIO_MRPRO);
  }
  if constexpr (PR_PRIO_SPECW && SPEC && NT == 1024) {
    if (wave < 4) __builtin_amdgcn_s_setprio(3);
    else if (wave < 8) __builtin_amdgcn_s_setprio(2);
    else if (wave < 12) __builtin_amdgcn_s_setprio(1);
  }
  if constexpr (X1) {
    static_assert(!SPEC && PF && !NX && !SB && !W1 && NSPLIT == 1 && NT == 1024, "X1: plain latency-bound whole-state kernel only");
    int x1_;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(x1_));
    if ((x1_ & 15) != X1_XCC) return;
  }
  const int H = k_H;
  const int split = NSPLIT == 1 ? 0 : (int)blockIdx.x % NSPLIT;
  const int h = X1 ? (int)blockIdx.x >> 3 : (int)blockIdx.x / NSPLIT, n = blockIdx.y;
  const int g = n * H + h;
  const int b = g * NSPLIT + split;
  const int dim = H * D;
  const int nxd_ = AB_NXMUL == 2 ? p.nxt : p.nxt * AB_NXMUL / 2;  // next-round prefetch distance
  const int c = tid & (D - 1);
  const bool qk = wave < 2;  // waves 0-1: q/k channel c. waves 2-3: decay channel c, v row (c % VB)
  const int64_t tok0 = (int64_t)n * S;
  const bf16* xrow = k_x + tok0 * p.x_stride;
  const int vrow = split * VB + (c & (VB - 1));

  // ---- 0. translation warm-up. Under the eval's 512 MB flush every dependent cold hop costs ~1.3 us, much of it
  // page walks. The history / state loads (second hop, after idx) touch unknown slots of the conv and state pools,
  // so the last wave touches one word per PFSTRIDE of both pools (and the counter / o-exchange words) right away;
  // the loads are never waited on (asm, invisible to the waitcnt pass; issued first, so in-order vmcnt stays
  // conservative) and their destination VGPRs are kept live to the end of the kernel.
  // plain: the slot chain (acc / idx from preloaded kernarg pointers) is issued at entry, so its s_load overlaps the
  // Params kernarg s_loads (one lgkmcnt(0) for both); the warm-up's a0 (preloaded pointers only) goes out before the
  // Params wait, a1 (Params pointers) later. Spec kernels keep the original order (A/B: entry slot chain loses there).
  int a, rd, cs;
  int wsl[S];
  bool slot_ok = true;
  if constexpr (SLE_) slot_ok = slots<S, SPEC>(k_idx, k_acc, n, a, rd, cs, wsl);
  uint32_t pf0 = 0, pf1 = 0;
  if constexpr (WU0E_ && A0_) {
    if (wave == WUW_ && (PR_WUB == 0 || b < PR_WUB)) {
      const char* a0 = wu_a0(k_conv, k_state, lane, k_nslots, k_css, k_sss);
      asm volatile("global_load_dword %0, %1, off" : "=v"(pf0) : "v"(a0));
    }
  }
  auto stagger_ = [&]() {
  if constexpr (!SPEC && NX && NT == 1024 && BST) {
    // First co-resident generation (blocks < p.nxt = CUs x blocks per CU): its blocks wait 0..STAG ticks once (STAGL levels,
    // hash of the sequence n, so the H heads of a sequence stay together), so co-resident blocks and CUs do not run in
    // lockstep; later generations inherit the offsets.
    if (STAG > 0 && b < p.nxt) {
      unsigned x_ = (unsigned)(AB_STAGKEY ? b : n) * 2654435761u; x_ ^= x_ >> 15; x_ *= 0x2c1b3c6dU; x_ ^= x_ >> 12;  // per sequence: its H heads stay together
#if STAGL > 2
      const unsigned d_ = (x_ % STAGL) * (STAG / (STAGL - 1));
#else
      const unsigned d_ = (x_ & 1) ? STAG : 0;
#endif
      if (d_) { const unsigned long long t_ = __builtin_amdgcn_s_memrealtime(); while (__builtin_amdgcn_s_memrealtime() < t_ + d_) __builtin_amdgcn_s_sleep(1); }
    }
  }
  };
  if constexpr (PR_STAGPOS == 0) stagger_();
  const uint64_t t_start = HOLDK_ ? __builtin_amdgcn_s_memrealtime() : 0;  // read-first hold (plain), see below
  if (wave == WUW_) {
    const int64_t cbytes = (int64_t)k_nslots * k_css * 2, sbytes = (int64_t)k_nslots * k_sss * 4;
    int64_t o0 = (int64_t)(lane & 31) * PFSTRIDE;
    const char* a0 = lane < 32 ? reinterpret_cast<const char*>(k_conv) + (o0 < cbytes ? o0 : cbytes - 4)
                               : reinterpret_cast<const char*>(k_state) + (o0 < sbytes ? o0 : sbytes - 4);
    if constexpr (WU0E_ || !A0_) { }
    else if constexpr (PR_WUB > 0) { if (b < PR_WUB) asm volatile("global_load_dword %0, %1, off" : "=v"(pf0) : "v"(a0)); }
    else if constexpr (!WU0E_) asm volatile("global_load_dword %0, %1, off" : "=v"(pf0) : "v"(a0));
    if constexpr (SPEC && !PR_SWU0 && !(PR_NOA1 & 1)) {
      const char* a1 = lane == 0 ? reinterpret_cast<const char*>(p.cnt + g)
                                 : reinterpret_cast<const char*>(p.ow + (int64_t)g * S * D);
      if (NSPLIT == 1) a1 = reinterpret_cast<const char*>(p.out + tok0 * H * D);
      asm volatile("global_load_dword %0, %1, off" : "=v"(pf1) : "v"(a1));
    }
  }
  // ---- 1. slot-independent loads (unconditional; role picks the address)
  // qk:     xa = x_q, xb = x_k, w0 = w_q, w1 = w_k
  // others: xa = gate, xb = x_v[vrow], w0 = w_v[vrow], w1 = w_v[vrow] (unused)
  const int64_t xa_off = qk ? (int64_t)(h * D + c) : 0;
  const int64_t xb_off = qk ? (int64_t)(dim + h * D + c) : (int64_t)(2 * dim + h * D + vrow);
  const int w0_off = qk ? h * D + c : 8 * dim + h * D + vrow;
  const int w1_off = qk ? 4 * dim + h * D + c : 8 * dim + h * D + vrow;
  // Only the prologue waves (0-3) load the prologue inputs, and only waves with an epilogue token load og / nw. The
  // explicit wave-uniform branches keep these loads above the slot chain (LLVM would otherwise sink them into the
  // role branches of the prologue, below the state-row loads, and in-order vmcnt would make the prologue wait for
  // the block's 64 KB state) without making the other waves issue them.
  // Values stay undefined in the waves that do not load them (never used there in control flow): a defined
  // alternative (e.g. 0) makes the compiler fold conversions / insert phi copies inside the branch, i.e. a wait.
  bf16 xa[S], xb[S];
  float w0[4], w1[4];
  float dtc, alog;
  bf16 beta_b;
  const bf16* pa = qk ? xrow + xa_off : p.gate + (tok0 * H + h) * D + c;
  const int64_t sa = qk ? p.x_stride : (int64_t)H * D;
  // CG_ML: every address is (wave-uniform pointer) + (unsigned 32-bit lane offset): the per-token / per-tap stepping
  // runs on the SALU and each value costs exactly its load instruction
  constexpr bool ML_ = CG_ML == 1 || (CG_ML == 2 && NT == 1024 && PF && !NX) || (CG_ML == 3 && SPEC && NT == 1024 && PF && !NX) ||
                       (CG_ML == 4 && SPEC && NT == 1024 && PF && !NX && !SB);
  const uint32_t cu_ = (uint32_t)c;
  const uint32_t cbu_ = qk ? cu_ : (uint32_t)(c & (VB - 1));  // q/k: channel; others: v row within the block
  if (!PF || wave < 4 || AB_ALLW) {
    if constexpr (ML_) {
      const bf16* ua = qk ? xrow + h * D : p.gate + (tok0 * H + h) * D;
      const bf16* ub = xrow + (qk ? dim : 2 * dim + split * VB) + h * D;
      const float* uw0 = p.cw + (qk ? 0 : 8 * dim + split * VB) + h * D;
      const float* uw1 = p.cw + (qk ? 4 * dim : 8 * dim + split * VB) + h * D;
#pragma unroll
      for (int t = 0; t < S; ++t) {
        xa[t] = ldb1<NTL_>(opq(ua + t * sa) + cu_);
        xb[t] = ldb1<NTL_>(opq(ub + t * p.x_stride) + cbu_);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        w0[j] = opq(uw0 + j * dim)[cbu_];
        w1[j] = opq(uw1 + j * dim)[cbu_];
      }
      dtc = opq(p.dt + h * D)[cu_];
      alog = p.A_log[h];
      beta_b = ldb1<NTB_>(opq(p.beta + tok0 * H + h) + (uint32_t)((c < S ? c : S - 1) * H));
    } else {
#pragma unroll
    for (int t = 0; t < S; ++t) {
      xa[t] = ldb<NTL_>(pa + t * sa);
      xb[t] = ldb<NTL_>(xrow + t * p.x_stride + xb_off);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      w0[j] = p.cw[j * dim + w0_off];
      w1[j] = p.cw[j * dim + w1_off];
    }
    dtc = p.dt[h * D + c];
    alog = p.A_log[h];
    if constexpr (PR_BETA32 && SPEC) beta_b = ldb<NTB_>(p.beta + (tok0 * H + h) + (int64_t)((c < S ? c : S - 1) * H));
    else beta_b = ldb<NTB_>(p.beta + (tok0 + (c < S ? c : S - 1)) * H + h);
    }
  }
  float nw_r[2];
  bf16 og_b[OGN][2];
  if (!PF || EW_ < (S < NW ? S : NW)) {
    nw_r[0] = p.nw[lane];
    nw_r[1] = p.nw[lane + 64];
#pragma unroll
    for (int i = 0; i < OGN; ++i) {
      const int t = (EW_ + NW * i) < S ? EW_ + NW * i : S - 1;
      const bf16* ogp = p.og + (tok0 + t) * dim + h * D;
      og_b[i][0] = ldb<NTL_>(ogp + lane);
      og_b[i][1] = ldb<NTL_>(ogp + lane + 64);
    }
  }

  if constexpr ((!SPEC && !(PR_NOA1 & 2)) || (SPEC && PR_SWU0 && !(PR_NOA1 & 1))) {
    if (wave == WUW_) {  // exchange-word / out warm-up (Params pointers): after the first Params use
      const char* a1 = lane == 0 ? reinterpret_cast<const char*>(p.cnt + g)
                                 : reinterpret_cast<const char*>(p.ow + (int64_t)g * S * D);
      if (NSPLIT == 1) a1 = reinterpret_cast<const char*>(p.out + tok0 * H * D);
      asm volatile("global_load_dword %0, %1, off" : "=v"(pf1) : "v"(a1));
    }
  }
  // ---- 2. slot chain
  if constexpr (PR_STAGPOS == 2) stagger_();
  if constexpr (!SLE_) slot_ok = slots<S, SPEC>(k_idx, k_acc, n, a, rd, cs, wsl);
  if (!slot_ok) {
    // fake uses pin the slot-independent loads above this branch: LLVM otherwise sinks them into the role branches
    // of the prologue, i.e. below the state-row loads, and in-order vmcnt then makes the prologue wait for the state
    // (plain only: the store-bound spec grids measured better with the state rows issued first)
    if (PF && !(AB_NOPIN & 1)) {
#pragma unroll
    for (int t = 0; t < S; ++t) asm volatile("" :: "v"(xa[t]), "v"(xb[t]));
#pragma unroll
    for (int j = 0; j < 4; ++j) asm volatile("" :: "v"(w0[j]), "v"(w1[j]));
    asm volatile("" :: "v"(dtc), "v"(alog), "v"(beta_b), "v"(nw_r[0]), "v"(nw_r[1]));
#pragma unroll
    for (int i = 0; i < OGN; ++i) asm volatile("" :: "v"(og_b[i][0]), "v"(og_b[i][1]));
    }
    for (int i = tid; i < S * VB; i += NT) {
      const int t = i / VB, v = split * VB + i % VB;
      p.out[(tok0 + t) * dim + h * D + v] = __float2bfloat16(0.0f);
    }
    asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
    return;
  }

  const int start = a - 1;
  bf16* crow = k_conv + SLOTMUL(cs, k_css);
  const int64_t ctap = 3 * (int64_t)dim;
  // conv history: qk threads q (ha) / k (hb); others v (hb), ha duplicates hb (unused)
  const int64_t ha_off = qk ? (int64_t)(h * D + c) : xb_off;
  bf16 ha[3], hb[3];
#define KDA_HIST_ \
  if (!PF || wave < 4 || AB_ALLW) { \
    if constexpr (ML_) { \
      const bf16* uha_ = crow + start * ctap + (qk ? 0 : 2 * dim + split * VB) + h * D; \
      const bf16* uhb_ = crow + start * ctap + (qk ? dim : 2 * dim + split * VB) + h * D; \
      _Pragma("unroll") \
      for (int j = 0; j < 3; ++j) { \
        ha[j] = ldb1<NTL_>(opq(uha_ + j * ctap) + cbu_); \
        hb[j] = ldb1<NTL_>(opq(uhb_ + j * ctap) + cbu_); \
      } \
    } else { \
      _Pragma("unroll") \
      for (int j = 0; j < 3; ++j) { \
        ha[j] = ldb<NTL_>(crow + (start + j) * ctap + ha_off); \
        hb[j] = ldb<NTL_>(crow + (start + j) * ctap + xb_off); \
      } \
    } \
  }
  if (!PF || wave < 4 || AB_ALLW) {
    if constexpr (ML_) {
      const bf16* uha = crow + start * ctap + (qk ? 0 : 2 * dim + split * VB) + h * D;
      const bf16* uhb = crow + start * ctap + (qk ? dim : 2 * dim + split * VB) + h * D;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        ha[j] = ldb1<NTL_>(opq(uha + j * ctap) + cbu_);
        hb[j] = ldb1<NTL_>(opq(uhb + j * ctap) + cbu_);
      }
    } else {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      ha[j] = ldb<NTL_>(crow + (start + j) * ctap + ha_off);
      hb[j] = ldb<NTL_>(crow + (start + j) * ctap + xb_off);
    }
    }
  }

  // state rows after the history: vmcnt is in order, so the prologue's wait for the history must not also wait
  // for 64 KB of state (needed only in the recurrence).
  const int rlane = tid & 15, rgroup = tid >> 4;
  const float* sp = k_state + SLOTMUL(rd, k_sss) + (int64_t)h * D * D +
                    (split * VB + rgroup) * D + 4 * rlane;
  f4 hr[RPT][2];
  // multi-round plain: the state rows of all waves go out only after the prologue waves have issued their inputs and
  // conv history (the block's small loads reach the memory system ahead of its 64 KB)
  if constexpr (!SPEC && PF && NX && NSPLIT == 1 && SBAR) __syncthreads();
  if constexpr (PR_STAGPOS == 1) stagger_();
  // spec whole-state prologue-first grid of one round (1024 threads): the same ordering, chosen by the cost model
  // (state_after_prologue() in kernel_*.py, passed in p.hold, which the spec kernels do not use otherwise). The
  // prologue waves' loads are order-critical against the 64 KB state burst of all 16 waves (delay injection: delaying
  // only the prologue waves costs 1.3-1.5x the delay, delaying only the state-row waves is free up to ~0.5 us), so
  // while the state read phase is shorter than the prologue chain the state rows go out behind a block barrier.
  // (SB: its own instantiation - a runtime branch here costs the grids that do not take it +0.06..+0.18 us)
  if constexpr (SB) __syncthreads();
#pragma unroll
  for (int i = 0; i < RPT; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#if BLD
      hr[i][j] = __builtin_bit_cast(f4, __builtin_amdgcn_raw_buffer_load_b128(
          __builtin_amdgcn_make_buffer_rsrc(k_state + SLOTMUL(rd, k_sss) + (int64_t)h * D * D + split * VB * D, 0, VB * D * 4, 0x00020000),
          ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, BLDAUX));
#else
      hr[i][j] = __builtin_nontemporal_load(reinterpret_cast<const f4*>(sp + i * RG * D + 64 * j));
#endif
    }
#undef KDA_HIST_

  // NX (multi-round plain grids, cost model next_round()): block b + p.nxt runs on this block's XCD one round later
  // (round-robin dispatch; 'blocks per round' is a multiple of 8). Wave NW-1 pulls that block's slot index, conv
  // history and inputs (~3.6 KB, not the state) into the XCD's L2 with LDS-direct loads into a dummy LDS sink (no VGPR
  // destination, so no register hazard), shortening the next block's dependent first hops.
  if constexpr (NX && !SPEC && !(AB_NONX & 1)) {
    __shared__ float s_pfsink[64];
    if (wave == NXW_ && b + nxd_ < NBLK_) {
      if constexpr (PR_PRIO_NX) __builtin_amdgcn_s_setprio(3);
      const unsigned pfm = (unsigned)(uintptr_t)s_pfsink;
      const int gn = b + nxd_, hn = gn % H, nn = gn / H;
      const int rdn = k_idx[nn];
      if (rdn >= 0) {
        const bf16* crn = k_conv + SLOTMUL(rdn, k_css);
        const bf16* xrn = k_x + (int64_t)nn * p.x_stride;
        const int seg = lane >> 1, half = (lane & 1) * 64;
        const bf16* pa;
        if (seg < 9) pa = crn + (int64_t)(seg / 3) * (3 * dim) + (seg % 3) * dim + hn * D + half;
        else if (seg < 12) pa = xrn + (seg - 9) * dim + hn * D + half;
        else if (seg == 12) pa = p.gate + ((int64_t)nn * H + hn) * D + half;
        else pa = p.og + (int64_t)nn * dim + hn * D + half;
        asm volatile("s_mov_b32 m0, %1\n global_load_lds_dword %0, off" KDA_NXNT_ :: "v"(pa), "s"(pfm) : "memory", "m0");
      }
      if constexpr (PR_PRIO_NX) __builtin_amdgcn_s_setprio(0);
    }
  }
  __shared__ float s_q[S][D], s_k[S][D], s_d[S][D];
  __shared__ float s_v[S][VB];
  __shared__ float s_o[S][VB];  // o of this block's rows (all tokens)
  __shared__ float s_red[2][3 * S];
  __shared__ float s_beta[S];
  __shared__ int s_last;
  __shared__ __attribute__((aligned(16))) float s_nrm2[(S2_) ? S : 1][4];  // S2: rq * scale, rk, kq, beta per token

  // ---- 3. prologue math
  const float ea = __expf(alog);
  const float beta_raw = b2f(beta_b);
  float og_r[OGN][2];
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    og_r[i][0] = b2f(og_b[i][0]);
    og_r[i][1] = b2f(og_b[i][1]);
  }
  if (wave >= 4) {
  } else if (qk) {
#pragma unroll
    for (int t = 0; t < S; ++t) {
      float q = 0.f, k = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        q += (t + j < 3 ? (!PF ? b2f(ha[t + j]) : b2f_o(ha[t + j])) : b2f(xa[t + j - 3])) * w0[j];
        k += (t + j < 3 ? (!PF ? b2f(hb[t + j]) : b2f_o(hb[t + j])) : b2f(xb[t + j - 3])) * w1[j];
      }
      q = q * sigm(q);
      k = k * sigm(k);
      s_q[t][c] = q;
      s_k[t][c] = k;
      const float rq = wave_sum(q * q), rk = wave_sum(k * k), rqk = wave_sum(q * k);
      if (lane == (PR_WS3L ? 63 : 0)) {
        s_red[wave][3 * t] = rq;
        s_red[wave][3 * t + 1] = rk;
        s_red[wave][3 * t + 2] = rqk;
      }
    }
    if (!(SPEC && PF && NT == 1024 && S >= LCS) && !(PR_MRLATE && !SPEC && NX) && !LBL_ && NSPLIT == 1) {  // long-window spec 1024-thread PF kernels commit after the token loop
      // q/k taps: [old start+1, old start+2, x_0 .. x_{S-1}] -> taps [0, S + 2)
#pragma unroll
      for (int t = 0; t < S + 2; ++t) {
        stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(crow + t * ctap + ha_off, t < 2 ? ha[t + 1] : xa[t - 2]);
        stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(crow + t * ctap + xb_off, t < 2 ? hb[t + 1] : xb[t - 2]);
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < S; ++t) s_d[t][c] = __expf(p.lb * sigm(ea * (b2f(xa[t]) + dtc)));
    if (c < VB) {
#pragma unroll
      for (int t = 0; t < S; ++t) {
        float v = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) v += (t + j < 3 ? (!PF ? b2f(hb[t + j]) : b2f_o(hb[t + j])) : b2f(xb[t + j - 3])) * w0[j];
        if constexpr ((SPEC || PR_SVC) && RPT >= 2) s_v[t][(c % RG) * RPT + c / RG] = v * sigm(v);  // thread-contiguous rows
        else s_v[t][c] = v * sigm(v);
      }
      if (!(SPEC && PF && NT == 1024 && S >= LCS) && !(PR_MRLATE && !SPEC && NX) && !LBL_)
#pragma unroll
        for (int t = 0; t < S + 2; ++t) stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(crow + t * ctap + xb_off, t < 2 ? hb[t + 1] : xb[t - 2]);
    }
    if (c < S) s_beta[c] = sigm(beta_raw);
  }
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    og_r[i][0] = sigm(og_r[i][0]);
    og_r[i][1] = sigm(og_r[i][1]);
  }
  __syncthreads();
  if constexpr (PR_PRIO_MRPRO && !SPEC && NX) {
    if (wave < 4) __builtin_amdgcn_s_setprio(0);
  }

  // Read-first hold (plain, fully co-resident grids; p.hold from the cost model in kernel_*.py): every block holds
  // its state stores until ~ the time the DRAM needs to serve all blocks' state reads. Otherwise the early blocks'
  // writes interleave with the remaining reads and the last blocks receive their state ~2x later (stamps: state
  // arrival p50 13 us, max 28 us at 1024 groups), leaving a long tail of late writers.
  if (HOLDK_ && p.hold > 0)
    while (__builtin_amdgcn_s_memrealtime() < t_start + (uint64_t)p.hold) __builtin_amdgcn_s_sleep(PR_HSLEEP);
  // ---- 4. recurrence
  const float scale = 0.08838834764831845f;
  if constexpr (S2_) {  // S2: per-token norm scalars once (wave 0, lane t), not in every wave for every token
    if (wave == 0 && lane < S) {
      const int t = lane;
      const float sq = s_red[0][3 * t] + s_red[1][3 * t];
      const float sk = s_red[0][3 * t + 1] + s_red[1][3 * t + 1];
      const float sqk = s_red[0][3 * t + 2] + s_red[1][3 * t + 2];
      const float rq = __builtin_amdgcn_rsqf(sq + 1e-6f) * scale;
      const float rk = __builtin_amdgcn_rsqf(sk + 1e-6f);
      *reinterpret_cast<f4*>(&s_nrm2[t][0]) = f4{rq, rk, sqk * rq * rk, s_beta[t]};
    }
    __syncthreads();
  }
  float* const sbase = k_state + (int64_t)h * D * D + (split * VB + rgroup) * D + 4 * rlane;
#if TP_WS
  // WS: in the spec multi-round kernel the upper half of the block's waves enters the token loop TP_WS x s_sleep(1)
  // (~29 ns each) after the lower half, once per block (NOTES_tpmodel.md): all 16 waves otherwise run the token steps
  // (LDS reads -> row math -> 4 store issues) in phase behind the norm barrier.
  if constexpr (SPEC && S >= 2 && NX && PF && NT == 1024 && NSPLIT == 1) {  // S = 1: a single token step, nothing to offset
    // WM (NOTES_xcd.md): not only the first generation starts in phase. The 32 blocks of an XCD's generation end within < 1 us of
    // each other, so every later generation re-forms the lock per XCD (per-replay stamps: the slow replays lose 7 us GPU-wide in the
    // token loop of generation 1, where v67 has no offset). The offset therefore goes to every block that has a successor on its
    // CU (b < blocks - p.nxt); the tail blocks (no successor) run alone at the end and would pay it 1 : 1 (NOTES_tpmodel.md).
    if (wave >= NW / 2 && (AB_WSALL || b < p.nxt || (TP_WSM && S >= TP_WSM && b < NBLK_ - p.nxt))) { for (int i_ = 0; i_ < TP_WS; ++i_) __builtin_amdgcn_s_sleep(1); }
  }
#endif
#if TP_WS && TP_WS1
  // W1: the same offset in the single-round spec kernel, as its own instantiation (launched where the cost model wave_offset() in
  // kernel_*.py asks for it, p.hold < 0): a run-time branch here cost the small S = 4 grids that do not take it +0.1..+0.4 us.
  if constexpr (W1 && !NX) {
    if (wave >= NW / 2) { for (int i_ = 0; i_ < TP_WS; ++i_) __builtin_amdgcn_s_sleep(1); }
  }
#endif
  // TN: the token loop (the KDA_TOKLOOP_BODY section at the top of this file, included textually) exists twice in the spec
  // multi-round kernel (S >= ST_TNS), once per state-store cache policy (SPOL_SPEC / ST_POL), behind ONE block-uniform branch: a branch
  // inside the token step is a scheduling barrier between a thread's two rows (NOTES_tpmodel VB: +2.3..+4.8 us), a second
  // copy of the loop is not. Every other instantiation includes the body once, unchanged.
  constexpr bool TN_ = ST_TN && SPEC && (S >= ST_TNS || W1) && NX && PF && NT == 1024 && NSPLIT == 1;  // W1 on a multi-round kernel = the two-copy form for S < ST_TNS
  if constexpr (TN_) {
    // p.hold (not used by the spec multi-round kernel otherwise) = first selected block b0 (0 = none): blocks b >= b0 on odd XCDs
    int xcc_;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_));
    const bool tn_ = (unsigned)(p.hold - 1) < (unsigned)b && ST_PM;
    // (the two arms start and end with different (empty) asm statements: otherwise the compiler hoists the code the
    // copies have in common above the branch, i.e. everything up to the first store of token 0, and sinks their common
    // tail - 128 VGPRs and stack spills in the S >= 7 instantiations)
    if (tn_) {
      asm volatile("; TN: token loop, state stores with policy ST_POL");
#define KDA_TOKLOOP_BODY
#define KDA_TOKPOL ST_POL
#include KDA_SELF
#undef KDA_TOKPOL
#undef KDA_TOKLOOP_BODY
      asm volatile("; TN: end of the ST_POL copy");
    } else {
      asm volatile("; TN: token loop, state stores with policy SPOL_SPEC");
#define KDA_TOKLOOP_BODY
#define KDA_TOKPOL SPOL_SPEC
#include KDA_SELF
#undef KDA_TOKPOL
#undef KDA_TOKLOOP_BODY
      asm volatile("; TN: end of the SPOL_SPEC copy");
    }
  } else {
#define KDA_TOKLOOP_BODY
#define KDA_TOKPOL SPOL_SPEC
#include KDA_SELF
#undef KDA_TOKPOL
#undef KDA_TOKLOOP_BODY
  }

  // ---- 5. epilogue
  if (NSPLIT > 1) {
    // o of all tokens goes out as full 16 B system-scope (sc0 sc1) stores from wave 0, which alone waits for them
    // (and for its own earlier state stores) before the arrival atomic; the other waves never wait on stores.
    __syncthreads();
    if (wave == 0) {
      const auto orsrc = __builtin_amdgcn_make_buffer_rsrc(p.ow + (int64_t)g * S * D, 0, 0x7fffffff, 0x00020000);
      for (int e = lane * 4; e < S * VB; e += 256) {
        const int t = e / VB, v = e % VB;
        __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u4, *reinterpret_cast<const f4*>(&s_o[t][v])),
                                               orsrc, (t * D + split * VB + v) * 4, 0, 17);
      }
      __builtin_amdgcn_s_waitcnt(0);
      if (lane == 0) {
        const int old = __hip_atomic_fetch_add(p.cnt + g, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old == NSPLIT - 1;
      }
      constexpr int t = S - 1;
      if (SPEC ? (wsl[t] > 0) : true) {
        float* ubase = k_state + SLOTMUL(wsl[t], k_sss) + (int64_t)h * D * D + split * VB * D;
#pragma unroll
        for (int i = 0; i < RPT; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            __builtin_amdgcn_raw_buffer_store_b128(
                __builtin_bit_cast(u4, hr[i][j]), __builtin_amdgcn_make_buffer_rsrc(ubase, 0, 0x7fffffff, 0x00020000),
                ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPEC ? SPOL_SPEC : SPOL_PLAIN);
      }
    }
    __syncthreads();
    if (!s_last) {
      asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
      return;
    }
    if (tid == 0) __hip_atomic_store(p.cnt + g, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (qk) {
#pragma unroll
      for (int t = 0; t < S + 2; ++t) {
        stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(crow + t * ctap + ha_off, t < 2 ? ha[t + 1] : xa[t - 2]);
        stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(crow + t * ctap + xb_off, t < 2 ? hb[t + 1] : xb[t - 2]);
      }
    }
  } else {
    if constexpr (PR_OGS) {
#pragma unroll
      for (int i = 0; i < OGN; ++i) asm volatile("" :: "v"(og_r[i][0]), "v"(og_r[i][1]));
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    const int t = EW_ + NW * i;
    if (t < S) {
      float o0, o1;
      if (NSPLIT == 1) {
        o0 = bfround(s_o[t][lane]);
        o1 = bfround(s_o[t][lane + 64]);
      } else {
        const float* wp = p.ow + ((int64_t)g * S + t) * D;
        o0 = bfround(__hip_atomic_load(wp + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        o1 = bfround(__hip_atomic_load(wp + lane + 64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
      }
      const float ss = EPWSUM(o0 * o0 + o1 * o1);
      const float rstd = RSQRT(ss * (1.0f / D) + p.eps);
      bf16* op = p.out + (tok0 + t) * dim + h * D;
      stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(op + lane, __float2bfloat16(o0 * rstd * nw_r[0] * og_r[i][0]));
      stbs<SPEC || NTSP, SNTK_ || NTSP, SCP>(op + lane + 64, __float2bfloat16(o1 * rstd * nw_r[1] * og_r[i][1]));
    }
  }
  if constexpr (LBL_ && LBLP_ == 2) {
      if (wave < 2) {
#pragma unroll
        for (int u = 0; u < S + 2; ++u) {
          stbs<false, NTSP, SCP>(crow + u * ctap + ha_off, u < 2 ? ha[u + 1] : xa[u - 2]);
          stbs<false, NTSP, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
        }
      } else if (wave < 4 && c < VB) {
#pragma unroll
        for (int u = 0; u < S + 2; ++u) stbs<false, NTSP, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
      }
  }
  if constexpr ((PR_LCEP && SPEC && PF && NT == 1024 && S >= LCS && NSPLIT == 1) || (PR_MRLATE == 3 && !SPEC && NX && NSPLIT == 1)) {
    // conv commit after the epilogue out stores
    if (wave < 2) {
#pragma unroll
      for (int u = 0; u < S + 2; ++u) {
        stbs<SPEC, SNTK_, SCP>(crow + u * ctap + ha_off, u < 2 ? ha[u + 1] : xa[u - 2]);
        stbs<SPEC, SNTK_, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
      }
    } else if (wave < 4 && c < VB) {
#pragma unroll
      for (int u = 0; u < S + 2; ++u) stbs<SPEC, SNTK_, SCP>(crow + u * ctap + xb_off, u < 2 ? hb[u + 1] : xb[u - 2]);
    }
  }
  asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
}

// Prologue staging layout (bytes) in LDS: raw inputs loaded as 16 B chunks.
template <int VB, int S>
struct Stage {
  static constexpr int W = 0;                        // float [8][128]: (group q/k, tap) x channel
  static constexpr int WV = W + 8 * 128 * 4;         // float [4][VB]
  static constexpr int DT = WV + 4 * VB * 4;         // float [128]
  static constexpr int X = DT + 128 * 4;             // bf16 [S][2][128]
  static constexpr int XV = X + S * 2 * 128 * 2;     // bf16 [S][VB]
  static constexpr int G = XV + S * VB * 2;          // bf16 [S][128]
  static constexpr int HQ = G + S * 128 * 2;         // bf16 [3][2][128]
  static constexpr int HV = HQ + 3 * 2 * 128 * 2;    // bf16 [3][VB]
  static constexpr int BYTES = HV + 3 * VB * 2;
  // chunk counts (16 B each)
  static constexpr int NW_ = 256, NWV = VB, NDT = 32, NX = 32 * S, NXV = S * VB / 8, NG = 16 * S;
  static constexpr int NA = NW_ + NWV + NDT + NX + NXV + NG;  // first hop (slot independent)
  static constexpr int NB = 96 + 3 * VB / 8;                   // second hop (conv history)
  // A "run" is a maximal range of chunk ids whose 16 B chunks are contiguous in memory (one weight row, one token's q
  // or k channels, ...). run start of first-hop chunk i / history chunk i:
  static constexpr int run_a(int i) {
    constexpr int A1 = NW_, A2 = A1 + NWV, A3 = A2 + NDT, A4 = A3 + NX, A5 = A4 + NXV;
    if (i < A1) return i & ~31;
    if (i < A2) return A1 + (i - A1) / (VB / 4) * (VB / 4);
    if (i < A3) return A2;
    if (i < A4) return A3 + ((i - A3) & ~15);
    if (i < A5) return A4 + (i - A4) / (VB / 8) * (VB / 8);
    return A5 + ((i - A5) & ~15);
  }
  static constexpr int run_b(int i) { return i < 96 ? (i & ~15) : 96 + (i - 96) / (VB / 8) * (VB / 8); }
  struct Runs { int n; int j[64]; };  // lane index at which each run of a wave's 64 chunks starts (j[0] = 0)
  static constexpr Runs runs(int i0, int cnt, bool hist) {
    Runs r{};
    r.n = 1;
    r.j[0] = 0;
    for (int j = 1; j < 64 && i0 + j < cnt; ++j)
      if ((hist ? run_b(i0 + j) : run_a(i0 + j)) == i0 + j) r.j[r.n++] = j;
    return r;
  }
};

template <int VB, int S, bool SPEC, int NT, bool LATE>
__global__ __launch_bounds__(NT + (VB < D ? 64 : 0)) void kda_vec(const int* __restrict__ k_idx, const int* __restrict__ k_acc,
                                                float* k_state, bf16* k_conv, const bf16* __restrict__ k_x, int k_H,
                                                int k_sss, int k_css, int k_nslots, Params p) {
  constexpr int NSPLIT = D / VB;
  constexpr int NW = NT / 64;             // waves per block
  constexpr int TT = 16 * VB < NT ? 16 * VB : NT;  // token-loop threads (16 lanes per state row)
  constexpr int RG = TT / 16;             // row groups
  constexpr int RPT = VB / RG;            // state rows per thread
  constexpr int OGN = (S + NW - 1) / NW;  // epilogue tokens per wave
  constexpr int TPW = (S + NW - 1) / NW;  // prologue tokens per wave (token t -> wave t % NW)
  constexpr int VPL = (VB + 63) / 64;     // v rows per lane in the prologue
  using St = Stage<VB, S>;
  static_assert(RPT >= 1, "block too large for VB");
  static_assert(VB >= 16, "v chunks need >= 8 rows per tap");
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);  // wave-uniform
  const int lane = tid & 63;
  const int H = k_H;
  const int split = NSPLIT == 1 ? 0 : (int)blockIdx.x % NSPLIT;
  const int h = (int)blockIdx.x / NSPLIT, n = blockIdx.y;
  const int g = n * H + h;
  const int b = g * NSPLIT + split;
  const int dim = H * D;
  const int64_t tok0 = (int64_t)n * S;
  const int64_t xs = p.x_stride;
  const int vr0 = split * VB;  // first state row of this block
  __shared__ __attribute__((aligned(16))) char stg[St::BYTES];
  __shared__ float s_q[S][D], s_k[S][D], s_d[S][D];
  __shared__ float s_v[S][VB];
  __shared__ float s_o[S][VB];   // o of this block's rows (all tokens)
  __shared__ float s_nrm[S][4];  // rq * scale, rk, (q.k) * rq * rk
  __shared__ float s_beta[S];
  __shared__ int s_last;
  __shared__ __attribute__((aligned(16))) int s_rdy[8];
  // Prologue token t runs on wave t. With more tokens than token waves (one per SIMD), a younger wave shares its SIMD with a
  // token wave's own prologue token and finishes ~0.2 us later (age-ordered issue); the token waves then idle at the
  // prologue barrier. Replace that barrier by LDS ready flags: the loop starts when the token waves' own tokens are
  // announced and waits for the younger waves' tokens only before it first reads them (NOTES_b1model.md).
  constexpr int NBS = TT / 64;  // flags only when S > NBS (else the prologue barrier: nothing to gain, a poll costs more)
  constexpr bool NBF = NSPLIT > 1 && TT < NT && S > NBS && !AB_NONBF;  // no prologue barrier: token t's operands are announced by s_rdy[t]
  constexpr int NBW = TT / 64;  // tokens awaited before token 0
  if (wave == 0) STAMP(0);

  // ---- V-split exchange wave (wave NW): its own program, three block barriers like the token waves
  constexpr bool POLL = NSPLIT > 1;
  if constexpr (POLL) {
    if (wave == NW) {
      TLDECL
      TL(0);
      TLR(20);
      constexpr int NSS = NSPLIT * S;  // <= 64
      constexpr uint32_t SENT = 0xbf800000u;  // -1.0f: a sum of squares is >= 0 or NaN (canonicalized below)
      constexpr int OPL = (S * VB + 63) / 64;  // outputs per lane (consecutive in t * VB + v)
      // translation warm-up of the conv / state pools (lanes < 32 / >= 32) and of this block's mailbox
      uint32_t pf0, pf1;
      {
        const int64_t cbytes = (int64_t)k_nslots * k_css * 2, sbytes = (int64_t)k_nslots * k_sss * 4;
        int64_t o0 = (int64_t)(lane & 31) * PFSTRIDE;
        const char* a0 = lane < 32 ? reinterpret_cast<const char*>(k_conv) + (o0 < cbytes ? o0 : cbytes - 4)
                                   : reinterpret_cast<const char*>(k_state) + (o0 < sbytes ? o0 : sbytes - 4);
        if constexpr (PR_VNOA & 2) pf0 = 0;
        else asm volatile("global_load_dword %0, %1, off" : "=v"(pf0) : "v"(a0));
        if constexpr (PR_V16WU0) __builtin_amdgcn_sched_barrier(0);
        const char* a1 = reinterpret_cast<const char*>(p.mb + (int64_t)(g * NSPLIT + split) * NSS);
        if constexpr (PR_VNOA & 1) pf1 = 0;
        else asm volatile("global_load_dword %0, %1, off" : "=v"(pf1) : "v"(a1));
      }
      float xog[OPL], xnw[OPL];
#pragma unroll
      for (int k = 0; k < OPL; ++k) {
        int e = lane * OPL + k;
        e = e < S * VB ? e : S * VB - 1;
        const int t = e / VB, v = vr0 + e % VB;
        xog[k] = b2f(p.og[(tok0 + t) * dim + h * D + v]);
        xnw[k] = p.nw[v];
      }
      int a, rd, cs;
      int wsl[S];
      const bool ok = slots<S, SPEC>(k_idx, k_acc, n, a, rd, cs, wsl);
#if MB_CLR
      // self-healing mailboxes (REDTEAM R2): every block returns the words it is going to publish (its own slots in every
      // sibling's mailbox; nobody else writes them but the reader's reset) to the sentinel before it publishes, so a word left
      // behind by an aborted or overlapped launch cannot outlive the next call. Position / lane mapping: MB_CLR above.
#if MB_CLR == 1
      if (lane < NSS)
        __hip_atomic_store(p.mb + ((int64_t)(g * NSPLIT + lane / S) * NSPLIT + split) * S + lane % S, SENT, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#elif MB_CLR == 2 || MB_CLR == 5 || MB_CLR == 7
      if (lane < NSS && (MB_CLR != 5 || !(VB == 16 && S >= MB_LATE_S)))
        __hip_atomic_store(p.mb + ((int64_t)(g * NSPLIT + (lane & (NSPLIT - 1))) * NSPLIT + split) * S + lane / NSPLIT, SENT, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#endif
#endif
      __syncthreads();  // staging
#if MB_CLR == 4 || MB_CLR == 5
      if (lane < NSS && (MB_CLR == 4 || (VB == 16 && S >= MB_LATE_S)))
        __hip_atomic_store(p.mb + ((int64_t)(g * NSPLIT + (lane & (NSPLIT - 1))) * NSPLIT + split) * S + lane / NSPLIT, SENT, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#endif
      if constexpr (!NBF) __syncthreads();  // prologue math
      if (!ok) {  // the token waves wrote zeros and returned before the token loop (no third barrier)
        asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
        return;
      }
#if MB_CLR == 3 || MB_CLR == 7
      if (lane < NSS)
        __hip_atomic_store(p.mb + ((int64_t)(g * NSPLIT + (lane & (NSPLIT - 1))) * NSPLIT + split) * S + lane / NSPLIT, SENT, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
#endif
      __syncthreads();  // token loop done: s_o complete
      TL(15);
      // partial sum of squares of this block's (bf16-rounded) o per token: lane -> (t = lane / 8, rows lane % 8 + 8 r)
      const int pt = lane >> 3, pq = lane & 7;
      float ps = 0.f;
#pragma unroll
      for (int r = 0; r < VB / 8; ++r) {
        const float o = bfround(s_o[pt < S ? pt : 0][pq + 8 * r]);
        ps += o * o;
      }
      ps = dpp_add<0xb1>(ps);
      ps = dpp_add<0x4e>(ps);
      ps = dpp_add<0x141>(ps);  // every lane of an 8-lane group holds the group sum
      // publish into every sibling's mailbox: lane L = i * S + t -> mailbox (g, i), slot (split, t)
      const int Lt = lane % S, di = lane / S;
      const float pv = __shfl(ps, 8 * Lt);
      const uint32_t pb = pv >= 0.f ? __float_as_uint(pv) : 0x7fc00000u;
      if (lane < NSS)
        __hip_atomic_store(p.mb + ((int64_t)(g * NSPLIT + di) * NSPLIT + split) * S + Lt, pb, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
      TL(16);
      // poll this block's mailbox until every sibling's partials are in
      unsigned* mine = p.mb + (int64_t)(g * NSPLIT + split) * NSS;
      uint32_t v;
      do {
        v = lane < NSS ? __hip_atomic_load(mine + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) : 0u;
      } while (__builtin_amdgcn_ballot_w64(v == SENT) != 0);
      TL(17);
      if constexpr (!PR_V16RST) { if (lane < NSS) __hip_atomic_store(mine + lane, SENT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }  // reset
      __shared__ float s_ss[64];
      s_ss[lane] = __uint_as_float(v);
#pragma unroll
      for (int k = 0; k < OPL; ++k) {
        const int e = lane * OPL + k;
        if (e < S * VB) {
          const int t = e / VB, vv = e % VB;
          float ss = 0.f;
#pragma unroll
          for (int i = 0; i < NSPLIT; ++i) ss += s_ss[i * S + t];
          const float rstd = RSQRT(ss * (1.0f / D) + p.eps);
          const float o = bfround(s_o[t][vv]);
          p.out[(tok0 + t) * dim + h * D + vr0 + vv] = __float2bfloat16(o * rstd * xnw[k] * sigm(xog[k]));
        }
      }
      if constexpr (PR_V16RST) { if (lane < NSS) __hip_atomic_store(mine + lane, SENT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }  // reset after the out stores
      // split 0 commits the q/k conv taps: every sibling has published, so all history reads are done
      if (split == 0) {
        bf16* crow = k_conv + SLOTMUL(cs, k_css);
        const int64_t ctap = 3 * (int64_t)dim;
        constexpr int NQ = (S + 2) * 32;
        const auto crs = __builtin_amdgcn_make_buffer_rsrc(crow, 0, (int)((S + 2) * ctap * 2), 0x00020000);
#pragma unroll
        for (int it = 0; it < (NQ + 63) / 64; ++it) {
          const int i0 = lane + it * 64;
          const int i = i0 < NQ ? i0 : 0;
          const int T = i >> 5, gq = (i >> 4) & 1, c = (i & 15) * 8;
          const u4 vq = *reinterpret_cast<const u4*>(T < 2 ? stg + St::HQ + (((T + 1) * 2 + gq) * 128 + c) * 2
                                                           : stg + St::X + (((T - 2) * 2 + gq) * 128 + c) * 2);
          const uint32_t off = i0 < NQ ? (uint32_t)((T * ctap + gq * dim + h * D + c) * 2) : 0x7ffffff0u;
          __builtin_amdgcn_raw_buffer_store_b128(vq, crs, off, 0, PR_V16OUTNT ? 2 : 0);
        }
      }
      TL(19);
      TLR(21);
      TLFLUSH();
      asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
      return;
    }
  }

  // ---- 0. translation warm-up of the conv / state pools and the exchange words (see kda_opt7.cu)
  uint32_t pf0 = 0, pf1 = 0;
  if (!POLL && wave == NW - 1) {
    const int64_t cbytes = (int64_t)k_nslots * k_css * 2, sbytes = (int64_t)k_nslots * k_sss * 4;
    int64_t o0 = (int64_t)(lane & 31) * PFSTRIDE;
    const char* a0 = lane < 32 ? reinterpret_cast<const char*>(k_conv) + (o0 < cbytes ? o0 : cbytes - 4)
                               : reinterpret_cast<const char*>(k_state) + (o0 < sbytes ? o0 : sbytes - 4);
    const char* a1 = lane == 0 ? reinterpret_cast<const char*>(p.cnt + g)
                               : reinterpret_cast<const char*>(p.ow + (int64_t)g * S * D);
    if (NSPLIT == 1) a1 = reinterpret_cast<const char*>(p.out + tok0 * H * D);
    asm volatile("global_load_dword %0, %1, off" : "=v"(pf0) : "v"(a0));
    asm volatile("global_load_dword %0, %1, off" : "=v"(pf1) : "v"(a1));
  }

  // ---- 1. slot-independent inputs: 16 B chunks -> LDS (chunk id -> segment by compile-time bounds)
  constexpr int RA = (St::NA + NT - 1) / NT;
  u4 ca[RA];
  int cad[RA];
#if CG_FH
  // every Params field of the staging addresses is read here (one kernarg wait), not inside the per-wave paths
  const float* pcw = p.cw;
  const float* pdt = p.dt;
  const bf16* pgate = p.gate;
  const bf16* pbeta = p.beta;
#if CG_SL
  int a, rd, cs;
  int wsl[S];
  slots_load_pin<S, SPEC>(k_idx, k_acc, n, a, wsl, pcw, pdt, pgate, pbeta);
  const bool sok = slots_sel<S, SPEC>(a, rd, cs, wsl);
#else
  asm volatile("" : "+s"(pcw), "+s"(pdt), "+s"(pgate), "+s"(pbeta));
#endif
  constexpr int RBc = (St::NB + NT - 1) / NT;
  uint32_t hoff[RBc];  // conv-history chunk byte offset from (crow + start * ctap + h * D): slot independent
  int hdst[RBc];
#pragma unroll
  for (int k = 0; k < RA; ++k) { ca[k] = u4{0u, 0u, 0u, 0u}; cad[k] = 0; }
#pragma unroll
  for (int k = 0; k < RBc; ++k) { hoff[k] = 0; hdst[k] = 0; }
  // first-hop chunk i (uniform i): which tensor it is in (0 conv weights, 1 dt bias, 2 x, 3 gate) and its byte offset
  // from that tensor's per-block base tb[] (32-bit scalar arithmetic)
  constexpr auto fh_ten = [](int i) constexpr -> int {
    constexpr int A2 = St::NW_ + St::NWV, A3 = A2 + St::NDT, A5 = A3 + St::NX + St::NXV;
    return i < A2 ? 0 : (i < A3 ? 1 : (i < A5 ? 2 : 3));
  };
  const uint32_t xs32 = (uint32_t)xs;
  auto fh_off = [&](int i) -> uint32_t {
    constexpr int A1 = St::NW_, A2 = A1 + St::NWV, A3 = A2 + St::NDT, A4 = A3 + St::NX, A5 = A4 + St::NXV;
    if (i < A1) return (uint32_t)(((i >> 5) * dim + (i & 31) * 4) * 4);
    if (i < A2) return (uint32_t)(((8 + (i - A1) / (VB / 4)) * dim + vr0 + ((i - A1) % (VB / 4)) * 4) * 4);
    if (i < A3) return (uint32_t)((i - A2) * 16);
    if (i < A4) return ((uint32_t)((i - A3) >> 5) * xs32 + (uint32_t)((((i - A3) >> 4) & 1) * dim + ((i - A3) & 15) * 8)) * 2u;
    if (i < A5) return ((uint32_t)((i - A4) / (VB / 8)) * xs32 + (uint32_t)(2 * dim + vr0 + ((i - A4) % (VB / 8)) * 8)) * 2u;
    return (uint32_t)((((i - A5) >> 4) * dim + ((i - A5) & 15) * 8) * 2);
  };
  auto fh_base = [&](int ten) -> uint64_t {
    return ten == 0 ? (uint64_t)(pcw + h * D) : (ten == 1 ? (uint64_t)(pdt + h * D) : (ten == 2 ? (uint64_t)(k_x + tok0 * xs + h * D)
                                                                                         : (uint64_t)(pgate + (tok0 * H + h) * D)));
  };
  // byte offset of history chunk i from (crow + start * ctap + h * D), uniform i
  auto hh_off = [&](int i) -> uint32_t {
    if (i < 96) return (uint32_t)((((i >> 5) * 3 + ((i >> 4) & 1)) * dim + (i & 15) * 8) * 2);
    return (uint32_t)((((i - 96) / (VB / 8) * 3 + 2) * dim + vr0 + ((i - 96) % (VB / 8)) * 8) * 2);
  };
  // one compile-time instantiation per wave (wave-uniform if / else chain): a wave's 64 chunks are a few runs; per lane
  // the address is (run base selected by lane thresholds) + lane * 16, the LDS destination is chunk id * 16. The run
  // bases are opaque SGPR pairs: selects of plain registers stay v_cndmask (computed operands become exec-mask branches).
  wsel<0, NW>(wave, [&](auto wc) {
    constexpr int w = decltype(wc)::value;
    sfor<0, RA>([&](auto kc) {
      constexpr int k = decltype(kc)::value, i0 = w * 64 + k * NT;
      if constexpr (i0 < St::NA) {
        constexpr int n = St::NA - i0 < 64 ? St::NA - i0 : 64;
        static constexpr typename St::Runs R = St::runs(i0, St::NA, false);
        const uint32_t li = n < 64 ? (uint32_t)(lane < n ? lane : n - 1) : (uint32_t)lane;
        uint32_t ro[R.n];
#pragma unroll
        for (int r = 0; r < R.n; ++r) {
          ro[r] = r == 0 ? fh_off(St::run_a(i0)) + (uint32_t)((i0 - St::run_a(i0)) * 16) : fh_off(i0 + R.j[r]) - (uint32_t)(R.j[r] * 16);
          asm volatile("" : "+s"(ro[r]));
        }
        uint32_t o = ro[0];
        uint64_t tb = fh_base(fh_ten(i0));
        asm volatile("" : "+s"(tb));
#pragma unroll
        for (int r = 1; r < R.n; ++r) {
          o = lane >= R.j[r] ? ro[r] : o;
          if (fh_ten(i0 + R.j[r]) != fh_ten(i0 + R.j[r] - 1)) {  // compile-time: the run starts in another tensor
            uint64_t tn = fh_base(fh_ten(i0 + R.j[r]));
            asm volatile("" : "+s"(tn));
            tb = lane >= R.j[r] ? tn : tb;
          }
        }
        ca[k] = *reinterpret_cast<const u4 __attribute__((address_space(1)))*>(tb + (uint64_t)(o + li * 16u));  // global_load (an integer-derived pointer would be a flat_load)
        cad[k] = (int)((i0 + li) * 16u);
      }
    });
    sfor<0, RBc>([&](auto kc) {
      constexpr int k = decltype(kc)::value, i0 = w * 64 + k * NT;
      if constexpr (i0 < St::NB) {
        constexpr int n = St::NB - i0 < 64 ? St::NB - i0 : 64;
        static constexpr typename St::Runs R = St::runs(i0, St::NB, true);
        const uint32_t li = n < 64 ? (uint32_t)(lane < n ? lane : n - 1) : (uint32_t)lane;
        uint32_t ho[R.n];
#pragma unroll
        for (int r = 0; r < R.n; ++r) {
          ho[r] = r == 0 ? hh_off(St::run_b(i0)) + (uint32_t)((i0 - St::run_b(i0)) * 16)
                         : hh_off(i0 + R.j[r]) - (uint32_t)(R.j[r] * 16);
          asm volatile("" : "+s"(ho[r]));
        }
        uint32_t o = ho[0];
#pragma unroll
        for (int r = 1; r < R.n; ++r) o = lane >= R.j[r] ? ho[r] : o;
        hoff[k] = o + li * 16u;
        hdst[k] = (int)(St::HQ + (i0 + li) * 16u);
      }
    });
  });
#pragma unroll
  for (int k = 0; k < RBc; ++k) asm volatile("" : "+v"(hoff[k]));  // computed here, ahead of the slot wait
#else
#pragma unroll
  for (int k = 0; k < RA; ++k) {
    ca[k] = u4{0u, 0u, 0u, 0u};
    cad[k] = 0;
    // wave-uniform skip: a wave without a real chunk of this batch used to load, wait for and stage a duplicate; every
    // instruction a wave issues before the staging barrier is on the block's critical path (NOTES_b1model.md)
    if (wave * 64 + k * NT >= St::NA) continue;
    int i = tid + k * NT;
    i = i < St::NA ? i : St::NA - 1;
    const char* src;
    int dst;
    if (i < St::NW_) {
      const int gt = i >> 5, c = (i & 31) * 4;
      src = reinterpret_cast<const char*>(p.cw + gt * dim + h * D + c);
      dst = St::W + (gt * 128 + c) * 4;
    } else if (i < St::NW_ + St::NWV) {
      const int j = i - St::NW_, tap = j / (VB / 4), c = (j % (VB / 4)) * 4;
      src = reinterpret_cast<const char*>(p.cw + (8 + tap) * dim + h * D + vr0 + c);
      dst = St::WV + (tap * VB + c) * 4;
    } else if (i < St::NW_ + St::NWV + St::NDT) {
      const int c = (i - St::NW_ - St::NWV) * 4;
      src = reinterpret_cast<const char*>(p.dt + h * D + c);
      dst = St::DT + c * 4;
    } else if (i < St::NW_ + St::NWV + St::NDT + St::NX) {
      const int j = i - (St::NW_ + St::NWV + St::NDT), t = j >> 5, gq = (j >> 4) & 1, c = (j & 15) * 8;
      if constexpr (PR_V16M32) src = reinterpret_cast<const char*>(k_x + tok0 * xs + (int64_t)(t * (int)xs + gq * dim + h * D + c));
      else src = reinterpret_cast<const char*>(k_x + (tok0 + t) * xs + gq * dim + h * D + c);
      dst = St::X + ((t * 2 + gq) * 128 + c) * 2;
    } else if (i < St::NW_ + St::NWV + St::NDT + St::NX + St::NXV) {
      const int j = i - (St::NW_ + St::NWV + St::NDT + St::NX), t = j / (VB / 8), c = (j % (VB / 8)) * 8;
      if constexpr (PR_V16M32) src = reinterpret_cast<const char*>(k_x + tok0 * xs + (int64_t)(t * (int)xs + 2 * dim + h * D + vr0 + c));
      else src = reinterpret_cast<const char*>(k_x + (tok0 + t) * xs + 2 * dim + h * D + vr0 + c);
      dst = St::XV + (t * VB + c) * 2;
    } else {
      const int j = i - (St::NW_ + St::NWV + St::NDT + St::NX + St::NXV), t = j >> 4, c = (j & 15) * 8;
      if constexpr (PR_V16M32) src = reinterpret_cast<const char*>(p.gate + (tok0 * H + h) * D + (int64_t)(t * dim + c));
      else src = reinterpret_cast<const char*>(p.gate + ((tok0 + t) * H + h) * D + c);
      dst = St::G + (t * 128 + c) * 2;
    }
    if constexpr (PR_V16NT >= 2) ca[k] = __builtin_nontemporal_load(reinterpret_cast<const u4*>(src));
    else ca[k] = *reinterpret_cast<const u4*>(src);
    cad[k] = dst;
  }
#endif
  const float alog = p.A_log[h];
#if CG_FH
  // beta of token min(lane, S - 1): uniform descriptor (p.beta + tok0 * H + h), 32-bit lane offset
  const float beta_raw = b2f(__builtin_bit_cast(bf16, __builtin_amdgcn_raw_buffer_load_b16(
      __builtin_amdgcn_make_buffer_rsrc(const_cast<bf16*>(pbeta + tok0 * H + h), 0, 0x7fffffff, 0x00020000),
      (uint32_t)(lane < S ? lane : S - 1) * (uint32_t)(H * 2), 0, 0)));
#else
  const float beta_raw = b2f(p.beta[(tok0 + (lane < S ? lane : S - 1)) * H + h]);
#endif
  float og_r[OGN][2], nw_r[2];
  if (!POLL) {
  nw_r[0] = p.nw[lane];
  nw_r[1] = p.nw[lane + 64];
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    const int t = (wave + NW * i) < S ? wave + NW * i : S - 1;
    const bf16* ogp = p.og + (tok0 + t) * dim + h * D;
    og_r[i][0] = b2f(ogp[lane]);
    og_r[i][1] = b2f(ogp[lane + 64]);
  }
  }

  // ---- 2. slot chain, state rows, conv history chunks
#if CG_SL
  if (!sok) {
#else
  int a, rd, cs;
  int wsl[S];
  if (!slots<S, SPEC>(k_idx, k_acc, n, a, rd, cs, wsl)) {
#endif
    // fake uses pin the first-hop chunk loads above this branch (LLVM otherwise sinks them below the slot wait)
    if constexpr (!(AB_NOPIN & 2)) {
#pragma unroll
    for (int k = 0; k < RA; ++k) asm volatile("" :: "v"(ca[k]));
    asm volatile("" :: "v"(alog));
    }
    if constexpr (PR_V16BETA) asm volatile("" :: "v"(beta_raw));
    if (POLL) {
      __syncthreads();
      if constexpr (!NBF) __syncthreads();
    }
    for (int i = tid; i < S * VB; i += NT) {
      const int t = i / VB, v = vr0 + i % VB;
      p.out[(tok0 + t) * dim + h * D + v] = __float2bfloat16(0.0f);
    }
    asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
    return;
  }
  if (wave == 0) STAMP(1);
  const int start = a - 1;
  bf16* crow = k_conv + SLOTMUL(cs, k_css);
  const int64_t ctap = 3 * (int64_t)dim;
  constexpr int RB = (St::NB + NT - 1) / NT;
  constexpr bool LDSH = PR_V16LDS && RB == 1 && (LATE ? (RPT >= 2 ? RPT / 2 : 0) : RPT) == 0;  // P62: history via LDS-DMA
  u4 cb[RB];
  int cbd[RB];
  if constexpr (LDSH) {
    // conv-history chunks straight into the staging area (dst = HQ + 16 i, contiguous in i): 16 B LDS-DMA, M0 = wave base
    if (wave * 64 < St::NB) {
      const int i = tid;
      const bf16* src;
      if (i < 96) {
        const int tap = i >> 5, gq = (i >> 4) & 1, c = (i & 15) * 8;
        src = crow + (start + tap) * ctap + gq * dim + h * D + c;
      } else {
        const int j = i - 96, tap = j / (VB / 8), c = (j % (VB / 8)) * 8;
        src = crow + (start + tap) * ctap + 2 * dim + h * D + vr0 + c;
      }
      if (i < St::NB)
        __builtin_amdgcn_global_load_lds((void*)src, (__attribute__((address_space(3))) void*)(stg + St::HQ + wave * 64 * 16), 16, 0, 0);
    }
  }
#if CG_FH
  {
    // one descriptor for the block's history window; the per-lane offsets were computed before the slot wait, so the
    // loads issue right after the (scalar) slot arithmetic
    const auto hrs = __builtin_amdgcn_make_buffer_rsrc(crow + start * ctap + h * D, 0, 0x7fffffff, 0x00020000);
#pragma unroll
    for (int k = 0; k < RB; ++k) {
      cb[k] = u4{0u, 0u, 0u, 0u};
      cbd[k] = hdst[k];
      if (wave * 64 + k * NT >= St::NB) continue;  // wave-uniform: no real chunk in this wave
      cb[k] = __builtin_amdgcn_raw_buffer_load_b128(hrs, hoff[k], 0, 0);
    }
  }
#else
#pragma unroll
  for (int k = 0; k < (LDSH ? 0 : RB); ++k) {
    cb[k] = u4{0u, 0u, 0u, 0u};
    cbd[k] = 0;
    if (wave * 64 + k * NT >= St::NB) continue;  // wave-uniform: no real chunk in this wave
    int i = tid + k * NT;
    i = i < St::NB ? i : St::NB - 1;
    const bf16* src;
    int dst;
    if (i < 96) {
      const int tap = i >> 5, gq = (i >> 4) & 1, c = (i & 15) * 8;
      if constexpr (PR_V16H32) src = crow + start * ctap + (int64_t)(tap * 3 * dim + gq * dim + h * D + c);
      else src = crow + (start + tap) * ctap + gq * dim + h * D + c;
      dst = St::HQ + ((tap * 2 + gq) * 128 + c) * 2;
    } else {
      const int j = i - 96, tap = j / (VB / 8), c = (j % (VB / 8)) * 8;
      if constexpr (PR_V16H32) src = crow + start * ctap + (int64_t)(tap * 3 * dim + 2 * dim + h * D + vr0 + c);
      else src = crow + (start + tap) * ctap + 2 * dim + h * D + vr0 + c;
      dst = St::HV + (tap * VB + c) * 2;
    }
    if constexpr (PR_V16HDUP) {
      if (wave * 64 + k * NT < St::NB) cb[k] = *reinterpret_cast<const u4*>(src);
    } else if constexpr (PR_V16NT >= 1) cb[k] = __builtin_nontemporal_load(reinterpret_cast<const u4*>(src));
    else cb[k] = *reinterpret_cast<const u4*>(src);
    cbd[k] = dst;
  }
#endif
  // state rows after the history chunks (in-order vmcnt; see kda_main)
  // LATE: state rows are loaded after the staging barrier, so the history chunks do not queue behind B*H x 64 KB
  // of state reads at the DRAM; the state fetch then overlaps the prologue math.
  const int rlane = tid & 15, rgroup = (tid & (TT - 1)) >> 4;  // waves >= TT/64 load duplicate rows (no branch)
  const float* sp = k_state + SLOTMUL(rd, k_sss) + (int64_t)h * D * D + (vr0 + rgroup) * D + 4 * rlane;
  f4 hr[RPT][2];
  // HALF (RPT >= 2): the first half of each thread's rows is loaded early (with the slot chain), the rest after the
  // staging barrier: part of the store stream can start right after the prologue.
  constexpr int R0 = (CG_RE && TT < NT) ? RPT : ((LATE && RPT >= 2 && !AB_NOHALF) ? RPT / 2 : (LATE ? 0 : RPT));
  auto load_rows = [&](int i0, int i1) {
#pragma unroll
    for (int i = 0; i < RPT; ++i)
      if (i >= i0 && i < i1)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#if BLD
          hr[i][j] = __builtin_bit_cast(f4, __builtin_amdgcn_raw_buffer_load_b128(
              __builtin_amdgcn_make_buffer_rsrc(k_state + SLOTMUL(rd, k_sss) + (int64_t)h * D * D + vr0 * D, 0,
                                                (PR_V16DUP && TT < NT && wave >= TT / 64) ? 0 : VB * D * 4, 0x00020000),
              ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, BLDAUX));
#else
          hr[i][j] = __builtin_nontemporal_load(reinterpret_cast<const f4*>(sp + i * RG * D + 64 * j));
#endif
  };
  if constexpr (CG_RE == 2 && TT < NT) { if (wave < TT / 64) load_rows(0, R0); }
  else load_rows(0, R0);

#pragma unroll
  for (int k = 0; k < RA; ++k)
    if (wave * 64 + k * NT < St::NA) *reinterpret_cast<u4*>(stg + cad[k]) = ca[k];
#pragma unroll
  for (int k = 0; k < (LDSH ? 0 : RB); ++k)
    if (wave * 64 + k * NT < St::NB) *reinterpret_cast<u4*>(stg + cbd[k]) = cb[k];
  if constexpr (LDSH) {
    if (wave * 64 < St::NB) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  if constexpr (NBF) {
    if (wave == 0) s_rdy[lane & 7] = 0;
  }
  __syncthreads();  // staging complete; every conv-history read of this block is done
  if (wave == 0) STAMP(4);
  if (LATE) load_rows(R0, RPT);


  const float ea = __expf(alog);
  // ---- 3. prologue math from LDS: one wave per token, 2 q/k/decay channels and VPL v rows per lane
  const float* Lw = reinterpret_cast<const float*>(stg + St::W);
  const float* Lwv = reinterpret_cast<const float*>(stg + St::WV);
  const float* Ldt = reinterpret_cast<const float*>(stg + St::DT);
  const uint32_t* Lx = reinterpret_cast<const uint32_t*>(stg + St::X);   // bf16 pairs [S][2][64]
  const bf16* Lxv = reinterpret_cast<const bf16*>(stg + St::XV);
  const uint32_t* Lg = reinterpret_cast<const uint32_t*>(stg + St::G);   // [S][64]
  const uint32_t* Lh = reinterpret_cast<const uint32_t*>(stg + St::HQ);  // [3][2][64]
  const bf16* Lhv = reinterpret_cast<const bf16*>(stg + St::HV);
  const float scale = 0.08838834764831845f;
  const int c0 = 2 * lane;
#if CG_TP
  constexpr float L2E = 1.4426950408889634f;
  auto tokpro = [&](auto tc) {
    constexpr int t = decltype(tc)::value;
    {
      f2 q = {0.f, 0.f}, k = {0.f, 0.f};
      float v[VPL];
#pragma unroll
      for (int r = 0; r < VPL; ++r) v[r] = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int m = t + j;  // window position; < 3 -> history tap start + m, else x[m - 3]
        const uint32_t uq = m < 3 ? Lh[(m * 2) * 64 + lane] : Lx[((m - 3) * 2) * 64 + lane];
        const uint32_t uk = m < 3 ? Lh[(m * 2 + 1) * 64 + lane] : Lx[((m - 3) * 2 + 1) * 64 + lane];
        const f2 wq = *reinterpret_cast<const f2*>(Lw + j * 128 + c0);
        const f2 wk = *reinterpret_cast<const f2*>(Lw + (4 + j) * 128 + c0);
        q += f2{lo2f(uq), hi2f(uq)} * wq;
        k += f2{lo2f(uk), hi2f(uk)} * wk;
#pragma unroll
        for (int r = 0; r < VPL; ++r) {
          const int vr = (lane + 64 * r) & (VB - 1);
          v[r] += b2f(m < 3 ? Lhv[m * VB + vr] : Lxv[(m - 3) * VB + vr]) * Lwv[j * VB + vr];
        }
      }
      // x * sigm(x), sigm(x) = rcp(1 + exp2(-log2(e) x)): the multiplies and adds packed, exp2 / rcp per element
      auto sig2 = [&](f2 x, float c) {
        f2 e = x * c;
        e = f2{__builtin_amdgcn_exp2f(e.x), __builtin_amdgcn_exp2f(e.y)} + 1.0f;
        return f2{__builtin_amdgcn_rcpf(e.x), __builtin_amdgcn_rcpf(e.y)};
      };
      q *= sig2(q, -L2E);
      k *= sig2(k, -L2E);
      const uint32_t ug = Lg[t * 64 + lane];
      const f2 dt2 = *reinterpret_cast<const f2*>(Ldt + c0);
      // decay exp(lb * sigm(ea * (g + dt))) with the scalars folded: -log2(e) * ea and lb * log2(e)
      f2 d = sig2(f2{lo2f(ug), hi2f(ug)} + dt2, -L2E * ea) * (p.lb * L2E);
      d = f2{__builtin_amdgcn_exp2f(d.x), __builtin_amdgcn_exp2f(d.y)};
      *reinterpret_cast<f2*>(&s_d[t][c0]) = d;
      // sigm(v row 0) and sigm(beta) as one pair
      const f2 svb = sig2(f2{v[0], rl(beta_raw, t)}, -L2E);
      const float bt = svb.y;
      if constexpr (VB <= 64) s_v[t][lane & (VB - 1)] = v[0] * svb.x * bt;  // duplicate lanes write the same value
      else {
        if (lane < VB) s_v[t][lane] = v[0] * svb.x * bt;
#pragma unroll
        for (int r = 1; r < VPL; ++r)
          if (lane + 64 * r < VB) s_v[t][lane + 64 * r] = v[r] * sigm(v[r]) * bt;
      }
      const f2 qq = q * q, kk = k * k, qk = q * k;
      float sq = qq.x + qq.y, sk = kk.x + kk.y, sqk = qk.x + qk.y;
      wave_sum3c(sq, sk, sqk);
      sq = rl(sq, 63); sk = rl(sk, 63); sqk = rl(sqk, 63);
      const float rq = __builtin_amdgcn_rsqf(sq + 1e-6f) * scale;
      const float rk = __builtin_amdgcn_rsqf(sk + 1e-6f);
      *reinterpret_cast<f2*>(&s_q[t][c0]) = q * rq;
      *reinterpret_cast<f2*>(&s_k[t][c0]) = k * rk;
      *reinterpret_cast<float2*>(&s_nrm[t][0]) = make_float2(bt, sqk * rq * rk);  // every lane writes the same pair (no exec branch)
    }
  };
#else
  auto tokpro = [&](auto tc) {
    constexpr int t = decltype(tc)::value;
    {
      float q[2] = {0.f, 0.f}, k[2] = {0.f, 0.f}, v[VPL];
#pragma unroll
      for (int r = 0; r < VPL; ++r) v[r] = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int m = t + j;  // window position; < 3 -> history tap start + m, else x[m - 3]
        const uint32_t uq = m < 3 ? Lh[(m * 2) * 64 + lane] : Lx[((m - 3) * 2) * 64 + lane];
        const uint32_t uk = m < 3 ? Lh[(m * 2 + 1) * 64 + lane] : Lx[((m - 3) * 2 + 1) * 64 + lane];
        const float2 wq = *reinterpret_cast<const float2*>(Lw + j * 128 + c0);
        const float2 wk = *reinterpret_cast<const float2*>(Lw + (4 + j) * 128 + c0);
        q[0] += lo2f(uq) * wq.x;
        q[1] += hi2f(uq) * wq.y;
        k[0] += lo2f(uk) * wk.x;
        k[1] += hi2f(uk) * wk.y;
#pragma unroll
        for (int r = 0; r < VPL; ++r) {
          const int vr = (lane + 64 * r) & (VB - 1);
          v[r] += b2f(m < 3 ? Lhv[m * VB + vr] : Lxv[(m - 3) * VB + vr]) * Lwv[j * VB + vr];
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        q[e] *= sigm(q[e]);
        k[e] *= sigm(k[e]);
      }
      const uint32_t ug = Lg[t * 64 + lane];
      const float2 dt2 = *reinterpret_cast<const float2*>(Ldt + c0);
      const float d0 = __expf(p.lb * sigm(ea * (lo2f(ug) + dt2.x)));
      const float d1 = __expf(p.lb * sigm(ea * (hi2f(ug) + dt2.y)));
      *reinterpret_cast<float2*>(&s_d[t][c0]) = make_float2(d0, d1);
      // token loop works on normalized q^ (incl. scale), k^ and beta * v; s_nrm[t] = {beta, q^.k^}
      const float bt = sigm(rl(beta_raw, t));
#pragma unroll
      for (int r = 0; r < VPL; ++r)
        if constexpr (PR_V16VALL && VB <= 64) s_v[t][lane & (VB - 1)] = v[r] * sigm(v[r]) * bt;  // duplicate lanes write the same value
        else if (lane + 64 * r < VB) s_v[t][lane + 64 * r] = v[r] * sigm(v[r]) * bt;
      float sq = q[0] * q[0] + q[1] * q[1], sk = k[0] * k[0] + k[1] * k[1], sqk = q[0] * k[0] + q[1] * k[1];
      wave_sum3b(sq, sk, sqk);
      const float rq = __builtin_amdgcn_rsqf(sq + 1e-6f) * scale;
      const float rk = __builtin_amdgcn_rsqf(sk + 1e-6f);
      *reinterpret_cast<float2*>(&s_q[t][c0]) = make_float2(q[0] * rq, q[1] * rq);
      *reinterpret_cast<float2*>(&s_k[t][c0]) = make_float2(k[0] * rk, k[1] * rk);
      if (lane == 0) *reinterpret_cast<float2*>(&s_nrm[t][0]) = make_float2(bt, sqk * rq * rk);
    }
  };
#endif
  // token t -> wave t % NW; each wave runs its own compile-time instantiation (wave-uniform branches)
  sfor<0, S>([&](auto tc) {
    if (wave == (decltype(tc)::value + PR_V16TOK) % NW) {
      tokpro(tc);
      if constexpr (NBF) {
        asm volatile("" ::: "memory");
        s_rdy[decltype(tc)::value] = 1;  // after the token's operand writes (LDS writes of a wave execute in order)
        asm volatile("" ::: "memory");
      }
    }
  });
  if (!POLL) {
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    og_r[i][0] = sigm(og_r[i][0]);
    og_r[i][1] = sigm(og_r[i][1]);
  }
  }

  // conv commit from the staged raw inputs, 16 B per lane: taps [0, S + 2) = [old start+1, old start+2,
  // x_0 .. x_{S-1}]. v rows are owned by this block; q/k channels are shared by all splits (last arriver).
  auto commit = [&](bool do_qk, int i0, int step) {
    constexpr int NQ = (S + 2) * 32, NV = (S + 2) * (VB / 8);
    const int nch = do_qk ? NQ : NV;
    for (int i = i0; i < nch; i += step) {
      int T, c, gq;
      const u4* srcv;
      bf16* dstp;
      if (do_qk) {
        T = i >> 5; gq = (i >> 4) & 1; c = (i & 15) * 8;
        srcv = reinterpret_cast<const u4*>(T < 2 ? stg + St::HQ + (((T + 1) * 2 + gq) * 128 + c) * 2
                                                 : stg + St::X + (((T - 2) * 2 + gq) * 128 + c) * 2);
        dstp = crow + T * ctap + gq * dim + h * D + c;
      } else {
        T = i / (VB / 8); c = (i % (VB / 8)) * 8;
        srcv = reinterpret_cast<const u4*>(T < 2 ? stg + St::HV + ((T + 1) * VB + c) * 2
                                                 : stg + St::XV + ((T - 2) * VB + c) * 2);
        dstp = crow + T * ctap + 2 * dim + h * D + vr0 + c;
      }
      if constexpr (PR_V16OUTNT && POLL) __builtin_nontemporal_store(*srcv, reinterpret_cast<u4*>(dstp));
      else *reinterpret_cast<u4*>(dstp) = *srcv;
    }
  };
  // split: the v-tap commit is issued after the token loop by a wave that is idle during the o exchange (the history
  // reads all completed at the staging barrier), off the prologue-math critical path. No split: here (a commit at
  // the end would extend the kernel tail).
  if (NSPLIT == 1) {
    commit(false, tid, NT);
    commit(true, tid, NT);
  }
  if constexpr (!NBF) __syncthreads();
  if (wave == 0) STAMP(5);
  if constexpr (PR_V16VCE && POLL && TT < NT) {
    if (wave == NW - 1) commit(false, lane, 64);  // v taps of this block's rows, by a wave without token-loop work
  }
  auto wait_rdy = [&](int lo, int hi) {  // spin until tokens [lo, hi) are announced (wave-uniform LDS polls)
    for (;;) {
      const u4 f = *reinterpret_cast<const u4*>(&s_rdy[lo & ~3]);
      asm volatile("" ::: "memory");  // re-read every iteration
      int c = 0;
#pragma unroll
      for (int t = lo; t < hi; ++t) c += (int)f[t & 3];
      if (__builtin_amdgcn_readfirstlane(c) >= hi - lo) break;
    }
  };

  // ---- 4. recurrence
  // token t+1's LDS operands are read before token t's state stores (a store stalled at issue would otherwise
  // hold back the next token's reads and expose their latency)
  struct TokOps { f4 q[2], k[2], d[2]; float2 nb; float v[RPT]; };
  auto ld = [&](int t, TokOps& o) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int kb = 4 * rlane + 64 * j;
      o.q[j] = *reinterpret_cast<const f4*>(&s_q[t][kb]);
      o.k[j] = *reinterpret_cast<const f4*>(&s_k[t][kb]);
      o.d[j] = *reinterpret_cast<const f4*>(&s_d[t][kb]);
    }
    o.nb = *reinterpret_cast<const float2*>(&s_nrm[t][0]);
#pragma unroll
    for (int i = 0; i < RPT; ++i) o.v[i] = s_v[t][rgroup + RG * i];
  };
  // pipelined only with >= 2 rows per thread: with one row the token's math is too short to cover the reads, and
  // reading token 1's operands ahead of token 0 delays token 0 (measured: B=1 loop -0.09 us but token 0 +0.08 us)
  constexpr bool PIPE = (RPT >= 2 && !(AB_NOPIPE & 2)) || (TT < NT && !(AB_NOPIPE & 1));  // one token wave per SIMD: nothing else hides the LDS round trip
  TokOps cur;
  if (TT == NT || wave < TT / 64) {
  if constexpr (NBF) wait_rdy(0, S < NBW ? S : NBW);
  if (PIPE) ld(0, cur);
#pragma unroll
  for (int t = 0; t < S; ++t) {
    TokOps nxt;
    if constexpr (NBF) {
      if (t == NBW - 1 && S > NBW) wait_rdy(NBW, S);  // the younger waves' tokens, before the lookahead reads token NBW
    }
    if (!PIPE) ld(t, cur);
    else if (t + 1 < S) ld(t + 1, nxt);
    const float beta = cur.nb.x, kq = cur.nb.y;
    const f4* q4 = cur.q;
    const f4* k4 = cur.k;
    const f4* d4 = cur.d;
    const bool do_store = SPEC ? (wsl[t] > 0) : (t == S - 1);
    float* ubase = k_state + SLOTMUL(wsl[t], k_sss) + (int64_t)h * D * D + split * VB * D;  // uniform
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      f4 hd[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) hd[j] = hr[i][j] * d4[j];
      // packed (2-wide) dot partials
      const f2 ak = hd[0].lo * k4[0].lo + hd[0].hi * k4[0].hi + hd[1].lo * k4[1].lo + hd[1].hi * k4[1].hi;
      const f2 aq = hd[0].lo * q4[0].lo + hd[0].hi * q4[0].hi + hd[1].lo * q4[1].lo + hd[1].hi * q4[1].hi;
      float dk = ak.x + ak.y, dqi = aq.x + aq.y;
      if constexpr (TT < NT && !AB_NORR2V) row_reduce2(dk, dqi);  // latency-bound: both chains together
      else dk = row_reduce(dk);
      const int r = rgroup + RG * i;
      const float vn = __builtin_fmaf(-beta, dk, cur.v[i]);  // beta * (v - k^.h)
#pragma unroll
      for (int j = 0; j < 2; ++j) hr[i][j] = hd[j] + k4[j] * vn;
      // split: wave 0 defers its last-token stores until after the arrival atomic (it waits on its o stores)
      // branchless: a store that must not happen goes to a zero-size buffer (dropped by the bounds check), so the
      // unrolled token loop is one basic block and the scheduler can overlap consecutive tokens
      {
        const bool st = do_store && !(NSPLIT > 1 && !POLL && t == S - 1 && wave == 0);
        const auto rs = __builtin_amdgcn_make_buffer_rsrc(ubase, 0, st ? 0x7fffffff : 0, 0x00020000);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u4, hr[i][j]), rs,
                                                 ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0,
                                                 SPEC ? SPOL_SPEC : SPOL_PLAIN);
      }
      // the o reduction after the store block: the stores issue right after the update, dq runs while they drain
      const float dq = (TT < NT && !AB_NORR2V) ? dqi : row_reduce(dqi);
      const float o = __builtin_fmaf(vn, kq, dq);
      s_o[t][r] = o;  // all 16 lanes of the row write the same value (no exec branch)
    }
    if (PIPE && t + 1 < S) cur = nxt;
    if (t == 0 && wave == 0) STAMP(6);
    if constexpr (!(AB_NOSB & 1)) __builtin_amdgcn_sched_barrier(0);  // no cross-token hoisting (max-ILP build)
  }
  }  // token waves
  if (wave == 0) STAMP(7);

  // ---- 5. epilogue
  if constexpr (POLL) {
    __syncthreads();  // s_o complete; the exchange wave takes over
    if (!(PR_V16VCE && TT < NT) && wave == NW - 1) commit(false, lane, 64);  // v taps of this block's rows
    TL(19);
    TLR(21);
    TLFLUSH();
    asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
    return;
  }
  float ov[OGN][2];
  if (NSPLIT > 1) {
    // o of all tokens goes out as full 16 B system-scope (sc0 sc1) stores from wave 0, which alone waits for them
    // (and for its own earlier state stores) before the arrival atomic; the other waves never wait on stores.
    __syncthreads();
    if (wave == NW - 1) commit(false, lane, 64);  // wave 0 is busy with the o exchange
    if (wave == 0) {
      const auto orsrc = __builtin_amdgcn_make_buffer_rsrc(p.ow + (int64_t)g * S * D, 0, 0x7fffffff, 0x00020000);
      for (int e = lane * 4; e < S * VB; e += 256) {
        const int t = e / VB, v = e % VB;
        __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u4, *reinterpret_cast<const f4*>(&s_o[t][v])),
                                               orsrc, (t * D + split * VB + v) * 4, 0, 17);
      }
      __builtin_amdgcn_s_waitcnt(0);
      STAMP(8);
      if (lane == 0) {
        const int old = __hip_atomic_fetch_add(p.cnt + g, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old == NSPLIT - 1;
      }
      constexpr int t = S - 1;
      if (SPEC ? (wsl[t] > 0) : true) {
        float* ubase = k_state + SLOTMUL(wsl[t], k_sss) + (int64_t)h * D * D + split * VB * D;
#pragma unroll
        for (int i = 0; i < RPT; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            __builtin_amdgcn_raw_buffer_store_b128(
                __builtin_bit_cast(u4, hr[i][j]), __builtin_amdgcn_make_buffer_rsrc(ubase, 0, 0x7fffffff, 0x00020000),
                ((rgroup + i * RG) * D + 4 * rlane + 64 * j) * 4, 0, SPEC ? SPOL_SPEC : SPOL_PLAIN);
      }
    }
    __syncthreads();
    if (wave == 0) STAMP(9);
    if (!s_last) {
      asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
      return;
    }
    // last arriver: o loads first, then the counter reset and the q/k commit as straight-line buffer stores (fixed
    // count per wave, lanes without work dropped by the bounds check). vmcnt is in order: with the stores issued
    // before the loads (or in exec-masked branches) the norm's wait for the loads also waited for the stores.
#pragma unroll
    for (int i = 0; i < OGN; ++i) {
      const int t = wave + NW * i;
      if (t < S) {
        const float* wp = p.ow + ((int64_t)g * S + t) * D;
        ov[i][0] = __hip_atomic_load(wp + lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        ov[i][1] = __hip_atomic_load(wp + lane + 64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    asm volatile("" ::: "memory");
    constexpr uint32_t OOB = 0x7ffffff0u;  // offset + size > num_records: store dropped
    __builtin_amdgcn_raw_buffer_store_b32(0, __builtin_amdgcn_make_buffer_rsrc(p.cnt + g, 0, 4, 0x00020000),
                                          tid == 0 ? 0u : OOB, 0, 16);  // relaxed agent-scope reset (sc1)
    {
      constexpr int NQ = (S + 2) * 32;
      const auto crs = __builtin_amdgcn_make_buffer_rsrc(crow, 0, (int)((S + 2) * ctap * 2), 0x00020000);
#pragma unroll
      for (int it = 0; it < (NQ + NT - 1) / NT; ++it) {
        const int i0 = tid + it * NT;
        const int i = i0 < NQ ? i0 : 0;
        const int T = i >> 5, gq = (i >> 4) & 1, c = (i & 15) * 8;
        const u4 v = *reinterpret_cast<const u4*>(T < 2 ? stg + St::HQ + (((T + 1) * 2 + gq) * 128 + c) * 2
                                                        : stg + St::X + (((T - 2) * 2 + gq) * 128 + c) * 2);
        const uint32_t off = i0 < NQ ? (uint32_t)((T * ctap + gq * dim + h * D + c) * 2) : OOB;
        __builtin_amdgcn_raw_buffer_store_b128(v, crs, off, 0, 0);
      }
    }
  } else {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < OGN; ++i) {
      const int t = wave + NW * i;
      if (t < S) {
        ov[i][0] = s_o[t][lane];
        ov[i][1] = s_o[t][lane + 64];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < OGN; ++i) {
    const int t = wave + NW * i;
    if (t < S) {
      const float o0 = bfround(ov[i][0]), o1 = bfround(ov[i][1]);
      const float ss = EPWSUM(o0 * o0 + o1 * o1);
      const float rstd = RSQRT(ss * (1.0f / D) + p.eps);
      bf16* op = p.out + (tok0 + t) * dim + h * D;
      op[lane] = __float2bfloat16(o0 * rstd * nw_r[0] * og_r[i][0]);
      op[lane + 64] = __float2bfloat16(o1 * rstd * nw_r[1] * og_r[i][1]);
    }
  }
  if (wave == 0) STAMP(10);
  asm volatile("s_waitcnt vmcnt(0) ; keep %0 %1" :: "v"(pf0), "v"(pf1));
}

// One word per block: the physical XCC the block ran on (host-side check of the residue -> XCC map behind X1).
__global__ __launch_bounds__(1024) void kda_xcc_probe(int* out) {
  if (threadIdx.x != 0) return;
  int xcc_;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_));
  out[blockIdx.y * gridDim.x + blockIdx.x] = xcc_ & 15;
}

int g_query = 0, g_regs = 0;

#define KDA_GO(K, ns, nt) hipLaunchKernelGGL(K, GRIDD(B, p.H, (ns)), dim3(nt), 0, st, p.idx, p.acc, p.state, p.conv, p.x, p.H, \
                                            (int)p.state_slot_stride, (int)p.conv_slot_stride, p.nslots, p)
// The kernels the cost model (kernel_*.py) can ask for, and nothing else:
//   V-split (VB = 32 / 16 rows per block, spec, S >= 2): the staged kernel kda_vec<VB, S, spec, 512, LATE> + its exchange wave
//   whole-state plain: kda_main 256 threads (state-first / prologue-first hold kernel), 1024 threads prologue-first (+ next-round prefetch NX)
//   whole-state spec:  kda_main 1024 threads prologue-first (+ NX | SB state rows behind the block barrier | W1 wave offset)
// Anything else is "unsupported configuration" (launch() returns false), never a silent no-launch.
template <int VB, int S, bool SPEC, int NT>
bool launch(const Params& p, int B, hipStream_t st) {
  if constexpr (VB < 128) {
    if constexpr (SPEC && S >= 2 && NT == 512) {
      if (g_query || !p.vec) return false;
      KDA_GO((kda_vec<VB, S, SPEC, NT, true>), 128 / VB, NT + 64);
      return true;
    } else {
      return false;
    }
  } else {
    if (p.vec) return false;
    const void* fn = nullptr;  // 0 PF, 1 PF + NX, 2 PF + W1, 3 PF + SB, 4 state-first (plain 256 threads), 5 PF + NX + two-copy TN (short window)
    int k = -1;
    if constexpr (NT == 1024) {
      if (p.pf != 1) return false;
      if (p.nxt > 0) { k = 1; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true, true>); }
      else { k = 0; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true>); }
#if ST_TN
      if constexpr (SPEC && S < ST_TNS) {  // tail_nt() selected a nontemporal tail (p.hold = its first block) in a short window
        if (k == 1 && p.hold > 0) {
          if constexpr (S >= ST_TNV) { k = 5; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true, true, false, true>); }
          else return false;  // the cost model asks for a kernel this build does not carry: fail loudly
        }
      }
#endif
#if TP_WS && TP_WS1
      if constexpr (SPEC && S >= 2) {
        if (k == 0 && p.hold < 0) { k = 2; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true, false, false, true>); }  // wave_offset()
      }
#endif
      if constexpr (SPEC) {
        if (k == 0 && p.hold > 0) { k = 3; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true, false, true>); }  // state_after_prologue()
      }
    } else if constexpr (NT == 256 && !SPEC) {
      if (p.pf) { k = 0; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, true>); }
      else { k = 4; fn = reinterpret_cast<const void*>(&kda_main<VB, S, SPEC, NT, false>); }
    } else {
      return false;
    }
    if constexpr (NT == 1024 && !SPEC && S == 1) {
      if (k == 0 && p.hold == -2) {  // xcc_first() in kernel_*.py
        if (g_query) return false;
        Params q = p;
        q.hold = 0;
        hipLaunchKernelGGL((kda_main<VB, S, SPEC, NT, true, false, false, false, true>), dim3(8 * q.H, B), dim3(NT), 0, st, q.idx, q.acc, q.state,
                           q.conv, q.x, q.H, (int)q.state_slot_stride, (int)q.conv_slot_stride, q.nslots, q);
        return true;
      }
    }
    if (g_query) {  // cost-model query: report the compiled register count instead of launching
      hipFuncAttributes fa;
      g_regs = hipFuncGetAttributes(&fa, fn) == hipSuccess ? fa.numRegs : -1;
      return true;
    }
    if constexpr (NT == 1024) {
#if ST_TN
      if constexpr (SPEC && S >= ST_TNV && S < ST_TNS) {
        if (k == 5) { KDA_GO((kda_main<VB, S, SPEC, NT, true, true, false, true>), 1, NT); return true; }
      }
#endif
      if (k == 1) { KDA_GO((kda_main<VB, S, SPEC, NT, true, true>), 1, NT); return true; }
#if TP_WS && TP_WS1
      if constexpr (SPEC && S >= 2) {
        if (k == 2) { KDA_GO((kda_main<VB, S, SPEC, NT, true, false, false, true>), 1, NT); return true; }
      }
#endif
      if constexpr (SPEC) {
        if (k == 3) { KDA_GO((kda_main<VB, S, SPEC, NT, true, false, true>), 1, NT); return true; }
      }
      KDA_GO((kda_main<VB, S, SPEC, NT, true>), 1, NT);
      return true;
    } else if constexpr (NT == 256 && !SPEC) {
      if (k == 4) KDA_GO((kda_main<VB, S, SPEC, NT, false>), 1, NT);
      else KDA_GO((kda_main<VB, S, SPEC, NT, true>), 1, NT);
      return true;
    }
    return false;
  }
}
#undef KDA_GO

// (nsplit, threads) pairs: VB = 128 / nsplit rows per block, at least one row per 16-lane group.
template <int S, bool SPEC>
bool launch_s(const Params& p, int B, int nsplit, int nth, hipStream_t st) {
#define L(VB, NT) \
  if (nsplit == 128 / VB && nth == NT) return launch<VB, S, SPEC, NT>(p, B, st);
  L(128, 256) L(128, 1024) L(32, 512) L(16, 512)
#undef L
  return false;
}

bool dispatch(const Params& p, int B, int S, bool spec, int nsplit, int nth, hipStream_t st) {
#ifdef KDA_SPEC_ONLY
  // second build of this file (max-ILP machine scheduling, see kernel_v4_c.py): spec kernels only
  if (!spec) return false;
  switch (S) {
    case 1: return launch_s<1, true>(p, B, nsplit, nth, st);
    case 2: return launch_s<2, true>(p, B, nsplit, nth, st);
    case 3: return launch_s<3, true>(p, B, nsplit, nth, st);
    case 4: return launch_s<4, true>(p, B, nsplit, nth, st);
    case 5: return launch_s<5, true>(p, B, nsplit, nth, st);
    case 6: return launch_s<6, true>(p, B, nsplit, nth, st);
    case 7: return launch_s<7, true>(p, B, nsplit, nth, st);
    case 8: return launch_s<8, true>(p, B, nsplit, nth, st);
  }
  return false;
#else
  // default build: the plain kernels (the spec kernels live in the max-ILP build only)
  return !spec && S == 1 && launch_s<1, false>(p, B, nsplit, nth, st);
#endif
}

#ifdef KDA_DEVICE_ONLY
#ifdef KDA_SPEC_ONLY
template bool launch_s<8, true>(const Params&, int, int, int, hipStream_t);
#endif
#if defined(KDA_DEV_ALL) && defined(KDA_SPEC_ONLY)
template bool launch_s<1, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<2, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<3, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<4, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<5, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<6, true>(const Params&, int, int, int, hipStream_t);
template bool launch_s<7, true>(const Params&, int, int, int, hipStream_t);
#endif
#ifndef KDA_SPEC_ONLY
template bool launch_s<1, false>(const Params&, int, int, int, hipStream_t);
#endif
#endif
}  // namespace

#ifndef KDA_DEVICE_ONLY
void run(torch::Tensor x, torch::Tensor conv, torch::Tensor cw, torch::Tensor gate, torch::Tensor beta,
         torch::Tensor og, torch::Tensor A_log, torch::Tensor dt, torch::Tensor nw, torch::Tensor state,
         torch::Tensor idx, std::optional<torch::Tensor> acc, torch::Tensor out, torch::Tensor mb,
         double eps, double lb, int64_t nsplit, int64_t nth, int64_t vec, int64_t pf, int64_t hold, int64_t nxt) {
  Params p;
  const int H = A_log.numel();
  const int B = idx.size(0);
  const int T = x.size(0);
  p.x = reinterpret_cast<const bf16*>(x.data_ptr());
  p.conv = reinterpret_cast<bf16*>(conv.data_ptr());
  p.cw = cw.data_ptr<float>();
  p.gate = reinterpret_cast<const bf16*>(gate.data_ptr());
  p.beta = reinterpret_cast<const bf16*>(beta.data_ptr());
  p.og = reinterpret_cast<const bf16*>(og.data_ptr());
  p.A_log = A_log.data_ptr<float>();
  p.dt = dt.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.state = state.data_ptr<float>();
  p.idx = idx.data_ptr<int>();
  p.acc = acc.has_value() ? acc->data_ptr<int>() : nullptr;
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.cnt = nullptr;  // arrival counters / o-exchange buffer of the retired whole-state V-split exchange: no instantiated kernel reads them
  p.ow = nullptr;   // (the Params fields stay: removing them would move the preloaded kernel arguments)
  p.mb = reinterpret_cast<unsigned*>(mb.data_ptr());
  p.x_stride = x.stride(0);
  p.conv_slot_stride = conv.stride(0);
  p.state_slot_stride = state.stride(0);
  p.nslots = (int)std::min(state.size(0), conv.size(0));  // pool warm-up bound: never beyond the smaller pool (REDTEAM R3)
  p.vec = (int)vec;
  p.late = (int)vec;
  p.pf = (int)pf;
  p.hold = (int)hold;
  p.nxt = (int)nxt;
  p.H = H;
  p.dbg = 0;
  p.eps = (float)eps;
  p.lb = (float)lb;
  TORCH_CHECK(dispatch(p, B, T / B, acc.has_value(), (int)nsplit, (int)nth, c10::hip::getCurrentHIPStream()),
              "kda decode: unsupported configuration (S ", T / B, ", spec ", acc.has_value(), ", nsplit ", nsplit, ", threads ", nth, ", vec ", vec, ", pf ", pf, ")");
}

// VGPRs per lane of the kernel run() would launch for (S, spec, nsplit, nth); -1 if unsupported.
int64_t regs(int64_t S, bool spec, int64_t nsplit, int64_t nth, int64_t nx) {
  Params p{};
  p.pf = 1;
  p.nxt = (int)nx;
  g_query = 1;
  g_regs = -1;
  const bool ok = dispatch(p, 1, (int)S, spec, (int)nsplit, (int)nth, nullptr);
  g_query = 0;
  return ok ? g_regs : -1;
}

void xcc_probe(torch::Tensor buf, int64_t gx, int64_t gy) {
  TORCH_CHECK(buf.numel() >= gx * gy, "kda decode: probe buffer too small");
  hipLaunchKernelGGL(kda_xcc_probe, dim3((unsigned)gx, (unsigned)gy), dim3(1024), 0, c10::hip::getCurrentHIPStream(), buf.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("xcc_probe", &xcc_probe);
  m.def("x1_xcc", []() { return (int64_t)X1_XCC; });
  m.def("regs", &regs);
}
#endif
#endif  // KDA_TOKLOOP_BODY
