"""v13_a (v74) = v73 + X1: the plain latency-bound whole-state kernel (1024 threads, prologue-first) runs on the one XCC that starts first (found by HW_REG_XCC_ID, as TN does). The dispatch hands the workgroups of a grid to the 8 XCCs round-robin, and physical XCC 1 starts 0.14-0.45 us before the other seven on every MI350X measured (UNDERFLOOR.md E5 / E8b); a grid of 8 H x B blocks in which only the blocks on XCC 1 work, on head blockIdx.x / 8 (the others return at entry) therefore ends ~0.4 us earlier while one XCC's memory path carries the grid's bytes inside a block's own span (cost model xcc_first(): B H x 128 KB / X1_XCC_BW <= X1_SPAN_US, i.e. <= 16 blocks; measured: -0.4 us to 16 blocks, neutral at 24, +0.3 at 32). One working block per head needs the round-robin to be exact (8 consecutive workgroups on 8 different XCCs; its first XCC depends on the HSA queue, so blockIdx alone cannot choose the block): that is checked once per device by probe launches outside any capture (_x1_probe(): 3 grids x 4 dispatches, eager), and until then, or on a device where it does not hold, the grid is v73's. Own instantiation kda_main<..., X1 = true> (kda_v13_a.cu): every other kernel is instruction-identical to v73. Also: KDA_SELF of the .cu points at the file itself (v70-v73 compiled the token loop of kda_v12x_w1.cu). v12f_d (v73) = v72 with the mailbox reset of v70 (R2) issued right behind the staging barrier in the 16-row kda_vec kernel for S >= 4 (kda_v12f_d.cu MB_CLR = 5, MB_LATE_S = 4) and indexed without a division everywhere: before the staging barrier the store costs the latency-bound 16-row grids 0.06-0.17 us (hab 120 pairs); 32-row kernels and S <= 3 keep the early reset. Fault injection (scratch_final/poison_rate.py, every mailbox word stale before a call): NOTES_final.md section 10. Only the 14 kda_vec kernels change. v12f_c (v72) = v71 + the constants that did not hold on held-out shapes (NOTES_final.md section 4; hab pairs on the v70 base): (1) TN (nontemporal odd-XCD tail of spec multi-round grids of > 2 rounds) from S = 4: TN_FIXED_US re-fit 1.4 -> 1.1 (the tail\'s fixed cost fell when WM removed the middle generations\' slow state; six S = 4 grids of > 2 rounds: -0.03..-0.61 us, four beyond CI, one +0.08 +- 0.09); the S = 4 two-copy kernel is its own instantiation (kda_v12f_c.cu ST_TNV), so S = 4 grids of <= 2 rounds keep their v71 kernel; (2) CG_ML also in the state-row-barrier (SB) kernel: the exception was neutral on eight held-out SB shapes (-0.06..+0.08 us, mean -0.01), one rule less; (3) the 16-row V-split step as a cost model, S x V16_US_PER_TOKEN > V16_FIXED_US (160 pairs each: 16-row at S = 2 +0.01..+0.06 us on four shapes, 32-row at S = 3 +0.09..+0.13: same dispatch as the literal S >= 3). v12f_b (v71) = v70 without the dead code (REDTEAM.md section 5, ABLATIONS.md 2.2; kda_v12f_b.cu: the launcher references only the instantiations the cost model can ask for, 43 kernels in the two builds instead of 574; every one of them is instruction-identical to v70 and the dispatch is identical on 109 854 shapes; host: the unused arrival-counter / o-exchange workspaces, HOLD_BW, FORCE_LATE and the V-split arm of hold_ticks() are gone). v12f_a (v70) = v69 + the red-team fixes (REDTEAM.md, NOTES_final.md; no dispatch change, kda_vec gains one store): (R1) V-split mailboxes are no longer one buffer per device: every captured call owns its mailbox (taken from spares that the eager calls keep filled: nothing extra is replayed) and eager calls use one per (device, stream), so launches of different graphs / streams that overlap on the device cannot read each other's partial sums or spin on each other's resets; (R2) kda_vec blocks return the mailbox words they are going to publish to the sentinel at entry (kda_v12f_a.cu switch MB_CLR), so a stale word (aborted launch) is healed by the next call instead of poisoning every later one, and reset_workspace() refills everything; (R3) run() validates dtype / shape / contiguity of every tensor (strided state_indices / num_accepted_tokens go to the AITER path, which honours strides; anything else raises) and the pool warm-up bound is the smaller of the two pools; (R6) the ROCm version is part of the build digest and a build on a toolchain other than the audited one warns; (R7) a failed register query raises instead of silently changing the dispatch, the occupancy of spec multi-round grids is queried from the instantiation that is launched, the CU count is per device, experiment environment variables are ignored unless KDA_EXPERIMENT=1, and the unsupported 16-row / vec = 0 launch is an error. v12x_w1 (v69) = v68 + W1: the WS wave offset also in the single-round spec kda_main (1024 threads, prologue-first, no next-round prefetch, no state-row barrier) on grids whose state-write phase at the HBM write ceiling outlasts the block\'s own token loop by at least the offset (cost model wave_offset(): blocks x S x 64 KB / WS_HBM_W - S x WS_TOK_US >= WS_US, i.e. blocks >= 191 / 156 / 138 / 127 / 120 / 115 / 111 for S = 2..8; requested with p.hold = -1 and launched as its own instantiation kda_main<..., W1 = true>, kda_v12x_w1.cu switch TP_WS1; every other instantiation is byte-identical to v68). Latency-bound grids below it pay the offset (80-132 blocks at S = 2..4: +0.2..+0.8 us). Mechanism as WM (NOTES_xcd.md section 6). v12x_wm (v68) = v67 + WM: in the spec multi-round kda_main the WS wave offset (waves 8-15 enter the token loop 80 x s_sleep(1) late) is applied in every generation whose blocks have a successor on their CU (b < blocks - next_round()), not only in the first one, for S >= 3 (kda_v12x_wm.cu switch TP_WSM). Mechanism (NOTES_xcd.md): the offset breaks a GPU-wide in-phase state of the token steps (without any offset 12 % of the processes at S = 4 sit in a mode 6-7 us slower; an offset on any two XCDs removes it); the blocks of an XCD\'s generation end within < 1 us, so each later generation re-forms it (per-replay stamps: slow replays lose 7 us GPU-wide in the token loop of generation 1); tail blocks (no successor) are excluded, they pay a delay 1 : 1. Grids of <= 2 rounds have no middle generation and run v67 instructions\' paths. The dispatch is unchanged. v11a_ab (v67) = v66 after the first global leave-one-out (NOTES_ablate.md): (1) read-first hold re-fit as a two-term cost model, hold = HOLD_CU_US x ceil(blocks / CUs) + state bytes / HOLD_BW2 (the busiest CU's chain of co-resident state reads + the grid's bytes at the read rate; the v16-v66 form, bytes / 5.8e6, was a compromise between full grids and fractional fills: 36 held-out fills, 16 faster beyond CI by 0.4-1.3 us, one slower by 0.13 us); (2) features that no longer pay removed / unified, each neutral on the eval (160 pairs) and on held-out pairs: AB_NOPIN=2, PR_VNOA=2, AB_NORR2V=1, SNT=2, AB_PLD=12, PR_V16VCE=0, SPOL_PLAIN=17 (kda_vec fake-use pins, kda_vec exchange-wave pool warm-up, interleaved kda_vec row reductions, early kda_vec v-tap commit, per-class small-store / prologue-load / plain state-store policies now uniform). kda_v11a_ab.cu = kda_v11s_tn.cu + the leave-one-out switches with those defaults. v11s_tn (v66) = v65 + TN: in the spec multi-round kda_main (1024 threads, prologue-first, next-round prefetch) the blocks that write in the odd-XCD tail of the grid issue their state stores nontemporal (aux 2) instead of write-through (sc0 sc1): blocks without a successor on their CU, on odd XCDs (HW_REG_XCC_ID), in grids of more than two rounds with S >= 5 (cost model tail_nt(); first tail block passed in p.hold, which this kernel did not use; the S = 5..8 instantiations change, kda_v11s_tn.cu switch ST_TNS). The token loop text is a section at the top of kda_v11s_tn.cu that kda_main includes once per store policy; the multi-round instantiation carries two copies behind one block-uniform branch in front of the loop (a branch inside the token step is a scheduling barrier: +3.4..+6.6 us; every other instantiation includes the loop once and is instruction-identical to v65). Mechanism (NOTES_struct.md): XCDs 2k / 2k+1 share a write path on which the even one wins, so the odd XCDs finish the grid alone; a lone XCD writes 1.25 TB/s nontemporal vs 1.0 TB/s write-through, while under a saturated HBM nontemporal stores lose 13-21 % (so not on every block: odd XCDs in every generation +1.2..+3.6 us). A/B v65 -> v66: see NOTES_struct.md. The dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged. v10t_ws (v65) = v64 + WS: in the spec multi-round kda_main (1024 threads, prologue-first, next-round prefetch) the upper half of a block's waves (8-15) enters the token loop 80 x s_sleep(1) (~2.3 us) after the lower half, once per block, in the first generation only (b < next_round(): the blocks that all start together; later blocks start staggered, and a last-generation block that runs alone would pay the offset 1:1) (kda_v10t_ws.cu switch TP_WS = 80; NOTES_tpmodel.md). Delay injection: after the norm barrier all 16 waves of a block run the token steps in phase; a one-time offset of half of the waves costs nothing on any spec multi-round grid (waves 8-15 delayed 0.3 / 0.6 / 1.2 us at token 0: s4_k3_b64 -0.28 / -0.35 / -0.39 us on the harness build). The dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged; every other kernel runs v64 instructions. v10c_a (v64) = v63 + instruction-stream work on the critical phases (NOTES_codegen.md; kda_v10c_a.cu switches CG_FH = CG_SL = CG_TP = 1, CG_ML = 4): (FH) V-split kda_vec staging loads from per-wave compile-time run tables: per lane (tensor base select) + (32-bit run offset select) + lane * 16 with the bases / offsets on the SALU, instead of six 64-bit per-lane segment addresses and five selects (~100 VALU in front of every first-hop load); conv history as one buffer load whose lane offsets are computed ahead of the slot wait; beta as a buffer load. (SL) kda_vec slot-chain s_loads issued together with the Params kernarg read behind one wait. (TP) kda_vec token prologue 161 -> 137 non-scalar instructions: packed SiLU / decay math with folded scalars, the three wave sums as one DPP asm block. (ML) spec 1024-thread prologue-first single-round kda_main (mid-B) without the state-row barrier, i.e. where the prologue loads race the block's state rows: every prologue load as (wave-uniform base on the SALU) + (32-bit lane offset), one instruction per load. All four are per-instantiation compile-time forms; the dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged. A/B v63 -> the same stack with ML on every single-round spec kda_main, full-17: 160 pairs +0.0045 [+0.0037, +0.0053], 80 pairs +0.0041 [+0.0028, +0.0054] (s4_k3_b1 -0.33 / -0.35 us, s6_glm16_b16 -0.15 / -0.08, s6_glm8_b32 -0.11 / -0.10; s8_k3_b16, the barrier instantiation, -0.07 +- 0.08 / +0.17 +- 0.12 and its held-out shapes 0.00..+0.08 -> ML left out there). KDA_CG (env): extra -D switches for codegen experiments, empty by default. v9p_b (v63) = v62 after the leave-one-out of the probe stack (NOTES_probes2.md; kda_v10c_a.cu = scratch_probes2_run/kda_pr16.cu with PR_NOA1 = 3, PR_WUB = 64, PR_MRWU0 = PR_NX32 = PR_V16VALL = PR_ACLAMP = PR_V16VCE = PR_STAGPOS = 1): + plain multi-round first-generation stagger wait issued after the block's inputs and conv history, right before its state-row loads (was at block entry: the small loads of a delayed block no longer wait out the stagger; STAG re-swept at the new position, 150 ticks stays); - the passenger of v62 (host-side NX target without the runtime division: leave-one-out 0.00 on all five multi-round workloads). v9p_a (v62) = v61 + probe stack S3 (NOTES_probes2.md; kda_v9p_a.cu = scratch_probes2_run/kda_pr15.cu with PR_NOA1 = 3, PR_WUB = 64, PR_MRWU0 = PR_NXND = PR_NX32 = PR_V16VALL = PR_ACLAMP = PR_V16VCE = 1): kda_main without the out-page (a1) warm-up load and with the pool translation warm-up (a0, 64 loads on wave 4, ahead of its state rows in vmcnt order) only in the first 64 blocks of the grid (8 per XCD; later blocks find the translations warm); plain multi-round: a0 ahead of the first-generation stagger; multi-round next-round prefetch target (n, h) from host-side nxt / H, nxt % H (no runtime division) and, in spec, scalar bases + 32-bit lane offsets; kda_vec: v conv written by all lanes (no exec branch), accepted-count clamp on the SALU, V16 v-tap commit after the prologue. A/B v61 -> S3 full-17 160 pairs +0.0035 [+0.0027, +0.0043], every workload faster. v9m_sc (v61) = v60 with the barrier variant as its own kda_main instantiation (the v60 runtime branch in front of the state-row loads cost the grids that do not take it +0.06..+0.18 us). v9m_sb (v60) = v59 + spec whole-state prologue-first single-round grids (mid-B): the state rows of all waves are issued behind a block barrier (after the prologue waves issued their inputs and conv history) while the state read phase is shorter than the prologue chain (state_after_prologue(): B*H*64 KB / HOLD_BW_NT <= SPEC_PRO_US). From the delay-injection model of the mid-B kernel (NOTES_b1model.md section 6). v9m_hf (v59) = v58 + V-split kda_vec staging: a wave loads, waits for and stages a 16 B chunk batch (second first-hop batch, conv-history batch) only if it owns a real chunk of it (wave-uniform branch); the other waves used to run the address code, load and LDS write for a clamped duplicate chunk. Delay injection (NOTES_b1model.md) shows every instruction a wave issues before the staging barrier costs event time when the SIMDs carry two prologue waves each. A/B v58 -> v9m_hf: see NOTES_b1model.md. v9m_nb (v58) = v57 + 16-row V-split kda_vec with more prologue tokens than token waves (S > 4): the block barrier between the prologue math and the token loop is replaced by LDS ready flags. Prologue token t runs on wave t; waves 4-7 share a SIMD with the token waves 0-3, issue is age-ordered, so the token waves finish their own tokens ~0.2 us before the younger waves and used to idle at the barrier (delay-injection knees, NOTES_b1model.md: token waves 0.44 us, younger waves 0.66 us after the staging barrier). Now each prologue wave announces its token with one LDS word after its operand writes; the token waves start the loop when tokens 0-3 are announced and wait for tokens 4..S-1 once, before the lookahead first reads token 4. S <= 4 keeps the barrier (no younger prologue wave; a poll costs more than the barrier: +0.08 us). A/B v57 -> v9m_nb, 80 pairs full 17: +0.0020 [+0.0008, +0.0032] (s8_k3_b1 -0.06 +- 0.03, s6_glm16_b1 -0.11 +- 0.03 us, everything else unchanged). v8p_a (v57) = v56 + plain multi-round kda_main: conv commit after the token loop with the taps kept in VGPRs (72-77 VGPRs, so the VGPR-based occupancy model blocks_per_cu() runs the multi-round grid at 1 block per CU, nxt 256) and no block barrier before the state-row loads (SBAR 0: at 1 block per CU the barrier only delays the 64 KB read); plain kda_main s_o written by all 16 lanes (no exec branch) and thread-contiguous s_v rows (as in spec); epilogue rsq as raw v_rsq_f32 (ss/D + eps >= 1e-5); spec late conv commit only for S >= 7 (PROBES P60b, M1a, P46, P56, P30, P20; NOTES_probes.md; kda_v8p_a.cu = scratch_probes_run/kda_pr2.cu with PR_WUW = PR_GRID2D = PR_MRLATE = PR_SOALL = PR_SVC = PR_RSQ = 1, SBAR = 0, LCS = 7). v8i_g2 (v56) = v55 + 2-D launch grid (x = H * NSPLIT, y = B): (n, h, split) from blockIdx without a runtime 32-bit division ahead of every block's first address (PROBES P47; kda_v8i_g2.cu = the probe harness kda_pr.cu with PR_WUW = PR_GRID2D = 1, every other switch 0). v8i_wu (v55) = v54 + the kda_main translation / Params warm-up loads issued by wave 4 (the first non-prologue wave) instead of wave NW - 1, which is dispatched last (PROBES P42; ISSUE_PATH.md). A/B v54 -> v8i_wu 80 pairs +0.0029 [+0.0020, +0.0038]: s4_k3_b64 -0.79, s6_glm16_b64 -0.52, s4_k3_b16 -0.11 us. v7o_v54 (v54) = v7o_bl (v53) with the plain multi-round first-generation stagger recalibrated for the shorter v53 blocks: levels 0 / 0.75 / 1.5 us (STAG 340 -> 150 ticks, 3 levels; v53 blocks are ~3 % shorter and the 1.7 us level spacing fitted on v51 now over-delays). A/B v53 -> v54 (160 pairs) +0.0008 [+0.0001, +0.0016]: d_k3_b128 -0.36, d_glm16_b256 -0.32 us; STAG sweep (40 pairs each) 0: +0.09/+0.81, 150: -0.45/-0.34, 200: -0.33/-0.44, 250: -0.24/-0.49, 450: +0.32/+0.48 us. v7m_mr4 (v52) + every kda_main / kda_vec state-row load as a raw buffer load (buffer_load_dwordx4, SGPR descriptor of the block's own 64 KB slice + 32-bit lane offset, aux 2 = nt) instead of a 64-bit-address global_load_dwordx4 nt (CONVENTIONS rank 3 / U2: the floor harnesses always used buffer loads). A/B v52 -> v53 +0.0069 [+0.0058, +0.0081] (80 pairs): d_glm16_b256 -3.59, d_k3_b128 -1.06 us; aux 3 / 18 = aux 2 (n.s.). v6b_spl3 (v51) + plain multi-round kda_main at 2 blocks per CU: state stores batched after the row math (64 VGPRs; occupancy and next-round prefetch distance from the compiled multi-round instantiation itself) and a one-time desynchronisation of the first co-resident generation: blocks of sequence n with n * H < CUs * blocks per CU wait 0 / 1.7 / 3.4 us by a hash of n (MECHANISMS.md (a): identical co-resident blocks that start together stay phase-locked; keyed per sequence so its H heads stay together). v6b_spl3 = v6b_s1r + spec V-split ns 2 -> 4 step allowed while the split grid fits in all CUs (other split steps unchanged). v6b_s1r = v6b_k0 + spec 1024-thread grids of 1-2 rounds prologue-first with next-round prefetch (cost-model fix; no eval workload in this class). v6b_k0 = v6b_sb + the spec multi-round next-round prefetch issued after the first token (was at block start). v6b_sb = v6b_hb + a block barrier before the state-row loads in the plain multi-round kda_main (the prologue waves issue inputs + conv history first). v6b_hb = v6b_e2 + cached beta load in the plain hold kernel too (beta: a line shared by the H heads; only the spec multi-round kernel keeps nt beta). v6b_e2 = v6b_u + cached (MALL-allocating) beta load in the spec kda_main kernels without next-round prefetch (the beta line is shared by the H heads of a token). v6b_u = v6b_n + nontemporal LDS-DMA next-round prefetch (NX) loads in the plain and spec multi-round kda_main. v6b_n = v6b_q + nontemporal small stores (conv commit, out) in the spec multi-round kda_main. v6b_q = v6b_a + prologue-first load order (<= 2 blocks per CU) (nt prologue loads, default small stores) in the plain co-resident hold kernel. v6b_a = v41 + plain co-resident grids always on the whole-state 256-thread hold kernel (no plain V-split, no 1024-thread 1-2-round grids) + hold floor HOLD_MIN. Fused KDA decode / MTP verify for MI350X: HIP kernels in kda_v6l_pq.cu (JIT-built per scheduler variant, cached). v6l_pq (v41) = v40 + packed (2-wide) h.q dot in the spec kda_main token loop (as h.k already was). v6l_s2 (v40) = v39 + spec kda_main per-token norm scalars (rq * scale, rk, q.k, beta) computed once per block into LDS behind one extra barrier instead of in every wave for every token (TRANSFER S2). v6l_s4 (v39) = v38 (v5b_m) + the V-split kda_vec kernels from the max-ILP build with a sched_barrier at each token end (TRANSFER S4: the earlier "max-ILP slows kda_vec" result was cross-token hoisting to 168 VGPRs, not the scheduler; now 68-74 VGPRs). v5b_m = v37 (v5b_h) + system-scope small bf16 stores (conv commit, out) in the whole-state prologue-first plain kernels; v5b_h = v36 + sc1 nt state stores in the plain multi-round kernels; v5b_f = v35 (v5b_e: nontemporal per-(n, h) prologue loads in the plain whole-state state-first co-resident hold kernels) + the same in every spec kda_main kernel (read-first hold recalibrated). v34 = v33 (mailbox exchange; 16-row V-split for S >= 3) + kda_vec prologue with a compile-time token per wave, and in the 16-row mode a one-token LDS operand lookahead with interleaved k/q row reductions and DPP row-broadcast wave sums."""
import ctypes
import hashlib
import os
import warnings

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kda_v13_a.cu")
_SMAX = 8
_mods = {}
_cnt = {}
_NUM_CU = None
_cus = {}
_AUDITED_ROCM = "7.2.0"  # toolchain whose ISA passed the asm hazard / DPP / spill checks (scratch_final/validate_all.sh)
FORCE_NSPLIT = None
FORCE_NTH = None
FORCE_VEC = None
FORCE_PF = None
FORCE_HOLD = None
HOLD_MIN = 4.9  # us: hold floor for small co-resident grids (1.25-1.75 blocks per CU; hold sweeps NOTES_bw2)
HOLD_CU_US = 0.945  # us per co-resident block on the busiest CU: per-CU term of the read-first hold (hold_ticks(); fitted on 12 fills, checked on 24 others)
HOLD_BW2 = 9.1e6  # bytes per us: grid term of the read-first hold (state bytes of the whole grid at the read-only rate)
HOLD_BW_NT = 5.8e6  # bytes per us: state-read rate of a single-round spec grid beside its prologue chain (state_after_prologue(); it was the
# plain hold's rate before the two-term re-fit of v67: NOTES_bw M8, hold sweeps 5.6-6.2e6)


def _ext(ilp=False):
    """ilp=False: all kernels, default machine scheduler. ilp=True: spec kernels only, built with the max-ILP machine
    scheduler (-amdgpu-sched-strategy=max-ilp); used for every spec kernel: kda_main, whose token loop it schedules
    better (A/B: every spec kda_main workload -0.1..-1.6 us), and kda_vec (with per-token sched_barriers: B=1 spec
    -0.0..-0.17 us, held-out V-split shapes -0.03..-0.18 us; NOTES_lat2 S4). It slows the plain kernels (NOTES_opt4 "ILP")."""
    if ilp not in _mods:
        # experiment switches (store policies, warm-up stride, extra -D, debug / stamp builds) only with KDA_EXPERIMENT=1:
        # a variable left over in the environment must not change the kernel (REDTEAM R7.3)
        env = os.environ if os.environ.get("KDA_EXPERIMENT") == "1" else {}
        pol = env.get("SPOL", "17,17").split(",")
        pfs = env.get("PFSTRIDE", str(2 << 20))
        cg = env.get("KDA_CG", "").split()  # extra -D switches (codegen experiments; empty = defaults in the .cu)
        rocm = _rocm_version()
        if rocm != _AUDITED_ROCM:
            warnings.warn(f"kda decode kernel: building with ROCm {rocm}, audited on {_AUDITED_ROCM}: the hand-written asm loads / DPP blocks "
                          "were only checked in that toolchain's ISA; rerun scratch_final/validate_all.sh (asm checkers) before trusting it")
        digest = hashlib.sha1(open(_SRC, "rb").read() + " ".join(cg).encode() + rocm.encode()).hexdigest()[:10] + "_b" + "_p" + "_".join(pol) + "_s" + pfs + ("_ilp" if ilp else "")
        sfx = ("_dbg" if env.get("KDA_DBG") else "") + ("_st" if env.get("KDA_STAMP") else "")
        bdir = os.path.join(_HERE, "build", "v13a_" + digest + sfx)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        xf = ["-mllvm", "-amdgpu-sched-strategy=max-ilp", "-DKDA_SPEC_ONLY"] if ilp else []
        try:
            _mods[ilp] = load(name=f"kda_v13a_{digest}" + sfx, sources=[_SRC], extra_cflags=["-O3"],
                              extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-mllvm", "-amdgpu-kernarg-preload-count=16",
                                                 "-DSPOL_SPEC=" + pol[0], "-DSPOL_PLAIN=" + pol[1], "-DPFSTRIDE=" + pfs, "-DSTAG=150", "-DSTAGL=3", "-DBLD=2"] + xf + ["-D" + f for f in cg]
                              + (["-DKDA_DBG"] if env.get("KDA_DBG") else []) + (["-DKDA_STAMP"] if env.get("KDA_STAMP") else []),
                              build_directory=bdir, verbose=False)
        finally:  # leave the process environment as it was (REDTEAM R7.6)
            if arch0 is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = arch0
    return _mods[ilp]


def _rocm_version():
    try:
        from torch.utils.cpp_extension import ROCM_HOME
        return open(os.path.join(ROCM_HOME, ".info", "version")).read().strip()
    except Exception:
        return "unknown"


_retired = []
_SENT = -1082130432  # -1.0f: mailbox sentinel (a partial sum of squares is >= 0 or NaN)
_MB_WORDS = 8 * 64  # mailbox words per (n, h): NSPLIT x NSPLIT x S <= 8 x 8 x 8
_mbox = {}  # (device, stream) -> mailbox of the eager calls on that stream
_captured = []  # mailboxes owned by captured calls (kept alive: the graph points at them)
_spare = {}  # device -> sentinel-filled mailboxes ready for the next captured calls
_spare2 = {}  # device -> the same for grids above 64 pairs (128-pair mailboxes; 2-way split)
_NSPARE = 16  # per device and size class (401 kB / 803 kB each: 19 MB per device). A capture that finds no spare allocates on the capturing
# stream and its graph then refills that workspace at every replay: +4 us = +20..27 % on a B x H <= 82 spec call, for the life of the graph.
# HIP permits no eager work on ANY stream while one captures (tried: a fill on a side stream invalidates the capture), so the spares must
# exist beforehand. With 2 (24 and before) the third capture of V-split calls behind decode_warmup() with no eager call in between
# was hit (work_final_gatekeeper/tools/warmup_cover.py). An engine that captures more than 16 graphs with V-split calls in a row calls
# decode_warmup() again in between (it tops the spares up).


_MB_REC = 1056  # words per (n, h) in front of the mailbox words (kda_v14_b.cu MB_REC with MB_EPOCH): a line with the 8 leave flags, a line
# with the pair's epoch word (word 16: any content is an epoch), then S x 128 o words of a block that left


def _mb_new(dev, n):
    """A sentinel-filled workspace for n (n, h) pairs; returns the view the kernel is given: the mailbox words, with the leave
    records of the bounded poll (MB_BOUND) in the n * _MB_REC words in front of it (same storage; record g ends at view - g * _MB_REC)."""
    full = torch.full((n * (_MB_REC + _MB_WORDS),), _SENT, dtype=torch.int32, device=dev)
    return full[n * _MB_REC:]


# ---- workspaces of captured calls: one per (capture, stream), released when the graph is destroyed (iteration 7, backlog F).
# v77 gave every captured V-split CALL its own workspace and kept it for the life of the process (nothing told the launcher that a graph
# had died): 395 KiB per call (790 KiB for a 2-way call), 27 MiB for one graph holding the 69 KDA layers of a model, and the same again for
# every re-capture. Two facts of the HIP runtime end that (work_decode_gatekeeper/tests/userobj_probe.py):
#  - hipStreamGetCaptureInfo_v2 names the capture a call belongs to (id, graph handle). Calls captured on ONE stream in ONE capture are
#    chained in the graph: they are never in flight together, whatever stream the graph is replayed on. They share one workspace, exactly
#    as eager calls on one stream do (the kernel returns every word to the sentinel before it ends).
#  - a user object retained by the graph (hipUserObjectCreate / hipGraphRetainUserObject) is destroyed with the graph (torch drops the
#    graph when the CUDAGraph object goes): its callback names the capture, and the workspace goes back to the launcher. The callback only
#    appends to a list; the next launcher call (under the launcher's lock) moves the workspaces of dead captures to a free list, and the
#    next EAGER call refills them with the sentinel and makes them the spares of later captures. As for the memory torch itself
#    allocates inside a capture and releases with the graph: a graph must not be destroyed while one of its replays is in flight.
# If the runtime does not provide either call the behaviour is v77's (one workspace per captured call, never released).
SHARE_PER_CAPTURE = True  # False: v77
_NFREE = 8  # workspaces of dead graphs kept per device for reuse; more are handed back to the allocator
_cap = {}  # (device, stream, capture id) -> workspaces of that capture
_free = {}  # device -> workspaces of dead graphs (any content: refilled before they are used again)
_dead = []  # capture keys whose graph was destroyed (found by _reap() in the callback cells)
_uo = {}  # unused since iteration 8 (kept: the launcher's registry names it)
_hipl = []


def _hip_lib():
    """The HIP runtime this process runs on (the libamdhip64 in its address space), or None."""
    if not _hipl:
        lib = None
        try:
            with open("/proc/self/maps") as f:
                paths = {l.split()[5] for l in f if "libamdhip64" in l and len(l.split()) > 5}
            if len(paths) == 1:
                lib = ctypes.CDLL(sorted(paths)[0])
                lib.hipStreamGetCaptureInfo_v2, lib.hipUserObjectCreate, lib.hipGraphRetainUserObject, lib.hipUserObjectRelease  # noqa: B018
        except Exception:  # noqa: BLE001
            lib = None
        _hipl.append(lib)
    return _hipl[0]


# The destroy callback of the user object is libc's time(): it stores a non-zero word through the pointer it is given, a cell in a block
# that is never freed. No Python runs in the callback, so it is valid whenever the runtime fires it: from another thread, or while the
# interpreter shuts down (24 and before used a ctypes thunk: a process that exited while it still held a captured graph with a V-split call
# crashed at exit, status 139, because the thunk was freed before the graph; work_final_gatekeeper/tools/exit_probe.py).
_CELLS = 256  # cells per block
_cblk = []  # [address, view of the block's bytes, free cell indices]
_cell = {}  # capture key -> (block index, cell index)
_ZBLK = bytes(8 * _CELLS)
_libc = []


def _time_fn():
    """Address of libc's time(time_t*), or None."""
    if not _libc:
        fn = None
        try:
            c = ctypes.CDLL(None)
            c.calloc.restype = ctypes.c_void_p
            c.calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            fn = (c, ctypes.cast(c.time, ctypes.c_void_p).value)
        except Exception:  # noqa: BLE001
            fn = None
        _libc.append(fn)
    return _libc[0]


def _cell_new(key):
    """Address of a zeroed cell for `key`."""
    for bi, (addr, _view, free) in enumerate(_cblk):
        if free:
            ci = free.pop()
            break
    else:
        addr = _time_fn()[0].calloc(_CELLS, 8)
        if not addr:
            return None
        _cblk.append([addr, (ctypes.c_char * (8 * _CELLS)).from_address(addr), list(range(_CELLS - 1, 0, -1))])
        bi, ci = len(_cblk) - 1, 0
    ctypes.c_longlong.from_address(addr + 8 * ci).value = 0
    _cell[key] = (bi, ci)
    return addr + 8 * ci


def _capture_key(dev):
    """((device, stream, capture id), graph handle) of the capture the current stream is in; (None, None) if the runtime does not say."""
    lib = _hip_lib()
    if lib is None or not SHARE_PER_CAPTURE:
        return None, None
    st = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    status, cid, graph, deps, nd = ctypes.c_int(0), ctypes.c_ulonglong(0), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_size_t(0)
    rc = lib.hipStreamGetCaptureInfo_v2(st, ctypes.byref(status), ctypes.byref(cid), ctypes.byref(graph), ctypes.byref(deps), ctypes.byref(nd))
    if rc != 0 or status.value != 1 or not graph.value:
        return None, None
    return (dev, st.value or 0, cid.value), graph.value


def _retain(key, graph):
    """Tie `key` to the life of the graph being captured: a user object whose destroy callback marks the key's cell. False if the runtime refuses."""
    lib = _hip_lib()
    tf = _time_fn()
    cell = _cell_new(key) if tf is not None else None
    if cell is None:
        return False
    obj = ctypes.c_void_p()
    if lib.hipUserObjectCreate(ctypes.byref(obj), ctypes.c_void_p(cell), ctypes.c_void_p(tf[1]), ctypes.c_uint(1), ctypes.c_uint(1)) != 0:  # NoDestructorSync
        ctypes.c_longlong.from_address(cell).value = 1  # reaped with nothing attached
        return False
    if lib.hipGraphRetainUserObject(ctypes.c_void_p(graph), obj, ctypes.c_uint(1), ctypes.c_uint(1)) != 0:  # hipGraphUserObjectMove
        lib.hipUserObjectRelease(obj, ctypes.c_uint(1))  # (fires the callback: the key is reaped with nothing attached)
        return False
    return True


def _reap():
    """Workspaces of captures whose graph is gone: out of the live registry, onto the free list (class-sized ones, at most _NFREE)."""
    for blk in _cblk:
        if blk[1].raw != _ZBLK:  # some cell of this block was marked by the runtime
            for key, (bi, ci) in list(_cell.items()):
                if _cblk[bi] is blk and ctypes.c_longlong.from_address(blk[0] + 8 * ci).value:
                    ctypes.c_longlong.from_address(blk[0] + 8 * ci).value = 0
                    del _cell[key]
                    blk[2].append(ci)
                    _dead.append(key)
    while _dead:
        key = _dead.pop()
        for mb in _cap.pop(key, ()):
            for i, t in enumerate(_captured):
                if t is mb:
                    del _captured[i]
                    break
            fr = _free.setdefault(key[0], [])
            if mb.numel() in (64 * _MB_WORDS, 128 * _MB_WORDS) and len(fr) < _NFREE:
                fr.append(mb)


def _whole(mb):
    """The whole storage of a workspace (leave records + mailbox words) as one int32 tensor."""
    return torch.empty(0, dtype=torch.int32, device=mb.device).set_(mb.untyped_storage())


def _spare_new(dev, pairs):
    """A sentinel-filled spare: a dead graph's workspace of that size if there is one (refilled here, eagerly), else a new one."""
    fr = _free.get(dev, [])
    for i, t in enumerate(fr):
        if t.numel() == pairs * _MB_WORDS:
            mb = fr.pop(i)
            _whole(mb).fill_(_SENT)
            return mb
    return _mb_new(dev, pairs)


def workspace_report():
    """Live captured workspaces, their bytes, captures tracked, free-list workspaces, spares: for operators and tests."""
    _reap()
    size = lambda t: _whole(t).numel() * 4  # noqa: E731
    return dict(captured=len(_captured), captured_bytes=sum(size(t) for t in _captured), captures=len(_cap),
                free=sum(len(v) for v in _free.values()), spares=sum(len(v) for v in list(_spare.values()) + list(_spare2.values())),
                eager=len(_mbox), retired=len(_retired))


def _mailbox(dev, n):
    """V-split partial-sum mailboxes for n = B * H (n, h) pairs, all words at the sentinel between calls (the kernel returns
    them there). Two launches that are in flight together must not share one (REDTEAM R1: they would consume each other's
    partial sums or spin on each other's resets), so
    - calls under HIP-graph capture get one mailbox per (capture, stream) (64 pairs; 128 pairs for the 2-way split), see above.
      HIP permits no eager work
      while a stream captures, so it is taken from a few spares that every eager call keeps filled: the replay is the same
      single kernel launch. Only a capture with no eager call before it (or of a larger grid) allocates inside the capture;
      its sentinel fill is then part of the graph (correct, one small fill slower);
    - eager calls share one per (device, stream), grown on demand (outgrown ones are kept alive).
    Replaying one graph concurrently with itself stays undefined, as for any kernel that writes `out`."""
    spare = _spare.setdefault(dev, [])
    big = n > 64  # the 2-way split (split2(): up to 128 pairs) has its own spares, made at the first eager call of such a grid: the 64-pair
    # spares, and with them every allocation of a process that never runs a 2-way grid, stay exactly as they were
    if big:
        spare = _spare2.setdefault(dev, [])
    _reap()
    if torch.cuda.is_current_stream_capturing():
        key = graph = None
        try:
            key, graph = _capture_key(dev)
        except Exception:  # noqa: BLE001  (a runtime without the call: v77 behaviour)
            key = None
        if key is not None:
            for mb in _cap.get(key, ()):
                if n * _MB_WORDS <= mb.numel():
                    return mb  # an earlier call of this capture on this stream: chained in the graph, never in flight together
        if spare and n * _MB_WORDS <= spare[-1].numel():
            mb = spare.pop()
        else:
            mb = _mb_new(dev, n)
        _captured.append(mb)
        if key is not None:
            try:
                if key in _cap or _retain(key, graph):
                    _cap.setdefault(key, []).append(mb)
            except Exception:  # noqa: BLE001
                pass
        return mb
    while len(spare) < _NSPARE:
        spare.append(_spare_new(dev, 128 if big else 64))
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    mb = _mbox.get(key)
    if mb is None or mb.numel() < n * _MB_WORDS:
        if mb is not None:
            _retired.append(mb)
        mb = _mbox[key] = _mb_new(dev, max(n, 64))
    return mb


def _workspace(dev, n, vec):
    """The mailbox argument of the launch: _mailbox() for the V-split kda_vec kernels (vec), a device-wide placeholder that no
    kernel touches otherwise."""
    w = _cnt.get(dev)
    if w is None:
        w = _cnt[dev] = torch.full((64,), _SENT, dtype=torch.int32, device=dev)
    return _mailbox(dev, n) if vec else w


def reset_workspace():
    """Refill every mailbox with the sentinel (eager, synchronous). Not needed in normal operation: the kernel returns every
    word to the sentinel and resets its own words at entry; call it after a device error / aborted launch if in doubt."""
    _reap()
    for mb in (list(_mbox.values()) + _captured + [t for t in _retired if torch.is_tensor(t)]
               + [t for l in list(_spare.values()) + list(_spare2.values()) + list(_free.values()) for t in l]):
        _whole(mb).fill_(_SENT)  # the leave records in front of the view too
    torch.cuda.synchronize()


_MB_ST0, _MB_TOMB = 8, 0x40000000  # kda_v14_b.cu MB_GHOST: rec[8 + split] = stamps; a stamp with this bit = a tombstone
_MB_SEEN = 1  # kda_v14_b.cu MB_SEEN: sticky mark of a stamp word whose split once got a tombstone (kept by later stamps until a refill)


def workspace_health(heal=False):
    """Are the V-split workspaces of this process in the state every completed launch leaves them in? Host side, synchronous: it waits
    for the devices, so call it between steps, with no decode call of the process in flight.

    Every launch that runs inside the contract returns every mailbox word and every leave flag to the sentinel before it ends, also when
    its blocks leave / claim beside a co-tenant (work_release_gatekeeper/tools/ghost_probe.py, serve_soak.py). Words or flags that are
    NOT at the sentinel between steps were left by something outside the contract (a graph replayed concurrently with itself, a launch
    that was killed), and a later launch one of whose blocks is dispatched late can consume them: that launch returns a wrong result
    with no other sign (such leftovers can carry the pair's CURRENT epoch, so the epoch tag does not reject them: kda_v14_b.cu MB_GHOST).
    The kernel cuts that short by itself after at most two launches and leaves a tombstone where it did; this is the detector for
    both: `stale_words` / `raised_flags` = leftovers that have not been consumed yet, `ghost_events` = tombstones and the sticky marks
    they leave in the stamp words (MB_SEEN: kept until a refill, also behind the self-healing chain): launches of the workspace DID
    consume leftovers with a raised flag since the last refill, i.e. results since the last clean check are suspect. (Not a proof of
    the opposite: a leftover whose flag was already gone is consumed without a mark; only a check BEFORE the next launch sees it.)
    heal=True refills the workspaces found dirty (reset_workspace() refills all of them unconditionally).
    Returns {"ok", "workspaces", "dirty", "stale_words", "raised_flags", "ghost_events", "healed"}."""
    _reap()
    rep = dict(ok=True, workspaces=0, dirty=0, stale_words=0, raised_flags=0, ghost_events=0, healed=0)
    todo = [t for t in list(_mbox.values()) + _captured + _retired if torch.is_tensor(t)]
    for d in {t.device for t in todo}:
        torch.cuda.synchronize(d)
    counts = []
    for mb in todo:
        full = _whole(mb)
        nrec = full.numel() - mb.numel()
        if nrec:
            head = full[:nrec].view(-1, _MB_REC)
            st = head[:, _MB_ST0:_MB_ST0 + 8]
            flags, tomb = (head[:, :8] != _SENT).sum(), ((st >= 0) & ((st & (_MB_TOMB | _MB_SEEN)) != 0)).sum()
        else:
            flags = tomb = torch.zeros((), dtype=torch.int64, device=mb.device)
        counts.append(torch.stack(((mb != _SENT).sum(), flags, tomb)))
    for mb, c in zip(todo, counts):
        w, f, g = (int(v) for v in c.tolist())
        rep["workspaces"] += 1
        rep["stale_words"] += w
        rep["raised_flags"] += f
        rep["ghost_events"] += g
        if w or f or g:
            rep["dirty"] += 1
            if heal:
                _whole(mb).fill_(_SENT)
                rep["healed"] += 1
    if rep["healed"]:
        for d in {t.device for t in todo}:
            torch.cuda.synchronize(d)
    rep["ok"] = rep["dirty"] == 0
    return rep


_occ_cache = {}


def blocks_per_cu(S, spec, ns, nth, nx=0):
    """Resident blocks per CU from the compiled VGPR count (512 VGPRs per SIMD lane, <= 8 waves per SIMD); nx: of the
    multi-round (next-round prefetch) instantiation, whose batched-store plain build fits 64 VGPRs (2 blocks per CU)."""
    key = (S, spec, ns, nth, nx)
    if key not in _occ_cache:
        r = _ext(spec).regs(S, spec, ns, nth, nx)  # kda_main: the spec build for spec
        if r <= 0:  # no device / unsupported instantiation: a guessed occupancy would silently change the dispatch (REDTEAM R7.1)
            raise RuntimeError(f"kda decode kernel: register-count query failed for S={S} spec={spec} ns={ns} nth={nth} nx={nx}")
        waves = min(8, 512 // ((r + 7) // 8 * 8))
        _occ_cache[key] = max(1, waves * 4 // (nth // 64))
    return _occ_cache[key]


V16_US_PER_TOKEN = 0.14  # us per token step saved by 16-row blocks over 32-row ones (half the row work per SIMD in the token waves)
V16_FIXED_US = 0.31  # us: what the 16-row step costs whatever S is (twice the blocks: eight redundant q / k prologues per head, 8-way exchange)


V2_US_PER_TOKEN = 0.33  # us per token step saved by the 2-way split (64-row blocks, kda_v17_a.cu) over one 1024-thread block per pair: the
# whole-state block is VALU-issue bound (16 waves on one CU, ~0.9 us per token) while more than half of the CUs idle
V2_FIXED_US = 1.15  # us: what the 2-way split costs whatever S is (staged prologue twice per head, mailbox exchange)
V2_WRITE_US = 0.008  # us per token and per pair above half the CUs: the split grid becomes write-bound earlier (twice the store demand)
V2_MIN_US = 0.3  # us: predicted gain below which the grid stays whole-state. Fit and check: 35 dense + 20 page shapes of 65-92 pairs, S = 2..8
# (work_decode_latency/logs/split2_surface*.log): S = 8 -2.0..-0.7 us at 65-84 pairs, +0.1 at 88; S = 6 -0.8..-0.4 at 65-80, 0.0 at 84; S <= 5 +-0.3


def split2(B, H, S, spec=True):
    """Spec grid of more than half the CUs' worth of pairs (so that choose() cannot split it 4- or 8-way) that still fits the CUs when
    split 2-way: 1 = run it as 2 B H blocks of 64 rows. Only where the per-token gain outweighs the fixed cost and the earlier write bound."""
    n = B * H
    if not spec or S < 2 or 4 * n <= _NUM_CU or 2 * n > _NUM_CU:
        return 0
    return int(S * V2_US_PER_TOKEN - V2_FIXED_US - V2_WRITE_US * S * (n - _NUM_CU // 4) >= V2_MIN_US)


H2_TAIL_US = 0.9  # us per token of the second round of a spec grid just above one block per CU: its few blocks run latency-bound (VALU, ~0.9 us
# per token) behind a first round that was write-bound, while the memory system idles
H2_FIXED_US = 3.7  # us: what running the batch as two launches costs: a second graph node (1.7) and a second prologue with nothing to write (~2)
H2_MAX_OVER = 0.22  # share of the CUs by which the pairs may exceed them: above ~312 pairs the second round has enough blocks to keep the memory
# system busy and one grid wins again (halves - one call, us, work_decode_latency/logs/split_batch_1.log: S 8: -3.0..-4.7 at 264-312 pairs, -0.5 at
# 324, +0.5 at 336, +1.6 at 384; S 7: -2.5 / -0.8 / -1.0 at 264 / 288 / 312; S 6: -3.2 / -0.4 / -2.5; S 5: -1.8 at 264; S 4: 0.0)
H2_MIN_US = 0.5


def halves(B, H, S, spec=True):
    """Spec batch whose pairs exceed the CUs by a little (decode_adversary's notch at 257-300 pairs: a handful of blocks pay a whole second
    round): number of sequences of the first of TWO launches over disjoint sequence ranges, each a single-round grid, or 0 (one launch).
    Sequences are independent, so the two launches are the reference's result by construction; a captured graph holds two kernel nodes."""
    n = B * H
    if not spec or B < 2 or n <= _NUM_CU or n > _NUM_CU * (1.0 + H2_MAX_OVER) or ((B + 1) // 2) * H > _NUM_CU:
        return 0
    return (B + 1) // 2 if S * H2_TAIL_US - H2_FIXED_US >= H2_MIN_US else 0


def _num_cu(dev):
    if dev not in _cus:
        _cus[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _cus[dev]


def choose(B, H, S, spec=True):
    """Cost model -> (nsplit, threads per block).

    - V-split (spec only): one block per (n, h) streams (S + 1) x 64 KB through one CU, so when B*H blocks
      leave most CUs idle, split V. A split costs a redundant q/k prologue and a last-arriver epilogue
      (~2 us), so split only while the split grid fits in half the CUs and blocks keep >= 32 rows.
    - threads (no split): 1024 threads for latency-bound grids (<= one block per CU) and for grids that need
      several rounds even at 256 threads (4x the memory-level parallelism per state). In between, 256-thread
      blocks (co-resident) unless blocks/CUs sits mid-way between integers (see below).
    """
    global _NUM_CU
    if _NUM_CU is None:
        _NUM_CU = _num_cu(torch.cuda.current_device())
    ns = 1
    if FORCE_NSPLIT is not None:
        ns = FORCE_NSPLIT
    elif S >= 2:
        # one 64 KB-in / S x 64 KB-out stream per block: split V while the split grid stays within half the CUs
        # (every block alone on its CU, latency bound). 16-row blocks (ns = 8) keep the 512-thread staged prologue and
        # run the token loop on 4 waves (one per SIMD; kda_v5l_h.cu): per-SIMD token work halves, the mailbox
        # exchange cost does not grow with ns. Going from 4 to 8 splits saves ~0.06-0.09 us per token and costs
        # ~0.25 us (twice the blocks, redundant prologues): only for S >= 3 (held-out pairs: S = 2 +0.15 us,
        # S = 3..8 -0.0..-0.64 us; NOTES_lat "E").
        # Each split step has its own fixed cost. Entering the staged kda_vec kernel (ns 1 -> 2: mailbox exchange) and entering
        # the 16-row mode (ns 4 -> 8: eight redundant prologues per head, 4 token waves) need the split grid to fit in half
        # the CUs; the 32-row step (ns 2 -> 4) only halves each block's rows and pays while the split grid fits in all CUs
        # (held-out B*H 36-64: -0.7..-1.4 us; ns 4 -> 8 at a full grid +0.22 us, a first split at B*H 96-128 +0.2..+0.7 us;
        # NOTES_bw2 E42).
        # v72: the S gate of the 16-row step as its two measured terms (160 hab pairs per shape on v70, B*H <= 16: 16-row at S = 2
        # +0.03 us mean of four shapes, 32-row at S = 3 +0.11 mean of four -> 0.14 us per token against 0.31 fixed; S = 8 then
        # predicts -0.81, measured -0.64 on v33)
        while ns < (8 if S * V16_US_PER_TOKEN > V16_FIXED_US else 4) and B * H * ns * 2 * (1 if ns == 2 else 2) <= _NUM_CU:
            ns *= 2
        if ns == 1 and split2(B, H, S, spec):
            ns = 2  # 64 < B*H <= ~84 pairs at S >= 6: the 2-way split (its own code object, kernel_v15_a._ext2())
    nblk = B * H * ns
    if FORCE_NTH is not None:
        nth = FORCE_NTH
    elif ns > 1:
        nth = 512
    else:
        if spec or nblk <= _NUM_CU or nblk > _NUM_CU * blocks_per_cu(S, spec, 1, 256):
            # latency bound (one block per CU), or several rounds anyway; spec: the write-heavy 1024-thread kernel
            # (prologue-first + next-round prefetch in >= 2 rounds) beats a co-resident 256-thread grid (A/B s4/s8_k3_b64
            # -0.7/-1.4 us; heldout-type S=6 grids -2..-3 us; NOTES_opt3 "K1")
            nth = 1024
        else:
            # Co-resident 256-thread grid vs 1024-thread rounds (one block per CU). Spec never gets here. Plain: the
            # co-resident whole-state grid with the read-first hold (hold_ticks()) wins at every fill that fits, also
            # when it is per-CU unbalanced (1.25-1.75 blocks per CU): it reads all states in one phase and needs no
            # V-split exchange (screens: 1.25-1.75-round fills -1.5..-2.5 us vs 1024-thread rounds, d_k3_b32 -1.0 us
            # vs the V-split; NOTES_bw2 P2). (The older rule, 1024 threads at mid fills, predates the hold.)
            nth = 256
    if ns == 8:
        return 8, 512  # staged kernel: 512-thread prologue, token loop on 16 x 16 rows (kda_v5l_h.cu)
    return ns, min(nth, (128 // ns) * 16)  # at least one state row per 16-lane group


def use_vec(B, H, S, ns):
    """Staged prologue (16 B chunks -> LDS, one wave per token) vs per-thread loads.

    The staged prologue replaces ~2 narrow loads per token per wave (2S + ~25 per wave, all waves) by ~(1 KB of
    inputs per token) / (16 B x threads) wide loads, at the price of an extra barrier and LDS round trip (~0.3 us).
    That only pays when the kernel is latency bound (every block alone on its CU) and the per-token loads dominate
    (S >= 2); in throughput-bound grids the extra per-block latency costs more than the saved issue slots.
    """
    if FORCE_VEC is not None:
        return int(FORCE_VEC)
    # Whole-state blocks: the per-thread prologue with prologue-first load order (prologue_first()) and the max-ILP
    # scheduled kda_main token loop is at least as fast for every window length (crossover history: S <= 4 in v25,
    # S <= 6 in v27, all S in v28/v29: held-out S = 7/8 mid-B -0.5..-2.0 us, s8_k3_b16 -0.43 us; NOTES_opt4 "MV8").
    # The staged prologue remains for V-split grids, whose 16-32-row blocks keep the per-token inputs in LDS.
    return int(S >= 2 and B * H * ns <= _NUM_CU and ns > 1)


def prologue_first(B, H, S, spec, ns, nth):
    """kda_main load order. Prologue-first (slot-independent inputs, then slot chain, conv history, state rows) puts a
    block's own prologue ahead of its 64 KB state read; that pays when the prologue is on the block's critical path:
    latency-bound grids (every block alone on its CU) and grids of >= 2 rounds (later blocks start staggered, one per
    CU slot). In grids of one to two rounds all blocks start together and the DRAM queue sets the pace; issuing the
    state rows first gets them queued earliest (measured better there, plain and spec alike)."""
    if FORCE_PF is not None:
        return int(FORCE_PF)
    if ns != 1:
        return 0
    nblk = B * H
    if nblk <= _NUM_CU:
        return 1  # latency bound
    if spec and nth == 1024:
        # spec 1024-thread grids of more than one round: prologue-first too (and next_round() then prefetches the second
        # round's inputs). The state-first rule below was fitted on co-resident 256-thread spec grids (lo9), which no longer
        # exist; held-out 1-2-round spec grids -0.3..-2.5 us (NOTES_bw2 E41)
        return 1
    cap = _NUM_CU * blocks_per_cu(S, spec, ns, nth)
    if not spec and nth == 256 and nblk <= 2 * _NUM_CU:
        # plain co-resident hold grid (hold_ticks()) of <= 2 blocks per CU: each CU reads <= 2 states, so the prologue chain
        # (slot -> history -> math) is the critical path, not the read phase (d_k3_b32 -0.4 us, held-out <= 2 blk/CU -0.1..-0.3; at 3-4 blk/CU state-first stays: d_glm8_b128 +0.15 with PF; NOTES_bw2 P5)
        return 1
    return int(nblk >= 2 * cap)


def hold_ticks(B, H, S, spec, ns, nth):
    """Read-first hold for plain kda_main on a fully co-resident grid of more than one block per CU: all blocks start
    together and issue their 64 KB state reads at once; holding the state stores for ~(state bytes read) / (DRAM
    read rate) lets the reads drain before the writes start. Returns s_memrealtime ticks (100 MHz). The hold is only
    worth it with a 1:1 read/write mix (plain); write-heavy spec grids measured no gain."""
    if FORCE_HOLD is not None:
        return int(FORCE_HOLD)
    if spec:
        return 0
    nblk = B * H * ns
    if nblk <= _NUM_CU or nblk > _NUM_CU * blocks_per_cu(S, spec, ns, nth):
        return 0
    if True:  # (plain grids are never V-split: ns == 1)
        # the read phase lasts at least one block's own read chain under load (HOLD_MIN); above it, the two-term read phase (NOTES_ablate.md, re-fit 1): the busiest CU drains its own co-resident blocks' state reads
        # (HOLD_CU_US per block: 64 KB through that CU's L2 path) while the grid's bytes drain at the read-only fabric rate.
        # The optimum steps with ceil(blocks / CUs) (768 -> 864 blocks: +1.7 us for +12 % bytes): a hold proportional to
        # the bytes alone is short just above an occupancy step and long on full grids.
        return int(max(HOLD_MIN, HOLD_CU_US * -(-nblk // _NUM_CU) + B * H * 128 * 128 * 4 / HOLD_BW2) * 100)


LOAD_ISSUE_US = 25 / 2200.0  # us per cold load instruction issued by a wave (25 shader clocks at 2.2 GHz; ISSUE_PATH.md 1b)
SLOT_HOP_US = 0.40  # us: cold scalar (slot index) hop after the eval's cache flush (NOTES_lat section 1)
SPEC_PRO_US = 2.4  # us: spec prologue chain after the slot hop (history hop + prologue math + barrier), 1024-thread prologue-first kernel


def state_after_prologue(B, H, S, spec, ns, nth):
    """Spec whole-state prologue-first grid of one round: issue the state rows of all waves behind a block barrier, i.e.
    after the prologue waves have issued their inputs and conv history. The prologue loads are order-critical against
    the 64 KB-per-block state burst (NOTES_b1model.md section 6), while the state read has slack as long as the read
    phase (all state bytes at the DRAM read rate HOLD_BW_NT) is shorter than the prologue chain it runs beside. Held-out
    pairs: B*H 128-208 -0.2..-0.5 us (S >= 6; S <= 5 neutral), 216-240 neutral, 248-256 +0.1..+0.3 us without this limit."""
    if not spec or ns != 1 or nth != 1024 or B * H > _NUM_CU:
        return 0
    if not prologue_first(B, H, S, spec, ns, nth) or next_round(B, H, S, spec, ns, nth):
        return 0
    # (ii) the ordering only matters while the prologue waves are still issuing their slot-independent loads when the slot
    # hop returns and releases the other waves' state rows: ~2 S + 25 load instructions per prologue thread at ~25 clk
    # each (cold issue, ISSUE_PATH.md 1b) against the ~0.40 us scalar hop -> S >= 6 (held-out: S >= 6 -0.2..-0.5 us,
    # S = 4, 5 neutral, S = 3 +0.0..+0.26 us with the barrier)
    if (2 * S + 25) * LOAD_ISSUE_US <= SLOT_HOP_US:
        return 0
    return int(B * H * 128 * 128 * 4 / HOLD_BW_NT <= SPEC_PRO_US)


def next_round(B, H, S, spec, ns, nth):
    """Multi-round prologue-first grids (plain and spec): block b prefetches the small inputs of block b + (blocks per round),
    which runs on the same XCD (round-robin dispatch) one round later. 0 = off."""
    if ns != 1 or nth != 1024 or B * H <= _NUM_CU or not prologue_first(B, H, S, spec, ns, nth):
        return 0
    return _NUM_CU * blocks_per_cu(S, spec, ns, nth, 1)  # blocks per round of the multi-round (next-round prefetch) instantiation itself


TN_US_PER_TOKEN = 0.33  # us gained per token step whose state stores a tail block issues nontemporal instead of sc0 sc1 (a lone XCD
# writes 1.25 vs 1.0 TB/s, MEASURE.md 2.2: 2 MB per XCD and token step -> 0.41 us; fitted slope over S = 2..8 on held-out grids: 0.33)
TN_FIXED_US = 1.1  # us: what the nontemporal tail costs whatever S is: XCD 3's lead that the tail has to make up before it gains (NOTES_xcd.md section 3).
# Fitted intercept; 1.4 on v66 (break-even S ~ 4.2), re-fit on v70 where WM removed the slow state of the middle generations: S = 4 grids of > 2 rounds
# -0.22 us mean of six (NOTES_final.md section 4), S = 5 / 6 -0.37..-1.09; S <= 3 stays off (v66 pairs: +0.0..+0.7). kda_*.cu carries the two-copy kernel for
# S >= ST_TNV = 4 only: a tail requested for a shorter window is an "unsupported configuration" error, not a silent no-op.


def tail_nt(B, H, S, spec, ns, nth):
    """Spec multi-round grids: first block (0 = none) of the 'tail' blocks, whose state stores are nontemporal on odd XCDs.

    XCDs 2k and 2k + 1 share a write path on which the even one wins (MEASURE.md section 2), so the odd XCDs fall behind
    and finish the grid alone; alone, an XCD is bound by its own L2 -> fabric path, where write-through (sc1) stores run
    at 1.0 TB/s and nontemporal ones at 1.25, while under a saturated HBM nontemporal stores cost 13-21 % of the write
    phase (MECHANISMS.md C3). So only the blocks that write in the tail switch policy: blocks without a successor on
    their CU (b >= blocks - blocks per round), on odd XCDs (the kernel reads HW_REG_XCC_ID), and only
    - when a full round lies between the first generation and them (blocks > 2 rounds): in grids of <= 2 rounds the
      blocks without a successor run beside a half-idle or still saturated partner (held-out pairs -0.5..+2.9 us, no sign);
    - when the per-token gain outweighs the fixed cost: S * TN_US_PER_TOKEN > TN_FIXED_US, i.e. S >= 5 (held-out pairs,
      50 shapes of 264-2048 blocks on v66: S = 2, 3 +0.0..+0.7 us, S = 4 -0.65..+1.3 us, S >= 5 with > 2 rounds 21 of 22 faster,
      -0.2..-1.7 us; on v70: S = 4 with > 2 rounds five of six faster, -0.03..-0.61 us -> S >= 4)."""
    nxt = next_round(B, H, S, spec, ns, nth)
    if not spec or not nxt or B * H <= 2 * nxt or B * H >= 1 << 31:
        return 0
    if S * TN_US_PER_TOKEN <= TN_FIXED_US:
        return 0
    return B * H - nxt


WS_HBM_W = 6.1e6  # bytes per us: HBM write ceiling of the whole GPU (MECHANISMS.md (c))
WS_TOK_US = 0.9  # us per token step of a block's last wave while the HBM is not saturated (per-block stamps on 72-108-block grids: 0.90-1.0)
WS_US = 2.3  # us: the wave offset itself (TP_WS = 80 x s_sleep(1), kda_*.cu)


def wave_offset(B, H, S, spec, ns, nth):
    """Single-round spec 1024-thread kda_main (prologue-first, no next-round prefetch, no state-row barrier): 1 = waves 8-15 of every
    block enter the token loop WS_US late (the WS offset of the multi-round kernel; it breaks the collective in-phase state of a
    grid whose blocks all start together, NOTES_xcd.md sections 4-6: 8-12 % of the replays of the full S = 6 grids are 5-9 us slow).

    A latency-bound grid pays the offset (held-out pairs: 80-132 blocks at S = 2..4 +0.2..+0.8 us), a write-bound one hides it. So only
    where the grid's state-write phase at the HBM write ceiling outlasts the block's own token loop by at least the offset:
    B * H * S * 64 KB / WS_HBM_W - S * WS_TOK_US >= WS_US (blocks >= 191 / 156 / 138 / 127 / 120 / 115 / 111 for S = 2..8). The state-row-barrier
    grids (state_after_prologue(): S >= 6, <= 212 blocks) are left alone: 1.7 % slow replays, 0.00 us on 4 shapes."""
    if not spec or ns != 1 or nth != 1024 or S < 2 or not prologue_first(B, H, S, spec, ns, nth):
        return 0
    if next_round(B, H, S, spec, ns, nth) or state_after_prologue(B, H, S, spec, ns, nth):
        return 0
    return int(B * H * S * 128 * 128 * 4 / WS_HBM_W - S * WS_TOK_US >= WS_US)


X1_XCC = 1  # physical XCC that starts first in a graph replay on MI350X (UNDERFLOOR.md E5: 88-99 % of the replays on four GPUs, 0.14-0.45 us ahead)
X1_XCC_BW = 1.0e6  # bytes per us through one XCC's memory path (NOTES_struct.md section 2: a lone XCD writes 1.0 TB/s write-through)
X1_SPAN_US = 2.1  # us: memory phase of a latency-bound plain block (two cold hops, 64 KB in, 64 KB out); one XCC must carry the grid's bytes in it
_x1 = {}  # device -> probed: the dispatch is an exact round-robin over 8 XCCs
_X1_OK = False


def _x1_probe(dev):
    """Once per device, outside any capture: launch the probe kernel on grids of the X1 form (and one whose rows are not a multiple
    of 8) and check that workgroup i runs on XCC (first + i) % 8, i.e. that every 8 consecutive workgroups cover the 8 XCCs. A
    capture that comes before any eager call on its device leaves the device unprobed: the plain grid is used (no host work
    inside a capture)."""
    if dev not in _x1:
        if torch.cuda.is_current_stream_capturing():
            return False
        m = _ext(False)
        ok = _num_cu(dev) == 256 and m.x1_xcc() == X1_XCC
        for gx, gy in ((96, 1), (128, 2), (64, 3), (100, 3)) if ok else ():
            buf = torch.full((gx * gy,), -1, dtype=torch.int32, device=f"cuda:{dev}")
            with torch.cuda.device(dev):
                m.xcc_probe(buf, gx, gy)
            x = buf.cpu()
            ok = ok and bool(((x - torch.arange(gx * gy)) % 8 == x[0] % 8).all()) and 0 <= int(x.min()) and int(x.max()) == 7
        _x1[dev] = ok
    return _x1[dev]


def xcc_first(B, H, S, spec, ns, nth):
    """Plain latency-bound whole-state grid (1024 threads, prologue-first, one round): 1 = launch it as 8 H x B blocks of which only the
    ones on the first-started XCC work (X1). The lead is worth ~0.4 us; it is paid for as soon as one XCC's memory path
    (X1_XCC_BW) cannot carry the grid's state bytes (64 KB in + 64 KB out per block) inside the block's own memory phase:
    B H x 128 KB / X1_XCC_BW <= X1_SPAN_US, i.e. <= 16 blocks (pairs vs v73: 8 / 12 / 16 blocks -0.47 / -0.40 / -0.41 us,
    24 blocks -0.07 / -0.14, 32 blocks +0.27, 36-96 blocks +2.9..+6.5; UNDERFLOOR.md E8b)."""
    if spec or S != 1 or ns != 1 or nth != 1024 or not _X1_OK:
        return 0
    if not prologue_first(B, H, S, spec, ns, nth) or next_round(B, H, S, spec, ns, nth) or hold_ticks(B, H, S, spec, ns, nth):
        return 0
    return int(B * H * 2 * 128 * 128 * 4 / X1_XCC_BW <= X1_SPAN_US)


def hold_arg(B, H, S, spec, ns, nth):
    """The launch's multi-purpose argument: read-first hold ticks | state rows behind the barrier (1) | first nontemporal tail
    block | -1 wave offset | -2 X1 (each kernel class reads at most one of them)."""
    return (hold_ticks(B, H, S, spec, ns, nth) or state_after_prologue(B, H, S, spec, ns, nth) or tail_nt(B, H, S, spec, ns, nth)
            or -wave_offset(B, H, S, spec, ns, nth) or -2 * xcc_first(B, H, S, spec, ns, nth))


_BF16, _F32, _I32 = torch.bfloat16, torch.float32, torch.int32


def _validate(B, T, S, H, spec, mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
              state_indices, num_accepted_tokens, out):
    """Host-side input contract (REDTEAM R3; shapes, dtypes and strides only, nothing data dependent, so it costs nothing
    in a graph replay). The HIP kernels take raw pointers: a wrong dtype or an inner stride would be reinterpreted silently.
    Returns False when the HIP path cannot honour the layout but the AITER path can (strided index tensors)."""
    dim = H * 128
    def chk(name, t, dtype, shape, rows_only=False):
        if t.dtype != dtype or t.device != mixed_qkv.device:
            raise TypeError(f"kda decode: {name} must be {dtype} on {mixed_qkv.device}, got {t.dtype} on {t.device}")
        if shape is not None and tuple(t.shape) != tuple(shape):
            raise ValueError(f"kda decode: {name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
        if not (t[0].is_contiguous() if rows_only and t.shape[0] else t.is_contiguous()):
            raise ValueError(f"kda decode: {name} must be contiguous" + (" within a row / slot" if rows_only else ""))
    if B < 1 or T != B * S or S < 1:
        raise ValueError(f"kda decode: T = {T} is not B x S for B = {B}")
    taps = (2 + S) if spec else 3
    chk("mixed_qkv", mixed_qkv, _BF16, (T, 3 * dim), True)  # the row / slot stride of x, conv_state and ssm_state is honoured
    if conv_state.dim() != 3 or conv_state.shape[1] < taps or conv_state.shape[2] != 3 * dim:
        raise ValueError(f"kda decode: conv_state must be [slots, >= {taps}, {3 * dim}], got {tuple(conv_state.shape)}")
    chk("conv_state", conv_state, _BF16, None, True)
    chk("conv_weight", conv_weight, _F32, (3, 4, dim))
    chk("gate", gate, _BF16, (T, H, 128))
    chk("beta", beta, _BF16, (T, H), True)
    chk("out_gate", out_gate, _BF16, (T, dim), True)
    chk("A_log", A_log, _F32, (H,))
    chk("dt_bias", dt_bias, _F32, (dim,))
    chk("norm_weight", norm_weight, _F32, (128,))
    if ssm_state.dim() != 4 or tuple(ssm_state.shape[1:]) != (H, 128, 128):
        raise ValueError(f"kda decode: ssm_state must be [slots, {H}, 128, 128], got {tuple(ssm_state.shape)}")
    chk("ssm_state", ssm_state, _F32, None, True)
    chk("out", out, _BF16, (T, dim))
    for name, t in (("state_indices", state_indices), ("num_accepted_tokens", num_accepted_tokens)):
        if t is not None and t.device != mixed_qkv.device:  # the kernel takes raw pointers (decode_adversary: a CPU index tensor faulted the device)
            raise TypeError(f"kda decode: {name} must be on {mixed_qkv.device}, got {t.device}")
    if state_indices.dtype != _I32 or tuple(state_indices.shape) != ((B, S) if spec else (B,)):
        raise ValueError(f"kda decode: state_indices must be int32 {(B, S) if spec else (B,)}, got {state_indices.dtype} {tuple(state_indices.shape)}")
    if spec and (num_accepted_tokens.dtype != _I32 or tuple(num_accepted_tokens.shape) != (B,)):
        raise ValueError(f"kda decode: num_accepted_tokens must be int32 ({B},), got {num_accepted_tokens.dtype} {tuple(num_accepted_tokens.shape)}")
    return state_indices.is_contiguous() and (not spec or num_accepted_tokens.is_contiguous())


def run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
        state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    global _NUM_CU, _X1_OK
    B = cu_seqlens.shape[0] - 1
    T = mixed_qkv.shape[0]
    S = T // max(B, 1)
    H = A_log.numel()
    spec = num_accepted_tokens is not None
    if S > _SMAX or (not spec and S != 1) or not _validate(B, T, S, H, spec, mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias,
                                                         norm_weight, ssm_state, state_indices, num_accepted_tokens, out):
        import kernel_aiter_fallback as fb
        return fb.run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight,
                      ssm_state, state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out)
    _NUM_CU = _num_cu(mixed_qkv.device.index)  # the cost model below is evaluated for this call's device (REDTEAM R7.5)
    _X1_OK = not spec and _x1_probe(mixed_qkv.device.index)
    ns, nth = choose(B, H, S, spec)
    vec = use_vec(B, H, S, ns)  # staged prologue; it always loads the state after the staging barrier
    _ext(spec).run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
               state_indices, num_accepted_tokens, out, _workspace(mixed_qkv.device, B * H, vec), float(norm_eps),
               float(lower_bound), ns, nth, vec,
               prologue_first(B, H, S, spec, ns, nth), hold_arg(B, H, S, spec, ns, nth),
               next_round(B, H, S, spec, ns, nth))
