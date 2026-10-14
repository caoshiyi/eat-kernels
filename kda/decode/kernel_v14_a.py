"""v14_a = v74 + the per-XCC model of ASYMMETRY.md as one table and three rules (event = F + max over XCC x of (S_x + W_x); XCC_START_US = S_x, XCC_WRITE_EXCESS_US / XCC_WRITE_RATIO = how much longer XCC x writes): (H) hold_skew(): in the plain state-first hold kernel the XCCs whose write phase is >= HS_SKEW_US longer than the even XCCs' release their stores that much earlier (own instantiation kda_main<.., XV = 1>, kda_v14_a.cu HS); (U) xcc_items(): in a single-round spec whole-state grid with spare CUs the XCC with the slowest write path skips its last K = round((1 - UQ_FRAC) N / 8) items and blocks behind the grid on the two XCCs that start first take them (own instantiations XV = 2 of the PF / W1 / SB kernels; needs the exact round-robin); (X) xcc_fill(): X1 generalised, a latency-bound plain grid runs on the m XCCs with the smallest start offsets, m from the same per-XCC bandwidth bound (X1 is m = 1; XV = mask). Guard: every per-XCC rule needs _xcc_tab() (8 XCCs by the HIP device attribute, 256 CUs, the compiled constants equal to the tables), (U) and (X) also the round-robin probe, which is repeated on every stream an eager call uses and is sticky when it fails. Every kernel v74 can launch is instruction-identical. v13_a (v74) = v73 + X1: the plain latency-bound whole-state kernel (1024 threads, prologue-first) runs on the one XCC that starts first (found by HW_REG_XCC_ID, as TN does). The dispatch hands the workgroups of a grid to the 8 XCCs round-robin, and physical XCC 1 starts 0.14-0.45 us before the other seven on every MI350X measured (UNDERFLOOR.md E5 / E8b); a grid of 8 H x B blocks in which only the blocks on XCC 1 work, on head blockIdx.x / 8 (the others return at entry) therefore ends ~0.4 us earlier while one XCC's memory path carries the grid's bytes inside a block's own span (cost model xcc_first(): B H x 128 KB / X1_XCC_BW <= X1_SPAN_US, i.e. <= 16 blocks; measured: -0.4 us to 16 blocks, neutral at 24, +0.3 at 32). One working block per head needs the round-robin to be exact (8 consecutive workgroups on 8 different XCCs; its first XCC depends on the HSA queue, so blockIdx alone cannot choose the block): that is checked once per device by probe launches outside any capture (_x1_probe(): 3 grids x 4 dispatches, eager), and until then, or on a device where it does not hold, the grid is v73's. Own instantiation kda_main<..., X1 = true> (kda_v13_a.cu): every other kernel is instruction-identical to v73. Also: KDA_SELF of the .cu points at the file itself (v70-v73 compiled the token loop of kda_v12x_w1.cu). v12f_d (v73) = v72 with the mailbox reset of v70 (R2) issued right behind the staging barrier in the 16-row kda_vec kernel for S >= 4 (kda_v12f_d.cu MB_CLR = 5, MB_LATE_S = 4) and indexed without a division everywhere: before the staging barrier the store costs the latency-bound 16-row grids 0.06-0.17 us (hab 120 pairs); 32-row kernels and S <= 3 keep the early reset. Fault injection (scratch_final/poison_rate.py, every mailbox word stale before a call): NOTES_final.md section 10. Only the 14 kda_vec kernels change. v12f_c (v72) = v71 + the constants that did not hold on held-out shapes (NOTES_final.md section 4; hab pairs on the v70 base): (1) TN (nontemporal odd-XCD tail of spec multi-round grids of > 2 rounds) from S = 4: TN_FIXED_US re-fit 1.4 -> 1.1 (the tail\'s fixed cost fell when WM removed the middle generations\' slow state; six S = 4 grids of > 2 rounds: -0.03..-0.61 us, four beyond CI, one +0.08 +- 0.09); the S = 4 two-copy kernel is its own instantiation (kda_v12f_c.cu ST_TNV), so S = 4 grids of <= 2 rounds keep their v71 kernel; (2) CG_ML also in the state-row-barrier (SB) kernel: the exception was neutral on eight held-out SB shapes (-0.06..+0.08 us, mean -0.01), one rule less; (3) the 16-row V-split step as a cost model, S x V16_US_PER_TOKEN > V16_FIXED_US (160 pairs each: 16-row at S = 2 +0.01..+0.06 us on four shapes, 32-row at S = 3 +0.09..+0.13: same dispatch as the literal S >= 3). v12f_b (v71) = v70 without the dead code (REDTEAM.md section 5, ABLATIONS.md 2.2; kda_v12f_b.cu: the launcher references only the instantiations the cost model can ask for, 43 kernels in the two builds instead of 574; every one of them is instruction-identical to v70 and the dispatch is identical on 109 854 shapes; host: the unused arrival-counter / o-exchange workspaces, HOLD_BW, FORCE_LATE and the V-split arm of hold_ticks() are gone). v12f_a (v70) = v69 + the red-team fixes (REDTEAM.md, NOTES_final.md; no dispatch change, kda_vec gains one store): (R1) V-split mailboxes are no longer one buffer per device: every captured call owns its mailbox (taken from spares that the eager calls keep filled: nothing extra is replayed) and eager calls use one per (device, stream), so launches of different graphs / streams that overlap on the device cannot read each other's partial sums or spin on each other's resets; (R2) kda_vec blocks return the mailbox words they are going to publish to the sentinel at entry (kda_v12f_a.cu switch MB_CLR), so a stale word (aborted launch) is healed by the next call instead of poisoning every later one, and reset_workspace() refills everything; (R3) run() validates dtype / shape / contiguity of every tensor (strided state_indices / num_accepted_tokens go to the AITER path, which honours strides; anything else raises) and the pool warm-up bound is the smaller of the two pools; (R6) the ROCm version is part of the build digest and a build on a toolchain other than the audited one warns; (R7) a failed register query raises instead of silently changing the dispatch, the occupancy of spec multi-round grids is queried from the instantiation that is launched, the CU count is per device, experiment environment variables are ignored unless KDA_EXPERIMENT=1, and the unsupported 16-row / vec = 0 launch is an error. v12x_w1 (v69) = v68 + W1: the WS wave offset also in the single-round spec kda_main (1024 threads, prologue-first, no next-round prefetch, no state-row barrier) on grids whose state-write phase at the HBM write ceiling outlasts the block\'s own token loop by at least the offset (cost model wave_offset(): blocks x S x 64 KB / WS_HBM_W - S x WS_TOK_US >= WS_US, i.e. blocks >= 191 / 156 / 138 / 127 / 120 / 115 / 111 for S = 2..8; requested with p.hold = -1 and launched as its own instantiation kda_main<..., W1 = true>, kda_v12x_w1.cu switch TP_WS1; every other instantiation is byte-identical to v68). Latency-bound grids below it pay the offset (80-132 blocks at S = 2..4: +0.2..+0.8 us). Mechanism as WM (NOTES_xcd.md section 6). v12x_wm (v68) = v67 + WM: in the spec multi-round kda_main the WS wave offset (waves 8-15 enter the token loop 80 x s_sleep(1) late) is applied in every generation whose blocks have a successor on their CU (b < blocks - next_round()), not only in the first one, for S >= 3 (kda_v12x_wm.cu switch TP_WSM). Mechanism (NOTES_xcd.md): the offset breaks a GPU-wide in-phase state of the token steps (without any offset 12 % of the processes at S = 4 sit in a mode 6-7 us slower; an offset on any two XCDs removes it); the blocks of an XCD\'s generation end within < 1 us, so each later generation re-forms it (per-replay stamps: slow replays lose 7 us GPU-wide in the token loop of generation 1); tail blocks (no successor) are excluded, they pay a delay 1 : 1. Grids of <= 2 rounds have no middle generation and run v67 instructions\' paths. The dispatch is unchanged. v11a_ab (v67) = v66 after the first global leave-one-out (NOTES_ablate.md): (1) read-first hold re-fit as a two-term cost model, hold = HOLD_CU_US x ceil(blocks / CUs) + state bytes / HOLD_BW2 (the busiest CU's chain of co-resident state reads + the grid's bytes at the read rate; the v16-v66 form, bytes / 5.8e6, was a compromise between full grids and fractional fills: 36 held-out fills, 16 faster beyond CI by 0.4-1.3 us, one slower by 0.13 us); (2) features that no longer pay removed / unified, each neutral on the eval (160 pairs) and on held-out pairs: AB_NOPIN=2, PR_VNOA=2, AB_NORR2V=1, SNT=2, AB_PLD=12, PR_V16VCE=0, SPOL_PLAIN=17 (kda_vec fake-use pins, kda_vec exchange-wave pool warm-up, interleaved kda_vec row reductions, early kda_vec v-tap commit, per-class small-store / prologue-load / plain state-store policies now uniform). kda_v11a_ab.cu = kda_v11s_tn.cu + the leave-one-out switches with those defaults. v11s_tn (v66) = v65 + TN: in the spec multi-round kda_main (1024 threads, prologue-first, next-round prefetch) the blocks that write in the odd-XCD tail of the grid issue their state stores nontemporal (aux 2) instead of write-through (sc0 sc1): blocks without a successor on their CU, on odd XCDs (HW_REG_XCC_ID), in grids of more than two rounds with S >= 5 (cost model tail_nt(); first tail block passed in p.hold, which this kernel did not use; the S = 5..8 instantiations change, kda_v11s_tn.cu switch ST_TNS). The token loop text is a section at the top of kda_v11s_tn.cu that kda_main includes once per store policy; the multi-round instantiation carries two copies behind one block-uniform branch in front of the loop (a branch inside the token step is a scheduling barrier: +3.4..+6.6 us; every other instantiation includes the loop once and is instruction-identical to v65). Mechanism (NOTES_struct.md): XCDs 2k / 2k+1 share a write path on which the even one wins, so the odd XCDs finish the grid alone; a lone XCD writes 1.25 TB/s nontemporal vs 1.0 TB/s write-through, while under a saturated HBM nontemporal stores lose 13-21 % (so not on every block: odd XCDs in every generation +1.2..+3.6 us). A/B v65 -> v66: see NOTES_struct.md. The dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged. v10t_ws (v65) = v64 + WS: in the spec multi-round kda_main (1024 threads, prologue-first, next-round prefetch) the upper half of a block's waves (8-15) enters the token loop 80 x s_sleep(1) (~2.3 us) after the lower half, once per block, in the first generation only (b < next_round(): the blocks that all start together; later blocks start staggered, and a last-generation block that runs alone would pay the offset 1:1) (kda_v10t_ws.cu switch TP_WS = 80; NOTES_tpmodel.md). Delay injection: after the norm barrier all 16 waves of a block run the token steps in phase; a one-time offset of half of the waves costs nothing on any spec multi-round grid (waves 8-15 delayed 0.3 / 0.6 / 1.2 us at token 0: s4_k3_b64 -0.28 / -0.35 / -0.39 us on the harness build). The dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged; every other kernel runs v64 instructions. v10c_a (v64) = v63 + instruction-stream work on the critical phases (NOTES_codegen.md; kda_v10c_a.cu switches CG_FH = CG_SL = CG_TP = 1, CG_ML = 4): (FH) V-split kda_vec staging loads from per-wave compile-time run tables: per lane (tensor base select) + (32-bit run offset select) + lane * 16 with the bases / offsets on the SALU, instead of six 64-bit per-lane segment addresses and five selects (~100 VALU in front of every first-hop load); conv history as one buffer load whose lane offsets are computed ahead of the slot wait; beta as a buffer load. (SL) kda_vec slot-chain s_loads issued together with the Params kernarg read behind one wait. (TP) kda_vec token prologue 161 -> 137 non-scalar instructions: packed SiLU / decay math with folded scalars, the three wave sums as one DPP asm block. (ML) spec 1024-thread prologue-first single-round kda_main (mid-B) without the state-row barrier, i.e. where the prologue loads race the block's state rows: every prologue load as (wave-uniform base on the SALU) + (32-bit lane offset), one instruction per load. All four are per-instantiation compile-time forms; the dispatch (choose() / use_vec() / prologue_first() / hold_ticks() / state_after_prologue() / next_round()) is unchanged. A/B v63 -> the same stack with ML on every single-round spec kda_main, full-17: 160 pairs +0.0045 [+0.0037, +0.0053], 80 pairs +0.0041 [+0.0028, +0.0054] (s4_k3_b1 -0.33 / -0.35 us, s6_glm16_b16 -0.15 / -0.08, s6_glm8_b32 -0.11 / -0.10; s8_k3_b16, the barrier instantiation, -0.07 +- 0.08 / +0.17 +- 0.12 and its held-out shapes 0.00..+0.08 -> ML left out there). KDA_CG (env): extra -D switches for codegen experiments, empty by default. v9p_b (v63) = v62 after the leave-one-out of the probe stack (NOTES_probes2.md; kda_v10c_a.cu = scratch_probes2_run/kda_pr16.cu with PR_NOA1 = 3, PR_WUB = 64, PR_MRWU0 = PR_NX32 = PR_V16VALL = PR_ACLAMP = PR_V16VCE = PR_STAGPOS = 1): + plain multi-round first-generation stagger wait issued after the block's inputs and conv history, right before its state-row loads (was at block entry: the small loads of a delayed block no longer wait out the stagger; STAG re-swept at the new position, 150 ticks stays); - the passenger of v62 (host-side NX target without the runtime division: leave-one-out 0.00 on all five multi-round workloads). v9p_a (v62) = v61 + probe stack S3 (NOTES_probes2.md; kda_v9p_a.cu = scratch_probes2_run/kda_pr15.cu with PR_NOA1 = 3, PR_WUB = 64, PR_MRWU0 = PR_NXND = PR_NX32 = PR_V16VALL = PR_ACLAMP = PR_V16VCE = 1): kda_main without the out-page (a1) warm-up load and with the pool translation warm-up (a0, 64 loads on wave 4, ahead of its state rows in vmcnt order) only in the first 64 blocks of the grid (8 per XCD; later blocks find the translations warm); plain multi-round: a0 ahead of the first-generation stagger; multi-round next-round prefetch target (n, h) from host-side nxt / H, nxt % H (no runtime division) and, in spec, scalar bases + 32-bit lane offsets; kda_vec: v conv written by all lanes (no exec branch), accepted-count clamp on the SALU, V16 v-tap commit after the prologue. A/B v61 -> S3 full-17 160 pairs +0.0035 [+0.0027, +0.0043], every workload faster. v9m_sc (v61) = v60 with the barrier variant as its own kda_main instantiation (the v60 runtime branch in front of the state-row loads cost the grids that do not take it +0.06..+0.18 us). v9m_sb (v60) = v59 + spec whole-state prologue-first single-round grids (mid-B): the state rows of all waves are issued behind a block barrier (after the prologue waves issued their inputs and conv history) while the state read phase is shorter than the prologue chain (state_after_prologue(): B*H*64 KB / HOLD_BW_NT <= SPEC_PRO_US). From the delay-injection model of the mid-B kernel (NOTES_b1model.md section 6). v9m_hf (v59) = v58 + V-split kda_vec staging: a wave loads, waits for and stages a 16 B chunk batch (second first-hop batch, conv-history batch) only if it owns a real chunk of it (wave-uniform branch); the other waves used to run the address code, load and LDS write for a clamped duplicate chunk. Delay injection (NOTES_b1model.md) shows every instruction a wave issues before the staging barrier costs event time when the SIMDs carry two prologue waves each. A/B v58 -> v9m_hf: see NOTES_b1model.md. v9m_nb (v58) = v57 + 16-row V-split kda_vec with more prologue tokens than token waves (S > 4): the block barrier between the prologue math and the token loop is replaced by LDS ready flags. Prologue token t runs on wave t; waves 4-7 share a SIMD with the token waves 0-3, issue is age-ordered, so the token waves finish their own tokens ~0.2 us before the younger waves and used to idle at the barrier (delay-injection knees, NOTES_b1model.md: token waves 0.44 us, younger waves 0.66 us after the staging barrier). Now each prologue wave announces its token with one LDS word after its operand writes; the token waves start the loop when tokens 0-3 are announced and wait for tokens 4..S-1 once, before the lookahead first reads token 4. S <= 4 keeps the barrier (no younger prologue wave; a poll costs more than the barrier: +0.08 us). A/B v57 -> v9m_nb, 80 pairs full 17: +0.0020 [+0.0008, +0.0032] (s8_k3_b1 -0.06 +- 0.03, s6_glm16_b1 -0.11 +- 0.03 us, everything else unchanged). v8p_a (v57) = v56 + plain multi-round kda_main: conv commit after the token loop with the taps kept in VGPRs (72-77 VGPRs, so the VGPR-based occupancy model blocks_per_cu() runs the multi-round grid at 1 block per CU, nxt 256) and no block barrier before the state-row loads (SBAR 0: at 1 block per CU the barrier only delays the 64 KB read); plain kda_main s_o written by all 16 lanes (no exec branch) and thread-contiguous s_v rows (as in spec); epilogue rsq as raw v_rsq_f32 (ss/D + eps >= 1e-5); spec late conv commit only for S >= 7 (PROBES P60b, M1a, P46, P56, P30, P20; NOTES_probes.md; kda_v8p_a.cu = scratch_probes_run/kda_pr2.cu with PR_WUW = PR_GRID2D = PR_MRLATE = PR_SOALL = PR_SVC = PR_RSQ = 1, SBAR = 0, LCS = 7). v8i_g2 (v56) = v55 + 2-D launch grid (x = H * NSPLIT, y = B): (n, h, split) from blockIdx without a runtime 32-bit division ahead of every block's first address (PROBES P47; kda_v8i_g2.cu = the probe harness kda_pr.cu with PR_WUW = PR_GRID2D = 1, every other switch 0). v8i_wu (v55) = v54 + the kda_main translation / Params warm-up loads issued by wave 4 (the first non-prologue wave) instead of wave NW - 1, which is dispatched last (PROBES P42; ISSUE_PATH.md). A/B v54 -> v8i_wu 80 pairs +0.0029 [+0.0020, +0.0038]: s4_k3_b64 -0.79, s6_glm16_b64 -0.52, s4_k3_b16 -0.11 us. v7o_v54 (v54) = v7o_bl (v53) with the plain multi-round first-generation stagger recalibrated for the shorter v53 blocks: levels 0 / 0.75 / 1.5 us (STAG 340 -> 150 ticks, 3 levels; v53 blocks are ~3 % shorter and the 1.7 us level spacing fitted on v51 now over-delays). A/B v53 -> v54 (160 pairs) +0.0008 [+0.0001, +0.0016]: d_k3_b128 -0.36, d_glm16_b256 -0.32 us; STAG sweep (40 pairs each) 0: +0.09/+0.81, 150: -0.45/-0.34, 200: -0.33/-0.44, 250: -0.24/-0.49, 450: +0.32/+0.48 us. v7m_mr4 (v52) + every kda_main / kda_vec state-row load as a raw buffer load (buffer_load_dwordx4, SGPR descriptor of the block's own 64 KB slice + 32-bit lane offset, aux 2 = nt) instead of a 64-bit-address global_load_dwordx4 nt (CONVENTIONS rank 3 / U2: the floor harnesses always used buffer loads). A/B v52 -> v53 +0.0069 [+0.0058, +0.0081] (80 pairs): d_glm16_b256 -3.59, d_k3_b128 -1.06 us; aux 3 / 18 = aux 2 (n.s.). v6b_spl3 (v51) + plain multi-round kda_main at 2 blocks per CU: state stores batched after the row math (64 VGPRs; occupancy and next-round prefetch distance from the compiled multi-round instantiation itself) and a one-time desynchronisation of the first co-resident generation: blocks of sequence n with n * H < CUs * blocks per CU wait 0 / 1.7 / 3.4 us by a hash of n (MECHANISMS.md (a): identical co-resident blocks that start together stay phase-locked; keyed per sequence so its H heads stay together). v6b_spl3 = v6b_s1r + spec V-split ns 2 -> 4 step allowed while the split grid fits in all CUs (other split steps unchanged). v6b_s1r = v6b_k0 + spec 1024-thread grids of 1-2 rounds prologue-first with next-round prefetch (cost-model fix; no eval workload in this class). v6b_k0 = v6b_sb + the spec multi-round next-round prefetch issued after the first token (was at block start). v6b_sb = v6b_hb + a block barrier before the state-row loads in the plain multi-round kda_main (the prologue waves issue inputs + conv history first). v6b_hb = v6b_e2 + cached beta load in the plain hold kernel too (beta: a line shared by the H heads; only the spec multi-round kernel keeps nt beta). v6b_e2 = v6b_u + cached (MALL-allocating) beta load in the spec kda_main kernels without next-round prefetch (the beta line is shared by the H heads of a token). v6b_u = v6b_n + nontemporal LDS-DMA next-round prefetch (NX) loads in the plain and spec multi-round kda_main. v6b_n = v6b_q + nontemporal small stores (conv commit, out) in the spec multi-round kda_main. v6b_q = v6b_a + prologue-first load order (<= 2 blocks per CU) (nt prologue loads, default small stores) in the plain co-resident hold kernel. v6b_a = v41 + plain co-resident grids always on the whole-state 256-thread hold kernel (no plain V-split, no 1024-thread 1-2-round grids) + hold floor HOLD_MIN. Fused KDA decode / MTP verify for MI350X: HIP kernels in kda_v6l_pq.cu (JIT-built per scheduler variant, cached). v6l_pq (v41) = v40 + packed (2-wide) h.q dot in the spec kda_main token loop (as h.k already was). v6l_s2 (v40) = v39 + spec kda_main per-token norm scalars (rq * scale, rk, q.k, beta) computed once per block into LDS behind one extra barrier instead of in every wave for every token (TRANSFER S2). v6l_s4 (v39) = v38 (v5b_m) + the V-split kda_vec kernels from the max-ILP build with a sched_barrier at each token end (TRANSFER S4: the earlier "max-ILP slows kda_vec" result was cross-token hoisting to 168 VGPRs, not the scheduler; now 68-74 VGPRs). v5b_m = v37 (v5b_h) + system-scope small bf16 stores (conv commit, out) in the whole-state prologue-first plain kernels; v5b_h = v36 + sc1 nt state stores in the plain multi-round kernels; v5b_f = v35 (v5b_e: nontemporal per-(n, h) prologue loads in the plain whole-state state-first co-resident hold kernels) + the same in every spec kda_main kernel (read-first hold recalibrated). v34 = v33 (mailbox exchange; 16-row V-split for S >= 3) + kda_vec prologue with a compile-time token per wave, and in the 16-row mode a one-token LDS operand lookahead with interleaved k/q row reductions and DPP row-broadcast wave sums."""
import hashlib
import os
import warnings

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kda_v14_a.cu")
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
        bdir = os.path.join(_HERE, "build", "v14a_" + digest + sfx)
        os.makedirs(bdir, exist_ok=True)
        arch0 = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        xf = ["-mllvm", "-amdgpu-sched-strategy=max-ilp", "-DKDA_SPEC_ONLY"] if ilp else []
        try:
            _mods[ilp] = load(name=f"kda_v14a_{digest}" + sfx, sources=[_SRC], extra_cflags=["-O3"],
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
_NSPARE = 2


def _mailbox(dev, n):
    """V-split partial-sum mailboxes for n = B * H (n, h) pairs, all words at the sentinel between calls (the kernel returns
    them there). Two launches that are in flight together must not share one (REDTEAM R1: they would consume each other's
    partial sums or spin on each other's resets), so
    - a call under HIP-graph capture gets its own mailbox (128 KB: V-split needs B * H <= 64). HIP permits no eager work
      while a stream captures, so it is taken from a few spares that every eager call keeps filled: the replay is the same
      single kernel launch. Only a capture with no eager call before it (or of a larger grid) allocates inside the capture;
      its sentinel fill is then part of the graph (correct, one small fill slower);
    - eager calls share one per (device, stream), grown on demand (outgrown ones are kept alive).
    Replaying one graph concurrently with itself stays undefined, as for any kernel that writes `out`."""
    spare = _spare.setdefault(dev, [])
    if torch.cuda.is_current_stream_capturing():
        if spare and n * _MB_WORDS <= spare[-1].numel():
            mb = spare.pop()
        else:
            mb = torch.full((n * _MB_WORDS,), _SENT, dtype=torch.int32, device=dev)
        _captured.append(mb)
        return mb
    while len(spare) < _NSPARE:
        spare.append(torch.full((64 * _MB_WORDS,), _SENT, dtype=torch.int32, device=dev))
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    mb = _mbox.get(key)
    if mb is None or mb.numel() < n * _MB_WORDS:
        if mb is not None:
            _retired.append(mb)
        mb = _mbox[key] = torch.full((max(n, 64) * _MB_WORDS,), _SENT, dtype=torch.int32, device=dev)
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
    for mb in list(_mbox.values()) + _captured + [t for t in _retired if torch.is_tensor(t)] + [t for l in _spare.values() for t in l]:
        mb.fill_(_SENT)
    torch.cuda.synchronize()


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


# ---- per-XCC model (v14; ASYMMETRY.md): event = F + max over physical XCC x of (S_x + W_x). The tables are properties of MI350X
# (8 XCCs, 256 CUs); they are only used on a device that _xcc_tab() accepts.
XCC_START_US = (0.45, 0.0, 0.62, 0.45, 0.90, 0.62, 0.80, 0.57)  # S_x: start offset of XCC x through a ~700-instruction kernel, relative to
# XCC 1 (ASYMMETRY A.1: X1 launched on each target XCC, 18 fresh-process pairs x 3 shapes on HIP 0 / 2 / 3, +- 0.05 us; same order on HIP 6)
XCC_WRITE_EXCESS_US = (0.0, 0.4, 0.6, 1.3, 0.0, 1.3, 0.0, 0.6)  # W_x - W_even on a state-first plain hold grid: how much longer the
# store phase of XCC x lasts than that of XCCs 0 / 4 / 6 (ASYMMETRY A.4 / section 6: stamped token-loop durations on 16,64,1; the order
# is confirmed unstamped by delay and drop maps on four GPUs)
XCC_WRITE_RATIO = (1.0, 1.17, 1.18, 1.33, 1.0, 1.17, 1.0, 1.17)  # token-loop duration of XCC x relative to XCCs 0 / 4 / 6 on single-round
# spec grids (ASYMMETRY A.4: stamped, two HSA queues, 12,16,4 / 16,16,6 / 12,16,8: odd 1.12-1.22, XCC 2 1.12-1.25, XCC 3 1.22-1.44)
X1_XCC = min(range(8), key=XCC_START_US.__getitem__)  # = 1: the XCC that starts first (UNDERFLOOR.md E5: 88-99 % of the replays on four GPUs)
X1_XCC_BW = 1.0e6  # bytes per us through one XCC's memory path (NOTES_struct.md section 2: a lone XCD writes 1.0 TB/s write-through)
X1_SPAN_US = 2.1  # us: memory phase of a latency-bound plain block (two cold hops, 64 KB in, 64 KB out); one XCC must carry the grid's bytes in it
XF_MAX = 4  # xcc_fill(): most XCCs a latency-bound plain grid is placed on (1 = X1 only, v74). Measured (NOTES_v14 item X): m = 2 / 3 / 4
# -0.11..-0.29 us on twelve shapes of 17-64 blocks; m = 5 / 6 (the next XCCs start 0.62 us late, 0.28 us before the last) -0.16..+0.06 on eight
HS_SKEW_US = 0.25  # hold_skew(): how much earlier the slow-writing XCCs release their stores (0 = off, v74). Dose response on 15 state-first
# grids of 768-1024 blocks (NOTES_v14 item H): 0.25 us -0.26..-0.01 on every one, 0.5 us -0.21..+0.47, 0.75 us -0.06..+0.80
HS_EXCESS_MIN_US = 0.5  # hold_skew(): the XCCs whose write excess is at least this get the skew (XCCs 2, 3, 5, 7: the four with < 0.7 us of slack in
# ASYMMETRY A.2; the complementary mask reads +0.03..+0.51 us, the skew on all eight +0.2..+0.35 on 768-832-block grids)
UQ_FRAC = 0.92  # xcc_items(): share of its rows the slowest-writing XCC keeps (1 = off, v74): 22 of 24 (ASYMMETRY B.6: 23 gives half, 21 no more)
_x1 = {}  # device -> probed: the dispatch is an exact round-robin over 8 XCCs (False is sticky)
_x1_seen = set()  # (device, stream) pairs an eager call has probed
_tab = {}  # device -> the per-XCC tables apply (8 XCCs, 256 CUs, compiled constants as the tables)
_X1_OK = False  # this call's device: tables apply and the round-robin was probed exact (xcc_fill(), xcc_items())
_XCC_TAB = False  # this call's device: tables apply (hold_skew(): timing only, correct under any workgroup -> XCC mapping)


def _xcc_order():
    """XCCs by start offset (ties by id): 1, 0, 3, 7, 2, 5, 6, 4."""
    return sorted(range(8), key=lambda x: (XCC_START_US[x], x))


def _xf_mask(m):
    return sum(1 << x for x in _xcc_order()[:m])


def _uq_consts():
    """(slow XCC, receiver mask) of xcc_items(): the XCC with the largest write ratio, and the two XCCs that start first outside its
    write-path pair {2k, 2k + 1} (MEASURE.md 2.2)."""
    slow = max(range(8), key=XCC_WRITE_RATIO.__getitem__)
    rcv = [x for x in _xcc_order() if x >> 1 != slow >> 1][:2]
    assert rcv == sorted(rcv, reverse=True), "the kernel hands the moved items to the receivers in descending id order"
    return slow, sum(1 << x for x in rcv)


def _xcc_tab(dev):
    """The per-XCC tables are MI350X's: 8 XCCs as the HIP runtime reports them (hipDeviceAttributeNumberOfXccs; a CU count alone would
    also pass on another part, HW_RESEARCH.md card B), 256 CUs, and a build whose compiled XCC constants are the tables'."""
    if dev not in _tab:
        m = _ext(False)
        with torch.cuda.device(dev):
            c = list(m.xcc_consts())
            _tab[dev] = (m.num_xcc() == 8 and _num_cu(dev) == 256 and c[0] == X1_XCC and c[-2:] == list(_uq_consts())
                         and all(_xf_mask(i + 2) == v for i, v in enumerate(c[1:-2])))
    return _tab[dev]


def _rr_exact(x, n_xcc=8):
    """x[i] = XCC that workgroup i of one dispatch ran on (linear index). True iff the dispatch is an exact round-robin: workgroup i on
    XCC (x[0] + i) % n_xcc, every XCC id in range. Pure function of the probe result (tested on synthetic violations: chunked
    dispatch, fewer XCCs, permutations, a single misplaced workgroup; scratch_v14/xcc_tests.py)."""
    x = [int(v) for v in x]
    return len(x) >= n_xcc and all(0 <= v < n_xcc for v in x) and all((v - i) % n_xcc == x[0] % n_xcc for i, v in enumerate(x))


_PROBE_GRIDS = ((96, 1), (128, 2), (64, 3), (100, 3), (12, 17), (16, 9))  # X1 / XF forms (8 H' x B), a 1-D grid, UQ forms (H x (B + extra))
_probe_hook = None  # test hook: maps a probe result to what the checker sees (fault injection, scratch_v14/xcc_tests.py)


def _x1_probe(dev):
    """Outside any capture: launch the probe kernel on grids of the launch forms that depend on the workgroup -> XCC mapping and check
    that every dispatch is an exact round-robin over the 8 XCCs (_rr_exact()). Done on the first eager call of every (device, stream):
    the property belongs to the HSA queue type, and a stream is the only handle on a queue a caller has (HW_RESEARCH.md card B). A
    failure is sticky for the device and warns: graphs captured before it must be re-captured. A capture that comes before any eager
    call on its device leaves the device unprobed: the symmetric grid is used (no host work inside a capture)."""
    if torch.cuda.is_current_stream_capturing():
        return _x1.get(dev, False)
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    if key in _x1_seen or _x1.get(dev) is False:
        return _x1.get(dev, False)
    if len(_x1_seen) > 4096:
        _x1_seen.clear()
    _x1_seen.add(key)
    ok = _xcc_tab(dev)
    m = _ext(False)
    for gx, gy in _PROBE_GRIDS if ok else ():
        buf = torch.full((gx * gy,), -1, dtype=torch.int32, device=f"cuda:{dev}")
        with torch.cuda.device(dev):
            m.xcc_probe(buf, gx, gy)
        x = buf.cpu().tolist()
        if _probe_hook is not None:
            x = _probe_hook(x)
        ok = ok and _rr_exact(x)
    if not ok and _tab.get(dev):
        warnings.warn(f"kda decode kernel: the workgroup dispatch on cuda:{dev} is not an exact round-robin over 8 XCCs"
                      + (" any more: graphs captured on this device with the per-XCC launch forms are invalid and must be re-captured" if _x1.get(dev) else "")
                      + "; using the symmetric launch forms")
    _x1[dev] = ok
    return ok


def _single_round_plain_lb(B, H, S, spec, ns, nth):
    if spec or S != 1 or ns != 1 or nth != 1024:
        return False
    return bool(prologue_first(B, H, S, spec, ns, nth)) and not next_round(B, H, S, spec, ns, nth) and not hold_ticks(B, H, S, spec, ns, nth)


def xcc_fill(B, H, S, spec, ns, nth):
    """Plain latency-bound whole-state grid (1024 threads, prologue-first, one round): XCC mask to run it on (0 = every XCC, the
    symmetric grid). The grid is launched as 8 ceil(H / m) x B blocks; the block on the r-th XCC of the mask does head
    (blockIdx.x / 8) m + r, the others return at entry (m = 1 is v74's X1). The event is F + max over the used XCCs of S_x + W_x, so
    the grid goes on the m XCCs with the smallest start offsets, with the smallest m for which the busiest of them (B ceil(H / m)
    blocks: a row of 8 workgroups serves m heads of one sequence) carries its state bytes (64 KB in + 64 KB out per block) inside a
    block's own memory phase: B ceil(H / m) x 128 KB / X1_XCC_BW <= X1_SPAN_US, i.e. 16 blocks per XCC (X1 vs v73: 8 / 12 / 16
    blocks -0.47 / -0.40 / -0.41 us, 24 blocks -0.07 / -0.14, 32 blocks +0.27; UNDERFLOOR.md E8b), up to XF_MAX XCCs (beyond it the
    last XCC's start offset is nearly the grid's own: nothing left to gain)."""
    if not _X1_OK or not _single_round_plain_lb(B, H, S, spec, ns, nth):
        return 0
    for m in range(1, XF_MAX + 1):
        if B * -(-H // m) * 2 * 128 * 128 * 4 / X1_XCC_BW <= X1_SPAN_US:
            return _xf_mask(m)
    return 0


def xcc_first(B, H, S, spec, ns, nth):
    """1 = the grid is launched in an XCC-masked form (xcc_fill())."""
    return int(xcc_fill(B, H, S, spec, ns, nth) != 0)


def hold_skew(B, H, S, spec, ns, nth):
    """Plain state-first hold grid (256 threads) whose busiest CU is full: bits 16-31 of the hold argument = skew ticks | XCC mask << 8.
    Every block releases its state stores at its own start + hold; the grid ends on the XCCs with the longest write phase (XCC 3,
    then 5 / 7 / 2: slack 0.1 / 0.45-0.65 us against 0.9-1.0 on XCCs 0 / 1 / 4 / 6, ASYMMETRY A.2). Letting the XCCs whose write
    excess is at least HS_EXCESS_MIN_US release HS_SKEW_US earlier costs the read phase less than it gives their tail. Timing only:
    the kernel is correct whatever XCC a workgroup runs on, so the rule needs the tables (_xcc_tab()) but no round-robin probe.
    Gate: ceil(blocks / CUs) == blocks per CU, the same occupancy step the hold itself is modelled on (hold_ticks()). Measured against
    v74 (NOTES_v14 item H, 24 pairs per cell): 832-1024 blocks -0.26..-0.01 us (15 cells, eleven beyond CI), 640-768 -0.12..+0.08
    (nine cells, none beyond CI: off), 528-576 +0.18..+0.46 (off). Not on the prologue-first hold grids, where the block's own
    read chain, not the timer, releases the stores (ASYMMETRY B.3: every skew +0.1..+1.1 us)."""
    if HS_SKEW_US <= 0 or not _XCC_TAB or spec or ns != 1 or nth != 256 or prologue_first(B, H, S, spec, ns, nth):
        return 0
    t = hold_ticks(B, H, S, spec, ns, nth)
    if not t or -(-B * H // _NUM_CU) != blocks_per_cu(S, spec, ns, nth):
        return 0
    sk = int(round(HS_SKEW_US * 100))
    mask = sum(1 << x for x in range(8) if XCC_WRITE_EXCESS_US[x] >= HS_EXCESS_MIN_US)
    if t <= sk or t >= 1 << 16 or sk >= 256 or not mask:
        return 0
    return (sk | mask << 8) << 16


def xcc_items(B, H, S, spec, ns, nth):
    """Single-round spec whole-state grid (1024 threads, prologue-first) with spare CUs: N | K << 9 (0 = off). The grid ends on the
    XCC with the slowest write path (XCC 3: its token loop lasts 1.22-1.44 x the even XCCs'; slack <= 0.15 us on nine shapes,
    ASYMMETRY A.2 / A.4), so that XCC keeps UQ_FRAC of its N / 8 rows and its last K items run on blocks behind the grid on the two
    XCCs that start first outside its write-path pair (never XCC 2: NOTES_v13 item 3). UQ_FRAC = 0.92: 22 of 24 items; 23 of 24 gives
    half, 21 of 24 or other receivers no more (ASYMMETRY B.6). Needs (i) a spare CU for every moved item: the launched grid must be one
    round (a moved item on a full grid is a second round: +4.8..+6.0 us, NOTES_v14 item U2); (ii) a write-bound grid, i.e. the
    predicate of wave_offset(): the state-write phase at the HBM write ceiling outlasts the block's own token loop by WS_US (below
    it the grid is latency-bound and ends on the XCCs that start last: 108 blocks at S = 2 +0.14 us, NOTES_v14 item U); (iii) the
    exact round-robin (_x1_probe()). Measured against v74 (24 pairs per cell): 22 write-bound shapes of 120-248 blocks -0.33..+0.07
    us, mean -0.13, none slower beyond CI."""
    if UQ_FRAC >= 1 or not _X1_OK or not spec or ns != 1 or nth != 1024 or S < 2:
        return 0
    if not prologue_first(B, H, S, spec, ns, nth) or next_round(B, H, S, spec, ns, nth):
        return 0
    N = B * H
    K = int((1 - UQ_FRAC) * (N // 8) + 0.5)
    if K < 1 or K > 7 or H > 255 or N > 511:
        return 0
    rows = -(-N // 8) + -(-K // 2)  # the grid's rows of 8 workgroups + the receivers' rows (two receivers per row)
    if H * -(-8 * rows // H) > _NUM_CU:
        return 0
    if N * S * 128 * 128 * 4 / WS_HBM_W - S * WS_TOK_US < WS_US:
        return 0  # not write-bound (the predicate of wave_offset()): the grid does not end on the slow write path
    return N | K << 9


def hold_arg(B, H, S, spec, ns, nth):
    """The launch's multi-purpose argument: read-first hold ticks (| hold_skew() << 16) | state rows behind the barrier (1) | first
    nontemporal tail block | -1 wave offset | -2 XCC-masked plain grid (each kernel class reads at most one of them)."""
    t = hold_ticks(B, H, S, spec, ns, nth)
    if t:
        return t | hold_skew(B, H, S, spec, ns, nth)
    return (state_after_prologue(B, H, S, spec, ns, nth) or tail_nt(B, H, S, spec, ns, nth)
            or -wave_offset(B, H, S, spec, ns, nth) or -2 * xcc_first(B, H, S, spec, ns, nth))


def xv_arg(B, H, S, spec, ns, nth):
    """The launch's per-XCC argument: xcc_fill() mask (plain) | xcc_items() N | K << 9 (spec)."""
    return xcc_items(B, H, S, spec, ns, nth) if spec else xcc_fill(B, H, S, spec, ns, nth)


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
    if state_indices.dtype != _I32 or tuple(state_indices.shape) != ((B, S) if spec else (B,)):
        raise ValueError(f"kda decode: state_indices must be int32 {(B, S) if spec else (B,)}, got {state_indices.dtype} {tuple(state_indices.shape)}")
    if spec and (num_accepted_tokens.dtype != _I32 or tuple(num_accepted_tokens.shape) != (B,)):
        raise ValueError(f"kda decode: num_accepted_tokens must be int32 ({B},), got {num_accepted_tokens.dtype} {tuple(num_accepted_tokens.shape)}")
    return state_indices.is_contiguous() and (not spec or num_accepted_tokens.is_contiguous())


def run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
        state_indices, num_accepted_tokens, cu_seqlens, norm_eps, lower_bound, out):
    global _NUM_CU, _X1_OK, _XCC_TAB
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
    _X1_OK = _x1_probe(mixed_qkv.device.index)
    _XCC_TAB = _xcc_tab(mixed_qkv.device.index) if not torch.cuda.is_current_stream_capturing() else _tab.get(mixed_qkv.device.index, False)
    ns, nth = choose(B, H, S, spec)
    vec = use_vec(B, H, S, ns)  # staged prologue; it always loads the state after the staging barrier
    _ext(spec).run(mixed_qkv, conv_state, conv_weight, gate, beta, out_gate, A_log, dt_bias, norm_weight, ssm_state,
               state_indices, num_accepted_tokens, out, _workspace(mixed_qkv.device, B * H, vec), float(norm_eps),
               float(lower_bound), ns, nth, vec,
               prologue_first(B, H, S, spec, ns, nth), hold_arg(B, H, S, spec, ns, nth),
               next_round(B, H, S, spec, ns, nth), xv_arg(B, H, S, spec, ns, nth))
