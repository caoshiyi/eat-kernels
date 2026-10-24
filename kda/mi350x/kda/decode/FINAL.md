**Release check (iteration 19, 2026-10-11): ACCEPTED WITH DEVIATIONS — see `ACCEPTANCE.md`** (v77 correct in every check, 12 official runs 1.5969; deviations: gate 11 of `validate_all.sh` intermittent, first eager call about 2 s from a fresh process, switches set on the `kernel` module have no effect).
# FINAL — fused KDA decode + MTP verify for MI350X (`amd_kda_decode_mi350x`)

**End state: section 0000 (iteration 18).** The final kernel is v77, 1.5989 over 20 official runs; iterations 17 and 18 measured 16 in-kernel arms and
three placement-adaptive levers against it and committed no new kernel. Section 0000 gives the bounds per family, every class of lever with its
outcome, and the leads that remain; section 000 is the complete description of v77.

Hand-off of the kernel at **v77** (`kernel.py` -> `kernel_v16f_a.py`, commit 926e614a2), written 2026-10-11 by the finisher of iteration 17
(who resumed the iteration-16 finisher's files). Official score **1.5989** (20 runs; 000.4), 17/17 correct, `validate_all.sh` PASS.
**Section 000 is self-contained**: what the kernel is, every dispatch rule, every probe and guard, what a caller must do, scores and
the honest scope of each gain, the validation gates, the measurement procedure to trust, floors, limitations and leads.
v77 = v76's kernels, cost model and dispatch, byte for byte, with the host-side defects of the iteration-16 red-team review
(`REDTEAM16.md`) closed; the working log is `NOTES_final16.md`. Everything below section 000 is history, kept as written:
section 00 = v76 (iteration 15), section 0 = v75 (iteration 14), sections 1-7 = v74 / v73 (iterations 13 / 12). Where an older section
contradicts 000, 000 is right; in particular 00.6 (the measurement procedure) is replaced by 000.6.

## 0000. End state (iteration 18, closing statement)

*Written 2026-10-11 by the placement-adaptive closer of iteration 18 (`NOTES_v18.md`). Nothing below section 000 changed; this section is what a
reader can take as the end of the project. Where it and 000.9 disagree about a lead, this section is the later measurement.*

### 0000.1 The final kernel
**v77** (`kernel.py` -> `kernel_v16f_a.py`, commit 926e614a2; described completely in section 000). Official score **1.5989** (mean of 20
official runs, 5 on each of HIP 0 / 2 / 3 / 6; 17/17 correct in all 20; 000.4), `validate_all.sh` PASS on all 16 gates, `--heldout` 11/11
(1.483 against live SOTA). In this iteration's paired official-procedure runs the same kernel read 1.5984 and 1.5993 (two sessions' worth of
the kernel-argument-pool term, 000.6 item 4). Iterations 17 and 18 committed no kernel: 16 in-kernel arms and three placement-adaptive
levers were measured against v77 and none reached the commit line (+0.001 with a CI excluding zero).

### 0000.2 What bounds each workload family now
| family (eval workloads) | now, us (20-run means) | bound | what the remaining distance is made of, and who owns it |
|---|---|---|---|
| **P-LB** d_k3_b1, d_glm16_b1 | 13.30 / 13.29 | fixed launch cost 13.23 | at the floor; 0.02-0.04 us is the kernel-argument pool's die (runtime-owned) |
| **P-HOLD** d_k3_b32, d_glm16_b64, d_glm8_b128 | 20.08 / 33.59 / 34.04 | stream floors 18.8 / 32.2 / 32.4 | non-state bytes and the write tail of XCCs 2, 3, 5 (+ 4 or 7). The tail is convertible by a per-XCC hold skew by 0.12-0.31 us **in a process that allocates only plain-decode pools**, by 0.10 / 0.00 us in the eval's own process (0000.3 lever 1): which XCCs trail follows the allocation history in front of the pool (caller-owned). Kernarg pool's die 0.09-0.27 us (runtime-owned) |
| **P-MR** d_k3_b128, d_glm16_b256 | 45.20 / 98.15 | stream floors 41.8 / 91.6 | cold start of 256 CUs in the first generation; kernarg pool 0.09-0.14 us |
| **V16** s4_k3_b1, s6_glm16_b1, s8_k3_b1 | 14.39 / 15.12 / 15.33 | family floor 14.0 / 14.2 / 14.5 | design-owned 0.4-0.9 us in pieces of 0.1-0.3 us, each attacked on its own; eleven more arms screened in iteration 17 at +-0.02 us: two real (`bo` -0.02 us on S = 6, `vce1` -0.03 us on S >= 6 and +0.02 on S = 4), together +0.0006; 0.15-0.22 us is the kernarg pool's die |
| **S-1R / S-SB** s4_k3_b16, s6_glm16_b16, s6_glm8_b32, s8_k3_b16 | 22.05 / 30.94 / 31.15 / 30.73 | stream floors about 1 us below | prologue chain and store issue; XCC 3's write tail (UQ where there are spare CUs). The state pool's die phase moves them by -0.18..+0.15 us with a sign per (sample of slots, GPU): not predictable (0000.3 levers 2-3) |
| **S-MR** s4_k3_b64, s6_glm16_b64, s8_k3_b64 | 54.69 / 91.87 / 90.72 | write-stream floors 54.8 / 92.9 / 90.5 | at the bound. The pool's die phase moves them by up to +-0.6 us per GPU, sign per (sample of slots, GPU); the mixture of dies a random sample of slots gives is within 0.1-0.25 us of an exactly balanced one and 3-12 us better than a uniform one |

In score terms: about +0.0035 sits in the kernel-argument pool's die (not the kernel's to take), +0.0003 in the hold skew as measured in the
eval's process, +0.0006 in `bo` + gated `vce1`, and the rest in pieces below the resolution of 160 pairs (0.02 us on B = 1 spec, 0.03-0.12 us elsewhere).

### 0000.3 The classes of lever that were tried, and how each was closed
| class | iterations | outcome |
|---|---|---|
| The kernels and their cost model (one launch; whole-state and V-split templates; read-first hold; wave offsets; staged prologue; nontemporal tail; state rows behind the barrier; next-round prefetch; mailbox exchange) | 1-13 | 0.96 -> 1.59. Every feature has a leave-one-out and held-out evidence (sections 1-2, 000.2) |
| Per-XCC launch forms under an exact round-robin (X1 / XF, UQ) with a host-side guard | 13-14, reviewed 16 | +0.0065, kept; constants reviewed and upheld on 22 held-out shapes (000.2) |
| Placement of what the kernel owns: descriptor copy chosen by a timing probe | 15 | +0.0026, kept (v76). Code pages, mailbox, workspaces: below 0.05 us, dropped. Block-sized objects: -0.0013, killed |
| Host-side defects (lock, device, probes beside captures, reload) | 16 | closed in v77 at 0.0000 +- 0.0009 |
| In-kernel variants at the resolution limit (store policies, warm-up loads, pins, exchange order, row reductions, mailbox reset, wave offsets, nt tail, row ownership) | 17 | 16 arms, 160 pairs each: 6 worse, 7 neutral, 2 real but together +0.0006, 1 experiment (`rowx`); none committed |
| **Placement-adaptive choices from the caller's memory** (this iteration; `NOTES_v18.md`) | 18 | three levers + one new one, all closed; table below |

Iteration 18 in one table (us against v77, +- 95 %; "phase" = probed die of the state pool's first page, flipped by a 2 MiB offset of the pool;
cross-process runs of 288-576 fresh processes on four GPUs with a twin of v77 as null arm, inside its CI in every cell; decisions on the
official two-arm A/B, whose twin null read -0.0000 [-0.0013, +0.0012] in this session):

| lever | on the eval workloads | on held-out shapes of the family | by placement | verdict |
|---|---|---|---|---|
| **1. per-XCC read-first hold skew** (0.25 us earlier release on the slow-writing XCCs; state-first hold grids of 769-1024 blocks) | harness: d_glm16_b64 -0.15..-0.20, d_glm8_b128 -0.15..-0.21; **eval's own process: -0.05..-0.11 and 0.00 +- 0.04**; official A/B of the mask {2,3,5,7}: **+0.0002 [-0.0005, +0.0009]** | harness, seven shapes of 896-1024 blocks incl. re-sampled twins: -0.12..-0.31, four GPUs of one sign | die phase: no effect (same size in both). What matters is the allocation history in front of the pool: the harness run on the eval's own 17-workload sequence reads -0.04 / -0.05. Slow set {2,3,5,7} in the harness, {2,3,4,5} in the eval; eleven masks tried in the eval's process, none better than -0.11 / -0.06, complements +0.06..+0.17, 0.5 us +0.27 | real, never positive for a mask with XCC 3, **+0.0003 on the eval: below the line; nothing for a probe to select** |
| **2. row-ownership swap** in the whole-state spec kernels (`rowx`), chosen from the pool's phase | phase 1: s8_k3_b64 -0.51 +- 0.13, s4_k3_b64 -0.18, s8_k3_b16 -0.13, s6_glm8_b32 -0.13, **s6_glm16_b16 +0.23 +- 0.04**, s6_glm16_b64 +0.10 (n.s.); phase 0: the mirror image | phase 1: K3 S-MR -0.19..-0.38, 16,40,6 -0.16, 16,64,6 +0.07, **8,128,6 +0.17 +- 0.15** | per GPU the sign flips on 5 of 7 held-out / twin shapes (e.g. 12,48,8: -0.33 / +0.46 / -0.63 / +0.67 on HIP 0 / 2 / 3 / 6); the same shape with another sample of slots: +0.48 against +0.10 | **killed: the sign belongs to (sample of `state_indices`) x (GPU), not to the pool or the shape** |
| **3. the pool's die phase itself** | BASE phase 1 - 0: s8_k3_b64 +0.48, s4_k3_b64 +0.18, s6_glm8_b32 +0.15, s8_k3_b16 +0.10, s6_glm16_b16 -0.18, others 0 | -0.17..+0.35 | `rowx` IS the phase flip (24 of 28 mirror cells agree), so 3 = 2. A pool has no single phase: the die of a state is (constant of its 256 MiB segment) xor (bit 21 of its address), with a segment pattern per GPU (0 1 0 / 1 0 0 / 1 0 1 / 1 0 1). Pools above ~500 MiB get one first-page phase per GPU in every process; smaller ones draw. A pure address rule does not give the die (41-49 % agreement) | **killed with 2** |
| **per-block row map** from each block's own slot (new; sized on v77 with constructed index samples, no kernel built) | - | every slot in use on one die: **+3.4..+12 us**; reads only: +0.4..+2.4; exactly alternating against random: -0.26..+0.14 | - | **killed as a speed lever** (the random mixture is already the good state); robustness lead (0000.4 item 5) |

Why the three "per-workload signs" of iterations 14-17 looked like properties of the shapes: the eval seeds its inputs by the case name, so
each workload has ONE sample of slots; the large pools get one phase per GPU; and each harness has one allocation history. A fixed rule
fitted to those signs would have been a rule fitted to the eval's index samples and allocation sequence.

Two caller-side facts found on the way (not levers of the kernel; the eval's pools never have either property):
- a state pool whose states are not 64 KiB-aligned (pool starting 8 KiB into its allocation) costs the multi-round spec grids +0.8..+5.7 us;
- slots in use that all share bit 21 of their address (e.g. for H = 16 every slot with the same bit 1 of its index) cost every whole-state
  spec grid +3.4..+12 us: all blocks then write the same die at the same time.

### 0000.4 The leads that remain, and why each is out of the kernel's reach or below resolution
| # | lead | size | why it is not done |
|---|---|---|---|
| 1 | **Kernel-argument pool on the die of the late XCC quad** | +0.003..+0.004, and the largest variance term | the pool is allocated by the runtime at `hipGraphInstantiate`, after `run()` returned; nothing can be probed inside a capture. **The only kernel-side form — keeping dummy 132 KiB allocations alive so that the caller's next pool lands on a good piece — was ruled not permitted** (it steers an allocation made for the caller). Two argument blocks, no preload, a first node with padding: dead by arithmetic (NOTES_v15 section 3). Open to a caller (probe after `capture_end()` with `kda_pp`, instantiate again if on the early quad's die) or to the runtime |
| 2 | Per-XCC hold skew as a fixed rule (`kda_v18_k.cu -DKM_HS=1`, mask XCCs 2, 3, 5, 0.25 us) | +0.0003 on the eval; -0.12..-0.31 us on full plain hold grids in a plain-decode-only process; never positive | below the commit line alone; needs a ninth kernel per copy of the default object (descriptor pads, layout check, `place_tests.py`, dispatch reference, asm / ISA gates). Worth taking only together with another change to that object, or for a deployment that is plain-decode only (measure there first: its size is the allocation history's) |
| 3 | `bo` (branchless out stores) + `vce1` gated to long token windows | +0.0006 +- 0.0002 | below the line; the gate needs a boundary measured at S = 5 / 7 and the mailbox fault-injection gate (NOTES_v17 3.2) |
| 4 | Four descriptor windows (die x slow class) | <= +0.001 | four copies of every kernel: the max-ILP object grows to ~3 MiB and its buddy geometry changes (000.9 lead 3) |
| 5 | Per-block row map from (bit 21 of the block's slot address) xor (block parity): the die mixture exact for any sample of slots | 0 on the eval and on any random or sequential sample; removes a 3-12 us worst case | robustness only; costs run-time instructions on the slot path of the whole-state spec kernels, where one entry instruction has cost 1-2 us before |
| 6 | The unbuilt V16 switches of iteration 16 (`PR_V16QKSPREAD`, `PR_V16BL`, `PR_V16LDS`, `PR_V16M32`, `PR_V16H32`, `PR_V16BETA`, `PR_V16HDUP`, `PR_V16RST`, `PR_V16WU0`) | each a budget row of <= 0.1 us, <= +0.002 if a whole row were won | eight of the eleven arms of the same kind read 0 +- 0.0004 or worse |
| 7 | 17-24 blocks per XCC on three XCCs | held-out only (-0.055 us on one shape) | the same limit loses on four other shapes |
| - | **closed in iteration 18** (were 000.9 leads 2 and 6): the state pool's die phase on S-MR; the hold skew with a mask from the pool's address | - | 0000.3 |

The measurement tools to trust are those of 000.6, plus for placement questions `scratch_v18/`: `draw18.py` / `arms18.sh` / `strat.py` (per-process
die probes of the caller's pool, phase flipped by a 2 MiB offset — never by 8 KiB — and never left to chance for pools above 500 MiB),
`offarms.sh` (multi-arm screen inside the eval's own process), `orient.py` (constructed slot samples). A result obtained in a harness with its
own allocation history must be repeated in the eval's own process before it is believed: lever 1 is -0.2 us in one and -0.05 in the other.

---

## 000. v77 (iterations 16-17)

### 000.1 What the kernel is

One call of `run()` = host-side validation and dispatch + **exactly one kernel launch**; a HIP-graph replay is that one launch
and nothing else (no host work, no second node, no fill). S > 8 and strided index tensors go to the vendored AITER dispatcher
(`kernel_aiter_fallback.py`, `kernel_triton.py`).

**Device code**: one HIP C++ source, `kda_v14_b.cu` (~2800 lines; unchanged since v75), compiled through the wrapper `kda_v15_a.cu`,
which includes it **twice** (namespaces `kc0` / `kc1`: two byte-identical copies of every kernel, their descriptor tables exactly
8 KiB apart) and adds filler descriptors and two probe kernels (`kda_where`, `kda_pp`). Two JIT builds
(`torch.utils.cpp_extension.load`, cached under `build/v15a_<digest>` by a digest of both sources, the ROCm version and the pads):
the **default build** (plain kernels, 8 per copy; 296 KiB code object) and the **max-ILP build**
(`-mllvm -amdgpu-sched-strategy=max-ilp -DKDA_SPEC_ONLY`: spec kernels, 60 per copy; 1552 KiB).

Two kernel templates, one block per (sequence n, head h) or per V-slice of it:

| family | instantiation | what a block does | used for |
|---|---|---|---|
| `kda_main` whole-state | `kda_main<128, S, spec, NT, PF, NX, SB, W1, XV>` | all 128 state rows of one (n, h): per-thread prologue loads, conv + SiLU + L2 norms + per-token scalars into LDS, S token steps (row dots, delta-rule update, one fp32 state store per token), gated RMSNorm epilogue, conv-cache commit | every plain grid; spec grids of more than 64 (n, h) pairs |
| `kda_vec` V-split | `kda_vec<VB = 32 or 16, S, spec, 512, LATE>` | 128 / VB blocks per (n, h), each owns VB state rows; staged prologue (16 B chunks -> LDS); the RMSNorm sum of squares is assembled across sibling blocks through **mailboxes** (128 KiB of device memory per captured call / per (device, stream) for eager calls; each block resets, publishes, polls) | spec grids of <= 64 (n, h) pairs |

Kernel classes (each its own instantiation, so grids that do not use a feature run unchanged code) and the eval workloads on them:

| class | what | eval workloads |
|---|---|---|
| P-LB / **X1 / XF** | plain, 1024 threads, prologue-first; <= 16 blocks per XCC on the 1-4 XCCs that start first | d_k3_b1, d_glm16_b1 (X1) |
| P-HOLD | plain, 256 threads co-resident, read-first hold | d_k3_b32, d_glm16_b64, d_glm8_b128 |
| P-MR | plain, 1024 threads, multi-round, next-round prefetch | d_k3_b128, d_glm16_b256 |
| V16 / V32 | `kda_vec<16>` / `<32>` | s4_k3_b1, s6_glm16_b1, s8_k3_b1 (V16) |
| S-1R (W1) / S-SB, optionally **UQ** | spec, 1024 threads, single round; wave offset or state rows behind the barrier; XCC 3 hands its last K items to blocks behind the grid | s4_k3_b16, s8_k3_b16 (UQ); s6_glm16_b16, s6_glm8_b32 (full grids: no UQ) |
| S-MR | spec, 1024 threads, multi-round, wave offsets, nontemporal tail | s4_k3_b64, s6_glm16_b64, s8_k3_b64 |

**Host code, as it actually is: four Python layers, each a module that owns private instances of the ones below it.**

| layer (file) | version | owns | talks to the layer below by |
|---|---|---|---|
| `kernel.py` | - | `from kernel_v16f_a import run` | - |
| **`kernel_v16f_a.py`** | **v77** | the process-wide lock and state holder, device guard, best-effort probes (`_place`, `_probe_place`), `warmup()`, `HOST_PROBES` | a private instance of `kernel_v15_a.py` (`M`); replaces its `_place` / `_probe_place`; calls `M.run` |
| `kernel_v15_a.py` | v76 | the two builds (`_ext`), ELF layout check (`_layout`), the verdict rule (`_verdict`), copy selection (`copy_of`), the launch | private instances of `kernel_v14_b.py` (`K`) and `kernel_v13_a.py` (`V`), with their build hooks pointed at its builds |
| `kernel_v14_b.py` | v75 | the per-XCC tables and rules (`xcc_fill`, `xcc_items`), the guard (`_xcc_tab`, `_stream_ok`, `_x1_probe`, `_guard`) | `V` |
| `kernel_v13_a.py` | v74 | the cost model (`choose`, `use_vec`, `prologue_first`, `hold_ticks`, `state_after_prologue`, `next_round`, `tail_nt`, `wave_offset`), input validation (`_validate`), mailboxes (`_mailbox`, `_workspace`), occupancy from compiled VGPR counts (`blocks_per_cu`) | - |

Attribute access on `kernel` / `kernel_v16f_a` falls through the layers (a name is looked up in v77, then v76, v75, v74); an
assignment goes to the layer that defines the name. The lower three files are not modified by v77 and must stay as they are
(`isa15.py`, the dispatch reference and the red-team models bind to them). `kernel_v13_a.py`'s own source `kda_v13_a.cu` and v75's
`-DKDA_XONLY` builds are no longer built or launched; only `kda_v14_b.cu` through `kda_v15_a.cu` is.

What `run()` does on the host, in order (`kernel_v16f_a.run` -> `kernel_v15_a.run`):
1. tensors must be on a HIP device (`TypeError` otherwise); take the process-wide lock; if the tensors' device is not the current
   one, enter `torch.cuda.device(<it>)`;
2. `_validate()` (dtypes, shapes, contiguity; strided indices or S > 8 -> AITER fallback);
3. `_place(dev)`: once per device, only outside a capture (000.3);
4. cost model -> `(nsplit, threads)`, the launch arguments, the guard's verdict for this call, the descriptor copy;
5. the launch on the current stream of the tensors' device; `_X1_OK` reset; lock released.

### 000.2 Dispatch rules

Inputs of every rule: B, H, S, plain / spec, the device's CU count, compiled VGPR counts, and (for the two per-XCC rules) the
guard's verdict. No shape table, no workload names, nothing about call order or timing. The dispatch on 109 854 shapes is pinned
by validation step 9 (`scratch_final/val/disp_ref.txt`; v77 = v76 = v75 on all of them).

| rule (function, file) | cost model | constants (provenance) | evidence |
|---|---|---|---|
| V-split (`choose`, v13) | spec only: split V while the split grid leaves the blocks alone on their CUs: 32-row blocks if `4 B H <= CUs`, 16-row blocks if `16 B H <= CUs` and `S x V16_US_PER_TOKEN > V16_FIXED_US` | 0.14 us per token saved by 16-row blocks, 0.31 us fixed (160 pairs per shape on v70, NOTES_final 4.1) | boundary measured on both sides (S = 2 / 3); section 1 below |
| threads (`choose`) | 1024 threads if spec, or blocks <= CUs, or blocks > CUs x (256-thread blocks per CU); else 256 threads (co-resident) | blocks per CU from the compiled VGPR count of the launched instantiation (`blocks_per_cu`) | NOTES_bw2 P2, NOTES_opt3 K1 |
| load order (`prologue_first`) | prologue-first when the block's own prologue is on its critical path: latency-bound grids, every spec 1024-thread grid, plain hold grids of <= 2 blocks per CU; state-first on plain 3-4 blocks per CU | - | NOTES_bw2 P5, E41 |
| read-first hold (`hold_ticks`) | plain co-resident grids hold their state stores for `max(HOLD_MIN, HOLD_CU_US x ceil(blocks / CUs) + state bytes / HOLD_BW2)` | 4.9 us, 0.945 us per co-resident block, 9.1e6 B/us (two-term fit on 12 fills, checked on 24 others: NOTES_ablate re-fit 1) | 48 held-out fills |
| next-round prefetch (`next_round`) | multi-round prologue-first grids: block b prefetches the inputs of block b + (blocks per round) | blocks per round = CUs x blocks per CU of that instantiation | NOTES_ablate M17, S23 |
| state rows behind the barrier (`state_after_prologue`, SB) | single-round spec grids with `(2 S + 25) x LOAD_ISSUE_US > SLOT_HOP_US` (S >= 6) and `B H x 64 KB / HOLD_BW_NT <= SPEC_PRO_US` (<= 212 blocks) | 25 clocks per cold load issue (ISSUE_PATH 1b), 0.40 us slot hop (NOTES_lat 1), 5.8e6 B/us, 2.4 us prologue chain | held-out neighbours on both sides (NOTES_final rows 3-4) |
| nontemporal tail (`tail_nt`, TN) | spec grids of > 2 rounds with `S x TN_US_PER_TOKEN > TN_FIXED_US` (S >= 4): blocks without a successor on odd XCCs store nontemporal | 0.33 us per token, 1.1 us fixed (fitted slope / intercept over S = 2..8 on held-out grids, re-fit on v70) | 50 held-out shapes of 264-2048 blocks |
| wave offset (`wave_offset`, W1; WS / WM compiled into the multi-round kernel) | single-round spec grids whose write phase outlasts the block's token loop by the offset: `B H S x 64 KB / WS_HBM_W - S x WS_TOK_US >= WS_US` (blocks >= 191 / 156 / 138 / 127 / 120 / 115 / 111 for S = 2..8) | 6.1e6 B/us HBM write ceiling (MECHANISMS (c)), 0.9 us per token step (per-block stamps), 2.3 us = the offset itself | NOTES_xcd 4-6; six held-out shapes |
| **`xcc_fill` (X1 = m 1, XF = m 2..4; v14)** | plain latency-bound single-round grid: run it on the m XCCs that start first, smallest m <= `XF_MAX` with `B ceil(H / m) x 128 KB / X1_XCC_BW <= X1_SPAN_US`, **i.e. <= 16 blocks on the busiest XCC**; grid `8 ceil(H / m) x B`, the block on the r-th XCC of the mask does head `(blockIdx.x / 8) m + r`, the others return at entry. Needs the guard | `XCC_START_US` (start offsets, ASYMMETRY A.1) gives the order 1, 0, 3, 7; **the 16 is a measured block count** (the two constants only appear as their quotient); `XF_MAX = 4` (m = 5 / 6 measured: -0.16..+0.06 us) | iteration 16, 144 rounds on four GPUs (NOTES_final16 section 3): rule off +0.32..+0.35 us on 12-16 blocks, +0.12..+0.23 on two to four XCCs, 4 of 4 GPUs each; limit moved to 24 per XCC: four shapes slower (+0.03..+0.19), one faster (-0.055); shapes beyond the limit unchanged |
| **`xcc_items` (UQ; v14)** | write-bound (the predicate of `wave_offset`) single-round spec whole-state grid with spare CUs (`H x ceil(8 rows / H) <= CUs` including the receivers' rows): the XCC with the slowest write path keeps `UQ_FRAC` of its N / 8 rows; its last `K = round((1 - UQ_FRAC) N / 8)` items run on blocks behind the grid on the two XCCs that start first outside its write-path pair. Needs the guard | `XCC_WRITE_RATIO` (XCC 3: 1.22-1.44 x, ASYMMETRY A.4); `UQ_FRAC = 0.92` (read off N = 192: 22 of 24 rows, ASYMMETRY B.6; on other shapes it only decides K = 1 below N = 152, K = 2 above) | iteration 16: rule off +0.12..+0.39 us on five K = 2 shapes, +0.04..+0.09 on three of five K = 1 shapes (two neutral, none faster) |
| descriptor copy (`copy_of`, v15) | not a shape rule: every launch uses the copy of the kernel whose descriptor table the placement probe found on the I/O die of the XCC quad that starts last | 000.3 | 000.4 |
| in-kernel, compile time | late conv commit for S >= 7; multi-round stagger 0 / 0.75 / 1.5 us by a hash of the sequence; pool translation warm-up in the first 64 blocks; mailbox reset behind the staging barrier (`MB_CLR = 5`) | NOTES_probes, MECHANISMS (a), ISSUE_PATH | sections 1-2 below |

**The reviewer's observations on the two measured constants, and the verdicts** (REDTEAM16 section 5; evidence NOTES_final16 section 3:
144 cross-process rounds on four GPUs, 22 held-out shapes, four arms in random order with nulls):

- **`X1_SPAN_US = 2.1` with `X1_XCC_BW = 1.0e6`.** Observation: the two only ever appear as their quotient, "<= 16 blocks on the busiest
  XCC"; d_glm16_b1 has exactly 16 and drops out below 2.097, so the constant is a literal of an eval shape written as physics.
  **Verdict: legitimate, and to be read as a measured block count, not as a memory model.** The boundary is not cut to the eval
  shape's advantage: with the rule off nine held-out shapes on one to four XCCs are 0.12-0.35 us slower on 4 of 4 GPUs; with the limit
  moved to 24 blocks per XCC four shapes get slower (+0.03..+0.19 us) and one faster (-0.055); shapes beyond the limit are unchanged in
  every arm; and no eval workload is near the boundary's other side (the next plain eval shape has 384 blocks).
- **`UQ_FRAC = 0.92`.** Observation: "22 of 24 rows" is the row count of the eval's N = 192 (s4 / s8_k3_b16), the eval's launches are
  the same for 0.896..0.936, and elsewhere the constant only decides K = 1 (N <= 151) or K = 2; the K = 1 half was unproven and a
  candidate for "K >= 2 only". **Verdict: legitimate as it is, including K = 1.** K = 2 with the rule off: five held-out shapes
  +0.12..+0.39 us. K = 1 with the rule off: three of five shapes slower (+0.04..+0.09 us, beyond CI), two neutral, none faster; so K = 1
  gains about a third of K = 2 and never loses, behind the same guard. Had it been neutral it would have been cut.
- Neither constant can select a shape by form: the domain of each rule is set by block counts, write-boundness (`wave_offset()`'s
  existing predicate) and spare CUs. The public `--heldout` set has no X1 or UQ shape; gate 15 (`heldout16.py`) covers both rules and
  both sides of every boundary.
- Not inputs of any rule: workload order, allocation history, whether the call is timed. The eval's eager warm-up calls are what lets
  the probes run (000.3.1); since v77 any eager call of any shape, or `warmup()`, does.

Switches for an operator (set on the `kernel` module before the first call): `X1_SPAN_US = 0` (no X1 / XF), `UQ_FRAC = 1` (no UQ),
`FORCE_COPY = "off"` (no placement probe, copy 0), `HOST_PROBES = False` (none of the above and no blocking first call).

### 000.3 Probes and guards

| probe / guard | when | what it does | what it assumes | fallback |
|---|---|---|---|---|
| `_validate` (v13) | every call | dtype, shape, contiguity, one device | - | `TypeError` / `ValueError`; strided index tensors and S > 8: AITER path |
| device guard (v16f) | every call | tensors on a HIP device; host work and launch under the tensors' device | - | `TypeError` for CPU tensors |
| lock (v16f) | every call | one process-wide `RLock` around the host part and the launch | the layers below keep per-call decisions in module globals | - |
| `_xcc_tab(dev)` (v14) | once per device, host only | 8 XCCs (`hipDeviceAttributeNumberOfXccs`), 256 CUs, the compiled XCC constants equal the launcher's tables | the tables are MI350X's | symmetric forms, copy 0, no probe launched |
| `_stream_ok(dev)` (v14) | every call a per-XCC rule could apply to, host only | the current stream has no CU mask (`hipExtStreamGetCUMask`) | a masked queue is another queue type | symmetric form for that call |
| round-robin probe `_x1_probe` (v14; run by v16f's `_place`) | first eager call on a device (any shape), or `warmup()` | six probe grids of the launch forms in use; `_rr_exact()`: workgroup i ran on XCC (first + i) mod 8, every id < 8 | the dispatch is an exact round-robin over the XCCs with any origin (driver: MQD chunk size 1) and stays one for the life of the process | not exact: symmetric forms, sticky, one warning; `xcc_reprobe()` only ever turns it off |
| placement probe `_probe_place` (v16f, verdict rule `_verdict` of v15) | same call, per build | uploads the code object; one wave on XCC 1 and one on XCC 5 time 16 cold loads at filler lines of each descriptor window (72 launches, each behind a 512 MiB fill of a buffer taken with `hipMalloc` and freed before returning); the copy whose window reads >= 0.03 us slower from XCC 1 while the other reads >= 0.03 us faster | (i) a descriptor on the die of the quad that starts last is faster for every kernel class (measured on v75 in 600 processes); (ii) the loader places the object at offset = ELF address | copy 0 (= what v75 is: a draw) when: not `_xcc_tab`; `_layout()` refuses the build; **not the audited ROCm (7.2.0)**; **the lines are not inside one device mapping (`/proc/self/maps`)**; **a reading loaded other bytes than the ELF's**; no memory for the buffer; unclear readings; **any exception** (warning) |
| capture | every call | inside a capture nothing is probed, synchronised or (after the first eager call) allocated | - | a capture before any eager call on the device: symmetric forms, copy 0, placeholder / mailbox allocated inside the capture (one fill node each) |
| beside another thread's capture (v16f) | during `_place` | the calling thread is put in `hipStreamCaptureModeThreadLocal` for the probes and only its own stream is synchronised | (measured, NOTES_final16 section 2) | a failure that still names a capture: retried at the next eager calls, 3 attempts |
| reload (v16f) | import | mailbox registries and device verdicts on `sys.modules["_kda_decode_state"]` | - | - |
| `MB_CLR` mailbox reset (device) | every V-split launch | every block resets the words it will publish before use | sibling blocks of a head start together: best-effort after an aborted launch on a **shared** GPU (`reset_workspace()` then) | - |

First-call cost per device (warm build cache): ~30 ms, 512 MiB of device memory for that time (outside torch's accounting), 4 MiB
left (the two code objects). A cold build cache adds ~54 s of JIT. `warmup(device)` moves all of it to a moment of the caller's choice.

### 000.3.1 What the kernel needs from a caller

| topic | requirement / behaviour |
|---|---|
| **before the first capture** | one eager `run()` of **any shape** on the device, or `kernel.warmup(device)`, outside any capture and preferably while no other thread captures. It blocks ~30 ms and takes 512 MiB of device memory for that time (a cold build cache adds ~54 s of JIT). Without it the process is still correct but gets the symmetric launch forms and descriptor copy 0 until its first eager call (slower by what X1 / XF / UQ / placement are worth), and the placeholder / mailbox of a capture-first call are allocated inside the caller's capture (one fill node each). Graphs captured before the first eager call keep the symmetric form; re-capture them to get the per-XCC forms |
| a caller that cannot have a blocking first call | `kernel.HOST_PROBES = False` before the first call: no probe, no synchronisation, no transient allocation; symmetric forms and copy 0 (v73's dispatch) |
| **threads** | `run()` may be called from several threads: one process-wide lock serialises the host part (microseconds; ~30 ms once during the probes). A first eager call beside another thread's open capture is safe (thread-local capture mode; retried up to 3 eager calls if the runtime still refuses). Replays of captured graphs take no lock |
| **devices** | all tensors of a call on one HIP device; it need not be the current device (the call enters `torch.cuda.device` itself). CPU tensors: `TypeError`. Each device is probed at its own first eager call. Platform limit, not the kernel's: on ROCm 7.2.0 a *graph* of any custom kernel on a second device of one process faults (AITER's too); eager calls on several devices work |
| **streams** | any stream. A CU-masked stream gets the symmetric form (checked every call). A graph is expected to replay on the queue type it was captured on: torch always replays on the capture stream; a caller that launches the raw graph handle on a CU-masked stream is out of contract (correct on this driver for 8 masks tried, not guaranteed on another) |
| capture | `run()` is HIP-graph capturable: inside a capture nothing is probed, synchronised or launched besides the one kernel; after the first eager call nothing is allocated either (V-split captures take a pre-allocated spare mailbox; the third V-split capture in a row with no eager call between allocates one inside the capture) |
| memory left behind | 4 MiB of code objects per device; 128 KiB per captured V-split call (kept for the life of the process, graphs point at them); `sys.modules["_kda_decode_state"]` holds these across `importlib.reload` |
| after a device error | `reset_workspace()` (mailbox healing after an aborted launch is best-effort on a shared GPU) |
| other toolchains / devices | not ROCm 7.2.0: a warning, placement probe off (copy 0), re-run gate 8. Not an 8-XCC, 256-CU MI350X (MI300X, DPX / CPX partitions): symmetric forms, copy 0, no probe launched |

### 000.4 Scores
Official eval (`python tools/eval_amd_kda_decode_mi350x.py`), **5 runs on each GPU**, the four in parallel, NUMA node 0
(`bash scratch_final16/final_measure.sh 5 final17`; logs `scratch_final16/final17/`; canary before and after every run:
13.12-13.76 us, no GPU shared). All 20 runs 17/17 correct. Kernel as committed in 926e614a2.

| GPU (HIP) | runs | v77 mean_vs_base | range | v76 (iteration 15, one session earlier) |
|---|---|---|---|---|
| 0 | 5 | 1.6085 | 1.6056 - 1.6115 | 1.6109 |
| 2 | 5 | 1.6033 | 1.5975 - 1.6068 | 1.6045 |
| 3 | 5 | 1.6051 | 1.5990 - 1.6085 | 1.6071 |
| 6 | 5 | 1.5787 | 1.5697 - 1.5844 | 1.5800 |
| all | 20 | **1.5989** | 1.5697 - 1.6115 | 1.6006 |

**The two columns are not a comparison of v76 and v77.** They are two sessions; the comparison is the paired A/B below, in which
v76 itself reads 1.5991 in this session. The -0.0017 between the tables is the session term of 000.6 item 4 (the kernel-argument
pool's odds), inside what a 20-run table moves by.

| workload | SOTA us | HIP 0 us | HIP 2 us | HIP 3 us | HIP 6 us | mean us | vs_base |
|---|---|---|---|---|---|---|---|
| d_k3_b1 | 16.28 | 13.19 | 13.20 | 13.17 | 13.62 | 13.30 | 1.224 |
| d_k3_b32 | 23.44 | 19.98 | 19.93 | 20.10 | 20.30 | 20.08 | 1.168 |
| d_k3_b128 | 54.44 | 45.06 | 45.16 | 45.16 | 45.41 | 45.20 | 1.204 |
| s4_k3_b1 | 25.04 | 14.35 | 14.31 | 14.33 | 14.58 | 14.39 | 1.740 |
| s4_k3_b16 | 39.88 | 21.91 | 22.03 | 21.95 | 22.30 | 22.05 | 1.809 |
| s4_k3_b64 | 87.08 | 54.58 | 54.42 | 54.61 | 55.13 | 54.69 | 1.592 |
| s8_k3_b1 | 32.48 | 15.31 | 15.23 | 15.26 | 15.52 | 15.33 | 2.118 |
| s8_k3_b16 | 58.36 | 30.37 | 30.52 | 30.54 | 31.50 | 30.73 | 1.899 |
| s8_k3_b64 | 147.52 | 89.76 | 91.11 | 90.64 | 91.35 | 90.72 | 1.626 |
| d_glm16_b1 | 18.80 | 13.17 | 13.18 | 13.18 | 13.62 | 13.29 | 1.415 |
| d_glm16_b64 | 51.04 | 33.35 | 33.59 | 33.54 | 33.87 | 33.59 | 1.520 |
| d_glm16_b256 | 135.80 | 97.93 | 98.13 | 98.22 | 98.31 | 98.15 | 1.384 |
| s6_glm16_b1 | 28.68 | 15.10 | 14.97 | 15.10 | 15.31 | 15.12 | 1.897 |
| s6_glm16_b16 | 53.68 | 30.68 | 30.99 | 30.72 | 31.37 | 30.94 | 1.735 |
| s6_glm16_b64 | 148.28 | 90.63 | 92.53 | 91.57 | 92.74 | 91.87 | 1.614 |
| d_glm8_b128 | 51.48 | 33.96 | 34.10 | 33.82 | 34.26 | 34.04 | 1.513 |
| s6_glm8_b32 | 53.60 | 30.90 | 31.02 | 30.93 | 31.74 | 31.15 | 1.721 |

**Stack A/B v76 -> v77**, official procedure (`tools_ab/ab.py scratch_final16/arms/k_v76.py scratch_final16/arms/k_v77.py --pairs 40`:
two arms, AB / BA alternation, four GPUs in parallel; report `scratch_final16/ab_v76_v77.txt`):

| n | v76 | v77 | delta | per GPU (HIP 0 / 2 / 3 / 6) |
|---|---|---|---|---|
| 160 pairs | 1.5991 | 1.5991 | **-0.0000 [-0.0009, +0.0009]** | -0.0004 / -0.0000 / +0.0008 / -0.0004 |

No workload beyond its CI (largest: s8_k3_b16 +0.05 +- 0.07 us, s8_k3_b64 +0.04 +- 0.09, d_glm8_b128 -0.02 +- 0.04; the B = 1 and plain
workloads within 0.01 us). **The correctness fixes cost nothing measurable**: a replay is the same single launch from the same code
objects (both `.so` files unchanged by sha256, 136 kernel copies = v75's bytes).
**Held-out (unscored)**: `--heldout` 11/11, 1.483 against live SOTA (v76 1.483); boundary sweep worst 1.066 vs SOTA (y_h24_b21_s1);
`heldout16.py` 30 of 30 shapes correct with the documented launch form.

**The honest scope of each gain** (score points, official paired A/B at the time unless stated):

| gain | size | where it holds | where it does not |
|---|---|---|---|
| v1-v73: the kernels and the cost model (one launch, V-split, hold, wave offsets, staged prologue, ...) | 0.96 -> ~1.59 | every caller, every shape class; held-out 1.48 against live SOTA | - |
| X1 / XF (`xcc_fill`, v74 / v75) | +0.0049 (two eval workloads, 0.32-0.41 us each) | plain grids of <= 16 blocks per XCC on 1-4 XCCs; needs the guard: 8-XCC MI350X, unmasked stream, round-robin probe passed, i.e. **an eager call or `warmup()` before the capture** | capture-first processes, CU-masked streams, partitioned or other devices: symmetric form (v73's speed) |
| UQ (`xcc_items`, v75) | +0.0016 (two eval workloads, 0.12-0.39 us) | write-bound single-round spec grids with spare CUs (N = 111..248 blocks depending on S), same guard | same as above; full grids (no spare CU) |
| descriptor placement (v76) | +0.0026 | **largest under the eval's flush before every replay**: chosen copy against the other +0.10..+0.125 us on B = 1 spec | **shrinks without the flush**: +0.03..+0.04 us in single unflushed replays, 0.00 +- 0.02 in back-to-back chains of the same kernel (`exp_l4a.py`). Never negative (same instructions). A real decode step runs other layers' kernels between two replays, so it sits between the two regimes; not measured in a full model |
| v77 | 0.0000 +- 0.0009 | correctness and side effects only | - |

### 000.5 Validation gates
`bash scratch_final/validate_all.sh <tag>` (~25 min, all four GPUs; last line `VALIDATE_ALL PASS|FAIL`; logs `scratch_final/val/<tag>/`).
Status of the committed kernel: `validate_all.sh v77b` **VALIDATE_ALL PASS**, all 16 gates (`scratch_final/val/v77b/summary.txt`); 7b ran
on HIP 3 (canary 13.12 us): 0 stale of 4200.

| step | gate |
|---|---|
| 1-4 | official eval 17/17; `--heldout` 11/11; extra shapes; dispatch-boundary sweep (two halves) |
| 5-6 | replay stress on 22 shapes over every kernel class; every reachable V-split instantiation (`allvec.py`) |
| 7, 7b | iteration-12 red-team cases T1-T12 (`scratch_final/gpu_tests.py`); mailbox fault injection (`poison_rate.py`, exclusive GPU, after the parallel steps) |
| 8, 12 | CPU: both builds compile, asm hazard / DPP checkers clean, no spills; every kernel copy byte-identical to v75's binaries (`scratch_v15/isa15.py`) |
| 9 | dispatch table on 109 854 shapes identical to the reference |
| 10-11 | per-XCC launch forms (`scratch_v14/xcc_tests.py`); descriptor placement (`scratch_v15/place_tests.py`) |
| **13** | **iteration-16 red-team tests** (`scratch_final16/gpu_tests16.py`, `t3e.py`): T1 first-call cost / residue, T2 two threads x 300 000 calls with a rejected stream, T6 reload, T7 first call beside another thread's capture, T8 round-robin beside 10 loading processes, T9 rules on after any eager call, T11 fault inside the probe, T4 unmasked graphs launched on 8 CU-masked streams (under `timeout`), T3e first calls on two devices (threads / in order), T5 tensors on a non-current device; T10 informational |
| **14** | **portability** (`scratch_final16/port16.py`): 15 forced states, each in a fresh process: 6 XCCs, 128 CUs, failing attribute query, foreign XCC constants, five kinds of wrong XCC-id readings, a real CU-masked stream, unaudited ROCm, unmapped lines, foreign bytes, `HOST_PROBES` off, capture first, CPU tensors -> correct, symmetric / copy 0 as documented |
| **15** | **held-out coverage of every per-XCC rule** (`scratch_final16/heldout16.py`): 30 shapes on both sides of each boundary, launch form as documented, checked before and after replays (`--timed`: against live SOTA) |

Two facts about the platform that shape the tests: `torch.cuda.CUDAGraph.replay()` launches on the graph's capture stream whatever
stream is current (a masked replay needs `hipGraphLaunch` on the raw handle); and on ROCm 7.2.0 a graph of **any** custom kernel on
a second device of a process faults once that kernel was captured on the first (AITER's Triton kernel included), so two-device
tests are eager.

### 000.6 The measurement procedure
1. **Official score**: `python tools/eval_amd_kda_decode_mi350x.py`. For a number to quote: `bash scratch_final16/final_measure.sh 5 <tag>`
   (5 runs on each of HIP 0 / 2 / 3 / 6 in parallel, NUMA node 0, a canary before and after every run; a run whose canary is
   above 14.3 us ran beside another user's job and is listed, not averaged; HIP 6 reads 13.6-13.8 clean). Check
   `scratch_v14/freegpus.py` first.
2. **The procedure to trust for "is B better than A": the two-arm paired A/B**, `python tools_ab/ab.py A.py B.py --pairs 40`
   (160 pairs on four GPUs, AB / BA alternation inside every pair, each run the official eval's own `main()` in a fresh process;
   +- 0.0009 at 160 pairs, ~7 min). Its null holds: v76 against v77 (identical launches) reads -0.0000 +- 0.0009 with no workload
   beyond its CI and an order effect of -0.0007 +- 0.0009. Every keep decision should rest on it.
3. **Do not use the three-arm cross-process procedure of iteration 15 (`scratch_v15/arms.sh`) for a decision: it fails its own
   null.** Its fixed rotation gives each arm a fixed pattern of predecessors, and two identical arms then differ by about 0.001
   (NOTES_v16 1.1). The numbers it produced in iteration 15 (section 00.4 below, the "cross-process" rows) carry that uncertainty;
   the official-A/B rows there do not.
4. **Compare versions by paired runs, never by two `final_measure.sh` tables.** Per-process sd of the score within a GPU is 0.0038,
   and the session's kernel-argument pool odds move a 5-run mean on one GPU by about +- 0.002: v76 read 1.6006 in its own session
   and 1.5991 in this one.
5. For *why* a number moved, or for held-out shapes and rule switches: `bash scratch_final16/arms16.sh TAG ROUNDS A.py B.py ... [-- shapes]`
   (one fresh process per arm and round, **arm order a random permutation per round**, the die of the kernarg pool and of the
   caller's tensors probed after each timing) and `python scratch_final16/adj.py TAG --model raw|fe|feall`. Diagnostic only:
   always include the same file under a second name as a null arm (in `runs/leg1`, 39 of 40 null cells were inside their CI, all
   within +- 0.045 us), and confirm anything that would change the score with item 2.
6. A gain that exists only in the eval's workload order is a property of one allocation sequence: check with `SEEDED=1` (shuffled
   order, perturbed allocation history) before believing it.

### 000.7 Per family: now, the task-owned floor, what bounds the rest
"Now" = the 20-run means of 000.4. Floors as derived in FLOORS.md / NOTES_b1spec15 / UNDERFLOOR.md (unchanged by v77).

| family (eval workloads) | now, us | task-owned floor / bound | what bounds the rest |
|---|---|---|---|
| **P-LB** d_k3_b1, d_glm16_b1 | 13.30 / 13.29 | knee of the fixed launch cost 13.23 (13.04 / 13.08 / 13.20 / 13.60 by GPU) | at the floor. What is left is placement: 0.02-0.04 us from the kernel-argument pool's die (the caller's / runtime's allocation) |
| **P-HOLD** d_k3_b32, d_glm16_b64, d_glm8_b128 | 20.08 / 33.59 / 34.04 | stream floors 18.8 / 32.2 / 32.4 | non-state bytes, the odd-XCD write tail; kernarg pool's die 0.09-0.27 us |
| **P-MR** d_k3_b128, d_glm16_b256 | 45.20 / 98.15 | stream floors 41.8 / 91.6 | the first generation of blocks (cold start of 256 CUs); kernarg pool's die 0.09-0.14 us |
| **V16** s4_k3_b1, s6_glm16_b1, s8_k3_b1 | 14.39 / 15.12 / 15.33 | family floor 14.0 / 14.2 / 14.5 (entry 11.0 + two cold hops 0.9 + row hop 0.6 + S x 0.12 store issue + message 0.6 + out and write-back 0.4) | design-owned 0.4 / 0.9 / 0.8 us in pieces of 0.1-0.3 us (staging, younger-wave prologue contention, exchange pass, tail compute), each attacked and killed on its own; 0.15-0.22 us of each number is the kernarg pool's die |
| **S-1R / S-SB** s4_k3_b16, s6_glm16_b16, s6_glm8_b32, s8_k3_b16 | 22.05 / 30.94 / 31.15 / 30.73 | stream floors about 1 us below | prologue chain and store issue; XCC 3's write tail (converted by UQ where there are spare CUs; the two full grids have none); state pool phase -0.15 us on two of them (caller-owned) |
| **S-MR** s4_k3_b64, s6_glm16_b64, s8_k3_b64 | 54.69 / 91.87 / 90.72 | stream floors 54.8 / 92.9 / 90.5 | at the write-stream bound; the state pool's die phase is worth 0.36-0.39 us on the K3 two (caller-owned; lead 2) |

In score terms: the families at their floor (P-LB, S-MR) have nothing left; the others together hold perhaps +0.01, of which about
+0.0035 is the kernel-argument pool (not the kernel's to take, lead 1) and the rest is in pieces below 0.3 us that earlier
iterations priced and failed to convert.

### 000.8 Known limitations and residual risks
Correctness-relevant:
- **The per-XCC forms (X1, XF, UQ) are correct only under an exact round-robin dispatch over the 8 XCCs.** The kernels read
  `HW_REG_XCC_ID` to decide which blocks do the work; under any other mapping a head is skipped (its `out` columns, state and conv
  cache are not advanced) or computed twice, with no error. There is no run-time detection: the guard is host-side, before the
  launch (`_xcc_tab`, `_stream_ok`, one probe of six grids per device and process). Evidence that the contract holds here: 15 400
  checked replays and 770 probe dispatches beside 10 loading processes (T8), eight CU masks including "CU 0 only" (T4), five kinds
  of wrong register reading reach the symmetric path (port16). Not testable without root: a queue evicted and restored in the
  middle of a dispatch (inferred safe from the driver). Not detectable: a driver that changes its dispatch while the process runs.
  `X1_SPAN_US = 0` / `UQ_FRAC = 1` / `HOST_PROBES = False` switch the forms off; this trade buys +0.0065 of the score.
- **Red-team F3, judged out of contract after the hardware test**: a graph captured unmasked and launched on a CU-masked stream.
  Through torch it cannot happen (a graph replays on its capture stream); through `hipGraphLaunch` on the raw handle it is correct
  on this driver for every mask tried. A future driver that empties an XCC under a mask would make it wrong and nothing can run at
  replay to notice.
- **Mailbox healing after an aborted launch is best-effort on a shared GPU** (unchanged since v75; `reset_workspace()` after a device
  error). Unfaulted operation is unaffected.
- **Graphs of any custom kernel on a second device of one process fault on ROCm 7.2.0** once the kernel was captured on the first
  (the AITER baseline too). Platform; eager calls on several devices work and are tested (T3e, T5).
- **A capture before any eager call on the device** gets the symmetric forms and copy 0, and its placeholder / mailbox are
  allocated inside the capture (one fill node each). Correct, slower by what the rules are worth. Call `warmup()` first.

Side effects on a caller:
- **First eager call per device**: ~30 ms, 512 MiB of device memory for that time (`hipMalloc`, outside torch's accounting; if it
  cannot be had: copy 0), 4 MiB left; a cold build cache adds ~54 s of JIT, during which `PYTORCH_ROCM_ARCH` is rewritten in
  `os.environ`. `warmup()` / `HOST_PROBES = False` are the controls.
- **A process-wide lock** serialises the host part of `run()` across threads (microseconds; ~30 ms once, during the probes).
- **During the probes the calling thread is in HIP's thread-local capture mode** (restored afterwards). If the runtime library
  cannot be found in the process the mode is not switched; a first call beside another thread's capture then fails softly (copy 0,
  retried) but can still invalidate that capture, as in v76.
- `sys.modules["_kda_decode_state"]` holds the mailboxes and device verdicts for the life of the process (a reload keeps them).
- A failed round-robin probe is sticky for the process (by design).

Measurement / honesty:
- **The descriptor-placement gain (v76, +0.0026) is mostly a property of the cold regime the eval prescribes**: chosen against the
  other copy on the B = 1 spec shapes +0.10..+0.125 us with the eval's flush, +0.03..+0.04 in single replays without it, 0.00 +- 0.02
  in back-to-back chains (NOTES_final16 section 3). Never negative.
- **The kernel-argument pool's die is still a draw** (0.15-0.27 us on the B = 1 spec and hold workloads, per capture): the largest
  remaining variance term; a five-run mean on one GPU moves by +- 0.002 with a session's odds.
- Constants read off eval shapes (`X1_SPAN_US`: 16 blocks = d_glm16_b1; `UQ_FRAC`: 22 of 24 rows = s4 / s8_k3_b16) have held-out
  evidence on both sides of their boundaries (000.2), but they are measured numbers, not derived ones.
- Another toolchain than ROCm 7.2.0: a warning, the placement probe is off (copy 0), the asm checkers of step 8 must be re-run.

Red-team 16 findings accepted rather than fixed, with the reason (all others are fixed; NOTES_final16 sections 0-2):
- **F3** (unmasked graph on a masked stream): refuted on this driver, unreachable through torch, no zero-cost check exists at replay. Out of contract.
- **F7** (round-robin across queue eviction on an oversubscribed GPU): refuted as far as a test without root can (T8); inferred safe.
- **F8** (first-call cost): inherent to probing hardware; bounded (~30 ms, 512 MiB transient, fallback on allocation failure) and controllable.
- **F10** (the two descriptor windows differ in a second address term worth 0.04-0.08 us): confirmed; the proposed re-fit would make
  both copies share the term, so nothing was changed. Speed only (lead 3). The sentence "differ in bit 13 and nothing above" in
  `kernel_v15_a.py`'s docstring and in 00.1 below is wrong; the die claim it supports is right.
- **F13 (ii)** a failed round-robin probe stays failed for the process: by design. **(iv)** X1 masks the XCC id with 15, XF / UQ with 7: device
  code, and the probe rejects ids >= 8.
- A test, not the kernel: `scratch_v14/xcc_tests.py` now records the arguments of each launch; querying `hold_arg()` / `xv_arg()` after a
  call no longer tells what the call launched, because v77 resets the guard's verdict after every call. Tools that did that must record instead.

### 000.9 Untested leads, ranked
| # | lead | expected size | status / why not done |
|---|---|---|---|
| 1 | **Kernel-argument pool on the die of the late quad.** The runtime allocates the pool at `hipGraphInstantiate`, after `run()` returned; its die is worth 0.15-0.27 us on the B = 1 spec and hold workloads | **+0.003..+0.004**, and it removes the largest variance term | **The kernel-side form — holding dummy 132 KiB allocations alive so that the caller's next pool lands on a good piece — was ruled not permitted** (it steers an allocation made for the caller: PLACEMENT_RESEARCH section 4, confirmed for this iteration). Two argument blocks 8 KiB apart and a kernel without preload are dead by arithmetic (NOTES_v15 section 3). Open only to a caller (probe after `capture_end()`, instantiate again if on the wrong die: 16 launches of `kda_pp`) or to the runtime |
| 2 | State pool phase on S-MR K3 (s4 / s8_k3_b64): probe the pool's die phase at first use, choose which XCC group holds back | +0.0012 (0.36-0.39 us x 2 workloads) | in-kernel work; needs its own instantiation; the phase is caller-owned, so the rule must be a probe, not a constant. **Closed in iteration 18 (0000.3 levers 2-3): the sign belongs to the sample of slots and the GPU, not to the pool** |
| 3 | **Four descriptor windows (die x slow class)** — new, from T10: one window of every object carries a common-mode 0.04-0.08 us on cold loads (bit 14 with the piece's bits 19 / 26); the chosen copy has it in half of the processes | <= +0.001 (at most the cold-load term on the B = 1 workloads, half of the time) | needs four copies of every kernel: the max-ILP object grows to ~3 MiB and its buddy geometry changes (the layout check and `place_tests.py` depend on it). Re-fitting the pads into one 16 KiB block, as the review proposed, only makes both copies share the term |
| 4 | Phase 1's switch harness (`kernel_v16_a.py`, `kda_v16_k.cu`, `scratch_v16/`): ~20 compile-time switches on the V16 and spec kernels (store policies, row exchange, publish order, slot hop, token-step variants), each built and byte-diffed against v76 (`bld_*.log`), **none measured** | unknown; NOTES_b1spec15 prices the design-owned pieces of the B = 1 spec budget at 0.1-0.3 us each, all attacked before: expect <= +0.001 each | phase 1 stopped at its measurement step (its finding: `arms.sh` fails its null; measure these with `tools_ab/ab.py`, one switch per A/B). `scratch_v16/` holds the built arms and `NOTES_v16.md` the list |
| 5 | Re-take the B = 1 spec budget (NOTES_b1spec15) with the two placement covariates before attacking a row below 0.2 us | enabling | - |
| 6 | Per-XCC read-first hold on the state-first hold grids with a mask from the pool's address (NOTES_v14 brief 1) | +0.001 in one allocation order, 0 in the eval's | sign follows the caller's pool. **Closed in iteration 18 (0000.3 lever 1): +0.0003 in the eval's process with any mask containing XCC 3; its size follows the allocation history, no mask or address recovers it** |
| 7 | 17-24 blocks per XCC on three XCCs (12,6,1: -0.055 us with a limit of 24) | held-out only | the same limit loses on four other shapes; a per-m limit would be a fitted rule |

**Lead 4, screened (iteration 17 screen builder, `NOTES_v17.md`; no v78, `kernel.py` unchanged).** The harness was re-based onto v77
(`kernel_v17_a.py`: the same switches under v77's host layer) and 16 arms were measured with `tools_ab/ab.py` against the unmodified
v77, 160 pairs each, after a twin null of that procedure on v77 (-0.0002, CI [-0.0011, +0.0006]; resolution per workload 0.02 us =
+-0.00014 of score on each B = 1 spec workload, 0.03-0.08 us on B = 16 / 32 spec, 0.07-0.12 us on B = 64 spec). Worse: `rs` -0.0067
(the write-through mailbox reset is the fast form), `bors`, `tok1` -0.0019, `nopin0` -0.0008, `wsm0` -0.0006, `tn4off` -0.0003
(the shipped wave-offset and nt-tail rules are confirmed on their own workloads). Neutral within +-0.0004..0.0006: `outnt1`, `vnoa0`,
`vnoa3`, `dup1`, `norr2v0`, `ml4`, `spol16`. Real but below the commit line: `bo` (+0.00016, `s6_glm16_b1` only) and `vce1` (-0.03 us
at S = 6 / 8, +0.02..+0.03 us at S = 4); together and gated to long windows about **+0.0006 +- 0.0002**, which needs a cost model for
the gate (S = 5 / 7 held-out on both sides) and the full validation incl. 7b. `rowx` answered its question: swapping which waves own
which state rows moves `s8_k3_b64` by -0.31 us and `s6_glm16_b64` by +0.28 us, so lead 2 (pool phase on S-MR) can act through the row
map, with the sign taken from a probe of the pool, never from the shape. **Unscreened**: the switches iteration 16 did not build
(`PR_V16QKSPREAD`, `PR_V16BL`, `PR_V16LDS`, `PR_V16M32`, `PR_V16H32`, `PR_V16BETA`, `PR_V16HDUP`, `PR_V16RST`, `PR_V16WU0`; each a
budget row of <= 0.1 us), timing of any arm on held-out shapes (correctness only: 18 shapes per arm), and `arms16.sh`'s own null.

---

*(Header of the v76 hand-off, iteration 15, kept as written:)* Hand-off of the kernel at **v76** (`kernel.py` -> `kernel_v15_a.py`, a layer over private instances of v75's `kernel_v14_b.py` / v74's `kernel_v13_a.py`; builds of `kda_v15_a.cu`, which includes `kda_v14_b.cu` twice), iteration 15, placement builder. Written 2026-10-11. **Section 00 is v76** (what changed, the placement model and its guard, scores, A/B, validation, the measurement procedure to use from now on, risks, floors, leads). Section 0 is the v75 hand-off of iteration 14, sections 1-7 the v74 hand-off of iteration 13; statements that v76 changes are marked "v76". Iteration 15's log is `NOTES_v15.md`; its inputs are `PLACEMENT_RESEARCH.md` (placement researcher) and `NOTES_b1spec15.md` (B = 1 spec challenger: no new kernel, a phase budget and the family floor).

## 00. v76 (iteration 15)

### 00.1 What changed
One thing: **where the kernel descriptors are is chosen instead of drawn.** Every kernel, the cost model, the dispatch (identical
on 109 854 shapes) and the guard are v75's; the 136 kernel copies in the two new code objects are byte-identical to v75's kernels.

| piece | what it is | provenance | evidence |
|---|---|---|---|
| two copies | `kda_v15_a.cu` includes `kda_v14_b.cu` in namespaces `kc0` / `kc1` with filler kernels (64 B descriptor each) so that copy 1's descriptor table is exactly 8 KiB behind copy 0's, both inside the object's first buddy piece. One build per scheduler (default: 8 kernels per copy; max-ILP: 60): two code objects (296 KiB, 1552 KiB) instead of v75's four | two lines 8 KiB apart in one piece differ in address bit 13 and nothing above: always on different I/O dies (the die is a parity of bits 13, 21, 28: NOTES_v15 1.0) | `isa15.py`: 136 copies = v75's bytes; the probe reads opposite signs on the two windows in every process |
| `_layout()` | host-side check of the built extension (ELF inside `.hip_fatbin`): tables pairwise 8 KiB apart, each in one 8 KiB window, both in the first piece, >= 8 filler lines per window. Otherwise: warning, copy 0 | - | validation step 11 L, F |
| `_place(dev)` | replaces v75's `_upload()`: at the first eager call on a device both objects are uploaded and probed: one wave on XCC 1 and one on XCC 5 time 16 cold loads per window (72 launches per object, each behind a 512 MB fill of a buffer the extension takes with `hipMalloc` and frees before `_place()` returns). `_verdict()`: the copy whose window reads >= 0.03 us slower from XCC 1 while the other window reads >= 0.03 us faster | XCC 1 = `X1_XCC`, XCC 5 = the same position in the quad of the XCC that starts last (`XCC_START_US`); 0.03 us against a die difference of 0.08 us at 0.01 us resolution | wrong copy in 0 of 648 probes; no verdict in 1 of 120 first probes |
| fallbacks | capture before any eager call, unexpected layout, not an 8-XCC MI350X (`_xcc_tab()`), no memory for the flush buffer, unclear readings: **copy 0**, i.e. v75's situation (a draw). `FORCE_COPY` = 0 / 1 / "far" / "off" for tests | - | step 11 F: five fallback paths in fresh processes |

Measured and **not** in v76 (all in `NOTES_v15.md`): code objects padded to exactly 2 MiB (-0.0013: moves the caller's first torch
segment); kernels aligned to 4096 (0.00 +- 0.015 us); the X1 / XF kernels from v75's own builds (+0.02..+0.06 us); a probe without
a flush (wrong on 1 of 8 objects); anything for the kernel arguments (00.3).

### 00.2 The placement model (read before any A/B of two binaries; replaces 0.3)
What a dispatch reads before its first instruction, on every XCC: the kernel descriptor and the first 64 B of the kernel
arguments (the command processor's preload). MI350X interleaves 4 KiB pages over its two I/O dies; the die of a line is the
parity of physical address bits 13, 21 and 28 (a second term, +0.04 us for both quads, of bits 14, 19, 26; `scratch_v15/e1.py`). A
wave's cold load pays 0.08-0.12 us for the other die; the command processor's reads pay about twice that, and a dispatch ends on
the quad that starts last (XCCs 4-7).

Attribution on v75 (`drawmany.sh a1`: 600 fresh processes, 480 of them with a perturbed allocation history and shuffled workload
order, every object's page probed in the same process after each workload's timing; us between the two dies, negative =
faster on the die of XCCs 4-7; the full table with all nine items is NOTES_v15 1.1):

| workload | kernel arguments | descriptor | everything else the kernel owns |
|---|---|---|---|
| s4_k3_b1 / s8_k3_b1 / s6_glm16_b1 | -0.20 / -0.15 / -0.22 | -0.19 / -0.15 / -0.18 | code entry page < 0.02; mailbox die -0.02, slow bit +0.04 (s4 only) |
| d_k3_b32, d_glm16_b64, d_glm8_b128 | -0.09 / -0.09 / -0.13 | -0.13 / -0.07 / -0.08 | - |
| d_k3_b128, d_glm16_b256 | -0.09 / -0.14 | -0.07 / n.s. | - |
| s6_glm8_b32, s6_glm16_b16, s8_k3_b16 | -0.06 / -0.04 / -0.09 | -0.11 / -0.08 / -0.07 | - |
| d_k3_b1, d_glm16_b1 | -0.02 / -0.04 | -0.03 / -0.04 | - |
| s4_k3_b64, s8_k3_b64 (caller-owned: state pool phase +0.39 / +0.36) | n.s. | -0.13 / n.s. | - |

Both are drawn by the driver's buddy allocator from its small free pieces: the descriptor once per process and code object, the
argument pool at every `hipGraphInstantiate`. Their odds depend on the sizes of everything small the process loaded before and
on what earlier processes left: v75's max-ILP descriptors were on the fast die in 28 % of one session's processes (1 of 30 on
HIP 3) and in about two thirds of the next session's; the kernarg pool in 43 % / 81 % / 61 %. **This is the "session / build /
allocation term" of iterations 11-14** (the descriptor of the max-ILP object alone: +0.0049 of score per process).
v76 fixes the descriptor. The kernel arguments stay a draw (00.3).

### 00.3 The kernel arguments (not built)
Expected loss +0.0035 of score, same size as the descriptor. The pool is allocated by the runtime after `run()` has returned
inside the capture, the arguments sit at offset 0, and the die-sensitive read is the command processor's preload of line 0:
two argument blocks 8 KiB apart or a kernel without preload are dead by arithmetic; holding the "bad" 132 KiB pieces with dummy
graphs so that later captures find good ones only works by keeping dummy allocations alive to steer an allocation made for the
caller, which PLACEMENT_RESEARCH section 4 rules out. Table and numbers: NOTES_v15 section 3. It is the first lead of 00.8.

### 00.4 Final scores
Official eval (`python tools/eval_amd_kda_decode_mi350x.py`), **5 runs on each GPU**, the four in parallel, NUMA node 0
(`scratch_v15/final_measure.sh 5`; logs `scratch_v15/final/`; canary before and after every run: 13.16-13.76 us, no GPU shared).
All 20 runs 17/17 correct. v75 with the same procedure (iteration 14): 1.5963.

| GPU (HIP) | runs | mean_vs_base mean | range | v75 |
|---|---|---|---|---|
| 0 | 5 | 1.6109 | 1.6074 - 1.6151 | 1.6043 |
| 2 | 5 | 1.6045 | 1.5966 - 1.6085 | 1.6009 |
| 3 | 5 | 1.6071 | 1.6024 - 1.6125 | 1.6050 |
| 6 | 5 | 1.5800 | 1.5766 - 1.5845 | 1.5752 |
| all | 20 | **1.6006** | 1.5766 - 1.6151 | 1.5963 |

| workload | SOTA us | HIP 0 us | HIP 2 us | HIP 3 us | HIP 6 us | mean us | vs_base |
|---|---|---|---|---|---|---|---|
| d_k3_b1 | 16.28 | 13.11 | 13.20 | 13.14 | 13.63 | 13.27 | 1.227 |
| d_k3_b32 | 23.44 | 19.86 | 20.00 | 20.12 | 20.26 | 20.06 | 1.168 |
| d_k3_b128 | 54.44 | 44.97 | 45.13 | 45.22 | 45.40 | 45.18 | 1.205 |
| s4_k3_b1 | 25.04 | 14.31 | 14.25 | 14.33 | 14.59 | 14.37 | 1.743 |
| s4_k3_b16 | 39.88 | 21.78 | 21.94 | 22.00 | 22.21 | 21.98 | 1.814 |
| s4_k3_b64 | 87.08 | 54.51 | 54.54 | 54.34 | 54.94 | 54.58 | 1.595 |
| s8_k3_b1 | 32.48 | 15.20 | 15.13 | 15.20 | 15.51 | 15.26 | 2.128 |
| s8_k3_b16 | 58.36 | 30.65 | 30.81 | 30.26 | 30.98 | 30.67 | 1.903 |
| s8_k3_b64 | 147.52 | 90.25 | 90.95 | 90.66 | 91.27 | 90.78 | 1.625 |
| d_glm16_b1 | 18.80 | 13.12 | 13.14 | 13.17 | 13.62 | 13.26 | 1.418 |
| d_glm16_b64 | 51.04 | 33.39 | 33.64 | 33.54 | 33.86 | 33.61 | 1.519 |
| d_glm16_b256 | 135.80 | 98.04 | 98.09 | 98.20 | 98.46 | 98.20 | 1.383 |
| s6_glm16_b1 | 28.68 | 15.09 | 14.98 | 15.02 | 15.41 | 15.13 | 1.896 |
| s6_glm16_b16 | 53.68 | 30.44 | 30.83 | 30.67 | 31.50 | 30.86 | 1.739 |
| s6_glm16_b64 | 148.28 | 90.66 | 92.68 | 91.94 | 92.94 | 92.06 | 1.611 |
| d_glm8_b128 | 51.48 | 33.82 | 33.83 | 33.85 | 34.23 | 33.93 | 1.517 |
| s6_glm8_b32 | 53.60 | 30.99 | 31.16 | 30.93 | 31.81 | 31.22 | 1.717 |

**Stack A/B v75 -> v76** (tables in `scratch_v15/runs/`):

| procedure | n | delta | per GPU (HIP 0 / 2 / 3 / 6) |
|---|---|---|---|
| official: `tools_ab/ab.py kernel_v14_b.py kernel_v15_a.py --pairs 40` (final form) | 160 pairs | **+0.0026 [+0.0017, +0.0035]** | +0.0040 / +0.0033 / +0.0030 / +0.0002 |
| official, `--pairs 60`, the 8-reading probe | 240 pairs | +0.0030 [+0.0023, +0.0037] | +0.0035 / +0.0030 / +0.0027 / +0.0028 |
| cross-process, eval order (`scratch_v15/arms.sh f1`: one fresh process per arm and round, probes in every process) | 160 rounds | **+0.0018 [+0.0011, +0.0026]**; kernarg-adjusted +0.0020 +- 0.0006 | +0.0021 / +0.0020 / +0.0012 / +0.0021 |
| cross-process, shuffled workload order + perturbed allocation history + unrelated allocations on fresh streams between workloads (`SEEDED=1 arms.sh`) | 144 + 120 rounds | **+0.0025 [+0.0015, +0.0035]**, **+0.0025 [+0.0015, +0.0034]** | +0.0019 / +0.0030 / +0.0032 / +0.0020; +0.0031 / +0.0006 / +0.0039 / +0.0022 |
| the chosen copy against the other copy of the same binary | 3 x 144-160 rounds | +0.0066 / +0.0076 / +0.0058 | B = 1 spec +0.16..+0.19 us, d_glm16_b64 / d_glm8_b128 +0.17, 4 of 4 GPUs each |

Official A/B per workload (160 pairs): s4_k3_b1 -0.08 +- 0.02 (3/4 GPUs beyond CI), s8_k3_b1 -0.08 +- 0.02 (3/4), s6_glm16_b1 -0.09 +- 0.03
(4/4), d_glm8_b128 -0.09 +- 0.04, d_glm16_b64 -0.07 +- 0.03 (3/4), d_glm16_b256 -0.06, d_k3_b128 / d_k3_b32 -0.04, d_glm16_b1 -0.03 (4/4);
s4_k3_b16 +0.03 +- 0.03, s8_k3_b16 +0.07 +- 0.09 (HIP 6 alone +0.35 +- 0.16), s4_k3_b64 +0.02 +- 0.06: none slower beyond its pooled CI.
**Deviations from the keep rule, stated**: (i) in the cross-process harness in the eval's order d_k3_b1 reads +0.02 +- 0.013 us
(four runs: +0.015..+0.025; official A/B -0.01 +- 0.01; shuffled order -0.02 / -0.01), s6_glm16_b16 +0.06 +- 0.04 in one run of five;
(ii) held-out (`arms.sh ho1`, 22 shapes, 112 rounds): 13 faster beyond CI, two slower: 12,2,1 +0.037 +- 0.020 (the two-XCC XF kernel:
+0.014..+0.042 in four sessions) and 16,12,5 +0.045 +- 0.033. Both follow v75's own small objects being drawn again per
session (NOTES_v15 2.6: two hypotheses built and killed).
**Held-out (unscored)**: `--heldout` 11/11, 1.483 against live SOTA (v75 1.482).

### 00.5 Validation
`bash scratch_final/validate_all.sh v76a`: **PASS**, all 13 steps. New: **step 11** `scratch_v15/place_tests.py` (25 checks: layout;
the verdict rule on synthetic readings; the probe: a verdict on both builds, nothing left allocated, 20 repeats never the
other copy; 20 shapes x copy 0 / 1 / choice / far; repeated capture alternating copies; four graphs replayed on four threads;
five fallback paths in fresh processes incl. the flush buffer's allocation failure and a capture before any eager call; 25
workloads in three shuffled orders with unrelated allocations and fresh streams in between) and **step 12** `scratch_v15/isa15.py`
(kernel copies byte-identical to v75's). Step 8 audits `kda_v14_b.cu` (`_SRC`); step 9's reference is unchanged.

### 00.6 The measurement procedure from now on
1. **A/B of two binaries: cross-process with probes.** `scratch_v15/arms.sh TAG ROUNDS A.py B.py ...` runs one fresh process per arm
   and round on the four GPUs (arm order rotated, canary last) through `draw.py`, which after each workload's timing probes the
   die of the graph's kernarg pool and of the caller's tensors. `armrep.py TAG` = raw paired deltas; `armadj.py TAG` = the same
   with the kernarg pool's die as a covariate. Quote both. Three runs of the same comparison tonight: raw +0.0010 / +0.0014 /
   +0.0018, adjusted +0.0022 / +0.0021 / +0.0020 (+- 0.0006 at 160 rounds).
2. **Always carry the shuffled arm** (`SEEDED=1`): a gain that exists only in the eval's order is a property of one allocation
   sequence. A new binary's `FORCE_COPY`-style "off" arm replaces v14's "rules-off arm".
3. The official `tools_ab/ab.py` stays the last word for the score (+- 0.0009 at 160 pairs) but cannot say why a number moved.
4. Per-process sd of the score within a GPU: v75 0.0044 -> **v76 0.0038** (five runs; the descriptor's coin, sd 0.0023, is gone).
   What is left: the kernarg pool's die (as large) and replay noise of the large workloads. A five-run mean on one GPU can still
   move by about +- 0.002 with the session's kernarg odds (per GPU between 2 and 30 of 30 captures on the fast die): compare versions by paired runs, not by two `final_measure.sh` tables.
5. Canary as before (`scratch_v14/canary.py`, 14.3 us; HIP 6 reads 13.6-13.8 clean).

### 00.7 Risks added or changed by v76
- **The choice rests on one measured law** (descriptor near the quad that starts last is faster for every kernel class): MI350X
  only, behind `_xcc_tab()`. Wrong law on another part = the other copy = what v75 does half of the time; never a wrong result
  (both copies are the same instructions).
- **`_place()` allocates 512 MB of device memory for a moment** at the first eager call on a device (`hipMalloc`, freed before
  it returns; on failure: copy 0). It adds ~150 launches and their syncs to that first call (as v75's `_upload()` and probe, it
  syncs once). A capture that precedes every eager call gets copy 0 and no probe.
- **Layout by filler count.** `_PADS` (0 / 120 / 32 and 1 / 68 / 32 filler kernels) was fitted for ROCm 7.2.0's metadata sizes; another
  toolchain or a changed kernel count moves the tables, `_layout()` then refuses (warning, copy 0). Re-fit with
  `scratch_v15/cc.sh` (CPU only, 20-35 s per try).
- The code objects are 296 KiB + 1552 KiB (v75: 824 KiB in four); the max-ILP one is above ROCr's 1 MiB memcpy-upload limit (it
  is uploaded by the runtime's DMA path: no effect seen in any test).
- Private module instances: `kernel_v15_a` loads its own copies of `kernel_v14_b.py` / `kernel_v13_a.py` (so that redirecting their
  build hooks does not change those modules for other code in the process). Tools reach v75's / v74's switches through
  attribute fall-through, as before.
- Unchanged from v75: the round-robin guard, mailbox healing on a shared GPU, the allocation term for the caller's own pools.

### 00.8 Floors and leads after iteration 15
| family (eval workloads) | now, us (00.4) | floor / bound | what bounds the rest |
|---|---|---|---|
| **P-LB** d_k3_b1, d_glm16_b1 | 13.27 / 13.26 | knee of the fixed cost 13.23 (13.04 / 13.08 / 13.20 / 13.60 by GPU) | at the floor; placement is worth 0.03-0.04 us here (descriptor, taken) + 0.02-0.04 (kernarg pool, not the kernel's) |
| **P-HOLD** d_k3_b32, d_glm16_b64, d_glm8_b128 | 20.06 / 33.61 / 33.93 | stream floors 18.8 / 32.2 / 32.4 | non-state bytes; odd-XCD tail; **kernarg pool's die 0.09-0.27 us** (new) |
| **P-MR** d_k3_b128, d_glm16_b256 | 45.18 / 98.20 | 41.8 / 91.6 | first generation; kernarg pool's die 0.09-0.14 us |
| **V16** s4_k3_b1, s6_glm16_b1, s8_k3_b1 | 14.37 / 15.13 / 15.26 | **family floor 14.0 / 14.2 / 14.5** (NOTES_b1spec15 3.8: entry 11.0 + two cold hops 0.9 + row hop 0.6 + S x 0.12 store issue + message 0.6 + out and write-back 0.4) | the challenger's budget: 0.7 us + S x 0.12 us are the task's; design-owned 0.4 / 0.9 / 0.8 us in pieces of 0.1-0.3 us (staging 0.1-0.2, younger-wave prologue contention, exchange pass 0.06-0.2, tail compute 0.25-0.46), each attacked and killed in its own right (eight organisations priced, seven skeletons killed). **New: 0.15-0.22 us of every one of these numbers is the kernarg pool's die** (the entry cut 11.0 us is an average over it): with the pool on the late quad's die v76 is at about 14.27 / 15.02 / 15.18 (the mean minus half of the pool's column in 00.2) |
| **S-1R / S-SB** s4_k3_b16, s6_glm16_b16, s6_glm8_b32, s8_k3_b16 | 21.98 / 30.86 / 31.22 / 30.67 | stream-floor gaps ~1 us | prologue chain, store issue; XCC 3's tail converted in v75; state pool phase -0.15 us on two of them (caller-owned) |
| **S-MR** s4_k3_b64, s6_glm16_b64, s8_k3_b64 | 54.58 / 92.06 / 90.78 | stream floors 54.8 / 92.9 / 90.5 | write-stream-bound; **state pool's die phase 0.36-0.39 us on the K3 two** (new, caller-owned; lead 2) |

Untested leads, ranked (NOTES_v15 section 5 has the detail):
1. **Kernarg pool on the late quad's die: +0.003..+0.004** and the largest remaining variance term. Closed from inside the kernel;
   open to a caller (probe after `capture_end()`, instantiate again if near XCC 1: 16 launches of `kda_pp`) or to an operator
   ruling that the kernel may hold dummy 132 KiB pieces (then a few lines on top of `_place()`).
2. **State pool phase on S-MR K3: 0.36-0.39 us x 2 workloads = +0.0012**: the kernel can probe a pool's phase at first use and
   choose which XCC group holds back (NOTES_v14 lead 2, now with a measured size and the probe to do it). In-kernel work.
3. Re-take the B = 1 spec budget of NOTES_b1spec15 with the two placement covariates before attacking a row below 0.2 us.
4. Per-mask descriptor rule for the two- / three-XCC XF kernels: 0.01 us, held-out only. Not worth a rule.

---

## 0. v75 (iteration 14)

### 0.1 What changed
v75 = v74 + two rules of the per-XCC model `event = F + max over XCC x of (S_x + W_x)` (ASYMMETRY section 1), one guard, and one
loading rule. Everything v74 can launch is launched from **v74's own two builds** (same binaries, same code objects).

| piece | what it is | cost model / constants (provenance) | evidence (us against v74) |
|---|---|---|---|
| **`xcc_items()` (U)** | single-round spec whole-state grid: physical XCC 3 (slowest write path) skips its items in the last K rows of 8 workgroups; blocks launched behind the grid on XCC 1, then XCC 0, take them. Own instantiations `kda_main<.., W1 or SB, XV = 2>` (max-ILP per-XCC build): the v74 kernel + one scalar asm block at entry (5 instructions on the natural path) | K = round(0.08 N / 8) (ASYMMETRY B.6: 22 of 24); receivers = the two first-started XCCs outside XCC 3's write-path pair (S_x table); gate = write-bound in the sense of `wave_offset()` (existing constants) and the launched grid within one round | **s4_k3_b16 -0.11 +- 0.03, s8_k3_b16 -0.29 +- 0.07** (200 pairs, each beyond CI on 4 of 4 GPUs); leave-one-out +0.0016 [+0.0006, +0.0026]; held-out: 9 of 17 shapes faster on >= 3 of 4 GPUs (-0.14..-0.41), none slower |
| **`xcc_fill()` (X)** | v74's X1 generalised: a plain latency-bound grid runs on the m XCCs that start first (1; 1,0; 1,0,3; 1,0,3,7), grid `8 ceil(H / m) x B`, the block on the r-th XCC of the mask does head `(blockIdx.x / 8) m + r`. m = 1 is v74's X1 kernel from v74's build; m = 2..4 are `kda_main<.., X1, XV = mask>` (default per-XCC build) | smallest m for which the busiest XCC carries <= 16 blocks (`B ceil(H / m) x 128 KB / 1.0e6 <= 2.1 us`, v74's bound per XCC); `XF_MAX = 4` (m = 5 / 6 measured: -0.16..+0.06, not kept) | held-out only (17-64 blocks): -0.20..-0.28 on six m = 2 / 3 shapes (4 of 4 GPUs), -0.14 / -0.15 on two m = 4 shapes (3 of 4), 12,5,1 -0.07 +- 0.03; h_d_glm8_b7 14.2 us |
| **guard** | every per-XCC form (v74's X1 included) needs: 8 XCCs by `hipDeviceAttributeNumberOfXccs`, 256 CUs, compiled XCC constants equal to the launcher's tables (`_xcc_tab()`); the current stream without a CU mask (`hipExtStreamGetCUMask`, valid inside a capture); the dispatch probed to be an exact round-robin on six grids of the forms in use, once per device outside any capture (`_x1_probe()`, pure checker `_rr_exact()`), sticky and loud when it fails, `xcc_reprobe()` on request. Otherwise v74's symmetric grid | HW_RESEARCH card B: the round-robin is a driver / firmware contract (MQD chunk size 1), its origin is not (per queue: the kernels use `HW_REG_XCC_ID`), the chunk size is a driver constant | `scratch_v14/xcc_tests.py` (validation step 10) |
| **`_upload()`** | at the first eager call on a device the code objects of all four builds are uploaded (one 8-workgroup launch of the probe kernel from each), v74's two first | device memory the runtime takes later, between a caller's pool allocations, moves memory-bound workloads on unchanged binaries; taken ahead of v74's max-ILP build it moves that build's code (table in 0.3) | lazy: d_glm16_b64 +0.09 +- 0.03, d_glm8_b128 +0.17 +- 0.06; ahead of v74's: s4_k3_b1 +0.14 +- 0.05; as built: 13 of 15 v74-kernel workloads within +- 0.03 |

Built and **not** in v75 (all in `NOTES_v14.md`): the per-XCC read-first hold on the state-first hold grids (ASYMMETRY brief 1:
-0.13 / -0.14 us standalone at 0.25 us of skew, nothing in the eval's own process: its sign follows where the state pool sits;
form kept in `kernel_v14_a.py` / `kda_v14_a.cu`); `xcc_fill()` with 5 / 6 XCCs; UQ on full 256-block grids (+4.8..+6.0 us: a second round,
also with a 64-VGPR build); sub-item traffic off XCC 3 in the S-MR tail (timing-only: -0.58 us on s8_k3_b64, below the 1 us line);
fewer rows on late V16 siblings (0.00 +- 0.06); cached beta in the tail generation (0.00); die-aware mailbox rows (0.00 +- 0.03).

### 0.2 Final scores
Official eval (`python tools/eval_amd_kda_decode_mi350x.py`), **5 runs on each GPU**, the four in parallel, NUMA node 0
(`scratch_v14/final_measure.sh 5`; logs `scratch_v14/final/`; a canary, v74's 12,1,1, before and after every run: 13.1-13.7 us, i.e. no
GPU was shared during these runs). All 20 runs 17/17 correct. v74 with the same procedure (iteration 13): 1.5939.

| GPU (HIP) | runs | mean_vs_base mean | range |
|---|---|---|---|
| 0 | 5 | 1.6043 | 1.5981 - 1.6107 |
| 2 | 5 | 1.6009 | 1.5953 - 1.6053 |
| 3 | 5 | 1.6050 | 1.5985 - 1.6110 |
| 6 | 5 | 1.5752 | 1.5737 - 1.5768 |
| all | 20 | **1.5963** | 1.5737 - 1.6110 |

| workload | SOTA us | HIP 0 us | HIP 2 us | HIP 3 us | HIP 6 us | mean us | vs_base |
|---|---|---|---|---|---|---|---|
| d_k3_b1 | 16.28 | 13.18 | 13.19 | 13.13 | 13.65 | 13.29 | 1.225 |
| d_k3_b32 | 23.44 | 20.05 | 20.03 | 19.99 | 20.44 | 20.13 | 1.165 |
| d_k3_b128 | 54.44 | 45.23 | 45.18 | 45.08 | 45.62 | 45.28 | 1.202 |
| s4_k3_b1 | 25.04 | 14.38 | 14.36 | 14.28 | 14.70 | 14.43 | 1.735 |
| s4_k3_b16 | 39.88 | 21.70 | 21.91 | 21.83 | 22.35 | 21.95 | 1.817 |
| s4_k3_b64 | 87.08 | 54.31 | 54.58 | 54.73 | 55.30 | 54.73 | 1.591 |
| s8_k3_b1 | 32.48 | 15.27 | 15.31 | 15.30 | 15.58 | 15.37 | 2.114 |
| s8_k3_b16 | 58.36 | 31.04 | 30.70 | 30.74 | 31.09 | 30.89 | 1.889 |
| s8_k3_b64 | 147.52 | 89.90 | 90.31 | 90.48 | 91.05 | 90.44 | 1.631 |
| d_glm16_b1 | 18.80 | 13.23 | 13.25 | 13.16 | 13.65 | 13.32 | 1.411 |
| d_glm16_b64 | 51.04 | 33.47 | 33.80 | 33.66 | 34.08 | 33.75 | 1.512 |
| d_glm16_b256 | 135.80 | 98.25 | 97.98 | 98.11 | 98.52 | 98.22 | 1.383 |
| s6_glm16_b1 | 28.68 | 15.10 | 15.07 | 15.18 | 15.43 | 15.20 | 1.887 |
| s6_glm16_b16 | 53.68 | 30.81 | 30.92 | 30.76 | 31.53 | 31.00 | 1.731 |
| s6_glm16_b64 | 148.28 | 90.57 | 92.48 | 91.11 | 92.91 | 91.77 | 1.616 |
| d_glm8_b128 | 51.48 | 34.10 | 34.15 | 33.86 | 34.44 | 34.14 | 1.508 |
| s6_glm8_b32 | 53.60 | 31.23 | 31.08 | 31.01 | 31.63 | 31.24 | 1.716 |

all runs passed 17/17

**Paired A/B v74 -> v75, official procedure** (`tools_ab/ab.py kernel_v13_a.py kernel_v14_b.py --pairs 50`: 200 pairs, 50 per GPU, full 17;
`scratch_v14/runs/ab_v74_v75_4gpu.txt`):

    mean_vs_base A 1.5934 (sd 0.0138)  B 1.5946 (sd 0.0137)
    delta B-A = +0.0011  95% CI [+0.0004, +0.0019]  (paired sd 0.0054, n=200)
    per GPU: HIP0 +0.0005 (n=50)  HIP2 +0.0008 (n=50)  HIP3 +0.0018 (n=50)  HIP6 +0.0014 (n=50)

s4_k3_b16 22.12 -> 22.01 (-0.11 +- 0.03; per GPU -0.08 / -0.08 / -0.10 / -0.19), s8_k3_b16 31.16 -> 30.87 (-0.29 +- 0.07; -0.19 / -0.21 / -0.29 / -0.49): +0.0016.
On v74's binaries: d_k3_b128 -0.05 +- 0.03; **s8_k3_b1 +0.04 +- 0.02** (HIP 6 +0.12 +- 0.05) and **d_k3_b1 +0.02 +- 0.01** (HIP 6 +0.04 +- 0.02): -0.0004,
the price of `_upload()` (v74's max-ILP code object enters device memory at the first call instead of at the 4th workload; the
B = 1 spec kernels move per GPU in both directions, s4_k3_b1 -0.06 / +0.01 / -0.08 / +0.08); the other 12 within +- 0.03 us.
Leave-one-out of U (`ab.py scratch_v14/k_bnoU.py kernel_v14_b.py --pairs 30`, 120 pairs, four GPUs): **+0.0016 [+0.0006, +0.0026]**.
On HIP 3 alone, earlier, while another user's job held the other three GPUs: +0.0020 [+0.0010, +0.0031] (110 pairs).

**Held-out (unscored)**: `--heldout` on HIP 3: 11/11, mean 1.482 against live SOTA (v74: 1.481 / 1.482); h_d_glm8_b7 (an `xcc_fill()`
shape) 14.16 us, 1.387. Four-GPU sweep of 35 shapes around every rule boundary (`scratch_v14/runs/hv4.txt`): 17 shapes faster
on >= 3 of 4 GPUs, none slower beyond its pooled CI.

### 0.3 The allocation term (read before any A/B of two binaries) (v76: named and partly removed, section 00.2)
Identical instructions, different process history, official procedure, HIP 3 (us +- 95 % against v74):

| the per-XCC builds' code objects are uploaded | pairs | full 17 | d_glm16_b64 | d_glm8_b128 | s4_k3_b1 |
|---|---|---|---|---|---|
| lazily (the max-ILP one at the 5th workload) | 100 | +0.0002 [-0.0008, +0.0012] | **+0.09 +- 0.03** | **+0.17 +- 0.06** | 0.00 |
| at the first eager call, ahead of v74's max-ILP build | 50 | -0.0005 [-0.0018, +0.0007] | +0.03 | +0.02 | **+0.14 +- 0.05** |
| at the first eager call, v74's builds first (`_upload()`, v75) | 110 | +0.0020 [+0.0010, +0.0031] | 0.00 | +0.11 +- 0.05 (four GPUs: +0.01 +- 0.04) | -0.13 +- 0.04 (four GPUs: -0.01 +- 0.02) |

A one-source build of everything (`kernel_v14_a.py`, every reachable kernel instruction-identical, every rule off) is **-0.0018
[-0.0029, -0.0008]** against v74 (110 pairs, three GPUs): the same effect, and the "session / build term" of section 5. Where the
runtime puts a build's code and where that pushes the caller's pools is worth +- 0.1-0.2 us on single workloads. One physical
law behind it was measured (`scratch_v14/cprobe/`): a cold load costs 0.32 / 0.40 / 0.36 / 0.44 us by address bits 13 and 14, swapped
between the two XCC quads (the NPS1 page interleave over the two I/O dies, HW_RESEARCH card C).

### 0.4 Validation
`bash scratch_final/validate_all.sh v75c`: **PASS**, all 11 steps. Changes to the script: step 10 is `scratch_v14/xcc_tests.py` (supersedes
`scratch_v13/x1_tests.py`: checker on 14 synthetic violations; host model of the XF / UQ index maps on every accepted shape x 8
origins; 59 shapes on both sides of every rule boundary x 72 checked replays from 12 fresh streams; repeated capture; four
concurrent graphs; capture before any eager call; 3600 probe dispatches; fault injection of a failing and a late-failing probe);
step 9 compares `scratch_v14/dispatch_dump.py` (launch arguments incl. the per-XCC argument) with `val/disp_ref.txt` (v74's tables:
`val/disp_ref_v74.txt`, and `val/disp_ref_v74_xv.txt` in the new format: 119 XF rows, 2522 UQ rows, 50 X1 rows with v74's launch);
step 5 stresses five more shapes; **step 7b runs after the parallel steps on a GPU whose canary is clean** and fails if none is.
Step 8 compiles `kda_v14_b.cu` (v74's kernels in it are instruction-identical to v74's and are not the ones launched).

### 0.5 Risks added or changed by v75
- **`xcc_items()` and `xcc_fill()` are correct only under the exact round-robin** (a moved or masked item would run twice or not at
  all: wrong `out` / state for that head, no crash), like X1. Guard: 0.1. Not covered: a queue evicted and restored in the middle
  of a dispatch (inferred safe from the driver, not testable without root); a graph captured on an unmasked stream and replayed
  on a CU-masked one. `kernel_v14_b.UQ_FRAC = 1.0` / `XF_MAX = 1` / `X1_SPAN_US = 0` switch the forms off (v74's dispatch).
- **Mailbox healing needs an exclusive GPU (found in this run, holds for v74 too).** With another process's workgroups on the
  GPU, `poison_rate.py` shows a stale mailbox word consumed in ~35-40 % of the faulted calls (v74 138 of 400, v75 1671 of 4200;
  exclusive GPU: 0 of 4200). The best-effort reset of v70 / v73 relies on the sibling blocks of a head starting together.
  Unfaulted operation is unaffected (every other validation step passes on a shared GPU); after a device error on a shared GPU
  call `reset_workspace()`.
- **The allocation term** (0.3): the score of any new binary carries +- 0.001-0.002 that is not in its instructions. v75's +0.0011
  is +0.0016 from the two workloads that run new kernels and -0.0004 from two B = 1 workloads on unchanged kernels.
- **`_upload()` syncs once**, at the first eager call on a device (as the probe always did). A capture that precedes every eager
  call loads the builds lazily and takes the symmetric grids.
- Three more **device-specific tables** (`XCC_START_US`, `XCC_WRITE_RATIO`, the compiled `UQ_XCC` / `UQ_RM` / XF masks): MI350X only,
  checked by `_xcc_tab()`.

### 0.6 Floors and leads after iteration 14
| family | v74 statement | iteration 14 |
|---|---|---|
| P-LB plain grids of 17-64 blocks (held-out) | "could use two-XCC masks: held-out only" | **converted**: -0.14..-0.28 us (`xcc_fill()`); 65-96 blocks: the fifth / sixth XCC start too late (0.00 +- 0.06) |
| S-1R / S-SB with spare CUs | "no per-XCC lever: broken, not converted" (v13 item 3) | **converted**: -0.11 / -0.29 us on s4_k3_b16 / s8_k3_b16 (`xcc_items()`); on the 256-block grids (s6_glm16_b16, s6_glm8_b32) a moved item is a second round (+4.8..+6.0 us), also with a 64-VGPR build: holds |
| P-HOLD state-first | stream floors 32.2 / 32.4 | per-XCC hold skew is real in one allocation order (-0.13 / -0.14) and absent in the eval's: **not converted**; the hold grids follow their pool's place by +- 0.1-0.2 us |
| S-MR | at the stream floor; pair-{2,3} tail not recoverable with whole items | sub-item move off XCC 3, timing-only: -0.58 us (s8), ~0 against v74 (s6): **holds** |
| V16 | gated by the latest sibling | a late sibling with no state traffic at all ends no earlier (0.00 +- 0.06): **holds**; mailbox rows by die: 0.00 +- 0.03 |
| every B = 1 workload | fixed cost + cold hops | new: the place of the kernel's code is worth 0.1-0.14 us (s4_k3_b1 14.3 / 14.5 us between sessions of v74 itself) |

Most valuable untested lead: **choose where a kernel's code lands** (NOTES_v14 "leads" 1). A cold hop costs 0.32-0.44 us by 16 KiB of
address and by quad; the five B = 1 workloads carry 22e-3 per us; placement effects of 0.1-0.14 us on identical instructions were
measured three times today. Then: the hold skew with a mask from the pool's address quarter; an append-only build.

---

v74 = v73 (iteration 12, finisher: `kernel_v12f_d.py`) + **X1** (the plain latency-bound kernel on the XCC that starts first); every other
kernel is instruction-identical to v73, so everything below that is not marked "v74" / "iteration 13" is the v73 hand-off, measured in
iteration 12 (working log `NOTES_final.md`). Iteration 13's log is `NOTES_v13.md`; its inputs are `UNDERFLOOR.md` (floor breaker) and
`RUNTIME_RESEARCH.md` (runtime researcher).

## 1. What the kernel is

(v75: two more builds, of `kda_v14_b.cu` with `-DKDA_XONLY`, carry the per-XCC kernels; section 0.1.) One HIP C++ source (`kda_v13_a.cu`, ~2600 lines), JIT-built twice by `kernel_v13_a.py` (`torch.utils.cpp_extension.load`, cached by a digest of the
source + ROCm version): the **default build** carries the 5 plain-decode kernels (and the 19-instruction XCC probe), the **max-ILP build**
(`-mllvm -amdgpu-sched-strategy=max-ilp -DKDA_SPEC_ONLY`) the spec kernels. `run()` validates the inputs, evaluates a cost model on
`(B, H, S, CUs, compiled VGPR counts)` and launches **exactly one kernel**; S > 8 and malformed-but-legal layouts (strided index
tensors) go to the vendored AITER dispatcher (`kernel_aiter_fallback.py`). No host sync, no data-dependent host control flow, no
per-call allocation: capturable in a HIP graph.

Two kernel templates, one block per (sequence n, head h) or per V-slice of it:

| family | instantiation | what a block does | used for |
|---|---|---|---|
| **`kda_main`** whole-state | `kda_main<128, S, spec, NT, PF, NX, SB, W1>` | all 128 state rows of one (n, h): per-thread prologue loads (conv taps, gate, beta, weights), conv + SiLU + L2 norms + per-token scalars into LDS, then S token steps (h.k / h.q row dots, delta-rule update, one fp32 state store per token), gated RMSNorm epilogue, conv-cache commit | every plain grid; spec grids of more than 64 (n, h) pairs |
| **`kda_vec`** V-split | `kda_vec<VB = 32 or 16, S, spec, 512, LATE>` (+ one exchange wave: 576 threads) | 128 / VB blocks per (n, h), each owns VB state rows; **staged prologue** (16 B chunks -> LDS, one wave per token); token loop on 16 x VB threads; the RMSNorm sum of squares is assembled across the sibling blocks through **mailboxes** (each block resets the words it will publish, publishes its partial sums to every sibling, polls its own mailbox, resets it; one mailbox per captured call / per eager stream) | latency-bound spec grids: B*H <= 64 (32 rows), B*H <= 16 and S >= 3 (16 rows) |

`kda_main` variants (template flags; each is its own instantiation so that grids that do not use a feature run unchanged code):

| class | instantiation | eval workloads |
|---|---|---|
| P-LB | plain, 1024 threads, prologue-first (PF); **v74: <= 16 blocks as X1** (grid 8 H x B, only the blocks on physical XCC 1 work) | d_k3_b1, d_glm16_b1 (both X1) |
| P-HOLD | plain, 256 threads (4 blocks per CU co-resident), PF or state-first, **read-first hold** | d_k3_b32 (PF); d_glm16_b64, d_glm8_b128 (state-first) |
| P-MR | plain, 1024 threads, PF + next-round prefetch (NX), first-generation stagger, batched stores | d_k3_b128, d_glm16_b256 |
| V16 | `kda_vec<16>` (ready flags instead of the prologue barrier for S > 4, LDS lookahead) | s4_k3_b1, s8_k3_b1, s6_glm16_b1 |
| V32 | `kda_vec<32>` | none (held-out: 16 < B*H <= 64, or B*H <= 16 at S = 2) |
| S-1R | spec, 1024 threads, PF (`CG_ML` scalar-base loads), optionally W1 (wave offset) | s4_k3_b16, s6_glm16_b16, s6_glm8_b32 (all three W1) |
| S-SB | spec, 1024 threads, PF, state rows behind a block barrier | s8_k3_b16 |
| S-MR | spec, 1024 threads, PF + NX, WS / WM wave offset, late conv commit (S >= 7), nontemporal tail (TN) | s4_k3_b64, s6_glm16_b64, s8_k3_b64 |

### Dispatch rules (all in `kernel_v12f_d.py`; inputs are B, H, S, the device's CU count and compiled VGPR counts — no shape table)

| rule (function) | cost model | mechanism | evidence |
|---|---|---|---|
| V-split (`choose`) | split V while the split grid stays within the CUs: 32-row blocks if `4 B H <= CUs`, 16-row if `16 B H <= CUs` and `S x 0.14 us > 0.31 us` | a block streams (S + 1) x 64 KB through one CU; with idle CUs, more blocks = more memory-level parallelism. The 16-row step halves the token work per SIMD (0.14 us / token) for a fixed cost of twice the blocks (0.31 us) | D02 +0.060, D01 +0.011; boundary at 160 pairs on both sides (NOTES_final 4.1) |
| threads (`choose`) | 1024 threads if spec, or blocks <= CUs, or blocks > CUs x (256-thread blocks per CU, from the VGPR count); else 256 threads | latency-bound and multi-round grids want 4x the MLP per state; fully co-resident plain grids read all states in one phase | NOTES_bw2 P2, NOTES_opt3 K1 |
| load order (`prologue_first`) | PF when the block's own prologue is on its critical path: latency-bound grids, every spec 1024-thread grid, plain hold grids of <= 2 blocks per CU; state-first on plain 3-4 blocks per CU | with many co-resident blocks the DRAM queue sets the pace, so the 64 KB reads go first | D15 +0.016, D14 +0.002, D13 |
| read-first hold (`hold_ticks`) | plain co-resident grids hold their state stores for `max(4.9, 0.945 us x ceil(blocks / CUs) + state bytes / 9.1e6 B/us)` | all reads drain before the writes start; the busiest CU's read chain + the grid's bytes at the read-only rate | H01 +0.021; 48 held-out fills (NOTES_ablate re-fit 1); floor: NOTES_final row 8 |
| next-round prefetch (`next_round`) | multi-round PF grids: block b prefetches the small inputs of block b + CUs x blocks per CU (VGPR count of the launched instantiation) | that block runs on the same CU one round later | M17 +0.015, S23 +0.006 |
| state rows behind the barrier (`state_after_prologue`, SB) | single-round spec grids with S >= 6 (`(2S + 25) x 25 clk > 0.40 us` slot hop) and `B H x 64 KB / 5.8e6 <= 2.4 us` (<= 212 blocks) | the prologue loads are order-critical against the block's own 64 KB burst while the read phase is shorter than the prologue chain | S24 +0.0005; bounds on held-out neighbours (NOTES_final rows 3-4) |
| nontemporal tail (`tail_nt`, TN) | spec multi-round grids of > 2 rounds with `S x 0.33 us > 1.1 us`: blocks without a successor on odd XCDs store nontemporal | XCD 3 finishes the grid alone (its pair's write path is slower); alone, an XCD writes 1.25 TB/s nontemporal vs 1.0 write-through | T01 +0.003; NOTES_struct 2, NOTES_xcd 3, NOTES_final 4.1 |
| wave offset (`wave_offset` W1; WS / WM in the multi-round kernel) | single-round spec grids whose write phase outlasts the block's token loop by the offset: `B H S x 64 KB / 6.1e6 - S x 0.9 >= 2.3 us`; multi-round: first generation and (S >= 3) every generation with a successor | waves 8-15 enter the token loop 2.3 us late: breaks a GPU-wide in-phase state of the token steps (5-9 us slow replays) | S19 +0.001, NOTES_xcd 4-6; `TP_WS=0` on six held-out shapes +0.06..+0.54 us |
| **XCC-first, X1 (`xcc_first`), v74** | plain 1024-thread prologue-first single-round grid with `B H x 128 KB / 1.0e6 B/us <= 2.1 us` (<= 16 blocks), on a device whose dispatch was probed to be an exact round-robin over 8 XCCs (`_x1_probe`) | the dispatch reaches physical XCC 1 0.14-0.45 us before the other seven (UNDERFLOOR E5); a grid of 8 H x B blocks in which only the blocks on XCC 1 work (head = blockIdx.x / 8, the rest return at entry) ends ~0.4 us earlier while one XCC's memory path (~1 TB/s) carries the grid's bytes inside a block's own memory phase | A/B v73 -> v74 +0.0049 [+0.0038, +0.0060]; 8 / 12 / 16 blocks -0.34..-0.47 us, 24 blocks neutral (rule off), 32 blocks +0.27, 36+ +2.9..+6.5 (UNDERFLOOR E8b, NOTES_v13 item 1) |
| in-kernel, compile time | late conv commit for S >= 7 (`LCS`); plain multi-round stagger 0 / 0.75 / 1.5 us by a hash of the sequence (`STAG`, `STAGL`); pool translation warm-up in the first 64 blocks; V-split mailbox reset before the staging barrier, in the 16-row kernel for S >= 4 right behind it (`MB_CLR`, `MB_LATE_S`) | see NOTES_probes, MECHANISMS (a), ISSUE_PATH; the reset: an early store costs the latency-bound 16-row grids 0.06-0.17 us, a late one loses the healing on short windows and full grids | held-out: NOTES_final rows 2, 6; fault injection: NOTES_final 10 |

## 2. What each feature is worth

Single-feature leave-one-out on the eval (mean_vs_base units, v66/v67 base, 40-160 pairs each; **not additive**; IDs refer to the full
table in `NOTES_ablate.md` "Final attribution", 102 ablations):



Since v67: WM (v68) ~ +0.0004 pooled and unstable; W1 (v69) +0.0013 +- 0.0003 on its three workloads; v70 (red-team fixes) -0.0003
[-0.0011, +0.0005]; v71 (dead code) -0.0010 [-0.0019, -0.0000] in one session with instruction-identical kernels (session term, see
section 5); v72 (TN at S = 4, ML in the SB kernel) +0.0010 [+0.0000, +0.0019], of which +0.0005 on the two workloads that run changed code. Cumulative v67 -> v72: section 3. v73 (R2 reset behind the staging barrier in the 16-row kernel) +0.0017 [+0.0007, +0.0026] on all four GPUs, from the three B = 1 spec workloads. **v74 (X1, iteration 13): +0.0049 [+0.0038, +0.0060] (120 pairs, every GPU +0.0039..+0.0054), d_k3_b1 -0.43 +- 0.04 us and d_glm16_b1 -0.48 +- 0.03 us, the other 15 workloads within +- 0.05 us.**

## 3. Final scores (v74; v75: section 0.2)

Official eval (`python tools/eval_amd_kda_decode_mi350x.py`), **5 runs on each GPU** (the four in parallel, each pinned to NUMA node 0;
`scratch_v13/final_measure.sh 5`, logs `scratch_v13/final/`), kernel **v74**, iteration 13. `mean_vs_base` per run = mean of the 17 printed
ratios; all 20 runs 17/17 correct. (v73, same procedure, iteration 12: 1.6016 / 1.5889 / 1.5962 / 1.5695, all 20: 1.5891; logs
`scratch_final/final/`.)

| GPU (HIP) | runs | mean_vs_base mean | range |
|---|---|---|---|
| 0 | 5 | 1.6034 | 1.5982 - 1.6094 |
| 2 | 5 | 1.6001 | 1.5989 - 1.6031 |
| 3 | 5 | 1.6024 | 1.5999 - 1.6066 |
| 6 | 5 | 1.5695 | 1.5651 - 1.5761 |
| all | 20 | **1.5939** | 1.5651 - 1.6094 |

| workload | SOTA us | HIP 0 us | HIP 2 us | HIP 3 us | HIP 6 us | mean us | vs_base |
|---|---|---|---|---|---|---|---|
| d_k3_b1 | 16.28 | 13.14 | 13.18 | 13.13 | 13.62 | 13.27 | 1.227 |
| d_k3_b32 | 23.44 | 19.98 | 20.08 | 20.16 | 20.44 | 20.16 | 1.162 |
| d_k3_b128 | 54.44 | 45.05 | 45.18 | 45.10 | 45.51 | 45.21 | 1.204 |
| s4_k3_b1 | 25.04 | 14.38 | 14.33 | 14.38 | 14.83 | 14.48 | 1.729 |
| s4_k3_b16 | 39.88 | 21.81 | 22.11 | 21.93 | 22.52 | 22.09 | 1.805 |
| s4_k3_b64 | 87.08 | 54.44 | 54.74 | 54.86 | 55.26 | 54.82 | 1.588 |
| s8_k3_b1 | 32.48 | 15.27 | 15.18 | 15.29 | 15.74 | 15.37 | 2.113 |
| s8_k3_b16 | 58.36 | 30.89 | 30.82 | 30.98 | 31.78 | 31.12 | 1.875 |
| s8_k3_b64 | 147.52 | 90.03 | 91.12 | 90.86 | 91.37 | 90.84 | 1.624 |
| d_glm16_b1 | 18.80 | 13.17 | 13.24 | 13.14 | 13.64 | 13.30 | 1.414 |
| d_glm16_b64 | 51.04 | 33.54 | 33.67 | 33.57 | 34.06 | 33.71 | 1.514 |
| d_glm16_b256 | 135.80 | 98.32 | 98.14 | 98.30 | 98.60 | 98.34 | 1.381 |
| s6_glm16_b1 | 28.68 | 15.22 | 15.01 | 15.00 | 15.53 | 15.19 | 1.888 |
| s6_glm16_b16 | 53.68 | 30.80 | 30.93 | 30.71 | 31.41 | 30.96 | 1.734 |
| s6_glm16_b64 | 148.28 | 91.20 | 92.62 | 92.66 | 92.75 | 92.31 | 1.606 |
| d_glm8_b128 | 51.48 | 34.16 | 34.14 | 33.70 | 34.26 | 34.06 | 1.511 |
| s6_glm8_b32 | 53.60 | 31.14 | 31.09 | 30.95 | 31.88 | 31.26 | 1.714 |

all runs passed 17/17

The GPUs differ systematically (HIP 6 is ~2 % slower on every workload); `baseline_sota.json` was recorded once, so the per-GPU means are
1.570-1.603 for the same binary. HIP 6's fixed cost of a graph replay is 13.60 us against 13.04-13.20 on the others (UNDERFLOOR E6b), so
there the two X1 workloads end on that knee (13.62 / 13.64 us) and the run-to-run spread hides the step in the per-GPU mean; the paired
A/B below resolves it (+0.0039 on HIP 6).

**Paired A/B v73 -> v74** (`tools_ab/ab.py kernel_v12f_d.py kernel_v13_a.py --pairs 30`: 120 pairs, 30 per GPU, full 17,
`scratch_meas/ab/v13_v73_x1`, table `scratch_v13/ab_x1.txt`):

    mean_vs_base A 1.5875 (sd 0.0132)  B 1.5924 (sd 0.0136)
    delta B-A = +0.0049  95% CI [+0.0038, +0.0060]  (paired sd 0.0060, n=120)
    per GPU: HIP0 +0.0053 (n=30)  HIP2 +0.0050 (n=30)  HIP3 +0.0054 (n=30)  HIP6 +0.0039 (n=30)

d_k3_b1 13.70 -> 13.27 (-0.43 +- 0.04), d_glm16_b1 13.77 -> 13.30 (-0.48 +- 0.03); the other 15 workloads -0.05..+0.04 us (their kernels
are instruction-identical: `scratch_final/cpu.py` hashes, 50 of 50). Predicted by UNDERFLOOR brief 1: +0.0044.

**v73: final paired A/B against v67** (iteration 12; `tools_ab/ab.py kernel_v11a_ab.py kernel_v12f_d.py --pairs 40`: 160 pairs, 40 per GPU, full 17,
`scratch_meas/ab/fin_v67_v73`, table in `scratch_final/logs/ab_v67_v73.txt`):

    mean_vs_base A 1.5825 (sd 0.0132)  B 1.5879 (sd 0.0129)
    delta B-A = +0.0054  95% CI [+0.0045, +0.0063]  (paired sd 0.0058, n=160)
    per GPU: HIP0 +0.0042 (n=40)  HIP2 +0.0073 (n=40)  HIP3 +0.0045 (n=40)  HIP6 +0.0057 (n=40)

Per workload (B - A us, +- 95 %): 16 of 17 faster or equal. s6_glm16_b64 -0.48 +- 0.14 (WM), s4_k3_b64 -0.15 +- 0.06 (WM, TN at S = 4),
s6_glm16_b1 -0.15 +- 0.03 and s8_k3_b1 -0.09 +- 0.03 and s4_k3_b1 -0.06 +- 0.03 (v73 reset position), d_glm8_b128 -0.15 +- 0.04,
s4_k3_b16 -0.14 +- 0.04 and s6_glm8_b32 -0.07 +- 0.04 and s6_glm16_b16 -0.05 +- 0.04 (W1), s8_k3_b16 -0.09 +- 0.08 (ML in the SB kernel),
s8_k3_b64 -0.09 +- 0.06, d_k3_b1 -0.05 +- 0.04; d_k3_b32 +0.01 +- 0.03; the rest within +- 0.03.
Second session, 12 minutes later (`scratch_final/logs/ab_v67_v73_s2.txt`, another 160 pairs): **+0.0041 [+0.0032, +0.0049]**, per GPU
+0.0045 / +0.0048 / +0.0044 / +0.0026; same workloads faster (s6_glm16_b64 -0.35, s4_k3_b64 -0.24, s6_glm8_b32 -0.16, s8_k3_b16 -0.12,
s8_k3_b1 -0.11, s4_k3_b16 -0.10, s6_glm16_b16 -0.10, s6_glm16_b1 -0.09), d_k3_b32 +0.07 +- 0.03 and d_glm16_b64 +0.05 +- 0.03 slower
(hold kernels that are instruction-identical to v67's: the session term of section 5). **Pooled over the two sessions (320 pairs):
v67 -> v73 = +0.0047**, every GPU positive in both; the chain of step A/Bs (v67 -> v72 +0.0013, v72 -> v73 +0.0017) reads +0.0030.

**Held-out (unscored)**, v74: `--heldout` in the two validation runs (HIP 0): 11/11 correct, mean 1.481 / 1.482 vs live SOTA; boundary sweep worst 1.06-1.08 (`y_h24_b21_s1`, `y_h24_b11_s1`). v73: `--heldout` in the final validation run (HIP 0): 11/11 correct, mean 1.487 vs live SOTA (1.475-1.487 over the five validation runs of this phase); slowest ratios
the plain shapes h_d_h24_b16 and h_d_k3_b8 (1.12-1.13), h_d_k3_b200 (1.25); spec shapes 1.58-1.86. Boundary sweep (72 shapes, H in {8, 12, 16, 24}, vs live
SOTA with 30 replays): all correct, worst ratio 1.03-1.09 (`y_h24_b21_s1`, `y_h24_b11_s1`: plain H = 24, where vLLM's HIP kernel is
strongest). Nothing measured in this run is slower than SOTA.

## 4. Validation (v74; v75 changes: section 0.4)

`bash scratch_final/validate_all.sh TAG` (~2 min, uses HIP 0 / 2 / 3 / 6; logs `scratch_final/val/TAG/`; last line `VALIDATE_ALL PASS|FAIL`; v74: `val/v74b` PASS, all 10 steps):

1. official eval 17/17; 2. `--heldout` 11/11; 3. extra shapes (17, incl. S = 10 through the fallback); 4. dispatch-boundary sweep (72 shapes
vs live SOTA, worst 1.06x); 5. replay stress (14 shapes over every kernel class: 40 x (50 flushed replays + full check) + 2000
back-to-back replays); 6. every V-split instantiation forced on six (H, B) (`allvec.py`, 84 cases); 7. the red-team adversarial cases
(`gpu_tests.py` T1-T12: 504 slot-pattern x accepted-count cases on 39 shapes, padded rows vs AITER, 7 graphs sharing the workspace,
extreme-value numerics, 8.8 GiB pools, H in {1, 4, 20, 32, 48, 96} / B to 512, strided and mistyped inputs, poisoned mailboxes, two
graphs on two streams, a conv pool smaller than the state pool); 7b. mailbox fault injection (`poison_rate.py`: every mailbox word stale before each of 4200 calls on seven V-split shapes, at most 2 stale calls tolerated); 8. CPU compile of both builds with the two asm checkers
(`scratch_opt/asmcheck2.py` fire-and-forget asm loads, `scratch_codegen/asmcheck_dpp.py` DPP hazards) and no scratch / spills;
9. the dispatch table on 109 854 shapes against `scratch_final/val/disp_ref.txt` (delete that file when a dispatch change is intended;
`scratch_v13/dispatch_dump.py` regenerates it; v74 changes 50 rows, the plain grids of <= 16 blocks; v73's table is `val/disp_ref_v73.txt`); **10 (v74)** the X1 launch form (`scratch_v13/x1_tests.py`: 18 plain shapes on both sides of the rule, each graph replayed from 12 fresh streams with and without the flush and fully checked; 25 repeated captures and 2000 chained replays; two X1 graphs and a V-split graph replayed concurrently from three threads / streams; a capture before any eager call on the device, which must take the plain grid; 3000 probe dispatches over 16 streams and 10 grids, every one an exact round-robin).

For a source change that claims "no ISA change": `cd scratch_final && python cpu.py TAG --src ../kda_X.cu && python isacmp.py logs/OLD.json logs/TAG.json --reach`.
Timing: `tools_ab/ab.py A.py B.py --pairs 40` (160 pairs, full 17, ~5 min) and `scratch_final/hab/hab.py A.py B.py TAG PAIRS H,B,S ...`
(arbitrary shapes). Final measurement: `scratch_final/final_measure.sh`.

Red-team outcome (`REDTEAM.md`, tests first run on a GPU in this phase): R1 (overlapped launches hang), R2 (stale mailbox words
consumed), R3 (strided / mistyped inputs silently misread) **confirmed and fixed** in v70; R4 (padded rows), R8 (numerics) and the
slot / pool / odd-shape claims **refuted** (no defect); R6 / R7 host-side hardening done; R5 constants measured out of sample
(`NOTES_final.md` section 4: seven hold, one neutral gate kept, one neutral exception removed, one stale gate re-fit -> v72); R9 dead
code removed in v71; the cost of the R2 fix on the B = 1 spec workloads removed in v73 (`NOTES_final.md` section 10).

## 5. Known limitations and risks (v74; added by v75: section 0.5)

- **X1 rests on the dispatch being an exact round-robin over the 8 XCCs (v74).** The X1 kernel is launched on 8 H x B blocks and the
  block that finds itself on physical XCC 1 (`HW_REG_XCC_ID`) does head `blockIdx.x / 8`; one working block per head therefore needs
  every 8 consecutive workgroups to land on 8 different XCCs. blockIdx alone cannot choose the block: the first XCC of the round-robin
  is a property of the HSA queue (0 / 5 / 6 / 7 observed for different streams of one process, `scratch_v13/probe_map.py`), which is
  not known at capture time. The property held in every dispatch observed (4150 probe dispatches in this run, eager and graph, 16
  streams, 1-D and 2-D grids; the floor breaker's stamps on four GPUs), but it is firmware behaviour, not a documented contract.
  Guards (v75: X1 now runs behind `kernel_v14_b`'s guard: `num_xcc == 8` by the HIP attribute, no CU mask on the stream, six probe
  grids, sticky failure, `xcc_reprobe()`; section 0.1; the v74 text follows): `_x1_probe()` checks it on the device (4 grids, once, outside any capture) before `xcc_first()` may select X1, and only
  on a 256-CU device; a capture that precedes every eager call on its device uses the plain grid. If it failed after the probe, a
  head would be skipped or run twice (wrong `out` / state for that head, no crash). `kernel_v13_a.X1_SPAN_US = 0` disables the
  form (v73 dispatch).

- **Same graph replayed concurrently with itself** is undefined (shared mailbox and `out`). Different graphs / streams are safe since v70.
- **Stale mailbox word (after an aborted launch only)**: every block resets the words it will publish before it publishes (v70; in the
  16-row kernel behind the staging barrier since v73). This is **best-effort**: under fault injection (`scratch_final/poison_rate.py`,
  every word stale before each call) a stale word was still consumed in 8 of 300 000 calls on 16-row grids (v72's early reset on the
  same shapes: 1 of 48 000) and 1-2 of 60 000 on 32-row grids, i.e. 2-5 per 10^5 in either position (v69: every call); that one call returns a wrongly normalised `out` (state and conv cache stay correct) and the mailbox is clean
  afterwards. The guaranteed recovery after a device error is `reset_workspace()`. A per-launch tag would close it at ~2-5 us per V-split
  call (-0.03 on the score): not done. `gpu_tests.py T10b` (300 faulted calls) can therefore fail spuriously about once in 10^2 runs; the gate in `validate_all.sh` (step 7b, 4200 faulted calls) tolerates 2.
- **Memory**: each captured V-split call keeps a 128 KB mailbox for the life of the process.
- **Toolchain**: the kernel relies on hand-written asm loads the compiler cannot see, a reserved-register (`m0`) LDS-DMA sink and asm
  DPP blocks. They are correct in the ISA of ROCm 7.2.0 (checked by scripts, both builds); another toolchain only gets a warning
  (`_AUDITED_ROCM`) and must rerun step 8 of the validation.
- **Hardware specificity**: TN reads the XCC id and assumes the measured odd/even write-path pairing; WS / W1 / STAG / the hold constants
  are calibrated on MI350X (256 CUs, 8 XCDs). On other parts the kernel is correct (dispatch from the device's CU count) but untuned.
- **Constants without a mechanism** (they hold on held-out neighbours, but the *why* is open): `LCS = 7`; the SB bound of 212 blocks
  (does not scale with the CU count); `HOLD_MIN` below 1.25 blocks per CU (0.4 us too long at 288 blocks); TN on H = 16 grids at S = 4
  (+0.08 / +0.18 us within CI while H = 12 / 24 / 8 gain 0.14-0.90).
- **Measurement**: the full-17 A/B statistic has a session term of about +- 0.001-0.002 beyond its paired CI (three A/Bs on instruction-
  identical binaries did not close, NOTES_final section 6). Deltas below that are not evidence; v70, v71 and v72 are each inside it (v73 is not: +0.0017 on every GPU, from the predicted workloads).
- **Padded rows** follow AITER (row skipped, `out` zeroed), not `reference.py` (which has no padding rule). Tested against AITER (T2).
- **Dead code still in the source**: the `NSPLIT > 1` half of `kda_main`, the whole-state half of `kda_vec`, `if constexpr` arms of
  frozen switches, four unused `Params` fields (removing them moves the preloaded kernel arguments: needs its own ISA diff and A/B).

## 6. Remaining headroom per family (v74; moved in iteration 14: section 0.6)

Floors are the audited stream / fixed-cost floors of `FLOORS.md` and `NOTES_tpmodel.md` section 5 (version in parentheses where the
number predates this kernel); "now" is section 3.

| family (eval workloads) | now (us, section 3) | floor / bound | what bounds the rest | unbuilt candidates |
|---|---|---|---|---|
| **P-LB** d_k3_b1, d_glm16_b1 | **13.27 / 13.30 (v74; v73: 13.7 / 13.7)** | the knee of the fixed cost: 13.04 / 13.08 / 13.20 / 13.60 us on HIP 2 / 3 / 0 / 6 (UNDERFLOOR E6b), mean 13.23 | **at the floor (0.04-0.07 us)**: on XCC 1 the event is 10.2 + span and the block's span (~3.0 us) is hidden under the knee; a shorter block returns nothing | none with score (the 17-32-block plain grids could use two-XCC masks: held-out only) |
| **P-HOLD** d_k3_b32, d_glm16_b64, d_glm8_b128 | 20.1 / 33.6 / 34.2 | stream floors 18.8 / 32.2 / 32.4 (NOTES_tpmodel 5, v64) | non-state bytes 0.45 / 1.1 / 1.1 us (conv cache, inputs, out: in the contract); d_k3_b32 is read-latency-bound (0.5-0.8 us serial chain); odd-XCD tail 0.5-0.8 on the 4-blocks-per-CU grids | hold floor form below 1.6 blocks per CU (NOTES_final row 8; no eval gain); late slot chain in the PF hold kernel (+0.0002); slow-replay share 4-5 % has no wave-offset analogue (NOTES_xcd 7.3) |
| **P-MR** d_k3_b128, d_glm16_b256 | 45.2 / 98.2 | 41.8 / 91.6 (NOTES_tpmodel 5); HBM ceilings read ~7.0, write ~6.1, 1R:1W ~6.9 TB/s (FLOORS 0.2) | non-state bytes 1.6 / 4.2 us; first generation 1.3-1.8 us (cold hops + stagger); memory-bound with 0.25-0.4 us CU-side slack per block, so a shorter block does not convert | the first generation (~1 us); nothing named with a predicted gain |
| **V16** s4_k3_b1, s6_glm16_b1, s8_k3_b1 | 14.5 / 15.2 / 15.4 | forced floor of this decomposition 14.85 / 15.0 / 15.2 (NOTES_b1model 5, v59; s4 is already below it); audited bound 13.2-13.5 (NOTES_lat2) | 80-85 % of the event is replay + dispatch + two dependent cold hops (12.8-13.1 us); then SIMD issue (~5 cycles per instruction), per-CU store drain and one 0.6 us cross-CU message that cannot overlap the loop | V8 = 16 splits with a two-level mailbox: model -0.04 / +0.03 / -0.24 us (~+0.001-0.002 at full conversion); BAL, PA (NOTES_b1model 5, NOTES_struct 3) |
| **S-1R / S-SB** s4_k3_b16, s6_glm16_b16, s6_glm8_b32, s8_k3_b16 | 22.2 / 31.0 / 31.2 / 31.2 | stream-floor gaps 1.4 / 0.8 / 1.5 / 1.6 us at v51 (CONVENTIONS 1.2); since then -0.3..-0.6 | the prologue waves' load chain is critical (slope 1.3-1.55), then store issue serialised over the 16 waves; on 216-256-block grids the state read itself (NOTES_b1model 6, 8) | the prologue waves' entry on 216-256-block grids (untested, no prediction) |
| **S-MR** s4_k3_b64, s6_glm16_b64, s8_k3_b64 | 54.8 / 92.4 / 90.6 | stream floors 54.8 / 92.9 / 90.5 (v64: at the floor that includes the odd / even XCD skew) | write-stream-bound: XCD 2's write path is ~0.25 us slower per 2 MB burst while the other IOD writes, so XCD 3 alone sets the span (1.0-1.5 us); even XCDs idle for the last 3-6 us; slow-replay share 10-21 % (NOTES_xcd 7) | sub-block work shift on pair {2,3} (PV on one pair: "~2 us on s8 / s6" if it worked; the four-pair form was killed, NOTES_struct 1); wave offset in a write-bound tail generation: -0.07 us on six shapes (NOTES_final 8; +0.0003) |

**Floors that iteration 13 moved or re-tested** (`UNDERFLOOR.md` sections 2-4, `NOTES_v13.md`; the "now" column of the other rows is
still the v73 table, their kernels are unchanged):
- P-LB "hop-bound, 0.4-0.6 us above the fixed cost, no lever for the block-start skew": **broken and converted** (X1, -0.43 / -0.48 us).
- S-1R / S-SB "no per-XCC lever in a single-round grid": **broken, not converted**. XCC 3 ends the single-round spec grids too (the
  last 4 of its 24 items removed: -0.25 / -0.63 us on s4_k3_b16 / s8_k3_b16, any other XCC 0..-0.3); moving 2 of XCC 3's items to
  XCC 0 / 1 through spare CUs, results correct: s4_k3_b16 -0.19 +- 0.08, s8_k3_b16 -0.32 +- 0.11, held-out 12,16,6 -0.24 +- 0.09
  (control off XCC 1: +0.3..+0.7). Worth +0.0020 on the full 17 (only the 192-block grids have spare CUs), under the +0.004 bar of
  this run: parked with a production sketch (NOTES_v13 item 3). d_k3_b32 shows the opposite sign (XCC 0 / 1 carry its critical blocks).
- S-MR "at the stream floor" (UNDERFLOOR E9: a copy stream is 2.4-2.8 us faster at S >= 6): the floor is real for a stream, **not
  reachable by the V-split kernel** (forced `kda_vec` on 12,64,8 / 16,64,6: +33..+54 us); it needs split state traffic without a
  split prologue.
- Nontemporal stores on XCC 3 in single-round grids (TN's law applied to one XCC): **holds** (-0.03..+0.09 us; every other mask loses).
- Fixed cost of the graph replay (13.0-13.6 us, four CP packets), host outside the window, two cold hops at V16 entry, mixed store
  policies: **hold** (UNDERFLOOR E1-E3, E7; RUNTIME_RESEARCH).

Summary: every family sits within 0-7 % of a floor that moves only the bytes the contract requires; what is left with us-scale value
is under those floors (the XCD 2 / 3 write-path skew, the first generation of the multi-round grids) or in the fixed 13 us of a
graph replay. The search inside this architecture has converged: v68-v72 are each worth <= 0.001, and v73's +0.0017 only took back
what the correctness fix of v70 had cost.

## 7. Map of the notes

| file | contents / when to read |
|---|---|
| **FINAL.md** | this file |
| **NOTES_v15.md**, PLACEMENT_RESEARCH.md, NOTES_b1spec15.md | iteration 15: the address-bit law of the die interleave, the attribution of the draw to descriptor and kernarg pool (600 processes), the two-copy mechanism and its probe, every kill (2 MiB objects, page alignment, kernel-argument forms), the cross-process A/B procedure; source-level research on where each allocation comes from; the B = 1 spec phase budget and family floor |
| scratch_v15/ | `draw.py` + `arms.sh` + `armrep.py` / `armadj.py` (cross-process A/B with placement probes: the procedure of 00.6), `pp/` (raw-address cold-load probe), `e1.py`, `attr.py` / `table1.py`, `place_tests.py` (validation step 11), `isa15.py` (step 12), `cc.sh` (CPU layout check), `prel.py`, `final_measure.sh`, `runs/*.txt` (every table quoted), `runs/raw_json.tar.gz` |
| **NOTES_v14.md**, ASYMMETRY.md, HW_RESEARCH.md | iteration 14: the per-XCC model and its three rules (two kept), the guard, the method finding (an entry instruction re-allocates a kernel: asm entry blocks), the allocation term, every kill with numbers, the page -> die probe; the per-XCC critical-path map of every family; driver / patent / white-paper sources behind the per-XCC laws |
| scratch_v14/ | `mk_a.py` / `mk_a_py.py` / `mk_b.py` (how `kda_v14_a.cu`, `kernel_v14_a.py`, `kda_v14_b.cu` are generated from v74), `mk_x.py` (experiment copy), `mab.py` + `hrun.py` (canary-filtered multi-arm paired runs), `ab17.py`, `xcc_tests.py` + `xcc_shapes.py` (validation step 10), `dispatch_dump.py`, `final_measure.sh`, `canary.py`, `cprobe/`, `runs/*.txt` (every table quoted in NOTES_v14) |
| **NOTES_v13.md**, UNDERFLOOR.md, RUNTIME_RESEARCH.md | iteration 13: X1 (v74), the queue-dependent first XCC of the round-robin, the per-XCC item experiments on single-round grids (parked), kills; the assumption ledger under every floor with verdicts; CLR 7.2.1 source research on the fixed cost |
| **NOTES_final.md** | this run: red-team test outcomes, v70-v72, out-of-sample constant checks, dead-code proof, the session-term finding |
| REDTEAM.md | adversarial review of v67 (R1-R9), full list of calibrated constants and dead code; `scratch_redteam/` models and ISA audit scripts |
| NOTES_xcd.md | v67 -> v69 (WM, W1): the XCD pair {2,3} law, the collective slow state; read 7.4 before quoting a gain |
| NOTES_ablate.md, ABLATIONS.md | leave-one-out of every live feature (v66 -> v67): the attribution table, hold re-fit; the ablation catalog and harness |
| NOTES_struct.md | structural run v65 -> v66: TN, PV killed, B = 1 decompositions, unbuilt candidates |
| NOTES_tpmodel.md | throughput model of P-MR / HOLD / S-MR, remaining gap per workload (section 5), WS -> v65 |
| NOTES_b1model.md | delay-injection critical-path model for B = 1 spec and mid-B spec (v58-v61), forced floors, unbuilt candidates |
| NOTES_codegen.md, BUDGET_AUDIT.md, ISA_AUDIT.md | instruction-stream work v63 -> v64; emitted vs minimal instruction counts per phase; ISA audit of v25 |
| NOTES_probes.md / PROBES.md, NOTES_probes2.md / PROBES2.md | single-change probe catalogs and their screens (v54-v57, v57-v62) |
| MEASURE.md, tools_ab/README.md | the paired A/B tool, its validation, the XCD-skew mechanism: read before measuring anything |
| MECHANISMS.md | forensics: 2 blocks per CU, P-MR excess, odd / even XCD skew, B = 1 spec |
| FLOORS.md | audited fixed cost, bandwidth ceilings, per-workload stream floors (the reference for every "floor") |
| CONVENTIONS.md, ISSUE_PATH.md | design conventions never varied (C01-C20) with the v51 gap table; address / issue path (v55 warm-up wave, v56 2-D grid) |
| TRANSFER.md, SCOUT.md, RESEARCH.md | mechanism x instantiation matrix (v38); outside knowledge; iteration-2 runtime / toolchain research |
| CHALLENGER.md, DESIGNS.md | clean-room alternative structures (row-slab, WY, D1-D7) that lost: read before proposing a new structure |
| NOTES_bw.md, NOTES_bw2.md | plain B >= 32 bandwidth work v34 -> v38 and v41 -> v51 (hold kernel, NX prefetch, store policies) |
| NOTES_lat.md, NOTES_lat2.md | B = 1 latency runs v31 -> v34, v38 -> v39 (hardware cost table, the 13.2-13.5 us bound) |
| NOTES_solo.md, NOTES_opt.md, NOTES_opt3.md, NOTES_opt4.md, NOTES_opt7.md | iteration logs v1 -> v6, v6 -> v13, v13 -> v18, v25 -> v31, v52 -> v53 |
| results.tsv, solutions/ | one row per kept / discarded experiment; a snapshot of every committed version |
| scratch_v13/ | `x1_tests.py` (validation step 10), `dispatch_dump.py`, `mk_a.py` (how v74 was generated from v73), `mab.py` (multi-variant paired runs), `kda_uq.cu` + `kernel_uq*.py` (per-XCC item / store-policy experiments), `probe_map.py`, `final_measure.sh`, `mk_final.py` |
| scratch_final/ | `validate_all.sh`, `gpu_tests.py`, `allvec.py`, `cpu.py` + `isacmp.py` + `reach.py`, `dispatch_dump.py`, `hab/`, `var/` (constant variants), `mk_*.py` (how v70 / v71 / v72 were generated), `final_measure.sh` |
