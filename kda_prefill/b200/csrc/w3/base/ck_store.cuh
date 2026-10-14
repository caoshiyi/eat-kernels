// [contract2] full-line fp32 state stores from the tcgen05 16x256b fragment layout (checkpoints, snapshots).
//
// tcgen05.ld.16x256b.x4 hands lane (r = lane >> 2, q = lane & 3) the float pairs (r, 8j + 2q .. +1) and (r + 8, same columns) of fragment j = 0..3 (32 columns).  Storing them as
// float2 writes 8 B per lane: one 32 B sector of 8 different 128 B lines per warp instruction.  A 4x4 transpose of the float2 elements inside each quad (8 shuffles per row
// group) gives lane q the 8 contiguous columns 8q .. 8q + 7 of its row, i.e. one 32 B store per lane (st.global.v8.f32) and 128 B of one line per quad: every warp instruction
// writes 8 complete lines.  Measured: the checkpoint store time drops by ~30% (see work_contract2/NOTES.md).  The values written are identical.
#pragma once

__device__ __forceinline__ void ck_st_v8(float* p, const uint32_t* r) {
    asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(p), "f"(__uint_as_float(r[0])), "f"(__uint_as_float(r[1])), "f"(__uint_as_float(r[2])), "f"(__uint_as_float(r[3])),
                 "f"(__uint_as_float(r[4])), "f"(__uint_as_float(r[5])), "f"(__uint_as_float(r[6])), "f"(__uint_as_float(r[7])) : "memory");
}
// e[j] = lane's float2 of fragment j  ->  e[j] = float2 of fragment q (the lane's quad index) held by quad lane j
__device__ __forceinline__ void ck_quad_transpose(uint32_t (&e)[4][2], int lane) {
    const bool hi = (lane & 2) != 0, odd = (lane & 1) != 0;
    #pragma unroll
    for (int stage = 0; stage < 2; stage++) {
        const int xo = stage == 0 ? 2 : 1;
        const bool up = stage == 0 ? hi : odd;
        const int a0 = 0, a1 = stage == 0 ? 1 : 2;          // elements kept by the lanes with up == true ("low" pair)
        const int b0 = stage == 0 ? 2 : 1, b1 = 3;          // elements kept by the lanes with up == false
        const uint32_t s0x = up ? e[a0][0] : e[b0][0], s0y = up ? e[a0][1] : e[b0][1];
        const uint32_t s1x = up ? e[a1][0] : e[b1][0], s1y = up ? e[a1][1] : e[b1][1];
        const uint32_t r0x = __shfl_xor_sync(0xffffffffu, s0x, xo), r0y = __shfl_xor_sync(0xffffffffu, s0y, xo);
        const uint32_t r1x = __shfl_xor_sync(0xffffffffu, s1x, xo), r1y = __shfl_xor_sync(0xffffffffu, s1y, xo);
        if (up) { e[a0][0] = r0x; e[a0][1] = r0y; e[a1][0] = r1x; e[a1][1] = r1y; }
        else    { e[b0][0] = r0x; e[b0][1] = r0y; e[b1][0] = r1x; e[b1][1] = r1y; }
    }
}
// r: the 16 registers of one tcgen05.ld.16x256b.x4 (rows row0 / row0 + 8, columns col0 .. col0 + 31) -> g[row * 128 + col]
__device__ __forceinline__ void ck_store_tile(float* __restrict__ g, const uint32_t (&r)[16], int row0, int col0, int lane) {
    uint32_t a[4][2] = {{r[0], r[1]}, {r[4], r[5]}, {r[8], r[9]}, {r[12], r[13]}};
    uint32_t b[4][2] = {{r[2], r[3]}, {r[6], r[7]}, {r[10], r[11]}, {r[14], r[15]}};
    ck_quad_transpose(a, lane);
    ck_quad_transpose(b, lane);
    const uint32_t oa[8] = {a[0][0], a[0][1], a[1][0], a[1][1], a[2][0], a[2][1], a[3][0], a[3][1]};
    const uint32_t ob[8] = {b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1]};
    const int col = col0 + (lane & 3) * 8;
    ck_st_v8(g + (long long)row0 * 128 + col, oa);
    ck_st_v8(g + (long long)(row0 + 8) * 128 + col, ob);
}
// drop-in for sio_state_store / sio_state_store_cg: g = base of the [128][128] fp32 state, tm = TMEM address (lane 0, first column), q = lane quarter of this warp
__device__ __forceinline__ void sio_state_store_fl(float* __restrict__ g, uint32_t tm, int q, int lane) {
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int row0 = q * 32 + half * 16 + (lane >> 2);
        #pragma unroll
        for (int cp = 0; cp < 4; cp += 2) {
            uint32_t r[2][16];
            #pragma unroll
            for (int c2 = 0; c2 < 2; c2++)
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                  : "=r"(r[c2][0]),"=r"(r[c2][1]),"=r"(r[c2][2]),"=r"(r[c2][3]),"=r"(r[c2][4]),"=r"(r[c2][5]),"=r"(r[c2][6]),"=r"(r[c2][7]),"=r"(r[c2][8]),"=r"(r[c2][9]),"=r"(r[c2][10]),"=r"(r[c2][11]),"=r"(r[c2][12]),"=r"(r[c2][13]),"=r"(r[c2][14]),"=r"(r[c2][15])
                  : "r"(tm + ((uint32_t)(q * 32 + half * 16) << 16) + (cp + c2) * 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            #pragma unroll
            for (int c2 = 0; c2 < 2; c2++) ck_store_tile(g, r[c2], row0, (cp + c2) * 32, lane);
        }
    }
}

// out-of-line copy for kernels whose compute role is register-tight (cio: the inlined version made ptxas spill 8 values in the compute loop)
static __device__ __noinline__ void sio_state_store_fl_ni(float* __restrict__ g, uint32_t tm, int q, int lane) { sio_state_store_fl(g, tm, q, lane); }
