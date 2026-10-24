#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <vector>

// Kimi Delta Attention (KDA) prefill, Kimi-K3 shapes at TP8, AMD MI350X (gfx950).
//
// Gate / beta / normalization (all done inside the kernel):
//   g    = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h*128 + k]))   (log space)
//   beta = sigmoid(raw_beta)
//   q, k = L2-normalized over head_dim (eps 1e-6)
//
// State layout: V-major [N, H, V, K], fp32.
// bf16 buffers are passed as raw 16-bit words (uint16_t) because main.cpp is built by the
// host compiler, which cannot parse hip_bf16.h; kernel.cu reinterprets them.

using bf16_bits = uint16_t;

// Production contract: the recurrent state lives in a paged pool (row stride may exceed H*128*128 floats; rows dense),
// sequence n reads row sidx[n] if hh0[n] (else starts from zero and never reads it) and writes its final state there;
// ckoff[n] > 0 && ckidx[n] >= 0 also writes the state after ckoff[n] tokens to row ckidx[n]; raw_beta is read with a
// per-token stride of ld_beta elements. `meta` must be the device tables of kda_build_meta (cu32 | choff | cidx).
struct KdaPool {
    float* cache;
    int64_t row_stride;               // floats between consecutive rows
    const int32_t* sidx;              // [N]
    const bool* hh0;                  // [N]
    const int32_t* ckoff;             // [N] or nullptr
    const int32_t* ckidx;             // [N] or nullptr
    int64_t ld_beta;
};

void kda_prefill_kernel_launch(
    bf16_bits* output,                // [total_seq_len, H, 128]
    float* new_state,                 // [num_seqs, H, 128, 128]
    const bf16_bits* q,               // [total_seq_len, H, 128]
    const bf16_bits* k,               // [total_seq_len, H, 128]
    const bf16_bits* v,               // [total_seq_len, H, 128]
    const bf16_bits* raw_g,           // [total_seq_len, H, 128]
    const bf16_bits* raw_beta,        // [total_seq_len, H]
    const float* A_log,               // [H]
    const float* dt_bias,             // [H * 128]
    const float* state,               // [num_seqs, H, 128, 128] or nullptr (zero initial state)
    const int64_t* cu_seqlens,        // [num_seqs + 1]
    int32_t num_seqs,
    int32_t num_heads,
    float scale,
    float lower_bound,
    int64_t total_seq_len,
    float* workspace,                 // kda_workspace_floats(...) floats
    const int32_t* meta,              // prepare() metadata (device) or nullptr (built on device)
    const int32_t* hmeta,             // prepare() metadata (host header) or nullptr
    hipStream_t stream,
    const struct KdaPool* pool = nullptr, // production contract (below); new_state / state / cu_seqlens unused
    const struct KdaRelayCtx* rc = nullptr  // from kda_relay_acquire(); null / no flags: the relay is not taken
);

// Per-launch relay context: the launch's private chain-flag region (owned by its (device, stream)) and its epoch.
// kda_relay_acquire() returns false when this launch must not take the relay (metadata without relay tables, relay
// switched off, no flag region available); pass the same decision to kda_workspace_floats() and the launch.
struct KdaRelayCtx {
    unsigned long long* flags;
    unsigned long long epoch;
    int region = -1;                  // index of the flag region (diagnostics / tests)
};
bool kda_relay_acquire(const int32_t* hmeta, int32_t num_heads, int device, hipStream_t stream, struct KdaRelayCtx* ctx);
// Relay watchdog: != 0 once a relay item reported a broken assumption (bit 62: a chain wait timed out, bit 61: block id % 8
// was not the workgroup's XCD). The relay is off from then on; results of the launches in flight at that time are invalid.
// *newly (optional) is set on the first call that sees the fault.
unsigned long long kda_relay_fault(int* newly = nullptr);
// Watchdog recovery. A fault is recorded per flag region = per (device, stream): every relay launch issued on that
// (device, stream) since the last clean check is invalid (outputs, final and checkpoint pool rows of its sequences);
// launches on other streams are not affected. The relay is off on that DEVICE from then on (chunk-walk path).
// All of these only read host memory (no device synchronization): they are definitive for launches that have completed.
struct KdaRelayFaultRec { int device; unsigned long long stream; unsigned long long code; int pending; };
// != 0: fault code of (device, stream) not yet reported to a caller; marks it reported.
unsigned long long kda_relay_take_fault(int device, hipStream_t stream);
// Relay state of `device` (0 on, 1 switched off / epochs exhausted, 2 XCD probe failed, 3 watchdog fault); *serial = number
// of relay launches of the process so far (a fault code carries its launch's serial); recs (optional) += fault records of
// the device (device < 0: all). acknowledge: mark them reported (the caller handles them; prefill will not raise).
int kda_relay_health(int device, bool acknowledge, unsigned long long* serial, std::vector<KdaRelayFaultRec>* recs);
void kda_set_relay_off(int off);
// Range guard (default off; KDA_RANGE_GUARD=1): relay launches watch the value range of every V slice where its chain ends
// and recompute slices that left fp16's window (non-finite, tiny / underflowed state, tiny beta) in fp32, instead of
// writing non-finite or inaccurate outputs and pool rows. Costs ~5 % on the relay path while it is on.
void kda_set_range_guard(int on);
int kda_get_range_guard();
std::vector<int> kda_xcd_map(int n, hipStream_t stream);  // diagnostic: XCC id per workgroup of a probe launch  // 1: every later launch takes the prologue + chunk-walk path (tests, operations)

int64_t kda_workspace_floats(int64_t total_seq_len, int32_t num_seqs, int32_t num_heads,
                             const int32_t* hmeta, bool relay = true);

// Untimed metadata (built on the host by prepare()): int32 [cu32 (N+1) | pad | choff int64 (N+1) |
// chunk index pairs (seq, chunk) padded with (-1, 0) to T/64 + N entries].
int64_t kda_meta_choff_offset(int32_t num_seqs);
// Builds the device metadata (cu32, chunk offsets, chunk descriptors, per-sequence
// group segments) and its host header [H, nseg, maxG, seg offset, seqinfo offset].
#include <vector>
// mode 0: phase-1 policy (relay tables); 1: per-sequence group segments, no relay; 2: uniform groups (vLLM rule), no relay.
void kda_build_meta(const int64_t* cu_seqlens_host, int32_t num_seqs, int32_t num_heads,
                    std::vector<int32_t>& dev, std::vector<int32_t>& host, int mode = 0,
                    const int32_t* ck_chunks = nullptr);  // mode 3: relay on the production contract; ck_chunks[n] = checkpoint offset / 64 (0: none)

void kda_set_stages(int s);  // profiling hook: stop after stage s of the prologue + chunk-walk path (3 = all)
