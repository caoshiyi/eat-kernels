#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <pybind11/stl.h>
#include "kernel.h"

// Entry point used by tools/eval_amd_kda_mi350x.py.
std::map<std::string, torch::Tensor> run(
    torch::Tensor q,            // [1, T, H, 128] bf16
    torch::Tensor k,            // [1, T, H, 128] bf16
    torch::Tensor v,            // [1, T, H, 128] bf16
    torch::Tensor raw_g,        // [1, T, H, 128] bf16
    torch::Tensor raw_beta,     // [1, T, H] bf16
    torch::Tensor A_log,        // [H] f32
    torch::Tensor dt_bias,      // [H * 128] f32
    torch::Tensor state,        // [N, H, 128, 128] f32 (or empty)
    torch::Tensor cu_seqlens,   // [N + 1] int64
    double scale,
    double lower_bound,
    c10::optional<std::vector<torch::Tensor>> meta
) {
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                v.dtype() == torch::kBFloat16 && raw_g.dtype() == torch::kBFloat16 &&
                raw_beta.dtype() == torch::kBFloat16, "q/k/v/raw_g/raw_beta must be bfloat16");
    TORCH_CHECK(q.size(3) == 128, "head_dim must be 128");

    int64_t total_seq_len = q.size(1);
    int num_heads = q.size(2);
    int num_seqs = cu_seqlens.size(0) - 1;

    auto output = torch::empty_like(v);
    auto new_state = torch::empty({num_seqs, num_heads, 128, 128},
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    const int32_t* dmeta = meta.has_value() ? (*meta)[0].data_ptr<int32_t>() : nullptr;
    const int32_t* hmeta = meta.has_value() ? (*meta)[1].data_ptr<int32_t>() : nullptr;
    const hipStream_t stream = at::hip::getCurrentHIPStream(q.get_device()).stream();
    KdaRelayCtx rc;
    const bool relay = kda_relay_acquire(hmeta, num_heads, q.get_device(), stream, &rc);
    auto workspace = torch::empty({kda_workspace_floats(total_seq_len, num_seqs, num_heads, hmeta, relay)},
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    const float* state_ptr = (state.numel() > 0) ? state.data_ptr<float>() : nullptr;

    kda_prefill_kernel_launch(
        reinterpret_cast<bf16_bits*>(output.data_ptr()),
        new_state.data_ptr<float>(),
        reinterpret_cast<const bf16_bits*>(q.data_ptr()),
        reinterpret_cast<const bf16_bits*>(k.data_ptr()),
        reinterpret_cast<const bf16_bits*>(v.data_ptr()),
        reinterpret_cast<const bf16_bits*>(raw_g.data_ptr()),
        reinterpret_cast<const bf16_bits*>(raw_beta.data_ptr()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        state_ptr,
        cu_seqlens.data_ptr<int64_t>(),
        num_seqs, num_heads,
        static_cast<float>(scale), static_cast<float>(lower_bound),
        total_seq_len,
        workspace.data_ptr<float>(),
        dmeta, hmeta, stream, nullptr, &rc);

    std::map<std::string, torch::Tensor> result;
    result["output"] = output;
    result["new_state"] = new_state;
    return result;
}

// Untimed per-batch metadata derived from cu_seqlens only (see program.md "prepare"):
// [device int32 tables, host int32 header].  Built for the task's H = 12 heads; run()
// falls back to on-device metadata and the uniform group policy for any other H.
constexpr int kPrepareHeads = 12;
std::vector<torch::Tensor> prepare(torch::Tensor cu_seqlens) {
    auto cu = cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    const int N = cu.size(0) - 1;
    std::vector<int32_t> dev, host;
    kda_build_meta(cu.data_ptr<int64_t>(), N, kPrepareHeads, dev, host);
    auto d = torch::from_blob(dev.data(), {(int64_t)dev.size()}, torch::kInt32).clone();
    auto hh = torch::from_blob(host.data(), {(int64_t)host.size()}, torch::kInt32).clone();
    return {d.to(cu_seqlens.device()), hh};
}

// ---------------------------------------------------------------------------------------------------------------
// Production entry (tasks/amd_kda_prod_mi350x): paged state pool read / written in place, per-sequence
// has_initial_state, prefix-cache checkpoints, strided raw_beta, output written straight into the caller's slice.
// No host synchronization: everything layout-dependent comes from prepare_prod() (host tensors only).
// ---------------------------------------------------------------------------------------------------------------
constexpr int32_t kMetaMagic = 0x4b444131;

// Watchdog recovery: raises if a relay fault of (device, stream) has not been reported yet (once per fault). Host memory
// reads only. Called by run_prod when it does not take the relay and by prefill_prepare (kernel.py) at the start of a step.
static void relay_check(int device, hipStream_t stream) {
    const unsigned long long code = kda_relay_take_fault(device, stream);
    TORCH_CHECK(code == 0, "KDA prefill relay fault (watchdog code 0x", std::hex, code, std::dec, ", relay launch #",
                (code & ~(3ull << 61)) >> 24, ") on device ", device, " stream 0x", std::hex, reinterpret_cast<uintptr_t>(stream), std::dec,
                ": a chain handoff timed out. INVALID: the results (outputs, final-state and checkpoint pool rows of every sequence) of "
                "all prefill() calls issued on this device and stream since its last clean prefill_health(sync=True) and before this "
                "exception; other streams are not affected. The call that raised was not launched and touched nothing. The relay is "
                "now off on device ", device, ": retry the step from saved states (it runs on the chunk-walk path and is correct), or "
                "abort those requests. Set KDA_RELAY_OFF=1 if the process is restarted instead.");
}
// device_index >= 0: the device tables are returned on that device, copied (one blocking H2D copy) straight from a
// per-thread scratch vector that keeps its pages from call to call: a 64K-token H = 128 batch has 4.2 MB of tables, and a
// fresh host buffer per call (page faults) plus a host clone of it cost more than building them. -1: host tensor (clone).
std::vector<torch::Tensor> prepare_prod(torch::Tensor cu_seqlens, int64_t num_heads, int64_t mode,
                                        c10::optional<torch::Tensor> ck_off, c10::optional<torch::Tensor> ck_idx,
                                        int64_t device_index) {
    auto cu = cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    const int N = cu.size(0) - 1;
    static thread_local std::vector<int32_t> dev;
    std::vector<int32_t> host, kck;
    if (ck_off.has_value() && ck_idx.has_value()) {
        auto o = ck_off->to(torch::kCPU).to(torch::kInt32).contiguous(), r = ck_idx->to(torch::kCPU).to(torch::kInt32).contiguous();
        kck.resize(N);
        for (int n = 0; n < N; ++n) {
            const int32_t off = o.data_ptr<int32_t>()[n];
            kck[n] = off > 0 && off % 64 == 0 && r.data_ptr<int32_t>()[n] >= 0 ? off / 64 : 0;
        }
    }
    kda_build_meta(cu.data_ptr<int64_t>(), N, static_cast<int32_t>(num_heads), dev, host, static_cast<int>(mode),
                   kck.empty() ? nullptr : kck.data());
    // Trailer (checked by run_prod: the device tables are indexed with these, so a meta from another layout must not run):
    // [magic, N, T, checkpoints planned].
    bool any_ck = false;
    for (int32_t c : kck) any_ck = any_ck || c > 0;
    host.insert(host.end(), {kMetaMagic, static_cast<int32_t>(N), static_cast<int32_t>(cu.data_ptr<int64_t>()[N]), any_ck ? 1 : 0});
    auto blob = torch::from_blob(dev.data(), {(int64_t)dev.size()}, torch::kInt32);
    auto d = device_index >= 0 ? blob.to(torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index))) : blob.clone();
    auto hh = torch::from_blob(host.data(), {(int64_t)host.size()}, torch::kInt32).clone();
    if (dev.capacity() > (size_t(1) << 24)) std::vector<int32_t>().swap(dev);  // do not keep more than 64 MB of scratch
    return {d, hh};
}

void run_prod(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor raw_g,   // [1, T, H, 128] bf16, dense
    torch::Tensor raw_beta,                                                   // [1, T, H] bf16, any token stride
    torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor state_cache,                                                // [slots, H, 128, 128] f32, dense rows
    torch::Tensor state_indices, torch::Tensor has_initial_state,             // int32 [N], bool [N]
    torch::Tensor out,                                                        // [1, T, H, 128] bf16, dense
    c10::optional<torch::Tensor> ck_off, c10::optional<torch::Tensor> ck_idx, // int32 [N]
    double scale, double lower_bound,
    torch::Tensor dmeta, torch::Tensor hmeta) {
    const int64_t T = q.size(1);
    const int H = q.size(2);
    const int N = state_indices.size(0);
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.size(3) == 128, "q must be bf16 [1, T, H, 128]");
    TORCH_CHECK(state_cache.stride(3) == 1 && state_cache.stride(2) == 128 && state_cache.stride(1) == 128 * 128,
                "state_cache rows must be dense");
    TORCH_CHECK(raw_beta.stride(2) == 1 && out.is_contiguous(), "raw_beta rows / out must be dense");
    // The kernels index these raw: wrong dtype / stride / length would read or write other pool rows (no device sync here:
    // only tensor metadata is looked at).
    TORCH_CHECK(k.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16 && raw_g.dtype() == torch::kBFloat16 &&
                raw_beta.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16 && state_cache.dtype() == torch::kFloat32 &&
                A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kFloat32, "prefill: unexpected tensor dtype");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && raw_g.is_contiguous() &&
                k.sizes() == q.sizes() && v.sizes() == q.sizes() && raw_g.sizes() == q.sizes() && out.sizes() == q.sizes() &&
                raw_beta.dim() == 3 && raw_beta.size(1) == T && raw_beta.size(2) == H, "prefill: q / k / v / raw_g / raw_beta / out shapes");
    TORCH_CHECK(A_log.is_contiguous() && A_log.numel() == H && dt_bias.is_contiguous() && dt_bias.numel() == int64_t(H) * 128 &&
                state_cache.dim() == 4 && state_cache.size(1) == H, "prefill: A_log / dt_bias / state_cache shapes");
    TORCH_CHECK(state_indices.dtype() == torch::kInt32 && state_indices.dim() == 1 && state_indices.is_contiguous() &&
                has_initial_state.dtype() == torch::kBool && has_initial_state.is_contiguous() && has_initial_state.numel() == N,
                "state_indices must be a contiguous int32 [N] tensor and has_initial_state a contiguous bool [N] tensor");
    const bool ck_given = ck_off.has_value() && ck_idx.has_value();
    TORCH_CHECK(!ck_given || (ck_off->dtype() == torch::kInt32 && ck_idx->dtype() == torch::kInt32 && ck_off->is_contiguous() &&
                              ck_idx->is_contiguous() && ck_off->numel() == N && ck_idx->numel() == N),
                "checkpoint_offsets / checkpoint_state_indices must be contiguous int32 [N] tensors");
    {
        TORCH_CHECK(hmeta.dtype() == torch::kInt32 && hmeta.is_contiguous() && hmeta.numel() >= 12 && !hmeta.is_cuda() &&
                    dmeta.dtype() == torch::kInt32 && dmeta.is_contiguous() && dmeta.device() == q.device(),
                    "prefill: meta is not a prefill_prepare() result for this device");
        const int32_t* tr = hmeta.data_ptr<int32_t>() + hmeta.numel() - 4;
        TORCH_CHECK(tr[0] == kMetaMagic && hmeta.data_ptr<int32_t>()[0] == H && tr[1] == N && tr[2] == static_cast<int32_t>(T),
                    "prefill: meta was prepared for another layout (H / N / T differ)");
        TORCH_CHECK(tr[3] == 0 || ck_given, "prefill: meta was prepared with checkpoints but no checkpoint tensors were passed");
    }
    KdaPool pool;
    pool.cache = state_cache.data_ptr<float>();
    pool.row_stride = state_cache.size(0) > 1 ? state_cache.stride(0) : int64_t(H) * 128 * 128;
    pool.sidx = state_indices.data_ptr<int32_t>();
    pool.hh0 = has_initial_state.data_ptr<bool>();
    const bool ck = ck_off.has_value() && ck_idx.has_value();
    pool.ckoff = ck ? ck_off->data_ptr<int32_t>() : nullptr;
    pool.ckidx = ck ? ck_idx->data_ptr<int32_t>() : nullptr;
    pool.ld_beta = raw_beta.stride(1);
    const int32_t* hm = hmeta.data_ptr<int32_t>();
    const c10::cuda::CUDAGuard device_guard(q.device());  // symbol lookups / allocations below are per current device
    // The launch goes to the current stream of the data's device; that (device, stream) owns the relay's flag region.
    const hipStream_t stream = at::hip::getCurrentHIPStream(q.get_device()).stream();
    KdaRelayCtx rc;
    const bool relay = kda_relay_acquire(hm, H, q.get_device(), stream, &rc);
    // Not on the relay (layout, switch, or a watchdog fault on this device): report a fault of this (device, stream) that no
    // caller has seen yet, before anything is launched. (A granted relay implies no fault is known on the device.)
    if (!relay) relay_check(q.get_device(), stream);
    auto workspace = torch::empty({kda_workspace_floats(T, N, H, hm, relay)},
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    kda_prefill_kernel_launch(
        reinterpret_cast<bf16_bits*>(out.data_ptr()), nullptr,
        reinterpret_cast<const bf16_bits*>(q.data_ptr()),
        reinterpret_cast<const bf16_bits*>(k.data_ptr()),
        reinterpret_cast<const bf16_bits*>(v.data_ptr()),
        reinterpret_cast<const bf16_bits*>(raw_g.data_ptr()),
        reinterpret_cast<const bf16_bits*>(raw_beta.data_ptr()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), nullptr, nullptr,
        N, H, static_cast<float>(scale), static_cast<float>(lower_bound), T,
        workspace.data_ptr<float>(), dmeta.data_ptr<int32_t>(), hm, stream, &pool, &rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_prod", &prepare_prod, py::arg("cu_seqlens"), py::arg("num_heads"), py::arg("mode"),
          py::arg("checkpoint_offsets"), py::arg("checkpoint_state_indices"), py::arg("device_index") = -1);
    m.def("set_stages", &kda_set_stages);
    m.def("set_relay_off", &kda_set_relay_off);
    m.def("set_range_guard", &kda_set_range_guard);
    m.def("get_range_guard", &kda_get_range_guard);
    m.def("xcd_map", [](int n) { return kda_xcd_map(n, at::hip::getCurrentHIPStream().stream()); });
    m.def("relay_fault", []() { return kda_relay_fault(nullptr); });
    m.def("relay_check", [](int device) { relay_check(device, at::hip::getCurrentHIPStream(device).stream()); });
    m.def("relay_health", [](int device, bool acknowledge) {
        unsigned long long serial = 0;
        std::vector<KdaRelayFaultRec> recs;
        const int state = kda_relay_health(device, acknowledge, &serial, &recs);
        std::vector<std::tuple<int, unsigned long long, unsigned long long, int>> out;
        for (const auto& r : recs) out.emplace_back(r.device, r.stream, r.code, r.pending);
        return std::make_tuple(state, serial, out);
    });
    // Test hook (host keying only, nothing is launched besides the one-time probe): what kda_relay_acquire() decides for a
    // device index on the current stream -> (granted, flag region address, launch serial, region index).
    m.def("relay_acquire_test", [](torch::Tensor hmeta, int num_heads, int device) {
        KdaRelayCtx rc;
        const bool ok = kda_relay_acquire(hmeta.data_ptr<int32_t>(), num_heads, device, at::hip::getCurrentHIPStream().stream(), &rc);
        return std::make_tuple(ok, reinterpret_cast<uintptr_t>(rc.flags), rc.epoch >> 24, rc.region);
    });
    m.def("run_prod", &run_prod);
    m.def("prepare", &prepare, py::arg("cu_seqlens"));
    m.def("run", &run, "KDA prefill (Kimi-K3, MI350X)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("raw_g"), py::arg("raw_beta"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("state"), py::arg("cu_seqlens"),
          py::arg("scale"), py::arg("lower_bound"), py::arg("meta") = py::none());
}
