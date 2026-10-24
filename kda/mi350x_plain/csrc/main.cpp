#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
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
    auto workspace = torch::empty({kda_workspace_floats(total_seq_len, num_seqs, num_heads, hmeta)},
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
        dmeta, hmeta,
        at::hip::getCurrentHIPStream().stream());

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare", &prepare, py::arg("cu_seqlens"));
    m.def("run", &run, "KDA prefill (Kimi-K3, MI350X)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("raw_g"), py::arg("raw_beta"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("state"), py::arg("cu_seqlens"),
          py::arg("scale"), py::arg("lower_bound"), py::arg("meta") = py::none());
}
