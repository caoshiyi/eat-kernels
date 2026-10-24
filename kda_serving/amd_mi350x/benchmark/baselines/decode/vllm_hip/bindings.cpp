// Registers vLLM's ROCm Kimi-K3 fused KDA decode kernel as torch.ops.kda_decode_sota.fused_kda_decode.
// Schema copied from vllm/csrc/libtorch_stable/torch_bindings.cpp (vLLM main 0f112d1, 2026-10-08).

#include <optional>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

void fused_kda_decode(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    std::optional<torch::stable::Tensor> bias,
    torch::stable::Tensor& conv_state, torch::stable::Tensor const& raw_g,
    torch::stable::Tensor const& raw_beta, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices, torch::stable::Tensor& state,
    torch::stable::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::stable::Tensor> output_gate,
    std::optional<torch::stable::Tensor> norm_weight, double norm_eps);

STABLE_TORCH_LIBRARY(kda_decode_sota, ops) {
  ops.def(
      "fused_kda_decode("
      "Tensor x, Tensor weight, Tensor? bias, Tensor! conv_state, "
      "Tensor raw_g, Tensor raw_beta, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor! state, Tensor! out, "
      "float? lower_bound=None, Tensor? output_gate=None, "
      "Tensor? norm_weight=None, float norm_eps=1e-5) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(kda_decode_sota, CUDA, ops) {
  ops.impl("fused_kda_decode", TORCH_BOX(&fused_kda_decode));
}
