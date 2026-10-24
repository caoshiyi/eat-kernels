#!/bin/bash
# Builds the two HIP baselines the eval times against (the AITER decode baseline is Triton and needs no build):
#   baselines/prefill/libkda_sota.so              vLLM's ROCm Kimi-K3 fused KDA prefill (fused_kda_prologue + fused_kda_chunk)
#   baselines/decode/vllm_hip/libkda_decode_sota.so  vLLM's ROCm Kimi-K3 fused KDA decode
# Needs hipcc (ROCm 7.2.0) and torch in the active Python environment. Takes ~1-2 min.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/baselines/prefill/build.sh"
bash "$HERE/baselines/decode/vllm_hip/build.sh"
