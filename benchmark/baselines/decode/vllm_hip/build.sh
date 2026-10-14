#!/bin/bash
# Builds vLLM's ROCm Kimi-K3 fused KDA decode kernel into baselines/decode/vllm_hip/libkda_decode_sota.so.
# Run once on the MI350X host with the benchmark's Python environment active (needs hipcc and torch).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=$HERE/build
mkdir -p "$BUILD/kimi_k3"
cp "$HERE/fused_kda_decode_kernel_rocm.cu" "$BUILD/kimi_k3/"
cp "$HERE/torch_utils.h" "$BUILD/torch_utils.h"

TORCH_DIR=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")
FLAGS=(-O3 -std=c++20 -fPIC --offload-arch=gfx950 -DUSE_ROCM -D__HIP_PLATFORM_AMD__=1
       -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include")

hipcc "${FLAGS[@]}" -x hip -c "$BUILD/kimi_k3/fused_kda_decode_kernel_rocm.cu" -o "$BUILD/decode.o" &
hipcc "${FLAGS[@]}" -c "$HERE/bindings.cpp" -o "$BUILD/bindings.o" &
wait
hipcc -shared -fPIC --offload-arch=gfx950 "$BUILD/decode.o" "$BUILD/bindings.o" \
  -L"$TORCH_DIR/lib" -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip \
  -Wl,-rpath,"$TORCH_DIR/lib" -o "$HERE/libkda_decode_sota.so"
echo "built $HERE/libkda_decode_sota.so"
