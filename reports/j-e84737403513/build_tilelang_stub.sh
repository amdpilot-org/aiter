#!/usr/bin/env bash
set -euo pipefail

source_root=${1:?usage: build_tilelang_stub.sh TILELANG_SOURCE OUTPUT}
output=${2:?usage: build_tilelang_stub.sh TILELANG_SOURCE OUTPUT}

mkdir -p "$(dirname "$output")"
clang++ -std=c++17 -shared -fPIC -O2 \
  -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__=1 \
  -DTILELANG_HIP_STUB_EXPORTS \
  -I/opt/rocm/include \
  "$source_root/src/backend/rocm/stubs/hip.cc" \
  -ldl -o "$output"

readelf -Ws "$output" | grep -q 'hipGetDevicePropertiesR0600'
