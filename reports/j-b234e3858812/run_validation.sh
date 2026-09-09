#!/usr/bin/env bash

set -uo pipefail

ROOT=${AITER_VALIDATION_ROOT:-/job/aiter-validation}
CACHE=${AITER_BUILD_CACHE:-/job/aiter-build-cache}
RAW=${AITER_RAW_DIR:-$CACHE/raw}

export AITER_JIT_DIR=$CACHE/jit
export TRITON_CACHE_DIR=$CACHE/triton
export TORCH_EXTENSIONS_DIR=$CACHE/torch-extensions
export TMPDIR=$CACHE/tmp
export PYTHONPATH=$ROOT

mkdir -p "$RAW" "$CACHE/jit" "$CACHE/triton" "$CACHE/torch-extensions" "$CACHE/tmp"

run_harness() {
  local harness=$1
  shift
  python -c '
import pandas as pd
import runpy
import sys

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$harness" "$@"
}

cd "$ROOT"

for seq in 128 512; do
  for dims in "128 128" "192 128" "128 64" "256 256"; do
    set -- $dims
    dqk=$1
    dv=$2
    for causal in 0 1; do
      if [ "$causal" -eq 1 ]; then
        causal_arg=-c
      else
        causal_arg=
      fi
      name="fixed_no_lse_s${seq}_d${dqk}_${dv}_causal${causal}"
      run_harness op_tests/test_mha_fp8.py \
        -b 1 -n 8 -nk 8 -q "$seq" -k "$seq" -d "$dqk" -dv "$dv" $causal_arg \
        > "$RAW/$name.log" 2>&1
      printf 'fixed\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$dqk" "$dv" "$causal" "$?" \
        >> "$RAW/fixed_no_lse_results.tsv"
    done
  done
done

for dims in "128 128" "192 128" "128 64" "256 256"; do
  set -- $dims
  dqk=$1
  dv=$2
  for causal in 0 1; do
    if [ "$causal" -eq 1 ]; then
      causal_arg=-c
    else
      causal_arg=
    fi
    name="fixed_lse_s512_d${dqk}_${dv}_causal${causal}"
    run_harness op_tests/test_mha_fp8.py \
      -b 1 -n 8 -nk 8 -q 512 -k 512 -d "$dqk" -dv "$dv" --lse $causal_arg \
      > "$RAW/$name.log" 2>&1
    printf 'fixed_lse\t512\t%s\t%s\t%s\t%s\n' "$dqk" "$dv" "$causal" "$?" \
      >> "$RAW/fixed_lse_512_results.tsv"
  done
done

for seq in 128 512; do
  for dims in "128 128" "192 128" "128 64" "256 256"; do
    set -- $dims
    dqk=$1
    dv=$2
    for causal in 0 1; do
      if [ "$causal" -eq 1 ]; then
        causal_arg=-c
      else
        causal_arg=
      fi
      name="varlen_no_lse_s${seq}_d${dqk}_${dv}_causal${causal}"
      run_harness op_tests/test_mha_varlen_fp8.py \
        -b 1 -n 8 -nk 8 -q "$seq" -k "$seq" -d "$dqk" -dv "$dv" $causal_arg \
        > "$RAW/$name.log" 2>&1
      printf 'varlen\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$dqk" "$dv" "$causal" "$?" \
        >> "$RAW/varlen_no_lse_results.tsv"
    done
  done
done

for seq in 128 512; do
  for dims in "128 128" "192 128" "128 64" "256 256"; do
    set -- $dims
    dqk=$1
    dv=$2
    for causal in 0 1; do
      if [ "$causal" -eq 1 ]; then
        causal_arg=-c
      else
        causal_arg=
      fi
      name="varlen_lse_s${seq}_d${dqk}_${dv}_causal${causal}"
      run_harness op_tests/test_mha_varlen_fp8.py \
        -b 1 -n 8 -nk 8 -q "$seq" -k "$seq" -d "$dqk" -dv "$dv" --lse $causal_arg \
        > "$RAW/$name.log" 2>&1
      printf 'varlen_lse\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$dqk" "$dv" "$causal" "$?" \
        >> "$RAW/varlen_lse_results.tsv"
    done
  done
done
