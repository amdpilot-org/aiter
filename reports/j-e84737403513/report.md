# TileLang/AITER HIP symbol ownership investigation

## Result

The reported defect is real on the assigned AMD Instinct MI300X (`gfx942`).
When TileLang `v0.1.10`'s `libhip_stub.so` is loaded with `RTLD_GLOBAL`
before AITER imports its FMHA JIT extension, the extension binds
`hipGetDevicePropertiesR0600` and `hipModuleLaunchKernel` to the stub. The
same real GPU invocation then fails with:

```text
RuntimeError: invalid argument for fmha_v3_varlen_fwd
```

Loading the same stub with `RTLD_LOCAL`, loading it after the AITER extension,
or omitting it entirely leaves the extension bound to
`/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200` and the operation passes.

This PR adds a bounded subprocess reproducer and a minimal, validated fix for
the affected pybind extension loader. It does not add linker flags, locks, or
import-order guards.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Container hostname: `banff-cyxtera-cx57-5` (not used as image identity)
- OS: Ubuntu 22.04.5 LTS
- GPU: one AMD Instinct MI300X, serial `692440003992`, `gfx942`, 304 CUs
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- PyTorch HIP: `7.2.26015-fc0010cf6a`
- `hipcc`: HIP `7.2.26015-fc0010cf6a`, AMD clang `22.0.0git`
- Tested AITER revision: `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- TileLang stub source: tag `v0.1.10`, commit
  `69bc43e2cfbc38976b0c5e1c4b2299b66e8970ee`
- Preinstalled AITER source recorded separately: `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- Preinstalled TileLang source recorded separately: `a55a82302bf7f3c5af635b5c9146f728185cc900`

The preinstalled TileLang Python package reports `0.1.7.post3` and does not
contain `libhip_stub.so`. That is a missing-condition limitation, not the
interposition defect. The investigation therefore built the exact stub from
the `v0.1.10` source tag rather than treating the missing stub as a failure.

## Native provenance

| Object | Path | SHA-256 |
| --- | --- | --- |
| AITER FMHA extension | `/job/aiter-jit-cache/module_fmha_v3_varlen_fwd.so` | `d572524ded0b4d7ca875794b9b74b4ffa51e92c56439dbf5b39fb4e9bdda63af` |
| TileLang HIP stub | `/job/tilelang-stub-build/libhip_stub.so` | `9d5e3d49de2fcf2c537580780a4681054ed98c422dc7979d3ad306253e08bc3c` |
| Real HIP runtime | `/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200` | `36fb0b132fafa496996d2c5a433ed80f8fb3b10e8ea7966e67241d2ef89dc39d` |
| Torch HIP library | `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so` | `ebb6787911956c0f148b95f98330fc82ef1be423ae68ddea1e955f72e100aa72` |

`readelf -Ws` shows that the FMHA extension has undefined
`hipGetDevicePropertiesR0600@hip_6.0`, `hipModuleLaunchKernel@hip_4.2`,
`hipGetDevice@hip_4.2`, and `hipGetDeviceCount@hip_4.2`. The TileLang stub
exports all four names. The real HIP runtime exports
`hipGetDevicePropertiesR0600@@hip_6.0`.

Runtime ownership was recorded from `LD_DEBUG=bindings` for the extension and
from `dladdr` for the global, stub, and real-HIP handles. `/proc/<pid>/maps`
paths were also captured by the reproducer.

## Bounded matrix

Every case runs in a separate subprocess with a 300-second timeout. Each case
performs a nonzero float32 reference GPU computation and then calls AITER
`fmha_v3_varlen_fwd` on a `128 x 8 x 128` bf16 input with
`cu_seqlens = [0, 128]`. The reference is float32 softmax attention over the
same `q = k = v`. The accuracy gate is maximum absolute error below `0.05`.

### Without the fix (`AITER_DISABLE_DEEPBIND=1`)

| Case | `hipGetDevicePropertiesR0600` owner | AITER GPU result |
| --- | --- | --- |
| clean | real HIP | pass, max error `0.009514331817626953` |
| global stub before AITER | TileLang stub | fail, `invalid argument for fmha_v3_varlen_fwd` |
| local stub before AITER | real HIP | pass, max error `0.009514331817626953` |
| AITER extension before global stub | real HIP | pass, max error `0.009514331817626953` |

In the failing case, `LD_DEBUG=bindings` also records the extension binding
`hipModuleLaunchKernel`, `hipGetDevice`, and `hipGetDeviceCount` to the stub.
The float32 reference GPU operation still succeeds, so this is not a
missing-dependency or unavailable-device failure.

### With the fix

| Case | `hipGetDevicePropertiesR0600` owner | AITER GPU result |
| --- | --- | --- |
| clean | real HIP | pass, max error `0.009514331817626953` |
| global stub before AITER | real HIP | pass, max error `0.009514331817626953` |
| local stub before AITER | real HIP | pass, max error `0.009514331817626953` |
| AITER extension before global stub | real HIP | pass, max error `0.009514331817626953` |

All four protected cases bind the four recorded HIP symbols to
`/opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200`.

## Fix

`aiter/jit/core.py` now opens a pybind extension with
`RTLD_NOW | RTLD_DEEPBIND` before `importlib.import_module`. glibc deduplicates
the later `dlopen` by inode, so the import reuses the deep-bound handle. The
handle is retained because dropping a `ctypes.CDLL` wrapper can `dlclose` the
module while Python still references it.

The change is per-object and does not call `sys.setdlopenflags()`.
`AITER_DISABLE_DEEPBIND=1` is a process-start opt-out for tools such as ASan or
intentional `LD_PRELOAD` interposition. No linker flags or import-order guards
were added.

Upstream ROCm/aiter PR #4566 proposes broader `RTLD_DEEPBIND` coverage,
including standalone libraries and the C++ `SharedLibrary` helper. It was open
and unmerged when this branch was cut, and current `main` did not contain the
loader change. This PR intentionally includes only the pybind path reproduced
and validated here.

## Reproduction

From an AITER checkout with TileLang `v0.1.10` available as
`/path/to/tilelang`, use job-private caches:

```bash
mkdir -p /tmp/aiter-jit-cache /tmp/torch-extensions /tmp/stub-build

./reports/j-e84737403513/build_tilelang_stub.sh \
  /path/to/tilelang /tmp/stub-build/libhip_stub.so

python reports/j-e84737403513/reproduce_hip_interposition.py \
  --aiter-root "$PWD" \
  --stub-path /tmp/stub-build/libhip_stub.so \
  --jit-dir /tmp/aiter-jit-cache \
  --torch-extensions-dir /tmp/torch-extensions \
  --output-dir /tmp/aiter-results \
  --deepbind off \
  --timeout 300

python reports/j-e84737403513/reproduce_hip_interposition.py \
  --aiter-root "$PWD" \
  --stub-path /tmp/stub-build/libhip_stub.so \
  --jit-dir /tmp/aiter-jit-cache \
  --torch-extensions-dir /tmp/torch-extensions \
  --output-dir /tmp/aiter-results \
  --deepbind on \
  --timeout 300
```

The script writes `results-off.json`, `results-on.json`, and one
`LD_DEBUG=bindings` log per case.

## Limits

- Only the assigned single MI300X (`gfx942`) was tested.
- Only the AITER pybind FMHA JIT path was used for the numerical defect
  validation. Standalone `ctypes` and C++ `SharedLibrary` paths were not changed
  or claimed fixed by this PR.
- No end-to-end vLLM run was performed; the issue's smaller FMHA reproducer is
  the validated scope.
- The exact TileLang `v0.1.10` stub was built from source. The image's
  `0.1.7.post3` package lacks that stub and was not used as a substitute.
