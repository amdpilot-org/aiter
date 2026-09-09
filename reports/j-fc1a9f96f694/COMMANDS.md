# Command record

All commands below were run in the qualified MI300X image. Network calls used bounded retries. Build caches were kept under `/job/job-artifacts`; no model weights were downloaded and no node-wide state was changed.

```bash
# Mirror and upstream context
gh issue view 71 --repo amdpilot-org/aiter --json number,title,state,body,author,createdAt,comments,url
gh issue view 4701 --repo ROCm/aiter --json number,title,state,body,author,createdAt,comments,url
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS https://api.github.com/repos/ROCm/aiter/issues/4701
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS https://api.github.com/repos/ROCm/aiter/issues/4701/comments?per_page=100
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS https://api.github.com/repos/ROCm/aiter/pulls/2664
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS -H 'Accept: application/vnd.github.v3.diff' https://api.github.com/repos/ROCm/aiter/pulls/2664

# Environment
rocm-smi --showproduct --showid
rocminfo
/opt/venv/bin/python --version
/opt/venv/bin/python -c 'import sys, torch; print(sys.executable, torch.__version__, torch.__file__, torch.version.hip)'
/opt/venv/bin/python - <<'PY'
import flydsl, flydsl.expr, importlib.metadata
print(importlib.metadata.version("flydsl"))
print(flydsl.__file__)
print(flydsl.expr.__file__)
print(hasattr(flydsl.expr, "buffer_ops"))
PY

# Delivery clone
git clone --depth 50 https://github.com/amdpilot-org/aiter.git /job/aiter
git switch -C amdpilot/j-fc1a9f96f694
git rev-parse HEAD

# Source and dispatch inspection
rg -n "flash_attn_varlen_func|is_flydsl_available|fmha_kernels|flydsl_flash_attn_varlen_func" aiter
rg -n "from flydsl|import flydsl|buffer_ops" aiter/ops/flydsl -g '*.py'
rg -n "ENABLE_CK\\s*=|def get_gfx|def is_experimental_enabled" aiter/ops/mha.py aiter/jit -g '*.py'
rg -n "flash_attn_varlen_func\\(" tests op_tests

# Historical revisions
git ls-remote --tags https://github.com/ROCm/aiter.git 'v0.1.19*' 'refs/tags/v0.1.19*'
git fetch --depth 1 https://github.com/ROCm/aiter.git refs/tags/v0.1.19:refs/tags/upstream-v0.1.19
git worktree add --detach /job/job-artifacts/aiter-v0.1.19 31350226161346314b3d8882c8085bd31dce6a34
git fetch --depth 1 https://github.com/ROCm/aiter.git refs/tags/v0.1.19.post2:refs/tags/upstream-v0.1.19.post2
git worktree add --detach /job/job-artifacts/aiter-v0.1.19.post2 a63ede724b153564f3ed3fc538055fd15178c77d
git fetch --depth 1 https://github.com/ROCm/aiter.git 37b5395edf630a9eded64e7189c84d450644e571
git worktree add --detach /job/job-artifacts/aiter-37b5395 37b5395edf630a9eded64e7189c84d450644e571
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS https://api.github.com/repos/ROCm/aiter/commits?path=aiter%2Fops%2Fflydsl%2Fkernels%2Fbuffer_ops.py&per_page=20
curl --fail --retry 3 --retry-delay 2 --max-time 30 -sS https://api.github.com/repos/ROCm/aiter/commits/37b5395edf630a9eded64e7189c84d450644e571

# Affected import and call
PYTHONPATH=/job/job-artifacts/aiter-v0.1.19 AITER_JIT_DIR=/job/job-artifacts/aiter-jit-historical /opt/venv/bin/python - <<'PY'
import aiter
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_func
PY
PYTHONPATH=/job/job-artifacts/aiter-v0.1.19 AITER_JIT_DIR=/job/job-artifacts/aiter-jit-historical ENABLE_CK=0 /opt/venv/bin/python /job/aiter/reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py

# Fix and current-main validation
PYTHONPATH=/job/job-artifacts/aiter-37b5395 AITER_JIT_DIR=/job/job-artifacts/aiter-jit-37b5395 ENABLE_CK=0 /opt/venv/bin/python /job/aiter/reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py
PYTHONPATH=/job/job-artifacts/aiter-v0.1.19.post2 AITER_JIT_DIR=/job/job-artifacts/aiter-jit-post2 ENABLE_CK=0 /opt/venv/bin/python /job/aiter/reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py
PYTHONPATH=/job/aiter AITER_JIT_DIR=/job/job-artifacts/aiter-jit-current ENABLE_CK=0 /opt/venv/bin/python reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py

# Bounded CK attempt (failed for the documented shallow-submodule reason)
PYTHONPATH=/job/aiter ENABLE_CK=1 /opt/venv/bin/python reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py

# Delivery
git status --short --branch
git diff --check
git add reports/j-fc1a9f96f694
git commit
git diff f0321c0e8927d1d90a29385433f71e592b1c51f5..HEAD > /job/recovery.patch
git push --force-with-lease origin amdpilot/j-fc1a9f96f694
gh pr create --repo amdpilot-org/aiter --base main --head amdpilot/j-fc1a9f96f694 --draft --title ... --body-file ...
```
