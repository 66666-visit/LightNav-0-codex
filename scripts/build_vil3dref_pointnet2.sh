#!/usr/bin/env bash
# Build the official ViL3DRel PointNet++ CUDA extension in-place.
#
# This never installs into the LightNav/Habitat environment.  It expects a
# CUDA-enabled PyTorch interpreter whose Python/torch ABI matches the LightNav
# inference process (or a venv with the same torch), plus a CUDA toolkit with nvcc:
#
#   CUDA_HOME=/usr/local/cuda PYTHON=/path/to/cuda-venv/bin/python \
#     bash scripts/build_vil3dref_pointnet2.sh
#
# The official source is used unchanged.  The in-place `pointnet2` package is
# added to sys.path by lightnav.memory.l3_instance_grounder; no pip install is
# performed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIL_ROOT="$REPO_ROOT/references/vil3dref/og3d_src"
EXT_ROOT="$VIL_ROOT/model/external_tools/pointnet2"
PYTHON=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d "$EXT_ROOT" ]; then
    echo "[vil3dref] ERROR: official pointnet2 source missing: $EXT_ROOT" >&2
    exit 1
fi
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "[vil3dref] ERROR: nvcc not found under CUDA_HOME=$CUDA_HOME" >&2
    echo "[vil3dref]        set CUDA_HOME to a CUDA toolkit that matches the PyTorch wheel" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "[vil3dref] ERROR: this Python's torch has no usable CUDA device; "
        "the extension can be compiled but C/D require a GPU"
    )
print("[vil3dref] torch", torch.__version__, "cuda", torch.version.cuda)
PY

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-2}
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    # Match the actual device instead of assuming one architecture.
    TORCH_CUDA_ARCH_LIST=$("$PYTHON" - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)
fi
export TORCH_CUDA_ARCH_LIST
# setup.py uses a relative `-I_ext_src/include`; ninja runs from the build
# directory, so add the official include directory absolutely for both the
# host compiler and nvcc.
export CPLUS_INCLUDE_PATH="$EXT_ROOT/_ext_src/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$EXT_ROOT/_ext_src/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export NVCC_PREPEND_FLAGS="-I$EXT_ROOT/_ext_src/include ${NVCC_PREPEND_FLAGS:-}"

cd "$EXT_ROOT"
echo "[vil3dref] building official pointnet2 in-place: $EXT_ROOT"
"$PYTHON" setup.py build_ext --inplace
"$PYTHON" - <<'PY'
import sys
sys.path.insert(0, ".")
import pointnet2_utils
import torch
xyz = torch.rand(1, 256, 3, device="cuda", dtype=torch.float32)
idx = pointnet2_utils.furthest_point_sample(xyz, 16)
assert tuple(idx.shape) == (1, 16), idx.shape
print("[vil3dref] CUDA op smoke ok:", tuple(idx.shape))
PY
echo "[vil3dref] PASS: official pointnet2 CUDA extension built and executed"
