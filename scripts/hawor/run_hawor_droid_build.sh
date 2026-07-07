#!/usr/bin/env bash
# HaWoR P0: build its *masked* DROID-SLAM + lietorch on gfx942 (MI308X).
# HaWoR embeds a modified DROID-SLAM (thirdparty/DROID-SLAM) -> we apply the same
# ROCm patches as SM-5 in-place (do NOT swap the whole submodule). Runs on the GPU
# node; drives a ROCm PyTorch container via stdin heredoc (docker run -i).
set -uo pipefail

RUN_ID="${RUN_ID:-hawor_droid_build_$(date +%Y%m%d_%H%M%S)}"
WORK="${WORK:-/tmp/overnight-tests/hawor}"
REPO="${WORK}/HaWoR"
LOG="${WORK}/${RUN_ID}.log"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0}"
# reuse the torch_scatter wheel already fetched for SM-5
SCATTER_WHL_DIR="/tmp/overnight-tests/droid/pyg"
SCATTER_ZIP="https://github.com/Looong01/pyg-rocm-build/releases/download/15/torch-2.10.0-rocm-7.1-py310-linux_x86_64.zip"

echo "[host] RUN_ID=${RUN_ID} WORK=${WORK} IMAGE=${IMAGE}" | tee "${LOG}"
[ -d "${REPO}/.git" ] || { echo "[host] ERROR: ${REPO} not cloned"; exit 1; }

# torch_scatter wheel: reuse SM-5's, else fetch+unzip on host (container lacks unzip)
if ! ls "${SCATTER_WHL_DIR}"/torch_scatter*.whl >/dev/null 2>&1; then
  echo "[host] fetching torch_scatter wheel" | tee -a "${LOG}"
  mkdir -p "${SCATTER_WHL_DIR}"
  curl -sL "${SCATTER_ZIP}" -o "${SCATTER_WHL_DIR}/pyg.zip" && unzip -o -q "${SCATTER_WHL_DIR}/pyg.zip" -d "${SCATTER_WHL_DIR}"
fi
ls "${SCATTER_WHL_DIR}"/torch_scatter*.whl 2>&1 | tee -a "${LOG}"

echo "[host] launching container" | tee -a "${LOG}"
docker run -i --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -e HIP_VISIBLE_DEVICES=0 -e PYTORCH_ROCM_ARCH=gfx942 \
  -v "${WORK}:/work" -v "${SCATTER_WHL_DIR}:/pyg" -w /work/HaWoR \
  "${IMAGE}" bash -s 2>&1 <<'INNER' | tee -a "${LOG}"
set -uo pipefail
DROID=thirdparty/DROID-SLAM
LIET=$DROID/thirdparty/lietorch

echo "===== [ctr] env ====="
python -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip,'avail',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')" || { echo HAWOR_BUILD_FAIL torch-import; exit 1; }

echo "===== [ctr] apply ROCm patches to masked DROID-SLAM ====="
# P2a: masked DROID-SLAM's OWN setup.py hardcodes -gencode=arch=compute_* in the
# nvcc args; clang++ (hipcc backend) rejects them -> strip on ROCm (arch from
# PYTORCH_ROCM_ARCH=gfx942). vanilla DROID-SLAM lacked these, HaWoR's mask added them.
sed -i '/gencode=arch=compute/d' $DROID/setup.py && echo "  patched masked DROID-SLAM setup.py (removed -gencode)"
# P2b: same gate for lietorch setup.py (no-op if the fork already gates it)
sed -i '/gencode=arch=compute/d' $LIET/setup.py && echo "  patched lietorch setup.py (removed -gencode)"
# P3: dependent member-template call needs the template keyword under clang/hipcc
sed -i 's/Matrix4x4()\.block<3,3>/Matrix4x4().template block<3,3>/' $LIET/lietorch/src/lietorch_gpu.cu && echo "  patched lietorch_gpu.cu (.template)"
grep -c 'template block<3,3>' $LIET/lietorch/src/lietorch_gpu.cu
# P4: masked DROID-SLAM kernels use the deprecated x.type() in AT_DISPATCH_* macros;
# newer torch's DeprecatedTypeProperties no longer converts to ScalarType -> use .scalar_type().
sed -i 's/\.type()/.scalar_type()/g' $DROID/src/*.cu 2>/dev/null && echo "  patched masked kernels .type()->.scalar_type()"
# P5: bundled Eigen uses EIGEN_USING_STD(fill_n); under hipcc (__HIPCC__/__CUDACC__ set)
# that macro expands to `using ::fill_n;` (global ns) which does not exist -> force std::.
grep -rl 'EIGEN_USING_STD(fill_n)' $DROID/thirdparty/eigen/Eigen 2>/dev/null | xargs -r sed -i 's/EIGEN_USING_STD(fill_n);/using std::fill_n;/g' && echo "  patched eigen EIGEN_USING_STD(fill_n) -> using std::fill_n"

echo "===== [ctr] build tools + torch_scatter wheel ====="
pip install -q ninja 2>&1 | tail -2
pip install -q /pyg/torch_scatter*.whl 2>&1 | tail -3
python -c "import torch_scatter;from torch_scatter import scatter_mean;print('torch_scatter OK',torch_scatter.__version__)" || { echo HAWOR_BUILD_FAIL torch_scatter; exit 1; }

echo "===== [ctr] build lietorch (hipify) ====="
PYTORCH_ROCM_ARCH=gfx942 pip install --no-build-isolation $LIET 2>&1 | tail -40
python -c "import lietorch;from lietorch import SE3;print('lietorch import OK')" || { echo HAWOR_BUILD_FAIL lietorch-import; exit 1; }

echo "===== [ctr] build masked droid_backends (hipify, setup.py install) ====="
# HaWoR's documented path is `python setup.py install`, which bypasses the PEP517
# build_meta metadata hook whose _bubble_up_info_directory asserts on the >1
# in-tree .egg-info (lietorch.egg-info + droid_backends.egg-info) in this repo.
find $DROID -maxdepth 2 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
( cd $DROID && rm -rf build && PYTORCH_ROCM_ARCH=gfx942 python setup.py install > /tmp/droid_backends_build.log 2>&1 )
echo "----- [ctr] droid_backends FAILED/error lines -----"
grep -n 'FAILED:\|error:\|fatal error' /tmp/droid_backends_build.log | head -40
echo "----- [ctr] droid_backends build tail -----"
tail -25 /tmp/droid_backends_build.log
python -c "import torch,droid_backends;print('droid_backends import OK')" || { echo HAWOR_BUILD_FAIL droid_backends-import; exit 1; }

echo "HAWOR_DROID_BUILD_PASS"
INNER

echo "[host] done RUN_ID=${RUN_ID}" | tee -a "${LOG}"
