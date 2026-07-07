#!/usr/bin/env bash
# HaWoR P1 (scoped): functional GPU smoke of the ROCm-ported masked DROID-SLAM
# `droid_backends` on gfx942 (MI300X). Reinstalls lietorch + droid_backends +
# torch_scatter inside a fresh ROCm container (previous build container was --rm),
# applying the same 5 ROCm patches, then runs hawor_droid_smoke.py.
set -uo pipefail

RUN_ID="${RUN_ID:-hawor_droid_smoke_$(date +%Y%m%d_%H%M%S)}"
WORK="${WORK:-/tmp/overnight-tests/hawor}"
REPO="${WORK}/HaWoR"
LOG="${WORK}/${RUN_ID}.log"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0}"
SCATTER_WHL_DIR="/tmp/overnight-tests/droid/pyg"

echo "[host] RUN_ID=${RUN_ID} WORK=${WORK} IMAGE=${IMAGE}" | tee "${LOG}"
[ -d "${REPO}/.git" ] || { echo "[host] ERROR: ${REPO} not cloned"; exit 1; }
[ -f "${WORK}/hawor_droid_smoke.py" ] || { echo "[host] ERROR: smoke py missing in ${WORK}"; exit 1; }
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
python -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip,'avail',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')" || { echo HAWOR_SMOKE_FAIL torch-import; exit 1; }

echo "===== [ctr] apply ROCm patches (P1-P5) ====="
sed -i '/gencode=arch=compute/d' $DROID/setup.py && echo "  P1 masked DROID-SLAM setup.py -gencode"
sed -i '/gencode=arch=compute/d' $LIET/setup.py && echo "  P2 lietorch setup.py -gencode"
sed -i 's/Matrix4x4()\.block<3,3>/Matrix4x4().template block<3,3>/' $LIET/lietorch/src/lietorch_gpu.cu && echo "  P3 lietorch_gpu.cu .template"
sed -i 's/\.type()/.scalar_type()/g' $DROID/src/*.cu 2>/dev/null && echo "  P4 masked kernels .type()->.scalar_type()"
grep -rl 'EIGEN_USING_STD(fill_n)' $DROID/thirdparty/eigen/Eigen 2>/dev/null | xargs -r sed -i 's/EIGEN_USING_STD(fill_n);/using std::fill_n;/g' && echo "  P5 eigen EIGEN_USING_STD(fill_n)->using std::fill_n"

echo "===== [ctr] install deps (ninja + torch_scatter wheel) ====="
pip install -q ninja 2>&1 | tail -2
pip install -q /pyg/torch_scatter*.whl 2>&1 | tail -3

echo "===== [ctr] build+install lietorch ====="
PYTORCH_ROCM_ARCH=gfx942 pip install --no-build-isolation $LIET 2>&1 | tail -20
python -c "import lietorch;from lietorch import SE3;print('lietorch import OK')" || { echo HAWOR_SMOKE_FAIL lietorch-import; exit 1; }

echo "===== [ctr] build+install masked droid_backends ====="
find $DROID -maxdepth 2 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
( cd $DROID && rm -rf build && PYTORCH_ROCM_ARCH=gfx942 python setup.py install > /tmp/droid_build.log 2>&1 )
grep -n 'FAILED:\|error:' /tmp/droid_build.log | head -20
python -c "import torch,droid_backends;print('droid_backends import OK')" || { echo HAWOR_SMOKE_FAIL droid_backends-import; exit 1; }

echo "===== [ctr] run functional GPU smoke ====="
( cd $DROID && python /work/hawor_droid_smoke.py )
INNER

echo "[host] done RUN_ID=${RUN_ID}" | tee -a "${LOG}"
