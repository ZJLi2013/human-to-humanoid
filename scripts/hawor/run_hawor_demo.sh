#!/usr/bin/env bash
# Unattended HaWoR end-to-end demo on ROCm (MI300X / gfx942).
# Runs INSIDE the persistent `hawor_demo` container (torch 2.10 + rocm7.2).
# Staged + idempotent (sentinels in /work/.demo_state). Produces a rendered
# hand-motion video via aitviewer headless (Xvfb + llvmpipe).
set -uo pipefail

REPO=/work/HaWoR
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_ROCM_ARCH=gfx942
# GPU0 on this shared node is saturated (96% VRAM); use an idle device.
export HIP_VISIBLE_DEVICES=1
export HSA_OVERRIDE_GFX_VERSION=9.4.2
# torch 2.6+ defaults torch.load(weights_only=True) which rejects the Lightning
# checkpoints (omegaconf DictConfig). Weights are trusted (self-downloaded).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# cv2/matplotlib pull in Qt; force offscreen so they don't need the xcb plugin
# (aitviewer's GL still uses the Xvfb GLX context, unaffected).
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

STATE=/work/.demo_state
mkdir -p "$STATE"

# pkg_resources (needed by pytorch_lightning/lightning_fabric) ships with
# setuptools<81; ensure it every run (cheap, idempotent).
pip install -q "setuptools<81" wheel >/dev/null 2>&1 || true
# Metric3D needs mmcv.utils (Config/collect_env). mmcv 1.x `mmcv` pkg is the
# pure-python build (no compiled ops), which is all Metric3D imports here.
python -c "import mmcv" 2>/dev/null || pip install -q --no-build-isolation "mmcv==1.7.2" >/dev/null 2>&1 || true
# ffmpeg for frame extraction (detect_track_video -> extract_frames)
command -v ffmpeg >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null 2>&1; }
done_() { [ -f "$STATE/$1" ]; }
mark()  { touch "$STATE/$1"; }

DROID=thirdparty/DROID-SLAM
LIET=$DROID/thirdparty/lietorch

echo "############ STAGE 1: python deps ############"
if done_ deps; then echo "  [skip] deps already installed"; else
  python -m pip install -q --upgrade pip 2>&1 | tail -1
  # numpy<2 required by chumpy / smplx / old stack (HaWoR pins 1.26.4)
  pip install -q "numpy==1.26.4" 2>&1 | tail -2
  pip install -q \
    opencv-python scikit-image "smplx==0.1.28" yacs timm einops xtcocotools \
    pandas hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils rich \
    webdataset ultralytics pulp supervision pycocotools joblib natsort evo \
    pytorch-minimize "mmengine==0.10.4" HTML4Vision plyfile easydict loguru \
    dill lapx "pytorch-lightning==2.2.4" lightning-utilities "torchmetrics==1.4.0" \
    trimesh pyrender "moderngl-window==2.4.6" "aitviewer==1.13.0" \
    "huggingface_hub[cli]" gdown 2>&1 | tail -6
  # chumpy (MANO loader) -- old setup.py imports pip inside the build-isolation env
  # (ModuleNotFoundError: pip); --no-build-isolation uses outer env. Then patch numpy import.
  pip install -q --no-build-isolation "chumpy@git+https://github.com/mattloper/chumpy" 2>&1 | tail -3 || true
  CH=$(python -c "import chumpy,os;print(os.path.dirname(chumpy.__file__))" 2>/dev/null)
  if [ -n "$CH" ] && [ -f "$CH/__init__.py" ]; then
    sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CH/__init__.py"
    echo "  patched chumpy numpy import at $CH/__init__.py"
  fi
  # torch_scatter ROCm wheel (prebuilt)
  pip install -q /work/torch_scatter_rocm-2.1.2-cp310-cp310-linux_x86_64.whl 2>&1 | tail -2
  pip install -q ninja 2>&1 | tail -1
  mark deps
fi
python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__,'gpu',torch.cuda.is_available())" \
  || { echo "DEMO_FAIL torch/numpy import"; exit 1; }

echo "############ STAGE 2: build lietorch + droid_backends ############"
if done_ build && python -c "import torch,lietorch,droid_backends" 2>/dev/null; then
  echo "  [skip] extensions already built & importable"
else
  echo "  applying ROCm patches P1-P5"
  sed -i '/gencode=arch=compute/d' $DROID/setup.py
  sed -i '/gencode=arch=compute/d' $LIET/setup.py
  sed -i 's/Matrix4x4()\.block<3,3>/Matrix4x4().template block<3,3>/' $LIET/lietorch/src/lietorch_gpu.cu
  sed -i 's/\.type()/.scalar_type()/g' $DROID/src/*.cu 2>/dev/null
  # P6: lietorch DISPATCH call sites x.type() -> x.scalar_type() (else "cannot convert to c10::ScalarType").
  # Match any /DISPATCH/ line (lietorch DISPATCH_GROUP_AND_FLOATING_TYPES + torch AT_DISPATCH_*), covering
  # src/ and extras/ (corr_index_kernel). Line-scoped so x.device().type() (DeviceType) is untouched.
  sed -i '/DISPATCH/ s/\.type()/.scalar_type()/g' $LIET/lietorch/src/*.cu $LIET/lietorch/src/*.cpp $LIET/lietorch/extras/*.cu $LIET/lietorch/extras/*.cpp 2>/dev/null
  grep -rl 'EIGEN_USING_STD(fill_n)' $DROID/thirdparty/eigen/Eigen 2>/dev/null \
    | xargs -r sed -i 's/EIGEN_USING_STD(fill_n);/using std::fill_n;/g'
  echo "  building lietorch"
  PYTORCH_ROCM_ARCH=gfx942 pip install --no-build-isolation $LIET > /work/build_lietorch.log 2>&1
  python -c "import lietorch" || { echo "DEMO_FAIL lietorch build"; tail -30 /work/build_lietorch.log; exit 1; }
  echo "  building droid_backends"
  find $DROID -maxdepth 2 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null
  ( cd $DROID && rm -rf build && PYTORCH_ROCM_ARCH=gfx942 python setup.py install > /work/build_droid.log 2>&1 )
  python -c "import torch,droid_backends" || { echo "DEMO_FAIL droid_backends build"; tail -30 /work/build_droid.log; exit 1; }
  mark build
fi
echo "  extensions OK"

echo "############ STAGE 3: download weights ############"
if done_ weights; then echo "  [skip] weights present"; else
  echo "  downloading ThunderVVV/HaWoR (all model weights) via hf-mirror"
  hf download ThunderVVV/HaWoR --repo-type model \
    --local-dir /work/hf_hawor > /work/dl_hawor.log 2>&1 \
    || { echo "DEMO_FAIL hf download HaWoR"; tail -20 /work/dl_hawor.log; exit 1; }
  # place into repo-expected paths
  mkdir -p weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights
  cp -f /work/hf_hawor/external/detector.pt                        weights/external/
  cp -f /work/hf_hawor/external/droid.pth                          weights/external/
  cp -f /work/hf_hawor/hawor/checkpoints/hawor.ckpt               weights/hawor/checkpoints/
  cp -f /work/hf_hawor/hawor/checkpoints/infiller.pt              weights/hawor/checkpoints/
  cp -f /work/hf_hawor/hawor/model_config.yaml                    weights/hawor/
  cp -f /work/hf_hawor/external/metric_depth_vit_large_800k.pth   thirdparty/Metric3D/weights/

  echo "  downloading MANO (lithiumice/models_hub) via hf-mirror"
  hf download lithiumice/models_hub \
    4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl 4_SMPLhub/MANO/pkl/MANO_LEFT.pkl \
    --local-dir /work/hf_mano > /work/dl_mano.log 2>&1 \
    || { echo "DEMO_FAIL hf download MANO"; tail -20 /work/dl_mano.log; exit 1; }
  mkdir -p _DATA/data/mano _DATA/data_left/mano_left
  cp -f /work/hf_mano/4_SMPLhub/MANO/pkl/MANO_RIGHT.pkl _DATA/data/mano/MANO_RIGHT.pkl
  cp -f /work/hf_mano/4_SMPLhub/MANO/pkl/MANO_LEFT.pkl  _DATA/data_left/mano_left/MANO_LEFT.pkl
  mark weights
fi
echo "  weights layout:"
ls -la weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights _DATA/data/mano _DATA/data_left/mano_left 2>&1 | grep -E '\.pth|\.pt|\.ckpt|\.pkl|\.yaml' | head -20

echo "############ STAGE 3b: neutralize pytorch3d hand-mask render (ROCm) ############"
python - <<'PY'
import re
f = "scripts/scripts_test_video/hawor_video.py"
s = open(f).read()
if "ROCm stub" in s:
    print("  already patched")
else:
    s = s.replace("from lib.vis.renderer import Renderer",
                  "Renderer = None  # ROCm stub (pytorch3d hand-mask render disabled)")
    s = re.sub(r"renderer = Renderer\(.*?max_faces_per_bin=max_faces_per_bin\)",
               "renderer = None  # ROCm: hand-mask render disabled", s, flags=re.S)
    s = re.sub(r'vertices = outputs\["vertices"\]\[0\]\.cpu\(\).*?model_masks\[frame_ck\[img_i\]\] \+= mask',
               "pass  # ROCm: pytorch3d hand-mask render skipped (SLAM runs unmasked)", s, flags=re.S)
    open(f, "w").write(s)
    print("  patched hawor_video.py (pytorch3d neutralized)")
PY

echo "############ STAGE 4: run demo (Xvfb + llvmpipe) ############"
pkill Xvfb 2>/dev/null || true
Xvfb :99 -screen 0 1280x720x24 >/work/xvfb.log 2>&1 &
sleep 3
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export PYOPENGL_PLATFORM=glx

VIDEO="${DEMO_VIDEO:-./example/video_0.mp4}"
echo "  running: python demo.py --video_path $VIDEO --vis_mode world"
python demo.py --video_path "$VIDEO" --vis_mode world > /work/demo_run.log 2>&1
RC=$?
echo "  demo.py exit rc=$RC (tail below)"
tail -25 /work/demo_run.log

echo "############ STAGE 5: locate output video ############"
FOUND=$(find "$REPO" example -type f -name '*.mp4' -newer "$STATE/weights" 2>/dev/null | grep -iE 'aitviewer|vis_' | head -5)
echo "  produced videos:"
echo "$FOUND"
if [ -n "$FOUND" ]; then
  echo "$FOUND" | head -1 > /work/DEMO_VIDEO_PATH.txt
  echo "DEMO_DONE_OK"
else
  echo "DEMO_DONE_NO_VIDEO rc=$RC"
fi
