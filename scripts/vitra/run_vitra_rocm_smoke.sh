#!/usr/bin/env bash
# VITRA VLA (PaliGemma2-3B + DiT action head) ROCm inference smoke.
# Wh0 policy core (gap analysis SM-8). Uses the README "minimum usage" path:
# load_model + predict_action on one image. Avoids deepspeed (training-only) and
# MANO/HaWoR/pytorch3d (visualization-only). Weights: VITRA-VLA/VITRA-VLA-3B
# (free, MIT) + google/paligemma2-3b-mix-224 (gated, token has access).
set -uo pipefail

RUN_ID="${RUN_ID:-vitra_rocm_smoke}"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
DATA_DIR="${DATA_DIR:-/data}"
WORK="${DATA_DIR}/overnight-tests/vitra"
OUT="${WORK}/outputs_rocm/${RUN_ID}"
CACHE="${DATA_DIR}/cache/huggingface"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"

echo "=== VITRA ROCm smoke ${RUN_ID} ==="
echo "IMAGE=${IMAGE} WORK=${WORK}"

docker run --rm -i \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e RUN_ID="${RUN_ID}" -e WORK="${WORK}" -e OUT="${OUT}" \
  -e HF_HOME="${CACHE}" -e TRANSFORMERS_CACHE="${CACHE}" \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -w "${WORK}" \
  "${IMAGE}" bash -s <<'INNER'
set -uo pipefail
mkdir -p "${WORK}" "${OUT}" "${HF_HOME}" && cd "${WORK}"

echo "--- torch/HIP env ---"
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", getattr(torch.version,"hip",None),
      "cuda_ok", torch.cuda.is_available(),
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "--- HF login (gated paligemma2) ---"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
python -m pip install --no-input -q "huggingface_hub>=0.24" 2>&1 | tail -3 || true
python - <<'PY'
import os
from huggingface_hub import login, whoami
try:
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("hf whoami:", whoami()["name"])
except Exception as e:
    print("hf login WARN:", repr(e))
PY

echo "--- clone microsoft/VITRA ---"
[ -d VITRA ] || git clone --depth 1 https://github.com/microsoft/VITRA.git
cd VITRA
git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

echo "--- deps: install vitra package WITHOUT its dep resolution (keep ROCm torch; skip flash-attn/deepspeed which the inference path does not need) ---"
python -m pip install --no-input --no-deps -e . 2>&1 | tail -15 || echo "PIP_E_INSTALL_NONZERO"
# Explicitly install only the pure-Python runtime deps the inference path imports.
python -m pip install --no-input \
  "transformers==4.47.1" accelerate safetensors pillow numpy scipy \
  opencv-python-headless einops timm sentencepiece omegaconf tqdm \
  roma trimesh ffmpeg-python decord imageio imageio-ffmpeg \
  scikit-image matplotlib pandas h5py \
  "git+https://github.com/EasternJournalist/utils3d.git" 2>&1 | tail -8 || true
echo "--- confirm ROCm torch still intact ---"
python - <<'PY'
import torch
print("post-install torch", torch.__version__, "hip", getattr(torch.version,"hip",None),
      "cuda_ok", torch.cuda.is_available())
PY

echo "--- import smoke ---"
python - <<'PY'
import importlib, traceback
for m in ["vitra", "vitra.models"]:
    try:
        importlib.import_module(m); print("import OK:", m)
    except Exception as e:
        print("import FAIL:", m, repr(e)); traceback.print_exc()
PY

echo "--- model load + single predict_action smoke ---"
OUT="${OUT}" python - <<'PY'
import os, json, time, sys, traceback
import numpy as np, torch
from PIL import Image
out = os.environ["OUT"]; os.makedirs(out, exist_ok=True)
summary = {"stage": "start", "torch": torch.__version__,
           "hip": getattr(torch.version,"hip",None),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
try:
    from vitra.models.vla_builder import build_vla
    from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.utils.config_utils import load_config
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "VITRA-VLA/VITRA-VLA-3B"
    configs = load_config(repo)
    configs["model_load_path"] = repo
    configs["statistics_path"] = repo
    summary["stage"] = "config_loaded"

    t0 = time.time()
    # build_vla instantiates the VLA and loads the PaliGemma2 backbone via
    # from_pretrained. The VITRA-VLA-3B .pt holds only the action head/adapters,
    # so load it non-strictly (repo default load_model uses strict=True and fails
    # on the intentionally-absent backbone keys).
    model = build_vla(configs)
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")] \
            or [f for f in files if f.endswith(".pt")]
    ckpt_path = hf_hub_download(repo, ckpts[0])
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    summary["ckpt_file"] = ckpts[0]
    summary["missing_keys"] = len(missing)
    summary["unexpected_keys"] = len(unexpected)
    summary["missing_all_backbone"] = all("backbone" in k for k in missing) if missing else True
    model = model.cuda().eval()
    normalizer = load_normalizer(configs)
    summary["stage"] = "model_loaded"; summary["load_sec"] = round(time.time()-t0, 1)

    # Prefer a bundled example image; else synthesize one.
    img_path = None
    for cand in ["examples/0002.jpg", "examples/0001.jpg"]:
        if os.path.exists(cand): img_path = cand; break
    if img_path:
        image = Image.open(img_path).convert("RGB")
    else:
        image = Image.fromarray((np.random.rand(480,640,3)*255).astype("uint8"))
    image = resize_short_side_to_target(image, target=224)
    image = np.array(image)
    fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]], dtype=torch.float32)

    instruction = "Left hand: None. Right hand: Pick up the phone on the table."
    state = np.zeros((normalizer.state_mean.shape[0],))
    state_mask = np.array([False, True], dtype=bool)
    action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))
    norm_state = normalizer.normalize_state(state)
    unified_state_dim = StateFeature.ALL_FEATURES[1]
    unified_action_dim = ActionFeature.ALL_FEATURES[1]
    unified_state, unified_state_mask = pad_state_human(
        state=norm_state, state_mask=state_mask,
        action_dim=normalizer.action_mean.shape[0],
        state_dim=normalizer.state_mean.shape[0],
        unified_state_dim=unified_state_dim)
    _, unified_action_mask = pad_action(
        actions=None, action_mask=action_mask,
        action_dim=normalizer.action_mean.shape[0],
        unified_action_dim=unified_action_dim)

    # Model weights are float32; cast float inputs to match (numpy defaults to float64/Double).
    unified_state = unified_state.float()

    t1 = time.time()
    norm_action = model.predict_action(
        image=image, instruction=instruction,
        current_state=unified_state.unsqueeze(0),
        current_state_mask=unified_state_mask.unsqueeze(0),
        action_mask_torch=unified_action_mask.unsqueeze(0),
        num_ddim_steps=10, cfg_scale=5.0, fov=fov, sample_times=1)
    summary["stage"] = "action_predicted"; summary["infer_sec"] = round(time.time()-t1, 1)
    # predict_action already returns a numpy array (samples.cpu().numpy()).
    arr = np.asarray(norm_action, dtype="float32")
    summary["action_shape"] = list(arr.shape)
    summary["action_finite"] = bool(np.isfinite(arr).all())
    summary["status"] = "PASS" if np.isfinite(arr).all() and arr.size > 0 else "FAIL"
except Exception as e:
    summary["status"] = "FAIL"; summary["error"] = repr(e)
    summary["trace"] = traceback.format_exc()[-1500:]
with open(os.path.join(out, "vitra_rocm_smoke_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps({k:v for k,v in summary.items() if k!="trace"}, indent=2))
if summary.get("trace"): print(summary["trace"])
sys.exit(0 if summary.get("status")=="PASS" else 3)
PY
RC=$?
echo "PY_EXIT=$RC"
if [ "$RC" = "0" ]; then echo "VITRA_ROCM_SMOKE_PASS"; else echo "VITRA_ROCM_SMOKE_FAIL"; fi
INNER
echo "EXIT_CODE=$?"
