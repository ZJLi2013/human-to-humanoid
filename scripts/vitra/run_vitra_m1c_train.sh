#!/usr/bin/env bash
# M1-c: real short training on SSv2 10k crop, 8x MI300X FSDP (validates RCCL multi-GPU + real
# data pipeline on ROCm). Zero VITRA source patch; env/config only. Generates ssv2_10k.json
# from human_pretrain.json + M1-c overrides (see experiments.md param table).
set -uo pipefail

RUN_ID="${RUN_ID:-m1c_ssv2_10k}"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
DATA_DIR="${DATA_DIR:-/data}"
WORK="${DATA_DIR}/overnight-tests/vitra"
OUT="${WORK}/outputs_rocm/${RUN_ID}"
CACHE="${DATA_DIR}/cache/huggingface"
DATA_ROOT="${WORK}/data_ssv2_10k"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"
NPROC="${NPROC:-8}"

echo "=== VITRA M1-c train ${RUN_ID} (nproc=${NPROC}) ==="

docker run --rm -i \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e RUN_ID="${RUN_ID}" -e WORK="${WORK}" -e OUT="${OUT}" -e DATA_ROOT="${DATA_ROOT}" -e NPROC="${NPROC}" \
  -e HF_HOME="${CACHE}" -e TRANSFORMERS_CACHE="${CACHE}" \
  -e WANDB_MODE=offline -e WANDB_API_KEY=local-offline -e WANDB_DIR="${OUT}" \
  -e NCCL_DEBUG=WARN -e TOKENIZERS_PARALLELISM=false \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_IB_DISABLE=1 \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -w "${DATA_DIR}" \
  "${IMAGE}" bash -s <<'INNER'
set -uo pipefail
mkdir -p "${WORK}" "${OUT}" "${HF_HOME}" && cd "${WORK}"

echo "--- torch/HIP + GPU count ---"
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", getattr(torch.version,"hip",None),
      "n_gpu", torch.cuda.device_count())
PY

echo "--- HF token ---"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

echo "--- clone/reuse microsoft/VITRA ---"
[ -d VITRA ] || git clone --depth 1 https://github.com/microsoft/VITRA.git
cd VITRA
git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

# Local working-copy patch (NOT AMD-specific, NOT an upstream fork): decord's threaded
# decoder throws `avcodec_send_packet -11` under many concurrent readers (8 ranks x N workers).
# Pin single-thread decode -> deterministic fix for this known decord bug.
sed -i 's/decord\.VideoReader(name)/decord.VideoReader(name, num_threads=1)/g' vitra/datasets/video_utils.py
echo "decord num_threads=1 patch:"; grep -n 'decord.VideoReader' vitra/datasets/video_utils.py

echo "--- deps (no-deps vitra + pure-Python training deps; no deepspeed/flash-attn) ---"
python -m pip install --no-input -q --no-deps -e . 2>&1 | tail -3 || echo "PIP_E_NONZERO"
python -m pip install --no-input -q \
  "transformers==4.47.1" accelerate safetensors "numpy<2" pillow scipy \
  diffusers "timm==0.9.10" einops sentencepiece "tokenizers>=0.21" "peft==0.11.1" \
  hydra-core draccus wandb rich tqdm tensorboard tensorboardX pyyaml \
  opencv-python-headless decord scikit-image roma trimesh ffmpeg-python \
  imageio imageio-ffmpeg pympler json-numpy jsonlines matplotlib omegaconf \
  "huggingface_hub>=0.24" "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@d790d33" 2>&1 | tail -4 || true

echo "--- generate ssv2_10k.json (human_pretrain.json + M1-c overrides) ---"
DATA_ROOT="${DATA_ROOT}" OUT="${OUT}" python - <<'PY'
import json, os
c=json.load(open("vitra/configs/human_pretrain.json"))
OUT=os.environ["OUT"]; DR=os.environ["DATA_ROOT"]
c["batch_size"]=8
c["total_batch_size"]=64
c["save_steps"]=250
c["output_root"]=os.path.join(OUT,"checkpoints")
c["log_root"]=os.path.join(OUT,"logs")
c["cache_root"]=os.path.join(OUT,"cache")
c["wandb_entity"]=""
c["trainer"]["max_steps"]=1000
c["trainer"]["llm_freeze_step"]=200
c["train_dataset"]["data_root_dir"]=DR
c["train_dataset"]["data_mix"]="ssv2"
c["train_dataset"]["num_workers"]=4
json.dump(c, open("vitra/configs/ssv2_10k.json","w"), indent=2)
print("wrote ssv2_10k.json: batch=8 total=64 max_steps=1000 save=250 llm_freeze=200 data_mix=ssv2")
print("data_root:", DR)
PY

echo "--- launch torchrun FSDP (${NPROC} GPUs) ---"
torchrun --standalone --nproc_per_node="${NPROC}" scripts/train.py \
  --config vitra/configs/ssv2_10k.json
RC=$?
echo "TORCHRUN_EXIT=$RC"
echo "--- checkpoints ---"
find "${OUT}/checkpoints" -maxdepth 2 -type d 2>/dev/null | head
[ "$RC" = "0" ] && echo "M1C_TRAIN_DONE" || echo "M1C_TRAIN_FAIL"
INNER
echo "EXIT_CODE=$?"
