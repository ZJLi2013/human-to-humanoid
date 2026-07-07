#!/usr/bin/env bash
# VITRA training-env smoke on ROCm (M1-b, experiment E2).
# Goal: prove the TRAINING compute path (PaliGemma2-3B + DiT diffusion head, bf16,
# forward -> backward -> optimizer.step) runs on gfx942 with the pinned training deps,
# WITHOUT deepspeed / flash-attn (both confirmed unused: DiT uses torch SDPA, VLM uses
# transformers default attn; training loop is PyTorch FSDP). Single-GPU, synthetic batch,
# no dataset required. FSDP multi-rank is validated separately in M1-c.
set -uo pipefail

RUN_ID="${RUN_ID:-vitra_train_smoke}"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
DATA_DIR="${DATA_DIR:-/data}"
WORK="${DATA_DIR}/overnight-tests/vitra"
OUT="${WORK}/outputs_rocm/${RUN_ID}"
CACHE="${DATA_DIR}/cache/huggingface"
TOKEN_FILE="${TOKEN_FILE:-/tmp/overnight-tests/.hf_token}"

echo "=== VITRA ROCm TRAIN smoke ${RUN_ID} ==="
echo "IMAGE=${IMAGE} WORK=${WORK}"

docker run --rm -i \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e RUN_ID="${RUN_ID}" -e WORK="${WORK}" -e OUT="${OUT}" \
  -e HF_HOME="${CACHE}" -e TRANSFORMERS_CACHE="${CACHE}" \
  -e WANDB_MODE=offline \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -v "${TOKEN_FILE}:/root/.hf_token:ro" \
  -w "${DATA_DIR}" \
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
python -m pip install --no-input -q "huggingface_hub>=0.24" 2>&1 | tail -2 || true
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

echo "--- deps: vitra pkg WITHOUT dep resolution (keep ROCm torch; skip deepspeed/flash-attn) ---"
python -m pip install --no-input --no-deps -e . 2>&1 | tail -6 || echo "PIP_E_INSTALL_NONZERO"
# Training-path pure-Python deps (deepspeed/flash-attn intentionally omitted: unused).
python -m pip install --no-input \
  "transformers==4.47.1" accelerate safetensors "numpy<2" pillow scipy \
  diffusers "timm==0.9.10" einops sentencepiece "tokenizers>=0.21" "peft==0.11.1" \
  hydra-core draccus wandb rich tqdm tensorboard tensorboardX pyyaml \
  opencv-python-headless decord scikit-image roma trimesh ffmpeg-python \
  imageio imageio-ffmpeg pympler json-numpy jsonlines matplotlib omegaconf \
  "git+https://github.com/EasternJournalist/utils3d.git" 2>&1 | tail -8 || true

echo "--- confirm ROCm torch still intact ---"
python - <<'PY'
import torch
print("post-install torch", torch.__version__, "hip", getattr(torch.version,"hip",None),
      "cuda_ok", torch.cuda.is_available())
PY

echo "--- import training modules ---"
python - <<'PY'
import importlib, traceback
mods = ["vitra", "vitra.models.vla_builder", "vitra.models.vla",
        "vitra.models.action_model", "vitra.training.fsdp",
        "vitra.datasets.materialize", "vitra.utils.config_utils"]
for m in mods:
    try:
        importlib.import_module(m); print("import OK:", m)
    except Exception as e:
        print("import FAIL:", m, repr(e)); traceback.print_exc()
PY

echo "--- build_vla + single forward/backward/step smoke (synthetic batch) ---"
OUT="${OUT}" python - <<'PY'
import os, json, time, sys, traceback
import numpy as np, torch
from PIL import Image
out = os.environ["OUT"]; os.makedirs(out, exist_ok=True)
s = {"stage": "start", "torch": torch.__version__, "hip": getattr(torch.version,"hip",None),
     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
try:
    from vitra.models.vla_builder import build_vla
    from vitra.utils.config_utils import load_config
    cfg = load_config("vitra/configs/human_pretrain.json")
    cfg["model_load_path"] = None
    cfg["repeated_diffusion_steps"] = 2   # smaller for smoke
    s["stage"] = "config_loaded"

    t0 = time.time()
    model = build_vla(cfg)                 # downloads paligemma2-3b backbone
    model.use_bf16 = True
    model.trainable_params_setup()
    model = model.cuda()
    s["stage"] = "model_built"; s["build_sec"] = round(time.time()-t0, 1)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    s["trainable_params_M"] = round(n_train/1e6, 1)

    # --- synthetic batch via the real PaliGemma processor ---
    from transformers import PaliGemmaProcessor
    proc = PaliGemmaProcessor.from_pretrained(cfg["vlm"]["pretrained_model_name_or_path"])
    B, T = 1, model.chunk_size
    img = Image.fromarray((np.random.rand(224,224,3)*255).astype("uint8"))
    prompt = "Left hand: None. Right hand: pick up the cup."
    enc = proc(text=[prompt], images=[img], return_tensors="pt")
    pixel_values = enc["pixel_values"].cuda()
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].bool().cuda()

    adim = cfg["action_model"]["action_dim"]        # 192
    sdim = cfg["state_encoder"]["state_dim"]         # 212
    action_labels = torch.randn(B, T, adim).cuda()
    action_masks = torch.ones(B, T, adim).cuda()   # x_mask over action dims (B,T,192), float
    current_state = torch.zeros(B, sdim).cuda()
    current_state_mask = torch.ones(B, sdim).cuda()   # state_mask over state dims (B,212), float
    fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]], dtype=torch.float32).cuda()
    s["stage"] = "batch_ready"
    s["shapes"] = {"pixel_values": list(pixel_values.shape), "input_ids": list(input_ids.shape),
                   "action_labels": list(action_labels.shape)}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    t1 = time.time()
    loss_dict = model(
        pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
        action_labels=action_labels, action_masks=action_masks,
        current_state=current_state, current_state_mask=current_state_mask,
        fov=fov, mode="train")
    loss = sum(v for v in loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
    s["loss_keys"] = list(loss_dict.keys()) if isinstance(loss_dict, dict) else None
    s["loss_value"] = float(loss.detach().cpu())
    s["stage"] = "forward_done"; s["fwd_sec"] = round(time.time()-t1, 1)

    t2 = time.time()
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad.detach().float()**2).sum() for p in model.parameters() if p.grad is not None))
    opt.step(); opt.zero_grad()
    s["grad_norm"] = float(gnorm.cpu())
    s["bwd_step_sec"] = round(time.time()-t2, 1)
    s["mem_alloc_GB"] = round(torch.cuda.max_memory_allocated()/1e9, 2)
    s["stage"] = "step_done"
    s["status"] = "PASS" if (np.isfinite(s["loss_value"]) and s["grad_norm"] > 0) else "FAIL"
except Exception as e:
    s["status"] = "FAIL"; s["error"] = repr(e); s["trace"] = traceback.format_exc()[-2000:]
with open(os.path.join(out, "vitra_train_smoke_summary.json"), "w") as fh:
    json.dump(s, fh, indent=2)
print(json.dumps({k:v for k,v in s.items() if k!="trace"}, indent=2))
if s.get("trace"): print(s["trace"])
sys.exit(0 if s.get("status")=="PASS" else 3)
PY
RC=$?
echo "PY_EXIT=$RC"
if [ "$RC" = "0" ]; then echo "VITRA_TRAIN_SMOKE_PASS"; else echo "VITRA_TRAIN_SMOKE_FAIL"; fi
INNER
echo "EXIT_CODE=$?"
