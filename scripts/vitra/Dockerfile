# VITRA (VLA 本体) on AMD ROCm / gfx942 —— 推理 + 8xMI300X FSDP 训练闭环。
# 纯 PyTorch + transformers，零 fork / 零 hipify / 零自定义 kernel。
# 等价于 build_vitra_image.sh 的 docker-run+commit，改成可复现的 Dockerfile。
#
# build:  docker build -t vitra-rocm:m1c -f Dockerfile .
# run:    docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
#           --ipc=host --shm-size 32g -e HSA_NO_SCRATCH_RECLAIM=1 \
#           -v <data>:/data vitra-rocm:m1c bash
ARG BASE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE}

WORKDIR /opt
RUN git clone --depth 1 https://github.com/microsoft/VITRA.git && \
    git config --global --add safe.directory /opt/VITRA

WORKDIR /opt/VITRA
# --no-deps: 只装 VITRA 本身，显式列纯 Python 依赖，避开 flash-attn / deepspeed 的源码构建。
RUN python -m pip install --no-input -q --no-deps -e . && \
    python -m pip install --no-input -q \
      "transformers==4.47.1" accelerate safetensors "numpy<2" pillow scipy \
      diffusers "timm==0.9.10" einops sentencepiece "tokenizers>=0.21" "peft==0.11.1" \
      hydra-core draccus wandb rich tqdm tensorboard tensorboardX pyyaml \
      opencv-python-headless decord scikit-image roma trimesh ffmpeg-python \
      imageio imageio-ffmpeg pympler json-numpy jsonlines matplotlib omegaconf \
      "huggingface_hub>=0.24" \
      "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@d790d33"

# decord 多线程解码在高并发下崩（UnboundLocalError: imgs）——强制单线程。平台无关，NV 同踩。
RUN sed -i 's/decord.VideoReader(name)/decord.VideoReader(name, num_threads=1)/g' \
      vitra/datasets/video_utils.py && \
    python -c "import vitra, utils3d.numpy as n; assert hasattr(n,'image_uv'); print('BAKE_OK')"

# MI300X 固件项：RCCL 首个 barrier 需要它，否则 ncclSystemError（单卡不触发）。
ENV HSA_NO_SCRATCH_RECLAIM=1
WORKDIR /opt/VITRA
