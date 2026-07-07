# Do-as-I-Do (可选: 落真机重定向 human->灵巧手) on AMD ROCm / gfx942。
# 采样式 MPC 物理优化，跑在 ROCm 版 warp + mujoco_warp 上。
# 等价于 run_full_pipeline_rocm.sh 的容器内步骤 + rocm-warp L1/L2 的 warp/mujoco_warp 构建。
# 与 VITRA 同 base（rocm7.2.1 / py3.12 / torch2.9.1）。
#
# build:  docker build -t doasido-rocm:retarget -f Dockerfile .
#         (warp reduce.cu 在 gfx942+clang22 有病态 codegen 慢 -> 用 -O0 绕过，功能不受影响)
# run:    docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
#           --ipc=host --shm-size 32g --security-opt seccomp=unconfined \
#           -e HIP_VISIBLE_DEVICES=0 -v <work>:/work doasido-rocm:retarget bash
ARG BASE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE}

ARG HIP_ARCH=gfx942
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq ffmpeg && rm -rf /var/lib/apt/lists/*

# --- ROCm/warp: 源码 build + editable install ---
# reduce.cu 在 gfx942 上 -O3/-O1 会触发 clang 的病态 isel 慢（>17-52min）；用 -O0 规避，
# 只是关掉 reduce 的 runtime 优化，对重定向管线无影响，其余 .cu 秒级编译。
WORKDIR /opt
RUN git clone --depth 1 https://github.com/ROCm/warp.git && \
    cd warp && \
    python build_lib.py --hip-arch=${HIP_ARCH} --hipcc-options="-O0" && \
    pip install -e . --no-build-isolation && \
    python -c "import warp as wp; print('warp', wp.config.version)"

# --- ROCm/mujoco_warp (amd-integration) ---
RUN git clone --branch amd-integration https://github.com/ROCm/mujoco_warp.git && \
    cd mujoco_warp && \
    pip install -e . --no-deps && \
    pip install -q absl-py "etils[epath]" "mujoco>=3.8.0" numpy

# --- do-as-i-do fork (retargeting + reconstruction) ---
# --no-deps: pyproject 钉 CUDA 时代的 warp-lang==1.10 / 旧 mujoco_warp，改用上面 PYTHONPATH 上的 ROCm build。
RUN git clone --branch rocm-support https://github.com/ZJLi2013/do-as-i-do.git && \
    git config --global --add safe.directory '*' && \
    cd do-as-i-do/retargeting && \
    pip install -e . --no-deps && \
    pip install -q scipy loguru tyro tqdm filelock omegaconf loop-rate-limiters \
      "imageio[ffmpeg]" opencv-python trimesh rtree shapely networkx viser plotly \
      coacd mink "mujoco>=3.8.0" && \
    pip install -q "git+https://github.com/cvg/GeoCalib.git"

# warp / mujoco_warp 从源码树导入（避免 repo-root 空 namespace package 遮蔽内层真包）。
ENV PYTHONPATH=/opt/warp:/opt/mujoco_warp
WORKDIR /opt/do-as-i-do/retargeting
