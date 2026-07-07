# HaWoR (感知前端: video -> 世界坐标 3D 手轨) on AMD ROCm / gfx942。
# 管线里唯一的硬 CUDA 阻塞（masked DROID-SLAM + lietorch 自定义 kernel）就在这里，
# 本镜像在 build 时就把 5 处 ROCm patch 打好并 hipify 编译 lietorch + droid_backends。
# 等价于 run_hawor_demo.sh 的 STAGE 1-2（deps + build），已烘焙；权重(STAGE 3)因体积/MANO
# 注册墙留到运行时下载。旧栈（py3.10 / torch2.10），与 VITRA 镜像的 base 不同，故必须分开。
#
# build:  docker build -t hawor-rocm:demo -f Dockerfile .
#         (编译扩展不需要 GPU；kernel 优化慢的 reduce 不涉及，droid/lietorch 数分钟)
# run:    docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
#           --ipc=host --shm-size 16g -e HIP_VISIBLE_DEVICES=0 \
#           -v <work>:/work hawor-rocm:demo bash
#         (首次跑 run_hawor_demo.sh 会下 6 个权重 + MANO；MANO 需注册墙镜像)
ARG BASE=rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0
FROM ${BASE}

ARG PYTORCH_ROCM_ARCH=gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq ffmpeg unzip && rm -rf /var/lib/apt/lists/*

# --- Python 依赖（HaWoR 旧栈钉 numpy<2 / setuptools<81 / mmcv 1.7.2 纯 python 构建）---
RUN python -m pip install -q --upgrade pip && \
    pip install -q "setuptools<81" wheel "numpy==1.26.4" ninja && \
    pip install -q \
      opencv-python scikit-image "smplx==0.1.28" yacs timm einops xtcocotools \
      pandas hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils rich \
      webdataset ultralytics pulp supervision pycocotools joblib natsort evo \
      pytorch-minimize "mmengine==0.10.4" HTML4Vision plyfile easydict loguru \
      dill lapx "pytorch-lightning==2.2.4" lightning-utilities "torchmetrics==1.4.0" \
      trimesh pyrender "moderngl-window==2.4.6" "aitviewer==1.13.0" \
      "huggingface_hub[cli]" gdown && \
    pip install -q --no-build-isolation "mmcv==1.7.2"

# chumpy(MANO loader) 与 numpy>=1.24 不兼容 —— 装完就地补掉失效的 numpy import。
RUN pip install -q "chumpy@git+https://github.com/mattloper/chumpy" || true && \
    CH=$(python -c "import chumpy,os;print(os.path.dirname(chumpy.__file__))" 2>/dev/null) && \
    if [ -n "$CH" ]; then \
      sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CH/__init__.py"; \
    fi

# torch_scatter 用 pyg-rocm-build 的预编译 wheel（torch2.10 / rocm7.1 / py310，rocm7.2 runtime 可加载）。
RUN cd /tmp && \
    curl -sL "https://github.com/Looong01/pyg-rocm-build/releases/download/15/torch-2.10.0-rocm-7.1-py310-linux_x86_64.zip" -o pyg.zip && \
    unzip -o -q pyg.zip -d pyg && \
    pip install -q pyg/torch_scatter*.whl && \
    python -c "import torch_scatter; from torch_scatter import scatter_mean; print('torch_scatter', torch_scatter.__version__)" && \
    rm -rf /tmp/pyg /tmp/pyg.zip

# --- 克隆 ROCm fork（含 masked DROID-SLAM / lietorch 子模块）---
WORKDIR /opt
RUN git clone --recursive --branch amd_support https://github.com/ZJLi2013/HaWoR.git && \
    git config --global --add safe.directory '*'

WORKDIR /opt/HaWoR
# 5 处 ROCm patch（P1-P5，幂等；fork 已含部分 in-tree，重复应用无副作用）：
#  P1/P2 setup.py 去掉 nvcc 的 -gencode（clang/hipcc 拒收）
#  P3 lietorch_gpu.cu 依赖模板成员调用加 .template
#  P4 masked kernel 的 Tensor.type() -> .scalar_type()（新 torch 不再隐式转 ScalarType）
#  P5 bundled Eigen 的 EIGEN_USING_STD(fill_n) 在 hipcc 下展开为全局 fill_n（不存在）-> using std::
RUN DROID=thirdparty/DROID-SLAM && LIET=$DROID/thirdparty/lietorch && \
    sed -i '/gencode=arch=compute/d' $DROID/setup.py && \
    sed -i '/gencode=arch=compute/d' $LIET/setup.py && \
    sed -i 's/Matrix4x4()\.block<3,3>/Matrix4x4().template block<3,3>/' $LIET/lietorch/src/lietorch_gpu.cu && \
    sed -i 's/\.type()/.scalar_type()/g' $DROID/src/*.cu 2>/dev/null || true; \
    grep -rl 'EIGEN_USING_STD(fill_n)' $DROID/thirdparty/eigen/Eigen 2>/dev/null \
      | xargs -r sed -i 's/EIGEN_USING_STD(fill_n);/using std::fill_n;/g'

# hipify 编译扩展（无需 GPU）。droid_backends 走 setup.py install 绕开 PEP517 metadata hook 的 .egg-info 断言。
RUN DROID=thirdparty/DROID-SLAM && LIET=$DROID/thirdparty/lietorch && \
    pip install --no-build-isolation "$LIET" && \
    python -c "import lietorch; from lietorch import SE3; print('lietorch OK')" && \
    find $DROID -maxdepth 2 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null; \
    ( cd $DROID && rm -rf build && python setup.py install ) && \
    python -c "import torch, droid_backends; print('droid_backends OK')"

# 运行期环境（可视化 headless / 老 Lightning ckpt 加载）。demo 内 aitviewer 另需 Xvfb+llvmpipe（见 run_hawor_demo.sh）。
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    HF_ENDPOINT=https://hf-mirror.com
WORKDIR /opt/HaWoR
