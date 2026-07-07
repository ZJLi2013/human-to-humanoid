# scripts — ROCm 复现脚本 / 镜像构建

human video → 显式 3D 手部运动 → dexterous-hand VLA 管线各段在 AMD Instinct（MI300X / gfx942）上的复现脚本。均在 ROCm PyTorch 容器内运行。

## 镜像（Dockerfile，per-model）
每个模型一个 Dockerfile，把已验证的环境（deps + 修复 + hipify 编译）固化成可复现镜像：

| Dockerfile | 环境 | base 镜像 | 内容 |
|---|---|---|---|
| `vitra/Dockerfile` | VLA 本体（推理+训练） | rocm7.2.1 / py3.12 / torch2.9.1 | VITRA + 纯 Python 依赖 + decord 单线程修复；`human_video_factory/` 的 glue 也用它 |
| `hawor/Dockerfile` | 感知前端 | rocm7.2 / py3.10 / torch2.10.0 | HaWoR fork + 5 处 ROCm patch + hipify 编译 lietorch/droid_backends（权重运行时下载） |
| `do-as-i-do/Dockerfile` | 重定向（可选，落真机） | rocm7.2.1 / py3.12 / torch2.9.1 | ROCm warp + mujoco_warp 源码 build + retarget 依赖 |


```bash
docker build -t vitra-rocm:m1c        scripts/vitra
docker build -t hawor-rocm:demo       scripts/hawor
docker build -t doasido-rocm:retarget scripts/do-as-i-do
```
运行时挂 GPU：`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host`；VITRA 训练需 `-e HSA_NO_SCRATCH_RECLAIM=1`。各 Dockerfile 头部有完整 build/run 示例。

## 前置
- **节点**：8×MI300X（gfx942）+ docker + ROCm 驱动（`/dev/kfd`、`/dev/dri`）。单段 smoke 单卡即可。
- **HF token**：PaliGemma2 基座需申请权限；训练脚本从 `TOKEN_FILE`（默认 `/tmp/overnight-tests/.hf_token`）读。
- **MANO**：HaWoR 需 `MANO_RIGHT/LEFT.pkl`（注册墙，`run_hawor_demo.sh` 走 hf-mirror 拉，内网需自备镜像）。
- 脚本路径靠 env var 覆盖：`DATA_DIR` / `IMAGE` / `TOKEN_FILE` / `RUN_ID`（见各脚本头部）。

## 复现流程（按管线顺序）
数据流：**video →① 3D 手轨 →② 可训练三元组 →③ VLA 训练**。三段各自独立可跑。

**① HaWoR 感知前端（video → 世界坐标 3D 手轨）**
```bash
docker build -t hawor-rocm:demo scripts/hawor          # 建镜像（已 hipify 编译 kernel）
bash scripts/hawor/run_hawor_demo.sh                   # e2e demo（首跑下 6 权重+MANO）→ world_space_res.pth
python scripts/hawor/make_world_traj_video.py          # （可选）可视化 3D 手轨
```
**② 串成可训练样本（glue，在 vitra 镜像内跑）**
```bash
python scripts/human_video_factory/hawor_to_vitra.py   # HaWoR 输出 → VITRA episode
python scripts/human_video_factory/check_vitra_load.py # stock FrameDataset 加载 smoke
python scripts/human_video_factory/roundtrip_check.py  # 无 HaWoR 也能验 converter（离线 round-trip）
```
**③ VITRA 训练闭环**
```bash
docker build -t vitra-rocm:m1c scripts/vitra           # 建镜像
bash scripts/vitra/run_ssv2_download.sh                # 下 SSv2
bash scripts/vitra/run_ssv2_crop.sh                    # crop 10k episodes
bash scripts/vitra/run_vitra_train_smoke.sh            # （可选）单步 smoke
bash scripts/vitra/run_vitra_m1c_fast.sh               # 8 卡跑满 1000 步
```

**（可选）重定向（落真机）**：`docker build -t doasido-rocm:retarget scripts/do-as-i-do`，pipeline 运行脚本在 fork [`ZJLi2013/do-as-i-do@rocm-support`](https://github.com/ZJLi2013/do-as-i-do/tree/rocm-support)。

> 三段解耦：③ 训练按开源 VITRA-1M SSv2 crop 验证，**不依赖 ①②**；反之 ②→③ 也能吃 ① 的真实输出。②的 `roundtrip_check.py` 用真实 episode 就能验 converter，无需 ①。



## hawor/ — 感知前端（video → 世界坐标 3D 手轨）
| 脚本 | 作用 |
|---|---|
| `run_hawor_droid_build.sh` | hipify + 构建 masked DROID-SLAM (`droid_backends`) + `lietorch`（gfx942，唯一硬 CUDA 阻塞） |
| `run_hawor_droid_smoke.sh` / `hawor_droid_smoke.py` | droid/lietorch kernel 功能验证 |
| `run_hawor_demo.sh` | 全前端 e2e demo（WiLoR+HaWoR+DROID-SLAM+Metric3D+infiller）→ `world_space_res.pth` |
| `make_world_traj_video.py` / `make_overlay_video.py` | 世界坐标 3D 手轨 / 2D 叠加可视化（matplotlib，绕开 headless-GL） |

## human_video_factory/ — 感知前端 → VLA 训练格式的 glue
| 脚本 | 作用 |
|---|---|
| `hawor_to_vitra.py` | HaWoR `world_space_res.pth` + SLAM → VITRA-1M episode（`.npy` + index + stats）；stock `FrameDataset` 可直接吃，零 VITRA fork |
| `check_vitra_load.py` | 用 stock `FrameDataset` 加载 episode 的 smoke |
| `roundtrip_check.py` | 真实 episode → 逆向 → converter → 再加载 的离线 round-trip 验证（无需 HaWoR/MANO） |
| `probe_episode.py` | dump 真实 VITRA-1M episode 的 on-disk schema |

## vitra/ — VLA 本体（推理 + 8×MI300X FSDP 训练闭环）
| 脚本 | 作用 |
|---|---|
| `build_vitra_image.sh` | 烘焙 `vitra-rocm` 镜像（deps + VITRA + 3 处修复：`HSA_NO_SCRATCH_RECLAIM` env / decord `num_threads=1` / `utils3d` 钉版） |
| `run_ssv2_download.sh` / `run_ssv2_crop.sh` | 下载 + crop VITRA-1M SSv2 子集（10k episodes） |
| `run_vitra_rocm_smoke.sh` / `run_vitra_train_smoke.sh` | 单卡推理 / 单步训练 smoke |
| `run_vitra_m1c_train.sh` | 8×MI300X 训练闭环（从零装环境） |
| `run_vitra_m1c_fast.sh` | 同上，用烘焙镜像快速启动 |

> 进度 / 实验记录 / 战略定位见本仓 `docs/`（本地，未纳入版本库）。
