# WIP — 执行状态（单一事实源）

> 进度/已验证状态的唯一权威来源；README / why / study 只摘要并链接过来。
> 路线见 `study.md §1`（L1→L2→L3）；子任务设计 `docs/features/`，实验记录 `docs/experiments/`。

## 路线位置

**L1 pretrain ✅ → L2 post-train（下一步）→ L3 sim 验证 → L4 真机（无真机，排除）**（`study.md §1`）。
全链路 `human video → 3D 手轨 → dexterous VLA →（可选）重定向` 各段已在 MI300X（gfx942）ROCm 逐段验证。**下一步 = 拼成 robot 闭环、量抓取成功率**（feature8）。

## 全链路 ROCm 状态

| 阶段 | 组件 | 状态 | 产物 / 回馈 |
|---|---|---|---|
| 感知前端 video→3D 手轨 | HaWoR（WiLoR+ViT+masked DROID-SLAM+Metric3D+infiller） | ✅ e2e（含 hipify kernel） | [HaWoR fork](https://github.com/ZJLi2013/HaWoR/tree/amd_support) · [DROID-SLAM #171](https://github.com/princeton-vl/DROID-SLAM/pull/171) · [lietorch #53](https://github.com/princeton-vl/lietorch/pull/53) |
| 感知原语 | TAPIR / GroundingDINO / SAM2·3 / Depth-Anything-3 | ✅ | `ai_agents/rocm3d/README.md` |
| VLA 本体 | VITRA（PaliGemma2-3B + DiT 动作头） | ✅ 推理 + 8 卡 FSDP 训练 | [VITRA #42](https://github.com/microsoft/VITRA/issues/42) |
| 重定向（灵巧手，可选） | Do-as-I-Do（MuJoCo-Warp MPC） | ✅ 阶段 1–5 | [do-as-i-do fork](https://github.com/ZJLi2013/do-as-i-do/tree/rocm-support) |
| 真视频 e2e 闭合 + HaWoR 精度 | feature6（EgoDex 40 clips） | ✅ 腕 RTE median 1.9cm、40/40 成功 | `docs/features/feature6_hawor_egodex.md` |
| 可训练样本 glue | feature3 converter | ✅ round-trip ~1e-6 | `docs/features/feature3_triplet_export.md` |

## 已完成关键结论

- **VITRA on ROCm**：纯 PyTorch + `transformers`（DiT 用 torch SDPA），`rocm/pytorch:2.9.1` 开箱即用，无需 DeepSpeed/FlashAttention。8×MI300X FSDP 满 1000 步 loss 0.29→0.21，RCCL / ckpt / resume 正常，**零 fork**；唯一 AMD 专属适配 = `HSA_NO_SCRATCH_RECLAIM=1`（另有平台无关 decord fix）。复现见 `scripts/vitra/`。
- **HaWoR 前端**：全链路唯一的硬 CUDA 阻塞（masked DROID-SLAM + lietorch 自定义 kernel）已 hipify 并回馈上游（均单文件 PR）。License CC BY-NC-ND → 研究用途。
- **精度头条**：腕世界轨迹 RTE median **1.9cm**，足以驱动 VITRA 式 retargeting；指尖 9.5cm 有歧义，不作强结论。

## Backlog

| 优先级 | feature | 内容 |
|---|---|---|
| **P1** | feature8 | **L2 zero-shot baseline**（开环、无训练、**零渲染**）：VITRA 手部输出 → dex-retarget 到 sim floating 灵巧手（LEAP/Allegro，MuJoCo-Warp+Do-as-I-Do 现成）→ 零样本抓取成功率 baseline。据此决定要不要 feature9。设计见 `docs/features/feature8_l2_embodiment.md` |
| P2 | feature9 | **sim 数据生成 → post-train → 闭环验证**（依赖 feature8 结果）：① Do-as-I-Do 放量造 rollout → ② 续训 VITRA（保留 DiT 先验；**②抽 `f^c` 训 adapter 丢先验、不走**）→ ③ 闭环 policy-in-sim。观测 A(真视频·零渲染)/B(机器人视角·需渲染)分叉；渲染才评估 Genesis（与 MuJoCo-Warp 物理共存非替换）。三干预点+两 gap 见 `study.md §1`；设计见 `docs/features/feature9_l3_sim_deploy.md` |


## 约定

产物分工：设计/实验记录进 `docs/features/` + `docs/experiments/`；代码/长跑脚本进 `scripts/`（`human_video_factory/`、`do-as-i-do/`、`vitra/`、`hawor/`）；HaWoR / VITRA / Do-as-I-Do 各留原 fork，不做 submodule。**本项目自包含，不引用外部仓。**
