# human → humanoid


## §0 表征桥梁谱系 + 使能组件（方法分类 × ROCm 覆盖）

分类的是"**human video → robot action 用什么中间表征当桥梁**"（方法/模型，scalable ⟷ grounded），并标注本项目 ROCm(rocm3d) 覆盖：

| 表征桥梁 | 思路 | 代表工作 | ROCm(rocm3d) |
|---|---|---|---|
| latent action | 视频自监督学"潜动作"再对齐机器人 | LAPA / IGOR / Moto / GO-1 / villa-X / CLAP / Genie(-2) / Being-H0 / GR00T(latent) | — |
| 世界模型 / 视频预测 | 视频预测当 policy 或数据发生器 | GR-1 / GR-2 / VPP / mimic-video / **Wh0** / UniPi / Cosmos / DreamGen / WorldVLA | 多个 WM 已迁（Matrix-Game/le-wm/FastWAM ✅） |
| embodiment-agnostic intent | 意图(视频学)与控制(仿真学)解耦 | **LUCID** | — |
| 显式 2D 线索 | 点轨迹/光流/affordance/接触点作接口 | ATM / A0 / Track2Act / Gen2Act / Masquerade / point-flow | TAPIR ✅ |
| **显式 3D 重建**（主线） | hand-object 3D 轨迹 → retarget | **HaWoR / VITRA / Do-as-I-Do** / VidBot / EgoVLA / DexMV / OKAMI / ManipTrans / VideoManip / V2P-Manip / DexMan | HaWoR ✅ / Do-as-I-Do ✅ / VITRA ✅ |

```
                 task            obs             action
                 (做什么)         (看什么)         (怎么动)
弱接地/可扩展  ┌───────────────┬───────────────┬──────────────────┐
  latent      │               │  HVP 世界模型*  │  latent action    │← 2606: latent/WM
              │  任务意图      │  视觉嵌入       │                  │
  ────────────┤  任务结构      │  变换视频       │  affordance-2D    │← 2606: 显式2D
强接地/可执行  │               │               │  affordance-3D    │← 2606: 显式3D
              └───────────────┴───────────────┴──────────────────┘
                                                 ↑ HaWoR / Do-as-I-Do / VITRA
                                                   (action × 显式3D affordance)
  * world model 主放 obs（预测未来帧），但也可当 policy 直接生成动作（VPP/Wh0）→ 实际跨 obs/action。
```



**关键使能组件**（perception primitives，任何 pipeline 的地基）：

| 类别 | 代表 | ROCm(rocm3d) |
|------|------|--------------|
| 手部重建 | **HaWoR**、WiLoR、HaMeR、Dyn-HaMR | HaWoR 依赖 ✅ |
| 全身/人体重建 | SMPL-X、GVHMR、TRAM、WHAM | — |
| 物体位姿/网格 | FoundationPose、BundleSDF、SAM-3D | SAM3 ✅ |
| 分割/检测/跟踪 | SAM2/SAM3、GroundingDINO、TAPIR | ✅ 全套 |
| 相机/深度/SLAM | DROID-SLAM、MegaSAM、Metric3D、Depth-Anything-3 | DROID-SLAM ✅、DA3 ✅ |
| 重定向 | dex-retargeting、PHC、MuJoCo-Warp MPC(Do-as-I-Do) | Do-as-I-Do ✅ |

---

## §1 全局技术路线图：pretrain → post-train → sim deployment

human→humanoid 的完整路线是**三层数据 + 一层真机**。HaWoR→VITRA 覆盖第一层（L1 pretrain）。

| 层 | 学什么 | 数据来源 | 本项目资产 | 状态 |
|---|---|---|---|---|
| **L1 pretrain**（数据 + 模型先验） | 从人类视频学"给定指令**大致怎么动**"的动作**先验** | 无标注人类视频 → HaWoR 恢复 3D 手轨 | **HaWoR**（前端）+ **VITRA**（VLA 本体） | ✅ ROCm 已跑通（`WIP §1/§2`） |
| **L2 post-train**（embodiment alignment） | 手先验 **retarget 到具体本体** + 用带动作标签数据**微调成可执行策略** | sim rollout / Wh0 合成 | **Do-as-I-Do**（MPC 重定向）+ dex-retargeting | ⬜ 下一步 |
| **L3 sim deployment**（补物理 + 验证） | 物理引擎里补 **contact/force**、量抓取成功率、RL 修正闭环 | MuJoCo-Warp / Genesis 物理 rollout | **RoboSmith** / MuJoCo-Warp（Do-as-I-Do 已用） | ⬜ 未做 |
| （L4 真机部署） | 真实相机 / 控制器 / 延迟 | 真机 | — | ✗ 无真机，短期不做 |

### 三个判断

1. **HaWoR→VITRA = L1。** VITRA 是 pretraining，产出动作**先验**不是可部署策略（reasonable pose、只覆盖 short-horizon atomic、需真机数据微调）。`strategy.md §2.4` 的"数据工厂 + $/video-hour"是 L1 内部的 AMD-infra 视角。

2. **行业真正的难题在 L2/L3。** grounding（跨具身 retarget 欠约束）、物理可行性（video 无 contact/force）、真机 verification 全在 L2/L3（详见 `strategy.md`「Real Gap」段）。L1（视频抽 3D 手 / 造预训练数据）已基本工程化。

3. **没真机 → 走 L1→L2→L3 纯 sim 闭环。** VITRA 先验 → dex-retarget 到 **sim floating 灵巧手**（LEAP/Allegro，跑 MuJoCo-Warp/Genesis）→ 量**零样本 retarget 抓取成功率 baseline** → 用 **RL 修正** 或 **Wh0 合成数据** finetune。本体先 **floating dexhand**（VITRA 原生输出：手腕 6DoF + 15 手指关节，无需臂 IK），机械臂后续再加。

### L2 post-train 怎么做：human→robot 的 gap 与三种 adaptation

VITRA 是一条**端到端 VLA**（图像→latent→动作，最终输出是**人手动作**不是 latent）。落机器人有**两条正交的 gap**，且 3 个干预点**都作用在 VITRA 这一个模型上**（越往下越深、越吃机器人数据）：

```
                 输入 gap                                                        输出 gap
      (预训练看人手ego视频 / 真机看机械手相机)                          (人手动作空间 ↔ 机器人关节)
                    │                                                              │
   ego RGB ─────────┼──> [SigLIP 冻结] ──img feat──> [Gemma-2 LLM] ──f^c(latent)──> [DiT 动作头] ──> 手部动作 chunk
   手状态 s_t ───────────────────────────────────────────────────────────────────────┘             (手腕6DoF+手指关节角)
                                                                    │                                        │
   ③ 端到端续训 ◄─────────────────────────────────────────────────┘ (整个 VITRA:LLM+DiT 一起微调)          │
   ② latent 适配 ◄─── 抽 f^c 训小 adapter (VITRA 冻结)                                                       │
   ① 输出后处理 ◄──────────────────────────────────────────────────────────────────── 拿最终动作几何 retarget ┘ (零训练)
```

- **输入 gap**（图像域）：预训练输入是人手 ego 视频、真机输入是机器人相机里的机械手/夹爪 —— 由 vision encoder 侧解决（③续训时靠上层适应；或 SigLIP 泛化）。
- **输出 gap**（动作空间）：VITRA 输出在**人手动作空间**，机器人要**自身关节** —— 就是下表 3 种 adaptation 干预的地方。两条 gap 独立，别互相抵消。

跨**输出 gap** 的 adaptation 有三个层级：

| 层级 | 做法（干预点） | 需要的数据 | 本项目对应 |
|---|---|---|---|
| **① pose-level retargeting** | 拿 VITRA **最终输出**的手部动作 → 几何/优化 retarget 到机器人关节；**VITRA 全冻结** | 无需训练，几何/优化 | **Do-as-I-Do**（MuJoCo-Warp MPC）+ dex-retargeting，✅ 已在 ROCm |
| **② latent-level adaptation** ⚠️**最低优先级/最不推荐** | 抽中间 latent `f^c` → 训一个小 adapter/新头映射到机器人动作；**VITRA 大部分冻结，只训 adapter**（把 VITRA 当特征器，VITRA 官方未走此路）。**⚠️ 从 `f^c` 切出即绕过 DiT 动作头 → 丢弃 VITRA 在 1M episodes 上学到的动作先验（最值钱的部分），却仍要机器人数据训 adapter → 性价比最差；只在 ①③ 都不可行时才考虑** | 少量配对 / 机器人数据 | 待做（L2 微调，实际不走） |
| **③ policy fine-tuning** | **整个 VITRA VLA（LLM+DiT 头）端到端续训** + **动作空间对齐**（人手动作空间当机器人的超集，`transfer_xhand_to_human` 把机器人关节映射进人手表示、EEF 统一相机系）——**VITRA 官方真机微调 recipe** | 机器人 action 数据（sim rollout / Wh0） | 待做（L2 微调） |

没真机的落法：**先走 ①**（零训练）——VITRA 手轨 → dex-retarget 到 sim floating 灵巧手，量**零样本抓取成功率 baseline**（最省、最快出数字）；**baseline 不够再上 ③**（sim rollout / Wh0 数据端到端 policy fine-tuning，保留并微调 DiT 动作先验；**②因丢弃动作先验默认不走**），补物理靠 L3（sim contact/force + RL 修正）。

---

## §2 VITRA


### 2.1 训练输入

VITRA = VLM(PaliGemma-2 3B) + DiT 扩散动作头，**把"人手当灵巧机器人末端执行器"**。输入两路：

- **VLM 路（感知/条件）**：`ego RGB 帧(224²)` + `语言指令` + `相机 FoV token` + 可学习 `cognition token` → cognition feature `f^c`。
- **DiT 动作头路**：输入 = `concat(f^c, 当前手状态 s_t, 加噪动作 chunk a_t..a_{t+N})` + action mask → 迭代去噪预测未来动作（MSE on noise）。
  - **手状态 s_t** = 当前帧手腕**相机系平移+旋转** + **各手指关节角**。
  - **动作** = 未来一段**手部运动 chunk**（手腕位姿 + 关节角序列）。

### 2.2 产出数据样式（VITRA-1M / Wh0-WM-H 格式）

每 episode 一个 `.npy`，官方 schema：

```text
episode.npy
├── anno_type            # 该段主手部动作(分段依据)
├── language             # 自动生成的语言描述
├── camera               # 校正后内参/外参 (intrinsics/extrinsics)
└── left / right (每只手一份 MANO 3D 手部轨迹)
    ├── beta                       (10)         MANO 形状参数
    ├── global_orient_worldspace   (T×3×3)      canonical→world 手腕旋转
    ├── hand_pose                  (T×15×3×3)   15 个手指关节局部旋转
    ├── transl_worldspace          (T×3)        手腕世界系平移(米制)
    ├── joints_camspace            (T×21×3)     21 关节相机系 3D 位置
    └── joints_worldspace          (T×21×3)     21 关节世界系 3D 位置
```

- **本质**：不是像素/action token，而是"**分段 + 语言 + 相机 + 逐帧米制 3D 手轨(MANO)**"的结构化标注 —— humanoid-ready 中间态，落具体机器人手需 **retarget**（Do-as-I-Do MPC）。原始视频因版权不分发，只给 metadata + 复现脚本。
- **数据源**（官方 `data.md`）：**Ego4D 948k**（厨房 454k+其它 494k）· **EPIC 154k** · **EgoExo4D 67k** · **SSv2 53k** = 1.22M

### 2.3 VITRA-1M 外的可扩数据源（M1' 扩数据参考）


**第一人称 + 人手（同 VITRA-1M 范式，最省事，直接接 HaWoR）**

| 数据集 | 形态 | 可下性 | 需 HaWoR? | 差异化角度 |
|---|---|---|---|---|
| **EgoDex**（Apple, 2025） | Vision Pro ego，自带 3D 手轨（逐关节 4×4 transform，非 MANO），~829h/338k ep/194 任务 | **Apple CDN zip**（非 HF，无部分下载）：`ml-site.cdn-apple.com/datasets/egodex/{part1..5,test,extra}.zip`。**入门用 test.zip 16GB≈2.8k ep**（≤10k 够用） | 可跳过 HaWoR（自带手轨），但需薄适配 transform→MANO | 规模大、日常灵巧操作；DexWM 用的就是它；CC-BY-NC-ND |
| **HOT3D**（Meta, 2024） | Aria+Quest3 ego，手+物 3D（MANO/UmeTrack） | 开放 | 可跳过（GT 手位姿） | 手-物 3D 真值质量高 |
| **Nymeria**（Meta, 2024） | Aria ego，全身+手，日常活动，超大 | 开放 | 需（多为 body，手需补） | 户外/移动场景，VITRA-1M 偏厨房家务 |
| **H2O**（2021） | 第一人称手-物交互，MANO | 开放 | 可跳过 | 早期但干净 |


**EgoDex 的正确用法**（自带 Vision Pro 3D 手 GT）：✅ **当 HaWoR 精度标尺**——跑 EgoDex 视频→对比 GT，量 W-MPJPE/WA-MPJPE/RTE（不需 MANO 拟合），**首次给数据工厂产出质量一个数字**。

---


## §3 具身 world model / VLA 的 infra 命门（vs LLM 栈）

这条线不是单一同构 Transformer 图，而是**异构多阶段 pipeline**（感知原语 → 视频扩散/DiT WM → VLA → MuJoCo-Warp 部署，见 `§0` 使能组件表）。infra 命门与 LLM serving 完全不同，按重要性排：

1. **视频数据 IO**：视频帧存储带宽比 text token 大 1~2 量级，H264/JPEG 解码 + 重前处理（SLAM/深度/MANO）常先于 GPU 饱和。
2. **异构管线的自定义 kernel 可移植性**：lietorch / DROID-SLAM / gsplat / nvdiffrast 是 **ROCm 移植真正的阻塞**（`WIP §2`），LLM 移植几乎无此问题。
3. **计算模式**：WM/VLA 主流是 diffusion/DiT 的固定步数去噪（compute-bound），优化靠步数蒸馏 / 特征缓存，非 AR decode。
4. **物理仿真在环**：WM 当数据发生器时与几千并行 env 的 GPU 物理仿真耦合，LLM 栈无对应物。
5. **实时闭环控制延迟**：VLA 部署约束是控制频率下的硬实时单请求延迟（10~100Hz），非 LLM serving 的多用户吞吐。

**KV cache 在这边靠边站**：WM 主流是 diffusion（训练无 KV、推理是 compute-bound）；VLA 部署是单请求硬实时，continuous batching / PagedAttention 价值大幅下降。KV cache 恰是 LLM 侧最独特、这边最不重要的东西之一。

---

