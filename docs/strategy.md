# AMD 视角下的 human→humanoid：行业缺口、定位与竞争落地

> **AMD 视角的单一事实源**：行业卡在哪（§一，无 AMD 视角）→ 站 AMD 角度什么有 impact（§二 第一性原理）→ NVIDIA 怎么布局+变现、AMD 怎么非对称应对（§三起）。

---

## 一、Real Gap in Physical AI（行业事实）

> 依据：2026 三篇综述 —— [Feng et al. *From Human Videos to Robot Manipulation*](https://arxiv.org/html/2606.00054v1)、[*From Video to Control*](https://arxiv.org/html/2604.04974v3)、[IRMVLab *Robot Learning from Human Videos*](https://ar5iv.labs.arxiv.org/html/2604.27621)，及 [VITRA (ICRA 2026)](https://arxiv.org/html/2510.21571)。

**「跑通」证明了什么**：video → 3D hand → VITRA 跑通，只说明「大规模、廉价、可自动化的**预训练先验工厂**是通的」——全链条里最成熟、最像数据处理、最不需要研究突破的一段。两点分清：(1)「训练循环能跑」（工程事实、几天）≠「产出能用的机器人」（未解研究问题、多年）；(2) VITRA 是 pretraining 先验不是可部署策略（自述只教「大致怎么动」，部署仍需 real-robot data 微调、只覆盖 short-horizon atomic、3D 噪声直接进训练数据）。**越容易跑通的那段越不是护城河。**

**行业真正卡住处（三篇综述共识）= the robotics integration layer**，四块：
- **① Episodization**：web 视频非按操作单元切好（narration 弱对齐、频繁切视角、单 clip 混多意图）→ 污染 tokenization。
- **② Grounding / Heterogeneity（核心）**：*Embodiment mismatch*——高 DoF 人手→低 DoF 机器人 retarget 欠约束，有些动作根本无可行解（夹爪越简单越严重）；*Viewpoint mismatch*——web 视频多 exo/head-mounted，接触细节遮挡。
- **③ Evaluation**：LIBERO/CALVIN/SIMPLER 与互联网人类视频多样性/长程不匹配，答不了「固定真机预算下 human-video 预训练带来多少 transfer 效率」；真机评测又贵又有安全/平台差异。
- **④ 物理缺口**：video 记 kinematics 不记 contact force/dynamics = 「action label gap」本质。

**一句话**：行业卡的是 **grounding + 真机 verification 这个 integration layer**（研究问题、非 compute-bound）；「从视频抽 3D 手 / 造预训练数据」已基本工程化。前沿在数据工厂**下游**，不在工厂本身。

---

## 二、AMD 定位：为什么有意义（第一性原理）

**总姿态**：不开具身新战场，而是把 AMD 已赢的 **LLM-inference-agent workload 延伸进一个 named-TAM vertical（Physical AI）**——具身着力 workload 全是 inference 形态，是「domain 延展」非「新市场进入」，对无生态、资源弱势的 AMD 才 buy-in 得动。

### 2.1 算法不是 AMD 战场，compute workload 才是
- **「用哪座桥」（latent / 显式 3D / world model / sim）是算法研究**，MSFT/FAIR/NVIDIA/高校推，AMD lead 不动；理解它只为定位，不下场卷。
- 范式转移定义的**新 compute workload** 在 AMD scope，但限定：是 **inference 形态的数据准备**（§2.2），不是「让 PyTorch 能训」的通用活，也不是 AMD 没锚过的训练。

### 2.2 位置在数据准备，不在训练
```
[数据准备] 海量人类视频 → decode → detection → pose → SLAM/depth → MANO 拟合 → retarget → 可训练样本 (image, 3D state, action)
[模型训练] 样本 → (pre)train / distill → 策略（VLA / WM）
```
- **训练不押**：AMD 的 LLM 大单几乎全是推理/第二源，训练仍是 NVIDIA 地盘（NVLink/NCCL/生态）。连 LLM 都没锚训练，具身更不赌训练底座——训练只当 upside。
- **数据准备 = inference 性格**：固定模型跑海量视频 = 吞吐/batch-bound 大规模推理，正踩 AMD 已验证的 inference lane。
- ROCm 几何/SLAM 库的护城河不消失，但降级为「把 $/video-hour 打下来」的手段，非 headline（§2.4-A）。

### 2.3 数据中心，而非 edge
- **端侧实时控制**（System-1、0.1s 接 MPC）跑边缘 SoC（Jetson/高通/Ryzen AI/Versal），不给数据中心 GPU 立论。
- **数据量与部署量解耦**：驱动数据中心算力的是「web-scale 视频→训练数据」这道工序，不是机器人出货量（sim rollout + WM 合成为补充源）。
- **不从 edge 入手**：AMD AI 重心/客户/design-win 都在数据中心；edge 已被 Jetson 锁死、是慢生意、违反 §5.4「只从现有大账户 land-and-expand」。

→ **Instinct 具身 base-case = 数据中心侧 human-video 数据工厂**：compute-bound、与部署量解耦、现在就存在；体量今天中等，是「押范式会起来」的赌注。

### 2.4 两个着力 workload
**A. human-video 感知数据工厂（干净冷启动，低风险）**——经济命门：*MI300X 上 1 小时原始人类视频 → 一份可训练 `(image, 3D state, action)`，要多少 GPU-秒/多少钱*。范式中心（raw-video 轨唯一缺的工序）、AMD 独占且 LLM 栈碰不到（DROID/lietorch/pytorch3d/TAPIR/Metric3D）、不与 sim 象限（RoboSmith）重复。

**B. agentic 自改进 loop（CEO-facing lead，复用资产）**——AMD 自托管开源 LLM agent + ROCm sim（RoboSmith/Genesis）闭环，[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) 类 loop 的 all-AMD 参考实现（是 §三 地图 ④/⑤ 的现实实例）。

- **分工**：B 是 CEO-facing lead（literally LLM-agent 推理的延伸）；A 是低风险技术冷启动（real-video、无 frontier-LLM/sim 依赖）。同一条 inference lane，互补。
- **caveat**：(1) KISS——B 起步用最 ready 的 RoboSmith + 单任务，Genesis-ROCm 走 CI-pilot；(2) capture 前提是 agent 用**自托管开源模型**非 API；(3) 别押单一配方（ASPIRE 是 code-as-policy，与 VLA 预训练竞争）。

---

## 三、具身算力地图：把 2.2 细化到生命周期（用 LLM→agent 当镜子）

**LLM→agent 教训**：算力重心不在一次性预训练，而漂移到循环发生的**推理 + RL/后训练 + test-time reasoning**。所以该问的不是「谁训 foundation VLA」，而是「**哪些算力循环发生、随规模膨胀、inference 形态**」。

| 场景 | compute 性格 | 何时膨胀 | 谁的护城河 | AMD 契合 |
|---|---|---|---|---|
| **① 数据/世界工厂**（人类视频→标签、部署 log 回流） | 推理，循环 | **现在** | 缝隙（NV 无专门开源栈） | ✓ 滩头 |
| ② Foundation VLA 预训练 | 训练，集中 | 早期一次性 | **NVIDIA**（NVLink/NCCL/软件） | ✗ 不 lead |
| ③ sim + world-model rollout 造数据 | 物理 sim + 生成推理 | 若「学在 WM/sim」路线赢→巨大 | **NVIDIA 垂直堡垒**（Cosmos/Newton/Isaac） | △ RoboSmith 部分 |
| ④ RL / 每具身·每任务后训练 | rollout（推理）+ 训练 | 任务品类增多时 | **contested**（scale 时多流经 ③） | △ 仅 offline/WM 版本 |
| ⑤ **fleet 级 System-2 云端推理**（运行时重规划 offload，**非**数据清洗） | 推理 | 规模部署后（若云-边分担成立） | 开放（取决于云-边分担） | ? 投机 |
| ⑥ 持续学习 / 数据飞轮 retrain | 推理+训练，循环 | 商业期 | 运营平台 | ✓ retention |
| ⑦ Edge System-1 实时控制 | 推理，SoC | 出货量巨大 | Jetson/高通 | ✗ 非 Instinct |

**三条结论**：
1. **最大最持久的蛋糕是「循环推理脊柱」①→④→⑤→⑥，不是一次性 ② 预训练**，整条踩 AMD inference lane。**数据工厂 ① 是前门**；脊柱里只有 ① 和 ⑥ AMD 稳吃，④ contested、⑤ 投机。
2. **两个不 lead 的边界**：② 预训练（NVIDIA 护城河，只当 upside）、⑦ edge（Ryzen AI/Versal，非 Instinct）。
3. **唯一改写答案的情景 = ③ sim/WM 路线赢**：那最大蛋糕变成 NVIDIA 垂直堡垒。押 real-video 数据工厂 = 押「raw video 是范式主力轨」——**最大不确定性，须持续验证 real-video 轨 vs sim/WM 轨谁跑赢。**

> **④ 归属由同一岔口决定**：real-video/offline-RL 赢→④ 是 ①/⑥ 延伸（AMD lane）；online sim rollout 赢→④ 是 ③（NVIDIA）。关键 disanalogy：能 scale 的 RL 只发生在 sim/WM（真机被 wall-clock 限死）→ 回到 ③。**⑤ 是投机 upside**：默认「System-1 端侧、System-2 在云」这个未定架构；反方（断网自主/低延迟/隐私→自包含，类比 Tesla FSD）很强，若自包含赢则 ⑤ 落空、重推理转入 ⑦。**④/⑤ 的现实实例 = Workload B**：ASPIRE 已把假设落成证据，其「search loop 极耗算力、scale 需更便宜推理」= 明确的便宜推理需求信号；做 loop 底座 = AMD 的活。

→ **对全文**：NVIDIA「三台计算机」沿 ②③⑦ 布（各一卖硅入口）；AMD 对策收敛到「**吃 ① 前门、沿循环推理脊柱扩、绕开 ②/⑦、盯住 ③ 风险**」。

---

## 四、NVIDIA 布局与变现

### 4.1 三台计算机（把「买三次卡」写成一个必然工作流）
| 计算机 | 阶段 | 硬件 | 软件/模型 |
|---|---|---|---|
| **训练** | 训基础模型 | **DGX**（Blackwell） | NeMo、GR00T 训练 |
| **仿真/合成数据** | 生成数据 + 验证 | **RTX PRO Blackwell Server** | **Omniverse + Isaac Sim/Lab + Cosmos + Newton** |
| **运行时** | 端侧实时推理 | **Jetson AGX Thor**（Blackwell，128GB，130W） | Jetson 栈、Isaac ROS、Holoscan |

不是三个产品，是**三个 GPU 销售入口**：做机器人→训练买 DGX、仿真买 RTX PRO、端侧买 Jetson。

### 4.2 技术栈（下→上）
- **硅**：DGX / RTX PRO / Jetson Thor / DRIVE Thor。
- **仿真+数据**：Omniverse（OpenUSD）、Isaac Sim 5.0、**Isaac Lab**（开源 RL/IL, BSD-3）、**Newton**（GPU 物理引擎，NV+DeepMind+Disney，捐 Linux Foundation, Apache-2.0，基于 Warp+MuJoCo-Warp）。
- **世界模型**：**Cosmos**（WFM，可控合成数据；**Cosmos Reason** = 物理推理 VLM = System-2 大脑）。
- **机器人基础模型**：**Isaac GR00T** N1.5→N1.7（商业授权 + 灵巧手）→ 预览 **N2**（基于 DreamZero）= System-1 执行。
- **框架/编排**：Isaac ROS、Lab-Arena、**OSMO**（edge-to-cloud）；深整合 **HF/LeRobot**，早期采用 Figure/Agility/BD/Amazon。

### 4.3 领域研究团队 = 漏斗源头
Toronto（Fidler，生成/仿真/USD）、Seattle（Dieter Fox）、GEAR（Jim Fan/Yuke Zhu，GR00T）。**作用 = 需求生成器**：研究产出开源上 HF/LeRobot、进榜第一 = **心智占领** → 200 万开发者绑进 CUDA = **生态锁定** → 必买三台计算机 + AI Enterprise 订阅。类比「免费爆款内容→流量→广告变现」，ROI 不在论文，在几年后「因为大家都用 GR00T」多卖的卡。

### 4.4 razor-and-blades 变现
**NVIDIA 几乎不直接靠 Physical AI 的模型/仿真/框架赚钱——它们大多免费开源。钱在**：(1) **硬件（主体）**= 三台计算机 GPU；开源模型 = demand-gen/loss-leader，把工作流锁到 CUDA+硅；(2) **软件订阅**= NVAIE/Omniverse Enterprise/NIM（production 级付费）；(3) **平台/授权**= GR00T 商业授权、Cosmos Open Model License（须标注 "Built on NVIDIA Cosmos"）；(4) **AV 单列**= DRIVE 平台 + 车厂 design-win。**开源整条栈当诱饵，把全生命周期算力钉死在自家硅+CUDA。**

> **AV 关键教训**：连 NVIDIA 都没把 AV 数据闭环框架（MagLev）做成生意——各家自建闭环，NVIDIA 卖 DRIVE 平台 + 训练 GPU。

---

## 五、AMD 非对称打法

### 5.1 为什么不能正面复制
护城河是**十年软件+生态+研究品牌，不是硬件**（Omniverse/USD/Isaac/Cosmos/GR00T + 200 万开发者 + 顶级实验室），正面追必输——拥有闭环框架不是算力厂的赢法（连 NVIDIA/MagLev 都没做成）。**总原则：不 follow 产品形态（垂直全栈/闭环），follow 商业逻辑（开源诱饵→卖硅），打它打不到的非对称点。**

### 5.2 CEO framing
把 AMD 已赢的 LLM-inference-agent workload 延伸进 named-TAM vertical（§2.4 两 workload 都是 inference 形态，是 domain 延展）。**收紧点**：(1) 可 fund 的是 **domain-informed infra 优化**（长程 tool-use loop、sim-rollout 耦合、感知吞吐、几何 kernel），不是「embodied 能跑」这种 no-op；(2) 近期定性 = **廉价期权**（低成本卡位下一个 inference 需求来源），非现在营收。被 grill 的点答案都在前文（卖多少 GPU→§2.2/§2.4；护城河是被采纳的 ROCm 库、benchmark 只是 proof；有没有人用→§七）。

### 5.3 四条动作原则
1. **搭开放标准的便车，不建平行封闭栈**：NVIDIA 自己在开放关键层（Newton LF/Apache-2.0、Isaac Lab BSD、Cosmos 开放权重、OpenUSD、LeRobot HF）→ 把它们在 ROCm 上做成一等公民。**搭车不造车。**
2. **学变现逻辑不学产品形态**：开源工具 demand-gen、硅+订阅收钱。
3. **端侧不碰 Jetson Thor**（Ryzen AI/Versal 战场）。数据中心侧才是 Instinct 主场。
4. **一个垂直楔子 = human-video 数据工厂**：NVIDIA 合成数据主押 sim+WM，**raw human video 感知反推这条线它没有专门开源栈**——差异化缝隙。

### 5.4 客户打法：照搬 LLM anchor 模式 + 两条约束
LLM 里怎么赢：MI300X 大显存/带宽（192GB）→ 更少卡装大模型；hyperscaler 锚定（第二源+成本）；co-eng 落一个深耕一个、可复用件沉回 ROCm。**具身版同一套，但选锚标准收紧**——致命时序差异：LLM 大鱼已存在且巨大，具身当下没对等鲸（humanoid 公司小、pre-revenue、花在端侧 Jetson、day1 锁 NV）。

| 维度 | LLM 推理（已验证） | 具身版（对策） |
|---|---|---|
| 锚客户 | hyperscaler（已存在的鲸） | **现有 AMD LLM 大客户向机器人数据/训练的延伸** |
| co-opt 对象 | 客户的具体推理模型 | 数据中心侧 train + 数据生成；端侧交给 Ryzen AI/Versal |
| 技术差异点 | 大显存装大模型、降成本 | 大显存装多模态 VLA/WM + 长视频上下文；异构感知流水线吞吐/$（EPYC+Instinct） |
| 落地 | co-eng + 沉回 ROCm | 贴一个锚 pipeline 做深 → kernel/wheel/recipe 沉回 ROCm |

**两条约束**：(1) 开源适配是 low-visibility 的活，**必须 anchor 拉着做**（anchor 既给可见度又给方向——先适配哪个标准），不 speculative 做无人拉动的 plumbing；(2) **只能从现有 LLM 客户延伸、不冷启动新 logo**——在已买 MI300X 的 hyperscaler/云里 land-and-expand（候选池 §7.1）。

---

## 六、造 pull 与落地

### 6.1 怎么造 pull（能力排序）
**① 开源、开发者会用的 ROCm 工具/库**（hipified 感知/几何栈 + wheel + pipeline，最真实）＞ **② 被引的 3D-grounded 数据集**（AMD 工厂跑公开视频）＞ **③ release 小 WM/VLA**（最弱，只当「证明工厂数据可训」的 proof point）。
**anchor 来源**：成本敏感 robotics 初创（要第二源）、学术实验室（便宜算力换 lighthouse + 论文 "processed on AMD Instinct"）、**AMD 内部具身 initiative（第一个 pull 常来自内部）**。

### 6.2 关键判据：上游合入 vs 下游私有移植
落地 = 给 canonical repo 提「让 ROCm/AMD 成一等公民后端」的 PR 并推 merge（进 CI、进 support matrix）。分水岭**不是「发明 vs 移植算法」**，**而是「PR 有没有 merge 进上游、让下游默认继承」**。

> **★ 落地第一步（核心行动项）：给 human-video 数据管线用到的开源 repo（LeRobot/HF/pytorch3d/SLAM 库）提 ROCm/AMD 支持 PR 并推 merge（进 CI/support matrix）。** engineer 能独立启动、不依赖渠道政治/客户切栈、直接沉淀引力；合入上游一次、下游全继承。

### 6.3 为什么核心是 CI 不是 PR 本身
一次性移植会被下个 release 的 CUDA-specific 改动静默 break，退回私有移植坑。**CI 把「一次能跑」变「持久受支持」**——每个 PR 自动在 AMD 硬件测、谁 break AMD 谁红。给三样：不回退（durable）、可见度（README "tested on ROCm" badge = 下游选型会看的）、引力沉淀。

---

## 七、市场盘点（2026）

### 7.1 AMD Instinct 大客户（anchor 候选池，对应 §5.4 约束二）
| 客户 | 代次 | 规模/承诺 | 时间线 | 用途 |
|---|---|---|---|---|
| **OpenAI** | MI450→多代 | **6 GW** 多年；首期 1 GW binding | 2025-10 签；首期 **2H 2026** 起 | 训练+推理 |
| **Meta** | 定制 MI450 + Venice EPYC | **6 GW** 多年 | 首批 **2H 2026**（Helios+ROCm）；已部署 MI300X/MI350 | 推理为主+训练 |
| **Oracle (OCI)** | MI450（Helios） | 初期 **50,000 GPU**，2027+ 扩 | 公有云 **Q3 2026**；已 GA MI300X/MI355X | 云对外售卖 |
| **Microsoft (Azure)** | MI300X→MI400 | ND MI300X v5 生产 | 已在产；MI400 深度 design | 云对外 + 自用 |

### 7.2 主流云/neocloud 机器人方案
**贯穿结论：几乎每家云的「机器人方案」= NVIDIA Isaac/Cosmos/GR00T 栈 + 自家云胶水 + 自研推理模型，GPU 底座默认 NVIDIA。** 既是现状也是 AMD 缝隙。

| 云 | 机器人方案本质 | 自研模型 | AMD 现状 | anchor 适配度 |
|---|---|---|---|---|
| **OCI** | NVIDIA 栈转售 | 无 | MI450 50k + MI355X | ★★★ 缺口最清晰 |
| **Azure** | Physical AI Toolchain（Azure+NVIDIA） | **Rho-alpha**（自研 VLA+） | MI300X 在产 | ★★★ 最差异化（自家 FM 可 co-opt） |
| **AWS** | Physical AI 参考架构 + Bedrock AgentCore+GR00T+LeRobot | Nova/Claude | 较少 | ★★ 闭环最全但锁得紧 |
| **GCP** | TPU+自研；Genie3/Cosmos 造数据；MuJoCo-Warp | **Gemini Robotics-ER 1.6** | 弱 | ★ TPU 中心，最不适合 |
| **TensorWave** | 纯算力（AMD-native，8,192 MI325X） | 无 | AMD 原生 | ★★★ 零摩擦沙盒，体量小 |

其余 neocloud（CoreWeave/Lambda/Together/Nebius/Crusoe）以 NVIDIA 为主、纯算力供给。

**三条含义**：(1) 机器人 workload 在每家云都默认路由 NVIDIA shape，即便 OCI/Azure 为 LLM 卖 AMD——缺口 = 让 human-video 感知/数据 workload 跑在这些云**已在卖的 AMD shape** 上；(2) **别跟云胶水竞争，要插进去**（挂进 SageMaker/AgentCore/Azure ML/GKE，不另造）；(3) **优先 anchor：OCI + Azure**，TensorWave 作零摩擦沙盒先验证，AWS/GCP 后。

---

## 八、做什么 / 不做什么 + 一句话

**做**：★ 给 human-video 管线用到的开源 repo（LeRobot/HF/pytorch3d/SLAM）提 ROCm/AMD 支持 PR 并推 merge（判据 §6.2）。
**不做**：不追求「拥有具身数据 + 训练闭环框架」（连 NVIDIA/MagLev 都没做成）——目标是 per-stage 默认底座；无 pull 时不重投，停在「开源工具链 + 一张成本表」的低成本高杠杆形态。

**一句话**：不 follow「垂直全栈+闭环」形态，follow「开源诱饵→卖硅」逻辑；非对称动作 = 把 NVIDIA 开放的标准层（Newton/Cosmos/LeRobot/OpenUSD）在 ROCm 上做成一等公民 + 押它没覆盖的 raw-video 数据工厂缝隙；客户只从现有 AMD LLM 大账户 land-and-expand、用 anchor 拉动否则 low-visibility 的适配。**benchmark 是 proof、被牵引出的 ROCm 库是护城河；野心止于「每段 compute 的默认底座」，不越到「拥有整个闭环」。**

---

## 附录：个人定位与取舍（在算力厂做这件事的意义）

三档里只有一档是甜点区：

| 档位 | 处境 |
|---|---|
| 通用 infra（generic ROCm/kernel） | **商品化陷阱**：差异化弱、可替代。 |
| 垂直解决方案（自己训 VLA / 做 WM 产品） | **职业陷阱**：公司 capture 不了、与客户竞争、打不过 focused startup。 |
| **垂直知识驱动的 infra**（为一个垂直真实 workload 造可复用系统层） | **甜点区**：公司能 capture（GPU-hours/默认底座），你建立别人替代不了的专长。 |

解法不是逃去做垂直方案，而是**让 infra 深锚一个垂直的真实瓶颈**（= human-video 数据工厂）——NVIDIA 护城河正是「深懂 domain workload + 会造系统层」的人堆出来的。

**历史校准（「卖铲人」）**：infra 有复利护城河（Wintel/AWS/NVIDIA），但搬到具身要过两关，否则掉「太早/太通用」的坟场：
1. **卖铲子只在淘金潮被证实时才赢，具身的潮还没被证实**（没有 ChatGPT 时刻）。→ 做法必须是**廉价期权**：押冷启动期就有用、范式没起来也不亏的那块（inference 形态、现在就存在的数据工厂）。
2. **infra 主导期在「算法稀缺期」之后才到**：具身还早（配方仍在打），今天算法还值钱，照搬 LLM 晚期观察是 timing error。

**站位**：领域知识驱动的 infra——历史奖励「懂 workload 的 infra 人」，通用 infra 会商品化；风险调整后下行保护最好（技能可迁移、会复利，timing 押错也能带「懂 infra 又懂 embodied workload」的稀缺组合切进垂直）。
**身份分岔**：认同是 systems/infra 人 → AMD 是对的地方；真正在乎具身问题本身 → 算力厂长期是错的家，把这段当攒 systems 硬实力 + 建具身 domain 认知的跳板。**取舍**：短期锁第三档（吃透数据工厂一个垂直、产出 vertical-informed ROCm 系统层），别做通用（空）、别训 VLA（越界、不 capture）；中长期取决于身份问题。
