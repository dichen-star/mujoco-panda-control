# Day 21 日志 — 多模态绕障：BC 均值塌缩 vs DP 多模态采样

## 完成
- src/envs/obstacle2d_env.py：2D 绕障环境 + 脚本化双模态专家（左/右两条等价路径）
- scripts/34_collect_obstacle_demos.py：采集双模态示范，切成动作分块（H=8）
- scripts/35_train_obstacle.py：在分块示范上训练 chunked BC 与 chunked DP
- scripts/36_compare_obstacle.py：Expert / BC / DP 对比 + 三联轨迹图
- 模型 models/bc/bc_obstacle.pt、models/bc/dp_obstacle.pt
- 图 logs/day21_multimodal.png

## 实测（60 episodes）
| 策略 | 成功率 | 碰撞率 |
|---|---|---|
| Expert        | 100% | 0% |
| BC (chunked)  | 75%  | 25% |
| DP (chunked)  | 100% | 0% |

BC val_mse 0.035 | DP denoise_mse 0.04（均收敛）

## 结果判定：成功
DP 在多模态任务上 100%/0% 干净战胜 BC 75%/25%。三联图直观呈现：
专家两条弧 → BC 中心一簇直行撞墙（红）+ 外围较宽的绕行（teal）→ DP 两条紧致干净的弧。

## 关键分析
### BC 为何是 75% 而非预想的 ~0%
预想"BC 把左右 chunk 平均成正上方 → 几乎全撞"过于极端。实际有三个因素让 BC 大多数时候逃出均值陷阱：
1. 起点抖动 START_JITTER=0.15：多数 episode 不在正中起步，局部数据不对称 → 平均 chunk 略偏一侧 → drift 离心。
2. 动作分块 + 重规划：开环执行一段 8 步，只要略不对称就 drift 到偏心区；下一次重规划时 obs 已在
   单模态区（只有一侧续走）→ BC 顺势绕过。分块 + 闭环反而部分帮了 BC。
3. 数据几何：专家路点到 ±0.55、很快分叉，中心偏上区域几乎无数据，偏心 agent 被拉向最近的单侧数据。
但约 25% 近乎正中起步的 episode 拿到纯均值 chunk（≈直上）→ 在打破对称前撞障碍。这 25% 是均值塌缩的诚实签名。

### DP 为何 100%
从双峰的 chunk 分布里采出一整段连贯的左弧或右弧（绝不取均值），第一段就 committed 到一个峰，
与起点无关 → 无中心簇、0 碰撞、两条紧致弧。

### 两个诚实要点
- 即便 BC 75% 看着不低，那 25% 碰撞在真机上是灾难性的，DP 完全消除 —— 定性结论完全成立。
- BC val_mse 仅 0.035 却 25% 碰撞：数据里多数 (obs,chunk) 来自偏离中心的单模态状态（误差低），
  双模态决策点是少数 → 聚合监督指标掩盖了"少而关键"的多模态决策点。必须 rollout/看轨迹才发现得了
  （与 Day 18 的"val 低 / rollout 差"同类教训）。
- BC 成功的轨迹更宽更抖（中心低数据区外推）；DP 两簇紧致干净（采样定义良好的峰）。

## 关键认知
1. 多模态：回归输出条件均值，多个等价正确动作被平均成无效中值 → 系统性失败。
2. DP 输出分布样本，可二选一 → 解决多模态；这是 ACT / Diffusion Policy 落地的核心优势。
3. 动作分块(action chunking)提供时序一致性：一次预测一段、开环执行，避免逐步独立采样在两峰间横跳；
   分块既让 DP 干净 committed，也部分帮 BC 逃离均值陷阱。
4. 聚合指标（val MSE）会掩盖少而关键的多模态决策点 —— 评估必须看 rollout / 轨迹。

## 工程要点
- BCPolicy / DiffusionPolicy 的 act_dim 是构造参数，分块只需把 act_dim 设成 H×2，无需改类。
- chunked rollout：predict 出 H×2 维 → reshape(H,2) → 开环执行 → 重规划。
- 想要更极端对比（BC→~0%）：把 START_JITTER 调到接近 0，让每个 episode 正中起步。

## 下一步
Day 22：Week 4 总结。BC → DAgger → Diffusion+chunking 三者定位成表 + 博客，
并把 4 周（运动学 → 动力学控制 → 强化学习 → 模仿学习）串成完整求职项目叙事。