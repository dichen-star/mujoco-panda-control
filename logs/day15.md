# Day 15 日志 - 奖励工程 + 课程学习(部分成功 + 工程教训)

## 完成
- 扩展 PandaReachEnvV2:支持 sparse reward + 动态 target range
- 实现 CurriculumCallback:基于步数自动调整任务难度
- 三方对比训练 300k 步 × 3
- 产出 logs/day15_reward_engineering.png

## 实测结果
| 实验 | Final Success | Final Reward |
|------|--------------|--------------|
| Sparse (baseline) | 0.0% | 0.00 |
| Sparse + Curriculum | 0.0% | 0.00 |
| Dense (baseline) | 100.0% | -6.70 |

## 核心结论(预期 + 意外都有)
✅ **预期成功**:Dense baseline 100% 复现 Day 14 结果,验证实验装置可信
✅ **预期失败**:Sparse 完全学不动(300k 步 0% 成功率)——这本身就是
   今天的核心论点证据:zero-information episodes 让 policy gradient
   失去方向
⚠️ **意外失败**:简单的 Curriculum 没救回 sparse——这暴露了 curriculum
   learning 的工程难度

## 对"Curriculum 失败"的诊断假设
按可能性排序:

1. **Evaluation distribution mismatch**(最可能):训练时 stage 1 用
   scale=0.2(20% 工作空间),但 EvalCallback 永远用 scale=1.0 评估
   → 早期学到的能力无法外推到评估分布。dense 任务有距离信号兜底,
   sparse 任务下这种偏移是致命的。
2. **set_target_range_scale 可能没真正传到子环境**(待验证)
3. **ent_coef=0.01 对 sparse + curriculum 探索不够**

## 关键认知
1. **Sparse 失败的根因**:zero-information episodes,policy gradient 没方向
2. **Curriculum 不是免费午餐**:理论简单但工程上极易出错。需要谨慎
   设计:stage 边界、scale 跨度、评估策略、探索系数都会影响成败
3. **诚实的失败 > 表面的成功**:本次实验装置可信(dense 完美复现),
   sparse + curriculum 没救起来这一点本身就是有价值的发现


## 后续可能尝试
- 让 eval env 跟随 curriculum 切换 scale
- Stage 1 更小(scale=0.1),让"撞到目标"概率显著提升
- 用 HER + off-policy 算法(SAC/TD3)替换 PPO

## 工程坑
- ent_coef:sparse 任务要调到 0.01 鼓励探索,dense 任务可保持 0
- 不修改 v1 环境,继承新建 v2(开闭原则)
- EvalCallback 评估策略和训练分布的一致性,在 curriculum 下是个陷阱