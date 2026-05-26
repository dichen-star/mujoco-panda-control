# Day 14 日志 - PPO 训练 PandaReach

## 完成
- scripts/17_train_ppo.py: SB3 PPO + DummyVecEnv × 8 + 250k 步
- scripts/18_eval_ppo.py: 100 episodes 三方对比 + 收敛曲线
- 训练用时:~2 分 30 秒(CPU,fps ~1700)
- 最佳模型 models/best_model.zip(250k 步 eval 100%)
- 最终模型 models/ppo_panda_reach_final.zip
- 对比图 logs/day14_ppo_eval.png

## 实测结果(100 episodes)
| Policy | Success | Final dist | Note |
|--------|---------|-----------|------|
| Random     | 0%   | 310mm | baseline |
| Greedy IK  | 90%  | 65mm  | oracle, 用 IK 模型 |
| PPO        | 81%  | 65mm  | 学出来的, 无模型 |

PPO 平均误差与 oracle 持平(都是 65mm),只在 9% 的难例上失手。

## 训练戏剧三幕
| 阶段 | 步数 | eval success | 现象 |
|------|------|--------------|------|
| 黑夜 | 0-100k    | 0%      | 看似没学,实则攒势能 |
| 挣扎 | 100-200k  | 抖动至 30% | 偶有突破又回退 |
| 黄金 | 200-260k  | 0.3 → 1.0 | 指数级爆发 |

## 训练健康指标
- explained_variance: 涨到 0.97 → Critic 拟合极好
- clip_fraction: 0.06 → 0.22(梯度信号充足,在迭代)
- std (探索强度):1.00 → 0.94(健康收敛节奏)
- eval > rollout 的"剪刀差":策略成熟期标志,关掉噪声就起飞

