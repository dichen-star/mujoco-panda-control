# Day 13 日志 - Gym + MuJoCo 环境封装

## 完成
- 实现 src/envs/panda_reach_env.py(Gymnasium 0.29+ API)
- 任务:Franka Panda 末端触达 5cm 内
- 观测 23 维 / 动作 7 维(位置增量,[-1,1] 归一化)
- 三测试通过:env_checker / random policy / greedy IK policy
- 装好 gymnasium,Day 14 准备上 SB3+PPO

## 实测
- Random policy: return ~ -100,远不收敛(预期)
- Greedy IK policy: 成功率 X/5,终末距离 <Xmm(预期)

## 关键设计决策
1. 观测加冗余特征 (target - ee):学习速度 5-10x
2. 动作选位置增量而非力矩:Day 14 才能在 100k step 看到收敛
3. Dense reward + 触达奖金:Day 15 会对比 sparse 看难度
4. terminated vs truncated 必须分清,关系价值函数正确性

## 工程坑
- frame_skip 必须四舍五入取整,float 会炸
- 关节增量必须 clip 到 jnt_range,否则会撞限位
- check_env 一定要跑,接口错误肉眼看不出来