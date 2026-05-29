# MuJoCo Panda Control

从经典控制到强化学习:在 MuJoCo 仿真中,用 7-DoF Franka Emika Panda 系统实现机器人
运动学、动力学、四类控制器、强化学习与域随机化,并完成跨范式对比。下一步面向具身智能。

## 周总结博客

- [Week 1:从零搭建 7-DoF 机械臂运动学库](docs/week1_summary.md)
- [Week 2:从动力学到四种控制器](docs/week2_summary.md)
- [Week 3:从控制论到强化学习,与跨范式对决](docs/week3_summary.md)

## 关键数据

| 模块 | 指标 | 对照 |
|---|---|---|
| 正运动学 (POE) | 与 MuJoCo 误差 ~1e-7 | 100 随机位形 |
| 雅可比 | 1e-15(机器精度) | 100 随机位形 |
| 逆运动学 (DLS) | 0.34 mm,97% 成功率 | 100 随机目标 + 随机重启 |
| PD + 重力补偿 | 圆轨迹稳态 0.23 mm | Day 9 |
| 计算力矩控制 (CTC) | 圆轨迹稳态 0.13 mm | Day 10 |
| 笛卡尔阻抗控制 | 胡克定律误差 1% | Day 11 |
| PPO (位置控制) | 81% 触达成功率,65 mm | 100 随机目标,与 IK oracle 持平 |
| PPO + Domain Randomization | worst case 26%(vs nominal 8%) | 力矩控制下扰动鲁棒性 3.25× |

跨范式对决(Day 17):

| 方法 | 精度 (Scene 1) | 泛化 (Scene 2) | 鲁棒 (Scene 3) |
|---|---|---|---|
| CTC (model-based) | 0.148 mm | 100% | 100% |
| PPO_pos (model-free) | 220.7 mm | 82% | 82% |
| PPO_torque_DR (robust RL) | 277.5 mm | 17% | 20% |

每种范式都有自己的强项,不存在普适最优——选型决策树详见 Week 3 总结。

## 演示

### Week 3 三方对决
![Grand Comparison](logs/day17_grand_comparison.png)

### Week 2 四控制器对比(同一圆轨迹 + 15N 扰动)
![Controller Comparison](logs/day12_controller_comparison.png)

### Week 1 圆轨迹跟踪
![Circle Tracking](logs/day7_circle_tracking.png)

视频:[`logs/day7_circle_demo.mp4`](logs/day7_circle_demo.mp4)

## 技术栈

- **仿真**: MuJoCo 3.7,Franka Panda(mujoco_menagerie)
- **强化学习**: Stable-Baselines3, Gymnasium 0.29
- **数学/控制**: Modern Robotics(POE / SE(3) 李代数 / DLS),自实现动力学补偿
- **核心库**: NumPy, SciPy, PyTorch(SB3 后端)
- **测试**: 全部核心算法与 MuJoCo 内置实现对照,精度可复现
- **开发环境**: Ubuntu 24.04 (WSL2) + VS Code Remote + Git

## 项目结构

```
mujoco-panda-control/
├── src/
│   ├── kinematics/         # FK / Jacobian / IK
│   ├── controllers/        # PD+G / CTC / 阻抗
│   └── envs/               # Gymnasium 环境封装 + DR + 力矩控制版
├── scripts/                # 27 个递进的实验脚本(按 Day 编号)
├── tests/                  # 单元测试 + 与 MuJoCo 对照验证
├── models/                 # 训练好的 PPO / DR 策略权重
├── logs/                   # 每日日志 + 实验图 + 训练曲线 + 演示视频
└── docs/                   # 周总结博客
```

## 进度

### Week 1: 运动学 ✅
- [x] Day 1: 环境搭建(WSL2 + MuJoCo)
- [x] Day 2: Panda 加载 + 关节扫描 + 视频
- [x] Day 3: SO(3) / SE(3) 变换 + 单元测试
- [x] Day 4: 正运动学(POE)
- [x] Day 5: 雅可比 + 奇异性分析
- [x] Day 6: 逆运动学(DLS,97% 成功率)
- [x] Day 7: 圆轨迹跟踪综合演示

### Week 2: 动力学 + 控制器 ✅
- [x] Day 8: 动力学量提取(M, C, G)与一致性验证
- [x] Day 9: PD + 重力补偿(圆轨迹 0.23 mm)
- [x] Day 10: 计算力矩控制(0.13 mm)
- [x] Day 11: 笛卡尔阻抗控制(胡克定律误差 1%)
- [x] Day 12: 四控制器同任务 + 15N 扰动综合对比

### Week 3: 强化学习 ✅
- [x] Day 13: Gymnasium 环境封装(23-dim obs, 7-dim action)
- [x] Day 14: PPO 训练 250k 步,81% 成功率
- [x] Day 15: Sparse reward + Curriculum 对比实验
- [x] Day 16: Domain Randomization + 力矩控制对照消融
- [x] Day 17: Model-based vs Model-free vs Robust-RL 三方对决

### Week 4: 模仿学习 + 视觉感知(计划中)
- [ ] Diffusion Policy 或 ACT 实现
- [ ] 专家数据集生成与训练
- [ ] 视觉输入(图像 → 动作)端到端策略

## 快速开始

```bash
git clone https://github.com/dichen-star/mujoco-panda-control.git
cd mujoco-panda-control

conda create -n robot python=3.10 -y
conda activate robot
pip install -r requirements.txt

# 下载机器人模型
mkdir -p assets && cd assets
git clone https://gh-proxy.com/https://github.com/google-deepmind/mujoco_menagerie.git
cd ..

# 跑 Week 1 演示
python scripts/09_circle_tracking_offline.py

# 跑 Week 2 四控制器对比
python scripts/15_controller_comparison.py

# 跑 Week 3 跨范式对决
python scripts/26_grand_comparison.py
```

## 参考资料

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control*
- 赵世钰,《强化学习的数学原理》(西湖大学,B 站公开课)
- Schulman et al., *Proximal Policy Optimization Algorithms* (PPO)
- OpenAI, *Solving Rubik's Cube with a Robot Hand* (Domain Randomization)
- DeepMind MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

## License

MIT