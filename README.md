# MuJoCo Panda Control

基于 MuJoCo 的 7-DoF Franka Emika Panda 机械臂全栈控制系统。从运动学、动力学建模出发，逐步实现关节空间控制、任务空间控制、轨迹规划、模型预测控制（MPC）等核心算法，最终结合具身智能方向（Diffusion Policy / 模仿学习）。

## 演示

### Day 2 - 7 关节正弦运动

7 个关节按不同频率/振幅做协调正弦运动，末端在三维空间中画出复杂闭合轨迹。

![7 Joints Sinusoidal Motion](logs/day2_sine_motion.png)

视频效果：[`logs/day2_panda_sine.mp4`](logs/day2_panda_sine.mp4)

## 环境

- Ubuntu 24.04 (WSL2)
- Python 3.10
- MuJoCo 3.7
- 主要依赖：numpy, scipy, matplotlib, imageio

## 项目结构
mujoco-panda-control/
├── assets/
│   └── mujoco_menagerie/      # DeepMind 官方机器人模型库（panda.xml 等）
├── src/
│   ├── kinematics/             # 运动学模块：FK / Jacobian / IK
│   ├── controllers/            # 控制器模块：PD / CTC / 阻抗控制 / MPC
│   └── utils/                  # 工具函数
├── scripts/                    # 演示脚本
│   ├── 01_hello_mujoco.py     # 双摆 hello world
│   ├── 02_explore_panda.py    # 探索 Panda 模型结构
│   ├── 03_interactive_panda.py # 关节扫描可视化
│   ├── 04_panda_sine_motion.py # 7 关节正弦运动 + 数据记录
│   └── 05_record_video.py     # 离屏渲染录制 MP4
├── logs/                       # 学习日志、效果图、视频
└── README.md
## 进度

- [x] **Day 1**：环境搭建（WSL2 + MuJoCo），双摆仿真
- [x] **Day 2**：加载 Franka Panda，关节空间正弦运动，离屏渲染录制
- [ ] **Day 3**：刚体变换（SO(3) / SE(3) / 罗德里格斯公式）
- [ ] **Day 4**：正运动学（指数积公式 POE）
- [ ] **Day 5**：雅可比矩阵与奇异性分析
- [ ] **Day 6**：数值逆运动学（DLS 阻尼最小二乘法）
- [ ] **Day 7**：圆轨迹跟踪 demo + 第一周总结
- [ ] **Week 2**：动力学 + 经典控制器（PD / CTC / 阻抗控制）
- [ ] **Week 3**：轨迹规划 + 模型预测控制 (MPC)
- [ ] **Week 4**：具身智能模块（Diffusion Policy / 模仿学习）

## 快速开始

```bash
# 1. 克隆仓库及其子模块
git clone <repo-url>
cd mujoco-panda-control

# 2. 创建并激活 conda 环境
conda create -n robot python=3.10 -y
conda activate robot

# 3. 安装依赖
pip install -r requirements.txt

# 4. 克隆 mujoco_menagerie（通过镜像加速）
cd assets
git clone https://gh-proxy.com/https://github.com/google-deepmind/mujoco_menagerie.git
cd ..

# 5. 跑一个 demo
python scripts/04_panda_sine_motion.py
```

## 参考资料

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control*
- DeepMind MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- Franka Emika Panda 官方文档
