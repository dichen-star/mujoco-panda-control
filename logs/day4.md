# Day 4 日志 - 正运动学 (POE)

## 完成
- 精读 Modern Robotics 第 4.1 节：旋量 / SE(3) 矩阵指数 / POE 公式
- 在 transforms.py 新增 3 个函数：vec6_to_se3, matrix_exp6, fk_in_space
- 实现 src/kinematics/forward.py：Panda 7-DoF 正运动学
- 与 MuJoCo mj_forward 对照验证 100 个随机位形
- 误差 ~1e-7（受限于 MuJoCo 内部精度，已是工程级合格）

## 学到

### 数学概念
- 旋量 V = (ω, v)：6 维向量描述瞬时运动
- screw axis S：单位化旋量，纯旋转关节 S = (ω, -ω×q)
- SE(3) 矩阵指数：e^([S]θ) → 4×4 齐次变换
- POE 公式：T = e^([S_1]θ_1) ··· e^([S_n]θ_n) · M
  - S_i 用零位定义，靠左乘的伴随表示自动处理级联

### 关键 bug & 解决
- **末端坐标系不一致**：我最初的 PANDA_M 用了 flange 朝向（x/y 轴对齐），
  但 MuJoCo 的 hand body 在此基础上绕 z 转了 -45°（夹爪手指对角线方向）。
  误差从 0.7071 跳到 2.5e-8 - 一次性 45° 偏移修正。
- 这是 Panda 模型的硬件设计怪癖（夹爪手指对齐 x-y 对角线，不是轴向）。

### 工程经验
- screw axis 数据用 Lynch 教材 + Modern Robotics 配套代码
- "位置完全对、旋转完全错"是末端坐标系不一致的典型指纹
- np.sqrt(2)/2 比手敲 0.7071068 精度高 2 个数量级
- 容差 TOL_ROT 设到 1e-6 是因为 MuJoCo 内部精度的限制

## 验证结果
- 零位：位置误差 6.94e-17，旋转误差 2.47e-08
- 100 随机位形：最大误差量级 ~1e-7（受限于 MuJoCo 内部精度）

## 明天计划 (Day 5)
- 精读 Modern Robotics 第 5 章：雅可比矩阵
- 实现 panda_jacobian(theta) → 6×7
- 与 MuJoCo mj_jac 对照验证
- 观察奇异位形（手臂完全伸直时雅可比降秩）
