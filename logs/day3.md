# Day 3 日志 - SO(3) / SE(3) 刚体变换

## 完成
- 精读 Modern Robotics 第 3.1 - 3.2 节
- 实现 `src/kinematics/transforms.py`：8 个核心变换函数
- 写 `tests/test_transforms.py` 单元测试，全部通过（与 scipy 对照）
- 所有公式都手写实现，没有调 scipy 现成函数

## 学到

### 数学概念
- **SO(3)**：所有满足 R^T R = I 且 det(R) = 1 的 3×3 矩阵集合
- **欧拉旋转定理**：任意 3D 旋转可表示为"绕固定轴 ω 转角度 θ"
- **罗德里格斯公式**：R = I + sin(θ)[ω] + (1-cos(θ))[ω]²
- **四元数**：4 个数表示旋转，无奇异，比欧拉角稳定，比 R 紧凑
- **SE(3)**：4×4 齐次变换矩阵 T = [[R, p], [0, 1]]

### 工程经验
- 罗德里格斯公式实现时要处理 θ=0 的退化情况
- 从 R 提取轴角时 θ=π 是奇点，需要从 R 对角线特殊处理
- 四元数 q 和 -q 表示同一旋转，对照测试时要处理符号
- T 的逆要用结构化公式（R^T 替换求逆），不要 np.linalg.inv
- numpy 的数值精度限制：cos_theta 容易超出 [-1, 1]，要 clip

### 单元测试的价值
- 写完函数立刻测，比一次写完几百行再调试快 10 倍
- 用 scipy 作为"标准答案"是最直接的验证方式
- 边界情况（θ=0, θ=π, 奇异）必须显式覆盖

## 卡点 & 解决
- 实现 rot_to_axis_angle 时 θ=π 报除零错误 → 加特殊情况分支用 R 对角线推 axis
- 四元数对照 scipy 报错 → 发现 scipy 用 (x,y,z,w)，我用 (w,x,y,z)，需重排
- numpy clip 数值边界，避免 arccos 拿到 1.0000001 报 NaN

## 明天计划 (Day 4)
- 精读 Modern Robotics 第 4 章：正运动学 (Forward Kinematics)
- 实现 POE（指数积公式）
- 找到 Panda 的 7 个 screw axis，写 panda_fk(theta) → T_end
- 与 MuJoCo 的 mj_kinematics 结果对照验证
