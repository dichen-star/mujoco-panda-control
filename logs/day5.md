# Day 5 日志 - 雅可比矩阵 + 奇异性分析

## 完成
- 精读 Modern Robotics 第 5.1 节
- 在 transforms.py 新增 adjoint 函数（6×6 伴随矩阵）
- 新建 src/kinematics/jacobian.py：空间/物体雅可比 + 可操作度
- 在 forward.py 新增 panda_jacobian 接口
- 与 MuJoCo mj_jac 对照验证：零位误差 1.78e-15，100 随机位形误差 2.55e-15
- 误差达到机器精度，比 Day 4 的 FK 还要精确
- 奇异性可视化：扫描 joint4 观察可操作度变化

## 学到

### 数学概念
- 雅可比 J(θ) ∈ R^(6×n)：关节速度 → 末端旋量的线性映射
- 第 i 列 = 关节 i 单位速度时末端的旋量贡献
- 空间雅可比 J_s：末端旋量在世界坐标系下表达
- 物体雅可比 J_b：末端旋量在末端坐标系下表达
- 伴随表示 Ad_T：6×6 矩阵，把旋量在不同 frame 间变换
- 可操作度 μ = sqrt(det(J*J^T))：检测奇异位形的标量指标
- 奇异位形：J 不满秩，某些方向运动能力丢失

### MuJoCo 约定差异
- MuJoCo 的 mj_jac 返回 jacp（末端原点的世界速度）+ jacr（角速度）
- Modern Robotics 的 J_s 用的"空间旋量"定义不同
- 转换公式：v_s = v_o - omega × p
- 实现这个转换后两边数值差 1e-15（机器精度）

### Panda 的奇异位形
- 零位（theta=0）：所有关节轴在 yz 平面，绕 x 轴方向丢失 → 奇异
- joint4 接近 0（手臂完全伸直）：触发主奇异 → IK 会发散
- 工作姿态（home pose）远离奇异 → μ 在 0.08 量级

## 卡点 & 解决
- WSL 长文本粘贴反复截断，多次重写测试文件
- 改用 `cat > file << EOF` here-document 写法稳定写入
- Day 6 必须切换到 VS Code + WSL 集成

## 验证结果（机器精度级匹配）
- 零位：最大元素误差 = 1.78e-15
- 100 随机位形：最大元素误差 = 2.55e-15
- 比 Day 4 的 FK 精度高 10 万倍（因为雅可比公式更直接，没有那些 sqrt(2)/2 截断）

## 明天计划 (Day 6)
- 精读 Modern Robotics 第 6.2 节：数值逆运动学
- 实现 panda_ik(T_target, theta_init) 用 Damped Least Squares (DLS)
- 测试 100 个随机目标位姿，成功率 > 95%
- 切换到 VS Code 编辑代码，告别 nano 痛苦
