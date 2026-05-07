# Day 2 日志 - 加载真实机械臂 Franka Panda

## 完成
- 克隆 mujoco_menagerie（通过 gh-proxy.com 镜像绕过国内网络问题）
- 加载并可视化 Franka Panda 7-DoF 机械臂
- 写 4 个递进的脚本：模型探索、关节扫描、正弦运动、视频录制
- matplotlib 可视化关节角曲线 + 末端 3D 轨迹
- imageio + ffmpeg 离屏渲染录制 MP4
- 更新 README，加入演示图片和视频链接

## 学到

### Panda 的运动学结构
- 9 个广义坐标：7 个臂关节（hinge）+ 2 个手指关节
- 关节嵌套结构：link0 → link1 → ... → link7 → hand
- 每个 body 的 pos 是相对父 body 的零位偏移，正是 POE 公式中 M 矩阵的来源
- 7 个 actuator 与 7 个 joint 一一对应（位置伺服）

### MuJoCo API 关键点
- `mujoco.mj_id2name()` / `mj_name2id()` 在 body/joint/actuator 之间查询
- `model.jnt_axis[i]` 拿到关节的旋转轴
- `data.xpos[body_id]` 拿到 body 的世界系位置
- `data.ctrl[i]` 是给 actuator i 的控制指令
- `mujoco.Renderer` 离屏渲染，不依赖 viewer 窗口

### 多自由度协调运动的直觉
- 7 个独立正弦信号 → 末端复杂三维花瓣轨迹
- 这正是正运动学的"非线性映射"，下周用 POE 公式精确表达

## 卡点 & 解决
- mujoco_menagerie GitHub 直连慢 → 用 `gh-proxy.com` 镜像
- viewer 拖动物体功能在 WSLg 下不灵敏 → 改用脚本主动控制（更标准）
- mujoco.Renderer 报 "framebuffer width 640" → 把视频分辨率降到 640x480
- explorer.exe 默认打开 Windows 文档目录 → 用 `wslpath -w logs` 转换路径

## 直觉建立 ✅
- joint1 = 基座 z 轴旋转
- joint2/4/6 = 肩/肘/腕的俯仰
- joint3/5/7 = 各段杆自身的旋转
- 末端位姿是 7 个关节角的非线性函数

## 明天计划 (Day 3)
- 精读 Modern Robotics 第 3 章：刚体变换
- 实现 `src/kinematics/rotations.py`：罗德里格斯公式、旋转矩阵 ↔ 四元数 ↔ 欧拉角
- 配合 scipy.spatial.transform.Rotation 写单元测试
- 不写 Panda 相关代码，专注数学基础
