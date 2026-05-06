# Day 1 日志 - 2026/05/06

## 完成
- WSL2 + Ubuntu 24.04 + Miniconda + MuJoCo 3.7 全套环境搭建
- 建好项目骨架 mujoco-panda-control
- 跑通双摆物理仿真，viewer 实时显示，物理表现符合预期

## 学到
- mjModel(静态) / mjData(动态) / viewer(显示) 三者职责区分
- MJCF 基本结构：worldbody → body → joint + geom
- nq 是广义坐标维度（关节角），nv 是广义速度维度
- 双摆是混沌系统，初始位形微小扰动会导致完全不同的轨迹

## 卡点 & 解决
- raw.githubusercontent.com 被墙 → Microsoft Store 装 Ubuntu
- conda 26 需要先 accept TOS 才能创建环境
- Ubuntu 24.04 libgl1-mesa-glx 已废弃，改用 libgl1
- WSL 终端 cat << EOF 多行粘贴丢换行 → 改用 nano 编辑
- 退出 viewer 时 X Error BadDrawable 是 WSLg MIT-SHM 已知问题，无害

## viewer 操作
- 右键拖动：旋转视角
- Shift+左键 / 中键：平移
- 滚轮：缩放
- Ctrl+左键：拖动物体

## 明天计划 (Day 2)
- 克隆 mujoco_menagerie 仓库
- 加载 Franka Panda 7-DoF 模型
- 阅读 panda.xml 理解真实机器人 MJCF 结构
- 写脚本让 Panda 关节做正弦运动
