# Day 12 日志 - 四控制器综合对比 + Week 2 收官

## 完成
- 统一测试框架 scripts/15_controller_comparison.py
- 同一圆轨迹（圆心 [0.5,0,0.5], R=0.15m, T=10s）+ 15N 横向扰动
- 4 控制器同台 PK，三幕剧：自由跟踪 → 扰动 → 恢复
- 产出对比图 logs/day12_controller_comparison.png

## 实测结果（log 坐标对比）

| 控制器 | RMS(mm) | MaxDisturb(mm) | Recover(s) |
|--------|---------|----------------|------------|
| Position Servo      | 11.053 | 11.7 | not back |
| PD + Gravity        |  0.244 | 11.3 | 0.14 |
| Computed Torque     |  0.153 |  4.1 | 0.06 |
| Cartesian Impedance |  7.060 | 57.1 | 2.05 |

- PD+G 0.244mm 与 Day 9 (0.23mm) 一致 ✅
- CTC 0.153mm 与 Day 10 (0.13mm) 一致 ✅
- 阻抗 57mm = 15N / 300(N/m) = 50mm + 动态超调，胡克定律 ✅
- 伺服稳态 ~11mm 永远到不了 2mm 阈值 → not back 正常

## 关键工程认知

### 1. 力矩源冲突（项目最大坑）
- 位置伺服走 data.ctrl（内置 actuator）
- PD+G / CTC / 阻抗 走 data.qfrc_applied（自算力矩）
- 两者并存会叠加而非替换 → 自定义控制器必须 disable_actuators
- 故每个控制器需重新 from_xml_path 加载全新 model

### 2. 稳态指标必须跳过启动瞬态
- 首次跑全程 RMS 都 ~30mm，把 CTC 的 0.1mm 精度埋没
- 原因：t=0 末端在 home，圆轨迹起点不在 home，启动有 ~300mm 飞移
- 修复：稳态 RMS 从 t=1s 起算

### 3. panda_ik 返回签名不一致
- panda_ik 返回 (theta, ...) 而非纯数组
- 用 result[0] if isinstance(result, tuple) else result 兼容
- TODO: 后续统一 IK 返回签名

## 四控制器画像

| 控制器 | 跟踪精度 | 抗扰 | 柔顺 | 典型场景 |
|--------|---------|------|------|---------|
| Servo     | 差 (~11mm)  | 硬扛       | ❌ | 简单到点 |
| PD+G      | 优 (~0.2mm) | 一般       | ❌ | 通用伺服 |
| CTC       | 极优 (~0.15mm)| 优 (硬扛)| ❌ | 高精度轨迹 |
| Impedance | 良 (~2mm)   | 柔顺退让   | ✅ | 接触/协作 |

## Week 2 完成
动力学建模 → PD+G → CTC → 阻抗 → 综合对比，控制四件套闭环。