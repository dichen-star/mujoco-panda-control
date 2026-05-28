# Day 16 日志 — Domain Randomization + 力矩控制对照(方案 B)

## 一句话总结
DR 在位置控制下几乎无效(86%→88%),诊断出根因是内置 PD 伺服吸收了动力学扰动;
改成力矩控制重做后,DR 在最坏场景把成功率从 8%→26%、阻尼偏移从 4%→26%。
**DR 的有效性取决于动作空间是否把动力学暴露给 agent。**

## 完成
- `panda_reach_env_v3.py`:4 维 DR(payload/damping/gravity/ctrl noise),reset 重采样,备份 _nominal_*
- `panda_reach_env_torque.py`:力矩环境(disable_actuators + apply_torque + qfrc_bias 重力补偿)
- `21~25` 脚本:位置版 DR / 力矩冒烟 / 力矩训练(nominal+dr) / 力矩鲁棒性对比
- 两图:`logs/day16_robustness.png`(位置)、`logs/day16_torque_robustness.png`(力矩)

## 实测

位置控制版:
| 场景 | w/o DR | w/ DR |
|---|---|---|
| 无扰动 / Payload / Damping / Gravity | 86% | 88% |
| 最坏情况 | 88% | 90% |

→ 两模型纹丝不动,DR 没有作用对象。

力矩控制版:
| 场景 | w/o DR | w/ DR | 提升 |
|---|---|---|---|
| 无扰动 | 18% | 24% | +6 |
| Payload 0.3kg | 16% | 24% | +8 |
| Damping ×1.3 | 4% | 26% | **+22 (6.5×)** |
| Gravity ×1.05 | 18% | 24% | +6 |
| 最坏情况 | 8% | 26% | **+18 (3.25×)** |

→ 标称模型在阻尼/最坏场景崩溃,DR 模型保持稳定。

## 核心结论
- 位置控制:agent 输出目标角,内置 PD 伺服执行。PD 天生抗扰(加负载自动加力矩),
  扰动在 agent 看到前已被吸收 → DR 无用武之地。
- 力矩控制:agent 自己算对抗负载/阻尼的力矩,动力学直接暴露 → 标称模型没见过偏移就崩,
  DR 模型见过各种世界就扛得住。
- 推论:工业位置控制机械臂 sim-to-real 出乎意料地容易;力矩控制(腿足/接触/柔顺)才是 DR 主战场。

## 必须诚实标注的限制
1. 绝对成功率低(力矩标称 18%):7-DoF 纯力矩控制对 PPO 极难,300k 步学不到高水平。
   本图只做**相对对比**,不宣称"DR 让任务做得好"。
2. 测试只覆盖 3 个 DR 维度:ctrl_noise 是随机量、无法固定成可复现场景,测试时关闭。
3. 低成功率下场景间几个百分点是噪声:最坏(8%)略高于 Damping ×1.3(4%),
   因最坏里阻尼是 ×1.25 较轻,且 50 ep 下只差 2 次成功,不可过度解读。

## 工程坑
- `_nominal_*` 必须备份,否则每次 reset 在已改 model 上累积偏差。
- `set_dr_params()` 后必须 `mj_forward()`,否则 model 改了 data 没刷新。
- 评估固定偏移用 `enable_dr=False` + 手动 `set_dr_params()`。
- 力矩环境先跑冒烟测试确认 apply_torque/qfrc_bias 签名,再开训练。
- 看 log 轴图先看坐标范围:位置版右图 64.9–68.2mm 的 3mm 差异被 log 放大成假悬崖;
  力矩版改线性轴(数据跨度不到半个数量级,log 添乱)。
- 图例与坐标轴勿共用 "Nominal":已改为图例=模型(Trained w/o/w DR)、横轴=场景(No perturbation…)。

