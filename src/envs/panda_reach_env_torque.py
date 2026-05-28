"""
PandaReachEnvTorque: 力矩控制版的 DR 环境
============================================
继承 v3（保留全部 DR 能力），唯一区别：
  动作 = 7 维关节力矩（归一化到 [-1,1]，内部乘 torque_scale）
  不再用内置 PD 位置伺服，改为 disable_actuators + qfrc_applied

目的：让 agent 直接面对动力学，使 Domain Randomization 有作用对象。
"""
import numpy as np
import mujoco
from src.envs.panda_reach_env_v3 import PandaReachEnvV3

# 复用 Week 2 的力矩接口（PD+G 控制器模块）
from src.controllers.pd_gravity import disable_actuators, apply_torque

# 每个关节的力矩上限（Franka Panda 官方关节力矩极限，单位 N·m）
# 前 4 关节 87，后 3 关节 12，是 Panda 的标称连续力矩上限
PANDA_TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class PandaReachEnvTorque(PandaReachEnvV3):

    def __init__(self, torque_scale=None, gravity_comp=True, **kwargs):
        super().__init__(**kwargs)

        # 力矩缩放：action ∈ [-1,1] → tau = action * torque_scale
        # 默认用各关节力矩极限，让 agent 能用满动力学范围
        if torque_scale is None:
            torque_scale = PANDA_TORQUE_LIMIT.copy()
        self.torque_scale = np.asarray(torque_scale, dtype=np.float64)

        # gravity_comp: 是否给 agent 自动补偿重力
        #   True  → agent 只需学"动力学差量"，学得动，但削弱了重力 DR 的考验
        #   False → 纯力矩，最难，但 DR（含重力）考验最充分
        # 我们用 True：保证力矩控制版能在 300k 步内学出东西，
        # 同时 payload/damping 仍直接暴露给 agent，DR 依然有作用对象。
        self.gravity_comp = gravity_comp

        # 关键：禁用内置位置 actuator，改用自算力矩
        disable_actuators(self.model)

    def _compute_gravity_vec(self):
        """取当前位形的重力广义力 G(q)（7 维）。用 MuJoCo 的 qfrc_bias 的重力部分。
        简单稳健做法：把速度清零时的 qfrc_bias 当作纯重力项。"""
        # mj_forward 后 qfrc_bias = C(q,q̇)q̇ + G(q)。
        # 这里 agent 调用时刻速度不一定为零，但作为补偿项，
        # 直接用当前 qfrc_bias[:7] 作为重力+科氏的近似补偿即可（PD+G 思路）。
        return self.data.qfrc_bias[:7].copy()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # 控制噪声（继承 v3 的 sim-to-real 噪声，作用在归一化动作上）
        if self.ctrl_noise_std > 0:
            action = action + self.np_random.normal(
                0, self.ctrl_noise_std, size=action.shape)
            action = np.clip(action, -1.0, 1.0)

        # 归一化动作 → 物理力矩
        tau = action * self.torque_scale

        # 可选重力补偿（让任务可学，同时保留 payload/damping 的 DR 暴露）
        if self.gravity_comp:
            tau = tau + self._compute_gravity_vec()

        # 施加力矩，步进物理
        for _ in range(self.frame_skip):
            apply_torque(self.data, tau)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, distance = self._compute_reward()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {"distance": distance, "is_success": bool(terminated)}
        if self.enable_dr:
            info["dr_params"] = dict(self.current_dr)
        return obs, reward, terminated, truncated, info