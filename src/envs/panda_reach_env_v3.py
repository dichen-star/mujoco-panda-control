"""
PandaReachEnv v3:支持 Domain Randomization
==============================================
继承自 v2,新增 4 个可随机化的物理参数:
- 末端 payload 质量
- 关节阻尼倍率
- 重力倍率
- 控制噪声水平

每次 reset() 重新采样,episode 内保持不变。
"""
import numpy as np
import mujoco
from src.envs.panda_reach_env_v2 import PandaReachEnvV2


class PandaReachEnvV3(PandaReachEnvV2):

    def __init__(
        self,
        # DR 范围(low, high 都设为相同值即关闭 DR)
        payload_range=(0.0, 0.5),       # kg,末端附加质量
        damping_scale_range=(1.0, 1.0), # 阻尼倍率
        gravity_scale_range=(1.0, 1.0), # 重力倍率
        ctrl_noise_std=0.0,             # 控制噪声(动作单位)
        enable_dr=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.enable_dr = enable_dr
        self.payload_range = payload_range
        self.damping_scale_range = damping_scale_range
        self.gravity_scale_range = gravity_scale_range
        self.ctrl_noise_std = ctrl_noise_std

        # 备份原始物理参数(用于每次 reset 时重置)
        self._nominal_damping = self.model.dof_damping[:7].copy()
        self._nominal_gravity = self.model.opt.gravity.copy()
        self._nominal_hand_mass = self.model.body_mass[self.hand_id]

        # 当前 episode 的 DR 参数(reset 时填充)
        self.current_dr = {}

    def _sample_dr_params(self):
        """采样一组 DR 参数,返回字典"""
        rng = self.np_random
        return {
            "payload":       rng.uniform(*self.payload_range),
            "damping_scale": rng.uniform(*self.damping_scale_range),
            "gravity_scale": rng.uniform(*self.gravity_scale_range),
        }

    def _apply_dr_params(self, params):
        """把 DR 参数写入 MuJoCo 模型"""
        # 1. 末端 payload(加到 hand body 的质量上)
        self.model.body_mass[self.hand_id] = self._nominal_hand_mass + params["payload"]

        # 2. 关节阻尼
        self.model.dof_damping[:7] = self._nominal_damping * params["damping_scale"]

        # 3. 重力
        self.model.opt.gravity[:] = self._nominal_gravity * params["gravity_scale"]

    def set_dr_params(self, params):
        """手动设置 DR 参数(评估时用,固定 worst-case 偏移)"""
        self.current_dr = params
        self._apply_dr_params(params)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.enable_dr:
            self.current_dr = self._sample_dr_params()
            self._apply_dr_params(self.current_dr)
            # 重要:改了 model 后需要重新 forward 一次
            mujoco.mj_forward(self.model, self.data)

        info["dr_params"] = dict(self.current_dr)
        return self._get_obs(), info

    def step(self, action):
        # 加控制噪声
        if self.ctrl_noise_std > 0:
            action = action + self.np_random.normal(0, self.ctrl_noise_std, size=action.shape)
            action = np.clip(action, -1.0, 1.0)
        return super().step(action)