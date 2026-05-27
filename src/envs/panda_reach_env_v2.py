"""
PandaReachEnv v2:支持 sparse reward 和 curriculum learning
==============================================================
继承自 v1,保持完全向后兼容。新增:
- reward_type: "dense" | "sparse"
- target_range_scale: 0.0-1.0,缩放目标采样范围(用于 curriculum)
"""
import numpy as np
from src.envs.panda_reach_env import PandaReachEnv


class PandaReachEnvV2(PandaReachEnv):
    """
    新增参数:
        reward_type: "dense" (Day 14 默认) 或 "sparse"
        target_range_scale: 1.0 = 完整工作空间,0.5 = 半径减半,etc.
    """

    def __init__(
        self,
        reward_type: str = "dense",
        target_range_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert reward_type in ("dense", "sparse"), \
            f"reward_type must be 'dense' or 'sparse', got {reward_type}"
        self.reward_type = reward_type

        # 工作空间中心(home 朝向时末端大致位置附近)
        self._target_center = (self.target_low + self.target_high) / 2.0
        self._target_half_range = (self.target_high - self.target_low) / 2.0

        self.set_target_range_scale(target_range_scale)

    def set_target_range_scale(self, scale: float):
        """动态调整目标采样范围(curriculum 调用)"""
        assert 0.0 < scale <= 1.0, f"scale must be in (0, 1], got {scale}"
        self._range_scale = scale
        # 围绕中心点,按 scale 缩放半径
        self.target_low = self._target_center - self._target_half_range * scale
        self.target_high = self._target_center + self._target_half_range * scale

    def _compute_reward(self):
        ee_pos = self.data.xpos[self.hand_id]
        dist = float(np.linalg.norm(ee_pos - self.target))
        terminated = dist < self.success_thresh

        if self.reward_type == "dense":
            reward = -dist
            if terminated:
                reward += 10.0
        else:  # sparse
            reward = 1.0 if terminated else 0.0

        return reward, terminated, dist