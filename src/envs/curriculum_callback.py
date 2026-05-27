"""
Curriculum Learning 回调:基于训练步数动态调整目标采样范围
"""
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    根据训练步数,动态调整所有并行环境的 target_range_scale。

    schedule: list of (timestep_threshold, scale) 元组,按步数升序。
              例如 [(0, 0.2), (50000, 0.5), (150000, 1.0)]
              表示:
                0-50k:   scale=0.2(目标在 home 附近 20% 工作空间内)
                50k-150k: scale=0.5
                150k+:    scale=1.0(完整工作空间)
    """

    def __init__(self, schedule, verbose=0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_scale = None

    def _on_step(self) -> bool:
        # 找到当前 timestep 应该用的 scale
        scale = self.schedule[0][1]
        for threshold, s in self.schedule:
            if self.num_timesteps >= threshold:
                scale = s
            else:
                break

        # 如果 scale 变了,更新所有并行环境
        if scale != self.current_scale:
            self.current_scale = scale
            self.training_env.env_method("set_target_range_scale", scale)
            if self.verbose:
                print(f"\n[Curriculum] step {self.num_timesteps}: scale → {scale}")

        return True