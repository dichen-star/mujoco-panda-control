"""
PandaReachEnv: Franka Panda 触达任务的 Gymnasium 环境
======================================================
任务:让末端执行器触达一个随机采样的目标点(5cm 容差)

观测 (23 维):joint_pos(7) + joint_vel(7) + ee_pos(3) + target(3) + (target-ee)(3)
动作 (7 维): 关节位置增量,归一化到 [-1, 1],内部乘 action_scale
奖励:        dense,-||ee - target|| + 触达奖励 +10
终止:        ||ee - target|| < 0.05 (terminated) 或 step >= max_steps (truncated)
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

PANDA_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "assets", "mujoco_menagerie", "franka_emika_panda", "panda.xml"
)
PANDA_HOME_THETA = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])


class PandaReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        max_steps: int = 200,
        action_scale: float = 0.05,   # 每步最大关节位移 ~3°
        success_thresh: float = 0.05, # 5cm 视为成功
        render_mode: str | None = None,
    ):
        super().__init__()

        # ---- MuJoCo 模型 ----
        self.model = mujoco.MjModel.from_xml_path(PANDA_XML)
        self.data = mujoco.MjData(self.model)
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # 控制频率 50Hz / 物理仿真步长 ~0.002s → frame_skip ~10
        self.control_dt = 0.02
        self.frame_skip = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        # ---- 任务参数 ----
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.success_thresh = success_thresh

        # 目标采样范围(Panda 工作空间内)
        self.target_low = np.array([0.3, -0.3, 0.2])
        self.target_high = np.array([0.7,  0.3, 0.7])

        # 关节限位(用于 clip 防止超出物理极限)
        self.joint_low = self.model.jnt_range[:7, 0]
        self.joint_high = self.model.jnt_range[:7, 1]

        # ---- Gym 空间 ----
        obs_dim = 7 + 7 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # 内部状态
        self.target = None
        self.step_count = 0

        # 渲染
        self.render_mode = render_mode
        self.viewer = None

    # -------------------- Gym 接口 --------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 回到 home,清零速度
        self.data.qpos[:7] = PANDA_HOME_THETA
        self.data.qvel[:] = 0.0
        if self.model.nq > 7:
            self.data.qpos[7:] = 0.04
        mujoco.mj_forward(self.model, self.data)

        # 随机采样目标点
        self.target = self.np_random.uniform(self.target_low, self.target_high)
        self.step_count = 0

        return self._get_obs(), {"target": self.target.copy()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # 当前关节 + 缩放后的增量,再 clip 到关节限位
        current_q = self.data.qpos[:7].copy()
        target_q = current_q + self.action_scale * action
        target_q = np.clip(target_q, self.joint_low, self.joint_high)

        # MuJoCo 内置 actuator 跟踪 target_q
        self.data.ctrl[:7] = target_q
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, terminated, distance = self._compute_reward()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {
            "distance": distance,
            "is_success": bool(terminated),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # -------------------- 内部方法 --------------------
    def _get_obs(self):
        ee_pos = self.data.xpos[self.hand_id].copy()
        return np.concatenate([
            self.data.qpos[:7],
            self.data.qvel[:7],
            ee_pos,
            self.target,
            self.target - ee_pos,
        ]).astype(np.float32)

    def _compute_reward(self):
        ee_pos = self.data.xpos[self.hand_id]
        dist = float(np.linalg.norm(ee_pos - self.target))

        # Dense reward: 负距离(归一化到大致 [-1, 0])+ 触达奖励
        reward = -dist
        terminated = dist < self.success_thresh
        if terminated:
            reward += 10.0
        return reward, terminated, dist