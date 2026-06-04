"""
Obstacle2DEnv: 2D 点导航，中间有圆障。
多模态专家：绕障可走左可走右，两者都对。
obs = [pos_x, pos_y, goal_x-pos_x, goal_y-pos_y] (4)   action = [dx,dy]∈[-1,1] (2)
"""
import numpy as np

GOAL = np.array([0.0, 2.0])
OBSTACLE = np.array([0.0, 1.0])
OBSTACLE_R = 0.30
GOAL_R = 0.12
START_JITTER = 0.15

WAYPOINTS = {
    -1: np.array([[-0.55, 0.6], [-0.55, 1.4], [0.0, 2.0]]),
     1: np.array([[ 0.55, 0.6], [ 0.55, 1.4], [0.0, 2.0]]),
}


def expert_action(pos, side, idx, step_size):
    """脚本化专家：沿所选一侧路点前进。返回 (动作, 更新后的路点索引)。"""
    wps = WAYPOINTS[side]
    if idx < len(wps) - 1 and np.linalg.norm(pos - wps[idx]) < 0.15:
        idx += 1
    a = np.clip((wps[idx] - pos) / step_size, -1.0, 1.0)
    return a.astype(np.float32), idx


class Obstacle2DEnv:
    def __init__(self, max_steps=70, step_size=0.06):
        self.max_steps, self.step_size = max_steps, step_size
        self.obs_dim, self.act_dim = 4, 2
        self.pos, self.t = None, 0
        self._rng = np.random.RandomState()

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.pos = np.array([self._rng.uniform(-START_JITTER, START_JITTER), 0.0])
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.pos, GOAL - self.pos]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.pos = self.pos + self.step_size * np.asarray(action, dtype=np.float64)
        self.t += 1
        dist_goal = float(np.linalg.norm(self.pos - GOAL))
        dist_obs = float(np.linalg.norm(self.pos - OBSTACLE))
        terminated = truncated = success = collision = False
        reward = -dist_goal
        if dist_obs < OBSTACLE_R:
            terminated = collision = True; reward -= 10.0
        elif dist_goal < GOAL_R:
            terminated = success = True; reward += 10.0
        if (not terminated) and self.t >= self.max_steps:
            truncated = True
        return self._obs(), reward, terminated, truncated, {
            "success": success, "collision": collision, "dist_goal": dist_goal}