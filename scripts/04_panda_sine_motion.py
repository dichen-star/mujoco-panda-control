"""
Day 2 - Panda 正弦运动 + 数据记录
功能：
  - 7 个关节按不同频率做正弦运动（避免对称，让运动更"丰富"）
  - 实时仿真 + viewer 显示
  - 记录每一步的关节角和末端位姿
  - 仿真结束后自动画出关节角曲线和末端 3D 轨迹
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ============== 参数 ==============
PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
SIM_DURATION = 10.0   # 仿真总时长（秒）
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0, 0.04, 0.04])
# 每个关节的正弦运动振幅（弧度）
AMPLITUDES = np.array([0.6, 0.4, 0.5, 0.3, 0.5, 0.4, 0.8])
# 每个关节的正弦运动频率（Hz）
FREQUENCIES = np.array([0.3, 0.4, 0.5, 0.4, 0.6, 0.5, 0.7])

# ============== 加载模型 ==============
model = mujoco.MjModel.from_xml_path(PANDA_XML)
data = mujoco.MjData(model)
data.qpos[:] = HOME_QPOS
data.ctrl[:7] = HOME_QPOS[:7]
mujoco.mj_forward(model, data)

# 找末端 body 的 id（用于读取末端位姿）
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
print(f"末端 body 'hand' id = {hand_id}")

# ============== 数据记录 ==============
log_t = []           # 时间
log_q = []           # 关节角 [n_steps, 7]
log_ee_pos = []      # 末端位置 [n_steps, 3]

# ============== 主仿真循环 ==============
print(f"\n开始仿真，时长 {SIM_DURATION}s")
print(f"关节振幅 (rad): {AMPLITUDES}")
print(f"关节频率 (Hz): {FREQUENCIES}\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_start = time.time()
    while viewer.is_running() and time.time() - sim_start < SIM_DURATION:
        step_start = time.time()
        t = time.time() - sim_start
        
        # 计算 7 个关节的目标角度（HOME + 正弦扰动）
        target = HOME_QPOS.copy()
        target[:7] = HOME_QPOS[:7] + AMPLITUDES * np.sin(2 * np.pi * FREQUENCIES * t)
        
        # 把目标角写到控制器
        data.ctrl[:7] = target[:7]
        
        # 物理仿真前进一步
        mujoco.mj_step(model, data)
        
        # 记录数据（每步都记）
        log_t.append(t)
        log_q.append(data.qpos[:7].copy())
        log_ee_pos.append(data.xpos[hand_id].copy())
        
        # 同步显示 + 实时速度
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print(f"\n仿真结束，总步数 = {len(log_t)}")

# ============== 转成 numpy 数组方便画图 ==============
log_t = np.array(log_t)
log_q = np.array(log_q)
log_ee_pos = np.array(log_ee_pos)

print(f"末端位置范围:")
print(f"  x: [{log_ee_pos[:,0].min():+.3f}, {log_ee_pos[:,0].max():+.3f}] m")
print(f"  y: [{log_ee_pos[:,1].min():+.3f}, {log_ee_pos[:,1].max():+.3f}] m")
print(f"  z: [{log_ee_pos[:,2].min():+.3f}, {log_ee_pos[:,2].max():+.3f}] m")

# ============== 画图 ==============
fig = plt.figure(figsize=(14, 6))

# 子图 1：7 个关节角随时间变化
ax1 = fig.add_subplot(1, 2, 1)
for i in range(7):
    ax1.plot(log_t, log_q[:, i], label=f'joint{i+1}')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Joint Angle (rad)')
ax1.set_title('7 Joints Sinusoidal Motion')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# 子图 2：末端 3D 轨迹
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(log_ee_pos[:, 0], log_ee_pos[:, 1], log_ee_pos[:, 2], 'b-', linewidth=0.8)
ax2.scatter(log_ee_pos[0, 0], log_ee_pos[0, 1], log_ee_pos[0, 2],
            c='green', s=80, label='Start', marker='o')
ax2.scatter(log_ee_pos[-1, 0], log_ee_pos[-1, 1], log_ee_pos[-1, 2],
            c='red', s=80, label='End', marker='X')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Z (m)')
ax2.set_title('End-Effector Trajectory in 3D')
ax2.legend()

plt.tight_layout()

# 保存图（不直接 show，避免 WSL2 卡顿）
out_path = 'logs/day2_sine_motion.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\n图已保存到: {out_path}")
plt.show()
