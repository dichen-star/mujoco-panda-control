"""
Day 2 - 录制 Panda 正弦运动 MP4 视频不用 viewer，用离屏渲染（offscreen rendering）逐帧抓画面
"""
import numpy as np
import mujoco
import imageio.v2 as imageio
import os

# ============== 参数 ==============
PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
SIM_DURATION = 8.0    # 视频时长（秒）
FPS = 30              # 帧率
WIDTH, HEIGHT = 640, 480  # 分辨率
OUTPUT_PATH = "logs/day2_panda_sine.mp4"

HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0, 0.04, 0.04])
AMPLITUDES = np.array([0.6, 0.4, 0.5, 0.3, 0.5, 0.4, 0.8])
FREQUENCIES = np.array([0.3, 0.4, 0.5, 0.4, 0.6, 0.5, 0.7])

# ============== 加载模型 ==============
model = mujoco.MjModel.from_xml_path(PANDA_XML)
data = mujoco.MjData(model)
data.qpos[:] = HOME_QPOS
data.ctrl[:7] = HOME_QPOS[:7]
mujoco.mj_forward(model, data)

# ============== 离屏渲染器 ==============
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# 设置一个好看的相机视角（从前侧斜上方看）
camera = mujoco.MjvCamera()
camera.lookat = np.array([0.4, 0.0, 0.4])  # 相机看向的点
camera.distance = 1.8
camera.azimuth = 130    # 水平角度
camera.elevation = -20  # 俯仰角度

# ============== 仿真 + 录制 ==============
n_frames = int(SIM_DURATION * FPS)
sim_dt = model.opt.timestep
steps_per_frame = max(1, int(round(1.0 / FPS / sim_dt)))

print(f"开始录制视频")
print(f"  时长 = {SIM_DURATION}s")
print(f"  帧率 = {FPS} fps")
print(f"  分辨率 = {WIDTH}x{HEIGHT}")
print(f"  总帧数 = {n_frames}")
print(f"  仿真步长 = {sim_dt*1000:.2f}ms, 每帧 {steps_per_frame} 步")
print()

frames = []
for frame_idx in range(n_frames):
    t = frame_idx / FPS
    
    # 计算目标关节角
    target = HOME_QPOS.copy()
    target[:7] = HOME_QPOS[:7] + AMPLITUDES * np.sin(2 * np.pi * FREQUENCIES * t)
    data.ctrl[:7] = target[:7]
    
    # 推进物理（一帧画面对应多个仿真步）
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)
    
    # 渲染当前画面
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    frames.append(pixels)
    
    # 进度
    if (frame_idx + 1) % 30 == 0:
        print(f"  已渲染 {frame_idx + 1}/{n_frames} 帧 ({100*(frame_idx+1)/n_frames:.0f}%)")

# ============== 写入 MP4 ==============
print(f"\n正在写入视频...")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
imageio.mimsave(OUTPUT_PATH, frames, fps=FPS, codec='libx264', quality=8)

file_size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
print(f"\n✅ 视频已保存到: {OUTPUT_PATH}")
print(f"   文件大小: {file_size_mb:.2f} MB")
print(f"\n用以下命令打开（或 Windows 资源管理器进 logs/ 文件夹）：")
print(f"   explorer.exe logs/")
