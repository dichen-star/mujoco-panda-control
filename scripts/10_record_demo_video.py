"""
Day 7 - 录制圆轨迹跟踪演示视频（无 viewer，离屏渲染）
"""
import numpy as np
import mujoco
import imageio.v2 as imageio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_HOME_THETA


PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
OUTPUT_PATH = "logs/day7_circle_demo.mp4"

# 圆轨迹参数
CIRCLE_CENTER = np.array([0.5, 0.0, 0.5])
CIRCLE_RADIUS = 0.15
CIRCLE_PERIOD = 5.0

# 视频参数
FPS = 30
WIDTH, HEIGHT = 640, 480
TOTAL_DURATION = 8.0  # 1 秒预热 + 5 秒画圆 + 2 秒收尾


def main():
    print("Generating IK sequence...")
    n_frames = int(TOTAL_DURATION * FPS)
    
    # 计算每帧对应的关节角
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    T_home = panda_fk(PANDA_HOME_THETA)
    R_target = T_home[:3, :3]
    
    theta_sequence = []
    theta_current = PANDA_HOME_THETA.copy()
    
    for frame_idx in range(n_frames):
        t = frame_idx / FPS
        
        if t < 1.0:
            # 前 1 秒：HOME 静止
            theta_sequence.append(PANDA_HOME_THETA.copy())
            continue
        elif t > 6.0:
            # 6 秒后：保持最后位姿
            theta_sequence.append(theta_sequence[-1])
            continue
        
        # 1-6 秒：画圆
        circle_t = (t - 1.0) / CIRCLE_PERIOD * 2 * np.pi
        pos = CIRCLE_CENTER + CIRCLE_RADIUS * np.array(
            [np.cos(circle_t), np.sin(circle_t), 0]
        )
        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = pos
        
        theta_solved, success, _ = panda_ik(
            T_target, theta_init=theta_current, max_iter=100
        )
        if success:
            theta_current = theta_solved
        theta_sequence.append(theta_current.copy())
    
    print("  Generated {} frames of joint angles".format(len(theta_sequence)))
    
    # 离屏渲染
    print("\nRendering video...")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    
    # 相机视角
    camera = mujoco.MjvCamera()
    camera.lookat = np.array([0.4, 0.0, 0.5])
    camera.distance = 1.6
    camera.azimuth = 130
    camera.elevation = -25
    
    frames = []
    for frame_idx, theta in enumerate(theta_sequence):
        data.qpos[:7] = theta
        data.qpos[7:] = 0.04
        mujoco.mj_forward(model, data)
        
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)
        
        if (frame_idx + 1) % 30 == 0:
            print("  Rendered {}/{} frames".format(frame_idx + 1, n_frames))
    
    # 保存
    print("\nWriting MP4...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimsave(OUTPUT_PATH, frames, fps=FPS, codec='libx264', quality=8)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print("\nVideo saved to: {}".format(OUTPUT_PATH))
    print("File size: {:.2f} MB".format(size_mb))


if __name__ == "__main__":
    main()