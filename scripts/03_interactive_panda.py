"""
Day 2 - 交互式控制 Panda 关节
目的：观察单个关节转动对机械臂姿态的影响
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
model = mujoco.MjModel.from_xml_path(PANDA_XML)
data = mujoco.MjData(model)

# Panda 的"工厂默认"姿态（一个常用的初始位形，让机械臂垂直立起）
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0, 0.04, 0.04])

# 设到初始位形
data.qpos[:] = HOME_QPOS

# 让 actuator 维持当前位置（Panda 模型里是位置伺服，所以 ctrl 设到目标关节角就会保持）
data.ctrl[:7] = HOME_QPOS[:7]

# 启动 viewer
print("=" * 60)
print("Panda 关节扫描测试")
print("=" * 60)
print("机械臂将依次扫描每个关节，观察哪节杆在动")
print("按 Ctrl+C 中断，或关闭窗口退出")
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 先静止 3 秒让你看清初始姿态
    print("初始姿态（HOME），3 秒后开始扫描...")
    t_start = time.time()
    while time.time() - t_start < 3.0 and viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    
    # 依次让每个关节做一次往复运动
    for joint_idx in range(7):
        if not viewer.is_running():
            break
        
        print(f">>> 现在测试 joint{joint_idx + 1}（观察哪节杆在动）")
        
        # 让该关节做一个 ±0.8 弧度（约 ±45°）的往复运动
        t_start = time.time()
        duration = 4.0  # 每个关节测试 4 秒
        
        while time.time() - t_start < duration and viewer.is_running():
            t = time.time() - t_start
            offset = 0.8 * np.sin(2 * np.pi * t / duration)
            
            target = HOME_QPOS.copy()
            target[joint_idx] = HOME_QPOS[joint_idx] + offset
            
            data.ctrl[:7] = target[:7]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    # 回到 HOME 静止
    print(">>> 测试结束，回到 HOME 姿态")
    data.ctrl[:7] = HOME_QPOS[:7]
    t_start = time.time()
    while time.time() - t_start < 3.0 and viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("结束")
