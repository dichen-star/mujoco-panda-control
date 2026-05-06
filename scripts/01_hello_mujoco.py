"""
Day 1 - 第一个 MuJoCo 脚本
"""
import time
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light name="top" pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="link1" pos="0 0 1.2">
      <joint name="joint1" type="hinge" axis="0 1 0"/>
      <geom name="link1" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" rgba="0.8 0.2 0.2 1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint name="joint2" type="hinge" axis="0 1 0"/>
        <geom name="link2" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" rgba="0.2 0.8 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("=" * 50)
print("模型信息")
print("=" * 50)
print(f"广义坐标维度 nq = {model.nq}")
print(f"广义速度维度 nv = {model.nv}")
print(f"刚体数量 nbody = {model.nbody}")
print()

data.qpos[0] = 1.0
data.qpos[1] = 0.5

print("启动 viewer，观察双摆自由摆动")
print("按 ESC 或关闭窗口退出，30 秒后自动结束")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("仿真结束")
