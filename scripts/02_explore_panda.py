"""
Day 2 - 用代码探索 Panda 的结构
目的：搞清楚 Panda 有哪些关节、怎么连接、末端在哪
"""
import mujoco
import numpy as np

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
model = mujoco.MjModel.from_xml_path(PANDA_XML)
data = mujoco.MjData(model)

# 让 mujoco 把所有 body 的位姿算出来（前向运动学）
mujoco.mj_forward(model, data)

print("=" * 70)
print("【1】Panda 有几个能转动的关节？")
print("=" * 70)
print(f"总关节数 njnt = {model.njnt}")
print(f"广义坐标维度 nq = {model.nq}（每个旋转关节占 1 维）")
print(f"控制器数量 nu = {model.nu}（你能直接控制几个关节）")
print()

print("=" * 70)
print("【2】每个关节叫什么名字？绕哪个轴转？范围是多少？")
print("=" * 70)
print(f"{'编号':<4} {'关节名':<15} {'类型':<8} {'轴向':<20} {'范围(弧度)':<25} {'范围(角度)'}")
print("-" * 100)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    
    # 关节类型：0=free, 1=ball, 2=slide, 3=hinge
    jtype = {0:"free", 1:"ball", 2:"slide", 3:"hinge"}[model.jnt_type[i]]
    
    axis = model.jnt_axis[i]
    axis_str = f"[{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]"
    
    low_rad, high_rad = model.jnt_range[i]
    low_deg, high_deg = np.degrees([low_rad, high_rad])
    range_rad_str = f"[{low_rad:+.3f}, {high_rad:+.3f}]"
    range_deg_str = f"[{low_deg:+.1f}°, {high_deg:+.1f}°]"
    
    print(f"{i:<4} {name:<15} {jtype:<8} {axis_str:<20} {range_rad_str:<25} {range_deg_str}")

print()
print("=" * 70)
print("【3】Body 是怎么连接的？（运动学链）")
print("=" * 70)
print(f"总 body 数量 nbody = {model.nbody}")
print()
print(f"{'编号':<4} {'body 名':<20} {'父 body':<20} {'初始位置 (x,y,z)'}")
print("-" * 80)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(无名)"
    parent_id = model.body_parentid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "(无名)"
    pos = data.xpos[i]
    pos_str = f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
    print(f"{i:<4} {name:<20} {parent_name:<20} {pos_str}")

print()
print("=" * 70)
print("【4】末端在哪里？")
print("=" * 70)
# 找末端 body（Panda 一般是 'hand' 或 'attachment'）
candidates = ["hand", "attachment", "link7", "end_effector"]
for cand in candidates:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
    if idx >= 0:
        pos = data.xpos[idx]
        rot = data.xmat[idx].reshape(3, 3)
        print(f"找到末端 body：'{cand}' (id={idx})")
        print(f"  零位时位置 = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  零位时旋转矩阵 =")
        print(f"  {rot}")
        break
else:
    print("没找到末端 body，请看上面 body 列表手动找")

print()
print("=" * 70)
print("【5】控制器（actuator）信息")
print("=" * 70)
print(f"{'编号':<4} {'actuator 名':<18} {'控制哪个关节':<15} {'控制范围'}")
print("-" * 70)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = model.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    low, high = model.actuator_ctrlrange[i]
    print(f"{i:<4} {name:<18} {joint_name:<15} [{low:+.3f}, {high:+.3f}]")
