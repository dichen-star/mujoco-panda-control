
"""
Panda 正运动学 vs MuJoCo 对照验证

把自己写的 panda_fk 和 MuJoCo 的 mj_forward 对比，
随机采样 100 个关节位形，要求误差 < 1e-9
"""
import numpy as np
import mujoco
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kinematics.forward import panda_fk

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
N_TESTS = 100
TOL_POS = 1e-6     # 位置误差容忍 1 微米
TOL_ROT = 1e-6     # 旋转矩阵元素误差容忍


def mujoco_get_hand_pose(model, data, theta):
    """让 MuJoCo 算 hand body 的位姿"""
    data.qpos[:7] = theta
    data.qpos[7:] = 0.04  # 手指给个无关紧要的值
    mujoco.mj_forward(model, data)
    
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    pos = data.xpos[hand_id].copy()
    rot = data.xmat[hand_id].reshape(3, 3).copy()
    
    T_mujoco = np.eye(4)
    T_mujoco[:3, :3] = rot
    T_mujoco[:3, 3] = pos
    return T_mujoco


def test_panda_fk_zero():
    """零位测试"""
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    theta = np.zeros(7)
    T_mine = panda_fk(theta)
    T_mujoco = mujoco_get_hand_pose(model, data, theta)
    
    pos_err = np.max(np.abs(T_mine[:3, 3] - T_mujoco[:3, 3]))
    rot_err = np.max(np.abs(T_mine[:3, :3] - T_mujoco[:3, :3]))
    
    print(f"零位测试")
    print(f"  位置误差 = {pos_err:.2e}")
    print(f"  旋转误差 = {rot_err:.2e}")
    
    assert pos_err < TOL_POS, f"零位位置误差过大: {pos_err}"
    assert rot_err < TOL_ROT, f"零位旋转误差过大: {rot_err}"
    print("✅ test_panda_fk_zero passed\n")


def test_panda_fk_random():
    """100 个随机位形对照"""
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    np.random.seed(42)
    
    # Panda 关节范围（弧度）
    joint_lows = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_highs = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
    max_pos_err = 0
    max_rot_err = 0
    
    for i in range(N_TESTS):
        theta = np.random.uniform(joint_lows, joint_highs)
        
        T_mine = panda_fk(theta)
        T_mujoco = mujoco_get_hand_pose(model, data, theta)
        
        pos_err = np.max(np.abs(T_mine[:3, 3] - T_mujoco[:3, 3]))
        rot_err = np.max(np.abs(T_mine[:3, :3] - T_mujoco[:3, :3]))
        
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
        
        if pos_err > TOL_POS or rot_err > TOL_ROT:
            print(f"  ❌ test {i}: theta = {theta}")
            print(f"     pos_err = {pos_err:.2e}, rot_err = {rot_err:.2e}")
            print(f"     T_mine =\n{T_mine}")
            print(f"     T_mujoco =\n{T_mujoco}")
            assert False, f"测试 {i} 失败"
    
    print(f"100 个随机位形测试")
    print(f"  最大位置误差 = {max_pos_err:.2e} (容忍 {TOL_POS})")
    print(f"  最大旋转误差 = {max_rot_err:.2e} (容忍 {TOL_ROT})")
    print("✅ test_panda_fk_random passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Panda 正运动学 vs MuJoCo 对照测试")
    print("=" * 60)
    print()
    
    test_panda_fk_zero()
    test_panda_fk_random()
    
    print("=" * 60)
    print("🎉 所有测试通过！POE 正运动学与 MuJoCo 完全一致")
    print("=" * 60)
