"""Panda Jacobian vs MuJoCo test"""
import numpy as np
import mujoco
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kinematics.forward import panda_jacobian

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"
N_TESTS = 100
TOL = 1e-5


def mujoco_get_jacobian(model, data, theta):
    data.qpos[:7] = theta
    data.qpos[7:] = 0.04
    mujoco.mj_forward(model, data)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[hand_id], hand_id)
    jacp = jacp[:, :7]
    jacr = jacr[:, :7]
    hand_pos = data.xpos[hand_id]
    J_s = np.zeros((6, 7))
    J_s[:3, :] = jacr
    for i in range(7):
        J_s[3:, i] = jacp[:, i] - np.cross(jacr[:, i], hand_pos)
    return J_s


def test_jacobian_zero():
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    theta = np.zeros(7)
    J_mine = panda_jacobian(theta)
    J_mujoco = mujoco_get_jacobian(model, data, theta)
    err = np.max(np.abs(J_mine - J_mujoco))
    print("Zero pose test")
    print("  Max element error = {:.2e}".format(err))
    assert err < TOL, "Zero pose error too large: {}".format(err)
    print("OK test_jacobian_zero passed")
    print()


def test_jacobian_random():
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    np.random.seed(42)
    joint_lows = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_highs = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    max_err = 0
    for i in range(N_TESTS):
        theta = np.random.uniform(joint_lows, joint_highs)
        J_mine = panda_jacobian(theta)
        J_mujoco = mujoco_get_jacobian(model, data, theta)
        err = np.max(np.abs(J_mine - J_mujoco))
        max_err = max(max_err, err)
        if err > TOL:
            print("FAIL test {}: err = {:.2e}".format(i, err))
            assert False
    print("100 random poses test")
    print("  Max element error = {:.2e}".format(max_err))
    print("OK test_jacobian_random passed")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Panda Jacobian vs MuJoCo Test")
    print("=" * 60)
    print()
    test_jacobian_zero()
    test_jacobian_random()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
