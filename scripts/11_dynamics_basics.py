"""
Day 8 - 动力学方程基础 (V4)

机器人动力学方程：
    M(q)·q̈ + C(q,q̇)·q̇ + G(q) = τ
    
其中：
    q (qpos):    关节位置 (joint position)，单位 rad
    q̇ (qvel):    关节速度 (joint velocity)，单位 rad/s
    q̈ (qacc):    关节加速度 (joint acceleration)，单位 rad/s²
    M(q):        惯性矩阵 (inertia matrix, mass matrix)，7×7
    C(q,q̇):     科氏-离心矩阵 (Coriolis-centrifugal matrix)，7×7
    G(q):        重力矢量 (gravity vector)，7×1
    τ (tau):     关节力矩 (joint torque)，单位 N·m，控制器输出

MuJoCo API 关键点：
    mj_fullM:     提取完整 M 矩阵
    mj_rne(flg_acc=0):  输出 C·q̇ + G (bias terms)
    mj_rne(flg_acc=1):  在 MuJoCo 3.7 中行为不稳定，不要用
"""
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.kinematics.inverse import (
    PANDA_HOME_THETA,
    PANDA_JOINT_LOWS,
    PANDA_JOINT_HIGHS,
)

PANDA_XML = "assets/mujoco_menagerie/franka_emika_panda/panda.xml"


# ==================== 动力学基础函数 ====================

def get_M(model, data, q):
    """
    提取惯性矩阵 M(q)
    
    Args:
        q: 关节位置 (7,)
    Returns:
        M: (7, 7) 对称正定矩阵
    """
    data.qpos[:7] = q
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    nv = model.nv
    M_full = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    return M_full[:7, :7]


def get_bias(model, data, q, qdot):
    """
    提取偏置项 b(q, q̇) = C(q,q̇)·q̇ + G(q)
    
    用 mj_rne(flg_acc=0)，这是唯一可靠的方式
    
    Args:
        q:    关节位置 (7,)
        qdot: 关节速度 (7,)
    Returns:
        bias: (7,) = C·q̇ + G
    """
    data.qpos[:7] = q
    data.qpos[7:] = 0.04
    data.qvel[:7] = qdot
    data.qvel[7:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    nv = model.nv
    bias = np.zeros(nv)
    mujoco.mj_rne(model, data, 0, bias)
    return bias[:7]


def get_G(model, data, q):
    """
    提取重力矢量 G(q)
    
    令 q̇ = 0，则 bias = C·0 + G = G
    """
    return get_bias(model, data, q, np.zeros(7))


def get_Cqdot(model, data, q, qdot):
    """
    提取科氏-离心力矢量 C(q,q̇)·q̇
    
    C·q̇ = bias - G
    """
    return get_bias(model, data, q, qdot) - get_G(model, data, q)


def inverse_dynamics(model, data, q, qdot, qddot):
    """
    反向动力学：给定运动状态，计算所需关节力矩
    
    τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)
      = M(q)·q̈ + bias(q, q̇)
    
    显式计算，不依赖 mj_rne 的 flg_acc=1
    
    Args:
        q, qdot, qddot: 期望的关节运动状态
    Returns:
        tau: (7,) 所需关节力矩
    """
    M = get_M(model, data, q)
    bias = get_bias(model, data, q, qdot)
    tau = M @ qddot + bias
    return tau


def forward_dynamics_from_tau(model, data, q, qdot, tau):
    """
    正向动力学：给定力矩，计算关节加速度
    
    q̈ = M⁻¹ (τ - C·q̇ - G) = M⁻¹ (τ - bias)
    
    Args:
        q, qdot: 当前关节状态
        tau:     施加的关节力矩
    Returns:
        qddot: (7,) 关节加速度
    """
    M = get_M(model, data, q)
    bias = get_bias(model, data, q, qdot)
    qddot = np.linalg.solve(M, tau - bias)
    return qddot


# ==================== 实验 1: M 与 G 随姿态变化 ====================

def experiment_1_M_at_different_poses():
    print("=" * 70)
    print("Experiment 1: M(q) and G(q) at different poses")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    poses = {
        "HOME (curled up)": PANDA_HOME_THETA,
        "Slight extension":  np.array([0, 0.3, 0, -1.0, 0, 1.3, 0.785]),
        "Rotated sideways":  np.array([1.0, 0.3, 0, -1.0, 0, 1.3, 0.785]),
        "More curled":       np.array([0, -1.5, 0, -2.8, 0, 1.3, 0.785]),
    }
    
    for name, q in poses.items():
        if np.any(q < PANDA_JOINT_LOWS) or np.any(q > PANDA_JOINT_HIGHS):
            print("\nWARN: '{}' violates limits, skip".format(name))
            continue
        
        M = get_M(model, data, q)
        G = get_G(model, data, q)
        
        print("\nPose: {}".format(name))
        print("  joint angles = {}".format(np.round(q, 3)))
        print("  M diagonal = {}".format(np.round(np.diag(M), 3)))
        print("  M[0,0] = {:.4f} kg·m²".format(M[0, 0]))
        print("  G(q) = {}".format(np.round(G, 3)))
        print("  ||G(q)|| = {:.3f} N·m".format(np.linalg.norm(G)))


# ==================== 实验 2: 科氏力的速度二次关系 ====================

def experiment_2_coriolis_with_velocity():
    print("\n" + "=" * 70)
    print("Experiment 2: Coriolis C(q,q̇)·q̇ vs velocity (should be quadratic)")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    q = PANDA_HOME_THETA
    
    print("\nFixed pose: HOME, scaling all joint velocities uniformly")
    print("\n{:<15} {:<25} {:<25}".format(
        "speed", "||C·q̇|| (N·m)", "Dominant joint"))
    print("-" * 70)
    
    speeds = [0.1, 0.5, 1.0, 2.0, 4.0]
    norms = []
    
    for speed in speeds:
        qdot = speed * np.ones(7)
        Cqdot = get_Cqdot(model, data, q, qdot)
        norm = np.linalg.norm(Cqdot)
        norms.append(norm)
        
        max_idx = np.argmax(np.abs(Cqdot))
        print("{:<15.2f} {:<25.3f} {:<25}".format(
            speed, norm,
            "joint{}: {:.3f} N·m".format(max_idx + 1, Cqdot[max_idx])))
    
    print("\nQuadratic scaling check:")
    for i in range(1, len(speeds)):
        ratio_v = speeds[i] / speeds[i-1]
        ratio_C = norms[i] / norms[i-1]
        expected = ratio_v ** 2
        print("  speed {:.1f}x -> Cqdot {:.2f}x (expected {:.2f}x)".format(
            ratio_v, ratio_C, expected))


# ==================== 实验 3: 反向-正向动力学一致性 ====================

def experiment_3_inverse_forward_consistency():
    print("\n" + "=" * 70)
    print("Experiment 3: Inverse-Forward dynamics consistency")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    n_tests = 5
    max_errors = []
    
    print("\n{:<8} {:<20} {:<20} {:<20}".format(
        "Test", "||q̈_target||", "||τ||", "||q̈_computed - target||"))
    print("-" * 70)
    
    for i in range(n_tests):
        np.random.seed(i)
        margin = 0.3
        q = np.random.uniform(
            PANDA_JOINT_LOWS + margin, PANDA_JOINT_HIGHS - margin)
        qdot = np.random.uniform(-0.5, 0.5, 7)
        qddot_target = np.random.uniform(-1.0, 1.0, 7)
        
        # 反向动力学
        tau = inverse_dynamics(model, data, q, qdot, qddot_target)
        
        # 正向动力学
        qddot_computed = forward_dynamics_from_tau(
            model, data, q, qdot, tau)
        
        err = np.linalg.norm(qddot_computed - qddot_target)
        max_errors.append(err)
        
        print("{:<8} {:<20.4f} {:<20.4f} {:<20.2e}".format(
            i + 1, np.linalg.norm(qddot_target), np.linalg.norm(tau), err))
    
    max_err = max(max_errors)
    print("\nMax error: {:.2e}".format(max_err))
    if max_err < 1e-10:
        print("PASS: Machine-precision consistency.")
    elif max_err < 1e-3:
        print("PASS: Acceptable consistency.")
    else:
        print("FAIL: error too large.")


# ==================== 实验 4: 重力补偿可视化 ====================

def experiment_4_gravity_compensation_torques():
    print("\n" + "=" * 70)
    print("Experiment 4: Gravity torque G(q) sweep over joint2")
    print("=" * 70)
    
    model = mujoco.MjModel.from_xml_path(PANDA_XML)
    data = mujoco.MjData(model)
    
    j2_min = -1.3
    j2_max = 1.3
    j2_values = np.linspace(j2_min, j2_max, 30)
    
    G_records = []
    for j2 in j2_values:
        q = PANDA_HOME_THETA.copy()
        q[1] = j2
        if np.any(q < PANDA_JOINT_LOWS) or np.any(q > PANDA_JOINT_HIGHS):
            continue
        G = get_G(model, data, q)
        G_records.append(G)
    
    G_records = np.array(G_records)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.arange(7))
    for i in range(7):
        ax.plot(j2_values[:len(G_records)], G_records[:, i],
                color=colors[i], linewidth=2, label='joint{}'.format(i + 1))
    
    ax.set_xlabel('joint2 angle (rad)', fontsize=11)
    ax.set_ylabel('Gravity torque G(q) (N·m)', fontsize=11)
    ax.set_title('Gravity G(q) vs joint2 sweep [{:.1f}, {:.1f}] rad\n'
                 '(other joints at HOME)'.format(j2_min, j2_max), fontsize=12)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('logs/day8_gravity_torques.png', dpi=120, bbox_inches='tight')
    print("\nFigure saved to: logs/day8_gravity_torques.png")
    print("\nMax |G(q)|: {:.2f} N·m".format(np.max(np.abs(G_records))))
    
    plt.show()


if __name__ == "__main__":
    print("\nDay 8: Robot Dynamics Fundamentals (V4 - Explicit Formulas)\n")
    
    experiment_1_M_at_different_poses()
    experiment_2_coriolis_with_velocity()
    experiment_3_inverse_forward_consistency()
    experiment_4_gravity_compensation_torques()
    
    print("\n" + "=" * 70)
    print("Day 8 complete!")
    print("=" * 70)