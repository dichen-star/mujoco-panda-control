"""
计算力矩控制 (Computed Torque Control)

τ = M(q) [q̈_d + Kp·(q_d - q) + Kd·(q̇_d - q̇)] + C(q,q̇)·q̇ + G(q)
  = M(q) [q̈_d + Kp·e + Kd·ė] + bias(q, q̇)

其中 bias = C·q̇ + G 通过 mj_rne(flg_acc=0) 提取
"""
import numpy as np
import mujoco


def compute_ctc_torque(model, data, q_d, qdot_d, qddot_d, Kp, Kd):
    """
    计算 CTC 控制力矩
    
    Args:
        model, data: MuJoCo 模型和数据
        q_d:    期望关节位置 (7,)
        qdot_d: 期望关节速度 (7,)
        qddot_d: 期望关节加速度 (7,)
        Kp:     比例增益 (7,) 或标量
        Kd:     微分增益 (7,) 或标量
    
    Returns:
        tau: (7,) 关节力矩
    """
    # 当前状态
    q = data.qpos[:7].copy()
    qdot = data.qvel[:7].copy()
    
    # 误差
    e = q_d - q
    edot = qdot_d - qdot
    
    # 算 M(q) 和 bias(q, q̇)
    M = compute_M(model, data, q)
    bias = compute_bias(model, data, q, qdot)
    
    # 控制律
    # 虚拟控制 u = q̈_d + Kp·e + Kd·ė
    u = qddot_d + Kp * e + Kd * edot
    
    # τ = M·u + bias
    tau = M @ u + bias
    return tau


def compute_M(model, data, q):
    """
    提取惯性矩阵 M(q)
    
    通过 mj_fullM 获取
    """
    # 暂存状态
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    
    # 设置纯静态状态以提取 M
    data.qpos[:7] = q
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    nv = model.nv
    M_full = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M_arm = M_full[:7, :7].copy()
    
    # 恢复状态
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)
    
    return M_arm


def compute_bias(model, data, q, qdot):
    """
    提取偏置项 bias(q, q̇) = C(q,q̇)·q̇ + G(q)
    
    使用 mj_rne(flg_acc=0)
    """
    # 暂存状态
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    
    # 设置 q 和 q̇，q̈ = 0
    data.qpos[:7] = q
    data.qpos[7:] = 0.04
    data.qvel[:7] = qdot
    data.qvel[7:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    bias_arm = bias[:7].copy()
    
    # 恢复状态
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)
    
    return bias_arm


# CTC 推荐增益（CTC 的 Kp/Kd 含义与 PD 不同）
# 因为系统已经线性化为 q̈ = u，Kp 直接对应自然频率平方
# Kp = omega_n^2,  Kd = 2·zeta·omega_n
# 推荐 omega_n = 30 rad/s, zeta = 1.0（临界阻尼）
# → Kp = 900, Kd = 60
PANDA_CTC_KP = np.array([900, 900, 900, 900, 900, 900, 900])
PANDA_CTC_KD = np.array([60, 60, 60, 60, 60, 60, 60])