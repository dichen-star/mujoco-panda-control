"""
PD + 重力补偿控制器

τ = Kp * (q_d - q) + Kd * (q̇_d - q̇) + G(q)
"""
import numpy as np
import mujoco


def compute_pd_gravity_torque(model, data, q_d, qdot_d, Kp, Kd):
    """
    计算 PD + 重力补偿控制力矩
    
    Args:
        model, data: MuJoCo 模型和数据
        q_d:    期望关节位置 (7,)
        qdot_d: 期望关节速度 (7,)
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
    
    # 重力补偿 G(q)
    G = compute_gravity(model, data, q)
    
    # 控制律
    tau = Kp * e + Kd * edot + G
    return tau


def compute_gravity(model, data, q):
    """
    计算重力补偿项 G(q)
    
    使用 mj_rne(flg_acc=0) + qvel=0 提取
    """
    # 暂存当前状态
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    
    # 设置纯静态
    data.qpos[:7] = q
    data.qpos[7:] = 0.04
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    G = bias[:7].copy()
    
    # 恢复状态
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)
    
    return G


def disable_actuators(model):
    """
    完全禁用所有 actuator
    
    Panda 的 actuator 是 PD 位置伺服器：
        τ_act = gainprm[0] · (ctrl - qpos) + biasprm[1] · qpos + biasprm[2] · qvel
    
    把 gainprm[0]、biasprm[1]、biasprm[2] 都设为 0，actuator 输出恒为 0
    """
    for i in range(model.nu):
        model.actuator_gainprm[i][0] = 0
        model.actuator_biasprm[i][1] = 0  # 清除位置反馈
        model.actuator_biasprm[i][2] = 0  # 清除速度反馈


def apply_torque(data, tau):
    """
    把关节力矩施加到 MuJoCo qfrc_applied
    
    前提：调用前必须先 disable_actuators(model)
    """
    data.qfrc_applied[:7] = tau
    data.qfrc_applied[7:] = 0


# Panda 推荐增益（按关节大小递减）
PANDA_KP_DEFAULT = np.array([600, 600, 600, 600, 250, 150, 50])
PANDA_KD_DEFAULT = np.array([50, 50, 50, 50, 30, 25, 15])