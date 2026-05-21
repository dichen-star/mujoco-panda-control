"""
笛卡尔阻抗控制 (Cartesian Impedance Control)

3-DoF 平动版本：

τ = J(q)^T [K_cart·(x_d - x) + D_cart·(ẋ_d - ẋ)] + G(q)

末端表现为虚拟"弹簧 + 阻尼"挂在目标位置 x_d
"""
import numpy as np
import mujoco
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

from src.kinematics.forward import panda_fk, panda_jacobian
from src.controllers.pd_gravity import compute_gravity


def compute_impedance_torque(model, data, x_d, xdot_d, K_cart, D_cart, 
                              include_gravity=True):
    """
    笛卡尔阻抗控制（3-DoF 平动）
    
    Args:
        model, data: MuJoCo 模型和数据
        x_d:    末端期望位置 (3,)
        xdot_d: 末端期望速度 (3,)
        K_cart: 笛卡尔刚度 (3,) 或 (3,3) 矩阵
        D_cart: 笛卡尔阻尼 (3,) 或 (3,3) 矩阵
        include_gravity: 是否加重力补偿（默认 True）
    
    Returns:
        tau: (7,) 关节力矩
    """
    q = data.qpos[:7].copy()
    qdot = data.qvel[:7].copy()
    
    # 1. 末端当前位置（用 panda_fk）
    T_current = panda_fk(q)
    x = T_current[:3, 3]  # 平动部分
    
    # 2. 末端当前速度
    # 用 MuJoCo 的 mj_jac 接口直接获取"末端原点"线速度雅可比
    # 这避免了空间雅可比的扭曲约定问题
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jacp = np.zeros((3, model.nv))  # 线速度雅可比
    jacr = np.zeros((3, model.nv))  # 角速度雅可比
    mujoco.mj_jac(model, data, jacp, jacr, T_current[:3, 3], hand_id)
    
    J_v = jacp[:, :7]            # 取前 7 列（机械臂关节，跳过手指）
    xdot = J_v @ qdot            # 3, 末端原点世界速度
    
    # 3. 末端位置/速度误差
    x_err = x_d - x              # 3,
    xdot_err = xdot_d - xdot     # 3,
    
    # 4. 笛卡尔虚拟弹簧 + 阻尼力
    if K_cart.ndim == 1:
        F_spring = K_cart * x_err
        F_damper = D_cart * xdot_err
    else:
        F_spring = K_cart @ x_err
        F_damper = D_cart @ xdot_err
    
    F_cart = F_spring + F_damper  # 3,
    
    # 5. 雅可比转置映射到关节力矩
    tau_cart = J_v.T @ F_cart    # 7,
    
    # 6. 加重力补偿（只加 G，不加科氏力，否则高速运动时会失控）
    if include_gravity:
        G = compute_gravity(model, data, q)
        tau = tau_cart + G
    else:
        tau = tau_cart
    
    return tau


# 默认参数：中等刚度
PANDA_K_CART_DEFAULT = np.array([200.0, 200.0, 200.0])  # N/m
PANDA_D_CART_DEFAULT = np.array([30.0, 30.0, 30.0])     # N·s/m

# 高刚度（接近位置控制）
PANDA_K_CART_STIFF = np.array([1000.0, 1000.0, 1000.0])
PANDA_D_CART_STIFF = np.array([60.0, 60.0, 60.0])

# 低刚度（很容易推动，柔顺）
PANDA_K_CART_SOFT = np.array([50.0, 50.0, 50.0])
PANDA_D_CART_SOFT = np.array([15.0, 15.0, 15.0])