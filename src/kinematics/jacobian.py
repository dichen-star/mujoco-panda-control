"""
雅可比矩阵 - 关节速度到末端旋量的映射

J(θ) ∈ R^(6×n)：n 是关节数
V = J(θ) * θ̇

空间雅可比 J_s：末端旋量在世界坐标系下表达
物体雅可比 J_b：末端旋量在末端坐标系下表达
"""
import numpy as np
from .transforms import (
    vec_to_so3,
    vec6_to_se3,
    matrix_exp6,
    adjoint,
    trans_inv,
)


def jacobian_space(S_list, theta_list):
    """
    空间雅可比 J_s
    
    J_{s,i} = Ad_{e^[S_1]θ_1 ... e^[S_{i-1}]θ_{i-1}} * S_i
    
    Args:
        S_list: (n, 6) screw axes（零位下，空间坐标系）
        theta_list: (n,) 关节角
    Returns:
        J_s: (6, n) 空间雅可比
    """
    S_list = np.asarray(S_list)
    theta_list = np.asarray(theta_list).flatten()
    n = len(theta_list)
    
    J = np.zeros((6, n))
    T = np.eye(4)  # 累积变换
    
    for i in range(n):
        if i == 0:
            # 第 1 列：J_s,1 = S_1（伴随 = I）
            J[:, 0] = S_list[0]
        else:
            # 累乘前面的指数项
            S_prev = S_list[i - 1]
            T = T @ matrix_exp6(vec6_to_se3(S_prev * theta_list[i - 1]))
            J[:, i] = adjoint(T) @ S_list[i]
    
    return J


def jacobian_body(B_list, theta_list):
    """
    物体雅可比 J_b
    
    J_{b,i} = Ad_{e^[-B_{i+1}]θ_{i+1} ... e^[-B_n]θ_n} * B_i
    
    Args:
        B_list: (n, 6) screw axes（零位下，物体坐标系）
        theta_list: (n,) 关节角
    Returns:
        J_b: (6, n) 物体雅可比
    """
    B_list = np.asarray(B_list)
    theta_list = np.asarray(theta_list).flatten()
    n = len(theta_list)
    
    J = np.zeros((6, n))
    T = np.eye(4)
    
    # 从最后一列开始反向累积
    J[:, n - 1] = B_list[n - 1]
    for i in range(n - 2, -1, -1):
        B_next = B_list[i + 1]
        T = T @ matrix_exp6(vec6_to_se3(-B_next * theta_list[i + 1]))
        J[:, i] = adjoint(T) @ B_list[i]
    
    return J


def manipulability(J):
    """
    可操作度 μ = sqrt(det(J J^T))
    
    对 6×n 矩阵 J（n >= 6），J J^T 是 6×6 正定矩阵。
    μ = 0 表示奇异位形。
    
    Args:
        J: (6, n) 雅可比矩阵
    Returns:
        μ: 标量，可操作度
    """
    JJT = J @ J.T
    det = np.linalg.det(JJT)
    # 数值噪声可能导致微小负值
    if det < 0:
        det = 0
    return np.sqrt(det)


def smallest_singular_value(J):
    """
    雅可比的最小奇异值（数值上比可操作度更敏感地检测奇异）
    
    Args:
        J: (6, n) 雅可比矩阵
    Returns:
        sigma_min: 最小奇异值
    """
    sigmas = np.linalg.svd(J, compute_uv=False)
    return sigmas[-1]
