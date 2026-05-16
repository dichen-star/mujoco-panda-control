"""
SO(3) / SE(3) 基础变换函数
手写实现，与 scipy.spatial.transform.Rotation 对照验证

约定：
- 旋转矩阵 R: shape (3, 3)
- 旋转轴 axis: shape (3,) 单位向量
- 四元数 q: shape (4,) 顺序 (w, x, y, z) - 标量在前
- 齐次变换 T: shape (4, 4)
- 平移 p: shape (3,)
"""
import numpy as np


def vec_to_so3(omega):
    """
    向量 → 反对称矩阵 (hat operator)
    
    Args:
        omega: (3,) 数组
    Returns:
        (3, 3) 反对称矩阵
    """
    omega = np.asarray(omega).flatten()
    return np.array([
        [0,         -omega[2],  omega[1]],
        [omega[2],   0,        -omega[0]],
        [-omega[1],  omega[0],  0       ]
    ])


def so3_to_vec(M):
    """
    反对称矩阵 → 向量 (vee operator)
    
    Args:
        M: (3, 3) 反对称矩阵
    Returns:
        (3,) 向量
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def axis_angle_to_rot(axis, theta):
    """
    罗德里格斯公式：(单位轴 + 角度) → 旋转矩阵
    
    R = I + sin(θ)[ω] + (1-cos(θ))[ω]²
    
    Args:
        axis: (3,) 数组（不必是单位向量，函数内部归一化）
        theta: 标量，弧度
    Returns:
        (3, 3) 旋转矩阵
    """
    axis = np.asarray(axis, dtype=float).flatten()
    norm = np.linalg.norm(axis)
    
    # 处理零角度（避免除以零）
    if norm < 1e-12 or abs(theta) < 1e-12:
        return np.eye(3)
    
    # 归一化
    axis = axis / norm
    omega_hat = vec_to_so3(axis)
    
    # 罗德里格斯公式
    R = np.eye(3) + np.sin(theta) * omega_hat + (1 - np.cos(theta)) * (omega_hat @ omega_hat)
    return R


def rot_to_axis_angle(R):
    """
    旋转矩阵 → (单位轴, 角度)
    
    θ = arccos((tr(R) - 1) / 2)
    [ω] = (R - R^T) / (2 sin θ)
    
    Args:
        R: (3, 3) 旋转矩阵
    Returns:
        axis: (3,) 单位向量
        theta: 标量，弧度，范围 [0, π]
    """
    R = np.asarray(R)
    
    # 算角度
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差超出 [-1, 1]
    theta = np.arccos(cos_theta)
    
    # 处理特殊情况
    if abs(theta) < 1e-6:
        # θ ≈ 0，无旋转
        return np.array([1.0, 0.0, 0.0]), 0.0
    
    if abs(theta - np.pi) < 1e-6:
        # θ ≈ π，sin(θ) ≈ 0，公式失效
        # 用 R = I + 2[ω]² 反推
        # R 的对角线给出 axis 各分量平方
        diag = np.diag(R)
        # 找最大对角元，从这一列取 axis
        i = np.argmax(diag)
        axis = np.zeros(3)
        axis[i] = np.sqrt((R[i, i] + 1) / 2)
        # 用其他元素确定剩余分量符号
        for j in range(3):
            if j != i:
                axis[j] = R[i, j] / (2 * axis[i])
        axis = axis / np.linalg.norm(axis)
        return axis, np.pi
    
    # 一般情况
    omega_hat = (R - R.T) / (2 * np.sin(theta))
    axis = so3_to_vec(omega_hat)
    return axis, theta


def rot_to_quat(R):
    """
    旋转矩阵 → 四元数 (w, x, y, z)，标量在前
    
    使用 Shepperd 方法：选取最大对角元素以避免数值问题
    
    Args:
        R: (3, 3) 旋转矩阵
    Returns:
        q: (4,) 四元数 (w, x, y, z)
    """
    R = np.asarray(R)
    trace = np.trace(R)
    
    if trace > 0:
        s = 2 * np.sqrt(trace + 1.0)  # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])  # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])  # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])  # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quat_to_rot(q):
    """
    四元数 (w, x, y, z) → 旋转矩阵
    
    Args:
        q: (4,) 四元数（标量在前）
    Returns:
        R: (3, 3) 旋转矩阵
    """
    q = np.asarray(q, dtype=float).flatten()
    q = q / np.linalg.norm(q)  # 归一化
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)]
    ])
    return R


def rp_to_trans(R, p):
    """
    R + p → 齐次变换矩阵 T
    
    T = [R p]
        [0 1]
    
    Args:
        R: (3, 3) 旋转矩阵
        p: (3,) 平移向量
    Returns:
        T: (4, 4) 齐次变换
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p).flatten()
    return T


def trans_inv(T):
    """
    齐次变换的逆（结构化公式，比 np.linalg.inv 更快更稳）
    
    T = [R p]   T^-1 = [R^T  -R^T p]
        [0 1]          [0    1     ]
    
    Args:
        T: (4, 4) 齐次变换
    Returns:
        T_inv: (4, 4) 齐次变换
    """
    R = T[:3, :3]
    p = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv
def adjoint(T):
    """
    SE(3) 的伴随表示 Ad_T，6×6 矩阵
    
    用来把旋量从一个 frame 变换到另一个 frame：
    V_b = Ad_{T_ba} * V_a
    
    Args:
        T: (4, 4) 齐次变换
    Returns:
        (6, 6) 伴随矩阵
    """
    R = T[:3, :3]
    p = T[:3, 3]
    
    p_hat = vec_to_so3(p)
    
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = p_hat @ R
    Ad[3:, 3:] = R
    return Ad
def vec6_to_se3(V):
    """
    6 维旋量向量 → 4x4 se(3) 矩阵
    
    V = (omega, v) → [V] = [[omega], v]
                            [0,      0]]
    
    Args:
        V: (6,) 数组，前 3 是 omega，后 3 是 v
    Returns:
        (4, 4) se(3) 矩阵
    """
    V = np.asarray(V).flatten()
    omega = V[:3]
    v = V[3:]
    
    se3_mat = np.zeros((4, 4))
    se3_mat[:3, :3] = vec_to_so3(omega)
    se3_mat[:3, 3] = v
    return se3_mat


def matrix_exp6(se3_mat):
    """
    SE(3) 矩阵指数
    
    输入 [V]θ（已经把 θ 吸收进去了），输出 e^([V]θ)
    
    分两种情况：
    - 旋转分量为 0（纯平移）：e^([V]θ) = I + [V]θ
    - 否则：用罗德里格斯 + G(θ) 公式
    
    Args:
        se3_mat: (4, 4) se(3) 矩阵 = [V]θ 形式
    Returns:
        (4, 4) SE(3) 齐次变换矩阵
    """
    omega_theta = so3_to_vec(se3_mat[:3, :3])  # 提取 omega*theta
    theta = np.linalg.norm(omega_theta)
    
    if theta < 1e-12:
        # 纯平移情况：e^([V]) = I + [V]（因为 [V]^2 = 0 当 omega=0）
        T = np.eye(4) + se3_mat
        return T
    
    # 一般情况：分离 omega_hat 和 theta
    omega_hat = omega_theta / theta
    omega_hat_mat = vec_to_so3(omega_hat)
    
    # 罗德里格斯公式（旋转部分）
    R = np.eye(3) + np.sin(theta) * omega_hat_mat + (1 - np.cos(theta)) * (omega_hat_mat @ omega_hat_mat)
    
    # G(theta) 矩阵（平移部分的积分系数）
    G = (np.eye(3) * theta 
         + (1 - np.cos(theta)) * omega_hat_mat 
         + (theta - np.sin(theta)) * (omega_hat_mat @ omega_hat_mat))
    
    # 提取 v_theta（已经包含 theta）
    v_theta = se3_mat[:3, 3]
    v = v_theta / theta
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T
def matrix_log6(T):
    """
    SE(3) 矩阵对数：齐次变换 T → 4x4 se(3) 矩阵 [V]*theta
    
    这是 matrix_exp6 的逆运算。
    
    Args:
        T: (4, 4) 齐次变换矩阵
    Returns:
        (4, 4) se(3) 矩阵 = [V]*theta 形式
    """
    R = T[:3, :3]
    p = T[:3, 3]
    
    # Step 1: 从 R 算轴角
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    if abs(cos_theta - 1) < 1e-9:
        # theta ≈ 0：纯平移
        se3_log = np.zeros((4, 4))
        se3_log[:3, 3] = p
        return se3_log
    
    if abs(cos_theta + 1) < 1e-9:
        # theta ≈ pi：奇点，用对角线推 axis
        theta = np.pi
        diag = np.diag(R)
        i = np.argmax(diag)
        axis = np.zeros(3)
        axis[i] = np.sqrt((R[i, i] + 1) / 2)
        for j in range(3):
            if j != i:
                axis[j] = R[i, j] / (2 * axis[i])
        omega_hat = vec_to_so3(axis)
    else:
        theta = np.arccos(cos_theta)
        omega_hat = (R - R.T) / (2 * np.sin(theta))
    
    # Step 2: 算 G^{-1}(theta)
    cot_half = 1.0 / np.tan(theta / 2)
    G_inv = (np.eye(3) / theta 
             - 0.5 * omega_hat 
             + (1.0 / theta - 0.5 * cot_half) * (omega_hat @ omega_hat))
    
    # Step 3: 拼成 se(3) 矩阵
    se3_log = np.zeros((4, 4))
    se3_log[:3, :3] = omega_hat * theta
    se3_log[:3, 3] = G_inv @ p * theta
    return se3_log


def se3_to_vec6(se3_mat):
    """
    se(3) 矩阵 → 6 维旋量向量（vec6_to_se3 的逆运算）
    
    Args:
        se3_mat: (4, 4) se(3) 矩阵
    Returns:
        (6,) 旋量向量 (omega, v)
    """
    omega = so3_to_vec(se3_mat[:3, :3])
    v = se3_mat[:3, 3]
    return np.concatenate([omega, v])

def fk_in_space(M, S_list, theta_list):
    """
    空间坐标系 POE 正运动学公式
    
    T = e^([S_1]θ_1) · e^([S_2]θ_2) ··· e^([S_n]θ_n) · M
    
    Args:
        M: (4, 4) 零位末端位姿
        S_list: (n, 6) 数组，每行是一个 screw axis
        theta_list: (n,) 关节角度
    Returns:
        T: (4, 4) 末端在空间坐标系下的位姿
    """
    M = np.asarray(M)
    S_list = np.asarray(S_list)
    theta_list = np.asarray(theta_list).flatten()
    
    T = M.copy()
    # 从最后一个关节往前累乘（注意公式是 S_1 在最左，但我们从右往左乘，先乘 M）
    for i in range(len(theta_list) - 1, -1, -1):
        S_theta = S_list[i] * theta_list[i]
        T = matrix_exp6(vec6_to_se3(S_theta)) @ T
    
    return T

if __name__ == "__main__":
    # 简单 smoke test：跑通就行，正式测试在 tests/test_transforms.py
    print("Smoke test...")
    R = axis_angle_to_rot([0, 0, 1], np.pi / 4)
    print(f"绕 z 轴 45° 的 R:\n{R}")
    print(f"R^T R =\n{R.T @ R}")  # 应接近 I
    print(f"det(R) = {np.linalg.det(R):.6f}")  # 应为 1
    print("OK")
