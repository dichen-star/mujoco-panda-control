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


if __name__ == "__main__":
    # 简单 smoke test：跑通就行，正式测试在 tests/test_transforms.py
    print("Smoke test...")
    R = axis_angle_to_rot([0, 0, 1], np.pi / 4)
    print(f"绕 z 轴 45° 的 R:\n{R}")
    print(f"R^T R =\n{R.T @ R}")  # 应接近 I
    print(f"det(R) = {np.linalg.det(R):.6f}")  # 应为 1
    print("OK")
