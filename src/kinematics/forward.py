"""
Franka Panda 7-DoF 正运动学（基于 POE）

使用 Modern Robotics 第 4 章的指数积公式：
    T(theta) = e^([S_1]θ_1) · e^([S_2]θ_2) ··· e^([S_7]θ_7) · M

screw axes 数据来源：Modern Robotics 配套代码 + Franka 官方文档
"""
import numpy as np
from .transforms import fk_in_space


# Panda 的 7 个 screw axes（在空间/世界坐标系下，零位时）
# 每行 = (omega_x, omega_y, omega_z, v_x, v_y, v_z)
PANDA_S_LIST = np.array([
    [ 0,  0,  1,  0,      0,      0    ],   # joint1: 绕 z, q=(0,0,0)
    [ 0,  1,  0, -0.333,  0,      0    ],   # joint2: 绕 y, q=(0,0,0.333)
    [ 0,  0,  1,  0,      0,      0    ],   # joint3: 绕 z, q=(0,0,0)
    [ 0, -1,  0,  0.649,  0,     -0.0825], # joint4: 绕 -y, q=(0.0825,0,0.649)
    [ 0,  0,  1,  0,      0,      0    ],   # joint5: 绕 z, q=(0,0,0)
    [ 0, -1,  0,  1.033,  0,      0    ],   # joint6: 绕 -y, q=(0,0,1.033)
    [ 0,  0, -1,  0,      0.088,  0    ],   # joint7: 绕 -z, q=(0.088,0,0)
])


# Panda 零位末端（hand body）位姿
# 末端朝下，位置 (0.088, 0, 0.926)
# 注意：hand body 相对 flange 有 -45° z 轴旋转
# cos(-π/4) = √2/2, sin(-π/4) = -√2/2
_c = np.sqrt(2) / 2
PANDA_M = np.array([
    [ _c,  _c,  0,  0.088],
    [ _c, -_c,  0,  0    ],
    [ 0,   0,  -1,  0.926],
    [ 0,   0,   0,  1    ],
])

def panda_fk(theta):
    """
    Panda 7-DoF 正运动学
    
    Args:
        theta: (7,) 关节角度（弧度）
    Returns:
        T: (4, 4) hand body 在世界坐标系下的位姿
    """
    theta = np.asarray(theta).flatten()
    if len(theta) != 7:
        raise ValueError(f"Panda 需要 7 个关节角，但收到 {len(theta)}")
    
    return fk_in_space(PANDA_M, PANDA_S_LIST, theta)
def panda_jacobian(theta):
    """
    Panda 7-DoF 空间雅可比
    
    Args:
        theta: (7,) 关节角度
    Returns:
        J_s: (6, 7) 空间雅可比矩阵
    """
    from .jacobian import jacobian_space
    
    theta = np.asarray(theta).flatten()
    if len(theta) != 7:
        raise ValueError(f"Panda 需要 7 个关节角，但收到 {len(theta)}")
    
    return jacobian_space(PANDA_S_LIST, theta)

if __name__ == "__main__":
    # 零位测试：T(0) 应该等于 M
    print("零位测试")
    print("=" * 60)
    T_zero = panda_fk(np.zeros(7))
    print(f"T(0) =\n{T_zero}")
    print(f"\nPANDA_M =\n{PANDA_M}")
    print(f"\n误差 = {np.max(np.abs(T_zero - PANDA_M)):.2e}")
    
    # 几个测试位形
    print("\n" + "=" * 60)
    print("几个测试位形")
    print("=" * 60)
    
    test_thetas = [
        np.zeros(7),
        np.array([0, -np.pi/4, 0, -np.pi/2, 0, np.pi/4, 0]),  # 类似 home
        np.array([np.pi/4, 0, 0, 0, 0, 0, 0]),  # 只转 joint1
    ]
    
    for i, th in enumerate(test_thetas):
        T = panda_fk(th)
        pos = T[:3, 3]
        print(f"\n位形 {i}: theta = {th}")
        print(f"  末端位置 = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
