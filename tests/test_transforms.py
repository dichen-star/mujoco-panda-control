"""
SO(3) / SE(3) 变换函数的单元测试
对照 scipy.spatial.transform.Rotation 验证手写实现
"""
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kinematics.transforms import (
    vec_to_so3, so3_to_vec,
    axis_angle_to_rot, rot_to_axis_angle,
    rot_to_quat, quat_to_rot,
    rp_to_trans, trans_inv
)


# 容差
TOL = 1e-9


def assert_close(a, b, tol=TOL, name=""):
    """通用接近性断言"""
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.max(np.abs(a - b))
    assert diff < tol, f"{name}: max diff = {diff} > {tol}"
    return True


def test_vec_so3_inverse():
    """vec_to_so3 与 so3_to_vec 互为逆"""
    np.random.seed(42)
    for _ in range(20):
        v = np.random.randn(3)
        M = vec_to_so3(v)
        v_back = so3_to_vec(M)
        assert_close(v, v_back, name="vec → so3 → vec")
        # 反对称性
        assert_close(M, -M.T, name="反对称性 M = -M^T")
    print("✅ test_vec_so3_inverse passed")


def test_axis_angle_to_rot():
    """罗德里格斯公式 vs scipy"""
    np.random.seed(0)
    for _ in range(50):
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        theta = np.random.uniform(-np.pi, np.pi)
        
        R_mine = axis_angle_to_rot(axis, theta)
        R_scipy = scipy_R.from_rotvec(axis * theta).as_matrix()
        
        assert_close(R_mine, R_scipy, tol=1e-10, name="axis_angle → R")
        # 验证 R 是合法的旋转矩阵
        assert_close(R_mine.T @ R_mine, np.eye(3), tol=1e-10, name="R^T R = I")
        assert abs(np.linalg.det(R_mine) - 1.0) < 1e-10
    print("✅ test_axis_angle_to_rot passed")


def test_rot_to_axis_angle():
    """旋转矩阵 → 轴角 → 旋转矩阵 应该回到原 R"""
    np.random.seed(1)
    for _ in range(50):
        # 随机生成 R
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        theta = np.random.uniform(0.01, np.pi - 0.01)  # 避开特殊情况
        R_orig = axis_angle_to_rot(axis, theta)
        
        # 提取轴角再转回 R
        axis_back, theta_back = rot_to_axis_angle(R_orig)
        R_back = axis_angle_to_rot(axis_back, theta_back)
        
        assert_close(R_orig, R_back, tol=1e-9, name="R → (axis, θ) → R")
    print("✅ test_rot_to_axis_angle passed")


def test_rot_to_axis_angle_special():
    """轴角的特殊情况：θ=0 和 θ≈π"""
    # θ = 0
    R = np.eye(3)
    axis, theta = rot_to_axis_angle(R)
    assert abs(theta) < 1e-6, f"θ=0 failed: theta={theta}"
    
    # θ = π
    R_pi = axis_angle_to_rot([0, 0, 1], np.pi)
    axis, theta = rot_to_axis_angle(R_pi)
    R_back = axis_angle_to_rot(axis, theta)
    assert_close(R_pi, R_back, tol=1e-9, name="θ=π special case")
    print("✅ test_rot_to_axis_angle_special passed")


def test_rot_quat_roundtrip():
    """R → quat → R 应该回到原 R"""
    np.random.seed(2)
    for _ in range(50):
        # 生成随机旋转矩阵
        R_orig = scipy_R.random(random_state=np.random.randint(10000)).as_matrix()
        
        q = rot_to_quat(R_orig)
        # 验证四元数模长为 1
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9
        
        R_back = quat_to_rot(q)
        assert_close(R_orig, R_back, tol=1e-9, name="R → quat → R")
    print("✅ test_rot_quat_roundtrip passed")


def test_quat_vs_scipy():
    """四元数与 scipy 对照（注意 scipy 是 (x,y,z,w) 而我们是 (w,x,y,z)）"""
    np.random.seed(3)
    for _ in range(30):
        R_orig = scipy_R.random(random_state=np.random.randint(10000)).as_matrix()
        
        q_mine = rot_to_quat(R_orig)  # (w, x, y, z)
        q_scipy = scipy_R.from_matrix(R_orig).as_quat()  # (x, y, z, w)
        # 重排成 (w, x, y, z)
        q_scipy_reordered = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
        
        # 注意 q 和 -q 表示同一个旋转
        if np.dot(q_mine, q_scipy_reordered) < 0:
            q_scipy_reordered = -q_scipy_reordered
        
        assert_close(q_mine, q_scipy_reordered, tol=1e-9, name="quat vs scipy")
    print("✅ test_quat_vs_scipy passed")


def test_rp_to_trans():
    """rp_to_trans 结构正确"""
    R = axis_angle_to_rot([0, 0, 1], np.pi / 3)
    p = np.array([1, 2, 3])
    T = rp_to_trans(R, p)
    
    assert T.shape == (4, 4)
    assert_close(T[:3, :3], R, name="T 的旋转部分")
    assert_close(T[:3, 3], p, name="T 的平移部分")
    assert_close(T[3, :], [0, 0, 0, 1], name="T 的最后一行")
    print("✅ test_rp_to_trans passed")


def test_trans_inv():
    """T @ T^-1 = I"""
    np.random.seed(4)
    for _ in range(30):
        R = scipy_R.random(random_state=np.random.randint(10000)).as_matrix()
        p = np.random.randn(3)
        T = rp_to_trans(R, p)
        
        T_inv_mine = trans_inv(T)
        T_inv_numpy = np.linalg.inv(T)
        
        # 对照 numpy 的通用逆
        assert_close(T_inv_mine, T_inv_numpy, tol=1e-9, name="T_inv vs np.inv")
        # T @ T^-1 = I
        assert_close(T @ T_inv_mine, np.eye(4), tol=1e-9, name="T @ T^-1 = I")
    print("✅ test_trans_inv passed")


if __name__ == "__main__":
    print("=" * 60)
    print("运行 SO(3) / SE(3) 变换函数测试")
    print("=" * 60)
    
    test_vec_so3_inverse()
    test_axis_angle_to_rot()
    test_rot_to_axis_angle()
    test_rot_to_axis_angle_special()
    test_rot_quat_roundtrip()
    test_quat_vs_scipy()
    test_rp_to_trans()
    test_trans_inv()
    
    print()
    print("=" * 60)
    print("🎉 所有测试通过")
    print("=" * 60)
