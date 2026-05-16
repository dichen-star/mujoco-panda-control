"""
逆运动学（IK）- Damped Least Squares 数值算法

给定目标位姿 T_target，反推关节角 theta
"""
import numpy as np
from .transforms import matrix_log6, se3_to_vec6, trans_inv
from .forward import panda_fk, panda_jacobian


def ik_dls_space(
    T_target,
    theta_init,
    fk_func,
    jacobian_func,
    joint_lows=None,
    joint_highs=None,
    lambda_damp=0.01,
    tol_pos=1e-4,
    tol_rot=1e-3,
    max_iter=200,
):
    """
    通用 DLS 逆运动学（空间雅可比版本）
    
    Args:
        T_target: (4, 4) 目标位姿
        theta_init: (n,) 初始关节角
        fk_func: 正运动学函数 theta -> T
        jacobian_func: 雅可比函数 theta -> J
        joint_lows, joint_highs: 关节限位（可选）
        lambda_damp: DLS 阻尼系数
        tol_pos: 位置容差（米）
        tol_rot: 旋转容差（弧度）
        max_iter: 最大迭代次数
    
    Returns:
        theta: (n,) 最终关节角
        success: bool 是否收敛
        info: dict 调试信息
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    n = len(theta)
    
    history = {
        'theta': [theta.copy()],
        'pos_err': [],
        'rot_err': [],
    }
    
    for iter_count in range(max_iter):
        T_current = fk_func(theta)
        
        T_err = T_target @ trans_inv(T_current)
        err_vec = se3_to_vec6(matrix_log6(T_err))
        
        omega_err = err_vec[:3]
        v_err = err_vec[3:]
        
        pos_err_norm = np.linalg.norm(v_err)
        rot_err_norm = np.linalg.norm(omega_err)
        
        history['pos_err'].append(pos_err_norm)
        history['rot_err'].append(rot_err_norm)
        
        if pos_err_norm < tol_pos and rot_err_norm < tol_rot:
            return theta, True, {
                'iters': iter_count + 1,
                'history': history,
                'final_pos_err': pos_err_norm,
                'final_rot_err': rot_err_norm,
            }
        
        J = jacobian_func(theta)
        JJT = J @ J.T
        damped = JJT + lambda_damp**2 * np.eye(6)
        delta_theta = J.T @ np.linalg.solve(damped, err_vec)
        
        theta = theta + delta_theta
        
        if joint_lows is not None and joint_highs is not None:
            theta = np.clip(theta, joint_lows, joint_highs)
        
        history['theta'].append(theta.copy())
    
    return theta, False, {
        'iters': max_iter,
        'history': history,
        'final_pos_err': pos_err_norm,
        'final_rot_err': rot_err_norm,
    }


# Panda 关节限位
PANDA_JOINT_LOWS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_HIGHS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

PANDA_HOME_THETA = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])


def panda_ik(T_target, theta_init=None, **kwargs):
    """
    Panda 7-DoF 逆运动学（DLS）
    """
    if theta_init is None:
        theta_init = PANDA_HOME_THETA.copy()
    
    return ik_dls_space(
        T_target,
        theta_init,
        fk_func=panda_fk,
        jacobian_func=panda_jacobian,
        joint_lows=PANDA_JOINT_LOWS,
        joint_highs=PANDA_JOINT_HIGHS,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test
    print("Smoke test: FK -> IK -> FK")
    print("=" * 60)
    
    np.random.seed(0)
    theta_truth = np.random.uniform(PANDA_JOINT_LOWS + 0.5, PANDA_JOINT_HIGHS - 0.5)
    T_target = panda_fk(theta_truth)
    
    print(f"真值 theta = {theta_truth}")
    print(f"目标位姿 T = \n{T_target}")
    
    theta_solved, success, info = panda_ik(T_target)
    
    print(f"\nIK 结果:")
    print(f"  收敛 = {success}")
    print(f"  迭代次数 = {info['iters']}")
    print(f"  最终位置误差 = {info['final_pos_err']:.2e}")
    print(f"  最终旋转误差 = {info['final_rot_err']:.2e}")
    print(f"  解 theta = {theta_solved}")
    
    T_solved = panda_fk(theta_solved)
    pos_diff = np.linalg.norm(T_solved[:3, 3] - T_target[:3, 3])
    rot_diff = np.linalg.norm(T_solved[:3, :3] - T_target[:3, :3])
    print(f"\nFK 验证（解的位姿 vs 目标）:")
    print(f"  位置差 = {pos_diff:.2e}")
    print(f"  旋转差 = {rot_diff:.2e}")