"""
Panda IK 测试：100 个随机目标，统计成功率和闭环误差
使用多次随机重启策略提高成功率
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik, PANDA_JOINT_LOWS, PANDA_JOINT_HIGHS, PANDA_HOME_THETA


N_TESTS = 100
TOL_POS = 1e-3
TOL_ROT = 1e-2
MAX_RESTARTS = 4  # 最多尝试 4 个不同初值


def test_ik_random_targets():
    """100 个随机目标位姿测试，每个目标允许多次重启"""
    np.random.seed(42)
    
    n_success = 0
    n_first_try = 0  # 首次尝试就成功的（HOME 初值）
    iter_counts = []
    pos_errs = []
    rot_errs = []
    failed_indices = []
    
    for i in range(N_TESTS):
        margin = 0.3
        theta_truth = np.random.uniform(
            PANDA_JOINT_LOWS + margin, 
            PANDA_JOINT_HIGHS - margin
        )
        T_target = panda_fk(theta_truth)
        
        # 构造多个初值候选
        initial_guesses = [PANDA_HOME_THETA]
        rng = np.random.RandomState(i + 1000)
        for _ in range(MAX_RESTARTS - 1):
            random_init = rng.uniform(
                PANDA_JOINT_LOWS + margin,
                PANDA_JOINT_HIGHS - margin
            )
            initial_guesses.append(random_init)
        
        # 依次尝试每个初值，找到一个收敛就停
        success_this_target = False
        for try_idx, theta_init in enumerate(initial_guesses):
            theta_solved, success, info = panda_ik(
                T_target, theta_init=theta_init, max_iter=500
            )
            T_solved = panda_fk(theta_solved)
            pos_err = np.linalg.norm(T_solved[:3, 3] - T_target[:3, 3])
            rot_err = np.linalg.norm(T_solved[:3, :3] - T_target[:3, :3])
            
            if success and pos_err < TOL_POS and rot_err < TOL_ROT:
                success_this_target = True
                iter_counts.append(info['iters'])
                pos_errs.append(pos_err)
                rot_errs.append(rot_err)
                if try_idx == 0:
                    n_first_try += 1
                break
        
        if success_this_target:
            n_success += 1
        else:
            failed_indices.append(i)
    
    success_rate = n_success / N_TESTS
    first_try_rate = n_first_try / N_TESTS
    
    print("100 random target IK test")
    print("  Success rate = {}/{} = {:.1f}%".format(n_success, N_TESTS, success_rate * 100))
    print("  HOME init success = {}/{} = {:.1f}%".format(n_first_try, N_TESTS, first_try_rate * 100))
    if iter_counts:
        print("  Avg iterations = {:.1f}".format(np.mean(iter_counts)))
        print("  Median iterations = {:.0f}".format(np.median(iter_counts)))
        print("  Max position error = {:.2e}".format(np.max(pos_errs)))
        print("  Max rotation error = {:.2e}".format(np.max(rot_errs)))
    if failed_indices:
        print("  Failed test indices: {}".format(failed_indices[:10]))
    
    assert success_rate >= 0.90, "Success rate {:.1f}% below 90%".format(success_rate * 100)
    print("OK test_ik_random_targets passed")
    print()


def test_ik_home_target():
    """HOME 位姿 IK 测试"""
    T_home = panda_fk(PANDA_HOME_THETA)
    theta_solved, success, info = panda_ik(T_home, theta_init=PANDA_HOME_THETA)
    
    print("HOME target IK test (should converge immediately)")
    print("  Iterations = {}".format(info['iters']))
    print("  Success = {}".format(success))
    
    assert success
    assert info['iters'] <= 3, "HOME target should not take many iterations"
    print("OK test_ik_home_target passed")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Panda IK Test")
    print("=" * 60)
    print()
    
    test_ik_home_target()
    test_ik_random_targets()
    
    print("=" * 60)
    print("All IK tests passed!")
    print("=" * 60)