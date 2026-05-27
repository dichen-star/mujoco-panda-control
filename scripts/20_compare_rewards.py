"""
Day 15 - 对比 sparse / sparse+curriculum / dense 三个实验
========================================================
读取三个实验的 EvalCallback 日志,画对比曲线
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

LOG_ROOT = "logs/day15"

EXPERIMENTS = [
    ("A_sparse_no_curriculum", "Sparse (baseline)",     "#888888", "--"),
    ("B_sparse_curriculum",    "Sparse + Curriculum",   "#1f77b4", "-"),
    ("C_dense_no_curriculum",  "Dense (Day 14 ceiling)","#d62728", "-"),
]


def load_eval(exp_name):
    """EvalCallback 把数据存到 log_path/evaluations.npz"""
    path = os.path.join(LOG_ROOT, exp_name, "evaluations.npz")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping")
        return None
    data = np.load(path)
    timesteps = data["timesteps"]
    # data['results'] shape: (n_evals, n_eval_episodes)
    results = data["results"]
    successes = data["successes"] if "successes" in data.files else None
    return {
        "timesteps": timesteps,
        "rewards": results.mean(axis=1),
        "reward_std": results.std(axis=1),
        "success_rate": successes.mean(axis=1) if successes is not None else None,
    }


def main():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ============ 左:成功率曲线 ============
    ax = axes[0]
    for name, label, color, ls in EXPERIMENTS:
        d = load_eval(name)
        if d is None or d["success_rate"] is None:
            continue
        ax.plot(d["timesteps"] / 1000, d["success_rate"] * 100,
                label=label, color=color, linestyle=ls, lw=2)
    # 标注 curriculum 升级时刻
    for boundary in [50, 150]:
        ax.axvline(boundary, color="black", alpha=0.2, ls=":", lw=1)
    ax.text(25, 5, "Stage 1\n(small)", ha="center", fontsize=8, alpha=0.6)
    ax.text(100, 5, "Stage 2\n(medium)", ha="center", fontsize=8, alpha=0.6)
    ax.text(225, 5, "Stage 3\n(full)", ha="center", fontsize=8, alpha=0.6)

    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Eval success rate (%)")
    ax.set_title("Success rate over training")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # ============ 右:reward 曲线 ============
    ax = axes[1]
    for name, label, color, ls in EXPERIMENTS:
        d = load_eval(name)
        if d is None:
            continue
        ax.plot(d["timesteps"] / 1000, d["rewards"],
                label=label, color=color, linestyle=ls, lw=2)
        ax.fill_between(d["timesteps"] / 1000,
                        d["rewards"] - d["reward_std"],
                        d["rewards"] + d["reward_std"],
                        color=color, alpha=0.12)
    for boundary in [50, 150]:
        ax.axvline(boundary, color="black", alpha=0.2, ls=":", lw=1)

    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Eval mean reward")
    ax.set_title("Eval reward over training (±1 std)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = "logs/day15_reward_engineering.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Figure saved: {out}")

    # ============ 终末成绩单 ============
    print("\nFinal eval results (last evaluation):")
    print("-" * 60)
    print(f"{'Experiment':<28} {'Success':<10} {'Mean reward':<15}")
    print("-" * 60)
    for name, label, _, _ in EXPERIMENTS:
        d = load_eval(name)
        if d is None:
            continue
        sr = d["success_rate"][-1] * 100 if d["success_rate"] is not None else float("nan")
        mr = d["rewards"][-1]
        print(f"{label:<28} {sr:<10.1f} {mr:<15.2f}")
    print("-" * 60)


if __name__ == "__main__":
    main()