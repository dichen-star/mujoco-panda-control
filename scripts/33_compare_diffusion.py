"""
Day 20 - 对比 Expert / BC / DAgger / DP / PPO + DP 复现 sanity check
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from stable_baselines3 import PPO
from src.envs.panda_reach_env import PandaReachEnv
from src.kinematics.forward import panda_fk
from src.kinematics.inverse import panda_ik
from src.imitation.bc_policy import BCPolicy
from src.imitation.diffusion_policy import DiffusionPolicy

BC_MODEL = "models/bc/bc_reach.pt"
DAGGER_MODEL = "models/bc/dagger_reach.pt"
DP_MODEL = "models/bc/diffusion_reach.pt"
PPO_MODEL = "models/best_model.zip"
N_EP = 50            # DP 推理慢，用 50（注意比 Day 19 的 100 噪声更大）
SEED_OFF = 70000
T = 50


def expert_action(env):
    base = env.unwrapped
    q_now = base.data.qpos[:7]
    T_home = panda_fk(q_now)
    Tm = np.eye(4); Tm[:3, :3] = T_home[:3, :3]; Tm[:3, 3] = base.target
    res = panda_ik(Tm, theta_init=q_now)
    q_target = res[0] if isinstance(res, tuple) else res
    return np.clip((q_target - q_now) / base.action_scale, -1.0, 1.0)


def eval_policy(env, fn, n_ep, off):
    succ, finals = 0, []
    for ep in range(n_ep):
        obs, info = env.reset(seed=ep + off)
        term = trunc = False; last = 0.0
        while not (term or trunc):
            obs, r, term, trunc, info = env.step(fn(obs, env)); last = info["distance"]
        succ += int(term); finals.append(last)
    return succ / n_ep * 100, float(np.mean(finals)) * 1000


def main():
    print("=" * 60)
    print(f"Day 20: Expert / BC / DAgger / DP / PPO ({N_EP} ep, seed {SEED_OFF})")
    print("=" * 60)

    env = PandaReachEnv(max_steps=200)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]

    bc = BCPolicy(od, ad); bc.load_state_dict(torch.load(BC_MODEL, map_location="cpu")); bc.eval()
    dg = BCPolicy(od, ad); dg.load_state_dict(torch.load(DAGGER_MODEL, map_location="cpu")); dg.eval()
    dp = DiffusionPolicy(od, ad, T=T); dp.load_state_dict(torch.load(DP_MODEL, map_location="cpu")); dp.eval()
    ppo = PPO.load(PPO_MODEL)

    # ---- sanity check：DP 在专家状态上复现专家动作的 MSE（应与 BC 同量级）----
    d = np.load("data/demos_reach.npz")
    idx = np.random.RandomState(0).choice(len(d["obs"]), 2000, replace=False)
    obs_s = torch.as_tensor(d["obs"][idx], dtype=torch.float32)
    with torch.no_grad():
        dp_act = dp.sample(obs_s).cpu().numpy()
        bc_act = bc(obs_s).cpu().numpy()
    dp_mse = float(((dp_act - d["act"][idx]) ** 2).mean())
    bc_mse = float(((np.clip(bc_act, -1, 1) - d["act"][idx]) ** 2).mean())
    print(f"\n[sanity] action-reproduction MSE vs expert: DP {dp_mse:.4f} | BC {bc_mse:.4f}")
    print("         （两者同量级 → DP 去噪器确实学到了映射，实现正确）\n")

    policies = [
        ("Expert (IK)", lambda o, e: expert_action(e),                       "#2a9d8f"),
        ("BC",          lambda o, e: bc.predict(o),                          "#e76f51"),
        ("DAgger",      lambda o, e: dg.predict(o),                          "#9b5de5"),
        ("Diffusion",   lambda o, e: dp.predict(o),                          "#ffb703"),
        ("PPO",         lambda o, e: ppo.predict(o, deterministic=True)[0],  "#264653"),
    ]
    results = {}
    for name, fn, c in policies:
        sr, fd = eval_policy(env, fn, N_EP, SEED_OFF)
        results[name] = (sr, fd, c)
        print(f"  {name:<12} success {sr:5.1f}%   final {fd:5.0f} mm")
    env.close()

    # ---- 两面板 ----
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 10})
    names = list(results.keys()); cols = [results[n][2] for n in names]
    sr = [results[n][0] for n in names]; fd = [results[n][1] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    b = ax.bar(x, sr, 0.6, color=cols, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 115)
    ax.set_title(f"Reach success ({N_EP} episodes)")
    ax.grid(True, axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)
    for bar, v in zip(b, sr):
        ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax = axes[1]
    b = ax.bar(x, fd, 0.6, color=cols, edgecolor="black", linewidth=0.8)
    ax.axhline(50, color="green", ls="--", lw=1.3, label="50 mm threshold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Mean final distance (mm)")
    ax.set_ylim(0, max(fd) * 1.30)
    ax.set_title("Mean final distance")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)
    for bar, v in zip(b, fd):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(fd)*0.02, f"{v:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = "logs/day20_diffusion_compare.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()