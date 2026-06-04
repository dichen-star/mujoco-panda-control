"""Day 21 - 在多模态示范上训练 chunked BC 与 chunked DP"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from src.imitation.bc_policy import BCPolicy
from src.imitation.diffusion_policy import DiffusionPolicy

DATA = "data/demos_obstacle.npz"
BC_OUT = "models/bc/bc_obstacle.pt"
DP_OUT = "models/bc/dp_obstacle.pt"
T = 50


def train_bc(X, Y, mean, std, device, epochs=200):
    n = len(X); idx = torch.randperm(n); nv = int(0.1 * n)
    vi, ti = idx[:nv], idx[nv:]
    loader = DataLoader(TensorDataset(X[ti], Y[ti]), batch_size=256, shuffle=True)
    Xv, Yv = X[vi].to(device), Y[vi].to(device)
    m = BCPolicy(obs_dim=X.shape[1], act_dim=Y.shape[1]); m.set_normalizer(mean, std); m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lf = torch.nn.MSELoss()
    best, best_state = 1e9, None
    for ep in range(epochs):
        m.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = lf(m(Xv), Yv).item()
        if v < best:
            best = v; best_state = {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, best


def train_dp(X, Y, mean, std, device, epochs=400):
    loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)
    m = DiffusionPolicy(obs_dim=X.shape[1], act_dim=Y.shape[1], T=T); m.set_normalizer(mean, std); m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        m.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); B = yb.shape[0]
            t = torch.randint(0, T, (B,), device=device); eps = torch.randn_like(yb)
            abar = m.abar[t].unsqueeze(-1)
            a_t = torch.sqrt(abar) * yb + torch.sqrt(1 - abar) * eps
            loss = ((m.eps(a_t, xb, t) - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * B
        if (ep + 1) % 50 == 0:
            print(f"  DP epoch {ep+1:3d}  denoise_mse {tot/len(X):.4f}")
    return m


def main():
    os.makedirs("models/bc", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(DATA); obs, act = d["obs"], d["act"]
    print(f"device={device}  pairs={len(obs)}  obs_dim={obs.shape[1]}  chunk_dim={act.shape[1]}")
    mean, std = obs.mean(0), obs.std(0) + 1e-6
    X = torch.as_tensor(obs, dtype=torch.float32); Y = torch.as_tensor(act, dtype=torch.float32)

    print("training chunked BC ...")
    bc, vbc = train_bc(X, Y, mean, std, device); torch.save(bc.state_dict(), BC_OUT)
    print(f"  BC val_mse {vbc:.4f}  saved {BC_OUT}")

    print("training chunked DP ...")
    dp = train_dp(X, Y, mean, std, device); torch.save(dp.state_dict(), DP_OUT)
    print(f"  saved {DP_OUT}")


if __name__ == "__main__":
    main()