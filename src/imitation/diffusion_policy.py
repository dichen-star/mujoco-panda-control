"""
DiffusionPolicy: 条件 DDPM 动作生成器（从零实现）
=====================================================
ε-prediction + 线性噪声调度 + 正弦时间步嵌入 + 观测条件。
单步单动作（不做 action chunking）。
"""
import math
import torch
import torch.nn as nn


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=23, act_dim=7, t_dim=32, hidden=256, T=50):
        super().__init__()
        self.obs_dim, self.act_dim, self.t_dim, self.T = obs_dim, act_dim, t_dim, T

        # 观测归一化（随 state_dict 保存）
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        # 线性噪声调度
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

        # 时间步嵌入 MLP
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
        # 去噪网络 ε_θ(a_t, s, t)
        in_dim = act_dim + obs_dim + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def _temb(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def eps(self, a_t, obs, t):
        """预测噪声。a_t:(B,act)  obs:(B,obs)  t:(B,) long"""
        obs_n = (obs - self.obs_mean) / self.obs_std
        temb = self.t_mlp(self._temb(t))
        return self.net(torch.cat([a_t, obs_n, temb], dim=-1))

    @torch.no_grad()
    def sample(self, obs):
        """DDPM 祖先采样：obs (B,obs) -> action (B,act)，clip 到 [-1,1]"""
        device = next(self.parameters()).device
        B = obs.shape[0]
        a = torch.randn(B, self.act_dim, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self.eps(a, obs, t)
            beta, alpha, abar = self.betas[i], self.alphas[i], self.abar[i]
            mean = (a - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
            a = mean + (torch.sqrt(beta) * torch.randn_like(a) if i > 0 else 0.0)
        return a.clamp(-1.0, 1.0)

    @torch.no_grad()
    def predict(self, obs):
        """单条 obs(np) -> 单条 action(np)"""
        self.eval()
        device = next(self.parameters()).device
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return self.sample(o).squeeze(0).cpu().numpy()