import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=23, act_dim=7, hidden=(256, 256)):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, act_dim)]   # 线性输出，不再用 Tanh
        self.net = nn.Sequential(*layers)

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        x = (x - self.obs_mean) / self.obs_std
        return self.net(x)

    @torch.no_grad()
    def predict(self, obs):
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = self.forward(x).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0)   # 推理端裁剪，等价于专家的 clip