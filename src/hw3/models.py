"""DQN 與 Dueling DQN 的 PyTorch 模型。"""
from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """教科書 Listing 3.2 的基線網路。"""

    def __init__(self, obs_dim: int = 64, n_actions: int = 4, hidden=(150, 100)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQN(nn.Module):
    """Value / Advantage 分支的 Dueling 架構。

    Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))
    """

    def __init__(self, obs_dim: int = 64, n_actions: int = 4, hidden=(150, 100)):
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(h2, 1)
        self.adv_head = nn.Linear(h2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
