"""DQN 與 Dueling DQN 的 PyTorch 模型。"""
from __future__ import annotations

import math

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


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy layer（Fortunato et al., 2017）。

    參數: μ + σ ⊙ ε，其中 ε 為 factorised noise（行/列各取一次再外積）。
    每次 forward 都 resample ε（train 時），eval 時用 μ。
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        # σ 起始值 = sigma_init / sqrt(in_features)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device=None) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, device=self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return torch.nn.functional.linear(x, w, b)


class RainbowDQN(nn.Module):
    """Dueling 架構 + NoisyLinear 取代後段的 Linear。

    Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))
    """

    def __init__(self, obs_dim: int = 64, n_actions: int = 4, hidden=(150, 100), sigma_init: float = 0.17):
        super().__init__()
        h1, h2 = hidden
        # 前段 trunk 仍用普通 Linear（避免 noise 過早干擾特徵抽取）
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # value / advantage 兩個 head 用 NoisyLinear，把探索性放進輸出層
        # sigma_init 從 paper 預設 0.5 調到 0.17（Categorical DQN paper 對 toy env 的設定）。
        self.value_head = NoisyLinear(h2, 1, sigma_init=sigma_init)
        self.adv_head = NoisyLinear(h2, n_actions, sigma_init=sigma_init)

    def reset_noise(self) -> None:
        self.value_head.reset_noise()
        self.adv_head.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class CategoricalRainbowDQN(nn.Module):
    """C51（Distributional） + Dueling + 可選 NoisyLinear。

    Bellemare et al., *A Distributional Perspective on RL*, ICML 2017.

    輸出: 對每個 action 預測 N 個 atom 的 logits，softmax 後成為機率分布。
    Q(s, a) = Σ_i z_i · p(z_i | s, a)，z_i 為固定 support。
    """

    def __init__(
        self,
        obs_dim: int = 64,
        n_actions: int = 4,
        hidden=(150, 100),
        n_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 2.0,
        use_noisy: bool = False,
        sigma_init: float = 0.17,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.use_noisy = use_noisy
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))

        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        if use_noisy:
            self.value_head = NoisyLinear(h2, n_atoms, sigma_init=sigma_init)
            self.adv_head = NoisyLinear(h2, n_actions * n_atoms, sigma_init=sigma_init)
        else:
            self.value_head = nn.Linear(h2, n_atoms)
            self.adv_head = nn.Linear(h2, n_actions * n_atoms)

    def reset_noise(self) -> None:
        if self.use_noisy:
            self.value_head.reset_noise()
            self.adv_head.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """回傳每個 (state, action) 的 atom logits，shape = (B, n_actions, n_atoms)。"""
        h = self.trunk(x)
        v = self.value_head(h).view(-1, 1, self.n_atoms)  # (B, 1, N)
        a = self.adv_head(h).view(-1, self.n_actions, self.n_atoms)  # (B, A, N)
        # Dueling 合併: Q_logits(s, a) = V(s) + A(s, a) − mean_a A(s, a)
        return v + (a - a.mean(dim=1, keepdim=True))

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """回傳機率分布，shape = (B, A, N)。"""
        logits = self.forward(x)
        # 對 atom 維度做 softmax；clamp 防止極端值
        return torch.softmax(logits, dim=-1).clamp_min(1e-8)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Σ_i z_i · p_i，shape = (B, A)。"""
        p = self.dist(x)
        return (p * self.atoms).sum(dim=-1)


def project_distribution(
    next_dist: torch.Tensor,  # (B, N)  target net 的 p(s', a*)
    rewards: torch.Tensor,    # (B,)
    dones: torch.Tensor,      # (B,)
    gamma_n: torch.Tensor,    # (B,)  每筆樣本實際的 γ^n
    atoms: torch.Tensor,      # (N,)
    v_min: float,
    v_max: float,
    n_atoms: int,
) -> torch.Tensor:
    """C51 projection step (對應論文 Algorithm 1)。

    target 分布 = 將 atom 經 Bellman 操作 (r + γ·z_i)·(1−done) clamp 後，
    依距離最近的兩個 atom 線性內插回原 support。
    回傳 m，shape = (B, N)。
    """
    B = rewards.size(0)
    N = n_atoms
    delta_z = (v_max - v_min) / (N - 1)

    # τ_i = clamp(r + γ^n · z_i · (1 − done), v_min, v_max)
    tz = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * atoms.unsqueeze(0) * (1.0 - dones.unsqueeze(1))
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / delta_z                  # (B, N)，fractional bin
    l = b.floor().long().clamp(0, N - 1)        # (B, N)
    u = b.ceil().long().clamp(0, N - 1)
    eq_mask = (l == u).float()                  # b 剛好是整數的格子，避免 mass 消失

    m = torch.zeros(B, N, dtype=next_dist.dtype, device=next_dist.device)
    # m[l] += p · (u + (l==u) − b)；m[u] += p · (b − l)
    m.scatter_add_(1, l, next_dist * (u.float() + eq_mask - b))
    m.scatter_add_(1, u, next_dist * (b - l.float()))
    return m


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
