"""HW3-4（加分題）：Rainbow DQN 解 Gridworld random mode。

組合的改進（在 HW3-3 Dueling Double DQN 基礎上再加 3 樣，最後再加 C51）：
- ✅ Double DQN（HW3-2/3）
- ✅ Dueling DQN（HW3-2/3）
- 🌟 Prioritized Experience Replay（PER, α=0.6, β 0.4→1.0）
- 🌟 Multi-step / n-step return（n=3）
- 🌟 Distributional RL — C51（51 atoms over [v_min, v_max]）
- 🟡 Noisy Networks（可選；用 `--noisy` 開啟，否則退回 ε-greedy）

預設配置 = 5/6 Rainbow（不含 NoisyNet，因為實測在此 toy env 反而變差）。
加上 `--noisy` 後即為 6/6（含 ablation）；`--no-distributional` 可退回 4/6 Rainbow-Lite。
"""
from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from .env import GridworldEnv
from .models import (
    CategoricalRainbowDQN,
    DuelingDQN,
    RainbowDQN,
    hard_update,
    project_distribution,
    soft_update,
)
from .nstep import NStepBuffer
from .prioritized_replay import PrioritizedReplayBuffer
from .viz import plot_curves, plot_winrate, render_rollout_gif

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CKPT = ARTIFACTS / "checkpoints"


class _ExperienceStream(IterableDataset):
    def __init__(self, agent: "RainbowLightning"):
        self.agent = agent

    def __iter__(self):
        while True:
            self.agent._play_step()
            yield self.agent.buffer.sample(self.agent.hparams.batch_size, self.agent.total_env_steps)


def _collate_passthrough(batch):
    return batch[0]


class RainbowLightning(pl.LightningModule):
    def __init__(
        self,
        mode: str = "random",
        capacity: int = 10000,
        batch_size: int = 128,
        warmup_steps: int = 1000,
        gamma: float = 0.9,
        n_step: int = 3,
        lr: float = 1e-3,
        tau: float = 0.005,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        total_steps: int = 25000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 8000,
        use_noisy: bool = False,
        distributional: bool = True,
        n_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 2.0,
        seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if distributional:
            self.online = CategoricalRainbowDQN(
                n_atoms=n_atoms, v_min=v_min, v_max=v_max, use_noisy=use_noisy
            )
        elif use_noisy:
            self.online = RainbowDQN()
        else:
            self.online = DuelingDQN()
        self.target = copy.deepcopy(self.online)
        hard_update(self.target, self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.env = GridworldEnv(mode=mode, seed=seed)
        self.state = self.env.reset()
        self.nstep = NStepBuffer(n=n_step, gamma=gamma)
        self.buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_steps=total_steps,
            rng=random.Random(seed),
        )

        self.total_env_steps = 0
        self.episode_reward = 0.0
        self.episode_rewards: list[float] = []

    def prefill(self) -> None:
        """預熱 buffer：warmup 期完全 uniform 探索，保證初始多樣性。"""
        while len(self.buffer) < self.hparams.warmup_steps:
            self._play_step(force_random=True)

    def _epsilon(self) -> float:
        if self.hparams.use_noisy:
            return 0.0
        k = min(1.0, self.total_env_steps / self.hparams.eps_decay_steps)
        return self.hparams.eps_end + (self.hparams.eps_start - self.hparams.eps_end) * math.exp(-3 * k)

    def _greedy_action(self) -> int:
        """純 greedy（依當前 Q 值挑最大），呼叫前確保已 reset_noise 與 to(device)。"""
        st = torch.from_numpy(self.state).float().to(self.device)
        if self.hparams.distributional:
            q = self.online.q_values(st)  # (1, A)
        else:
            q = self.online(st)
        return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def _play_step(self, force_random: bool = False) -> None:
        if force_random or (not self.hparams.use_noisy and random.random() < self._epsilon()):
            action = random.randint(0, 3)
        else:
            if self.hparams.use_noisy:
                self.online.reset_noise()
            action = self._greedy_action()

        s_next, r, done, _ = self.env.step(action)
        for tr in self.nstep.push(self.state, action, r, s_next, done):
            self.buffer.push(tr)
        self.state = s_next
        self.episode_reward += r
        self.total_env_steps += 1
        if done:
            self.nstep.reset()
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
            self.state = self.env.reset()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _step_distributional(self, s, a, r, s_next, done, n, w):
        """C51 + Double + n-step + PER 的 training step。"""
        B = s.size(0)
        N = self.hparams.n_atoms
        atoms = self.online.atoms

        # 預測 log p(s, a)
        log_p_all = torch.log_softmax(self.online(s), dim=-1)  # (B, A, N)
        a_exp = a.view(B, 1, 1).expand(B, 1, N)
        log_p_a = log_p_all.gather(1, a_exp).squeeze(1)  # (B, N)

        with torch.no_grad():
            if self.hparams.use_noisy:
                self.target.eval()
            # Double DQN: 用 online 選 next action（依 Q 期望值）
            next_q = self.online.q_values(s_next)  # (B, A)
            a_star = next_q.argmax(dim=1)
            # target 算分布
            next_dist = self.target.dist(s_next)  # (B, A, N)
            next_dist_a = next_dist.gather(
                1, a_star.view(B, 1, 1).expand(B, 1, N)
            ).squeeze(1)  # (B, N)

            gamma_n = self.hparams.gamma ** n  # (B,)
            m = project_distribution(
                next_dist=next_dist_a,
                rewards=r,
                dones=done,
                gamma_n=gamma_n,
                atoms=atoms,
                v_min=self.hparams.v_min,
                v_max=self.hparams.v_max,
                n_atoms=N,
            )  # (B, N)

        # 交叉熵 per-sample → PER 用作 priority
        ce = -(m * log_p_a).sum(dim=1)  # (B,)
        loss = (w * ce).mean()
        return loss, ce.detach()

    def _step_qlearning(self, s, a, r, s_next, done, n, w):
        """非 distributional path（4/6 或 + NoisyNet 5/6）。"""
        if self.hparams.use_noisy:
            self.target.eval()
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(s_next).argmax(dim=1, keepdim=True)
            q_next = self.target(s_next).gather(1, next_actions).squeeze(1)
            gamma_n = self.hparams.gamma ** n
            target = r + gamma_n * q_next * (1.0 - done)
        td_error = target - q_pred
        loss = (w * torch.nn.functional.smooth_l1_loss(q_pred, target, reduction="none")).mean()
        return loss, td_error.detach()

    def training_step(self, batch, batch_idx):
        s, a, r, s_next, done, n, idxs, w = batch
        s = s.to(self.device); s_next = s_next.to(self.device)
        a = a.to(self.device); r = r.to(self.device)
        done = done.to(self.device); n = n.to(self.device); w = w.to(self.device)

        if self.hparams.distributional:
            loss, priority_signal = self._step_distributional(s, a, r, s_next, done, n, w)
        else:
            loss, priority_signal = self._step_qlearning(s, a, r, s_next, done, n, w)

        self.buffer.update_priorities(idxs, priority_signal.cpu().numpy())
        soft_update(self.target, self.online, self.hparams.tau)

        self.log("loss", loss, prog_bar=True)
        if self.episode_rewards:
            self.log("win200", float(np.mean(np.array(self.episode_rewards[-200:]) > 0)), prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.online.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.total_steps, eta_min=1e-5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(
            _ExperienceStream(self), batch_size=1, num_workers=0, collate_fn=_collate_passthrough
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="random")
    parser.add_argument("--accelerator", default="auto", help="auto / cpu / gpu / cuda")
    parser.add_argument("--noisy", action="store_true", help="enable NoisyNet (6/6 Rainbow)")
    parser.add_argument("--no-distributional", action="store_true", help="disable C51 (= 4/6 Rainbow-Lite)")
    parser.add_argument("--n-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-2.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--tag", default="rainbow", help="artifact 檔名 prefix（用於 ablation 區分）")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    distributional = not args.no_distributional
    n_components = 4 + int(distributional) + int(args.noisy)

    model = RainbowLightning(
        mode=args.mode,
        total_steps=args.steps,
        use_noisy=args.noisy,
        distributional=distributional,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        seed=args.seed,
    )
    print(
        f"[rainbow] config: distributional={distributional}, noisy={args.noisy}, "
        f"Rainbow components = {n_components}/6, tag={args.tag}"
    )
    model.prefill()
    print(
        f"[rainbow] buffer warmed ({len(model.buffer)} transitions). "
        f"starting Trainer for {args.steps} steps on {args.accelerator}..."
    )

    trainer = pl.Trainer(
        max_steps=args.steps,
        accelerator=args.accelerator,
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=100,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    rewards = np.array(model.episode_rewards, dtype=np.float32)
    np.save(ARTIFACTS / f"{args.tag}_rewards.npy", rewards)
    torch.save(model.online.state_dict(), CKPT / f"{args.tag}_random.pt")

    label = f"Rainbow ({n_components}/6)"
    plot_curves(
        {label: rewards},
        f"HW3-4 · Gridworld {args.mode} mode — 學習曲線",
        ARTIFACTS / f"{args.tag}_rewards.png",
        smooth=max(20, rewards.size // 30),
        colors={label: "#f43f5e"},
    )
    plot_winrate(
        {label: rewards},
        f"HW3-4 · Gridworld {args.mode} mode — 勝率",
        ARTIFACTS / f"{args.tag}_winrate.png",
        window=max(50, rewards.size // 10),
        colors={label: "#f43f5e"},
    )

    net = model.online
    net.eval()

    if distributional:
        def act(state):
            with torch.no_grad():
                st = torch.from_numpy(state).float()
                q = net.q_values(st)
                return int(torch.argmax(q, dim=1).item())
    else:
        def act(state):
            with torch.no_grad():
                return int(torch.argmax(net(torch.from_numpy(state).float()), dim=1).item())

    # 用 seed=99 重現「需要實際導航 7 步、最後吃到 goal」的 rollout（seed=777 玩家
    # 剛好就在 goal 隔壁，1 步結束、不太能展示策略）。
    res = render_rollout_gif(
        GridworldEnv(mode=args.mode, seed=99),
        act,
        ARTIFACTS / f"rollout_{args.tag}_random.gif",
        title=label + " · random",
        max_steps=40,
    )
    final_window = min(300, rewards.size)
    wr = float(np.mean(rewards[-final_window:] > 0)) if rewards.size else 0.0
    print(f"[rainbow] episodes={rewards.size}, win_rate_last_{final_window}={wr:.3f}, rollout={res}")


if __name__ == "__main__":
    main()
