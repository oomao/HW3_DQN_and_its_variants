"""HW3-3：PyTorch Lightning 實作 Dueling Double DQN + 訓練穩定性技巧。

訓練技巧：
- 梯度裁剪 (gradient clipping)：Lightning `gradient_clip_val=1.0`
- Cosine learning-rate annealing
- Target network soft update（τ=0.005），每步更新
- ε 指數衰減 + warm-up buffer
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
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from .env import GridworldEnv
from .models import DuelingDQN, hard_update, soft_update
from .replay import ReplayBuffer
from .viz import plot_curves, plot_winrate, render_rollout_gif

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CKPT = ARTIFACTS / "checkpoints"


class _ExperienceStream(IterableDataset):
    """每個 training step 都從 buffer 抽一個 batch（buffer 永遠非空，因為預先 warmup）。"""

    def __init__(self, agent: "DQNLightning"):
        self.agent = agent

    def __iter__(self):
        while True:
            self.agent._play_step()
            yield self.agent.buffer.sample(self.agent.hparams.batch_size)


def _collate_passthrough(batch):
    return batch[0]


class DQNLightning(pl.LightningModule):
    def __init__(
        self,
        mode: str = "random",
        capacity: int = 10000,
        batch_size: int = 128,
        warmup_steps: int = 500,
        gamma: float = 0.9,
        lr: float = 1e-3,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 3000,
        total_steps: int = 6000,
        seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.online = DuelingDQN()
        self.target = copy.deepcopy(self.online)
        hard_update(self.target, self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.env = GridworldEnv(mode=mode, seed=seed)
        self.state = self.env.reset()
        self.buffer = ReplayBuffer(capacity, rng=random.Random(seed))
        self.loss_fn = nn.MSELoss()

        self.total_env_steps = 0
        self.episode_reward = 0.0
        self.episode_rewards: list[float] = []

    def prefill(self) -> None:
        """預熱 buffer，避免 dataloader 一開始拿到空 batch。"""
        while len(self.buffer) < self.hparams.warmup_steps:
            self._play_step()

    def _epsilon(self) -> float:
        k = min(1.0, self.total_env_steps / self.hparams.eps_decay_steps)
        return self.hparams.eps_end + (self.hparams.eps_start - self.hparams.eps_end) * math.exp(-3 * k)

    @torch.no_grad()
    def _play_step(self) -> None:
        eps = self._epsilon()
        if random.random() < eps:
            action = random.randint(0, 3)
        else:
            st = torch.from_numpy(self.state).float().to(self.device)
            action = int(torch.argmax(self.online(st), dim=1).item())
        s_next, r, done, _ = self.env.step(action)
        self.buffer.push(self.state, action, r, s_next, done)
        self.state = s_next
        self.episode_reward += r
        self.total_env_steps += 1
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
            self.state = self.env.reset()

    def training_step(self, batch, batch_idx):
        s, a, r, s_next, done = batch
        s = s.to(self.device)
        s_next = s_next.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        done = done.to(self.device)
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(s_next).argmax(dim=1, keepdim=True)
            q_next = self.target(s_next).gather(1, next_actions).squeeze(1)
            target = r + self.hparams.gamma * q_next * (1.0 - done)
        loss = self.loss_fn(q_pred, target)
        soft_update(self.target, self.online, self.hparams.tau)

        self.log("loss", loss, prog_bar=True)
        self.log("eps", self._epsilon(), prog_bar=True)
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
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="random")
    parser.add_argument("--accelerator", default="auto", help="auto / cpu / gpu / cuda")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    model = DQNLightning(mode=args.mode, total_steps=args.steps, seed=args.seed)
    model.prefill()
    print(
        f"[lightning] buffer warmed ({len(model.buffer)} transitions). "
        f"starting Trainer for {args.steps} steps on {args.accelerator}..."
    )

    trainer = pl.Trainer(
        max_steps=args.steps,
        accelerator=args.accelerator,
        devices=1 if args.accelerator != "cpu" else None,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        log_every_n_steps=100,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    rewards = np.array(model.episode_rewards, dtype=np.float32)
    np.save(ARTIFACTS / "lightning_rewards.npy", rewards)
    torch.save(model.online.state_dict(), CKPT / "lightning_random.pt")

    plot_curves(
        {"Dueling Double DQN (Lightning)": rewards},
        f"HW3-3 · Gridworld {args.mode} mode — 學習曲線",
        ARTIFACTS / "lightning_rewards.png",
        smooth=max(20, rewards.size // 30),
        colors={"Dueling Double DQN (Lightning)": "#a78bfa"},
    )
    plot_winrate(
        {"Dueling Double DQN (Lightning)": rewards},
        f"HW3-3 · Gridworld {args.mode} mode — 勝率",
        ARTIFACTS / "lightning_winrate.png",
        window=max(50, rewards.size // 10),
        colors={"Dueling Double DQN (Lightning)": "#a78bfa"},
    )

    net = model.online
    net.eval()

    def act(state):
        with torch.no_grad():
            return int(torch.argmax(net(torch.from_numpy(state).float()), dim=1).item())

    res = render_rollout_gif(
        GridworldEnv(mode=args.mode, seed=777),
        act,
        ARTIFACTS / "rollout_lightning_random.gif",
        title="Dueling Double DQN · random",
        max_steps=40,
    )
    final_window = min(300, rewards.size)
    wr = float(np.mean(rewards[-final_window:] > 0)) if rewards.size else 0.0
    print(f"[lightning] episodes={rewards.size}, win_rate_last_{final_window}={wr:.3f}, rollout={res}")


if __name__ == "__main__":
    main()
