"""HW3-1：Naive DQN（有/無 Experience Replay）於 Gridworld static mode。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import GridworldEnv
from .models import DQN
from .replay import ReplayBuffer
from .viz import plot_curves, plot_winrate, render_rollout_gif

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CKPT = ARTIFACTS / "checkpoints"


def epsilon_schedule(ep: int, total: int, start: float = 1.0, end: float = 0.1) -> float:
    return max(end, start - (start - end) * (ep / total))


def train(
    episodes: int = 3000,
    replay: bool = True,
    capacity: int = 1000,
    batch_size: int = 200,
    gamma: float = 0.9,
    lr: float = 1e-3,
    seed: int = 0,
    mode: str = "static",
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GridworldEnv(mode=mode, seed=seed)
    model = DQN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buf = ReplayBuffer(capacity, rng=random.Random(seed)) if replay else None

    rewards = np.zeros(episodes, dtype=np.float32)
    losses: list[float] = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        eps = epsilon_schedule(ep, episodes)
        while not done:
            st = torch.from_numpy(s).float()
            with torch.no_grad():
                q = model(st)
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                a = int(torch.argmax(q, dim=1).item())
            s_next, r, done, _ = env.step(a)
            total_r += r

            if buf is not None:
                buf.push(s, a, r, s_next, done)
                if len(buf) >= batch_size:
                    bs, ba, br, bsn, bd = buf.sample(batch_size)
                    q_pred = model(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = model(bsn).max(1).values
                        target = br + gamma * q_next * (1 - bd)
                    loss = loss_fn(q_pred, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss))
            else:
                # Naive (no replay): single-step update.
                q_pred = model(st)[0, a]
                with torch.no_grad():
                    q_next = model(torch.from_numpy(s_next).float()).max().item()
                target = torch.tensor(r + gamma * q_next * (0 if done else 1))
                loss = loss_fn(q_pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))

            s = s_next
        rewards[ep] = total_r

    return {"rewards": rewards, "model": model, "losses": np.array(losses, dtype=np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    print("[naive] training WITHOUT replay...")
    naive = train(episodes=args.episodes, replay=False, seed=args.seed)
    print("[naive] training WITH replay...")
    with_replay = train(episodes=args.episodes, replay=True, seed=args.seed)

    torch.save(with_replay["model"].state_dict(), CKPT / "naive_static.pt")
    np.save(ARTIFACTS / "naive_rewards_no_replay.npy", naive["rewards"])
    np.save(ARTIFACTS / "naive_rewards_replay.npy", with_replay["rewards"])

    plot_curves(
        {
            "Naive DQN (無 replay)": naive["rewards"],
            "DQN + Experience Replay": with_replay["rewards"],
        },
        "HW3-1 · Gridworld static mode — 學習曲線",
        ARTIFACTS / "naive_rewards.png",
        smooth=100,
        colors={
            "Naive DQN (無 replay)": "#94a3b8",
            "DQN + Experience Replay": "#0ea5e9",
        },
    )
    plot_winrate(
        {
            "Naive DQN (無 replay)": naive["rewards"],
            "DQN + Experience Replay": with_replay["rewards"],
        },
        "HW3-1 · Gridworld static mode — 勝率",
        ARTIFACTS / "naive_winrate.png",
        window=200,
        colors={
            "Naive DQN (無 replay)": "#94a3b8",
            "DQN + Experience Replay": "#0ea5e9",
        },
    )

    # Rollout GIF using the trained (replay) model.
    model = with_replay["model"]
    model.eval()

    def act(state):
        with torch.no_grad():
            return int(torch.argmax(model(torch.from_numpy(state).float()), dim=1).item())

    result = render_rollout_gif(
        GridworldEnv(mode="static", seed=123),
        act,
        ARTIFACTS / "rollout_naive_static.gif",
        title="Naive DQN · static",
    )
    print(f"[naive] final win rate (last 200 eps, w/ replay): "
          f"{(with_replay['rewards'][-200:] > 0).mean():.3f}; rollout: {result}")


if __name__ == "__main__":
    main()
