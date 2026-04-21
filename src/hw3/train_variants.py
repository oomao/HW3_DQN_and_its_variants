"""HW3-2：Vanilla DQN / Double DQN / Dueling DQN 於 Gridworld player mode。"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .env import GridworldEnv
from .models import DQN, DuelingDQN, hard_update
from .replay import ReplayBuffer
from .viz import plot_curves, plot_winrate, render_rollout_gif

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CKPT = ARTIFACTS / "checkpoints"


def epsilon_schedule(ep: int, total: int, start: float = 1.0, end: float = 0.1) -> float:
    return max(end, start - (start - end) * (ep / total))


def train(
    variant: str,
    episodes: int,
    mode: str = "player",
    seed: int = 0,
    gamma: float = 0.9,
    lr: float = 1e-3,
    capacity: int = 10000,
    batch_size: int = 200,
    sync_every: int = 500,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GridworldEnv(mode=mode, seed=seed)
    if variant == "dueling":
        online = DuelingDQN()
    else:
        online = DQN()
    target = copy.deepcopy(online)
    hard_update(target, online)
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    buf = ReplayBuffer(capacity, rng=random.Random(seed))

    rewards = np.zeros(episodes, dtype=np.float32)
    step_count = 0
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        eps = epsilon_schedule(ep, episodes)
        while not done:
            st = torch.from_numpy(s).float()
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(online(st), dim=1).item())
            s_next, r, done, _ = env.step(a)
            buf.push(s, a, r, s_next, done)
            s = s_next
            total_r += r
            step_count += 1

            if len(buf) >= batch_size:
                bs, ba, br, bsn, bd = buf.sample(batch_size)
                q_pred = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if variant == "double" or variant == "dueling":
                        next_actions = online(bsn).argmax(dim=1, keepdim=True)
                        q_next = target(bsn).gather(1, next_actions).squeeze(1)
                    else:  # vanilla DQN using target network for stability
                        q_next = target(bsn).max(1).values
                    tgt = br + gamma * q_next * (1 - bd)
                loss = loss_fn(q_pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if step_count % sync_every == 0:
                hard_update(target, online)
        rewards[ep] = total_r

    return {"rewards": rewards, "model": online}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="player")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    out: dict[str, dict] = {}
    for name, variant in [
        ("Vanilla DQN", "vanilla"),
        ("Double DQN", "double"),
        ("Dueling DQN", "dueling"),
    ]:
        print(f"[variants] training {name} on {args.mode} mode...")
        out[name] = train(variant, episodes=args.episodes, mode=args.mode, seed=args.seed)

    for name, res in out.items():
        key = name.lower().replace(" ", "_")
        np.save(ARTIFACTS / f"variants_{key}_rewards.npy", res["rewards"])
        torch.save(res["model"].state_dict(), CKPT / f"variants_{key}.pt")

    colors = {
        "Vanilla DQN": "#94a3b8",
        "Double DQN": "#0ea5e9",
        "Dueling DQN": "#f59e0b",
    }
    plot_curves(
        {k: v["rewards"] for k, v in out.items()},
        f"HW3-2 · Gridworld {args.mode} mode — 學習曲線",
        ARTIFACTS / "variants_rewards.png",
        smooth=100,
        colors=colors,
    )
    plot_winrate(
        {k: v["rewards"] for k, v in out.items()},
        f"HW3-2 · Gridworld {args.mode} mode — 勝率",
        ARTIFACTS / "variants_winrate.png",
        window=200,
        colors=colors,
    )

    # Rollout GIF for Dueling DQN (usually the best)
    model = out["Dueling DQN"]["model"]
    model.eval()

    def act(state):
        with torch.no_grad():
            return int(torch.argmax(model(torch.from_numpy(state).float()), dim=1).item())

    res = render_rollout_gif(
        GridworldEnv(mode=args.mode, seed=321),
        act,
        ARTIFACTS / "rollout_dueling_player.gif",
        title="Dueling DQN · player",
    )
    for k, v in out.items():
        wr = (v["rewards"][-200:] > 0).mean()
        print(f"[variants] {k}: final win rate = {wr:.3f}")
    print(f"[variants] Dueling rollout: {res}")


if __name__ == "__main__":
    main()
