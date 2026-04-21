"""共用的繪圖與 GIF 產生工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams

from .env import GridworldEnv

rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "PingFang TC",
    "Noto Sans CJK TC",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = np.full(w - 1, x[0])
    padded = np.concatenate([pad, x])
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def plot_curves(
    curves: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    ylabel: str = "Reward per episode",
    smooth: int = 50,
    colors: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, arr in curves.items():
        y = moving_average(arr, smooth)
        kwargs = {"lw": 1.6, "label": label}
        if colors and label in colors:
            kwargs["color"] = colors[label]
        ax.plot(np.arange(1, y.size + 1), y, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_winrate(
    curves: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    window: int = 200,
    colors: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, arr in curves.items():
        y = moving_average((arr > 0).astype(np.float32), window)
        kwargs = {"lw": 1.6, "label": label}
        if colors and label in colors:
            kwargs["color"] = colors[label]
        ax.plot(np.arange(1, y.size + 1), y, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Episodes")
    ax.set_ylabel(f"Win rate (rolling {window})")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


_ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def _draw_frame(ax, size, positions, title, hint=None):
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(size):
        for c in range(size):
            ax.add_patch(
                patches.Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor="#f8fafc",
                    edgecolor="#334155",
                )
            )
    # Wall
    wr, wc = positions["Wall"]
    ax.add_patch(patches.Rectangle((wc - 0.5, wr - 0.5), 1, 1, facecolor="#475569", edgecolor="#1e293b"))
    ax.text(wc, wr, "W", ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    # Pit
    pr, pc = positions["Pit"]
    ax.add_patch(patches.Rectangle((pc - 0.5, pr - 0.5), 1, 1, facecolor="#b91c1c", edgecolor="#7f1d1d"))
    ax.text(pc, pr, "X", ha="center", va="center", color="white", fontsize=20, fontweight="bold")
    # Goal
    gr, gc = positions["Goal"]
    ax.add_patch(patches.Rectangle((gc - 0.5, gr - 0.5), 1, 1, facecolor="#15803d", edgecolor="#14532d"))
    ax.text(gc, gr, "+", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
    # Player
    plr, plc = positions["Player"]
    ax.add_patch(patches.Circle((plc, plr), 0.30, facecolor="#f59e0b", edgecolor="#78350f", zorder=3))
    ax.set_title(title, fontsize=12)
    if hint:
        ax.text(size - 1, -0.9, hint, ha="right", va="center", fontsize=10, color="#475569")


def render_rollout_gif(
    env: GridworldEnv,
    act_fn,
    out_path: Path,
    max_steps: int = 30,
    title: str = "rollout",
) -> dict:
    """Record a greedy rollout using act_fn(state)-> action, returning result dict."""
    state = env.reset()
    positions = [env.piece_positions()]
    actions = []
    total_r = 0.0
    for _ in range(max_steps):
        a = int(act_fn(state))
        state, r, done, info = env.step(a)
        actions.append(a)
        positions.append(env.piece_positions())
        total_r += r
        if done:
            break
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(exist_ok=True)
    frames = []
    size = env.size
    for i, pos in enumerate(positions):
        fig, ax = plt.subplots(figsize=(4, 4))
        if i == 0:
            hint = "start"
        elif i == len(positions) - 1:
            hint = ("win" if info.get("won") else "lose" if info.get("lost") else "stop")
        else:
            hint = f"step {i}  action={_ACTION_ARROWS[actions[i-1]]}"
        _draw_frame(ax, size, pos, f"{title}  step {i}", hint)
        fig.tight_layout()
        path = tmp_dir / f"frame_{i:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        frames.append(imageio.imread(path))
    imageio.mimsave(out_path, frames, duration=0.45, loop=0)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()
    return {"reward": total_r, "steps": len(actions), "won": bool(info.get("won", False))}
