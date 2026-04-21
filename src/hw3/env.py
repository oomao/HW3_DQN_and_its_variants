"""Thin numpy/torch-friendly wrapper around the vendored Gridworld."""
from __future__ import annotations

import numpy as np

from .gridworld_vendored import Gridworld

ACTION_CHARS = ("u", "d", "l", "r")


class GridworldEnv:
    def __init__(self, mode: str = "static", size: int = 4, max_moves: int = 50, seed: int | None = None):
        self.mode = mode
        self.size = size
        self.max_moves = max_moves
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
        self.game: Gridworld | None = None
        self.moves = 0

    def reset(self) -> np.ndarray:
        self.game = Gridworld(size=self.size, mode=self.mode)
        self.moves = 0
        return self._state()

    def _state(self) -> np.ndarray:
        board = self.game.board.render_np().reshape(1, -1).astype(np.float32)
        board += self._rng.random(board.shape, dtype=np.float32) / 100.0
        return board

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.game.makeMove(ACTION_CHARS[action])
        self.moves += 1
        reward = self.game.reward()
        done = reward != 0 or self.moves >= self.max_moves
        info = {"won": reward > 0, "lost": reward < 0, "truncated": reward == 0 and done}
        return self._state(), float(reward), done, info

    def piece_positions(self) -> dict[str, tuple[int, int]]:
        c = self.game.board.components
        return {k: c[k].pos for k in ("Player", "Goal", "Pit", "Wall")}


def observation_size(size: int = 4) -> int:
    return 4 * size * size
