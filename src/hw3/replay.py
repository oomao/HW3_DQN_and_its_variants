"""經驗回放緩衝區（deque 實作）。"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, rng: random.Random | None = None):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng or random.Random()

    def push(self, s, a, r, s_next, done) -> None:
        self.buffer.append(Transition(s, a, r, s_next, done))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = self._rng.sample(list(self.buffer), batch_size)
        s = torch.from_numpy(np.concatenate([t.s for t in batch], axis=0)).float()
        s_next = torch.from_numpy(np.concatenate([t.s_next for t in batch], axis=0)).float()
        a = torch.tensor([t.a for t in batch], dtype=torch.long)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return s, a, r, s_next, done
