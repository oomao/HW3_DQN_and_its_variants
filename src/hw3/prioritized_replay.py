"""Prioritized Experience Replay（PER）— sum-tree 實作。

Schaul et al., *Prioritized Experience Replay*, ICLR 2016.

- 每筆 transition 被抽到的機率 P(i) ∝ p_i^α，其中 p_i = |TD-error| + ε
- 損失加上 IS weight w_i = (N · P(i))^(-β)，β 從 0.4 線性 anneal 到 1.0
- 用 sum-tree（陣列實作）讓 sample / update 都 O(log N)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NStepTransition:
    s: np.ndarray
    a: int
    r: float          # discounted n-step return
    s_next: np.ndarray  # state at t+n
    done: bool        # whether episode ended within the n-step window
    n: int            # actual horizon used (may be < n if episode ended)


class _SumTree:
    """葉節點數固定為 capacity（2 的冪次以上即可，這裡直接用任意大小）。

    tree[0] 是根，tree[capacity-1 : 2*capacity-1] 是葉節點。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[NStepTransition | None] = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, idx: int, s: float) -> int:
        # iterative descent to a leaf whose cumulative sum covers s
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: NStepTransition) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, NStepTransition]:
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]  # type: ignore[return-value]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 25000,
        eps: float = 1e-6,
        rng: random.Random | None = None,
    ):
        self.capacity = capacity
        self.tree = _SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.eps = eps
        self._rng = rng or random.Random()
        self._max_priority = 1.0  # 新樣本用最大優先度寫入，保證至少被抽到一次

    def __len__(self) -> int:
        return self.tree.size

    def _beta(self, step: int) -> float:
        k = min(1.0, step / max(1, self.beta_steps))
        return self.beta_start + (self.beta_end - self.beta_start) * k

    def push(self, transition: NStepTransition) -> None:
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, step: int):
        assert self.tree.size >= batch_size, "buffer 未填滿，先 prefill"
        total = self.tree.total
        segment = total / batch_size
        idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        batch: list[NStepTransition] = []
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = self._rng.uniform(lo, hi)
            idx, p, data = self.tree.get(s)
            # 極少數情況 data 還沒寫入（priority 為 0 之類），fallback
            if data is None:
                s = self._rng.uniform(0.0, total)
                idx, p, data = self.tree.get(s)
            idxs[i] = idx
            priorities[i] = p
            batch.append(data)  # type: ignore[arg-type]

        probs = priorities / max(total, 1e-12)
        beta = self._beta(step)
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max() + 1e-12  # normalize for stability

        s = torch.from_numpy(np.concatenate([t.s for t in batch], axis=0)).float()
        s_next = torch.from_numpy(np.concatenate([t.s_next for t in batch], axis=0)).float()
        a = torch.tensor([t.a for t in batch], dtype=torch.long)
        r = torch.tensor([t.r for t in batch], dtype=torch.float32)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32)
        n = torch.tensor([t.n for t in batch], dtype=torch.float32)
        w = torch.from_numpy(weights.astype(np.float32))
        return s, a, r, s_next, done, n, idxs, w

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(td_errors) + self.eps
        for idx, e in zip(idxs, td):
            p = float(e) ** self.alpha
            self.tree.update(int(idx), p)
            if float(e) > self._max_priority:
                self._max_priority = float(e)
