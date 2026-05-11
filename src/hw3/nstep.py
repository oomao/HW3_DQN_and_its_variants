"""n-step return 累積器。

把連續最多 n 個 1-step transition 合成一筆 n-step transition：
    r_n = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1}
    s_next = s_{t+n}    （若中途 done，則 s_next 為 done 那一步的 s_next）
    done  = 該段內是否觸發過 episode 結束
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .prioritized_replay import NStepTransition


class NStepBuffer:
    """單軌跡 n-step 緩衝器。episode 結束時要 flush 殘餘 transition。"""

    def __init__(self, n: int = 3, gamma: float = 0.9):
        self.n = n
        self.gamma = gamma
        self._buf: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, s, a, r, s_next, done) -> list[NStepTransition]:
        """把 1-step transition 餵進來，回傳本次新增的 n-step transitions 清單。

        - 當 deque 滿到 n 時，吐出以 buf[0] 為起點、horizon=n 的 transition。
        - episode 結束時，flush 剩下所有 partial transition（horizon < n）。
        """
        self._buf.append((s, a, r, s_next, done))
        out: list[NStepTransition] = []

        # 滿 n 步 → 吐出最舊的一筆 full-horizon transition
        if len(self._buf) == self.n:
            out.append(self._make_from(0, self.n))

        # episode 結束 → 把剩下的 partial-horizon transition 全部吐出
        if done:
            # 若剛剛已吐出 buf[0]，這次從 buf[1] 開始；否則 buf 還沒滿，從 0 開始
            start = 1 if len(self._buf) == self.n else 0
            for i in range(start, len(self._buf)):
                k = len(self._buf) - i
                out.append(self._make_from(i, k))
            self._buf.clear()

        return out

    def _make_from(self, start: int, k: int) -> NStepTransition:
        """以 buf[start] 為起點、最多累積 k 步合成一筆 n-step transition。"""
        s0, a0, _, _, _ = self._buf[start]
        R = 0.0
        done_in_window = False
        s_next = self._buf[start + k - 1][3]
        actual_k = k
        for i in range(k):
            _, _, r_i, s_next_i, d_i = self._buf[start + i]
            R += (self.gamma ** i) * r_i
            if d_i:
                done_in_window = True
                s_next = s_next_i
                actual_k = i + 1
                break
        return NStepTransition(
            s=s0,
            a=int(a0),
            r=float(R),
            s_next=s_next,
            done=bool(done_in_window),
            n=int(actual_k),
        )
