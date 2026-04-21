# Proposal: Naive DQN + Experience Replay on Gridworld (static)

## Why
HW3-1 要求以「DRL in Action」教科書第三章的 starter code 為基礎，在 Gridworld
的 **static** 模式下，實作最基本的 DQN 網路並加上 Experience Replay Buffer，
藉此驗證 agent 能在固定地圖上學到最優路徑。這是後續兩個進階題的地基。

## What Changes
- 新增 `src/hw3/` 套件，含：
  - `gridworld_vendored.py`：抄自上游（MIT 授權）之 Gridworld 與 GridBoard。
  - `env.py`：把 Gridworld 包成 numpy 友善的 `reset/step` 介面，並內建
    state 前處理（`render_np().reshape(1,64) + 微量雜訊`）。
  - `models.py`：Listing 3.2 的三層全連接網路 `DQN`。
  - `replay.py`：`ReplayBuffer`（`deque`、random sample）。
  - `train_naive.py`：同時訓練「無 replay」與「有 replay」兩種設定，輸出
    reward 曲線、勝率曲線與 policy rollout GIF。
- 新增 `requirements.txt`、`.gitignore`（忽略 `.claude/`、`_upstream/`）。
- 新增 `scripts/startup.sh`、`scripts/ending.sh`（沿用 HW2 的流程規範）。

## Impact
- 新能力：`dqn`（見 spec delta）。
- 不影響既有程式（repo 全新建立）。
- 訓練時間 < 2 分鐘（CPU）。
