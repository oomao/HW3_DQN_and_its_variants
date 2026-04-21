# Proposal: Double DQN 與 Dueling DQN on Gridworld (player)

## Why
HW3-2 要求在 Gridworld 的 **player** 模式（只有玩家位置隨機）下，實作並比較
兩種 DQN 的改良版：
- **Double DQN**：以 online 網路挑動作、target 網路估值，緩解 max-operator
  造成的過度估計（overestimation bias）。
- **Dueling DQN**：把 Q 拆成 V(s) 與 A(s,a)，讓網路學「這個狀態多好」與
  「選這個動作多好」兩件事，提升資料效率。

## What Changes
- 在 `models.py` 新增 `DuelingDQN`（trunk + V-head + A-head）。
- 在 `models.py` 新增 `soft_update` / `hard_update` 工具函式。
- 新增 `src/hw3/train_variants.py`：統一的訓練流程，以 `variant` 參數切換
  `vanilla` / `double` / `dueling` 三種設定；皆使用 target network（sync 每 500 步）。
- 輸出三條曲線比較圖、勝率圖與 Dueling DQN 的 rollout GIF。

## Impact
- 新 requirement：Dueling architecture、Double-DQN update。
- 訓練時間 < 4 分鐘（CPU）。
