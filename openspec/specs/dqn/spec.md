# dqn Specification

## Purpose
TBD - created by archiving change 01-naive-dqn-static. Update Purpose after archive.
## Requirements
### Requirement: Gridworld Environment Wrapper
系統 SHALL 將上游教科書 Gridworld 包裝成 numpy/torch 友善介面，支援
`static` / `player` / `random` 三種模式。

#### Scenario: Reset returns flattened state
- **WHEN** `GridworldEnv(mode="static").reset()` 被呼叫
- **THEN** 回傳 shape `(1, 64)`、dtype `float32` 的 numpy 陣列

#### Scenario: Step returns SARS'
- **WHEN** 呼叫 `step(action)` 且 action ∈ {0,1,2,3}
- **THEN** 回傳 `(next_state, reward, done, info)`；reward 為 −1/0/1；
  done 在 reward≠0 或 `moves >= max_moves` 時為 True

### Requirement: DQN Network
系統 SHALL 提供教科書 Listing 3.2 的三層全連接 DQN。

#### Scenario: Shape
- **WHEN** 輸入 shape `(B, 64)`
- **THEN** 輸出 shape `(B, 4)` 的 Q 值

### Requirement: Experience Replay Buffer
系統 SHALL 提供 deque-based replay buffer，支援隨機抽樣批次。

#### Scenario: Sampling
- **WHEN** buffer 長度 ≥ batch_size 時呼叫 `sample(batch_size)`
- **THEN** 回傳 `(s, a, r, s_next, done)`，皆為 torch tensor，s 與 s_next
  shape 為 `(batch_size, 64)`

### Requirement: Naive DQN Training
系統 SHALL 提供同時比較有/無 replay 的訓練腳本，並輸出學習曲線、勝率曲線與
rollout GIF。

#### Scenario: Convergence on static mode
- **WHEN** 有 replay 的 DQN 在 static mode 訓練 3000 回合
- **THEN** 最後 200 回合的勝率 ≥ 0.95，且 rollout GIF 能從起點走到 goal

### Requirement: Dueling DQN Architecture
系統 SHALL 提供 Dueling DQN，以 V-head 與 A-head 組合出 Q 值。

#### Scenario: Advantage centered
- **WHEN** 任意輸入 `(B, 64)`
- **THEN** 輸出 Q 的 advantage 部分以 `A − mean_a A` 標準化，避免 V 與 A 不可辨識

### Requirement: Double DQN Update
系統 SHALL 在訓練時使用 Double DQN 目標：`argmax` 來自 online 網路、
Q 值來自 target 網路。

#### Scenario: Stable sync
- **WHEN** 每經過 500 個 env step
- **THEN** target 網路以 hard copy 同步 online 權重

### Requirement: Variant Comparison
系統 SHALL 提供單一訓練腳本同時跑三種 variant 並輸出比較圖。

#### Scenario: Player mode convergence
- **WHEN** 三種 variant 皆在 player mode 訓練 4000 回合
- **THEN** 最後 200 回合勝率 ≥ 0.90

### Requirement: Lightning Training Loop
系統 SHALL 以 `pytorch_lightning.LightningModule` 封裝 Dueling Double DQN，
使用 `pl.Trainer` 驅動訓練。

#### Scenario: Gradient clipping
- **WHEN** `pl.Trainer(gradient_clip_val=1.0)` 設定
- **THEN** 每步更新的梯度 L2 norm 被裁剪到 ≤ 1.0

#### Scenario: Cosine LR schedule
- **WHEN** 訓練達到 `total_steps`
- **THEN** 學習率由 1e-3 退火至 1e-5（CosineAnnealingLR）

### Requirement: Soft Target Update
系統 SHALL 在每個訓練步以 Polyak averaging 更新 target 網路。

#### Scenario: Tau
- **WHEN** `tau=0.005` 時
- **THEN** `target ← (1 − τ) target + τ online`

### Requirement: Exponential Epsilon Schedule
系統 SHALL 對隨機動作的機率以指數衰減方式降低，加速收斂但保留必要探索。

#### Scenario: Decay formula
- **WHEN** 環境互動步數 `t`
- **THEN** `ε(t) = ε_end + (ε_start − ε_end) · exp(−3 · min(1, t/decay_steps))`

### Requirement: Random Mode Performance
系統 SHALL 於 Gridworld random mode 下的最後 300 回合達到勝率 ≥ 0.60。

#### Scenario: Smoke-test rollout
- **WHEN** 使用訓練完成的網路在 random mode 跑 1 回合 greedy rollout
- **THEN** 能在 `max_steps` 內抵達 goal（贏）或避開 pit（至少不在第一步送頭）

