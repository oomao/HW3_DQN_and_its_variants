# Spec Delta: dqn (ADDED)

## ADDED Requirements

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
